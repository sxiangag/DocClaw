"""General VLM-backed OCR implementation."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from docclaw.agent.tool.ocr.ocr import (
    OcrTool,
    PixelBBox,
    _batch_ocr_message,
    _existing_ocr_metadata,
    _existing_target_ocr,
    _ocr_message,
    _resolve_ocr_targets,
    clamp_bbox,
)
from docclaw.agent.tool.ocr.paddleocrvl import (
    _build_skipped_image_like_result,
    _region_is_image_like_for_broad_ocr,
)
from docclaw.agent.tool.vlm_client import VLMClient
from docclaw.agent.utils import Action, Observation, RunState, plannerize_page_refs
from docclaw.provider.base import LLMProvider


class VLMOcrTool(OcrTool):
    """Recognize page or region text with a multimodal language model."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        artifact_dir: str | Path | None = None,
        allow_chart_region_ocr: bool = False,
    ) -> None:
        super().__init__(artifact_dir=artifact_dir)
        self.client = VLMClient(
            provider,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.allow_chart_region_ocr = allow_chart_region_ocr

    @property
    def source_name(self) -> str:
        return "vlm_ocr"

    async def execute(self, state: RunState, action: Action) -> Observation:
        force = bool(action.parameters.get("force", False))
        targets, error = _resolve_ocr_targets(state, action)
        if error is not None:
            return self.error(action, error)
        assert targets is not None

        results: list[dict[str, Any]] = []
        artifacts: list[str] = []
        source_counts: dict[str, int] = {}
        for target in targets:
            result, target_artifacts, target_error = await self._execute_single_target(
                state,
                action,
                target,
                force=force,
            )
            if target_error is not None:
                return self.error(action, target_error)
            if result is None:
                continue
            results.append(result)
            artifacts.extend(target_artifacts)
            source = str(result.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1

        message = (
            _ocr_message(results[0], source_counts)
            if len(results) == 1
            else _batch_ocr_message(results, source_counts)
        )
        return Observation(
            action_id=action.action_id,
            success=True,
            data={"results": results, "sources": source_counts},
            message=message,
            artifacts=artifacts,
        )

    async def _execute_single_target(
        self,
        state: RunState,
        action: Action,
        target: dict[str, Any],
        *,
        force: bool,
    ) -> tuple[dict[str, Any] | None, list[str], str | None]:
        if _should_skip_image_like_region_target(
            state,
            target,
            allow_chart_region_ocr=self.allow_chart_region_ocr,
        ):
            skipped = _build_skipped_image_like_result(target)
            if skipped is not None:
                return skipped, [], None

        existing_payload = _existing_target_ocr(state, target)
        if not force and existing_payload is not None:
            return (
                {
                    **target["data"],
                    "text": existing_payload["text"],
                    "source": existing_payload.get("source", "state"),
                    "confidence": existing_payload["confidence"],
                    **_existing_ocr_metadata(existing_payload),
                },
                [],
                None,
            )

        source_path, artifacts, error = self._prepare_ocr_image(state, action, target)
        if error is not None:
            return None, [], error
        assert source_path is not None

        try:
            result = await self.recognize_text(state, action, target, source_path)
        except Exception as exc:
            return None, [], f"ocr failed: {exc}"

        text = result.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        confidence = result.get("confidence")
        if isinstance(target["data"].get("region_id"), str):
            confidence = None
        return (
            {
                **target["data"],
                "text": text,
                "source": self.source_name,
                "confidence": confidence,
            },
            artifacts,
            None,
        )

    async def recognize_text(
        self,
        state: RunState,
        action: Action,
        target: dict[str, Any],
        image_path: Path,
    ) -> dict[str, Any]:
        target_payload = plannerize_page_refs(
            {
                "task": state.task.to_dict(),
                "action_target": target.get("data") or {},
                "request_kind": _request_kind_for_target(state, target),
                "instructions": "Return a faithful transcription for this document image or crop.",
            }
        )
        text, response = await self.client.complete_text(
            system_prompt=_ocr_system_prompt(state, target),
            user_payload=target_payload,
            image_path=image_path,
        )
        if _request_kind_for_target(state, target) == "formula":
            text = _normalize_formula_text_for_target(state, target, text)
        return {
            "text": text,
            "confidence": None,
            "usage": response.usage,
        }

    def render_crop(
        self,
        source_path: Path,
        output_path: Path,
        bbox: PixelBBox,
    ) -> None:
        from PIL import Image

        with Image.open(source_path) as image:
            width, height = image.size
            x0, y0, x1, y1 = clamp_bbox(bbox, width, height)
            if x1 <= x0 or y1 <= y0:
                raise ValueError("bbox is outside the page image")
            crop = image.crop((x0, y0, x1, y1))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(output_path)


def _ocr_system_prompt(state: RunState, target: dict[str, Any]) -> str:
    request_kind = _request_kind_for_target(state, target)
    common_rules = (
        "Perform OCR on the provided document page or crop and return only OCR output for visible content.\n"
        "Do not summarize. Do not explain. Do not add commentary.\n"
        "Do not invent missing content. If part of the image is unreadable, stay conservative.\n"
        "Do not rewrite, normalize, or improve the content.\n"
        "Do not introduce extra line breaks for readability.\n"
        "Keep block-level breaks that are clearly visible in the document page."
    )
    if request_kind == "table":
        return (
            common_rules
            + "The target is a table.\n"
            + "Return only one OCR result as an HTML table.\n"
            + "Preserve cell text faithfully.\n"
            + "Use <table>, <thead>, <tbody>, <tr>, <th>, and <td> when appropriate.\n"
            + "Use colspan and rowspan when clearly needed.\n"
            + "Do not wrap the result in markdown fences."
        )
    if request_kind == "formula":
        formula_style = _formula_style_for_target(state, target)
        if formula_style == "display":
            delimiter_rule = (
                "The formula is display-style. Return the OCR result wrapped in $$...$$.\n"
            )
        elif formula_style == "inline":
            delimiter_rule = (
                "The formula is inline-style. Return the OCR result wrapped in $...$.\n"
            )
        else:
            delimiter_rule = (
                "Wrap inline formulas in $...$ and standalone display formulas in $$...$$ based on the visible layout.\n"
            )
        return (
            common_rules
            + "The target is a formula.\n"
            + "Return only the OCR result for the formula as LaTeX-style text.\n"
            + delimiter_rule
            + "Preserve symbols, superscripts, subscripts, fractions, matrices, and alignment faithfully."
        )
    if request_kind == "chart":
        return (
            common_rules
            + "The target is a chart.\n"
            + "Return only OCR output as concise plain text containing directly visible chart content.\n"
            + "Focus on titles, axis labels, legend items, series names, and directly readable values.\n"
            + "Do not infer hidden data points."
        )
    return (
        common_rules
        + "The target is regular document text.\n"
        + "Return only the OCR transcription text."
    )


def _request_kind_for_target(state: RunState, target: dict[str, Any]) -> str:
    data = target.get("data")
    if not isinstance(data, dict):
        return "ocr"
    region_id = data.get("region_id")
    if not isinstance(region_id, str):
        return "ocr"
    region = state.get_region(region_id)
    if region is None:
        return "ocr"
    normalized = str(region.type or "").strip().lower()
    if normalized in {"table", "formula", "chart"}:
        return normalized
    return "ocr"


def _formula_style_for_target(state: RunState, target: dict[str, Any]) -> str | None:
    data = target.get("data")
    if not isinstance(data, dict):
        return None
    region_id = data.get("region_id")
    if not isinstance(region_id, str):
        return None
    region = state.get_region(region_id)
    return _formula_style_for_region(region)


def _formula_style_for_region(region: Any) -> str | None:
    if region is None:
        return None
    raw_type = str(getattr(region, "raw_type", "") or "").strip().lower()
    if raw_type in {"display_formula", "equation_isolated"}:
        return "display"
    if raw_type in {"inline_formula", "equation_inline"}:
        return "inline"

    metadata = getattr(region, "metadata", None)
    if isinstance(metadata, dict):
        layout = metadata.get("layout")
        if isinstance(layout, dict):
            label = str(layout.get("label") or "").strip().lower()
            if label in {"display_formula", "equation_isolated"}:
                return "display"
            if label in {"inline_formula", "equation_inline"}:
                return "inline"
        render = metadata.get("render")
        if isinstance(render, dict):
            markdown_label = str(render.get("markdown_label") or "").strip().lower()
            if markdown_label == "display_formula":
                return "display"
            if markdown_label == "inline_formula":
                return "inline"
    return None


def _normalize_formula_text_for_target(
    state: RunState,
    target: dict[str, Any],
    text: str,
) -> str:
    style = _formula_style_for_target(state, target)
    return _normalize_formula_text(text, style=style)


def _normalize_formula_text(text: str, *, style: str | None) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return normalized

    if style == "display":
        inner = _strip_formula_delimiters(normalized)
        return f"$$\n{inner}\n$$" if inner else normalized
    if style == "inline":
        inner = _strip_formula_delimiters(normalized)
        return f"${inner}$" if inner else normalized
    return normalized


def _strip_formula_delimiters(text: str) -> str:
    normalized = text.strip()
    if normalized.startswith("$$") and normalized.endswith("$$") and len(normalized) >= 4:
        return normalized[2:-2].strip()
    if normalized.startswith(r"\[") and normalized.endswith(r"\]") and len(normalized) >= 4:
        return normalized[2:-2].strip()
    if normalized.startswith("$") and normalized.endswith("$") and len(normalized) >= 2:
        return normalized[1:-1].strip()
    if normalized.startswith(r"\(") and normalized.endswith(r"\)") and len(normalized) >= 4:
        return normalized[2:-2].strip()

    match = re.fullmatch(r"\\begin\{(?:aligned|align\*?|equation\*?|gather\*?|matrix|bmatrix|pmatrix|vmatrix|Vmatrix)\}(.*)\\end\{(?:aligned|align\*?|equation\*?|gather\*?|matrix|bmatrix|pmatrix|vmatrix|Vmatrix)\}", normalized, re.DOTALL)
    if match is not None:
        return normalized
    return normalized


def _should_skip_image_like_region_target(
    state: RunState,
    target: dict[str, Any],
    *,
    allow_chart_region_ocr: bool,
) -> bool:
    data = target.get("data")
    if not isinstance(data, dict):
        return False
    region_id = data.get("region_id")
    if not isinstance(region_id, str):
        return False
    region = state.get_region(region_id)
    if region is None:
        return False
    if allow_chart_region_ocr and str(region.type or "").strip().lower() == "chart":
        return False
    return _region_is_image_like_for_broad_ocr(region)
