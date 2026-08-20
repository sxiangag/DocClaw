"""General VLM-backed formula parsing implementation."""

from __future__ import annotations

from typing import Any

from docclaw.agent.tool.formula.formula import FormulaTool, _resolve_formula_target
from docclaw.agent.tool.formula.paddleocrvl import (
    _Candidate,
    _cached_formulas_for_page,
    _formula_candidates_for_page,
    _uses_crop_regions,
    _uses_refined_regions,
    _uses_zoom_regions,
    _candidate_image_input,
)
from docclaw.agent.tool.ocr.vlm import _formula_style_for_region, _normalize_formula_text
from docclaw.agent.tool.vlm_client import VLMClient
from docclaw.agent.utils import Action, PageState, Region, RunState, plannerize_page_refs
from docclaw.provider.base import LLMProvider


class VLMFormulaTool(FormulaTool):
    """Parse formulas with a general multimodal language model."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.client = VLMClient(
            provider,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @property
    def description(self) -> str:
        return (
            "Parse formulas from known formula regions or page-level formula candidates "
            "with a general multimodal language model. This keeps DocClaw layout "
            "targeting and uses VLM recognition for formula transcription."
        )

    async def execute(self, state: RunState, action: Action):
        pages, regions, error = _resolve_formula_target(state, action)
        if error is not None:
            return self.error(action, error)
        assert pages is not None

        try:
            results = await self.parse_formulas_async(state, pages, regions, action)
        except Exception as exc:
            return self.error(action, str(exc))

        source_counts: dict[str, int] = {}
        for result in results:
            source = str(result.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1

        from docclaw.agent.utils import Observation

        total_formulas = len(results)
        message = (
            f"Formula parsing ready for {len(pages)} page(s); "
            f"{total_formulas} formula(s) available."
        )
        return Observation(
            action_id=action.action_id,
            success=True,
            data={"results": results, "sources": source_counts},
            message=message,
        )

    def parse_formulas(
        self,
        state: RunState,
        pages: list[Any],
        regions: list[Region] | None,
        action: Action,
    ) -> list[dict[str, Any]]:
        raise RuntimeError("VLMFormulaTool.parse_formulas is async-only; call execute")

    async def parse_formulas_async(
        self,
        state: RunState,
        pages: list[PageState],
        regions: list[Region] | None,
        action: Action,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for page in pages:
            page_regions = (
                [region for region in regions if region.page_index == page.page_index]
                if regions is not None
                else None
            )
            if regions is not None and not page_regions:
                continue

            cached_formulas = _cached_formulas_for_page(page, regions=page_regions)
            if cached_formulas is not None and not _uses_refined_regions(action):
                results.extend(cached_formulas)
                continue

            candidates = _formula_candidates_for_page(
                state,
                page,
                explicit_regions=page_regions,
                use_zoom_artifacts=_uses_zoom_regions(action),
                use_crop_artifacts=_uses_crop_regions(action),
            )
            for candidate in candidates:
                formula = await self._parse_candidate(state, page, candidate)
                if formula is not None:
                    results.append(formula)
        return results

    async def _parse_candidate(
        self,
        state: RunState,
        page: PageState,
        candidate: _Candidate,
    ) -> dict[str, Any] | None:
        payload, response = await self.client.complete_json(
            system_prompt=_formula_system_prompt(candidate.region),
            user_payload=plannerize_page_refs(
                {
                    "task": state.task.to_dict(),
                    "page_index": page.page_index,
                    "region_id": candidate.region_id,
                    "request_kind": "formula_to_latex",
                }
            ),
            image_path=_candidate_image_input(page, candidate),
        )
        latex = _normalize_formula_text(
            _extract_latex(payload),
            style=_formula_style_for_region(candidate.region),
        )
        if not latex:
            return None
        confidence = _extract_confidence(payload)
        if candidate.region is not None:
            candidate.region.text = latex
            candidate.region.confidence = confidence
        return {
            "page_index": page.page_index,
            "region_id": candidate.region_id,
            "text": latex,
            "source": "vlm_formula",
            "confidence": confidence,
            "usage": response.usage,
        }


def _formula_system_prompt(region: Region | None) -> str:
    style = _formula_style_for_region(region)
    common_rules = (
        "Perform OCR on the provided document page or crop and return only OCR output for visible content.\n"
        "Do not summarize. Do not explain. Do not add commentary.\n"
        "Do not invent missing content. If part of the image is unreadable, stay conservative.\n"
        "Do not rewrite, normalize, or improve the content.\n"
    )
    if style == "display":
        delimiter_rule = (
            "The formula is display-style. Return the OCR result wrapped in $$...$$.\n"
        )
    elif style == "inline":
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
        + "Return exactly one JSON object with this shape:\n"
        + "{\n"
        + '  "latex": "formula OCR output with the required math delimiters",\n'
        + '  "confidence": 0.0\n'
        + "}\n"
        + "Rules:\n"
        + "- latex must contain only the OCR result for the formula content.\n"
        + delimiter_rule
        + "- Preserve symbols, superscripts, subscripts, fractions, matrices, and alignment faithfully.\n"
        + "- Do not output prose outside the JSON object.\n"
        + '- If no clear formula is visible, return {"latex": "", "confidence": 0.0}.'
    )


def _extract_latex(payload: dict[str, Any]) -> str:
    for key in ("latex", "text", "formula"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_confidence(payload: dict[str, Any]) -> float | None:
    value = payload.get("confidence")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
