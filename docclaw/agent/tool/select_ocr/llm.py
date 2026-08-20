"""LLM-backed OCR refinement candidate selection."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import re
from typing import Any

from docclaw.agent.debug import dump_jsonl_from_env
from docclaw.agent.tool.tool import Tool
from docclaw.agent.utils import (
    Action,
    ActionType,
    Observation,
    RunState,
)
from docclaw.provider.base import LLMProvider


class LLMSelectOcrTool(Tool):
    """Select the best OCR candidate for refined regions."""

    _JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    _ALLOWED_CANDIDATES = {"original", "zoom", "crop", "rotate", "no_text"}

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.provider = provider
        self.model = model or provider.get_default_model()
        self.max_tokens = max_tokens
        self.temperature = temperature

    @property
    def action_type(self) -> ActionType:
        return "select_ocr"

    @property
    def description(self) -> str:
        return (
            "Compare original OCR text against existing refinement candidates "
            "such as zoom, crop, or rotate OCR for known regions, "
            "then write back the best candidate."
        )

    @property
    def target_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "Select one or more known region_ids whose refinement candidates should be judged.",
            "properties": {
                "region_ids": {
                    "type": "array",
                    "description": "Known region identifiers to compare across original and available refinement candidates.",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "additionalProperties": False,
        }

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "No additional parameters.",
            "properties": {},
            "additionalProperties": False,
        }

    async def execute(self, state: RunState, action: Action) -> Observation:
        regions, error = _resolve_selection_regions(state, action)
        if error is not None:
            return self.error(action, error)
        assert regions is not None

        judge_inputs: list[dict[str, Any]] = []
        kept_original_region_ids: list[str] = []
        selections_by_region_id: dict[str, str] = {}
        debug_artifacts: list[str] = []

        for region in regions:
            selection_input = _build_selection_input(
                state,
                action,
                region,
            )
            if selection_input is None:
                kept_original_region_ids.append(region.region_id)
                selections_by_region_id[region.region_id] = "original"
                continue
            for artifact_path in selection_input.get("debug_artifact_paths", []):
                if isinstance(artifact_path, str) and artifact_path:
                    debug_artifacts.append(artifact_path)
            judge_inputs.append(selection_input)

        if judge_inputs:
            messages, user_payload = self._build_messages(state, action, judge_inputs)
            dump_jsonl_from_env(
                "DOCCLAW_OCR_SELECTION_DEBUG_PATH",
                {
                    "kind": "select_ocr_input",
                    "model": self.model,
                    "document_id": state.document.document_id,
                    "run_id": state.run_id,
                    "action_id": action.action_id,
                    "task": state.task.to_dict(),
                    "target": action.target,
                    "parameters": action.parameters,
                    "judge_region_ids": [str(item.get("region_id") or "") for item in judge_inputs],
                    "user_payload": user_payload,
                },
            )
            response = await self.provider.chat(
                messages,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            if response.error:
                return self.error(action, response.error)

            try:
                payload = self._parse_payload(response.content)
                selections_by_region_id.update(
                    self._resolve_selections(
                        judge_inputs=judge_inputs,
                        payload=payload,
                    )
                )
            except Exception as exc:
                return self.error(action, f"invalid candidate selection response: {exc}")
            dump_jsonl_from_env(
                "DOCCLAW_OCR_SELECTION_DEBUG_PATH",
                {
                    "kind": "select_ocr_output",
                    "model": self.model,
                    "document_id": state.document.document_id,
                    "run_id": state.run_id,
                    "action_id": action.action_id,
                    "raw_content": response.content,
                    "parsed_payload": payload,
                    "resolved_selections": selections_by_region_id,
                    "usage": response.usage,
                },
            )
            usage = response.usage
        else:
            usage = {}

        updated_region_ids: list[str] = []
        final_kept_original_region_ids: list[str] = []
        selection_rows: list[dict[str, Any]] = []
        for region in regions:
            selected_candidate = selections_by_region_id.get(region.region_id, "original")
            if selected_candidate == "original":
                final_kept_original_region_ids.append(region.region_id)
            else:
                updated_region_ids.append(region.region_id)
            selection_rows.append(
                {
                    "region_id": region.region_id,
                    "selected_candidate": selected_candidate,
                }
            )

        return Observation(
            action_id=action.action_id,
            success=True,
            data={
                "selections": selection_rows,
                "updated_region_ids": updated_region_ids,
                "kept_original_region_ids": final_kept_original_region_ids,
                "source": "llm",
                "usage": usage,
            },
            message=(
                f"Selected final OCR candidates for {len(regions)} region(s); "
                f"updated {len(updated_region_ids)} and kept {len(final_kept_original_region_ids)} original."
            ),
            artifacts=debug_artifacts,
        )

    def update_state(
        self,
        state: RunState,
        action: Action,
        observation: Observation,
    ) -> None:
        selections = observation.data.get("selections")
        if not isinstance(selections, list):
            return
        for selection in selections:
            if not isinstance(selection, dict):
                continue
            region_id = selection.get("region_id")
            selected_candidate = selection.get("selected_candidate")
            if not isinstance(region_id, str) or not region_id.strip():
                continue
            if not isinstance(selected_candidate, str) or not selected_candidate.strip():
                continue
            region = state.get_region(region_id)
            if region is None:
                continue
            _apply_selected_candidate(
                region,
                selected_candidate=selected_candidate,
            )

    def _build_messages(
        self,
        state: RunState,
        action: Action,
        judge_inputs: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        system_prompt = (
            "You are the DocClaw OCR refinement candidate selection tool.\n"
            "For each provided region, choose the single best OCR result among the "
            "available choices.\n"
            "You will receive a JSON payload containing one entry per region, followed "
            "by one matching region crop image for each region.\n"
            "For each region, first read that region's JSON entry, then find the "
            "matching 'Region <region_id>: region crop' image below, and choose the "
            "best text from choices based on that crop.\n"
            "Use only the provided region crop image, bbox, region type, and choice texts.\n"
            "You may also choose no_text, but use it conservatively. Choose no_text "
            "only when the region truly should not produce any text output, such as "
            "when it contains only noise, artifacts, or non-text marks rather than "
            "recoverable readable text. If the crop contains clearly visible readable "
            "text, do not choose no_text.\n"
            "Return only JSON with this shape:\n"
            "{\n"
            '  "selections": [\n'
            "    {\n"
            '      "region_id": "region id",\n'
            '      "best_candidate": "original"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Allowed best_candidate values are only: original, zoom, crop, rotate, no_text.\n"
            "Do not invent region ids. Choose exactly one candidate per provided region."
        )
        user_payload = {
            "task": state.task.to_dict(),
            "target": action.target,
            "parameters": action.parameters,
            "regions": [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"region_crop_image_url", "debug_artifact_paths"}
                }
                for item in judge_inputs
            ],
        }
        user_prompt = json.dumps(
            user_payload,
            ensure_ascii=False,
            indent=2,
        )
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": user_prompt},
        ]
        for item in judge_inputs:
            region_id = str(item.get("region_id") or "")
            if region_id:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"Region {region_id}: region crop",
                    }
                )
            region_crop_image_url = item.get("region_crop_image_url")
            if isinstance(region_crop_image_url, str) and region_crop_image_url:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": region_crop_image_url}}
                )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ], user_payload

    @classmethod
    def _parse_payload(cls, content: str | None) -> dict[str, Any]:
        if content is None or not content.strip():
            raise ValueError("empty response")
        candidate = content.strip()
        match = cls._JSON_BLOCK_RE.search(candidate)
        if match:
            candidate = match.group(1).strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start < 0 or end < start:
                raise ValueError("response must contain a JSON object")
            payload = json.loads(candidate[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("response JSON must be an object")
        return payload

    def _resolve_selections(
        self,
        *,
        judge_inputs: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> dict[str, str]:
        items = payload.get("selections")
        if not isinstance(items, list):
            raise ValueError("selections must be a list")

        available_by_region_id = {
            str(item["region_id"]): set(item["choices"])
            for item in judge_inputs
        }
        resolved: dict[str, str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            region_id = item.get("region_id")
            best_candidate = item.get("best_candidate")
            if not isinstance(region_id, str) or not region_id.strip():
                continue
            if not isinstance(best_candidate, str) or not best_candidate.strip():
                continue
            candidate_kind = best_candidate.strip()
            if candidate_kind not in self._ALLOWED_CANDIDATES:
                continue
            allowed = available_by_region_id.get(region_id)
            if allowed is None:
                continue
            if candidate_kind not in allowed:
                continue
            resolved[region_id] = candidate_kind

        for region_id in available_by_region_id:
            resolved.setdefault(region_id, "original")
        return resolved


def _resolve_selection_regions(
    state: RunState,
    action: Action,
) -> tuple[list[Any] | None, str | None]:
    if "region_id" in action.target or "page_index" in action.target:
        return None, "select_ocr requires target.region_ids"
    region_ids = action.target.get("region_ids")
    if not isinstance(region_ids, list) or not region_ids:
        return None, "select_ocr target.region_ids must be a non-empty list"

    regions = []
    seen: set[str] = set()
    for raw_region_id in region_ids:
        region_id = str(raw_region_id).strip()
        if not region_id:
            continue
        if region_id in seen:
            continue
        region = state.get_region(region_id)
        if region is None:
            return None, f"unknown region_id: {region_id}"
        regions.append(region)
        seen.add(region_id)
    if not regions:
        return None, "select_ocr target.region_ids must contain non-empty strings"
    return regions, None

def _build_selection_input(
    state: RunState,
    action: Action,
    region: Any,
) -> dict[str, Any] | None:
    candidates = region.metadata.get("ocr_refinement_candidates")
    if not isinstance(candidates, dict):
        return None

    choices: dict[str, str] = {
        "original": region.text or "",
        "no_text": "",
    }
    for candidate_kind in ("zoom", "crop", "rotate"):
        candidate = candidates.get(candidate_kind)
        if not isinstance(candidate, dict):
            continue
        text = candidate.get("text")
        if not isinstance(text, str):
            continue
        choices[candidate_kind] = text

    if len(choices) <= 2:
        return None

    original_artifact_path = _render_original_region_crop(state, action, region)
    zoom_artifact_path = _candidate_artifact_path(candidates.get("zoom"))
    crop_artifact_path = _candidate_artifact_path(candidates.get("crop"))
    rotate_artifact_path = _candidate_artifact_path(candidates.get("rotate"))
    debug_artifact_paths = [
        path
        for path in [
            original_artifact_path,
            zoom_artifact_path,
            crop_artifact_path,
            rotate_artifact_path,
        ]
        if isinstance(path, str) and path
    ]

    return {
        "region_id": region.region_id,
        "region_type": region.type,
        "bbox": list(region.bbox),
        "coordinate_space": region.coordinate_space,
        "choices": choices,
        "region_crop_image_url": _image_data_url(original_artifact_path),
        "debug_artifact_paths": debug_artifact_paths,
    }


def _candidate_artifact_path(candidate: Any) -> str | None:
    if not isinstance(candidate, dict):
        return None
    artifact_path = candidate.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        return None
    candidate_path = Path(artifact_path).expanduser()
    if not candidate_path.exists():
        return None
    return str(candidate_path)


def _render_original_region_crop(
    state: RunState,
    action: Action,
    region: Any,
) -> str | None:
    page = state.get_page(region.page_index)
    if page is None or not page.image_path:
        return None
    source_path = Path(page.image_path).expanduser()
    if not source_path.exists():
        return None

    artifact_dir = _resolve_artifact_dir(state, action)
    if artifact_dir is None:
        return None

    output_path = artifact_dir / f"{action.action_id}_{_safe_cache_fragment(region.region_id)}_original.png"
    try:
        _render_crop(
            source_path,
            output_path,
            tuple(float(value) for value in region.bbox),
            str(region.coordinate_space or "pixel"),
            page.width,
            page.height,
        )
    except Exception:
        return None
    return str(output_path)


def _resolve_artifact_dir(
    state: RunState,
    action: Action,
) -> Path | None:
    value = action.parameters.get("artifact_dir")
    if value is None:
        value = state.metadata.get("artifact_dir")
    if value is None:
        value = state.document.metadata.get("artifact_dir")
    if value is None:
        return None
    path = Path(str(value)).expanduser() / "selection_debug"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _render_crop(
    source_path: Path,
    output_path: Path,
    bbox: tuple[float, float, float, float],
    coordinate_space: str,
    page_width: int | None,
    page_height: int | None,
) -> None:
    from PIL import Image

    pixel_bbox = _to_pixel_bbox(
        bbox,
        coordinate_space,
        page_width,
        page_height,
    )
    if pixel_bbox is None:
        raise ValueError("invalid bbox")

    with Image.open(source_path) as image:
        width, height = image.size
        x0, y0, x1, y1 = _clamp_bbox(pixel_bbox, width, height)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("bbox is outside the page image")
        crop = image.crop((x0, y0, x1, y1))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path)


def _to_pixel_bbox(
    bbox: tuple[float, float, float, float],
    coordinate_space: str,
    page_width: int | None,
    page_height: int | None,
) -> tuple[int, int, int, int] | None:
    if coordinate_space == "relative":
        if page_width is None or page_height is None:
            return None
        values = (
            float(bbox[0]) * page_width,
            float(bbox[1]) * page_height,
            float(bbox[2]) * page_width,
            float(bbox[3]) * page_height,
        )
    else:
        values = tuple(float(value) for value in bbox)
    x0, y0, x1, y1 = (int(round(value)) for value in values)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _clamp_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        max(0, min(width, x0)),
        max(0, min(height, y0)),
        max(0, min(width, x1)),
        max(0, min(height, y1)),
    )


def _safe_cache_fragment(value: str) -> str:
    fragment = "".join(ch if ch.isalnum() else "_" for ch in value)
    fragment = fragment.strip("_")
    return fragment[:80] or "target"


def _image_data_url(path_value: str | None) -> str | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value).expanduser()
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _apply_selected_candidate(
    region: Any,
    *,
    selected_candidate: str,
) -> None:
    original_snapshot = {
        "text": region.text,
        "confidence": region.confidence,
    }
    if selected_candidate == "no_text":
        region.metadata["select_ocr_debug"] = {
            "original": original_snapshot,
            "selected_candidate": selected_candidate,
            "selected": None,
        }
        region.text = ""
        region.confidence = None
        region.metadata.pop("ocr_refinement_candidates", None)
        return

    candidates = region.metadata.get("ocr_refinement_candidates")
    if selected_candidate != "original":
        if not isinstance(candidates, dict):
            return
        candidate = candidates.get(selected_candidate)
        if not isinstance(candidate, dict):
            return
        text = candidate.get("text")
        if not isinstance(text, str):
            return
        confidence = candidate.get("confidence")
        region.metadata["select_ocr_debug"] = {
            "original": original_snapshot,
            "selected_candidate": selected_candidate,
            "selected": {
                "text": text,
                "confidence": (
                    float(confidence) if isinstance(confidence, (int, float)) else None
                ),
                "source": candidate.get("source"),
                "artifact_path": candidate.get("artifact_path"),
                "target_long_side_px": candidate.get("target_long_side_px"),
                "left_px": candidate.get("left_px"),
                "right_px": candidate.get("right_px"),
                "top_px": candidate.get("top_px"),
                "bottom_px": candidate.get("bottom_px"),
                "angle_degree": candidate.get("angle_degree"),
            },
        }
        region.text = text
        region.confidence = float(confidence) if isinstance(confidence, (int, float)) else None
    else:
        region.metadata["select_ocr_debug"] = {
            "original": original_snapshot,
            "selected_candidate": selected_candidate,
            "selected": {
                "text": region.text,
                "confidence": region.confidence,
            },
        }

    region.metadata.pop("ocr_refinement_candidates", None)
