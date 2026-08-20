"""LLM-backed OCR quality inspection."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from docclaw.agent.debug import dump_jsonl_from_env
from docclaw.agent.tool.tool import Tool
from docclaw.agent.tool.vlm_client import image_data_url
from docclaw.agent.utils import (
    Action,
    ActionType,
    Observation,
    RunState,
    page_index_from_id,
    plannerize_page_refs,
)
from docclaw.provider.base import LLMProvider


class LLMInspectOcrTool(Tool):
    """Inspect OCR/parsing outputs and propose refinement targets."""

    _JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    _OVERVIEW_MAX_SIDE_PX = 768

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        default_max_actions: int | None = None,
    ) -> None:
        self.provider = provider
        self.model = model or provider.get_default_model()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.default_max_actions = default_max_actions

    @property
    def action_type(self) -> ActionType:
        return "inspect_ocr"

    @property
    def description(self) -> str:
        return (
            "Inspect existing region-level OCR/parsing outputs and identify "
            "specific OCR/parsing-oriented regions that should be rerun with "
            "parameterized zoom, crop, or rotate actions."
        )

    @property
    def target_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": (
                "Optional focused page set or region set for OCR quality inspection. "
                "Use region_ids when specific regions are already known. Use "
                "page_ids to inspect all OCRed regions on those pages."
            ),
            "properties": {
                "page_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Focused page set for OCR quality inspection.",
                },
                "region_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Focused region set for OCR quality inspection.",
                },
            },
            "additionalProperties": False,
        }

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "OCR quality inspection controls.",
            "properties": {
                "max_actions": {
                    "type": "integer",
                    "description": "Maximum refinement actions to return.",
                },
                "max_refine_regions": {
                    "type": "integer",
                    "description": "Backward-compatible alias for max_actions.",
                },
            },
            "additionalProperties": False,
        }

    async def execute(self, state: RunState, action: Action) -> Observation:
        regions = _select_regions(
            state,
            target=action.target,
        )
        if not regions:
            return Observation(
                action_id=action.action_id,
                success=True,
                data={
                    "inspected_region_ids": [],
                    "refinement_actions": [],
                    "inspected_region_count": 0,
                    "source": "llm",
                },
                message="No OCR/parsing-oriented regions available for OCR quality inspection.",
            )

        messages, user_payload = self._build_messages(state, action, regions)
        dump_jsonl_from_env(
            "DOCCLAW_OCR_QUALITY_DEBUG_PATH",
            {
                "kind": "inspect_ocr_input",
                "model": self.model,
                "document_id": state.document.document_id,
                "run_id": state.run_id,
                "action_id": action.action_id,
                "task": state.task.to_dict(),
                "target": action.target,
                "parameters": action.parameters,
                "inspected_region_ids": [
                    str(region["region_id"])
                    for region in regions
                    if isinstance(region.get("region_id"), str)
                ],
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
            max_actions = _optional_int(action.parameters.get("max_actions"))
            if max_actions is None:
                max_actions = _optional_int(action.parameters.get("max_refine_regions"))
            if max_actions is None:
                max_actions = self.default_max_actions
            refinement_targets = self._refine_targets_from_payload(
                state,
                regions,
                payload,
                max_actions=max_actions,
            )
        except Exception as exc:
            return self.error(action, f"invalid OCR quality response: {exc}")
        dump_jsonl_from_env(
            "DOCCLAW_OCR_QUALITY_DEBUG_PATH",
            {
                "kind": "inspect_ocr_output",
                "model": self.model,
                "document_id": state.document.document_id,
                "run_id": state.run_id,
                "action_id": action.action_id,
                "raw_content": response.content,
                "parsed_payload": payload,
                "refinement_actions": refinement_targets["refinement_actions"],
                "usage": response.usage,
            },
        )

        message = (
            f"Inspected OCR quality for {len(regions)} region(s); "
            f"{len(refinement_targets['refinement_actions'])} refinement action(s) proposed."
        )
        return Observation(
            action_id=action.action_id,
            success=True,
            data={
                "inspected_region_ids": [
                    str(region["region_id"])
                    for region in regions
                    if isinstance(region.get("region_id"), str)
                ],
                "refinement_actions": refinement_targets["refinement_actions"],
                "inspected_region_count": len(regions),
                "source": "llm",
                "usage": response.usage,
            },
            message=message,
        )

    def update_state(
        self,
        state: RunState,
        action: Action,
        observation: Observation,
    ) -> None:
        inspected_region_ids = observation.data.get("inspected_region_ids")
        if not isinstance(inspected_region_ids, list):
            return
        for region_id in inspected_region_ids:
            if not isinstance(region_id, str) or not region_id.strip():
                continue
            state.mark_region_inspected(region_id)
        refinement_actions = observation.data.get("refinement_actions")
        if not isinstance(refinement_actions, list):
            return
        for item in refinement_actions:
            if not isinstance(item, dict):
                continue
            region_id = item.get("region_id")
            if not isinstance(region_id, str) or not region_id.strip():
                continue
            region = state.get_region(region_id)
            if region is None:
                continue
            existing = region.metadata.get("ocr_inspection_actions")
            history = list(existing) if isinstance(existing, list) else []
            history.append(json.loads(json.dumps(item, ensure_ascii=False)))
            region.metadata["ocr_inspection_actions"] = history

    def _build_messages(
        self,
        state: RunState,
        action: Action,
        regions: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        system_prompt = (
            "You are the DocClaw OCR quality inspection tool.\n"
            "Review region-level recognition output and identify only the regions "
            "that should be rerun with refinement actions.\n"
            "You will receive one page overview thumbnail per page, followed by one matching "
            "region crop for each region.\n"
            "Use the page overview to judge layout context, neighboring content, and whether "
            "a box may be too tight or misplaced. Use the matching region crop to judge "
            "whether the OCR text matches the visible content.\n"
            "Only choose OCR/parsing-oriented regions whose current OCR/parsing output looks "
            "unreliable enough that crop-based refinement is likely to help.\n"
            "If max_actions is provided in the input parameters, do not exceed it, "
            "but do not feel obliged to use all available slots.\n"
            "If a region has visible OCR/parsing errors and a crop-based refinement is likely "
            "to help, propose an action for it.\n"
            "Use zoom when the visible content is present but visually hard to read, such as "
            "small, dense, faint, blurry, low-contrast, or compressed text. Small or medium boxes "
            "with dense content are more suitable zoom candidates. Choose target_long_side_px with "
            "the bbox size in mind, and prefer roughly two to three times the current bbox "
            "long side. Overly large target_long_side_px can make the result worse.\n"
            "Use crop when the crop itself looks too tight or incomplete, such as clipped "
            "text, cut-off lines, incomplete cells, incomplete formulas, or missing prefixes "
            "or suffixes, or other nearby visible content that appears cut off by the current "
            "box. Use both the region crop and the page overview to judge which side is missing "
            "content. Then make conservative directional adjustments, using the bbox size as a "
            "rough reference for what counts as a small change. Overly large crop adjustments "
            "can make the result worse.\n"
            "Use rotate when orientation itself looks like a real OCR obstacle, especially "
            "for sideways or vertically arranged readable text that is likely to become "
            "readable after rotation.\n"
            "Return only JSON with this shape:\n"
            "{\n"
            '  "refinement_actions": [\n'
            "    {\n"
            '      "region_id": "region id",\n'
            '      "reason": "short_simple_reason",\n'
            '      "action": "crop",\n'
            '      "left_px": 0,\n'
            '      "right_px": 10,\n'
            '      "top_px": 0,\n'
            '      "bottom_px": 0\n'
            "    },\n"
            "    {\n"
            '      "region_id": "region id",\n'
            '      "reason": "short_simple_reason",\n'
            '      "action": "zoom",\n'
            '      "target_long_side_px": 1024\n'
            "    },\n"
            "    {\n"
            '      "region_id": "region id",\n'
            '      "reason": "short_simple_reason",\n'
            '      "action": "rotate",\n'
            '      "angle_degree": 90\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Requirements:\n"
            "- reason must be short and simple, preferably an underscore-separated phrase.\n"
            "- action must be one of: crop, zoom, rotate.\n"
            "- for crop, provide integer left_px/right_px/top_px/bottom_px. Use negative values to shrink.\n"
            "- for zoom, provide integer target_long_side_px.\n"
            "- for rotate, provide numeric angle_degree. Use positive values for clockwise rotation and negative values for counterclockwise rotation.\n"
            "- return at most one action per region.\n"
            "Do not invent region ids beyond the provided set.\n"
            "Return only actions that should be rerun."
        )
        region_payloads = [
            {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "page_overview_image_url",
                    "region_crop_image_url",
                    "source_image_path",
                    "page_width",
                    "page_height",
                }
            }
            for item in regions
        ]
        user_payload = plannerize_page_refs(
            {
                "task": state.task.to_dict(),
                "target": action.target,
                "parameters": action.parameters,
                "regions": region_payloads,
            },
            document=state.document,
        )
        user_prompt = json.dumps(
            user_payload,
            ensure_ascii=False,
            indent=2,
        )
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": user_prompt},
        ]
        page_indexes_in_order: list[int] = []
        seen_page_indexes: set[int] = set()
        for item in regions:
            page_index = item.get("page_index")
            if isinstance(page_index, int) and page_index not in seen_page_indexes:
                page_indexes_in_order.append(page_index)
                seen_page_indexes.add(page_index)

        for page_index in page_indexes_in_order:
            overview_image_url = next(
                (
                    item.get("page_overview_image_url")
                    for item in regions
                    if item.get("page_index") == page_index
                    and isinstance(item.get("page_overview_image_url"), str)
                    and item.get("page_overview_image_url")
                ),
                None,
            )
            if isinstance(overview_image_url, str) and overview_image_url:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"Page {page_index + 1}: page overview thumbnail with all inspected regions highlighted and labeled by region id.",
                    }
                )
                user_content.append(
                    {"type": "image_url", "image_url": {"url": overview_image_url}}
                )

        for item in regions:
            region_id = str(item.get("region_id") or "")
            if region_id:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"Region {region_id}: region crop.",
                    }
                )
            image_url = item.get("region_crop_image_url")
            if isinstance(image_url, str) and image_url:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": image_url}}
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

    def _refine_targets_from_payload(
        self,
        state: RunState,
        inspected_regions: list[dict[str, Any]],
        payload: dict[str, Any],
        *,
        max_actions: int | None,
    ) -> dict[str, Any]:
        allowed_region_ids = {
            str(item["region_id"])
            for item in inspected_regions
            if isinstance(item.get("region_id"), str)
        }
        refinement_actions = _normalize_refinement_actions(
            state=state,
            allowed_region_ids=allowed_region_ids,
            payload=payload,
        )
        if max_actions is not None:
            refinement_actions = refinement_actions[:max_actions]
        return {
            "refinement_actions": refinement_actions,
        }


def _select_regions(
    state: RunState,
    *,
    target: dict[str, Any],
) -> list[dict[str, Any]]:
    page_indices = _normalize_page_indices(
        target.get("page_indices")
        if "page_indices" in target
        else [page_index_from_id(str(item), document=state.document) for item in target.get("page_ids", [])]
    )
    region_ids = _normalize_region_ids(target.get("region_ids"))

    focused_page_indexes: set[int] = set()
    for page_ref in page_indices:
        focused_page_indexes.add(page_ref)
    if region_ids:
        for region_id in region_ids:
            region = state.get_region(region_id)
            if region is not None:
                focused_page_indexes.add(region.page_index)

    results: list[dict[str, Any]] = []
    for page in state.document.pages:
        if focused_page_indexes and page.page_index not in focused_page_indexes:
            continue
        page_results: list[dict[str, Any]] = []
        for region in page.regions:
            if region_ids and region.region_id not in region_ids:
                continue
            if not _is_refinable_region(region):
                continue
            page_results.append(
                {
                    "page_index": page.page_index,
                    "region_id": region.region_id,
                    "type": region.type,
                    "bbox": list(region.bbox),
                    "coordinate_space": region.coordinate_space,
                    "text": region.text if isinstance(region.text, str) else "",
                    "page_width": page.width,
                    "page_height": page.height,
                    "source_image_path": page.image_path,
                    "region_crop_image_url": _region_crop_image_url(
                        page_image_path=page.image_path,
                        bbox=tuple(region.bbox),
                        coordinate_space=str(region.coordinate_space or "pixel"),
                        page_width=page.width,
                        page_height=page.height,
                    ),
                }
            )
        if not page_results:
            continue
        overview_image_url = _page_overview_image_url(
            page_image_path=page.image_path,
            regions=page_results,
            page_width=page.width,
            page_height=page.height,
            max_side_px=LLMInspectOcrTool._OVERVIEW_MAX_SIDE_PX,
        )
        for item in page_results:
            item["page_overview_image_url"] = overview_image_url
        results.extend(page_results)
    return results


def _normalize_page_indices(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    page_indices: list[int] = []
    seen: set[int] = set()
    for item in value:
        maybe = _optional_int(item)
        if maybe is None or maybe < 0 or maybe in seen:
            continue
        page_indices.append(maybe)
        seen.add(maybe)
    return page_indices


def _normalize_region_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    region_ids: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        region_id = item.strip()
        if not region_id or region_id in seen:
            continue
        region_ids.append(region_id)
        seen.add(region_id)
    return region_ids


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _filter_requested_region_ids(
    state: RunState,
    *,
    allowed_region_ids: set[str],
    requested_region_ids: list[str],
) -> list[str]:
    results: list[str] = []
    for region_id in requested_region_ids:
        if region_id not in allowed_region_ids:
            continue
        if state.get_region(region_id) is None:
            continue
        results.append(region_id)
    return results


def _normalize_refinement_actions(
    *,
    state: RunState,
    allowed_region_ids: set[str],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_actions = payload.get("refinement_actions")
    if not isinstance(raw_actions, list):
        raise ValueError("response JSON must contain refinement_actions as a list")
    return _filter_requested_actions(
        state,
        allowed_region_ids=allowed_region_ids,
        requested_actions=raw_actions,
    )


def _filter_requested_actions(
    state: RunState,
    *,
    allowed_region_ids: set[str],
    requested_actions: list[Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen_region_ids: set[str] = set()
    for item in requested_actions:
        normalized = _normalize_refinement_action(item)
        if normalized is None:
            continue
        region_id = normalized["region_id"]
        if region_id not in allowed_region_ids:
            continue
        if state.get_region(region_id) is None or region_id in seen_region_ids:
            continue
        results.append(normalized)
        seen_region_ids.add(region_id)
    return results


def _normalize_refinement_action(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    raw_region_id = value.get("region_id")
    if not isinstance(raw_region_id, str) or not raw_region_id.strip():
        return None
    raw_action = value.get("action")
    if not isinstance(raw_action, str):
        return None
    action = raw_action.strip().lower()
    if action not in {"crop", "zoom", "rotate"}:
        return None
    normalized: dict[str, Any] = {
        "region_id": raw_region_id.strip(),
        "reason": _normalize_reason(value.get("reason")),
        "action": action,
    }
    if action == "crop":
        normalized.update(
            {
                "left_px": _optional_int(value.get("left_px")) or 0,
                "right_px": _optional_int(value.get("right_px")) or 0,
                "top_px": _optional_int(value.get("top_px")) or 0,
                "bottom_px": _optional_int(value.get("bottom_px")) or 0,
            }
        )
        return normalized
    if action == "zoom":
        target_long_side_px = _optional_int(value.get("target_long_side_px"))
        if target_long_side_px is None or target_long_side_px <= 0:
            return None
        normalized["target_long_side_px"] = target_long_side_px
        return normalized
    angle_degree = _optional_float(value.get("angle_degree"))
    if angle_degree is None:
        return None
    normalized["angle_degree"] = angle_degree
    return normalized


def _normalize_reason(value: Any) -> str:
    if not isinstance(value, str):
        return "needs_refinement"
    compact = "_".join(value.strip().split())
    compact = re.sub(r"[^a-zA-Z0-9_-]", "", compact)
    return compact[:64] or "needs_refinement"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _page_overview_image_url(
    *,
    page_image_path: str | None,
    regions: list[dict[str, Any]],
    page_width: int | None,
    page_height: int | None,
    max_side_px: int,
) -> str | None:
    image = _render_page_overview_thumbnail(
        page_image_path=page_image_path,
        regions=regions,
        page_width=page_width,
        page_height=page_height,
        max_side_px=max_side_px,
    )
    if image is None:
        return None
    return image_data_url(image)


def _region_crop_image_url(
    *,
    page_image_path: str | None,
    bbox: tuple[float, float, float, float],
    coordinate_space: str,
    page_width: int | None,
    page_height: int | None,
) -> str | None:
    image = _render_region_crop(
        page_image_path=page_image_path,
        bbox=bbox,
        coordinate_space=coordinate_space,
        page_width=page_width,
        page_height=page_height,
    )
    if image is None:
        return None
    return image_data_url(image)


def _render_page_overview_thumbnail(
    *,
    page_image_path: str | None,
    regions: list[dict[str, Any]],
    page_width: int | None,
    page_height: int | None,
    max_side_px: int,
) -> Any:
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError:
        return None
    source_path = _resolve_existing_path(page_image_path)
    if source_path is None:
        return None
    pixel_regions: list[tuple[str, tuple[int, int, int, int]]] = []
    for item in regions:
        region_id = str(item.get("region_id") or "").strip()
        bbox = item.get("bbox")
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            continue
        coordinate_space = str(item.get("coordinate_space") or "pixel")
        pixel_bbox = _to_pixel_bbox(tuple(bbox), coordinate_space, page_width, page_height)
        if pixel_bbox is None:
            continue
        pixel_regions.append((region_id, pixel_bbox))
    if not pixel_regions:
        return None
    try:
        with Image.open(source_path) as image:
            canvas = image.convert("RGB")
            width, height = canvas.size
            scale = min(
                1.0,
                float(max_side_px) / float(max(width, height)),
            )
            if scale < 1.0:
                resized = canvas.resize(
                    (
                        max(1, int(round(width * scale))),
                        max(1, int(round(height * scale))),
                    )
                )
                canvas = resized
            draw = ImageDraw.Draw(canvas)
            outline_width = max(2, int(round(4 * scale)) if scale > 0 else 2)
            for region_id, pixel_bbox in pixel_regions:
                x0, y0, x1, y1 = pixel_bbox
                scaled_bbox = (
                    int(round(x0 * scale)),
                    int(round(y0 * scale)),
                    int(round(x1 * scale)),
                    int(round(y1 * scale)),
                )
                draw.rectangle(scaled_bbox, outline=(255, 128, 0), width=outline_width)
                if region_id:
                    label_y = max(0, scaled_bbox[1] - 14)
                    draw.text((scaled_bbox[0], label_y), region_id, fill=(255, 128, 0))
            return canvas
    except Exception:
        return None


def _render_region_crop(
    *,
    page_image_path: str | None,
    bbox: tuple[float, float, float, float],
    coordinate_space: str,
    page_width: int | None,
    page_height: int | None,
) -> Any:
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return None
    source_path = _resolve_existing_path(page_image_path)
    if source_path is None:
        return None
    pixel_bbox = _to_pixel_bbox(bbox, coordinate_space, page_width, page_height)
    if pixel_bbox is None:
        return None
    try:
        with Image.open(source_path) as image:
            width, height = image.size
            x0, y0, x1, y1 = _clamp_bbox(pixel_bbox, width, height)
            if x1 <= x0 or y1 <= y0:
                return None
            return image.crop((x0, y0, x1, y1)).copy()
    except Exception:
        return None


def _resolve_existing_path(path_value: str | None) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value).expanduser()
    if not path.exists():
        return None
    return path


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


def _is_refinable_region(region: Any) -> bool:
    normalized_type = str(region.type or "").strip().lower()
    if normalized_type == "chart":
        return False
    render = region.metadata.get("render")
    if isinstance(render, dict) and isinstance(render.get("image_like"), bool):
        if bool(render.get("image_like")):
            return False
    return True
