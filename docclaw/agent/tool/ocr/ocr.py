"""Abstract OCR tool."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

from docclaw.agent.tool.layout.ppdoclayout import (
    _ensure_post_layout_groups,
    _is_ignorable_for_markdown,
    _is_image_like_label,
    _markdown_label_for_layout_label,
    _normalize_layout_label,
    _normalize_region_type,
    _render_kind_for_label,
    _safe_id,
    _unique_region_id,
)
from docclaw.agent.tool.ocr.assemble import assemble_complete_region_page_text
from docclaw.agent.tool.tool import Tool
from docclaw.agent.utils import (
    Action,
    ActionType,
    Observation,
    Region,
    RunState,
    has_searchable_text,
    page_id_from_index,
    page_number_from_index,
)


PixelBBox = tuple[int, int, int, int]
OCR_REGION_CROP_PADDING_PX = 2
PAGE_OCR_LAYOUT_SOURCE = "paddleocr_vl_page_ocr"


class OcrTool(Tool):
    """Base class for tools that recognize text from pages or regions."""

    def __init__(
        self,
        *,
        artifact_dir: str | Path | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None

    @property
    def action_type(self) -> ActionType:
        return "ocr"

    @property
    def description(self) -> str:
        return (
            "Recognize text from a page set, region set, zoomed region set, "
            "crop-adjusted region set, or rotated region set. "
            "OCR returns normalized text results with source and confidence, and writes "
            "searchable OCR text back into document state for later retrieval and "
            "reasoning. It only supports known page/region targets provided as explicit "
            "sets. Repeating OCR on the same pages or regions has no value, but "
            "other pages or regions may not yet have been OCRed. Use page_ids "
            "exposed in document memory for page-level OCR, use known region_ids "
            "from document memory or recent action_history outputs for region-level "
            "OCR, and use action_history to avoid repeating OCR on targets that "
            "already have searchable text. Region ids encode page number, region "
            "type, and local order, for example p14_chart_13 or p7_text_5."
        )

    @property
    def target_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": (
                "Select exactly one target mode, setting mode explicitly. "
                "Use page_ids for full-page OCR "
                "over those pages. Use region_ids only for specific known regions. "
                "If you want all content from selected pages, use page_ids and "
                "set region_ids to [] or omit it entirely. Do not use placeholder "
                "values such as '__all__', 'string', empty strings, or numeric dummy values."
            ),
            "properties": {
                "mode": {
                    "type": "string",
                    "description": (
                        "Explicit OCR target mode. Use 'page' for page_ids, "
                        "'region' for region_ids, 'zoom_region' for zoom_region_ids, "
                        "'crop_region' for crop_region_ids, and 'rotate_region' "
                        "for rotate_region_ids."
                    ),
                    "enum": ["page", "region", "zoom_region", "crop_region", "rotate_region"],
                },
                "region_ids": {
                    "type": "array",
                    "description": "Known region identifiers to OCR as an explicit region set. When using page_ids, set this to [] or omit it.",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "zoom_region_ids": {
                    "type": "array",
                    "description": "Known region identifiers whose zoom artifacts should be OCRed as an explicit region set.",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "crop_region_ids": {
                    "type": "array",
                    "description": "Known region identifiers whose crop artifacts should be OCRed as an explicit region set.",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "rotate_region_ids": {
                    "type": "array",
                    "description": "Known region identifiers whose rotate artifacts should be OCRed as an explicit region set.",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "page_ids": {
                    "type": "array",
                    "description": "Page ids for full-page OCR as an explicit page set. When using page_ids, region_ids should be [] or omitted.",
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
            "description": "OCR controls.",
            "properties": {
                "artifact_dir": {
                    "type": "string",
                    "description": "Directory for temporary OCR crop artifacts.",
                },
            },
            "additionalProperties": False,
        }

    def document_overview_fragment(self, state: RunState) -> dict[str, Any]:
        all_page_ids = [page_id_from_index(page.page_index, document=state.document) for page in state.document.pages]
        page_indexes_with_page_ocr = [
            page_id_from_index(page.page_index, document=state.document)
            for page in state.document.pages
            if _has_canonical_page_ocr(page)
        ]
        page_indexes_with_region_ocr = sorted(
            {
                page_id_from_index(page.page_index, document=state.document)
                for page in state.document.pages
                if any(has_searchable_text(region.text) for region in page.regions)
            }
        )
        refinement_candidate_entries = sorted(
            [
                {
                    "region_id": region.region_id,
                    "page_index": page.page_index,
                    "type": region.type,
                    "candidate_kinds": sorted(
                        [
                            candidate_kind
                            for candidate_kind, candidate_payload in candidates.items()
                            if isinstance(candidate_kind, str)
                            and candidate_kind.strip()
                            and isinstance(candidate_payload, dict)
                            and isinstance(candidate_payload.get("text"), str)
                        ]
                    ),
                }
                for page in state.document.pages
                for region in page.regions
                if isinstance((candidates := region.metadata.get("ocr_refinement_candidates")), dict)
            ],
            key=lambda item: (
                int(item["page_index"]),
                str(item["region_id"]),
            ),
        )
        region_ocr_entries = sorted(
            [
                {
                    "region_id": region.region_id,
                    "page_index": page.page_index,
                    "type": region.type,
                    "confidence": (
                        float(region.confidence)
                        if isinstance(region.confidence, (int, float))
                        else None
                    ),
                }
                for page in state.document.pages
                for region in page.regions
                if has_searchable_text(region.text)
            ],
            key=lambda item: (
                int(item["page_index"]),
                str(item["region_id"]),
            ),
        )
        return {
            "ocr": {
                "pages_have_been_page_ocred": page_indexes_with_page_ocr,
                "pages_have_not_been_page_ocred": [
                    page_id
                    for page_id in all_page_ids
                    if page_id not in set(page_indexes_with_page_ocr)
                ],
                "pages_have_been_region_ocred": page_indexes_with_region_ocr,
                "pages_have_not_been_region_ocred": [
                    page_id
                    for page_id in all_page_ids
                    if page_id not in set(page_indexes_with_region_ocr)
                ],
                "regions_have_been_ocred": [
                    item["region_id"] for item in region_ocr_entries
                ],
                "refinement_candidate_region_ids": [
                    item["region_id"] for item in refinement_candidate_entries
                ],
                "refinement_candidates": refinement_candidate_entries,
            }
        }

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
            result, target_artifacts, error = self._execute_single_target(
                state,
                action,
                target,
                force=force,
            )
            if error is not None:
                return self.error(action, error)
            assert result is not None
            results.append(result)
            artifacts.extend(target_artifacts)
            source = str(result.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1

        if len(results) == 1:
            message = _ocr_message(results[0], source_counts, document=state.document)
        else:
            message = _batch_ocr_message(results, source_counts, document=state.document)

        return Observation(
            action_id=action.action_id,
            success=True,
            data={
                "results": results,
                "sources": source_counts,
            },
            message=message,
            artifacts=artifacts,
        )

    def _execute_single_target(
        self,
        state: RunState,
        action: Action,
        target: dict[str, Any],
        *,
        force: bool,
    ) -> tuple[dict[str, Any] | None, list[str], str | None]:
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
            result = self.recognize_text(source_path)
        except Exception as exc:
            return None, [], f"ocr failed: {exc}"

        text = result.get("text", "")
        if not isinstance(text, str):
            text = str(text)

        data = {
            **target["data"],
            "text": text,
            "source": self.source_name,
            "confidence": result.get("confidence"),
        }

        return data, artifacts, None

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Short source name to store in observations."""

    @abstractmethod
    def recognize_text(self, image_path: Path) -> dict[str, Any]:
        """Run OCR on an image path and return normalized OCR data."""

    @abstractmethod
    def render_crop(
        self,
        source_path: Path,
        output_path: Path,
        bbox: PixelBBox,
    ) -> None:
        """Write a cropped image for region OCR."""

    def update_state(
        self,
        state: RunState,
        action: Action,
        observation: Observation,
    ) -> None:
        results = observation.data.get("results")
        if isinstance(results, list):
            for result in results:
                if isinstance(result, dict):
                    _apply_ocr_result(state, result)
            return
        _apply_ocr_result(state, observation.data)

    def _prepare_ocr_image(
        self,
        state: RunState,
        action: Action,
        target: dict[str, Any],
    ) -> tuple[Path | None, list[str], str | None]:
        artifact_path = target.get("artifact_path")
        if isinstance(artifact_path, str) and artifact_path.strip():
            source_path = Path(artifact_path).expanduser()
            if not source_path.exists():
                return None, [], f"ocr artifact_path does not exist: {source_path}"
            return source_path, [], None

        page = state.require_page(target["page_index"])
        if not page.image_path:
            return None, [], f"page {page.page_index} has no image_path"
        source_path = Path(page.image_path).expanduser()
        if not source_path.exists():
            return None, [], f"page image_path does not exist: {source_path}"

        pixel_bbox = target.get("pixel_bbox")
        if pixel_bbox is None:
            return source_path, [], None

        artifact_dir = _resolve_artifact_dir(state, action, self.artifact_dir)
        if artifact_dir is None:
            return None, [], "ocr region target requires artifact_dir"
        output_path = artifact_dir / f"{action.action_id}_{_safe_cache_fragment(str(target['artifact_name']))}_ocr.png"
        padded_pixel_bbox = _expand_bbox(
            pixel_bbox,
            padding=OCR_REGION_CROP_PADDING_PX,
            width=page.width,
            height=page.height,
        )
        self.render_crop(source_path, output_path, padded_pixel_bbox)
        return output_path, [str(output_path)], None


def _resolve_ocr_targets(
    state: RunState,
    action: Action,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if "region_id" in action.target or "page_index" in action.target:
        return None, "ocr requires target.region_ids, target.zoom_region_ids, target.crop_region_ids, target.rotate_region_ids, or target.page_ids"

    mode, error = _optional_ocr_mode(action.target.get("mode"))
    if error is not None:
        return None, error

    raw_region_ids = action.target.get("region_ids")
    cleaned_region_ids = (
        [
            region_id
            for raw_region_id in raw_region_ids
            if (region_id := _optional_non_empty_string(raw_region_id)) is not None
        ]
        if isinstance(raw_region_ids, list)
        else None
    )
    raw_zoom_region_ids = action.target.get("zoom_region_ids")
    cleaned_zoom_region_ids = (
        [
            region_id
            for raw_region_id in raw_zoom_region_ids
            if (region_id := _optional_non_empty_string(raw_region_id)) is not None
        ]
        if isinstance(raw_zoom_region_ids, list)
        else None
    )
    raw_crop_region_ids = action.target.get("crop_region_ids")
    cleaned_crop_region_ids = (
        [
            region_id
            for raw_region_id in raw_crop_region_ids
            if (region_id := _optional_non_empty_string(raw_region_id)) is not None
        ]
        if isinstance(raw_crop_region_ids, list)
        else None
    )
    raw_rotate_region_ids = action.target.get("rotate_region_ids")
    cleaned_rotate_region_ids = (
        [
            region_id
            for raw_region_id in raw_rotate_region_ids
            if (region_id := _optional_non_empty_string(raw_region_id)) is not None
        ]
        if isinstance(raw_rotate_region_ids, list)
        else None
    )

    if mode == "region":
        if action.target.get("page_indices") is not None and not cleaned_region_ids:
            return _resolve_region_targets_for_pages(state, action.target)
        return _resolve_explicit_region_targets(state, action.target, cleaned_region_ids)
    if mode == "zoom_region":
        return _resolve_explicit_zoom_region_targets(state, action.target, cleaned_zoom_region_ids)
    if mode == "crop_region":
        return _resolve_explicit_crop_region_targets(state, action.target, cleaned_crop_region_ids)
    if mode == "rotate_region":
        return _resolve_explicit_rotate_region_targets(state, action.target, cleaned_rotate_region_ids)
    if mode == "page":
        return _resolve_explicit_page_targets(state, action.target)

    has_page_indices = action.target.get("page_indices") is not None
    has_zoom_region_ids = bool(cleaned_zoom_region_ids)
    has_crop_region_ids = bool(cleaned_crop_region_ids)
    has_rotate_region_ids = bool(cleaned_rotate_region_ids)
    if has_page_indices and cleaned_region_ids:
        cleaned_region_ids = [
            region_id for region_id in cleaned_region_ids if state.get_region(region_id) is not None
        ]
    if has_page_indices and cleaned_zoom_region_ids:
        cleaned_zoom_region_ids = [
            region_id
            for region_id in cleaned_zoom_region_ids
            if state.get_zoom_region_view(region_id) is not None
        ]
        has_zoom_region_ids = bool(cleaned_zoom_region_ids)
    if has_page_indices and cleaned_crop_region_ids:
        cleaned_crop_region_ids = [
            region_id
            for region_id in cleaned_crop_region_ids
            if state.get_crop_region_view(region_id) is not None
        ]
        has_crop_region_ids = bool(cleaned_crop_region_ids)
    if has_page_indices and cleaned_rotate_region_ids:
        cleaned_rotate_region_ids = [
            region_id
            for region_id in cleaned_rotate_region_ids
            if state.get_rotate_region_view(region_id) is not None
        ]
        has_rotate_region_ids = bool(cleaned_rotate_region_ids)
    has_region_ids = bool(cleaned_region_ids)
    mode_count = (
        int(has_region_ids)
        + int(has_zoom_region_ids)
        + int(has_crop_region_ids)
        + int(has_rotate_region_ids)
        + int(has_page_indices)
    )
    if mode_count > 1:
        return None, "ocr target must specify exactly one of target.region_ids, target.zoom_region_ids, target.crop_region_ids, target.rotate_region_ids, or target.page_ids"

    region_ids = action.target.get("region_ids")
    if region_ids is not None:
        return _resolve_explicit_region_targets(state, action.target, cleaned_region_ids)

    zoom_region_ids = action.target.get("zoom_region_ids")
    if zoom_region_ids is not None:
        return _resolve_explicit_zoom_region_targets(state, action.target, cleaned_zoom_region_ids)

    crop_region_ids = action.target.get("crop_region_ids")
    if crop_region_ids is not None:
        return _resolve_explicit_crop_region_targets(state, action.target, cleaned_crop_region_ids)

    rotate_region_ids = action.target.get("rotate_region_ids")
    if rotate_region_ids is not None:
        return _resolve_explicit_rotate_region_targets(state, action.target, cleaned_rotate_region_ids)

    page_indices = action.target.get("page_indices")
    if page_indices is not None:
        return _resolve_explicit_page_targets(state, action.target)

    return None, "ocr requires target.region_ids, target.zoom_region_ids, target.crop_region_ids, target.rotate_region_ids, or target.page_ids"


def _resolve_explicit_region_targets(
    state: RunState,
    target_payload: dict[str, Any],
    cleaned_region_ids: list[str] | None,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    region_ids = target_payload.get("region_ids")
    if not isinstance(region_ids, list) or not region_ids:
        return None, "ocr target.region_ids must be a non-empty list"
    if not cleaned_region_ids:
        return None, "ocr target.region_ids must contain non-empty strings"
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for region_id in cleaned_region_ids:
        if region_id in seen:
            continue
        target, error = _resolve_region_ocr_target(state, region_id)
        if error is not None:
            return None, error
        assert target is not None
        targets.append(target)
        seen.add(region_id)
    return targets, None


def _resolve_region_targets_for_pages(
    state: RunState,
    target_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    page_targets, error = _resolve_explicit_page_targets(state, target_payload)
    if error is not None:
        return None, error
    assert page_targets is not None

    targets: list[dict[str, Any]] = []
    for page_target in page_targets:
        page = state.require_page(int(page_target["page_index"]))
        for region in page.regions:
            target, region_error = _resolve_region_ocr_target(state, region.region_id)
            if region_error is not None:
                return None, region_error
            assert target is not None
            targets.append(target)

    if not targets:
        page_indices = target_payload.get("page_indices")
        return None, (
            "ocr region-mode page_ids resolved no known regions; "
            f"run parse_layout first or use page mode for pages "
            f"{[page_id_from_index(page_index, document=state.document) for page_index in page_indices] if isinstance(page_indices, list) else page_indices}"
        )

    return targets, None


def _resolve_explicit_zoom_region_targets(
    state: RunState,
    target_payload: dict[str, Any],
    cleaned_zoom_region_ids: list[str] | None,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    zoom_region_ids = target_payload.get("zoom_region_ids")
    if not isinstance(zoom_region_ids, list) or not zoom_region_ids:
        return None, "ocr target.zoom_region_ids must be a non-empty list"
    if not cleaned_zoom_region_ids:
        return None, "ocr target.zoom_region_ids must contain non-empty strings"
    targets = []
    seen: set[str] = set()
    for region_id in cleaned_zoom_region_ids:
        if region_id in seen:
            continue
        target, error = _resolve_zoom_region_ocr_target(state, region_id)
        if error is not None:
            return None, error
        assert target is not None
        targets.append(target)
        seen.add(region_id)
    return targets, None


def _resolve_explicit_crop_region_targets(
    state: RunState,
    target_payload: dict[str, Any],
    cleaned_crop_region_ids: list[str] | None,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    crop_region_ids = target_payload.get("crop_region_ids")
    if not isinstance(crop_region_ids, list) or not crop_region_ids:
        return None, "ocr target.crop_region_ids must be a non-empty list"
    if not cleaned_crop_region_ids:
        return None, "ocr target.crop_region_ids must contain non-empty strings"
    targets = []
    seen: set[str] = set()
    for region_id in cleaned_crop_region_ids:
        if region_id in seen:
            continue
        target, error = _resolve_crop_region_ocr_target(state, region_id)
        if error is not None:
            return None, error
        assert target is not None
        targets.append(target)
        seen.add(region_id)
    return targets, None


def _resolve_explicit_rotate_region_targets(
    state: RunState,
    target_payload: dict[str, Any],
    cleaned_rotate_region_ids: list[str] | None,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    rotate_region_ids = target_payload.get("rotate_region_ids")
    if not isinstance(rotate_region_ids, list) or not rotate_region_ids:
        return None, "ocr target.rotate_region_ids must be a non-empty list"
    if not cleaned_rotate_region_ids:
        return None, "ocr target.rotate_region_ids must contain non-empty strings"
    targets = []
    seen: set[str] = set()
    for region_id in cleaned_rotate_region_ids:
        if region_id in seen:
            continue
        target, error = _resolve_rotate_region_ocr_target(state, region_id)
        if error is not None:
            return None, error
        assert target is not None
        targets.append(target)
        seen.add(region_id)
    return targets, None


def _resolve_explicit_page_targets(
    state: RunState,
    target_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    page_indices = target_payload.get("page_indices")
    if not isinstance(page_indices, list) or not page_indices:
        return None, "ocr target.page_ids must be a non-empty list"
    cleaned_page_indices: list[int] = []
    for raw_page_index in page_indices:
        page_index, error = _optional_page_index(raw_page_index)
        if error is not None:
            return None, error
        if page_index is not None:
            cleaned_page_indices.append(page_index)
    if not cleaned_page_indices:
        return None, "ocr target.page_ids must contain valid page ids"
    targets = []
    seen: set[int] = set()
    for page_index in cleaned_page_indices:
        if page_index in seen:
            continue
        target, error = _resolve_page_ocr_target(state, page_index)
        if error is not None:
            return None, error
        assert target is not None
        targets.append(target)
        seen.add(page_index)
    return targets, None


def _resolve_region_ocr_target(
    state: RunState,
    region_id: str,
) -> tuple[dict[str, Any] | None, str | None]:
    region = state.get_region(region_id)
    if region is None:
        return None, f"unknown region_id: {region_id}"
    page = state.require_page(region.page_index)
    pixel_bbox, error = _to_pixel_bbox(
        region.bbox,
        region.coordinate_space,
        page.width,
        page.height,
    )
    if error is not None:
        return None, error
    return {
        "page_index": region.page_index,
        "pixel_bbox": pixel_bbox,
        "artifact_name": region.region_id,
        "data": {"page_index": region.page_index, "region_id": region.region_id},
    }, None


def _resolve_page_ocr_target(
    state: RunState,
    page_index: int,
) -> tuple[dict[str, Any] | None, str | None]:
    page = state.get_page(page_index)
    if page is None:
        return None, f"unknown page_index: {page_index}"
    return {
        "page_index": page.page_index,
        "artifact_name": f"page_{page.page_index}",
        "data": {
            "page_index": page.page_index,
            "region_id": None,
        },
    }, None


def _resolve_zoom_region_ocr_target(
    state: RunState,
    region_id: str,
) -> tuple[dict[str, Any] | None, str | None]:
    region = state.get_region(region_id)
    if region is None:
        return None, f"unknown region_id: {region_id}"
    zoom_view = state.get_zoom_region_view(region_id)
    if zoom_view is None:
        return None, f"unknown zoom_region_id: {region_id}"
    return {
        "page_index": region.page_index,
        "artifact_name": f"zoom_region_{region_id}",
        "artifact_path": zoom_view.artifact_path,
        "data": {
            "page_index": region.page_index,
            "region_id": region.region_id,
            "zoom_artifact_path": zoom_view.artifact_path,
            "target_long_side_px": zoom_view.target_long_side_px,
        },
    }, None


def _resolve_crop_region_ocr_target(
    state: RunState,
    region_id: str,
) -> tuple[dict[str, Any] | None, str | None]:
    region = state.get_region(region_id)
    if region is None:
        return None, f"unknown region_id: {region_id}"
    crop_view = state.get_crop_region_view(region_id)
    if crop_view is None:
        return None, f"unknown crop_region_id: {region_id}"
    return {
        "page_index": region.page_index,
        "artifact_name": f"crop_region_{region_id}",
        "artifact_path": crop_view.artifact_path,
        "data": {
            "page_index": region.page_index,
            "region_id": region.region_id,
            "crop_artifact_path": crop_view.artifact_path,
            "left_px": crop_view.left_px,
            "right_px": crop_view.right_px,
            "top_px": crop_view.top_px,
            "bottom_px": crop_view.bottom_px,
        },
    }, None


def _resolve_rotate_region_ocr_target(
    state: RunState,
    region_id: str,
) -> tuple[dict[str, Any] | None, str | None]:
    region = state.get_region(region_id)
    if region is None:
        return None, f"unknown region_id: {region_id}"
    rotate_view = state.get_rotate_region_view(region_id)
    if rotate_view is None:
        return None, f"unknown rotate_region_id: {region_id}"
    return {
        "page_index": region.page_index,
        "artifact_name": f"rotate_region_{region_id}",
        "artifact_path": rotate_view.artifact_path,
        "data": {
            "page_index": region.page_index,
            "region_id": region.region_id,
            "rotate_artifact_path": rotate_view.artifact_path,
            "angle_degree": rotate_view.angle_degree,
        },
    }, None


def _to_pixel_bbox(
    bbox: tuple[float, float, float, float],
    coordinate_space: str,
    page_width: int | None,
    page_height: int | None,
) -> tuple[PixelBBox, str | None]:
    if len(bbox) != 4:
        return (0, 0, 0, 0), "bbox must contain four coordinates"

    if coordinate_space == "relative":
        if page_width is None or page_height is None:
            return (0, 0, 0, 0), "relative bbox requires page width and height"
        values = (
            float(bbox[0]) * page_width,
            float(bbox[1]) * page_height,
            float(bbox[2]) * page_width,
            float(bbox[3]) * page_height,
        )
    elif coordinate_space == "pixel":
        values = tuple(float(value) for value in bbox)
    else:
        return (0, 0, 0, 0), f"unsupported coordinate_space: {coordinate_space}"

    x0, y0, x1, y1 = (int(round(value)) for value in values)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0), "bbox must have positive width and height"

    if page_width is not None and page_height is not None:
        x0, y0, x1, y1 = clamp_bbox((x0, y0, x1, y1), page_width, page_height)
        if x1 <= x0 or y1 <= y0:
            return (0, 0, 0, 0), "bbox is outside the page bounds"

    return (x0, y0, x1, y1), None


def _ocr_message(
    result: dict[str, Any],
    source_counts: dict[str, int],
    *,
    document: Any | None = None,
) -> str:
    page_index = result.get("page_index")
    region_id = result.get("region_id")
    text = result.get("text")
    source = next(iter(source_counts.keys()), "unknown")
    if isinstance(region_id, str) and region_id:
        target_name = f"region {region_id}"
    elif isinstance(page_index, int):
        target_name = f"page_id {page_id_from_index(page_index, document=document)}"
    else:
        target_name = "target"
    return f"OCR ready for {target_name} (chars={len(text) if isinstance(text, str) else 0}, source={source})."


def _batch_ocr_message(
    results: list[dict[str, Any]],
    source_counts: dict[str, int],
    *,
    document: Any | None = None,
) -> str:
    page_indexes = sorted(
        {
            page_index
            for result in results
            if isinstance((page_index := result.get("page_index")), int)
        }
    )
    page_ids = [page_id_from_index(page_index, document=document) for page_index in page_indexes]
    page_label = ", ".join(page_ids)
    source_summary = ", ".join(
        f"{count} from {source}"
        for source, count in sorted(source_counts.items())
    )
    return (
        f"OCR ready for {len(results)} target(s) across {len(page_indexes)} page(s) ({page_label})"
        + (f" ({source_summary})" if source_summary else "")
        + "."
    )


def _optional_non_empty_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"all", "dummy", "string", ":all:", ":dummy:", ":string:", "__all__", "__dummy__", "__string__"}:
        return None
    if not any(ch.isalnum() for ch in text):
        return None
    return text


def _optional_ocr_mode(value: Any) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    text = str(value).strip()
    if not text:
        return None, None
    if text not in {"page", "region", "zoom_region", "crop_region", "rotate_region"}:
        return None, "ocr target.mode must be one of 'page', 'region', 'zoom_region', 'crop_region', or 'rotate_region'"
    return text, None


def _optional_page_index(value: Any) -> tuple[int | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, "ocr target.page_ids must contain valid page ids"
    if isinstance(value, int):
        return value, None
    text = str(value).strip()
    if not text:
        return None, None
    try:
        return int(text), None
    except ValueError:
        return None, "ocr target.page_ids must contain valid page ids"


def _safe_cache_fragment(value: str) -> str:
    fragment = "".join(ch if ch.isalnum() else "_" for ch in value)
    fragment = fragment.strip("_")
    return fragment[:80] or "target"


def _apply_ocr_result(state: RunState, data: dict[str, Any]) -> None:
    text = data.get("text")
    if not isinstance(text, str) or not text:
        return
    confidence = data.get("confidence")
    normalized_confidence = (
        float(confidence) if isinstance(confidence, (int, float)) else None
    )

    region_id = data.get("region_id")
    if isinstance(region_id, str):
        region = state.get_region(region_id)
        if region is not None:
            if _is_refined_region_result(data):
                _apply_refined_region_ocr_candidate(
                    region,
                    text=text,
                    confidence=normalized_confidence,
                    data=data,
                )
            else:
                region.text = text
                region.confidence = normalized_confidence
        return

    page_index = data.get("page_index")
    if isinstance(page_index, int):
        page = state.get_page(page_index)
        if page is not None:
            page.ocr_text = text
            page.ocr_confidence = normalized_confidence
            _maybe_store_canonical_page_ocr(page, text, normalized_confidence, data)
            page_ocr_markdown = data.get("page_ocr_markdown")
            if has_searchable_text(page_ocr_markdown):
                page.metadata["page_ocr_markdown"] = page_ocr_markdown
            source_kind = data.get("source_kind")
            if isinstance(source_kind, str) and source_kind.strip():
                page.metadata["ocr_source_kind"] = source_kind
            region_ids = data.get("region_ids")
            if isinstance(region_ids, list):
                page.metadata["ocr_assembled_region_ids"] = [
                    region_id for region_id in region_ids if isinstance(region_id, str)
                ]
            page_ocr_blocks = _page_ocr_blocks_from_raw(data.get("page_ocr_blocks"))
            if page_ocr_blocks:
                _apply_page_ocr_blocks(page, page_ocr_blocks)


def _existing_target_ocr(
    state: RunState,
    target: dict[str, Any],
) -> dict[str, Any] | None:
    data = target.get("data")
    if not isinstance(data, dict):
        return None
    if _is_refined_region_result(data):
        return None

    region_id = data.get("region_id")
    if isinstance(region_id, str):
        region = state.get_region(region_id)
        if region is not None and has_searchable_text(region.text):
            return {
                "text": region.text,
                "confidence": region.confidence,
            }
        return None

    page_index = data.get("page_index")
    if isinstance(page_index, int):
        page = state.get_page(page_index)
        if page is not None:
            cached_page_text = page.metadata.get("page_ocr_text")
            if has_searchable_text(cached_page_text):
                return {
                    "text": cached_page_text,
                    "confidence": _optional_float(page.metadata.get("page_ocr_confidence")),
                    "source": "state",
                    "source_kind": "page_ocr",
                }
            source_kind = page.metadata.get("ocr_source_kind")
            if (
                has_searchable_text(page.ocr_text)
                and not (
                    isinstance(source_kind, str)
                    and source_kind.startswith("region_")
                )
            ):
                return {
                    "text": page.ocr_text,
                    "confidence": page.ocr_confidence,
                    "source": "state",
                    "source_kind": "page_ocr",
                }
            assembled = assemble_complete_region_page_text(page)
            if assembled is not None:
                text, source_kind, region_ids = assembled
                return {
                    "text": text,
                    "confidence": None,
                    "source": "state",
                    "source_kind": source_kind,
                    "region_ids": region_ids,
                }
    return None


def _has_canonical_page_ocr(page) -> bool:
    if has_searchable_text(page.metadata.get("page_ocr_text")):
        return True
    source_kind = page.metadata.get("ocr_source_kind")
    if isinstance(source_kind, str) and source_kind.startswith("region_"):
        return False
    return has_searchable_text(page.ocr_text)


def _maybe_store_canonical_page_ocr(
    page,
    text: str,
    confidence: float | None,
    data: dict[str, Any],
) -> None:
    source = data.get("source")
    source_kind = data.get("source_kind")
    if source == "state" and source_kind != "page_ocr":
        return
    if isinstance(source_kind, str) and source_kind.startswith("region_"):
        return
    if isinstance(data.get("region_ids"), list) and data.get("region_ids"):
        return
    page.metadata["page_ocr_text"] = text
    page.metadata["page_ocr_confidence"] = confidence
    if isinstance(source, str) and source.strip() and source != "state":
        page.metadata["page_ocr_source"] = source


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _existing_ocr_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    source_kind = payload.get("source_kind")
    if isinstance(source_kind, str) and source_kind.strip():
        metadata["source_kind"] = source_kind
    region_ids = payload.get("region_ids")
    if isinstance(region_ids, list):
        metadata["region_ids"] = [
            region_id for region_id in region_ids if isinstance(region_id, str)
        ]
    return metadata


def _page_ocr_blocks_from_raw(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    blocks: list[dict[str, Any]] = []
    for index, raw_block in enumerate(value):
        block = _normalize_page_ocr_block(raw_block, fallback_index=index)
        if block is not None:
            blocks.append(block)
    return blocks


def _normalize_page_ocr_block(
    raw_block: Any,
    *,
    fallback_index: int,
) -> dict[str, Any] | None:
    if not isinstance(raw_block, dict):
        return None

    bbox = _bbox_from_block(raw_block)
    if bbox is None:
        return None

    label = _normalize_layout_label(
        str(raw_block.get("label") or raw_block.get("block_label") or "text")
    )
    text = raw_block.get("text")
    if not isinstance(text, str):
        text = raw_block.get("block_content")
    if not isinstance(text, str):
        text = ""

    block_id = raw_block.get("block_id")
    if not isinstance(block_id, int):
        block_id = fallback_index

    order = raw_block.get("order")
    if not isinstance(order, int):
        order = raw_block.get("block_order")
    if not isinstance(order, int):
        order = None

    group_id = raw_block.get("group_id")
    if not isinstance(group_id, (int, str)) or not str(group_id).strip():
        group_id = None

    polygon_points = raw_block.get("polygon_points")
    if polygon_points is None:
        polygon_points = raw_block.get("block_polygon_points")

    return {
        "block_id": block_id,
        "order": order,
        "label": label,
        "type": _normalize_region_type(label),
        "text": text,
        "bbox": list(bbox),
        "group_id": group_id,
        "polygon_points": polygon_points,
    }


def _bbox_from_block(block: dict[str, Any]) -> tuple[float, float, float, float] | None:
    for key in ("bbox", "block_bbox", "coordinate", "box"):
        value = block.get(key)
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list) or len(value) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(item) for item in value)
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        return (x0, y0, x1, y1)
    return None


def _apply_page_ocr_blocks(page, blocks: list[dict[str, Any]]) -> None:
    page.metadata["page_ocr_blocks"] = blocks
    page.metadata["page_ocr_layout_source"] = PAGE_OCR_LAYOUT_SOURCE

    if not page.regions:
        created_regions = _regions_from_page_ocr_blocks(page, blocks)
        for region in created_regions:
            page.add_region(region)
        if created_regions:
            page.metadata["layout_source"] = PAGE_OCR_LAYOUT_SOURCE
            page.metadata["layout_raw_boxes"] = [
                _layout_raw_box_from_page_ocr_block(block) for block in blocks
            ]
            _apply_fallback_post_layout(page, created_regions)
        return

    for block in blocks:
        if not has_searchable_text(block.get("text")):
            continue
        region = _find_region_for_page_ocr_block(page, block)
        if region is None or has_searchable_text(region.text):
            continue
        region.text = str(block["text"])
        region.confidence = None
        region.metadata["ocr_source_kind"] = "page_ocr_block"
        region.metadata["page_ocr_block_id"] = block.get("block_id")
    _ensure_post_layout_groups(page)


def _regions_from_page_ocr_blocks(page, blocks: list[dict[str, Any]]) -> list[Region]:
    regions: list[Region] = []
    used_ids = {region.region_id for region in page.regions}
    for index, block in enumerate(blocks):
        bbox = _bbox_from_block(block)
        if bbox is None:
            continue
        label = _normalize_layout_label(str(block.get("label") or "text"))
        block_id = block.get("block_id")
        suffix = block_id if isinstance(block_id, int) else index
        region_id = _unique_region_id(
            f"p{page.page_number}_{_safe_id(label)}_{suffix}",
            used_ids,
        )
        markdown_label = _markdown_label_for_layout_label(label)
        metadata = {
            "raw": _layout_raw_box_from_page_ocr_block(block),
            "layout": {
                "source": PAGE_OCR_LAYOUT_SOURCE,
                "label": label,
                "score": None,
                "sequence_order": suffix,
                "reading_order": (
                    block.get("order") if isinstance(block.get("order"), int) else None
                ),
                "order_label": None,
                "polygon_points": block.get("polygon_points"),
                "group_id": block.get("group_id"),
                "model_name": "PaddleOCRVL",
            },
            "render": {
                "kind": _render_kind_for_label(label),
                "image_like": _is_image_like_label(label),
                "ignore_for_markdown": _is_ignorable_for_markdown(markdown_label),
                "markdown_label": markdown_label,
            },
            "ocr_source_kind": "page_ocr_block",
            "page_ocr_block_id": block.get("block_id"),
        }
        text = block.get("text")
        regions.append(
            Region(
                page_index=page.page_index,
                bbox=bbox,
                region_id=region_id,
                label=label,
                raw_type=label,
                type=_normalize_region_type(label),
                text=text if has_searchable_text(text) else None,
                confidence=None,
                coordinate_space="pixel",
                metadata=metadata,
            )
        )
    return regions


def _layout_raw_box_from_page_ocr_block(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": block.get("label"),
        "coordinate": list(block.get("bbox") or []),
        "block_id": block.get("block_id"),
        "block_order": block.get("order"),
        "block_content": block.get("text"),
        "group_id": block.get("group_id"),
        "polygon_points": block.get("polygon_points"),
    }


def _apply_fallback_post_layout(page, regions: list[Region]) -> None:
    _ensure_post_layout_groups(page, regions)


def _find_region_for_page_ocr_block(page, block: dict[str, Any]):
    label = _normalize_layout_label(str(block.get("label") or "text"))
    block_id = block.get("block_id")
    if isinstance(block_id, int):
        region = page.get_region(f"p{page.page_number}_{_safe_id(label)}_{block_id}")
        if region is not None:
            return region

    block_bbox = _bbox_from_block(block)
    if block_bbox is None:
        return None
    for region in page.regions:
        region_label = _normalize_layout_label(
            str(region.label or region.raw_type or "text")
        )
        if region_label != label:
            continue
        if tuple(float(item) for item in region.bbox) == block_bbox:
            return region
    return None


def _is_refined_region_result(data: dict[str, Any]) -> bool:
    return (
        data.get("zoom_artifact_path") is not None
        or data.get("target_long_side_px") is not None
        or data.get("crop_artifact_path") is not None
        or data.get("left_px") is not None
        or data.get("right_px") is not None
        or data.get("top_px") is not None
        or data.get("bottom_px") is not None
        or data.get("rotate_artifact_path") is not None
        or data.get("angle_degree") is not None
    )


def _apply_refined_region_ocr_candidate(
    region,
    *,
    text: str,
    confidence: float | None,
    data: dict[str, Any],
) -> None:
    if data.get("zoom_artifact_path") is not None or data.get("target_long_side_px") is not None:
        candidate_kind = "zoom"
    elif (
        data.get("crop_artifact_path") is not None
        or data.get("left_px") is not None
        or data.get("right_px") is not None
        or data.get("top_px") is not None
        or data.get("bottom_px") is not None
    ):
        candidate_kind = "crop"
    elif data.get("rotate_artifact_path") is not None or data.get("angle_degree") is not None:
        candidate_kind = "rotate"
    else:
        return

    candidates = region.metadata.get("ocr_refinement_candidates")
    if not isinstance(candidates, dict):
        candidates = {}
        region.metadata["ocr_refinement_candidates"] = candidates
    candidates[candidate_kind] = {
        "text": text,
        "confidence": confidence,
        "artifact_path": (
            data.get("zoom_artifact_path")
            if candidate_kind == "zoom"
            else data.get("crop_artifact_path")
            if candidate_kind == "crop"
            else data.get("rotate_artifact_path")
        ),
        "target_long_side_px": (
            data.get("target_long_side_px")
            if candidate_kind == "zoom"
            else None
        ),
        "left_px": data.get("left_px") if candidate_kind == "crop" else None,
        "right_px": data.get("right_px") if candidate_kind == "crop" else None,
        "top_px": data.get("top_px") if candidate_kind == "crop" else None,
        "bottom_px": data.get("bottom_px") if candidate_kind == "crop" else None,
        "angle_degree": data.get("angle_degree") if candidate_kind == "rotate" else None,
        "source": data.get("source"),
    }


def clamp_bbox(bbox: PixelBBox, width: int, height: int) -> PixelBBox:
    x0, y0, x1, y1 = bbox
    return (
        max(0, min(width, x0)),
        max(0, min(height, y0)),
        max(0, min(width, x1)),
        max(0, min(height, y1)),
    )


def _expand_bbox(
    bbox: PixelBBox,
    *,
    padding: int,
    width: int | None,
    height: int | None,
) -> PixelBBox:
    if padding <= 0:
        return bbox
    x0, y0, x1, y1 = bbox
    expanded = (x0 - padding, y0 - padding, x1 + padding, y1 + padding)
    if width is not None and height is not None:
        return clamp_bbox(expanded, width, height)
    return expanded


def _resolve_artifact_dir(
    state: RunState,
    action: Action,
    default_artifact_dir: Path | None,
) -> Path | None:
    value = action.parameters.get("artifact_dir")
    if value is None:
        value = state.metadata.get("artifact_dir")
    if value is None:
        value = state.document.metadata.get("artifact_dir")
    if value is None:
        return default_artifact_dir
    return Path(str(value)).expanduser()
