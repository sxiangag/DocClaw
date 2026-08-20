"""PaddleOCR-VL-backed table implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docclaw.agent.tool.ocr.ocr import clamp_bbox
from docclaw.agent.tool.ocr.paddleocrvl import (
    _extract_scores,
    _result_to_dict,
    _to_jsonable,
    _unwrap_result_item,
)
from docclaw.agent.tool.quiet import suppress_vendor_init_output
from docclaw.agent.tool.table.ppstructure import (
    _bboxes_overlap,
    _estimate_shape_from_html,
    _extract_bbox as _extract_any_bbox,
    _html_has_spans,
)
from docclaw.agent.tool.table.table import TableTool
from docclaw.agent.utils import Action, PageState, Region, RunState, has_searchable_text


class PaddleOCRVLTableTool(TableTool):
    """Parse tables with PaddleOCR-VL stage-II prompts."""

    def __init__(
        self,
        *,
        pipeline: Any | None = None,
        pipeline_kwargs: dict[str, Any] | None = None,
        predict_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self.pipeline_kwargs = {
            "device": "gpu:0",
            **dict(pipeline_kwargs or {}),
        }
        self.predict_kwargs = {
            "use_layout_detection": False,
            "prompt_label": "table",
            **dict(predict_kwargs or {}),
        }

    @property
    def description(self) -> str:
        return (
            "Parse tables from known table regions or page-level table candidates with "
            "PaddleOCR-VL using the table prompt. This uses DocClaw for layout "
            "targeting and PaddleOCR-VL for region-level table recognition."
        )

    def parse_tables(
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

            cached_tables = _cached_tables_for_page(page, regions=page_regions)
            if cached_tables is not None and not _uses_refined_regions(action):
                results.extend(cached_tables)
                continue

            candidates = _table_candidates_for_page(
                state,
                page,
                explicit_regions=page_regions,
                use_zoom_artifacts=_uses_zoom_regions(action),
                use_crop_artifacts=_uses_crop_regions(action),
            )
            for candidate in candidates:
                table = self._parse_candidate(page, candidate)
                if table is not None:
                    results.append(table)

        return results

    def _parse_candidate(
        self,
        page: PageState,
        candidate: _Candidate,
    ) -> dict[str, Any] | None:
        raw = self._recognize_table(page, candidate)
        blocks = _extract_table_blocks(raw["raw"])
        if not blocks:
            return None
        primary = blocks[0]
        html = _extract_block_content(primary)
        if not html:
            return None

        if candidate.region is not None:
            candidate.region.text = html
            candidate.region.confidence = raw["confidence"]
        return {
            "page_index": page.page_index,
            "region_id": candidate.region_id,
            "text": html,
            "source": "paddleocr_vl",
            "confidence": raw["confidence"],
        }

    def _recognize_table(self, page: PageState, candidate: _Candidate) -> dict[str, Any]:
        image_input = _candidate_image_input(page, candidate)
        with suppress_vendor_init_output():
            result = self._get_pipeline().predict(image_input, **self.predict_kwargs)
        result_item = _unwrap_result_item(result)
        result_data = _to_jsonable(_result_to_dict(result_item))
        scores = _extract_scores(result_data)
        confidence = sum(scores) / len(scores) if scores else None
        return {
            "confidence": confidence,
            "raw": result_data,
        }

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            from paddleocr import PaddleOCRVL

            with suppress_vendor_init_output():
                self._pipeline = PaddleOCRVL(**self.pipeline_kwargs)
        return self._pipeline


class _Candidate:
    def __init__(
        self,
        *,
        region: Region | None,
        region_id: str | None,
        bbox: tuple[float, float, float, float] | None,
        artifact_path: str | None = None,
    ) -> None:
        self.region = region
        self.region_id = region_id
        self.bbox = bbox
        self.artifact_path = artifact_path


def _cached_tables_for_page(
    page: PageState,
    *,
    regions: list[Region] | None,
) -> list[dict[str, Any]] | None:
    selected_regions = regions if regions is not None else [
        region for region in page.regions
        if region.type == "table"
    ]
    if not selected_regions:
        return None
    if not all(region.type == "table" and has_searchable_text(region.text) for region in selected_regions):
        return None
    return [_region_to_table(region) for region in selected_regions]

def _region_to_table(region: Region) -> dict[str, Any]:
    html = region.text or ""
    return {
        "page_index": region.page_index,
        "region_id": region.region_id,
        "text": html,
        "source": "paddleocr_vl",
        "confidence": region.confidence,
    }


def _table_candidates_for_page(
    state: RunState,
    page: PageState,
    *,
    explicit_regions: list[Region] | None,
    use_zoom_artifacts: bool,
    use_crop_artifacts: bool,
) -> list[_Candidate]:
    if explicit_regions is not None:
        return [
            _Candidate(
                region=region,
                region_id=region.region_id,
                bbox=region.bbox,
                artifact_path=(
                    _zoom_artifact_path(state, region)
                    if use_zoom_artifacts
                    else _crop_artifact_path(state, region)
                    if use_crop_artifacts
                    else None
                ),
            )
            for region in explicit_regions
        ]

    if page.regions:
        return [
            _Candidate(region=region, region_id=region.region_id, bbox=region.bbox)
            for region in page.regions
            if region.type == "table"
        ]

    return [
        _Candidate(
            region=None,
            region_id=None,
            bbox=(0.0, 0.0, float(page.width), float(page.height)),
        )
    ]


def _crop_region(source_path: Path, bbox: tuple[float, float, float, float]) -> Any:
    from PIL import Image

    with Image.open(source_path) as image:
        width, height = image.size
        x0, y0, x1, y1 = clamp_bbox(bbox, width, height)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("bbox is outside the page image")
        return image.crop((x0, y0, x1, y1)).convert("RGB")


def _candidate_image_input(page: PageState, candidate: _Candidate) -> Any:
    if candidate.artifact_path:
        artifact_path = Path(candidate.artifact_path).expanduser()
        if not artifact_path.exists():
            raise ValueError(f"zoom artifact_path does not exist: {artifact_path}")
        return str(artifact_path)
    source_path = Path(page.image_path or "").expanduser()
    if not source_path.exists():
        raise ValueError(f"page image_path does not exist: {source_path}")
    if candidate.bbox is not None:
        return _crop_region(source_path, candidate.bbox)
    return str(source_path)


def _uses_zoom_regions(action: Action) -> bool:
    return isinstance(action.target.get("zoom_region_ids"), list)


def _uses_crop_regions(action: Action) -> bool:
    return isinstance(action.target.get("crop_region_ids"), list)


def _uses_refined_regions(action: Action) -> bool:
    return _uses_zoom_regions(action) or _uses_crop_regions(action)


def _zoom_artifact_path(state: RunState, region: Region) -> str | None:
    zoom_view = state.get_zoom_region_view(region.region_id)
    if zoom_view is None:
        return None
    return zoom_view.artifact_path


def _crop_artifact_path(state: RunState, region: Region) -> str | None:
    crop_view = state.get_crop_region_view(region.region_id)
    if crop_view is None:
        return None
    return crop_view.artifact_path


def _extract_table_blocks(raw: dict[str, Any]) -> list[dict[str, Any]]:
    parsing_res_list = raw.get("parsing_res_list")
    if not isinstance(parsing_res_list, list):
        return []
    blocks: list[dict[str, Any]] = []
    for item in parsing_res_list:
        if not isinstance(item, dict):
            continue
        label = str(item.get("block_label") or item.get("label") or "").strip().lower()
        if label == "table":
            blocks.append(item)
    return blocks


def _extract_block_content(block: dict[str, Any]) -> str | None:
    value = block.get("block_content") or block.get("content")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
