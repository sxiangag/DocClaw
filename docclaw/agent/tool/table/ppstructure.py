"""PP-Structure-backed table implementation."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from docclaw.agent.tool.quiet import suppress_vendor_init_output
from docclaw.agent.tool.table.table import TableTool
from docclaw.agent.utils import Action, PageState, Region, RunState, has_searchable_text


class PPStructureTableTool(TableTool):
    """Parse tables with PaddleOCR PP-StructureV3."""

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
            "enable_mkldnn": False,
            **dict(pipeline_kwargs or {}),
        }
        self.predict_kwargs = {
            "use_table_recognition": True,
            **dict(predict_kwargs or {}),
        }

    @property
    def description(self) -> str:
        return (
            "Parse document tables from a page set or region set with PaddleOCR "
            "PP-StructureV3. Parsed table results are written back into page state, "
            "and repeating table parsing on the same targets usually has no value."
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
            if cached_tables is not None:
                results.extend(cached_tables)
                continue

            if not page.image_path:
                raise ValueError(f"page {page.page_index} has no image_path")
            image_path = Path(page.image_path).expanduser()
            if not image_path.exists():
                raise ValueError(f"page image_path does not exist: {image_path}")

            result = self._predict_page(image_path)
            result_data = _result_to_dict(result)
            tables = _tables_from_result(page, result_data)
            _write_tables_to_regions(page, tables)
            selected_tables = _filter_tables(tables, regions=page_regions)
            results.extend(selected_tables)

        return results

    def _predict_page(self, image_path: Path) -> Any:
        with suppress_vendor_init_output():
            return self._get_pipeline().predict(str(image_path), **self.predict_kwargs)

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            from paddleocr import PPStructureV3

            with suppress_vendor_init_output():
                self._pipeline = PPStructureV3(**self.pipeline_kwargs)
        return self._pipeline

def _cached_tables_for_page(
    page: PageState,
    *,
    regions: list[Region] | None,
) -> list[dict[str, Any]] | None:
    selected_regions = regions if regions is not None else [
        region for region in page.regions
        if str(region.label or "").strip().lower() == "table"
    ]
    if not selected_regions:
        return None
    if not all(region.type == "table" and has_searchable_text(region.text) for region in selected_regions):
        return None
    return [_region_to_table(region) for region in selected_regions]


def _filter_tables(
    tables: list[dict[str, Any]],
    *,
    regions: list[Region] | None,
) -> list[dict[str, Any]]:
    if not regions:
        return [dict(table) for table in tables]

    filtered: list[dict[str, Any]] = []
    for table in tables:
        bbox = _extract_bbox(table.get("bbox"))
        if bbox is None or any(_bboxes_overlap(bbox, region.bbox) for region in regions):
            filtered.append(dict(table))
    return filtered


def _write_tables_to_regions(page: PageState, tables: list[dict[str, Any]]) -> None:
    for table in tables:
        bbox = _extract_bbox(table.get("bbox"))
        if bbox is None:
            continue
        html = str(table.get("text") or "").strip()
        if not html:
            continue
        confidence = table.get("confidence")
        normalized_confidence = float(confidence) if isinstance(confidence, (int, float)) else None
        for region in page.regions:
            if str(region.label or "").strip().lower() != "table":
                continue
            if _bboxes_overlap(bbox, region.bbox):
                region.text = html
                region.confidence = normalized_confidence


def _region_to_table(region: Region) -> dict[str, Any]:
    return {
        "page_index": region.page_index,
        "region_id": region.region_id,
        "text": region.text or "",
        "source": "ppstructure",
        "confidence": region.confidence,
    }


def _tables_from_result(page: PageState, result: dict[str, Any]) -> list[dict[str, Any]]:
    table_results = result.get("table_res_list")
    if not isinstance(table_results, list):
        return []

    table_blocks = _table_blocks(result)
    tables: list[dict[str, Any]] = []
    for index, item in enumerate(table_results):
        table_result = _normalize_result_dict(item)
        if table_result is None:
            continue

        bbox_candidates = [
            table_result.get("bbox"),
            table_result.get("block_bbox"),
        ]
        if index < len(table_blocks):
            bbox_candidates.append(table_blocks[index].get("block_bbox"))
        bbox = None
        for candidate in bbox_candidates:
            bbox = _extract_bbox(candidate)
            if bbox is not None:
                break
        html = _extract_html(table_result)
        cell_boxes = _normalize_cell_boxes(table_result.get("cell_box_list"))
        ocr_lines = _normalize_table_ocr(table_result.get("table_ocr_pred"))
        tables.append(
            {
                "page_index": page.page_index,
                "region_id": None,
                "text": html,
                "source": "ppstructure",
                "confidence": None,
                "bbox": list(bbox) if bbox is not None else None,
            }
        )
    return tables


def _table_blocks(result: dict[str, Any]) -> list[dict[str, Any]]:
    parsing_blocks = result.get("parsing_res_list")
    if not isinstance(parsing_blocks, list):
        return []
    blocks: list[dict[str, Any]] = []
    for item in parsing_blocks:
        block = _normalize_result_dict(item)
        if block is None:
            continue
        label = str(block.get("block_label") or block.get("label") or "").strip().lower()
        if label == "table":
            blocks.append(block)
    return blocks


def _normalize_table_ocr(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    texts = _to_sequence(value.get("rec_texts"))
    scores = _to_sequence(value.get("rec_scores"))
    boxes = _first_present_sequence(value.get("rec_boxes"), value.get("rec_polys"))

    lines: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        line: dict[str, Any] = {"text": str(text)}
        if index < len(scores):
            score = _to_float(scores[index])
            if score is not None:
                line["score"] = score
        if index < len(boxes):
            box = _to_list(boxes[index])
            if box is not None:
                line["box"] = box
        lines.append(line)
    return lines


def _normalize_cell_boxes(value: Any) -> list[list[Any]]:
    boxes = _to_sequence(value)
    normalized: list[list[Any]] = []
    for box in boxes:
        item = _to_list(box)
        if item is not None:
            normalized.append(item)
    return normalized


def _extract_html(table_result: dict[str, Any]) -> str | None:
    value = table_result.get("pred_html")
    if isinstance(value, str) and value.strip():
        return value
    return None


def _extract_bbox(value: Any) -> tuple[float, float, float, float] | None:
    values = _to_list(value)
    if values is None or len(values) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in values)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _estimate_shape_from_html(html: str | None) -> tuple[int | None, int | None]:
    if not html:
        return None, None
    rows = re.findall(r"<tr\b", html, flags=re.IGNORECASE)
    row_count = len(rows) if rows else None
    column_count: int | None = None
    for row_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL):
        cells = re.findall(r"<t[dh]\b", row_html, flags=re.IGNORECASE)
        if column_count is None or len(cells) > column_count:
            column_count = len(cells)
    return row_count, column_count


def _html_has_spans(html: str | None) -> bool:
    if not html:
        return False
    lowered = html.lower()
    return "rowspan=" in lowered or "colspan=" in lowered


def _bboxes_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    return lx0 < rx1 and rx0 < lx1 and ly0 < ry1 and ry0 < ly1


def _result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, list):
        if not result:
            return {}
        return _result_to_dict(result[0])
    normalized = _normalize_result_dict(result)
    if normalized is not None:
        return normalized
    raise TypeError(f"unsupported PP-Structure result type: {type(result).__name__}")


def _normalize_result_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return _to_jsonable(value)

    json_attr = getattr(value, "json", None)
    if isinstance(json_attr, dict):
        maybe_res = json_attr.get("res")
        if isinstance(maybe_res, dict):
            return _to_jsonable(maybe_res)
        return _to_jsonable(json_attr)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, dict):
            return _to_jsonable(data)

    return None


def _first_present_sequence(*values: Any) -> list[Any]:
    for value in values:
        sequence = _to_sequence(value)
        if sequence:
            return sequence
    return []


def _to_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return value
    return []


def _to_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return value
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
