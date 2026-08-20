"""PP-DocLayout-backed layout implementation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from docclaw.agent.tool.layout.layout import LayoutTool
from docclaw.agent.tool.quiet import suppress_vendor_init_output
from docclaw.agent.utils import Action, PageState, Region, RunState

_MARKDOWN_IGNORE_LABELS = {
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
}

_DEFAULT_IMAGE_LIKE_LABELS = {
    "chart",
    "image",
    "header_image",
    "footer_image",
    "seal",
}

_OFFICIAL_LAYOUT_THRESHOLD = 0.3
_OFFICIAL_LAYOUT_MERGE_BBOXES_MODE = {
    0: "union",  # abstract
    1: "union",  # algorithm
    2: "union",  # aside_text
    3: "large",  # chart
    4: "union",  # content
    5: "large",  # display_formula
    6: "large",  # doc_title
    7: "union",  # figure_title
    8: "union",  # footer
    9: "union",  # footer_image
    10: "union",  # footnote
    11: "union",  # formula_number
    12: "union",  # header
    13: "union",  # header_image
    14: "union",  # image
    15: "large",  # inline_formula
    16: "union",  # number
    17: "large",  # paragraph_title
    18: "union",  # reference
    19: "union",  # reference_content
    20: "union",  # seal
    21: "union",  # table
    22: "union",  # text
    23: "union",  # title_text
    24: "union",  # vision_footnote
}


class PPDocLayoutTool(LayoutTool):
    """Parse layout regions with PaddleOCR LayoutDetection."""

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
            "model_name": "PP-DocLayoutV3",
            "threshold": _OFFICIAL_LAYOUT_THRESHOLD,
            "layout_nms": True,
            "layout_unclip_ratio": [1.0, 1.0],
            "layout_merge_bboxes_mode": dict(_OFFICIAL_LAYOUT_MERGE_BBOXES_MODE),
            "enable_mkldnn": False,
            **dict(pipeline_kwargs or {}),
        }
        self.predict_kwargs = dict(predict_kwargs or {})

    @property
    def description(self) -> str:
        return (
            "Parse layout for one page, a page set, or the whole document and return "
            "page/region inventory. This writes structural layout regions into document "
            "state for later OCR and reasoning, but does not create searchable text on "
            "its own. Repeating layout parsing on the same pages usually has no value."
        )

    def parse_layout(
        self,
        state: RunState,
        pages: list[PageState],
        action: Action,
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for page in pages:
            if page.regions:
                payload.append(_page_payload(page, skipped=True))
                continue

            if not page.image_path:
                raise ValueError(f"page {page.page_index} has no image_path")
            image_path = Path(page.image_path).expanduser()
            if not image_path.exists():
                raise ValueError(f"page image_path does not exist: {image_path}")

            result = self._predict_page(image_path)
            result_data = _result_to_dict(result)
            regions, normalized_boxes = _regions_from_result(
                page,
                result_data,
                model_name=str(self.pipeline_kwargs.get("model_name") or ""),
            )
            _annotate_post_layout_groups(
                page,
                regions,
                normalized_boxes,
                image_path=image_path,
            )
            page.metadata["layout_source"] = "pp_doclayout"
            if self.pipeline_kwargs.get("model_name") is not None:
                page.metadata["layout_model_name"] = self.pipeline_kwargs["model_name"]
            page.metadata["layout_raw_boxes"] = normalized_boxes
            for region in regions:
                page.add_region(region)
            payload.append(_page_payload(page, layout_detection_result=result_data))
        return payload

    def _predict_page(self, image_path: Path) -> Any:
        with suppress_vendor_init_output():
            pipeline = self._get_pipeline()
            if hasattr(pipeline, "predict"):
                result = pipeline.predict(str(image_path), **self.predict_kwargs)
            else:
                result = pipeline(str(image_path), **self.predict_kwargs)
            if isinstance(result, Iterator):
                return next(result)
            return result

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            from paddlex import create_model

            with suppress_vendor_init_output():
                model_name = self.pipeline_kwargs.get("model_name", "PP-DocLayoutV3")
                kwargs = dict(self.pipeline_kwargs)
                kwargs.pop("model_name", None)
                kwargs.pop("enable_mkldnn", None)
                self._pipeline = create_model(model_name, **kwargs)
        return self._pipeline


def _page_payload(
    page: PageState,
    *,
    skipped: bool = False,
    layout_detection_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "page_index": page.page_index,
        "width": page.width,
        "height": page.height,
        "image_path": page.image_path,
        "regions": [region.to_dict() for region in page.regions],
    }
    if skipped:
        payload["skipped"] = True
    if layout_detection_result is not None:
        payload["layout_detection"] = {
            "input_path": layout_detection_result.get("input_path"),
            "page_index": layout_detection_result.get("page_index"),
            "box_count": len(layout_detection_result.get("boxes", [])),
        }
    return payload


def _result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, list):
        if not result:
            return {}
        return _result_to_dict(result[0])
    if isinstance(result, dict):
        if isinstance(result.get("res"), dict):
            return result["res"]
        return result

    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, dict):
            if isinstance(data.get("res"), dict):
                return data["res"]
            return data

    json_attr = getattr(result, "json", None)
    if isinstance(json_attr, dict):
        if isinstance(json_attr.get("res"), dict):
            return json_attr["res"]
        return json_attr

    if hasattr(result, "keys"):
        try:
            data = {key: result[key] for key in result.keys()}
            if isinstance(data.get("res"), dict):
                return data["res"]
            return data
        except Exception:
            pass

    raise TypeError(f"unsupported layout detection result type: {type(result).__name__}")


def _regions_from_result(
    page: PageState,
    result: dict[str, Any],
    *,
    model_name: str,
) -> tuple[list[Region], list[dict[str, Any]]]:
    boxes = _extract_boxes(result)
    _annotate_sequence_order(page, boxes)
    regions: list[Region] = []
    used_ids = {region.region_id for region in page.regions}
    for index, box in enumerate(boxes):
        bbox = _extract_bbox(box)
        if bbox is None:
            continue

        raw_label = _extract_label(box)
        normalized_label = _normalize_layout_label(raw_label)
        region_type = _normalize_region_type(raw_label)
        markdown_label = _markdown_label_for_layout_label(normalized_label)
        sequence_order = _extract_sequence_order(box)
        score = _extract_score(box)
        polygon_points = _extract_polygon_points(box)
        group_id = _extract_group_id(box)
        render_kind = _render_kind_for_label(normalized_label)
        image_like = _is_image_like_label(normalized_label)
        ignore_for_markdown = _is_ignorable_for_markdown(markdown_label)
        suffix = sequence_order if sequence_order is not None else index
        region_id = _unique_region_id(
            f"p{page.page_number}_{_safe_id(normalized_label)}_{suffix}",
            used_ids,
        )
        metadata = {
            "raw": box,
            "layout": {
                "source": "pp_doclayout",
                "label": normalized_label,
                "score": score,
                "sequence_order": sequence_order,
                "reading_order": _extract_reading_order(box),
                "order_label": _extract_order_label(box),
                "polygon_points": polygon_points,
                "group_id": group_id,
                "model_name": model_name or None,
            },
            "render": {
                "kind": render_kind,
                "image_like": image_like,
                "ignore_for_markdown": ignore_for_markdown,
                "markdown_label": markdown_label,
            },
        }

        regions.append(
            Region(
                page_index=page.page_index,
                bbox=bbox,
                region_id=region_id,
                label=normalized_label,
                raw_type=normalized_label,
                type=region_type,
                coordinate_space="pixel",
                metadata=metadata,
            )
        )
    return regions, boxes


def _annotate_post_layout_groups(
    page: PageState,
    regions: list[Region],
    boxes: list[dict[str, Any]],
    *,
    image_path: Path,
) -> None:
    try:
        post_layout = _build_post_layout_metadata(
            page,
            regions,
            boxes,
            image_path=image_path,
        )
    except Exception:
        post_layout = _build_fallback_post_layout_metadata(regions)
    page.metadata["post_layout"] = post_layout
    _annotate_regions_from_post_layout(regions, post_layout)


def _ensure_post_layout_groups(
    page: PageState,
    regions: list[Region] | None = None,
    boxes: list[dict[str, Any]] | None = None,
    *,
    image_path: Path | None = None,
) -> bool:
    """Ensure page has post-layout grouping metadata without overwriting parsed layout."""
    if isinstance(page.metadata.get("post_layout"), dict):
        return False

    resolved_regions = list(regions) if regions is not None else list(page.regions)
    if not resolved_regions:
        return False

    resolved_boxes = (
        [dict(item) for item in boxes if isinstance(item, dict)]
        if boxes is not None
        else _post_layout_boxes_from_state(page, resolved_regions)
    )
    resolved_image_path = image_path if image_path is not None else _page_image_path(page)

    post_layout: dict[str, Any] | None = None
    if resolved_boxes and resolved_image_path is not None and resolved_image_path.exists():
        try:
            candidate = _build_post_layout_metadata(
                page,
                resolved_regions,
                resolved_boxes,
                image_path=resolved_image_path,
            )
            if _has_post_layout_groups(candidate):
                post_layout = candidate
        except Exception:
            post_layout = None

    if post_layout is None:
        post_layout = _build_fallback_post_layout_metadata(resolved_regions)

    page.metadata["post_layout"] = post_layout
    _annotate_regions_from_post_layout(resolved_regions, post_layout)
    return True


def _annotate_regions_from_post_layout(
    regions: list[Region],
    post_layout: dict[str, Any],
) -> None:
    region_by_id = {region.region_id: region for region in regions}

    groups = post_layout.get("groups")
    if not isinstance(groups, dict):
        return

    for group_id, group in groups.items():
        if not isinstance(group_id, str) or not isinstance(group, dict):
            continue
        region_ids = group.get("region_ids")
        if not isinstance(region_ids, list):
            continue
        anchor_region_id = group.get("anchor_region_id")
        group_order = group.get("order")
        for member_index, region_id in enumerate(region_ids):
            region = region_by_id.get(region_id)
            if region is None:
                continue
            layout = region.metadata.setdefault("layout", {})
            if not isinstance(layout, dict):
                layout = {}
                region.metadata["layout"] = layout
            layout["merge_group_id"] = group_id
            layout["merge_order"] = group_order
            layout["merge_index"] = member_index


def _post_layout_boxes_from_state(
    page: PageState,
    regions: list[Region],
) -> list[dict[str, Any]]:
    raw_boxes = page.metadata.get("layout_raw_boxes")
    if isinstance(raw_boxes, list):
        boxes = [dict(item) for item in raw_boxes if isinstance(item, dict)]
        if len(boxes) == len(regions):
            return boxes

    boxes: list[dict[str, Any]] = []
    for region in regions:
        box = _post_layout_box_from_region(region)
        if box is None:
            return []
        boxes.append(box)
    return boxes


def _post_layout_box_from_region(region: Region) -> dict[str, Any] | None:
    raw = region.metadata.get("raw")
    box = dict(raw) if isinstance(raw, dict) else {}
    if _extract_bbox(box) is None:
        box["coordinate"] = list(region.bbox)
    if not str(box.get("label") or "").strip():
        box["label"] = region.label or region.raw_type or region.type or "layout"
    layout = region.metadata.get("layout")
    if isinstance(layout, dict):
        if "group_id" not in box and layout.get("group_id") is not None:
            box["group_id"] = layout.get("group_id")
        sequence_order = layout.get("sequence_order")
        if "block_order" not in box and isinstance(sequence_order, int):
            box["block_order"] = sequence_order
        polygon_points = layout.get("polygon_points")
        if "polygon_points" not in box and polygon_points is not None:
            box["polygon_points"] = polygon_points
    return box if _extract_bbox(box) is not None else None


def _page_image_path(page: PageState) -> Path | None:
    if not page.image_path:
        return None
    return Path(page.image_path).expanduser()


def _has_post_layout_groups(post_layout: dict[str, Any]) -> bool:
    group_ids = post_layout.get("group_ids")
    groups = post_layout.get("groups")
    return isinstance(group_ids, list) and bool(group_ids) and isinstance(groups, dict)


def _build_post_layout_metadata(
    page: PageState,
    regions: list[Region],
    boxes: list[dict[str, Any]],
    *,
    image_path: Path,
) -> dict[str, Any]:
    groups = _infer_official_style_groups(regions, boxes, image_path=image_path)
    ordered_group_ids = [group["group_id"] for group in groups]
    return {
        "group_ids": ordered_group_ids,
        "groups": {
            group["group_id"]: {
                key: value
                for key, value in group.items()
                if key != "group_id"
            }
            for group in groups
        },
    }


_DEFAULT_NON_MERGE_LABELS = (
    "image",
    "header_image",
    "footer_image",
    "chart",
    "seal",
    "table",
)


def _build_fallback_post_layout_metadata(regions: list[Region]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    ordered_group_ids: list[str] = []
    for group_order, region in enumerate(regions):
        label = _normalize_layout_label(region.label or region.raw_type or "layout")
        group_id = f"merge_group_{group_order}"
        ordered_group_ids.append(group_id)
        groups[group_id] = {
            "label": label,
            "order": group_order,
            "anchor_region_id": region.region_id,
            "region_ids": [region.region_id],
            "image_like": _is_image_like_label(label),
            "ignore": _is_ignorable_for_markdown(
                _markdown_label_for_layout_label(label)
            ),
        }
    return {
        "group_ids": ordered_group_ids,
        "groups": groups,
    }


def _infer_official_style_groups(
    regions: list[Region],
    boxes: list[dict[str, Any]],
    *,
    image_path: Path,
) -> list[dict[str, Any]]:
    block_entries = _build_merge_block_entries(boxes, image_path=image_path)
    if not block_entries:
        return []

    ordered_groups: list[dict[str, Any]] = []
    groups_by_id: dict[str, dict[str, Any]] = {}
    source_region_ids = [region.region_id for region in regions]

    for merged_index, block in enumerate(block_entries):
        source_index = block.get("_docclaw_source_index")
        if not isinstance(source_index, int):
            continue
        if source_index < 0 or source_index >= len(source_region_ids):
            continue

        group_anchor_index = block.get("group_id")
        if isinstance(group_anchor_index, int):
            group_id = f"merge_group_{group_anchor_index}"
        else:
            group_id = f"merge_group_{source_index}"

        group = groups_by_id.get(group_id)
        if group is None:
            label = _normalize_layout_label(str(block.get("label") or "layout"))
            group = {
                "group_id": group_id,
                "label": label,
                "order": len(ordered_groups),
                "anchor_region_id": source_region_ids[source_index],
                "region_ids": [],
                "image_like": _is_image_like_label(label),
                "ignore": _is_ignorable_for_markdown(
                    _markdown_label_for_layout_label(label)
                ),
            }
            groups_by_id[group_id] = group
            ordered_groups.append(group)

        region_id = source_region_ids[source_index]
        if region_id not in group["region_ids"]:
            group["region_ids"].append(region_id)

    return ordered_groups


def _build_merge_block_entries(
    boxes: list[dict[str, Any]],
    *,
    image_path: Path,
) -> list[dict[str, Any]]:
    if not boxes:
        return []
    try:
        import numpy as np
        from PIL import Image
        from paddlex.inference.pipelines.components.common.crop_image_regions import (
            CropByBoxes,
        )
        from paddlex.inference.pipelines.paddleocr_vl.uilts import merge_blocks
    except Exception:
        return _fallback_merge_block_entries(boxes)

    with Image.open(image_path) as image:
        image_array = np.array(image.convert("RGB"))

    cropper = CropByBoxes()
    crop_inputs = []
    for index, box in enumerate(boxes):
        crop_input = {
            "cls_id": int(box.get("cls_id", index)),
            "label": _normalize_layout_label(_extract_label(box)),
            "coordinate": list(_extract_bbox(box) or ()),
        }
        polygon_points = _extract_polygon_points(box)
        if polygon_points is not None:
            crop_input["polygon_points"] = polygon_points
        crop_inputs.append(crop_input)
    cropped_blocks = cropper(image_array, crop_inputs, layout_shape_mode="auto")
    for index, block in enumerate(cropped_blocks):
        block["_docclaw_source_index"] = index
        block["_docclaw_original_group_id"] = _extract_group_id(boxes[index])
        polygon_points = _extract_polygon_points(boxes[index])
        if polygon_points is not None:
            block["polygon_points"] = polygon_points

    merged_blocks = merge_blocks(
        cropped_blocks,
        non_merge_labels=list(_DEFAULT_NON_MERGE_LABELS),
        layout_shape_mode="auto",
    )
    normalized_entries: list[dict[str, Any]] = []
    for block in merged_blocks:
        normalized_entries.append(
            {
                "label": _normalize_layout_label(str(block.get("label") or "layout")),
                "box": list(block.get("box") or []),
                "group_id": block.get("group_id"),
                "_docclaw_source_index": block.get("_docclaw_source_index"),
            }
        )
    return normalized_entries


def _fallback_merge_block_entries(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": _normalize_layout_label(_extract_label(box)),
            "box": list(_extract_bbox(box) or []),
            "group_id": None,
            "_docclaw_source_index": index,
        }
        for index, box in enumerate(boxes)
    ]


def _extract_boxes(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw_boxes = result.get("boxes")
    if not isinstance(raw_boxes, list):
        return []

    boxes = [
        normalized
        for item in raw_boxes
        if (normalized := _normalize_box(item)) is not None
    ]
    boxes = _filter_overlap_boxes(boxes)
    return boxes


def _annotate_sequence_order(page: PageState, boxes: list[dict[str, Any]]) -> None:
    if not boxes:
        return
    try:
        from paddlex.inference.pipelines.layout_parsing.layout_objects import (
            LayoutBlock,
            LayoutRegion,
        )
        from paddlex.inference.pipelines.layout_parsing.setting import BLOCK_LABEL_MAP
        from paddlex.inference.pipelines.layout_parsing.xycut_enhanced.xycuts import (
            xycut_enhanced,
        )
    except Exception:
        return

    page_bbox = _page_bbox(page, boxes)
    layout_blocks = []
    for index, box in enumerate(boxes):
        bbox = _extract_bbox(box)
        if bbox is None:
            continue
        block = LayoutBlock(
            label=_normalize_layout_label(_extract_label(box)),
            bbox=list(map(int, bbox)),
            group_id=_extract_group_id(box),
        )
        block.text_line_height = max(2, min(int(block.height), 10))
        block.text_line_width = max(2, min(int(block.width), 20))
        setattr(block, "_docclaw_source_index", index)
        layout_blocks.append(block)
    if not layout_blocks:
        return

    try:
        sorted_blocks = xycut_enhanced(LayoutRegion(bbox=page_bbox, blocks=layout_blocks))
    except Exception:
        return

    visualize_index_labels = {
        label
        for label in BLOCK_LABEL_MAP["visualize_index_labels"]
        if label not in _MARKDOWN_IGNORE_LABELS
    }
    reading_order = 1
    for sequence_order, block in enumerate(sorted_blocks):
        source_index = getattr(block, "_docclaw_source_index", None)
        if not isinstance(source_index, int):
            continue
        boxes[source_index]["_docclaw_sequence_order"] = sequence_order
        boxes[source_index]["_docclaw_order_label"] = block.order_label
        if block.label in visualize_index_labels:
            boxes[source_index]["_docclaw_reading_order"] = reading_order
            reading_order += 1


def _page_bbox(page: PageState, boxes: list[dict[str, Any]]) -> list[int]:
    width = int(page.width) if isinstance(page.width, (int, float)) and page.width else 0
    height = int(page.height) if isinstance(page.height, (int, float)) and page.height else 0
    if width > 0 and height > 0:
        return [0, 0, width, height]

    max_x = 0.0
    max_y = 0.0
    for box in boxes:
        bbox = _extract_bbox(box)
        if bbox is None:
            continue
        _, _, x1, y1 = bbox
        max_x = max(max_x, x1)
        max_y = max(max_y, y1)
    return [0, 0, int(max_x), int(max_y)]


def _bbox_sort_key(box: dict[str, Any]) -> tuple[float, float]:
    bbox = _extract_bbox(box)
    if bbox is None:
        return (float("inf"), float("inf"))
    x0, y0, _, _ = bbox
    return (y0, x0)


def _extract_sequence_order(box: dict[str, Any]) -> int | None:
    value = box.get("_docclaw_sequence_order")
    if isinstance(value, int):
        return value
    return None


def _extract_reading_order(box: dict[str, Any]) -> int | None:
    value = box.get("_docclaw_reading_order")
    if isinstance(value, int):
        return value
    return None


def _extract_order_label(box: dict[str, Any]) -> str | None:
    value = box.get("_docclaw_order_label")
    if isinstance(value, str) and value.strip():
        return value
    return None


def _filter_overlap_boxes(
    boxes: list[dict[str, Any]],
    *,
    layout_shape_mode: str = "auto",
) -> list[dict[str, Any]]:
    filtered_boxes = [box for box in boxes if _normalize_layout_label(_extract_label(box)) != "reference"]
    dropped_indexes: set[int] = set()

    for i, left in enumerate(filtered_boxes):
        left_bbox = _extract_bbox(left)
        if left_bbox is None:
            dropped_indexes.add(i)
            continue
        x0, y0, x1, y1 = left_bbox
        if x1 - x0 < 6 or y1 - y0 < 6:
            dropped_indexes.add(i)
            continue

        for j in range(i + 1, len(filtered_boxes)):
            if i in dropped_indexes or j in dropped_indexes:
                continue
            right = filtered_boxes[j]
            right_bbox = _extract_bbox(right)
            if right_bbox is None:
                dropped_indexes.add(j)
                continue

            overlap_ratio = _bbox_overlap_ratio(left_bbox, right_bbox, mode="small")
            left_label = _normalize_layout_label(_extract_label(left))
            right_label = _normalize_layout_label(_extract_label(right))
            if "inline_formula" in {left_label, right_label} and overlap_ratio > 0.5:
                if left_label == "inline_formula":
                    dropped_indexes.add(i)
                if right_label == "inline_formula":
                    dropped_indexes.add(j)
                continue

            if overlap_ratio <= 0.7:
                continue

            if layout_shape_mode != "rect":
                left_poly = _extract_polygon_points(left)
                right_poly = _extract_polygon_points(right)
                if left_poly and right_poly:
                    poly_overlap_ratio = _polygon_overlap_ratio(
                        left_poly,
                        right_poly,
                        mode="small",
                    )
                    if poly_overlap_ratio is not None and poly_overlap_ratio < 0.7:
                        continue

            labels = {left_label, right_label}
            if labels & {"image", "table", "seal", "chart"} and len(labels) > 1:
                if "table" not in labels or labels <= {"table", "image", "seal", "chart"}:
                    continue

            if _bbox_area(left_bbox) >= _bbox_area(right_bbox):
                dropped_indexes.add(j)
            else:
                dropped_indexes.add(i)
                break

    return [box for idx, box in enumerate(filtered_boxes) if idx not in dropped_indexes]


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_overlap_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    *,
    mode: str = "union",
) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    ix0 = max(lx0, rx0)
    iy0 = max(ly0, ry0)
    ix1 = min(lx1, rx1)
    iy1 = min(ly1, ry1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    left_area = _bbox_area(left)
    right_area = _bbox_area(right)
    if mode == "small":
        denom = min(left_area, right_area)
    elif mode == "large":
        denom = max(left_area, right_area)
    else:
        denom = left_area + right_area - intersection
    if denom <= 0:
        return 0.0
    return intersection / denom


def _polygon_overlap_ratio(
    left: list[list[float]],
    right: list[list[float]],
    *,
    mode: str = "union",
) -> float | None:
    try:
        from shapely.geometry import Polygon
    except ImportError:
        return None

    left_poly = Polygon(left)
    right_poly = Polygon(right)
    if not left_poly.is_valid:
        left_poly = left_poly.buffer(0)
    if not right_poly.is_valid:
        right_poly = right_poly.buffer(0)
    intersection = left_poly.intersection(right_poly).area
    if intersection <= 0:
        return 0.0
    if mode == "small":
        denom = min(left_poly.area, right_poly.area)
    elif mode == "large":
        denom = max(left_poly.area, right_poly.area)
    else:
        denom = left_poly.union(right_poly).area
    if denom <= 0:
        return 0.0
    return float(intersection / denom)


def _extract_bbox(box: dict[str, Any]) -> tuple[float, float, float, float] | None:
    for key in ("coordinate", "bbox", "box"):
        bbox = _bbox_from_value(box.get(key))
        if bbox is not None:
            return bbox
    return None


def _extract_label(box: dict[str, Any]) -> str:
    value = box.get("label") or "layout"
    return str(value)


def _normalize_layout_label(label: str) -> str:
    return str(label).strip().lower() or "layout"


def _normalize_region_type(label: str) -> str:
    lowered = str(label).strip().lower()
    if "table" in lowered:
        return "table"
    if any(token in lowered for token in ("formula", "equation", "math")):
        return "formula"
    if "chart" in lowered:
        return "chart"
    return "text"


def _extract_score(box: dict[str, Any]) -> float | None:
    for key in ("score", "confidence"):
        value = box.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_polygon_points(box: dict[str, Any]) -> list[list[float]] | None:
    for key in ("polygon_points", "poly"):
        points = _polygon_points_from_value(box.get(key))
        if points is not None:
            return points
    return None


def _bbox_from_value(value: Any) -> tuple[float, float, float, float] | None:
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


def _polygon_points_from_value(value: Any) -> list[list[float]] | None:
    points_seq = _to_list(value)
    if points_seq is None:
        return None
    points: list[list[float]] = []
    for point in points_seq:
        point_values = _to_list(point)
        if point_values is None or len(point_values) != 2:
            return None
        try:
            x, y = (float(item) for item in point_values)
        except (TypeError, ValueError):
            return None
        points.append([x, y])
    return points or None


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


def _extract_group_id(box: dict[str, Any]) -> int | str | None:
    for key in ("group_id", "global_group_id"):
        value = box.get(key)
        if isinstance(value, (int, str)) and str(value).strip():
            return value
    return None


def _render_kind_for_label(label: str) -> str:
    lowered = _normalize_layout_label(label)
    if "table" in lowered:
        return "table"
    if any(token in lowered for token in ("formula", "equation", "math")):
        return "formula"
    if lowered == "chart":
        return "chart"
    if lowered in {"image", "header_image", "footer_image", "seal"}:
        return "image"
    return "text"


def _is_image_like_label(label: str) -> bool:
    return _normalize_layout_label(label) in _DEFAULT_IMAGE_LIKE_LABELS


def _is_ignorable_for_markdown(markdown_label: str | None) -> bool:
    return markdown_label in {
        "number",
        "footnote",
        "header",
        "header_image",
        "footer",
        "footer_image",
        "aside_text",
    }


def _markdown_label_for_layout_label(label: str) -> str | None:
    normalized = _normalize_layout_label(label)
    return _LAYOUT_LABEL_TO_MARKDOWN_LABEL.get(normalized)


def _normalize_box(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value

    json_attr = getattr(value, "json", None)
    if isinstance(json_attr, dict):
        return json_attr

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, dict):
            return data

    attr_names = (
        "cls_id",
        "label",
        "score",
        "coordinate",
        "bbox",
        "box",
        "order",
        "polygon_points",
    )
    attrs: dict[str, Any] = {}
    for name in attr_names:
        if hasattr(value, name):
            attrs[name] = getattr(value, name)
    if attrs:
        return attrs

    return None


def _safe_id(value: str | None) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in (value or "layout").lower())
    return text.strip("_") or "layout"


def _unique_region_id(prefix: str, used_ids: set[str]) -> str:
    candidate = prefix
    suffix = 1
    while candidate in used_ids:
        suffix += 1
        candidate = f"{prefix}_{suffix}"
    used_ids.add(candidate)
    return candidate


_LAYOUT_LABEL_TO_MARKDOWN_LABEL = {
    "abstract": "abstract",
    "abstract_title": "abstract_title",
    "algorithm": "algorithm",
    "aside_text": "aside_text",
    "chart": "chart",
    "chart_title": "chart_title",
    "content": "content",
    "content_title": "content_title",
    "display_formula": "display_formula",
    "doc_title": "doc_title",
    "equation_inline": "inline_formula",
    "equation_isolated": "display_formula",
    "figure_title": "figure_title",
    "footer": "footer",
    "footer_image": "footer_image",
    "footnote": "footnote",
    "formula": "formula",
    "formula_number": "formula_number",
    "header": "header",
    "header_image": "header_image",
    "image": "image",
    "inline_formula": "inline_formula",
    "number": "number",
    "ocr": "ocr",
    "paragraph_title": "paragraph_title",
    "reference": "reference",
    "reference_content": "reference_content",
    "reference_title": "reference_title",
    "seal": "seal",
    "spotting": "spotting",
    "table": "table",
    "table_title": "table_title",
    "text": "text",
    "vertical_text": "vertical_text",
    "vision_footnote": "vision_footnote",
}
