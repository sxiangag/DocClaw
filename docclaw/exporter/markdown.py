"""Markdown export from DocClaw DocumentState using PaddleOCR-VL formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docclaw.agent.utils import DocumentState, PageState, Region, has_searchable_text


_RAW_TYPE_TO_MARKDOWN_LABEL = {
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

_MARKDOWN_IGNORE_LABELS = {
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
}

_IMAGE_LIKE_LABELS = {
    "chart",
    "footer_image",
    "header_image",
    "image",
    "seal",
}


def export_page_markdown(
    page: PageState,
    *,
    pretty: bool = True,
    show_formula_number: bool = False,
) -> str:
    """Render one page into markdown using region.raw_type as the block label."""
    from paddlex.inference.common.result.converter import MarkdownConverter

    blocks, imgs_in_doc = _page_to_export_payload(page)
    handle_funcs_dict = _build_handle_funcs_dict(pretty=pretty)
    result = MarkdownConverter.convert(
        blocks,
        handle_funcs_dict=handle_funcs_dict,
        show_formula_number=show_formula_number,
        imgs_in_doc=imgs_in_doc,
    )
    markdown = result.get("markdown_texts")
    if not isinstance(markdown, str):
        raise TypeError("markdown converter returned non-string markdown_texts")
    return markdown


def export_document_markdown(
    document: DocumentState,
    *,
    pretty: bool = True,
    show_formula_number: bool = False,
) -> dict[int, str]:
    """Render all pages into markdown indexed by page_index."""
    return {
        page.page_index: export_page_markdown(
            page,
            pretty=pretty,
            show_formula_number=show_formula_number,
        )
        for page in document.pages
    }


def _page_to_blocks(page: PageState) -> list[Any]:
    blocks, _ = _page_to_export_payload(page)
    return blocks


def _page_to_export_payload(page: PageState) -> tuple[list[Any], list[dict[str, Any]]]:
    from paddlex.inference.pipelines.paddleocr_vl.result import PaddleOCRVLBlock

    artifact_dir = _page_artifact_dir(page)
    grouped_payload = _page_to_grouped_blocks(
        page,
        artifact_dir=artifact_dir,
        block_type=PaddleOCRVLBlock,
    )
    if grouped_payload is None:
        raise ValueError(
            f"page {page.page_index} is missing valid post_layout metadata for markdown export"
        )
    return grouped_payload


def _page_to_grouped_blocks(
    page: PageState,
    *,
    artifact_dir: Path | None,
    block_type: type[Any],
) -> tuple[list[Any], list[dict[str, Any]]] | None:
    post_layout = page.metadata.get("post_layout")
    if not isinstance(post_layout, dict):
        return None
    raw_group_ids = post_layout.get("group_ids")
    groups = post_layout.get("groups")
    if not isinstance(raw_group_ids, list) or not isinstance(groups, dict):
        return None

    region_by_id = {region.region_id: region for region in page.regions}
    blocks: list[Any] = []
    imgs_in_doc: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    for raw_group_id in raw_group_ids:
        group_id = str(raw_group_id).strip()
        if not group_id or group_id in seen_group_ids:
            continue
        seen_group_ids.add(group_id)
        group = groups.get(group_id)
        if not isinstance(group, dict):
            continue

        raw_region_ids = group.get("region_ids")
        if not isinstance(raw_region_ids, list) or not raw_region_ids:
            continue
        group_regions = [
            region_by_id[region_id]
            for region_id in raw_region_ids
            if isinstance(region_id, str) and region_id in region_by_id
        ]
        if not group_regions:
            continue
        ranked_group_regions = sorted(group_regions, key=_group_region_sort_key)
        anchor_region = _group_anchor_region(group, ranked_group_regions)
        markdown_label = _markdown_label_for_group(group, anchor_region)
        if markdown_label is None:
            continue
        if _group_ignore_for_markdown(group, anchor_region, markdown_label=markdown_label):
            continue

        block = _group_to_block(
            page,
            group_id=group_id,
            group=group,
            regions=ranked_group_regions,
            anchor_region=anchor_region,
            markdown_label=markdown_label,
            artifact_dir=artifact_dir,
            block_type=block_type,
            imgs_in_doc=imgs_in_doc,
        )
        if block is None:
            continue
        blocks.append(block)
    return blocks, imgs_in_doc


def _group_to_block(
    page: PageState,
    *,
    group_id: str,
    group: dict[str, Any],
    regions: list[Region],
    anchor_region: Region,
    markdown_label: str,
    artifact_dir: Path | None,
    block_type: type[Any],
    imgs_in_doc: list[dict[str, Any]],
) -> Any | None:
    text = _group_text(regions)
    bbox = _group_bbox(regions)
    if _group_is_image_like(group, anchor_region, markdown_label=markdown_label):
        image = _image_payload_for_bbox(
            page,
            bbox=bbox,
            artifact_dir=artifact_dir,
            artifact_key=group_id,
            markdown_label=markdown_label,
        )
        image_entry = _imgs_in_doc_entry_for_group(
            page,
            group=group,
            artifact_dir=artifact_dir,
            markdown_label=markdown_label,
            bbox=bbox,
        )
        if image_entry is None and not text:
            return None
        if image_entry is not None:
            imgs_in_doc.append(image_entry)
        block = block_type(
            label=markdown_label,
            bbox=bbox,
            content=text,
        )
        if image is not None:
            block.image = image
        return block

    if not text:
        return None
    return block_type(
        label=markdown_label,
        bbox=bbox,
        content=text,
    )


def _image_payload_for_bbox(
    page: PageState,
    *,
    bbox: tuple[float, float, float, float],
    artifact_dir: Path | None,
    artifact_key: str,
    markdown_label: str,
) -> dict[str, Any] | None:
    from PIL import Image

    if artifact_dir is None:
        return None
    source_path = _page_source_path(page)
    if source_path is None:
        return None

    output_path = artifact_dir / (
        f"{_safe_fragment(artifact_key)}_{_safe_fragment(markdown_label)}.png"
    )
    if not output_path.exists():
        _write_region_crop(source_path, output_path, bbox)
    with Image.open(output_path) as image:
        return {
            "path": str(output_path),
            "img": image.copy(),
        }


def _imgs_in_doc_entry_for_group(
    page: PageState,
    *,
    group: dict[str, Any],
    artifact_dir: Path | None,
    markdown_label: str,
    bbox: tuple[float, float, float, float],
) -> dict[str, Any] | None:
    image = _image_payload_for_bbox(
        page,
        bbox=bbox,
        artifact_dir=artifact_dir,
        artifact_key="imgs_in_doc",
        markdown_label=markdown_label,
    )
    if image is None:
        return None
    label = _group_image_label(group, markdown_label=markdown_label)
    return {
        "path": _construct_img_path(label, bbox),
        "img": image["img"].copy(),
        "label": label,
        "coordinate": _bbox_to_int_tuple(bbox),
    }


def _page_artifact_dir(page: PageState) -> Path | None:
    source_path = _page_source_path(page)
    if source_path is None:
        return None
    artifact_dir = (
        source_path.parent
        / ".docclaw_markdown_artifacts"
        / f"page_{page.page_index}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _page_source_path(page: PageState) -> Path | None:
    if not page.image_path:
        return None
    source_path = Path(page.image_path).expanduser()
    if not source_path.exists():
        return None
    return source_path.resolve()


def _write_region_crop(
    source_path: Path,
    output_path: Path,
    bbox: tuple[float, float, float, float],
) -> None:
    from PIL import Image

    with Image.open(source_path) as image:
        width, height = image.size
        x0, y0, x1, y1 = _clamp_bbox_to_image(bbox, width=width, height=height)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("bbox is outside the page image")
        crop = image.crop((x0, y0, x1, y1)).convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path)


def _clamp_bbox_to_image(
    bbox: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(round(value)) for value in bbox)
    return (
        max(0, min(width, x0)),
        max(0, min(height, y0)),
        max(0, min(width, x1)),
        max(0, min(height, y1)),
    )


def _bbox_to_int_tuple(
    bbox: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    return tuple(int(round(value)) for value in bbox)


def _normalized_region_text(region: Region) -> str:
    if not has_searchable_text(region.text):
        return ""
    assert isinstance(region.text, str)
    return region.text.strip()


def _group_text(regions: list[Region]) -> str:
    parts = [text for region in regions if (text := _normalized_region_text(region))]
    return "\n".join(parts)


def _group_bbox(regions: list[Region]) -> tuple[float, float, float, float]:
    x0 = min(region.bbox[0] for region in regions)
    y0 = min(region.bbox[1] for region in regions)
    x1 = max(region.bbox[2] for region in regions)
    y1 = max(region.bbox[3] for region in regions)
    return (float(x0), float(y0), float(x1), float(y1))


def _group_region_sort_key(region: Region) -> tuple[int, int, float, float, str]:
    layout = region.metadata.get("layout")
    merge_order = 10**9
    merge_index = 10**9
    if isinstance(layout, dict):
        raw_merge_order = layout.get("merge_order")
        raw_merge_index = layout.get("merge_index")
        if isinstance(raw_merge_order, (int, float)):
            merge_order = int(raw_merge_order)
        if isinstance(raw_merge_index, (int, float)):
            merge_index = int(raw_merge_index)
    x0, y0, _, _ = region.bbox
    return (merge_order, merge_index, float(y0), float(x0), region.region_id)


def _require_raw_type(region: Region) -> str:
    raw_type = region.raw_type
    if not isinstance(raw_type, str) or not raw_type.strip():
        raise ValueError(f"region {region.region_id} is missing raw_type")
    return raw_type.strip().lower()


def _markdown_label_for_region(region: Region) -> str | None:
    render = region.metadata.get("render")
    if isinstance(render, dict):
        markdown_label = render.get("markdown_label")
        if isinstance(markdown_label, str) and markdown_label.strip():
            return markdown_label.strip().lower()
    raw_type = _require_raw_type(region)
    return _markdown_label_for_raw_type(raw_type)


def _group_anchor_region(group: dict[str, Any], regions: list[Region]) -> Region:
    anchor_region_id = group.get("anchor_region_id")
    if isinstance(anchor_region_id, str):
        for region in regions:
            if region.region_id == anchor_region_id:
                return region
    return regions[0]


def _markdown_label_for_group(group: dict[str, Any], anchor_region: Region) -> str | None:
    label = group.get("label")
    if isinstance(label, str) and label.strip():
        normalized = label.strip().lower()
        if normalized in _RAW_TYPE_TO_MARKDOWN_LABEL:
            return _markdown_label_for_raw_type(normalized)
    return _markdown_label_for_region(anchor_region)


def _group_image_label(group: dict[str, Any], *, markdown_label: str) -> str:
    label = group.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip().lower()
    return markdown_label


def _markdown_label_for_raw_type(raw_type: str) -> str:
    if raw_type not in _RAW_TYPE_TO_MARKDOWN_LABEL:
        raise ValueError(f"unsupported raw_type for markdown export: {raw_type}")
    return _RAW_TYPE_TO_MARKDOWN_LABEL[raw_type]


def _region_ignore_for_markdown(region: Region, *, markdown_label: str) -> bool:
    render = region.metadata.get("render")
    if isinstance(render, dict) and isinstance(render.get("ignore_for_markdown"), bool):
        return bool(render.get("ignore_for_markdown"))
    return markdown_label in _MARKDOWN_IGNORE_LABELS


def _group_ignore_for_markdown(
    group: dict[str, Any],
    anchor_region: Region,
    *,
    markdown_label: str,
) -> bool:
    if isinstance(group.get("ignore"), bool):
        return bool(group.get("ignore"))
    return _region_ignore_for_markdown(anchor_region, markdown_label=markdown_label)


def _region_is_image_like(region: Region, *, markdown_label: str) -> bool:
    render = region.metadata.get("render")
    if isinstance(render, dict):
        image_like = render.get("image_like")
        if isinstance(image_like, bool):
            return image_like
        kind = render.get("kind")
        if isinstance(kind, str) and kind.strip().lower() in {"image", "chart"}:
            return True
    return markdown_label in _IMAGE_LIKE_LABELS


def _group_is_image_like(
    group: dict[str, Any],
    anchor_region: Region,
    *,
    markdown_label: str,
) -> bool:
    if isinstance(group.get("image_like"), bool):
        return bool(group.get("image_like"))
    return _region_is_image_like(anchor_region, markdown_label=markdown_label)


def _build_handle_funcs_dict(*, pretty: bool) -> dict[str, Any]:
    from paddlex.inference.common.result.converter.markdown_format_funcs import (
        build_handle_funcs_dict,
        format_centered_by_html,
        format_image_plain,
        format_image_scaled_by_html,
        format_table_center,
        format_text_plain,
        simplify_table,
    )

    if pretty:
        text_func = lambda block: format_centered_by_html(format_text_plain(block))
        image_func = lambda block: format_centered_by_html(
            format_image_scaled_by_html(
                block,
                original_image_width=_page_image_width_from_block(block),
                show_ocr_content=False,
            )
        )
        table_func = lambda block: "\n" + format_table_center(block)
    else:
        text_func = lambda block: block.content
        image_func = lambda block: format_image_plain(block, show_ocr_content=False)
        table_func = lambda block: simplify_table("\n" + block.content)

    chart_func = image_func
    formula_func = lambda block: block.content
    seal_func = image_func

    handle_funcs_dict = build_handle_funcs_dict(
        text_func=text_func,
        image_func=image_func,
        chart_func=chart_func,
        table_func=table_func,
        formula_func=formula_func,
        seal_func=seal_func,
    )
    for label in _MARKDOWN_IGNORE_LABELS:
        handle_funcs_dict.pop(label, None)
    return handle_funcs_dict


def _page_image_width_from_block(block: Any) -> int:
    bbox = getattr(block, "bbox", None)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 1000
    try:
        x0, _, x1, _ = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return 1000
    width = int(max(x1 - x0, 1.0))
    return max(width, 1)


def _safe_fragment(value: str) -> str:
    fragment = "".join(ch if ch.isalnum() else "_" for ch in value)
    fragment = fragment.strip("_")
    return fragment[:80] or "region"


def _construct_img_path(label: str, bbox: tuple[float, float, float, float]) -> str:
    x0, y0, x1, y1 = (int(round(value)) for value in bbox)
    return f"imgs/img_in_{label}_box_{x0}_{y0}_{x1}_{y1}.jpg"
