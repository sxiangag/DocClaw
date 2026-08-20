"""Deterministic page transcription from OCR results."""

from __future__ import annotations

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
)
from docclaw.agent.tool.tool import Tool
from docclaw.agent.utils import (
    Action,
    ActionType,
    Observation,
    PageState,
    Region,
    RunState,
    has_searchable_text,
)


class TranscribeTool(Tool):
    """Transcribe pages from OCR results into page-level text and a final answer."""

    @property
    def action_type(self) -> ActionType:
        return "transcribe"

    @property
    def description(self) -> str:
        return (
            "Transcribe a page into stable page-level text from OCR results. Prefer using "
            "region OCR results in reading order when they exist; otherwise reuse "
            "existing page OCR text. This tool writes transcribed page OCR back into "
            "state and can finalize the current run answer for transcription tasks."
        )

    @property
    def target_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "Explicit page set whose OCR text should be transcribed into final page text.",
            "properties": {
                "page_ids": {
                    "type": "array",
                    "description": "Page ids to transcribe into page-level OCR text.",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["page_ids"],
            "additionalProperties": False,
        }

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "Transcription controls.",
            "properties": {},
            "additionalProperties": False,
        }

    async def execute(self, state: RunState, action: Action) -> Observation:
        page_indices, error = _resolve_page_indices(state, action)
        if error is not None:
            return self.error(action, error)
        assert page_indices is not None

        pages: list[dict[str, Any]] = []
        answer_parts: list[str] = []
        for page_index in page_indices:
            page = state.require_page(page_index)
            assembled_text, source_kind, region_ids = _assemble_page_text(page)
            if not has_searchable_text(assembled_text):
                return self.error(
                    action,
                    f"page {page_index} has no OCR text available to transcribe",
                )
            page_payload = {
                "page_index": page_index,
                "text": assembled_text,
                "transcription_text": assembled_text,
                "source_kind": source_kind,
                "region_ids": region_ids,
                "region_count": len(region_ids),
            }
            markdown = _markdown_for_page_payload(state, page_payload)
            if has_searchable_text(markdown):
                assert isinstance(markdown, str)
                page_payload["text"] = markdown
                page_payload["markdown"] = markdown
                page_payload["format"] = "markdown"
                answer_parts.append(markdown)
            else:
                page_payload["format"] = "text"
                answer_parts.append(assembled_text)
            pages.append(page_payload)

        answer = "\n\n".join(part for part in answer_parts if part.strip())
        answer_format = (
            "markdown"
            if pages and all(page.get("format") == "markdown" for page in pages)
            else "text"
        )
        return Observation(
            action_id=action.action_id,
            success=True,
            data={
                "pages": pages,
                "answer": answer,
                "source": f"transcribed_page_{answer_format}",
                "format": answer_format,
            },
            message=(
                f"Transcribed {len(pages)} page(s) from OCR results "
                f"({sum(page['region_count'] for page in pages)} region(s) used)."
            ),
        )

    def update_state(
        self,
        state: RunState,
        action: Action,
        observation: Observation,
    ) -> None:
        pages = observation.data.get("pages")
        if isinstance(pages, list):
            for item in pages:
                if not isinstance(item, dict):
                    continue
                page_index = item.get("page_index")
                text = item.get("text")
                if not isinstance(page_index, int) or not isinstance(text, str):
                    continue
                page = state.get_page(page_index)
                if page is None:
                    continue
                page.ocr_text = text
                page.metadata["ocr_text_format"] = item.get("format") or "text"
                source_kind = item.get("source_kind")
                if isinstance(source_kind, str) and source_kind.strip():
                    page.metadata["ocr_source_kind"] = source_kind
                page.metadata["ocr_assembled_region_ids"] = list(item.get("region_ids") or [])
        final_answer = _build_final_answer_from_payload(pages)
        if isinstance(final_answer, str) and final_answer.strip():
            state.final_answer = final_answer
            state.status = "completed"


def _resolve_page_indices(
    state: RunState,
    action: Action,
) -> tuple[list[int] | None, str | None]:
    if "page_index" in action.target:
        return None, "transcribe requires target.page_ids"
    raw_page_indices = action.target.get("page_indices")
    if not isinstance(raw_page_indices, list) or not raw_page_indices:
        return None, "transcribe target.page_ids must be a non-empty list"
    page_indices: list[int] = []
    seen: set[int] = set()
    for raw_page_index in raw_page_indices:
        page_index = int(raw_page_index)
        state.require_page(page_index)
        if page_index in seen:
            continue
        page_indices.append(page_index)
        seen.add(page_index)
    return page_indices, None


def _assemble_page_text(page) -> tuple[str | None, str, list[str]]:
    complete_region_text = assemble_complete_region_page_text(page)
    if complete_region_text is not None:
        return complete_region_text

    cached_page_text = _canonical_page_ocr_text(page)
    if has_searchable_text(cached_page_text):
        assert isinstance(cached_page_text, str)
        return cached_page_text, "page_ocr", []

    grouped = _assemble_page_text_from_post_layout(page)
    if grouped is not None:
        return grouped

    ranked_regions = sorted(
        page.regions,
        key=_region_sort_key,
    )
    region_texts: list[str] = []
    used_region_ids: list[str] = []
    used_parsed = False
    for region in ranked_regions:
        region_text = _region_transcription_text(region)
        if not has_searchable_text(region_text):
            continue
        assert isinstance(region_text, str)
        region_texts.append(region_text.strip())
        used_region_ids.append(region.region_id)
        used_parsed = used_parsed or (
            has_searchable_text(region.text) and isinstance(region.type, str) and region.type != "text"
        )
    if region_texts:
        source_kind = "region_parsed" if used_parsed else "region_ocr"
        return "\n".join(region_texts), source_kind, used_region_ids
    if has_searchable_text(page.ocr_text):
        return page.ocr_text, "page_ocr", []
    return None, "none", []


def _canonical_page_ocr_text(page) -> str | None:
    text = page.metadata.get("page_ocr_text")
    if has_searchable_text(text):
        assert isinstance(text, str)
        return text

    source_kind = page.metadata.get("ocr_source_kind")
    if isinstance(source_kind, str) and source_kind.startswith("region_"):
        return None
    if has_searchable_text(page.ocr_text):
        return page.ocr_text
    return None


def assemble_complete_region_page_text(page) -> tuple[str, str, list[str]] | None:
    """Return page text assembled from region text only when layout coverage is complete."""
    if not _has_layout_context(page):
        return None

    eligible_regions = _eligible_text_regions(page)
    if not eligible_regions:
        return None

    for region in eligible_regions:
        if not has_searchable_text(_region_transcription_text(region)):
            return None

    eligible_region_ids = {region.region_id for region in eligible_regions}
    grouped = _assemble_page_text_from_post_layout(page)
    if grouped is not None:
        _, _, used_region_ids = grouped
        if set(used_region_ids) >= eligible_region_ids:
            return grouped

    return _assemble_regions_in_reading_order(eligible_regions)


def _build_final_answer_from_export(
    state: RunState,
    pages_payload: Any,
) -> str | None:
    if not isinstance(pages_payload, list) or not pages_payload:
        return None

    markdown_parts: list[str] = []
    for item in pages_payload:
        if not isinstance(item, dict):
            continue
        markdown = _markdown_for_page_payload(state, item)
        if has_searchable_text(markdown):
            assert isinstance(markdown, str)
            markdown_parts.append(markdown.strip())
    if not markdown_parts:
        return None
    return "\n\n".join(markdown_parts)


def _build_final_answer_from_payload(pages_payload: Any) -> str | None:
    if not isinstance(pages_payload, list) or not pages_payload:
        return None
    parts = [
        item["text"].strip()
        for item in pages_payload
        if isinstance(item, dict) and has_searchable_text(item.get("text"))
    ]
    if not parts:
        return None
    return "\n\n".join(parts)


def _markdown_for_page_payload(state: RunState, item: dict[str, Any]) -> str | None:
    page_index = item.get("page_index")
    if not isinstance(page_index, int):
        return None
    page = state.get_page(page_index)
    if page is None:
        return None

    from docclaw.exporter import export_page_markdown

    export_page = _page_ocr_blocks_export_page(page)
    if export_page is not None:
        try:
            return export_page_markdown(export_page, pretty=True)
        except (TypeError, ValueError):
            cached_markdown = page.metadata.get("page_ocr_markdown")
            if has_searchable_text(cached_markdown):
                assert isinstance(cached_markdown, str)
                return cached_markdown

    cached_markdown = page.metadata.get("page_ocr_markdown")
    if has_searchable_text(cached_markdown):
        assert isinstance(cached_markdown, str)
        return cached_markdown

    export_page = _state_regions_export_page(page)
    _ensure_post_layout_groups(export_page)
    try:
        return export_page_markdown(export_page, pretty=True)
    except (TypeError, ValueError):
        fallback_text = item.get("text")
        return fallback_text if has_searchable_text(fallback_text) else None


def _state_regions_export_page(page) -> PageState:
    metadata = dict(page.metadata)
    regions = list(page.regions)
    if not isinstance(metadata.get("post_layout"), dict):
        regions = sorted(regions, key=_region_sort_key)

    export_page = PageState(
        page_index=page.page_index,
        width=page.width,
        height=page.height,
        image_path=page.image_path,
        ocr_text=page.ocr_text,
        ocr_confidence=page.ocr_confidence,
        metadata=metadata,
    )
    export_page.regions.extend(Region.from_dict(region.to_dict()) for region in regions)
    return export_page


def _page_ocr_blocks_export_page(page) -> PageState | None:
    raw_blocks = page.metadata.get("page_ocr_blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        return None

    export_page = PageState(
        page_index=page.page_index,
        width=page.width,
        height=page.height,
        image_path=page.image_path,
        ocr_text=(
            page.metadata.get("page_ocr_text")
            if has_searchable_text(page.metadata.get("page_ocr_text"))
            else page.ocr_text
        ),
        ocr_confidence=page.ocr_confidence,
        metadata={
            "layout_source": page.metadata.get("page_ocr_layout_source")
            or page.metadata.get("layout_source"),
        },
    )

    regions: list[Region] = []
    boxes: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, raw_block in enumerate(raw_blocks):
        if not isinstance(raw_block, dict):
            continue
        region = _region_from_page_ocr_block(
            page,
            raw_block,
            fallback_index=index,
            used_ids=used_ids,
        )
        if region is None:
            continue
        regions.append(region)
        boxes.append(_layout_raw_box_from_page_ocr_block(raw_block))

    if not regions:
        return None

    export_page.regions.extend(regions)
    export_page.metadata["layout_raw_boxes"] = boxes
    _ensure_post_layout_groups(export_page, regions, boxes=boxes)
    return export_page


def _region_from_page_ocr_block(
    page,
    block: dict[str, Any],
    *,
    fallback_index: int,
    used_ids: set[str],
) -> Region | None:
    bbox = _bbox_from_page_ocr_block(block)
    if bbox is None:
        return None
    label = _normalize_layout_label(str(block.get("label") or "text"))
    block_id = block.get("block_id")
    suffix = block_id if isinstance(block_id, int) else fallback_index
    region_id = _unique_region_id(
        f"p{page.page_number}_{_safe_id(label)}_{suffix}",
        used_ids,
    )
    markdown_label = _markdown_label_for_layout_label(label)
    text = block.get("text")
    return Region(
        page_index=page.page_index,
        bbox=bbox,
        region_id=region_id,
        label=label,
        raw_type=label,
        type=_normalize_region_type(label),
        text=text if has_searchable_text(text) else None,
        confidence=None,
        coordinate_space="pixel",
        metadata={
            "raw": _layout_raw_box_from_page_ocr_block(block),
            "layout": {
                "source": page.metadata.get("page_ocr_layout_source"),
                "label": label,
                "score": None,
                "sequence_order": suffix,
                "reading_order": block.get("order") if isinstance(block.get("order"), int) else None,
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
        },
    )


def _bbox_from_page_ocr_block(block: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = block.get("bbox")
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


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


def _unique_region_id(base: str, used_ids: set[str]) -> str:
    candidate = base
    suffix = 1
    while candidate in used_ids:
        suffix += 1
        candidate = f"{base}_{suffix}"
    used_ids.add(candidate)
    return candidate


def _assemble_page_text_from_post_layout(page) -> tuple[str | None, str, list[str]] | None:
    post_layout = page.metadata.get("post_layout")
    if not isinstance(post_layout, dict):
        return None

    raw_group_ids = post_layout.get("group_ids")
    groups = post_layout.get("groups")
    if not isinstance(raw_group_ids, list) or not isinstance(groups, dict):
        return None

    region_by_id = {region.region_id: region for region in page.regions}
    group_texts: list[str] = []
    used_region_ids: list[str] = []
    used_parsed = False
    seen_groups: set[str] = set()

    for raw_group_id in raw_group_ids:
        group_id = str(raw_group_id).strip()
        if not group_id or group_id in seen_groups:
            continue
        seen_groups.add(group_id)
        group = groups.get(group_id)
        if not isinstance(group, dict):
            continue
        if bool(group.get("image_like")):
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
        group_parts: list[str] = []
        for region in ranked_group_regions:
            region_text = _region_transcription_text(region)
            if not has_searchable_text(region_text):
                continue
            assert isinstance(region_text, str)
            group_parts.append(region_text.strip())
            used_region_ids.append(region.region_id)
            used_parsed = used_parsed or (
                has_searchable_text(region.text)
                and isinstance(region.type, str)
                and region.type != "text"
            )

        if group_parts:
            group_texts.append("\n".join(group_parts))

    if group_texts:
        source_kind = "region_parsed" if used_parsed else "region_ocr"
        return "\n".join(group_texts), source_kind, used_region_ids
    return None


def _region_transcription_text(region) -> str | None:
    if has_searchable_text(region.text):
        return region.text
    return None


def _has_layout_context(page) -> bool:
    if isinstance(page.metadata.get("post_layout"), dict):
        return True
    if has_searchable_text(page.metadata.get("layout_source")):
        return True
    return any(isinstance(region.metadata.get("layout"), dict) for region in page.regions)


def _eligible_text_regions(page) -> list[Any]:
    return [
        region
        for region in page.regions
        if not _is_image_like_region(region)
    ]


def _is_image_like_region(region) -> bool:
    render = region.metadata.get("render")
    if isinstance(render, dict) and bool(render.get("image_like")):
        return True
    return str(region.type or region.label or "").strip().lower() in {
        "chart",
        "figure",
        "image",
        "picture",
    }


def _assemble_regions_in_reading_order(regions: list[Any]) -> tuple[str, str, list[str]]:
    ranked_regions = sorted(regions, key=_region_sort_key)
    region_texts: list[str] = []
    used_region_ids: list[str] = []
    used_parsed = False
    for region in ranked_regions:
        region_text = _region_transcription_text(region)
        if not has_searchable_text(region_text):
            continue
        assert isinstance(region_text, str)
        region_texts.append(region_text.strip())
        used_region_ids.append(region.region_id)
        used_parsed = used_parsed or (
            has_searchable_text(region.text)
            and isinstance(region.type, str)
            and region.type != "text"
        )
    source_kind = "region_parsed" if used_parsed else "region_ocr"
    return "\n".join(region_texts), source_kind, used_region_ids


def _region_sort_key(region) -> tuple[int, float, float, str]:
    raw_order = _region_sequence_order(region)
    order = int(raw_order) if isinstance(raw_order, (int, float)) else 10**9
    x0, y0, _, _ = region.bbox
    return (order, float(y0), float(x0), region.region_id)


def _region_sequence_order(region) -> int | float | None:
    layout = region.metadata.get("layout")
    if isinstance(layout, dict):
        order = layout.get("sequence_order")
        if isinstance(order, (int, float)):
            return order
    return None


def _group_region_sort_key(region) -> tuple[int, int, float, float, str]:
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
