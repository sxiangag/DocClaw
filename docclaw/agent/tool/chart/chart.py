"""Abstract chart parsing tool."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from docclaw.agent.tool.tool import Tool
from docclaw.agent.utils import (
    Action,
    ActionType,
    Observation,
    Region,
    RunState,
)


class ChartTool(Tool):
    """Base class for tools that expose chart content from document pages."""

    @property
    def action_type(self) -> ActionType:
        return "parse_chart"

    @property
    def description(self) -> str:
        return (
            "Parse charts from a page set or a known chart-candidate region set "
            "and return structured chart content. Parsed chart results are "
            "written back into page state, and repeating chart parsing on the same "
            "targets usually has no value. Use page_ids exposed in document memory "
            "for page-level parsing, and use known chart-like region_ids from "
            "document memory or recent action_history outputs when layout has "
            "already produced candidate regions. If relevant regions are not yet "
            "known, call parse_layout first."
        )

    @property
    def target_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": (
                "Select exactly one target mode. Use page_ids to parse charts "
                "from whole pages. Use region_ids only for specific known chart-candidate "
                "regions. If you want all charts from selected pages, use page_ids "
                "and set region_ids to [] or omit it entirely. Do not use placeholder "
                "values such as '__all__', 'string', empty strings, or numeric dummy "
                "values. Omit target to inspect all known pages."
            ),
            "properties": {
                "mode": {
                    "type": "string",
                    "description": (
                        "Explicit chart target mode. Use 'page' for page_ids, "
                        "'region' for region_ids, 'zoom_region' for "
                        "zoom_region_ids, and 'crop_region' for crop_region_ids."
                    ),
                    "enum": ["page", "region", "zoom_region", "crop_region"],
                },
                "page_ids": {
                    "type": "array",
                    "description": (
                        "Specific pages to inspect for whole-page chart parsing. "
                        "When using page_ids, region_ids should be [] or omitted."
                    ),
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "region_ids": {
                    "type": "array",
                    "description": (
                        "Known region identifiers to treat as explicit chart "
                        "candidates. When using page_ids, set this to [] or omit it."
                    ),
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "zoom_region_ids": {
                    "type": "array",
                    "description": (
                        "Known region identifiers whose zoom artifacts should be treated "
                        "as explicit chart candidates."
                    ),
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "crop_region_ids": {
                    "type": "array",
                    "description": (
                        "Known region identifiers whose crop artifacts should be treated "
                        "as explicit chart candidates."
                    ),
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
            "description": "Chart parsing controls.",
            "properties": {},
            "additionalProperties": False,
        }

    def document_overview_fragment(self, state: RunState) -> dict[str, Any]:
        return {}

    async def execute(self, state: RunState, action: Action) -> Observation:
        pages, regions, error = _resolve_chart_target(state, action)
        if error is not None:
            return self.error(action, error)
        assert pages is not None

        try:
            results = self.parse_charts(state, pages, regions, action)
        except Exception as exc:
            return self.error(action, str(exc))

        total_charts = len(results)
        source_counts: dict[str, int] = {}
        for result in results:
            source = str(result.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1
        message = (
            f"Chart parsing ready for {len(pages)} page(s); "
            f"{total_charts} chart(s) available"
        )
        message += "."
        return Observation(
            action_id=action.action_id,
            success=True,
            data={
                "results": results,
                "sources": source_counts,
            },
            message=message,
        )

    @abstractmethod
    def parse_charts(
        self,
        state: RunState,
        pages: list[Any],
        regions: list[Region] | None,
        action: Action,
    ) -> list[dict[str, Any]]:
        """Return normalized chart results for the selected pages or region target."""


def _resolve_chart_target(
    state: RunState,
    action: Action,
) -> tuple[list[Any] | None, list[Region] | None, str | None]:
    if "page_index" in action.target or "region_id" in action.target:
        return None, None, "parse_chart requires target.page_ids, target.region_ids, target.zoom_region_ids, or target.crop_region_ids"

    mode, error = _optional_target_mode(action.target.get("mode"), action_type="parse_chart")
    if error is not None:
        return None, None, error

    raw_region_ids = action.target.get("region_ids")
    cleaned_region_ids = (
        [
            region_id
            for raw_region_id in raw_region_ids
            if (region_id := _optional_non_empty_region_id(raw_region_id)) is not None
        ]
        if isinstance(raw_region_ids, list)
        else None
    )
    raw_zoom_region_ids = action.target.get("zoom_region_ids")
    cleaned_zoom_region_ids = (
        [
            region_id
            for raw_region_id in raw_zoom_region_ids
            if (region_id := _optional_non_empty_region_id(raw_region_id)) is not None
        ]
        if isinstance(raw_zoom_region_ids, list)
        else None
    )
    raw_crop_region_ids = action.target.get("crop_region_ids")
    cleaned_crop_region_ids = (
        [
            region_id
            for raw_region_id in raw_crop_region_ids
            if (region_id := _optional_non_empty_region_id(raw_region_id)) is not None
        ]
        if isinstance(raw_crop_region_ids, list)
        else None
    )

    if mode == "region":
        return _resolve_explicit_region_target(
            state,
            action.target,
            cleaned_region_ids,
            action_type="parse_chart",
        )
    if mode == "zoom_region":
        return _resolve_explicit_zoom_region_target(
            state,
            action.target,
            cleaned_zoom_region_ids,
            action_type="parse_chart",
        )
    if mode == "crop_region":
        return _resolve_explicit_crop_region_target(
            state,
            action.target,
            cleaned_crop_region_ids,
            action_type="parse_chart",
        )
    if mode == "page":
        return _resolve_explicit_page_target(
            state,
            action.target,
            action_type="parse_chart",
        )

    has_page_indices = action.target.get("page_indices") is not None
    if has_page_indices and cleaned_region_ids:
        cleaned_region_ids = [
            region_id for region_id in cleaned_region_ids if state.get_region(region_id) is not None
        ]
    if has_page_indices and cleaned_zoom_region_ids:
        cleaned_zoom_region_ids = [
            region_id for region_id in cleaned_zoom_region_ids if state.get_zoom_region_view(region_id) is not None
        ]
    if has_page_indices and cleaned_crop_region_ids:
        cleaned_crop_region_ids = [
            region_id for region_id in cleaned_crop_region_ids if state.get_crop_region_view(region_id) is not None
        ]
    has_region_ids = bool(cleaned_region_ids)
    has_zoom_region_ids = bool(cleaned_zoom_region_ids)
    has_crop_region_ids = bool(cleaned_crop_region_ids)
    mode_count = (
        int(has_page_indices)
        + int(has_region_ids)
        + int(has_zoom_region_ids)
        + int(has_crop_region_ids)
    )
    if mode_count > 1:
        return None, None, (
            "parse_chart target must specify exactly one of "
            "target.region_ids, target.zoom_region_ids, target.crop_region_ids, or target.page_ids"
        )

    if raw_region_ids is not None:
        return _resolve_explicit_region_target(
            state,
            action.target,
            cleaned_region_ids,
            action_type="parse_chart",
            allow_page_mode_omission=True,
        )

    if raw_zoom_region_ids is not None:
        return _resolve_explicit_zoom_region_target(
            state,
            action.target,
            cleaned_zoom_region_ids,
            action_type="parse_chart",
        )
    if raw_crop_region_ids is not None:
        return _resolve_explicit_crop_region_target(
            state,
            action.target,
            cleaned_crop_region_ids,
            action_type="parse_chart",
        )

    if action.target.get("page_indices") is not None:
        return _resolve_explicit_page_target(
            state,
            action.target,
            action_type="parse_chart",
        )
    return list(state.document.pages), None, None


def _resolve_explicit_region_target(
    state: RunState,
    target_payload: dict[str, Any],
    cleaned_region_ids: list[str] | None,
    *,
    action_type: str,
    allow_page_mode_omission: bool = False,
) -> tuple[list[Any] | None, list[Region] | None, str | None]:
    region_ids = target_payload.get("region_ids")
    if not isinstance(region_ids, list) or not region_ids:
        return None, None, f"{action_type} target.region_ids must be a non-empty list"
    if not cleaned_region_ids:
        if allow_page_mode_omission and target_payload.get("page_indices") is not None:
            return _resolve_explicit_page_target(state, target_payload, action_type=action_type)
        return None, None, f"{action_type} target.region_ids must contain non-empty strings"
    pages: list[Any] = []
    regions: list[Region] = []
    seen_page_indexes: set[int] = set()
    seen_region_ids: set[str] = set()
    for region_id in cleaned_region_ids:
        if region_id in seen_region_ids:
            continue
        region = state.get_region(region_id)
        if region is None:
            return None, None, f"unknown region_id: {region_id}"
        regions.append(region)
        seen_region_ids.add(region_id)
        if region.page_index not in seen_page_indexes:
            pages.append(state.require_page(region.page_index))
            seen_page_indexes.add(region.page_index)
    return pages, regions, None


def _resolve_explicit_zoom_region_target(
    state: RunState,
    target_payload: dict[str, Any],
    cleaned_zoom_region_ids: list[str] | None,
    *,
    action_type: str,
) -> tuple[list[Any] | None, list[Region] | None, str | None]:
    zoom_region_ids = target_payload.get("zoom_region_ids")
    if not isinstance(zoom_region_ids, list) or not zoom_region_ids:
        return None, None, f"{action_type} target.zoom_region_ids must be a non-empty list"
    if not cleaned_zoom_region_ids:
        return None, None, f"{action_type} target.zoom_region_ids must contain non-empty strings"
    pages: list[Any] = []
    regions: list[Region] = []
    seen_page_indexes: set[int] = set()
    seen_region_ids: set[str] = set()
    for region_id in cleaned_zoom_region_ids:
        if region_id in seen_region_ids:
            continue
        region = state.get_region(region_id)
        if region is None:
            return None, None, f"unknown region_id: {region_id}"
        if state.get_zoom_region_view(region_id) is None:
            return None, None, f"unknown zoom_region_id: {region_id}"
        regions.append(region)
        seen_region_ids.add(region_id)
        if region.page_index not in seen_page_indexes:
            pages.append(state.require_page(region.page_index))
            seen_page_indexes.add(region.page_index)
    return pages, regions, None


def _resolve_explicit_crop_region_target(
    state: RunState,
    target_payload: dict[str, Any],
    cleaned_crop_region_ids: list[str] | None,
    *,
    action_type: str,
) -> tuple[list[Any] | None, list[Region] | None, str | None]:
    crop_region_ids = target_payload.get("crop_region_ids")
    if not isinstance(crop_region_ids, list) or not crop_region_ids:
        return None, None, f"{action_type} target.crop_region_ids must be a non-empty list"
    if not cleaned_crop_region_ids:
        return None, None, f"{action_type} target.crop_region_ids must contain non-empty strings"
    pages: list[Any] = []
    regions: list[Region] = []
    seen_page_indexes: set[int] = set()
    seen_region_ids: set[str] = set()
    for region_id in cleaned_crop_region_ids:
        if region_id in seen_region_ids:
            continue
        region = state.get_region(region_id)
        if region is None:
            return None, None, f"unknown region_id: {region_id}"
        if state.get_crop_region_view(region_id) is None:
            return None, None, f"unknown crop_region_id: {region_id}"
        regions.append(region)
        seen_region_ids.add(region_id)
        if region.page_index not in seen_page_indexes:
            pages.append(state.require_page(region.page_index))
            seen_page_indexes.add(region.page_index)
    return pages, regions, None


def _resolve_explicit_page_target(
    state: RunState,
    target_payload: dict[str, Any],
    *,
    action_type: str,
) -> tuple[list[Any] | None, list[Region] | None, str | None]:
    raw_page_indices = target_payload.get("page_indices")
    if not isinstance(raw_page_indices, list) or not raw_page_indices:
        return None, None, f"{action_type} target.page_ids must be a non-empty list"
    pages: list[Any] = []
    seen_page_indexes: set[int] = set()
    for raw_page_index in raw_page_indices:
        try:
            page_index = int(raw_page_index)
        except (TypeError, ValueError):
            return None, None, f"{action_type} target.page_ids must contain valid page ids"
        if page_index in seen_page_indexes:
            continue
        try:
            pages.append(state.require_page(page_index))
        except ValueError as exc:
            return None, None, str(exc)
        seen_page_indexes.add(page_index)
    return pages, None, None


def _optional_non_empty_region_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {":dummy:", "__all__", "string"}:
        return None
    if not any(ch.isalnum() for ch in text):
        return None
    return text


def _optional_target_mode(
    value: Any,
    *,
    action_type: str,
) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    text = str(value).strip()
    if not text:
        return None, None
    if text not in {"page", "region", "zoom_region", "crop_region"}:
        return None, f"{action_type} target.mode must be one of 'page', 'region', 'zoom_region', or 'crop_region'"
    return text, None
