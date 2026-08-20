"""Abstract layout parsing tool."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from docclaw.agent.tool.tool import Tool
from docclaw.agent.utils import Action, ActionType, Observation, RunState, page_id_from_index


class LayoutTool(Tool):
    """Base class for tools that expose document layout regions."""

    @property
    def action_type(self) -> ActionType:
        return "parse_layout"

    @property
    def description(self) -> str:
        return (
            "Parse layout for one page, a page set, or the whole document and return "
            "page/region inventory. This creates structural regions but does not "
            "create searchable OCR text on its own. Successful results are written "
            "into document page/region state, and repeating layout parsing on the "
            "same pages usually has no value. Use page_ids exposed in document "
            "memory for targeting, and use task memory action_history to avoid "
            "repeating layout parsing on pages that have already produced region "
            "ids. Region ids encode page number, region type, and local order, for "
            "example p14_chart_13 or p7_text_5."
        )

    @property
    def target_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "Optional explicit page set for layout parsing. Omit target to parse all known pages.",
            "properties": {
                "page_ids": {
                    "type": "array",
                    "description": "Specific pages to inspect as an explicit set.",
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
            "description": "Layout parsing controls.",
            "properties": {},
            "additionalProperties": False,
        }

    def document_overview_fragment(self, state: RunState) -> dict[str, Any]:
        pages_with_layout = [
            page_id_from_index(page.page_index, document=state.document) for page in state.document.pages if page.regions
        ]
        unparsed_page_indexes = [
            page_id_from_index(page.page_index, document=state.document) for page in state.document.pages if not page.regions
        ]
        page_region_inventory = [
            {
                "page_index": page.page_index,
                "region_ids": [region.region_id for region in page.regions],
            }
            for page in state.document.pages
            if page.regions
        ]
        return {
            "layout": {
                "pages_have_been_layout_analyzed": pages_with_layout,
                "pages_have_not_been_layout_analyzed": unparsed_page_indexes,
                "pages_have_been_layout_analyzed_details": page_region_inventory,
            }
        }

    async def execute(self, state: RunState, action: Action) -> Observation:
        try:
            pages = _resolve_layout_pages(state, action)
        except ValueError as exc:
            return self.error(action, str(exc))

        try:
            payload = self.parse_layout(state, pages, action)
        except Exception as exc:
            return self.error(action, str(exc))
        skipped_pages = sum(1 for page in payload if page.get("skipped"))
        total_regions = sum(
            len(page_payload.get("regions", []))
            for page_payload in payload
            if isinstance(page_payload.get("regions"), list)
        )
        page_indexes = sorted(
            {
                page_index
                for page_payload in payload
                if isinstance((page_index := page_payload.get("page_index")), int)
            }
        )
        page_ids = [page_id_from_index(page_index, document=state.document) for page_index in page_indexes]
        page_label = ", ".join(page_ids)
        message = (
            f"Layout ready for {len(payload)} page(s)"
            + (f" ({page_label})" if page_label else "")
            + f"; {total_regions} region(s) available"
        )
        if skipped_pages:
            message += f", {skipped_pages} page(s) reused existing layout"
        message += "."
        return Observation(
            action_id=action.action_id,
            success=True,
            data={"pages": payload},
            message=message,
        )

    @abstractmethod
    def parse_layout(
        self,
        state: RunState,
        pages: list[Any],
        action: Action,
    ) -> list[dict[str, Any]]:
        """Return layout payloads for selected pages."""


def _resolve_layout_pages(state: RunState, action: Action) -> list[Any]:
    if "page_index" in action.target:
        raise ValueError("parse_layout requires target.page_ids")

    raw_page_indices = action.target.get("page_indices")
    if raw_page_indices is not None:
        if not isinstance(raw_page_indices, list) or not raw_page_indices:
            raise ValueError("parse_layout target.page_ids must be a non-empty list")
        pages: list[Any] = []
        seen: set[int] = set()
        for raw_page_index in raw_page_indices:
            page_index = int(raw_page_index)
            if page_index in seen:
                continue
            pages.append(state.require_page(page_index))
            seen.add(page_index)
        return pages

    return list(state.document.pages)
