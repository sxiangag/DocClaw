"""General VLM-backed table parsing implementation."""

from __future__ import annotations

from typing import Any

from docclaw.agent.tool.table.paddleocrvl import (
    _Candidate,
    _cached_tables_for_page,
    _table_candidates_for_page,
    _uses_crop_regions,
    _uses_refined_regions,
    _uses_zoom_regions,
    _candidate_image_input,
)
from docclaw.agent.tool.table.table import TableTool, _resolve_table_target
from docclaw.agent.tool.vlm_client import VLMClient
from docclaw.agent.utils import Action, PageState, Region, RunState, plannerize_page_refs
from docclaw.provider.base import LLMProvider


class VLMTableTool(TableTool):
    """Parse tables with a general multimodal language model."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.client = VLMClient(
            provider,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @property
    def description(self) -> str:
        return (
            "Parse tables from known table regions or page-level table candidates "
            "with a general multimodal language model. This keeps DocClaw layout "
            "targeting and uses VLM recognition for table structure."
        )

    async def execute(self, state: RunState, action: Action):
        pages, regions, error = _resolve_table_target(state, action)
        if error is not None:
            return self.error(action, error)
        assert pages is not None

        try:
            results = await self.parse_tables_async(state, pages, regions, action)
        except Exception as exc:
            return self.error(action, str(exc))

        source_counts: dict[str, int] = {}
        for result in results:
            source = str(result.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1
        return self._success_observation(
            action,
            pages=pages,
            results=results,
            source_counts=source_counts,
        )

    def parse_tables(
        self,
        state: RunState,
        pages: list[Any],
        regions: list[Region] | None,
        action: Action,
    ) -> list[dict[str, Any]]:
        raise RuntimeError("VLMTableTool.parse_tables is async-only; call execute")

    async def parse_tables_async(
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
                table = await self._parse_candidate(state, page, candidate)
                if table is not None:
                    results.append(table)
        return results

    async def _parse_candidate(
        self,
        state: RunState,
        page: PageState,
        candidate: _Candidate,
    ) -> dict[str, Any] | None:
        payload, response = await self.client.complete_json(
            system_prompt=_table_system_prompt(),
            user_payload=plannerize_page_refs(
                {
                    "task": state.task.to_dict(),
                    "page_index": page.page_index,
                    "region_id": candidate.region_id,
                    "request_kind": "table_to_html",
                }
            ),
            image_path=_candidate_image_input(page, candidate),
        )
        html = _extract_html(payload)
        if not html:
            return None
        confidence = _extract_confidence(payload)
        if candidate.region is not None:
            candidate.region.text = html
            candidate.region.confidence = confidence
        return {
            "page_index": page.page_index,
            "region_id": candidate.region_id,
            "text": html,
            "source": "vlm_table",
            "confidence": confidence,
            "usage": response.usage,
        }

    def _success_observation(
        self,
        action: Action,
        *,
        pages: list[PageState],
        results: list[dict[str, Any]],
        source_counts: dict[str, int],
    ):
        from docclaw.agent.utils import Observation

        total_tables = len(results)
        message = (
            f"Table parsing ready for {len(pages)} page(s); "
            f"{total_tables} table(s) available."
        )
        return Observation(
            action_id=action.action_id,
            success=True,
            data={"results": results, "sources": source_counts},
            message=message,
        )


def _table_system_prompt() -> str:
    return (
        "Perform OCR on the provided document page or crop and extract the visible table in this crop.\n"
        "Return exactly one JSON object with this shape:\n"
        "{\n"
        '  "html": "<table>...</table>",\n'
        '  "confidence": 0.0\n'
        "}\n"
        "Rules:\n"
        "- html must be valid HTML for one table only and should be the OCR result for the visible table content in this crop.\n"
        "- Preserve cell text faithfully.\n"
        "- Use <thead>, <tbody>, <tr>, <th>, and <td> when appropriate.\n"
        "- Use colspan and rowspan when clearly needed.\n"
        "- Do not complete truncated rows, columns, or cells that are not visible.\n"
        "- Do not summarize or restructure the table beyond what is needed for valid HTML.\n"
        "- Do not include markdown fences or explanation text.\n"
        "- If no clear table is visible, return {\"html\": \"\", \"confidence\": 0.0}."
    )


def _extract_html(payload: dict[str, Any]) -> str:
    for key in ("html", "text", "table_html"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_confidence(payload: dict[str, Any]) -> float | None:
    value = payload.get("confidence")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
