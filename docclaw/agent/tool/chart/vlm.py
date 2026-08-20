"""General VLM-backed chart parsing implementation."""

from __future__ import annotations

from typing import Any

from docclaw.agent.tool.chart.chart import ChartTool, _resolve_chart_target
from docclaw.agent.tool.chart.paddleocrvl import (
    _Candidate,
    _cached_charts_for_page,
    _candidate_image_input,
    _chart_candidates_for_page,
    _uses_crop_regions,
    _uses_refined_regions,
    _uses_zoom_regions,
)
from docclaw.agent.tool.vlm_client import VLMClient
from docclaw.agent.utils import Action, PageState, Region, RunState, plannerize_page_refs
from docclaw.provider.base import LLMProvider


class VLMChartTool(ChartTool):
    """Parse charts with a general multimodal language model."""

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
            "Parse charts from known chart regions or page-level chart candidates "
            "with a general multimodal language model. This keeps DocClaw layout "
            "targeting and uses VLM recognition for chart content extraction."
        )

    async def execute(self, state: RunState, action: Action):
        pages, regions, error = _resolve_chart_target(state, action)
        if error is not None:
            return self.error(action, error)
        assert pages is not None

        try:
            results = await self.parse_charts_async(state, pages, regions, action)
        except Exception as exc:
            return self.error(action, str(exc))

        source_counts: dict[str, int] = {}
        for result in results:
            source = str(result.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1

        from docclaw.agent.utils import Observation

        total_charts = len(results)
        message = (
            f"Chart parsing ready for {len(pages)} page(s); "
            f"{total_charts} chart(s) available."
        )
        return Observation(
            action_id=action.action_id,
            success=True,
            data={"results": results, "sources": source_counts},
            message=message,
        )

    def parse_charts(
        self,
        state: RunState,
        pages: list[Any],
        regions: list[Region] | None,
        action: Action,
    ) -> list[dict[str, Any]]:
        raise RuntimeError("VLMChartTool.parse_charts is async-only; call execute")

    async def parse_charts_async(
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

            cached_charts = _cached_charts_for_page(page, regions=page_regions)
            if cached_charts is not None and not _uses_refined_regions(action):
                results.extend(cached_charts)
                continue

            candidates = _chart_candidates_for_page(
                state,
                page,
                explicit_regions=page_regions,
                use_zoom_artifacts=_uses_zoom_regions(action),
                use_crop_artifacts=_uses_crop_regions(action),
            )
            for candidate in candidates:
                chart = await self._parse_candidate(state, page, candidate)
                if chart is not None:
                    results.append(chart)
        return results

    async def _parse_candidate(
        self,
        state: RunState,
        page: PageState,
        candidate: _Candidate,
    ) -> dict[str, Any] | None:
        payload, response = await self.client.complete_json(
            system_prompt=_chart_system_prompt(),
            user_payload=plannerize_page_refs(
                {
                    "task": state.task.to_dict(),
                    "page_index": page.page_index,
                    "region_id": candidate.region_id,
                    "request_kind": "chart_to_text",
                }
            ),
            image_path=_candidate_image_input(page, candidate),
        )
        content = _extract_chart_text(payload)
        if not content:
            return None
        confidence = _extract_confidence(payload)
        if candidate.region is not None:
            candidate.region.text = content
            candidate.region.confidence = confidence
        return {
            "page_index": page.page_index,
            "region_id": candidate.region_id,
            "text": content,
            "source": "vlm_chart",
            "confidence": confidence,
            "usage": response.usage,
        }


def _chart_system_prompt() -> str:
    return (
        "Perform OCR on the provided chart image or crop and return only visible chart text and directly readable numeric content.\n"
        "Return exactly one JSON object with this shape:\n"
        "{\n"
        '  "text": "plain text OCR output with visible titles, axis labels, legend items, annotations, and directly readable values",\n'
        '  "confidence": 0.0\n'
        "}\n"
        "Rules:\n"
        "- text should be OCR output focused on visible chart labels, axes, legend items, annotations, and directly readable values.\n"
        "- Do not infer hidden series values or approximate unreadable values.\n"
        "- Do not invent hidden data points.\n"
        "- Do not output markdown fences or prose outside the JSON object.\n"
        "- If no clear chart is visible, return {\"text\": \"\", \"confidence\": 0.0}."
    )


def _extract_chart_text(payload: dict[str, Any]) -> str:
    for key in ("text", "summary", "content"):
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
