"""Abstract page-selection tool."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

from docclaw.agent.tool.tool import Tool
from docclaw.agent.utils import (
    Action,
    ActionType,
    Observation,
    RunState,
    page_id_from_index,
)


class SelectPagesTool(Tool):
    """Select visually relevant candidate pages for the current question."""

    @property
    def action_type(self) -> ActionType:
        return "select_pages"

    @property
    def description(self) -> str:
        return (
            "Select relevant candidate pages for a specific question from page-level "
            "content. This tool only accepts target.page_ids and requires "
            "parameters.question plus parameters.mode. The question should state the "
            "page-selection objective for this selection operation. This includes "
            "finding the page with a specific page number, finding the pages that belong to a "
            "section, locating the page with a referenced chart, "
            "table, figure, or map, identifying pages that visually match "
            "a described object, layout, or scene, selecting the pages that "
            "should be kept for the next operation step, and etc. "
            "Use mode='coarse' for a high-recall first-pass page filter from images only. "
            "Use mode='refine' for a higher-precision second-pass filter over already "
            "narrowed pages, using page images plus best-available page text."
        )

    @property
    def target_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "Candidate page ids to filter at full-page scope.",
            "properties": {
                "page_ids": {
                    "type": "array",
                    "description": "Candidate page ids to filter.",
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
            "description": "Page-selection controls.",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Required page-selection question for this step. State the "
                        "selection objective that should be judged from the "
                        "candidate pages, not merely a verbatim restatement "
                        "of the original question. This may be a page-reference "
                        "resolution question, a question about which pages belong to "
                        "a named section, having specific table, chart, figure, or map, or "
                        "a question about which pages should be kept for the next "
                        "reasoning step, and etc."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["coarse", "refine"],
                    "description": (
                        "Required selection mode. Use 'coarse' for high-recall visual "
                        "page filtering before OCR/localization. Use 'refine' for a "
                        "higher-precision second-pass filter over already narrowed "
                        "pages after page text is available."
                    ),
                },
            },
            "required": ["question", "mode"],
            "additionalProperties": False,
        }

    async def execute(self, state: RunState, action: Action) -> Observation:
        targets, error = _resolve_select_page_targets(state, action)
        if error is not None:
            return self.error(action, error)
        assert targets is not None
        result = await self.select_pages(state, action, targets=targets)
        if not result["success"]:
            return self.error(action, str(result["error"]))
        payload = result["payload"]
        assert isinstance(payload, dict)
        selected_page_ids = payload.get("selected_page_ids")
        return Observation(
            action_id=action.action_id,
            success=True,
            data={
                "results": [payload],
                "source": payload.get("source", "unknown"),
            },
            message=selection_observation_message(
                selected_page_ids if isinstance(selected_page_ids, list) else [],
                count=len(targets),
            ),
            artifacts=result.get("artifacts") or [],
        )

    @abstractmethod
    async def select_pages(
        self,
        state: RunState,
        action: Action,
        *,
        targets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return selected candidate pages for one cross-page filtering step."""


def _resolve_select_page_targets(
    state: RunState,
    action: Action,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    question = action.parameters.get("question")
    if not isinstance(question, str) or not question.strip():
        return None, "select_pages requires parameters.question"
    mode = action.parameters.get("mode")
    if mode not in {"coarse", "refine"}:
        return None, "select_pages requires parameters.mode to be one of {'coarse', 'refine'}"

    raw_page_indices = action.target.get("page_indices")
    if not isinstance(raw_page_indices, list) or not raw_page_indices:
        return None, "select_pages requires target.page_ids"

    targets: list[dict[str, Any]] = []
    seen_page_indices: set[int] = set()
    for raw_page_index in raw_page_indices:
        if not isinstance(raw_page_index, int):
            return None, f"invalid page_id target: {raw_page_index!r}"
        if raw_page_index in seen_page_indices:
            continue
        page = state.get_page(raw_page_index)
        if page is None:
            return None, f"unknown page_id target: {page_id_from_index(raw_page_index, document=state.document)}"
        if not page.image_path:
            return None, f"page_id {page_id_from_index(page.page_index, document=state.document)} has no image_path"
        image_path = Path(page.image_path).expanduser()
        if not image_path.exists():
            return None, f"page image_path does not exist: {image_path}"
        targets.append(
            {
                "page_index": page.page_index,
                "artifact_path": str(image_path),
                "generated_artifact": False,
            }
        )
        seen_page_indices.add(raw_page_index)

    if not targets:
        return None, "select_pages found no eligible target pages"
    return targets, None


def selection_observation_message(selected_page_ids: list[str], *, count: int) -> str:
    if not selected_page_ids:
        return f"Page selection returned no matching pages from {count} candidate page(s)."
    if len(selected_page_ids) == 1:
        return f"Page selection chose {selected_page_ids[0]} from {count} candidate page(s)."
    page_label = ", ".join(selected_page_ids)
    return f"Page selection chose {page_label} from {count} candidate page(s)."
