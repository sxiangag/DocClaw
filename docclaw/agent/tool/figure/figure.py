"""Abstract figure understanding tool."""

from __future__ import annotations

from abc import abstractmethod
import hashlib
from typing import Any

from docclaw.agent.tool.tool import Tool
from docclaw.agent.utils import (
    Action,
    ActionType,
    FigureInsight,
    Observation,
    RunState,
    page_id_from_index,
    page_number_from_index,
)

class FigureTool(Tool):
    """Base class for tools that interpret non-text visual regions."""

    @property
    def action_type(self) -> ActionType:
        return "understand_figures"

    @property
    def description(self) -> str:
        return (
            "Understand page-level figures for a specific question. This tool only "
            "accepts target.page_ids and requires parameters.question plus "
            "parameters.mode. Use mode='inspection' for page-level visual "
            "understanding of selected candidate pages when local figure, chart, "
            "photo, screenshot, map, or other image semantics are needed to answer "
            "the question. Use mode='enumeration' for page-level logical-item "
            "counting, where the question should ask what this page contributes to "
            "the final count from the full page. Ask for the exact page-level "
            "judgment needed by the task, not a vague proxy. For counting tasks, "
            "ask what logical charts, figures, or items this page contributes. "
            "Results are stored as question-conditioned figure_insights for the "
            "current run."
        )

    @property
    def target_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "Select page ids to interpret at full-page scope.",
            "properties": {
                "page_ids": {
                    "type": "array",
                    "description": "Page ids to interpret at full-page scope.",
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
            "description": "Figure-understanding controls.",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Required page-level question that figure understanding must "
                        "answer for each page. Make it precise and task-specific; ask "
                        "for the exact judgment needed on that page rather than a vague "
                        "proxy. Because this tool is page-based, ask what this page "
                        "contributes to the task in terms of logical charts, figures, "
                        "or items. For example, for counting tasks, ask how many "
                        "matching logical charts, figures, or items this page "
                        "contributes."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["inspection", "enumeration"],
                    "description": (
                        "Required figure understanding mode. Use 'inspection' for "
                        "focus-area visual inspection and 'enumeration' for page-level "
                        "logical-item counting."
                    ),
                },
            },
            "required": ["question", "mode"],
            "additionalProperties": False,
        }

    async def execute(self, state: RunState, action: Action) -> Observation:
        targets, error = _resolve_figure_targets(state, action)
        if error is not None:
            return self.error(action, error)
        assert targets is not None

        results: list[dict[str, Any]] = []
        for target in targets:
            try:
                result = self.understand_figures(state, target, action)
            except Exception as exc:
                return self.error(action, str(exc))
            results.append(
                {
                    **target,
                    **result,
                }
            )

        if len(results) == 1:
            message = _single_observation_message(results[0], document=state.document)
        else:
            message = _batch_observation_message(results, document=state.document)

        return Observation(
            action_id=action.action_id,
            success=True,
            data={
                "results": results,
                "source": "tool",
            },
            message=message,
        )

    def update_state(
        self,
        state: RunState,
        action: Action,
        observation: Observation,
    ) -> None:
        results = observation.data.get("results")
        if not isinstance(results, list):
            return
        for item in results:
            if not isinstance(item, dict):
                continue
            page_index = item.get("page_index")
            if not isinstance(page_index, int):
                continue
            question = _resolve_figure_insight_question(state, action, item)
            answer = item.get("answer")
            reason = item.get("reason")
            cleaned_answer = answer.strip() if isinstance(answer, str) and answer.strip() else None
            cleaned_reason = reason.strip() if isinstance(reason, str) and reason.strip() else None
            if cleaned_answer is None and cleaned_reason is None:
                continue
            artifact_path = item.get("artifact_path")
            cleaned_artifact_path = (
                artifact_path.strip()
                if isinstance(artifact_path, str) and artifact_path.strip()
                else None
            )
            state.add_figure_insight(
                FigureInsight(
                    insight_key=_figure_insight_key(page_index=page_index, question=question),
                    page_index=page_index,
                    question=question,
                    answer=cleaned_answer,
                    reason=cleaned_reason,
                    artifact_path=cleaned_artifact_path,
                )
            )

    @abstractmethod
    def understand_figures(
        self,
        state: RunState,
        target: dict[str, Any],
        action: Action,
    ) -> dict[str, Any]:
        """Return structured figure understanding for one selected page."""


def _resolve_figure_targets(
    state: RunState,
    action: Action,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    question = action.parameters.get("question")
    if not isinstance(question, str) or not question.strip():
        return None, "understand_figures requires parameters.question"
    mode = action.parameters.get("mode")
    if mode not in {"inspection", "enumeration"}:
        return None, "understand_figures requires parameters.mode to be one of {'inspection', 'enumeration'}"
    return _resolve_page_targets(state, action)


def _resolve_page_targets(
    state: RunState,
    action: Action,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    raw_page_indices = action.target.get("page_indices")
    if not isinstance(raw_page_indices, list) or not raw_page_indices:
        return None, "understand_figures requires target.page_ids"

    targets: list[dict[str, Any]] = []
    seen_page_indices: set[int] = set()
    for raw_page_index in raw_page_indices:
        if not isinstance(raw_page_index, int):
            return None, f"invalid page_id target: {raw_page_index!r}"
        if raw_page_index in seen_page_indices:
            continue
        page = state.get_page(raw_page_index)
        if page is None:
            return None, f"unknown page_id target: {page_number_from_index(raw_page_index)}"
        if not page.image_path:
            return None, f"page {page.page_number} has no image_path"
        targets.append(
            {
                "page_index": page.page_index,
                "artifact_path": page.image_path,
                "generated_artifact": False,
            }
        )
        seen_page_indices.add(raw_page_index)

    if not targets:
        return None, "understand_figures found no eligible target pages"
    return targets, None


def _resolve_figure_insight_question(
    state: RunState,
    action: Action,
    item: dict[str, Any],
) -> str:
    raw_question = item.get("question")
    if isinstance(raw_question, str) and raw_question.strip():
        return raw_question.strip()
    action_question = action.parameters.get("question")
    if isinstance(action_question, str) and action_question.strip():
        return action_question.strip()
    return state.task.prompt.strip()


def _figure_insight_key(*, page_index: int, question: str) -> str:
    digest = hashlib.sha1(f"{page_index}\0{question}".encode("utf-8")).hexdigest()[:16]
    return f"fig_{digest}"


def _single_observation_message(data: dict[str, Any], *, document: Any | None = None) -> str:
    answer = data.get("answer")
    target_name = _target_name(data, document=document)
    message = f"Figure understanding ready for {target_name}"
    if isinstance(answer, str) and answer.strip():
        message += f": {answer.strip()}"
    else:
        message += "."
    return message


def _batch_observation_message(results: list[dict[str, Any]], *, document: Any | None = None) -> str:
    page_indexes = sorted(
        {
            page_index
            for item in results
            if isinstance((page_index := item.get("page_index")), int)
        }
    )
    page_ids = [page_id_from_index(page_index, document=document) for page_index in page_indexes]
    page_label = ", ".join(page_ids)
    return (
        f"Figure understanding ready for {len(results)} page(s) across "
        f"{len(page_indexes)} page(s) ({page_label})."
    )


def _target_name(target: dict[str, Any], *, document: Any | None = None) -> str:
    page_index = target.get("page_index")
    if isinstance(page_index, int):
        return f"page_id {page_id_from_index(page_index, document=document)}"
    return "target"
