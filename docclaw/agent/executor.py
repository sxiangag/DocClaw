"""Executor for DocClaw document actions."""

from __future__ import annotations

from typing import Any

from docclaw.agent.tool.tool import ToolRegistry, build_default_tool_registry
from docclaw.agent.utils import (
    Action,
    Observation,
    RunState,
    page_index_from_id,
)


class Executor:
    """Dispatch planner actions to registered tools."""

    def __init__(self, tools: ToolRegistry | None = None) -> None:
        self.tools = tools or build_default_tool_registry()

    async def execute(self, state: RunState, action: Action) -> Observation:
        action.target = _planner_target_to_internal_target(
            action.target,
            state=state,
        )
        tool = self.tools.get(action.action_type)
        if tool is None:
            return Observation(
                action_id=action.action_id,
                success=False,
                error=f"no tool registered for action_type: {action.action_type}",
            )
        if not tool.can_execute(action):
            return Observation(
                action_id=action.action_id,
                success=False,
                error=(
                    f"tool {tool.name} cannot execute action_type: "
                    f"{action.action_type}"
                ),
            )
        stop_guard_error = _guard_docqa_stop_answer(state, action)
        if stop_guard_error is not None:
            return Observation(
                action_id=action.action_id,
                success=False,
                error=stop_guard_error,
            )
        answer_guard_error = _guard_docqa_answer_from_evidence(state, action)
        if answer_guard_error is not None:
            return Observation(
                action_id=action.action_id,
                success=False,
                error=answer_guard_error,
            )
        recheck_guard_error = _guard_docqa_not_answerable_recheck(state, action)
        if recheck_guard_error is not None:
            return Observation(
                action_id=action.action_id,
                success=False,
                error=recheck_guard_error,
            )
        coarse_recall_guard_error = _guard_docqa_lookup_coarse_recall(state, action)
        if coarse_recall_guard_error is not None:
            return Observation(
                action_id=action.action_id,
                success=False,
                error=coarse_recall_guard_error,
            )
        try:
            return await tool.execute(state, action)
        except Exception as exc:
            return Observation(
                action_id=action.action_id,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )


def _planner_target_to_internal_target(
    target: dict[str, Any],
    *,
    state: RunState,
) -> dict[str, Any]:
    normalized = dict(target)

    raw_page_ids = normalized.pop("page_ids", None)
    if raw_page_ids is not None:
        normalized["page_indices"] = _page_ids_to_indices(
            raw_page_ids,
            state=state,
        )

    raw_page_id = normalized.pop("page_id", None)
    if raw_page_id is not None:
        try:
            normalized["page_index"] = page_index_from_id(
                str(raw_page_id),
                document=state.document,
            )
        except (TypeError, ValueError):
            normalized["page_index"] = raw_page_id

    return normalized


def _page_ids_to_indices(value: Any, *, state: RunState) -> Any:
    if not isinstance(value, list):
        return value
    page_indices: list[int] = []
    for item in value:
        try:
            page_indices.append(
                page_index_from_id(str(item), document=state.document)
            )
        except (TypeError, ValueError):
            page_indices.append(item)  # type: ignore[arg-type]
    return page_indices


def _guard_docqa_stop_answer(state: RunState, action: Action) -> str | None:
    if action.action_type != "stop":
        return None
    active_skill = state.get_active_skill()
    if active_skill is None or not _is_docqa_answer_guard_skill(active_skill.name):
        return None
    answer = action.parameters.get("answer")
    if not isinstance(answer, str):
        return None
    if not answer.strip():
        return None
    if answer.strip() == "Not answerable":
        if not state.action_trace:
            return (
                'docqa workflow violation: before stop with final answer "Not answerable", '
                "run extract_evidence to check the current focused state. Only if that "
                'extract_evidence result is still answerability_status="inconclusive" may '
                'you call stop with final answer "Not answerable".'
            )
        last_step = state.action_trace[-1]
        last_status = (
            (last_step.observation.data or {}).get("answerability_status")
            if last_step.observation.success
            else None
        )
        last_mode = _action_mode(last_step.action)
        if (
            last_step.action.action_type == "extract_evidence"
            and last_step.observation.success
            and last_status == "inconclusive"
            and last_mode == "not_answerable_recheck"
        ):
            return None
        if (
            last_step.action.action_type == "extract_evidence"
            and last_step.observation.success
            and last_status == "inconclusive"
        ):
            return (
                'docqa workflow violation: before stop with final answer "Not answerable", '
                "run extract_evidence with parameters.mode='not_answerable_recheck' on a "
                "planner-selected page_ids candidate set. Use this recheck only once, "
                "immediately before concluding 'Not answerable'."
            )
        return (
            'docqa workflow violation: before stop with final answer "Not answerable", '
            "run extract_evidence to check the current focused state. Only if that "
            'extract_evidence result is still answerability_status="inconclusive" may '
            'you call stop with final answer "Not answerable".'
        )
    return (
        'docqa workflow violation: use stop only for the final answer "Not answerable". '
        "For any other answer, first run extract_evidence; once the latest evidence "
        'assessment is answerable, use answer_from_evidence.'
    )


def _guard_docqa_answer_from_evidence(state: RunState, action: Action) -> str | None:
    if action.action_type != "answer_from_evidence":
        return None
    active_skill = state.get_active_skill()
    if active_skill is None or not _is_docqa_answer_guard_skill(active_skill.name):
        return None
    if not state.action_trace:
        return (
            "docqa workflow violation: before answer_from_evidence, run "
            "extract_evidence. answer_from_evidence must be called immediately "
            "after a successful extract_evidence step."
        )
    last_step = state.action_trace[-1]
    if (
        last_step.action.action_type == "extract_evidence"
        and last_step.observation.success
    ):
        return None
    return (
        "docqa workflow violation: before answer_from_evidence, run "
        "extract_evidence. answer_from_evidence must be called immediately "
        "after a successful extract_evidence step."
    )


def _guard_docqa_not_answerable_recheck(state: RunState, action: Action) -> str | None:
    if action.action_type != "extract_evidence":
        return None
    active_skill = state.get_active_skill()
    if active_skill is None or not _is_docqa_answer_guard_skill(active_skill.name):
        return None
    if _action_mode(action) != "not_answerable_recheck":
        return None
    target = action.target if isinstance(action.target, dict) else {}
    if not isinstance(target.get("page_indices"), list) or not target.get("page_indices"):
        return (
            "docqa workflow violation: extract_evidence with parameters.mode='not_answerable_recheck' "
            "must target a non-empty page_ids candidate set. Pass pages that are "
            "likely to contain the missing answer."
        )
    return None


def _guard_docqa_lookup_coarse_recall(state: RunState, action: Action) -> str | None:
    active_skill = state.get_active_skill()
    if active_skill is None or active_skill.name != "docqa-inspection":
        return None

    has_coarse_selection = any(
        step.observation.success
        and step.action.action_type == "select_pages"
        and _action_mode(step.action) == "coarse"
        for step in state.action_trace
    )
    has_page_ocr = any(
        step.observation.success
        and step.action.action_type == "ocr"
        and _action_target_mode(step.action) == "page"
        for step in state.action_trace
    )
    has_refine_selection = any(
        step.observation.success
        and step.action.action_type == "select_pages"
        and _action_mode(step.action) == "refine"
        for step in state.action_trace
    )

    if has_coarse_selection and has_page_ocr and has_refine_selection:
        return None

    if action.action_type == "select_pages":
        return None
    if action.action_type == "ocr" and _action_target_mode(action) == "page":
        return None

    missing_steps: list[str] = []
    if not has_coarse_selection:
        missing_steps.append("select_pages with parameters.mode='coarse'")
    if not has_page_ocr:
        missing_steps.append("page-mode ocr")
    if not has_refine_selection:
        missing_steps.append("select_pages with parameters.mode='refine'")
    missing_text = " and ".join(missing_steps)
    return (
        "docqa-inspection workflow violation: before inspection, extraction, parsing, or stop, "
        "complete page selection by running "
        f"{missing_text} at least once."
    )


def _action_mode(action: Action) -> str | None:
    parameters = action.parameters if isinstance(action.parameters, dict) else {}
    mode = parameters.get("mode")
    if not isinstance(mode, str):
        return None
    normalized = mode.strip()
    return normalized or None


def _action_target_mode(action: Action) -> str | None:
    target = action.target if isinstance(action.target, dict) else {}
    mode = target.get("mode")
    if not isinstance(mode, str):
        return None
    normalized = mode.strip()
    return normalized or None


def _is_docqa_answer_guard_skill(name: str) -> bool:
    return name in {"docqa", "docqa-inspection", "docqa-enumeration"}
