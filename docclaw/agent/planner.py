"""Planner interfaces for selecting DocClaw actions."""

from __future__ import annotations

from collections.abc import Iterable
import json
import re
from typing import Any
from typing import Protocol
from typing import TypeGuard

from docclaw.agent.debug import dump_jsonl_from_env
from docclaw.agent.tool.tool import ToolRegistry, build_default_tool_registry
from docclaw.agent.utils import (
    ACTION_TYPES,
    Action,
    ActionType,
    ActiveSkill,
    RunState,
    page_id_from_index,
)
from docclaw.provider.base import LLMProvider, LLMResponse, ToolCallRequest
from docclaw.skills import TaskSkillInfo, TaskSkillsLoader


class Planner(Protocol):
    """Select the next action for the current run state."""

    async def next_action(self, state: RunState) -> Action | None:
        """Return the next action, or None when no further action is available."""
        ...


class ScriptedPlanner:
    """Planner that returns a fixed action sequence.

    This is mainly useful for tests and early demos before an LLM planner exists.
    """

    def __init__(
        self,
        actions: Iterable[Action],
        *,
        stop_when_exhausted: bool = True,
        reason: str = "script exhausted",
    ) -> None:
        self._actions = list(actions)
        self._index = 0
        self.stop_when_exhausted = stop_when_exhausted
        self.reason = reason

    async def next_action(self, state: RunState) -> Action | None:
        if self._index < len(self._actions):
            action = self._actions[self._index]
            self._index += 1
            return action
        if self.stop_when_exhausted:
            return Action(action_type="stop", parameters={"reason": self.reason})
        return None


class LLMPlanner:
    """Planner that delegates next-action selection to a model."""

    _JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    _ACTIVE_SKILL_SELECTION_ATTEMPTED_KEY = "active_skill_selection_attempted"
    _MAX_FORMAT_RETRIES = 1

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str | None = None,
        tools: ToolRegistry | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        task_skills_loader: TaskSkillsLoader | None = None,
    ) -> None:
        self.provider = provider
        self.model = model or provider.get_default_model()
        self.tools = tools or build_default_tool_registry()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.task_skills_loader = task_skills_loader

    async def next_action(self, state: RunState) -> Action | None:
        await self._ensure_active_skill(state)
        messages = self._build_messages(state)
        function_tools = self.tools.function_definitions()

        invalid_output_error: str | None = None
        for attempt_index in range(self._MAX_FORMAT_RETRIES + 1):
            try:
                response = await self.provider.chat(
                    messages,
                    tools=function_tools,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    tool_choice="required",
                )
            except Exception as exc:
                return self._fallback_stop(
                    f"planner error: {type(exc).__name__}: {exc}",
                )
            if response.error:
                return self._fallback_stop(
                    f"planner error: {response.error}",
                )

            action, invalid_output_error = self._action_from_response(state, response)
            if action is not None:
                return action
            if attempt_index >= self._MAX_FORMAT_RETRIES:
                break

        assert invalid_output_error is not None
        return self._fallback_stop(
            f"invalid planner output after {self._MAX_FORMAT_RETRIES} retry: {invalid_output_error}",
        )

    def _build_messages(self, state: RunState) -> list[dict[str, str]]:
        document_memory = self.tools.build_document_overview(state)
        document_state = state.build_planner_context(
            document_memory=document_memory,
        )
        available_document_processing_actions = self.tools.definitions()
        system_prompt = (
            "You are the **DocClaw planner**, your job is to choose exactly one next document action for the current query.\n"
            "### Inputs\n"
            "You may use the following information:\n"
            "- **Active document skill**: task-oriented workflow guidance that specifies the interaction strategy and tool-use pattern for the current query.\n"
            "- **Document state**: a summary of the current document state, including both document memory and task memory.\n"
            "- **Available document processing actions**: executable actions and their tool descriptions.\n"
            "### Document State\n"
            "Interpret the document state as the agent's current memory. Use it to determine what is already known, what remains uncertain, and which document processing action should be taken next.\n"
            "Use **document memory** for document knowledge, and use **task memory** to track interaction context.\n"
            "Do not assume that information absent from the condensed document state is already available. If required information is missing, select an action that acquires or refines it. If sufficient information is already available, select an appropriate output action.\n"
            "### Output Contract\n"
            "- Produce exactly **one tool call** corresponding to one document processing action.\n"
            "- Select only from the available document processing actions and produce no extra prose outside the tool call.\n"
            "### Rules\n"
            "- Follow the active document skill as the workflow policy for the current query.\n"
            "- Select the next action that is best supported by the document state, active document skill, and available tool descriptions.\n"
            "- Target pages only with page ids exposed in the document state.\n"
            "- Do not assume that page ids are sequential page numbers. Page ids may be opaque and may differ from the document's actual page numbers.\n"
            "- Use region ids only when they appear in recent successful action outputs or in the document state.\n"
            "- If multiple known pages or regions need the same operation, prefer one batched action over many single-target actions.\n"
        )
        active_skill_prompt = self._active_skill_prompt(state)
        if active_skill_prompt is not None:
            system_prompt += (
                "\n\n### Active Document Skill\n"
                + active_skill_prompt
            )
        user_prompt = json.dumps(
            {
                "document_state": document_state,
                "available_document_processing_actions": available_document_processing_actions,
            },
            ensure_ascii=False,
            indent=2,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        messages.append({"role": "user", "content": user_prompt})
        _debug_dump_planner_round(
            state=state,
            messages=messages,
        )
        return messages

    async def _ensure_active_skill(self, state: RunState) -> None:
        if self.task_skills_loader is None:
            return
        if state.get_active_skill() is not None:
            return
        if bool(state.metadata.get(self._ACTIVE_SKILL_SELECTION_ATTEMPTED_KEY)):
            return

        state.metadata[self._ACTIVE_SKILL_SELECTION_ATTEMPTED_KEY] = True
        available_skills = self.task_skills_loader.list_skills()
        if not available_skills:
            return

        selected = await self._select_skill(state, available_skills)
        if selected is None:
            return
        state.set_active_skill(
            selected.name,
            reason=selected.reason,
            source=selected.source,
            path=selected.path,
        )
        state.add_pending_event(
            "skill_selected",
            {
                "skill": selected.to_dict(),
            },
        )

    async def _select_skill(
        self,
        state: RunState,
        available_skills: list[TaskSkillInfo],
    ) -> ActiveSkill | None:
        summary = self.task_skills_loader.build_skills_summary() if self.task_skills_loader else ""
        messages = self._build_skill_selection_messages(state, summary)
        try:
            response = await self.provider.chat(
                messages,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
        except Exception:
            return None
        if response.error:
            return None

        payload = self._parse_skill_selection_response(response.content)
        if payload is None:
            return None

        selected_name = payload.get("skill_name")
        if not isinstance(selected_name, str):
            return None
        normalized = selected_name.strip()
        if not normalized or normalized.lower() in {"none", "null"}:
            return None

        available_by_name = {skill.name: skill for skill in available_skills}
        selected = available_by_name.get(normalized)
        if selected is None:
            return None

        reason = payload.get("reason")
        return ActiveSkill(
            name=selected.name,
            reason=reason if isinstance(reason, str) and reason.strip() else None,
            source=selected.source,
            path=selected.path,
        )

    def _build_skill_selection_messages(
        self,
        state: RunState,
        skills_summary: str,
    ) -> list[dict[str, str]]:
        system_prompt = (
            "You are the **DocClaw task-oriented skill selector**.\n"
            "### Objective\n"
            "Given the user query and current document state summary, choose at most **one document skill** that best matches the query objective.\n"
            "A document skill is a task-oriented workflow policy. It defines an interaction strategy and tool-use pattern for a family of visual document processing tasks. It is not itself a document processing action.\n"
            "### Inputs\n"
            "You may use the following information:\n"
            "- **Query**: the user's question or instruction.\n"
            "- **Document state summary**: a summary of the current document state.\n"
            "- **Available document skills**: candidate skill names and descriptions.\n"
            "### Output Contract\n"
            "Return only a JSON object with the following schema:\n"
            "```json\n"
            "{\n"
            '  "skill_name": "<one available document skill name or null>",\n'
            '  "reason": "short reason"\n'
            "}\n"
            "```\n"
            "### Selection Rules\n"
            "- If none of the listed document skills is clearly appropriate, return `null` for `skill_name`.\n"
            "- Do not return Markdown fences or any extra prose."
        )
        user_prompt = json.dumps(
            {
                "query": state.task.prompt,
                "document_state_summary": state.document.summary(),
                "available_document_skills": skills_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @classmethod
    def _parse_skill_selection_response(cls, content: str | None) -> dict[str, Any] | None:
        if content is None or not content.strip():
            return None
        try:
            payload = cls._load_json_object(content)
        except ValueError:
            return None
        return payload if isinstance(payload, dict) else None

    def _active_skill_prompt(self, state: RunState) -> str | None:
        if self.task_skills_loader is None:
            return None
        active_skill = state.get_active_skill()
        if active_skill is None:
            return None
        body = self.task_skills_loader.load_skill_body(active_skill.name)
        if body is None or not body.strip():
            return None
        reason = f"Selected because: {active_skill.reason}\n\n" if active_skill.reason else ""
        return reason + body

    @classmethod
    def _parse_response(cls, content: str | None) -> tuple[Action | None, str | None]:
        if content is None or not content.strip():
            return None, "empty planner response"

        try:
            payload = cls._load_json_object(content)
        except ValueError as exc:
            return None, str(exc)

        if "action" in payload:
            nested = payload["action"]
            if not isinstance(nested, dict):
                return None, "planner action field must be an object"
            payload = nested

        action_type = payload.get("action_type")
        if not isinstance(action_type, str):
            return None, "planner action_type must be a string"
        if not _is_action_type(action_type):
            return None, f"unsupported action_type: {action_type}"

        target = payload.get("target", {})
        if target is None:
            target = {}
        if not isinstance(target, dict):
            return None, "planner target must be an object"
        target = _normalize_planner_mapping(target)
        target = _normalize_target_for_action(action_type, target)

        parameters = payload.get("parameters", {})
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            return None, "planner parameters must be an object"
        parameters = _normalize_planner_mapping(parameters)

        rationale = payload.get("rationale")
        if rationale is not None and not isinstance(rationale, str):
            rationale = str(rationale)

        try:
            return Action(
                action_type=action_type,
                target=target,
                parameters=parameters,
                rationale=rationale,
            ), None
        except ValueError as exc:
            return None, str(exc)

    @classmethod
    def _load_json_object(cls, content: str) -> dict[str, Any]:
        candidate = content.strip()
        match = cls._JSON_BLOCK_RE.search(candidate)
        if match:
            candidate = match.group(1).strip()

        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start < 0 or end < start:
                raise ValueError("planner response must contain one JSON object")
            try:
                payload = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError(f"planner returned invalid JSON: {exc.msg}") from exc

        if not isinstance(payload, dict):
            raise ValueError("planner response JSON must be an object")
        return payload

    def _action_from_tool_call(
        self,
        tool_call: ToolCallRequest,
    ) -> tuple[Action | None, str | None]:
        if not _is_action_type(tool_call.name):
            return None, f"planner tool call references unknown tool: {tool_call.name}"
        if self.tools.get(tool_call.name) is None:
            return None, f"planner tool call references unknown tool: {tool_call.name}"

        arguments = tool_call.arguments
        if not isinstance(arguments, dict):
            return None, "planner tool call arguments must be an object"

        target = arguments.get("target", {})
        if target is None:
            target = {}
        if not isinstance(target, dict):
            return None, "planner tool call target must be an object"
        target = _normalize_planner_mapping(target)
        target = _normalize_target_for_action(tool_call.name, target)

        parameters = arguments.get("parameters", {})
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            return None, "planner tool call parameters must be an object"
        parameters = _normalize_planner_mapping(parameters)

        rationale = arguments.get("rationale")
        if rationale is not None and not isinstance(rationale, str):
            rationale = str(rationale)

        try:
            return Action(
                action_type=tool_call.name,
                target=target,
                parameters=parameters,
                rationale=rationale,
            ), None
        except ValueError as exc:
            return None, str(exc)

    def _action_from_response(
        self,
        state: RunState,
        response: LLMResponse,
    ) -> tuple[Action | None, str | None]:
        if response.tool_calls:
            _debug_dump_planner_command(
                state=state,
                mode="tool_call",
                response=response,
                content=response.content,
            )
            if len(response.tool_calls) != 1:
                return None, f"expected 1 tool call, got {len(response.tool_calls)}"
            return self._action_from_tool_call(response.tool_calls[0])

        _debug_dump_planner_command(
            state=state,
            mode="json_response",
            response=response,
            content=response.content,
        )
        return self._parse_response(response.content)

    @staticmethod
    def _fallback_stop(reason: str) -> Action:
        return Action(
            action_type="stop",
            parameters={"reason": reason},
            rationale="Planner fallback due to invalid or failed model output.",
        )


def _is_action_type(value: str) -> TypeGuard[ActionType]:
    return value in ACTION_TYPES


def _normalize_planner_mapping(data: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in data.items():
        cleaned = _normalize_planner_value(value)
        if cleaned is not None:
            normalized[key] = cleaned
    return normalized


def _normalize_planner_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, list):
        cleaned_items = [
            cleaned
            for item in value
            if (cleaned := _normalize_planner_value(item)) is not None
        ]
        return cleaned_items or None
    if isinstance(value, dict):
        return _normalize_planner_mapping(value)
    return value


def _normalize_target_for_action(
    action_type: str,
    target: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(target)
    mode = normalized.get("mode")
    if not isinstance(mode, str):
        return normalized

    if action_type not in {"ocr", "parse_table", "parse_formula", "parse_chart"}:
        return normalized

    if mode == "page":
        normalized.pop("region_ids", None)
        normalized.pop("zoom_region_ids", None)
        normalized.pop("crop_region_ids", None)
        return normalized
    if mode == "region":
        normalized.pop("zoom_region_ids", None)
        normalized.pop("crop_region_ids", None)
        if action_type != "ocr":
            normalized.pop("page_ids", None)
            return normalized
        region_ids = normalized.get("region_ids")
        if isinstance(region_ids, list) and region_ids:
            normalized.pop("page_ids", None)
        return normalized
    if mode == "zoom_region":
        normalized.pop("page_ids", None)
        normalized.pop("region_ids", None)
        normalized.pop("crop_region_ids", None)
        normalized.pop("rotate_region_ids", None)
        return normalized
    if mode == "crop_region":
        normalized.pop("page_ids", None)
        normalized.pop("region_ids", None)
        normalized.pop("zoom_region_ids", None)
        normalized.pop("rotate_region_ids", None)
        return normalized
    if mode == "rotate_region":
        normalized.pop("page_ids", None)
        normalized.pop("region_ids", None)
        normalized.pop("zoom_region_ids", None)
        normalized.pop("crop_region_ids", None)
        return normalized
    return normalized

def _debug_dump_planner_round(
    *,
    state: RunState,
    messages: list[dict[str, str]],
) -> None:
    dump_jsonl_from_env(
        "DOCCLAW_PLANNER_DEBUG_PATH",
        {
            "kind": "planner_input",
            "document_id": state.document.document_id,
            "run_id": state.run_id,
            "active_skill": (
                state.get_active_skill().to_dict()
                if state.get_active_skill() is not None
                else None
            ),
            "action_trace_length": len(state.action_trace),
            "messages": messages,
            "page_debug": _planner_page_debug(state),
        },
    )


def _planner_page_debug(state: RunState) -> dict[str, Any]:
    try:
        return {
            "document_page_indices": [page.page_index for page in state.document.pages],
            "document_page_ids": [
                page_id_from_index(page.page_index, document=state.document)
                for page in state.document.pages
            ],
        }
    except Exception:
        return {}


def _debug_dump_planner_command(
    *,
    state: RunState,
    mode: str,
    response: LLMResponse,
    content: str | None,
) -> None:
    dump_jsonl_from_env(
        "DOCCLAW_PLANNER_DEBUG_PATH",
        {
            "kind": "planner_output",
            "document_id": state.document.document_id,
            "run_id": state.run_id,
            "active_skill": (
                state.get_active_skill().to_dict()
                if state.get_active_skill() is not None
                else None
            ),
            "action_trace_length": len(state.action_trace),
            "mode": mode,
            "content": content,
            "finish_reason": response.finish_reason,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                }
                for tool_call in response.tool_calls
            ],
            "usage": response.usage,
        },
    )
