"""Tool abstractions, registry, and general actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from docclaw.agent.utils import (
    Action,
    ActionType,
    Observation,
    RunState,
    page_id_from_index,
    plannerize_page_refs,
)
from docclaw.provider.base import LLMProvider


###############################################################################
# Tool Contract and Registry
###############################################################################


class Tool(ABC):
    """Executable document operation."""

    @property
    @abstractmethod
    def action_type(self) -> ActionType:
        """The action type this tool executes."""
        ...

    @property
    def name(self) -> str:
        """Human-readable tool name."""
        return self.action_type

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description for planners and diagnostics."""
        ...

    @property
    def target_schema(self) -> dict[str, Any]:
        """JSON schema for the action target payload."""
        return {
            "type": "object",
            "description": "Action target such as page_index, region_id, or bbox.",
            "additionalProperties": True,
        }

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON schema for the action parameters payload."""
        return {
            "type": "object",
            "description": "Additional action parameters for this tool.",
            "additionalProperties": True,
        }

    def can_execute(self, action: Action) -> bool:
        return action.action_type == self.action_type

    async def __call__(self, state: RunState, action: Action) -> Observation:
        return await self.execute(state, action)

    @abstractmethod
    async def execute(self, state: RunState, action: Action) -> Observation:
        """Execute an action and return an observation."""
        ...

    def update_state(
        self,
        state: RunState,
        action: Action,
        observation: Observation,
    ) -> None:
        """Apply this tool's successful observation to run state."""

    def document_overview_fragment(self, state: RunState) -> dict[str, Any]:
        """Return an optional planner-facing document overview fragment."""
        return {}

    def error(self, action: Action, message: str) -> Observation:
        return Observation(action_id=action.action_id, success=False, error=message)

    def function_definition(self) -> dict[str, Any]:
        """Return an OpenAI-style function definition for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.action_type,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": self.target_schema,
                        "parameters": self.parameters_schema,
                        "rationale": {
                            "type": "string",
                            "description": "Short reason for choosing this action.",
                        },
                    },
                    "required": ["target", "parameters"],
                    "additionalProperties": False,
                },
            },
        }

class ToolRegistry:
    """Registry mapping action types to tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.action_type] = tool

    def unregister(self, action_type: str) -> None:
        self._tools.pop(action_type, None)

    def get(self, action_type: str) -> Tool | None:
        return self._tools.get(action_type)

    def require(self, action_type: str) -> Tool:
        tool = self.get(action_type)
        if tool is None:
            raise KeyError(f"no tool registered for action_type: {action_type}")
        return tool

    @property
    def action_types(self) -> list[str]:
        return sorted(self._tools)

    def definitions(self) -> list[dict[str, str]]:
        return [
            {
                "action_type": tool.action_type,
                "name": tool.name,
                "description": tool.description,
            }
            for tool in sorted(self._tools.values(), key=lambda item: item.action_type)
        ]

    def function_definitions(self) -> list[dict[str, Any]]:
        return [
            tool.function_definition()
            for tool in sorted(self._tools.values(), key=lambda item: item.action_type)
        ]

    def build_document_overview(self, state: RunState) -> dict[str, Any]:
        return {
            "document_id": state.document.document_id,
            "page_ids": [page_id_from_index(page.page_index, document=state.document) for page in state.document.pages],
        }


def _merge_overview_fragment(base: dict[str, Any], fragment: dict[str, Any]) -> None:
    for key, value in fragment.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _merge_overview_fragment(current, value)
            continue
        base[key] = value


###############################################################################
# General Tools
###############################################################################


class StopTool(Tool):
    """Signal that execution should stop."""

    @property
    def action_type(self) -> ActionType:
        return "stop"

    @property
    def description(self) -> str:
        return (
            "End the current run explicitly, either with a final answer or with a "
            "bounded stop reason when no further safe progress is appropriate."
        )

    @property
    def target_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "No explicit target for stop.",
            "properties": {},
            "additionalProperties": False,
        }

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "Optional stop reason and optional final answer.",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Short reason for stopping.",
                },
                "answer": {
                    "type": "string",
                    "description": (
                        "Optional final answer to return when ending the run, "
                        "including outcomes such as Not answerable."
                    ),
                },
            },
            "additionalProperties": False,
        }

    async def execute(self, state: RunState, action: Action) -> Observation:
        reason = action.parameters.get("reason")
        answer = action.parameters.get("answer")
        return Observation(
            action_id=action.action_id,
            success=True,
            data={
                "reason": reason,
                "answer": answer,
            },
            message=f"Stop requested{': ' + str(reason) if reason else ''}.",
        )

    def update_state(
        self,
        state: RunState,
        action: Action,
        observation: Observation,
    ) -> None:
        answer = observation.data.get("answer")
        reason = observation.data.get("reason")
        if isinstance(reason, str) and reason:
            state.metadata["reason"] = reason
        if isinstance(answer, str) and answer:
            state.final_answer = answer
            state.status = "completed"
            return
        state.status = "stopped"


def build_default_tool_registry(
    *,
    evidence_provider: LLMProvider | None = None,
    answer_provider: LLMProvider | None = None,
    inspect_ocr_provider: LLMProvider | None = None,
) -> ToolRegistry:
    """Return a registry containing DocClaw's initial built-in tools."""
    from docclaw.agent.tool.answer import LLMAnswerTool, LLMJsonAnswerTool
    from docclaw.agent.tool.answer.answer import ExplicitAnswerOnlyTool
    from docclaw.agent.tool.chart import PaddleOCRVLChartTool
    from docclaw.agent.tool.enhancement import EnhancementTool
    from docclaw.agent.tool.evidence import LLMEvidenceTool
    from docclaw.agent.tool.formula import PaddleOCRVLFormulaTool
    from docclaw.agent.tool.internal_search import InternalSearchTool
    from docclaw.agent.tool.layout import PPDocLayoutTool
    from docclaw.agent.tool.crop import PillowCropTool
    from docclaw.agent.tool.ocr import PaddleOCRTool, TranscribeTool
    from docclaw.agent.tool.inspect_ocr import LLMInspectOcrTool
    from docclaw.agent.tool.select_ocr import LLMSelectOcrTool
    from docclaw.agent.tool.select_pages import VLMSelectPagesTool
    from docclaw.agent.tool.rotate import PillowRotateTool
    from docclaw.agent.tool.table import PaddleOCRVLTableTool
    from docclaw.agent.tool.zoom import PillowZoomTool

    answer_model_provider = answer_provider or evidence_provider
    quality_model_provider = inspect_ocr_provider or evidence_provider
    registry = ToolRegistry()
    crop_tool = PillowCropTool()
    rotate_tool = PillowRotateTool()
    zoom_tool = PillowZoomTool()
    layout_tool = PPDocLayoutTool()
    ocr_tool = PaddleOCRTool()
    for tool in (
        PaddleOCRVLChartTool(),
        PaddleOCRVLFormulaTool(),
        layout_tool,
        PaddleOCRVLTableTool(),
        crop_tool,
        rotate_tool,
        zoom_tool,
        ocr_tool,
        EnhancementTool(
            ocr_tool,
            zoom_tool=zoom_tool,
            crop_tool=crop_tool,
            rotate_tool=rotate_tool,
        ),
        InternalSearchTool(
            layout_tool=layout_tool,
            ocr_tool=ocr_tool,
        ),
        VLMSelectPagesTool(evidence_provider)
        if evidence_provider is not None
        else None,
        TranscribeTool(),
        LLMInspectOcrTool(quality_model_provider)
        if quality_model_provider is not None
        else None,
        LLMSelectOcrTool(quality_model_provider)
        if quality_model_provider is not None
        else None,
        LLMAnswerTool(answer_model_provider)
        if answer_model_provider is not None
        else ExplicitAnswerOnlyTool(),
        LLMJsonAnswerTool(answer_model_provider)
        if answer_model_provider is not None
        else None,
        StopTool(),
    ):
        if tool is not None:
            registry.register(tool)
    if evidence_provider is not None:
        registry.register(LLMEvidenceTool(evidence_provider))
    return registry
