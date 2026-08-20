"""Base answer tool."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from docclaw.agent.tool.tool import Tool
from docclaw.agent.utils import Action, ActionType, Observation, RunState


class AnswerTool(Tool):
    """Base class for tools that produce a final answer."""

    @property
    def action_type(self) -> ActionType:
        return "answer_from_evidence"

    @property
    def description(self) -> str:
        return (
            "Return a final answer using accumulated run-state evidence. "
            "Call this only after extract_evidence has already added "
            "evidence items to state.evidence."
        )

    @property
    def target_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "No explicit target for answer generation.",
            "properties": {},
            "additionalProperties": False,
        }

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "No additional answer-generation parameters.",
            "properties": {},
            "additionalProperties": False,
        }

    async def execute(self, state: RunState, action: Action) -> Observation:
        return await self.generate_answer(state, action)

    @abstractmethod
    async def generate_answer(self, state: RunState, action: Action) -> Observation:
        """Return an answer observation for the current state."""

    def update_state(
        self,
        state: RunState,
        action: Action,
        observation: Observation,
    ) -> None:
        answer = observation.data.get("answer")
        if isinstance(answer, str) and answer:
            state.final_answer = answer
            state.status = "completed"


class ExplicitAnswerOnlyTool(AnswerTool):
    """Fallback answer tool used when no answer provider is configured."""

    @property
    def description(self) -> str:
        return "Fail because no answer provider is configured."

    async def generate_answer(self, state: RunState, action: Action) -> Observation:
        return self.error(action, "answer provider is not configured")
