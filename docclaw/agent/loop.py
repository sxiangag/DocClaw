"""Agent loop for coordinating planning, execution, and state updates."""

from __future__ import annotations

from typing import Any, Callable

from docclaw.agent.executor import Executor
from docclaw.agent.planner import Planner
from docclaw.agent.tool.tool import Tool
from docclaw.agent.utils import Action, Observation, RunResult, RunState

RunEventCallback = Callable[[str, dict[str, Any]], None]


class AgentLoop:
    """Run the DocClaw action loop for one task and document."""

    def __init__(self, executor: Executor | None = None) -> None:
        self.executor = executor or Executor()

    async def run(
        self,
        state: RunState,
        planner: Planner,
        *,
        max_steps: int = 100,
        on_event: RunEventCallback | None = None,
    ) -> RunResult:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        steps = 0
        while state.status == "running" and steps < max_steps:
            action = await planner.next_action(state)
            if action is None:
                state.status = "stopped"
                break
            for event_name, event_payload in state.pop_pending_events():
                self._emit(on_event, event_name, event_payload)

            self._emit(
                on_event,
                "step_started",
                {
                    "step_index": steps,
                    "action": action,
                },
            )
            observation = await self.executor.execute(state, action)
            tool = self.executor.tools.get(action.action_type)
            self._apply_observation(state, action, observation, tool)
            self._emit(
                on_event,
                "step_finished",
                {
                    "step_index": steps,
                    "action": action,
                    "observation": observation,
                    "status": state.status,
                },
            )
            steps += 1

        if state.status == "running" and steps >= max_steps:
            state.status = "max_steps"

        result = RunResult(
            state=state,
            status=state.status,
            answer=state.final_answer,
            reason=state.metadata.get("reason"),
            error=state.metadata.get("last_error"),
        )
        self._emit(
            on_event,
            "run_finished",
            {
                "steps": steps,
                "result": result,
            },
        )
        return result

    def _apply_observation(
        self,
        state: RunState,
        action: Action,
        observation: Observation,
        tool: Tool | None,
    ) -> None:
        if not observation.success:
            state.metadata["last_error"] = observation.error or "action failed"
            state.metadata["failed_action"] = action.to_dict()
            state.add_trace_step(action, observation)
            return
        state.metadata.pop("last_error", None)
        state.metadata.pop("failed_action", None)
        if tool is not None:
            tool.update_state(state, action, observation)
        state.add_trace_step(action, observation)

    @staticmethod
    def _emit(
        callback: RunEventCallback | None,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        if callback is not None:
            callback(event, payload)
