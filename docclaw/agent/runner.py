"""Convenience runner for DocClaw agent runs."""

from __future__ import annotations

from pathlib import Path

from docclaw.agent.loop import AgentLoop, RunEventCallback
from docclaw.agent.planner import Planner
from docclaw.agent.utils import DocumentState, RunResult, RunState, Task
from docclaw.session.manager import Session, SessionManager


class DocClawRunner:
    """Create run state, execute the loop, and persist session history."""

    def __init__(
        self,
        loop: AgentLoop | None = None,
        session_manager: SessionManager | None = None,
        *,
        session_max_messages: int = 10,
    ) -> None:
        self.loop = loop or AgentLoop()
        self.session_manager = session_manager or SessionManager(
            Path.cwd() / ".docclaw" / "sessions"
        )
        self.session_max_messages = session_max_messages

    async def run(
        self,
        document: DocumentState,
        task: Task | str,
        planner: Planner,
        *,
        max_steps: int = 100,
        session: Session | None = None,
        session_id: str | None = None,
        on_event: RunEventCallback | None = None,
    ) -> RunResult:
        if session is not None and session_id is not None:
            raise ValueError("pass either session or session_id, not both")

        task_state = task if isinstance(task, Task) else Task(prompt=task)
        state = RunState(document=document, task=task_state)

        if session_id is not None:
            existing = self.session_manager.get(session_id)
            if existing is not None:
                session = existing
        if session is None:
            session = self.session_manager.create(
                session_id,
                metadata={"document_id": document.document_id},
            )
        else:
            session.metadata.setdefault("document_id", document.document_id)
        state.metadata["session_history"] = session.to_active_messages(
            max_messages=self.session_max_messages,
        )

        result = await self.loop.run(
            state,
            planner,
            max_steps=max_steps,
            on_event=on_event,
        )
        self._save_running_history(session, result)
        self.session_manager.save(session)
        return result


    @staticmethod
    def _save_running_history(session: Session, result: RunResult) -> None:
        session.add_task(result.state.task)
        for step in result.trace:
            session.add_action(step.action)
            session.add_observation(step.observation)
        assistant_content = _assistant_content(result)
        if assistant_content:
            session.add_message(
                "assistant",
                assistant_content,
                metadata={
                    "run_result": {
                        "status": result.status,
                        "answer": result.answer,
                        "error": result.error,
                    }
                },
            )


def _assistant_content(result: RunResult) -> str:
    if isinstance(result.answer, str) and result.answer:
        return result.answer
    if result.status == "failed":
        return result.error or "failed"
    if result.status == "stopped":
        reason = result.state.metadata.get("reason")
        if isinstance(reason, str) and reason:
            return f"Stopped: {reason}"
        return "stopped"
    if result.status == "max_steps":
        return "max_steps"
    return result.status
