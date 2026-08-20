"""Session management for DocClaw run history."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any

from docclaw.agent.utils import Action, Observation, Task, new_id, utc_now_iso


SESSION_ROLES = {
    "system",
    "user",
    "assistant",
    "tool",
    "task",
    "action",
    "observation",
    "event",
}

PROMPT_ROLE_MAP = {
    "task": "user",
    "action": "assistant",
    "observation": "tool",
    "event": "system",
}
ACTIVE_HISTORY_START_KEY = "active_start_index"
ACTIVE_HISTORY_ROLES = {"task", "assistant", "user"}


@dataclass(slots=True)
class SessionMessage:
    """One prompt-facing history entry in a DocClaw session."""

    role: str
    content: str
    message_id: str = field(default_factory=lambda: new_id("msg"))
    timestamp: str = field(default_factory=utc_now_iso)
    name: str | None = None
    action_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in SESSION_ROLES:
            raise ValueError(f"unsupported session role: {self.role}")
        if self.content is None:
            raise ValueError("session message content must not be None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "action_id": self.action_id,
            "metadata": self.metadata,
        }

    def to_prompt_message(self) -> dict[str, Any]:
        message = {
            "role": PROMPT_ROLE_MAP.get(self.role, self.role),
            "content": self.content,
        }
        if self.name is not None:
            message["name"] = self.name
        if self.action_id is not None:
            message["action_id"] = self.action_id
        return message

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMessage:
        return cls(
            message_id=str(data["message_id"]),
            timestamp=str(data["timestamp"]),
            role=str(data["role"]),
            content=str(data.get("content") or ""),
            name=data.get("name"),
            action_id=data.get("action_id"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class Session:
    """Prompt-facing history for one DocClaw interaction thread."""

    session_id: str = field(default_factory=lambda: new_id("session"))
    messages: list[SessionMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id must not be empty")

    def add_message(
        self,
        role: str,
        content: str,
        *,
        name: str | None = None,
        action_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMessage:
        message = SessionMessage(
            role=role,
            content=content,
            name=name,
            action_id=action_id,
            metadata=dict(metadata or {}),
        )
        self.messages.append(message)
        self.updated_at = utc_now_iso()
        return message

    def add_task(self, task: Task) -> SessionMessage:
        return self.add_message(
            "task",
            task.prompt,
            metadata={"task": task.to_dict()},
        )

    def add_action(self, action: Action) -> SessionMessage:
        target = f" target={action.target}" if action.target else ""
        rationale = f"\nRationale: {action.rationale}" if action.rationale else ""
        return self.add_message(
            "action",
            f"Action: {action.action_type}{target}{rationale}",
            name=action.action_type,
            action_id=action.action_id,
            metadata={"action": action.to_dict()},
        )

    def add_observation(self, observation: Observation) -> SessionMessage:
        if observation.success:
            content = observation.message or "Observation succeeded."
        else:
            content = observation.error or "Observation failed."
        return self.add_message(
            "observation",
            content,
            action_id=observation.action_id,
            metadata={"observation": observation.to_dict()},
        )

    def get_history(self, max_messages: int = 0) -> list[SessionMessage]:
        if max_messages <= 0 or len(self.messages) <= max_messages:
            return list(self.messages)
        return list(self.messages[-max_messages:])

    def get_active_history(self, max_messages: int = 0) -> list[SessionMessage]:
        start = self.active_start_index
        active = [
            message
            for message in self.messages[start:]
            if message.role in ACTIVE_HISTORY_ROLES
        ]
        if max_messages <= 0 or len(active) <= max_messages:
            return active
        return active[-max_messages:]

    def to_messages(self, max_messages: int = 0) -> list[dict[str, Any]]:
        return [
            message.to_prompt_message()
            for message in self.get_history(max_messages=max_messages)
        ]

    def to_active_messages(self, max_messages: int = 0) -> list[dict[str, Any]]:
        return [
            message.to_prompt_message()
            for message in self.get_active_history(max_messages=max_messages)
        ]

    @property
    def active_start_index(self) -> int:
        raw = self.metadata.get(ACTIVE_HISTORY_START_KEY, 0)
        try:
            index = int(raw)
        except (TypeError, ValueError):
            index = 0
        return max(0, min(index, len(self.messages)))

    def reset_active_history(self) -> None:
        self.metadata[ACTIVE_HISTORY_START_KEY] = len(self.messages)
        self.updated_at = utc_now_iso()

    def clear(self) -> None:
        self.messages.clear()
        self.metadata[ACTIVE_HISTORY_START_KEY] = 0
        self.updated_at = utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "messages": [message.to_dict() for message in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        return cls(
            session_id=str(data["session_id"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            metadata=dict(data.get("metadata") or {}),
            messages=[
                SessionMessage.from_dict(message)
                for message in data.get("messages", [])
            ],
        )


class SessionManager:
    """Persistent registry for DocClaw sessions."""

    def __init__(self, storage_dir: str | Path) -> None:
        root = Path(storage_dir)
        self.storage_dir = root.expanduser().resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}

    @staticmethod
    def _safe_stem(session_id: str) -> str:
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", session_id).strip("._")
        return stem or "session"

    def _session_path(self, session_id: str) -> Path:
        return self.storage_dir / f"{self._safe_stem(session_id)}.json"

    def _load(self, session_id: str) -> Session | None:
        path = self._session_path(session_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        session = Session.from_dict(data)
        self._sessions[session.session_id] = session
        return session

    def create(
        self,
        session_id: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        session = Session(
            session_id=session_id or new_id("session"),
            metadata=dict(metadata or {}),
        )
        self.save(session)
        return session

    def save(self, session: Session) -> None:
        self._sessions[session.session_id] = session
        path = self._session_path(session.session_id)
        tmp_path = path.with_suffix(".json.tmp")
        payload = json.dumps(_to_jsonable(session.to_dict()), ensure_ascii=False)
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, path)

    def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is not None:
            return session
        return self._load(session_id)

    def require(self, session_id: str) -> Session:
        session = self.get(session_id)
        if session is None:
            raise KeyError(f"unknown session_id: {session_id}")
        return session

    def delete(self, session_id: str) -> bool:
        self._sessions.pop(session_id, None)
        path = self._session_path(session_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_sessions(self) -> list[Session]:
        sessions = dict(self._sessions)
        for path in self.storage_dir.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            session = Session.from_dict(data)
            sessions.setdefault(session.session_id, session)
        return sorted(
            sessions.values(),
            key=lambda session: session.updated_at,
            reverse=True,
        )

    def _clear_cache(self) -> None:
        self._sessions.clear()

    def clear(self) -> None:
        self._clear_cache()
        for path in self.storage_dir.glob("*.json"):
            path.unlink()


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
