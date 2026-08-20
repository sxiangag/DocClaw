"""LLM provider abstractions for DocClaw."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
from typing import Any


@dataclass(slots=True)
class ToolCallRequest:
    """A tool call emitted by a provider response."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class LLMResponse:
    """Provider response returned to DocClaw."""

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    retry_after: float | None = None
    reasoning_content: str | None = None
    error: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(ABC):
    """Base interface for provider used by DocClaw planners and agents."""

    supports_progress_deltas = False

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Return a completion for the given messages."""

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Default streaming implementation falls back to one non-streaming call."""
        response = await self.chat(
            messages,
            tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        if on_content_delta and response.content:
            await on_content_delta(response.content)
        return response

    @staticmethod
    def _extract_retry_after(content: str | None) -> float | None:
        text = (content or "").lower()
        patterns = (
            r"retry after\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)?",
            r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            return LLMProvider._to_retry_seconds(float(match.group(1)), match.group(2))
        return None

    @staticmethod
    def _to_retry_seconds(value: float, unit: str | None) -> float:
        normalized = (unit or "s").lower()
        if normalized in {"ms", "milliseconds"}:
            return max(0.1, value / 1000.0)
        if normalized in {"m", "min", "minutes"}:
            return max(0.1, value * 60.0)
        return max(0.1, value)

    @staticmethod
    def _extract_retry_after_from_headers(headers: Any) -> float | None:
        if not headers:
            return None

        def _header_value(name: str) -> Any:
            if hasattr(headers, "get"):
                value = headers.get(name)
                if value is not None:
                    return value
                return headers.get(name.title())
            if isinstance(headers, dict):
                for key, value in headers.items():
                    if isinstance(key, str) and key.lower() == name.lower():
                        return value
            return None

        retry_ms = _header_value("retry-after-ms")
        if retry_ms is not None:
            try:
                return max(0.1, float(retry_ms) / 1000.0)
            except (TypeError, ValueError):
                pass

        retry_after = _header_value("retry-after")
        if retry_after is None:
            return None

        text = str(retry_after).strip()
        if not text:
            return None
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            return max(0.1, float(text))

        try:
            retry_at = parsedate_to_datetime(text)
        except Exception:
            return None

        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        remaining = (retry_at - datetime.now(retry_at.tzinfo)).total_seconds()
        return max(0.1, remaining) if remaining > 0 else 0.1

    @abstractmethod
    def get_default_model(self) -> str:
        """Return the provider's default model identifier."""
