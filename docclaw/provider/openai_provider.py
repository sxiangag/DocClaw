"""OpenAI Responses API provider for DocClaw."""

from __future__ import annotations

import importlib
import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from docclaw.provider.base import LLMProvider, LLMResponse, ToolCallRequest
from docclaw.provider.openai_codex_provider import (
    _coerce_delta_callback,
    convert_messages,
    convert_tools,
)

DEFAULT_OPENAI_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_MODEL = "gpt-5.5"
_FINISH_REASON_MAP = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "error",
    "cancelled": "error",
}


class OpenAIProvider(LLMProvider):
    """Use the standard OpenAI Responses API."""

    supports_progress_deltas = True

    def __init__(
        self,
        default_model: str = DEFAULT_OPENAI_MODEL,
        *,
        api_key: str,
        url: str = DEFAULT_OPENAI_URL,
        verify_ssl: bool = True,
        allow_insecure_ssl_fallback: bool = True,
        request_timeout: float = 60.0,
        requester: Callable[
            [str, dict[str, str], dict[str, Any], bool, Callable[[str], Awaitable[None]] | None],
            Awaitable[tuple[str, list[ToolCallRequest], str, str | None] | tuple[str, list[ToolCallRequest], str]],
        ] | None = None,
    ) -> None:
        self.default_model = default_model
        self.api_key = api_key
        self.url = url
        self.verify_ssl = verify_ssl
        self.allow_insecure_ssl_fallback = allow_insecure_ssl_fallback
        self.request_timeout = request_timeout
        self._requester = requester or self._request_openai

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
        return await self._call_openai(
            messages,
            tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )

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
        on_content_delta: Callable[[str], Any] | None = None,
    ) -> LLMResponse:
        return await self._call_openai(
            messages,
            tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
            on_content_delta=on_content_delta,
        )

    def get_default_model(self) -> str:
        return self.default_model

    async def _call_openai(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        model: str | None,
        max_tokens: int | None,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
        on_content_delta: Callable[[str], Any] | None = None,
    ) -> LLMResponse:
        resolved_model = model or self.default_model

        try:
            system_prompt, input_items = convert_messages(messages)
            delta_callback = _coerce_delta_callback(on_content_delta)
            headers = _build_headers(api_key=self.api_key)
            body: dict[str, Any] = {
                "model": resolved_model,
                "stream": True,
                "input": input_items,
                "tool_choice": tool_choice or "auto",
            }
            if system_prompt:
                body["instructions"] = system_prompt
            if max_tokens is not None:
                body["max_output_tokens"] = max_tokens
            if temperature != 0.0:
                body["temperature"] = temperature
            if reasoning_effort and reasoning_effort.lower() != "none":
                body["reasoning"] = {"effort": reasoning_effort}
            if tools:
                body["tools"] = convert_tools(tools)
                body["parallel_tool_calls"] = True

            requester = self._requester or self._request_openai
            try:
                result = await requester(
                    self.url,
                    headers,
                    body,
                    self.verify_ssl,
                    delta_callback,
                )
            except Exception as exc:
                if not self._should_retry_insecure(exc):
                    raise
                result = await requester(
                    self.url,
                    headers,
                    body,
                    False,
                    delta_callback,
                )

            content, tool_calls, finish_reason, reasoning_content, usage = _normalize_request_result(result)
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                reasoning_content=reasoning_content,
                usage=usage,
            )
        except Exception as exc:
            message = f"Error calling OpenAI: {exc}"
            retry_after = getattr(exc, "retry_after", None) or self._extract_retry_after(message)
            return LLMResponse(
                content=message,
                finish_reason="error",
                retry_after=retry_after,
                error=message,
            )

    async def _request_openai(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        verify: bool,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[ToolCallRequest], str, str | None, dict[str, int]]:
        try:
            openai_module = importlib.import_module("openai")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The `openai` package is required for OpenAIProvider requests."
            ) from exc

        timeout = httpx.Timeout(self.request_timeout)
        async with httpx.AsyncClient(timeout=timeout, verify=verify) as http_client:
            client = openai_module.AsyncOpenAI(
                api_key=self.api_key,
                base_url=_resolve_base_url(url),
                timeout=timeout,
                max_retries=0,
                http_client=http_client,
            )
            try:
                stream = await client.responses.create(**body)
            except Exception as exc:
                status_code = _status_code_from_exc(exc)
                if status_code is not None:
                    response = getattr(exc, "response", None)
                    retry_after = self._extract_retry_after_from_headers(getattr(response, "headers", None))
                    raise _OpenAIHTTPError(
                        _friendly_error(status_code, _error_detail_from_exc(exc)),
                        retry_after=retry_after,
                    ) from exc
                raise

            try:
                return await consume_openai_stream(stream, on_content_delta)
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    result = close()
                    if hasattr(result, "__await__"):
                        await result

    def _should_retry_insecure(self, exc: Exception) -> bool:
        if not self.allow_insecure_ssl_fallback or not self.verify_ssl:
            return False
        return "CERTIFICATE_VERIFY_FAILED" in str(exc)


class _OpenAIHTTPError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _build_headers(*, api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }


def _friendly_error(status_code: int, raw: str) -> str:
    if status_code == 401:
        return "OpenAI API authentication failed. Check the configured API key."
    if status_code == 429:
        return "OpenAI API rate limit or quota exceeded. Please try again later."
    return f"HTTP {status_code}: {raw}"


def _normalize_request_result(
    result: (
        tuple[str | None, list[ToolCallRequest], str, str | None, dict[str, int]]
        | tuple[str | None, list[ToolCallRequest], str, str | None]
        | tuple[str | None, list[ToolCallRequest], str]
    ),
) -> tuple[str | None, list[ToolCallRequest], str, str | None, dict[str, int]]:
    if len(result) == 5:
        content, tool_calls, finish_reason, reasoning_content, usage = result
        return content, tool_calls, finish_reason, reasoning_content, usage
    if len(result) == 4:
        content, tool_calls, finish_reason, reasoning_content = result
        return content, tool_calls, finish_reason, reasoning_content, {}
    content, tool_calls, finish_reason = result
    return content, tool_calls, finish_reason, None, {}


def _resolve_base_url(url: str) -> str:
    normalized = url.rstrip("/")
    if normalized.endswith("/responses"):
        normalized = normalized[: -len("/responses")]
    return normalized + "/"


def _status_code_from_exc(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def _error_detail_from_exc(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
        message = body.get("message")
        if isinstance(message, str) and message:
            return message
        return json.dumps(body, ensure_ascii=False)
    return str(exc)


def _map_finish_reason(status: Any) -> str:
    if not isinstance(status, str):
        return "stop"
    return _FINISH_REASON_MAP.get(status, "stop")


def _tool_calls_from_response(response: Any) -> list[ToolCallRequest]:
    tool_calls: list[ToolCallRequest] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "function_call":
            continue
        args_raw = getattr(item, "arguments", "") or "{}"
        try:
            arguments = json.loads(args_raw)
        except json.JSONDecodeError:
            arguments = {"raw": args_raw}
        if not isinstance(arguments, dict):
            arguments = {"raw": args_raw}
        tool_calls.append(
            ToolCallRequest(
                id=f"{getattr(item, 'call_id', '')}|{getattr(item, 'id', '')}".strip("|"),
                name=str(getattr(item, "name", "") or ""),
                arguments=arguments,
            )
        )
    return tool_calls


def _reasoning_content_from_response(response: Any) -> str | None:
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "reasoning":
            continue
        for summary in getattr(item, "summary", []) or []:
            text = getattr(summary, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
    return "".join(parts) or None


def _usage_from_response(response: Any) -> dict[str, int]:
    usage_payload = getattr(response, "usage", None)
    if usage_payload is None:
        return {}

    usage: dict[str, int] = {}
    for source_key, target_key in (
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = getattr(usage_payload, source_key, None)
        if isinstance(value, int):
            usage[target_key] = value

    input_details = getattr(usage_payload, "input_tokens_details", None)
    cached_tokens = getattr(input_details, "cached_tokens", None)
    if isinstance(cached_tokens, int):
        usage["cache_read_input_tokens"] = cached_tokens

    output_details = getattr(usage_payload, "output_tokens_details", None)
    reasoning_tokens = getattr(output_details, "reasoning_tokens", None)
    if isinstance(reasoning_tokens, int):
        usage["reasoning_tokens"] = reasoning_tokens

    return usage


async def consume_openai_stream(
    stream: Any,
    on_content_delta: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str | None, list[ToolCallRequest], str, str | None, dict[str, int]]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    completed_response: Any = None

    async for event in stream:
        event_type = getattr(event, "type", None)
        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "") or ""
            if delta:
                content_parts.append(delta)
                if on_content_delta is not None:
                    await on_content_delta(delta)
            continue
        if event_type == "response.reasoning_summary_text.delta":
            delta = getattr(event, "delta", "") or ""
            if delta:
                reasoning_parts.append(delta)
            continue
        if event_type == "response.completed":
            completed_response = getattr(event, "response", None)
            continue
        if event_type in {"error", "response.failed"}:
            detail = getattr(event, "error", None) or getattr(event, "message", None) or event
            raise RuntimeError(f"Response failed: {str(detail)[:500]}")

    if completed_response is None:
        raise RuntimeError("OpenAI Responses stream ended without response.completed")

    content = "".join(content_parts) or getattr(completed_response, "output_text", "") or None
    reasoning_content = _reasoning_content_from_response(completed_response) or "".join(reasoning_parts) or None
    tool_calls = _tool_calls_from_response(completed_response)
    finish_reason = _map_finish_reason(getattr(completed_response, "status", None))
    usage = _usage_from_response(completed_response)
    return content, tool_calls, finish_reason, reasoning_content, usage
