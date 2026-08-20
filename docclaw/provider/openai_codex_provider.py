"""OpenAI Codex Responses provider for DocClaw."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import hashlib
import importlib
import json
from typing import Any

from docclaw.provider.base import LLMProvider, LLMResponse, ToolCallRequest

DEFAULT_CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_CODEX_MODEL = "openai-codex/gpt-5.2"
DEFAULT_ORIGINATOR = "docclaw"
_FINISH_REASON_MAP = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "error",
    "cancelled": "error",
}


@dataclass(slots=True, frozen=True)
class CodexAccessToken:
    """Resolved Codex access token used to authenticate requests."""

    account_id: str
    access_token: str


class OpenAICodexProvider(LLMProvider):
    """Use Codex OAuth to call the Responses API."""

    supports_progress_deltas = True

    def __init__(
        self,
        default_model: str = DEFAULT_CODEX_MODEL,
        *,
        url: str = DEFAULT_CODEX_URL,
        originator: str = DEFAULT_ORIGINATOR,
        verify_ssl: bool = True,
        allow_insecure_ssl_fallback: bool = True,
        request_timeout: float = 60.0,
        token_provider: Callable[[], Awaitable[Any] | Any] | None = None,
        requester: Callable[
            [str, dict[str, str], dict[str, Any], bool, Callable[[str], Awaitable[None]] | None],
            Awaitable[tuple[str, list[ToolCallRequest], str, str | None] | tuple[str, list[ToolCallRequest], str]],
        ] | None = None,
    ) -> None:
        self.default_model = default_model
        self.url = url
        self.originator = originator
        self.verify_ssl = verify_ssl
        self.allow_insecure_ssl_fallback = allow_insecure_ssl_fallback
        self.request_timeout = request_timeout
        self._token_provider = token_provider
        self._requester = requester or _request_codex

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
        return await self._call_codex(
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
        return await self._call_codex(
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

    async def _call_codex(
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
            token = await self._get_token()
            delta_callback = _coerce_delta_callback(on_content_delta)
            headers = _build_headers(
                token.account_id,
                token.access_token,
                originator=self.originator,
            )
            body: dict[str, Any] = {
                "model": _strip_model_prefix(resolved_model),
                "store": False,
                "stream": True,
                "instructions": system_prompt,
                "input": input_items,
                "text": {"verbosity": "medium"},
                "include": ["reasoning.encrypted_content"],
                "prompt_cache_key": _prompt_cache_key(messages),
                "tool_choice": tool_choice or "auto",
                "parallel_tool_calls": True,
            }
            if reasoning_effort and reasoning_effort.lower() != "none":
                body["reasoning"] = {"effort": reasoning_effort}
            if tools:
                body["tools"] = convert_tools(tools)

            requester = self._requester or self._request_codex
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

            content, tool_calls, finish_reason, reasoning_content = _normalize_request_result(result)
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                reasoning_content=reasoning_content,
            )
        except Exception as exc:
            message = f"Error calling Codex: {exc}"
            retry_after = getattr(exc, "retry_after", None) or self._extract_retry_after(message)
            return LLMResponse(
                content=message,
                finish_reason="error",
                retry_after=retry_after,
                error=message,
            )

    async def _get_token(self) -> CodexAccessToken:
        provider = self._token_provider or _default_codex_token_provider
        result = provider()
        if hasattr(result, "__await__"):
            result = await result
        return _normalize_token(result)

    async def _request_codex(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        verify: bool,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str, list[ToolCallRequest], str, str | None]:
        return await _request_codex(
            url,
            headers,
            body,
            verify,
            on_content_delta,
            timeout=self.request_timeout,
        )

    def _should_retry_insecure(self, exc: Exception) -> bool:
        if not self.allow_insecure_ssl_fallback or not self.verify_ssl:
            return False
        return "CERTIFICATE_VERIFY_FAILED" in str(exc)


class _CodexHTTPError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _normalize_request_result(
    result: tuple[str, list[ToolCallRequest], str, str | None] | tuple[str, list[ToolCallRequest], str],
) -> tuple[str, list[ToolCallRequest], str, str | None]:
    if len(result) == 4:
        content, tool_calls, finish_reason, reasoning_content = result
        return content, tool_calls, finish_reason, reasoning_content
    content, tool_calls, finish_reason = result
    return content, tool_calls, finish_reason, None


def _normalize_token(value: Any) -> CodexAccessToken:
    if isinstance(value, CodexAccessToken):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return CodexAccessToken(account_id=str(value[0]), access_token=str(value[1]))

    account_id = getattr(value, "account_id", None)
    access_token = getattr(value, "access_token", None)
    if access_token is None:
        access_token = getattr(value, "access", None)
    if isinstance(account_id, str) and isinstance(access_token, str):
        return CodexAccessToken(account_id=account_id, access_token=access_token)

    raise TypeError("token provider must return CodexAccessToken, a 2-tuple, or an object with account_id/access")


def _strip_model_prefix(model: str) -> str:
    if model.startswith("openai-codex/") or model.startswith("openai_codex/"):
        return model.split("/", 1)[1]
    return model


def _build_headers(account_id: str, token: str, *, originator: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": originator,
        "User-Agent": "docclaw (python)",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }


def _prompt_cache_key(messages: list[dict[str, Any]]) -> str:
    raw = json.dumps(messages, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _default_codex_token_provider() -> Any:
    try:
        module = importlib.import_module("oauth_cli_kit")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The `oauth_cli_kit` package is required for OpenAICodexProvider. "
            "Install it or provide a custom token_provider."
        ) from exc
    return await asyncio.to_thread(module.get_token)


async def _request_codex(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    verify: bool,
    on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    *,
    timeout: float = 60.0,
) -> tuple[str, list[ToolCallRequest], str, str | None]:
    try:
        httpx = importlib.import_module("httpx")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The `httpx` package is required for OpenAICodexProvider requests."
        ) from exc

    async with httpx.AsyncClient(timeout=timeout, verify=verify) as client:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code != 200:
                text = await response.aread()
                retry_after = LLMProvider._extract_retry_after_from_headers(response.headers)
                raise _CodexHTTPError(
                    _friendly_error(response.status_code, text.decode("utf-8", "ignore")),
                    retry_after=retry_after,
                )
            return await consume_sse(response, on_content_delta)


def _friendly_error(status_code: int, raw: str) -> str:
    if status_code == 429:
        return "ChatGPT usage quota exceeded or rate limit triggered. Please try again later."
    return f"HTTP {status_code}: {raw}"


def _coerce_delta_callback(
    callback: Callable[[str], Any] | None,
) -> Callable[[str], Awaitable[None]] | None:
    if callback is None:
        return None

    async def _wrapped(text: str) -> None:
        result = callback(text)
        if hasattr(result, "__await__"):
            await result

    return _wrapped


def _map_finish_reason(status: str | None) -> str:
    return _FINISH_REASON_MAP.get(status or "completed", "stop")


def convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Convert chat-style messages into a Responses-style instruction/input pair."""
    system_parts: list[str] = []
    input_items: list[dict[str, Any]] = []

    for idx, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            if isinstance(content, str) and content.strip():
                system_parts.append(content)
            continue

        if role == "user":
            input_items.append(convert_user_message(content))
            continue

        if role == "assistant":
            if isinstance(content, str) and content:
                input_items.append({
                    "type": "message",
                    "id": f"msg_{idx}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": content}],
                })
            for tool_call in message.get("tool_calls") or []:
                fn = tool_call.get("function") or {}
                input_items.append({
                    "type": "function_call",
                    "id": tool_call.get("id") or f"fc_{idx}",
                    "call_id": tool_call.get("id") or f"call_{idx}",
                    "name": fn.get("name") or "",
                    "arguments": fn.get("arguments") or "{}",
                })
            continue

        if role == "tool":
            output = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            input_items.append({
                "type": "function_call_output",
                "call_id": message.get("tool_call_id") or f"call_{idx}",
                "output": output,
            })

    return "\n\n".join(system_parts), input_items


def convert_user_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {
            "role": "user",
            "content": [{"type": "input_text", "text": content}],
        }
    if isinstance(content, list):
        converted: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                converted.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                image_url = item.get("image_url") or {}
                url = image_url.get("url") if isinstance(image_url, dict) else None
                if url:
                    converted.append({
                        "type": "input_image",
                        "image_url": url,
                        "detail": "auto",
                    })
        if converted:
            return {"role": "user", "content": converted}
    return {"role": "user", "content": [{"type": "input_text", "text": ""}]}


def convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = fn.get("parameters")
        converted.append({
            "type": "function",
            "name": name,
            "description": fn.get("description") or "",
            "parameters": parameters if isinstance(parameters, dict) else {},
        })
    return converted


async def consume_sse(
    response: Any,
    on_content_delta: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, list[ToolCallRequest], str, str | None]:
    """Consume a Responses-style SSE stream."""
    content = ""
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCallRequest] = []
    tool_call_buffers: dict[str, dict[str, Any]] = {}
    finish_reason = "stop"
    buffer: list[str] = []

    def _flush_buffer() -> dict[str, Any] | None:
        data_lines = [line[5:].strip() for line in buffer if line.startswith("data:")]
        buffer.clear()
        if not data_lines:
            return None
        data = "\n".join(data_lines).strip()
        if not data or data == "[DONE]":
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None

    async for line in response.aiter_lines():
        if line == "":
            event = _flush_buffer()
            if event is None:
                continue
            event_type = event.get("type")
            if event_type == "response.output_item.added":
                item = event.get("item") or {}
                if item.get("type") == "function_call":
                    call_id = item.get("call_id")
                    if isinstance(call_id, str) and call_id:
                        tool_call_buffers[call_id] = {
                            "id": item.get("id") or "fc_0",
                            "name": item.get("name") or "",
                            "arguments": item.get("arguments") or "",
                        }
            elif event_type == "response.output_text.delta":
                delta = event.get("delta") or ""
                content += delta
                if on_content_delta and delta:
                    await on_content_delta(delta)
            elif event_type == "response.reasoning_summary_text.delta":
                delta = event.get("delta") or ""
                if delta:
                    reasoning_parts.append(delta)
            elif event_type == "response.function_call_arguments.delta":
                call_id = event.get("call_id")
                if call_id in tool_call_buffers:
                    tool_call_buffers[call_id]["arguments"] += event.get("delta") or ""
            elif event_type == "response.function_call_arguments.done":
                call_id = event.get("call_id")
                if call_id in tool_call_buffers:
                    tool_call_buffers[call_id]["arguments"] = event.get("arguments") or ""
            elif event_type == "response.output_item.done":
                item = event.get("item") or {}
                if item.get("type") == "function_call":
                    call_id = item.get("call_id")
                    if call_id in tool_call_buffers:
                        buf = tool_call_buffers[call_id]
                        args_raw = buf.get("arguments") or item.get("arguments") or "{}"
                        try:
                            arguments = json.loads(args_raw)
                        except json.JSONDecodeError:
                            arguments = {"raw": args_raw}
                        if not isinstance(arguments, dict):
                            arguments = {"raw": args_raw}
                        tool_calls.append(
                            ToolCallRequest(
                                id=f"{call_id}|{buf.get('id') or item.get('id') or 'fc_0'}",
                                name=buf.get("name") or item.get("name") or "",
                                arguments=arguments,
                            )
                        )
            elif event_type == "response.completed":
                finish_reason = _map_finish_reason((event.get("response") or {}).get("status"))
            elif event_type in {"error", "response.failed"}:
                detail = event.get("error") or event.get("message") or event
                raise RuntimeError(f"Response failed: {str(detail)[:500]}")
            continue

        buffer.append(line)

    event = _flush_buffer()
    if event is not None and event.get("type") == "response.completed":
        finish_reason = _map_finish_reason((event.get("response") or {}).get("status"))

    reasoning_content = "".join(reasoning_parts) or None
    return content, tool_calls, finish_reason, reasoning_content
