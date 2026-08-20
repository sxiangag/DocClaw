"""Anthropic Messages API provider for DocClaw."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import json
from typing import Any
from urllib.parse import urlparse

import httpx

from docclaw.provider.base import LLMProvider, LLMResponse, ToolCallRequest

DEFAULT_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_MAX_TOKENS = 8192
_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "error",
}


class AnthropicProvider(LLMProvider):
    """Use the Anthropic Messages API."""

    def __init__(
        self,
        default_model: str = DEFAULT_ANTHROPIC_MODEL,
        *,
        api_key: str,
        url: str = DEFAULT_ANTHROPIC_URL,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
        verify_ssl: bool = True,
        allow_insecure_ssl_fallback: bool = True,
        request_timeout: float = 60.0,
        requester: Callable[
            [str, dict[str, str], dict[str, Any], bool],
            Awaitable[dict[str, Any]],
        ] | None = None,
    ) -> None:
        self.default_model = default_model
        self.api_key = api_key
        self.url = url
        self.anthropic_version = anthropic_version
        self.verify_ssl = verify_ssl
        self.allow_insecure_ssl_fallback = allow_insecure_ssl_fallback
        self.request_timeout = request_timeout
        self._requester = requester or self._request_anthropic

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
        del reasoning_effort
        resolved_model = model or self.default_model
        try:
            system_prompt, anthropic_messages = convert_messages(messages)
            body: dict[str, Any] = {
                "model": resolved_model,
                "messages": anthropic_messages,
                "max_tokens": max_tokens if max_tokens is not None else DEFAULT_ANTHROPIC_MAX_TOKENS,
            }
            if system_prompt:
                body["system"] = system_prompt
            if temperature != 0.0:
                body["temperature"] = temperature
            if tools:
                body["tools"] = convert_tools(tools)
                body["tool_choice"] = convert_tool_choice(tool_choice)

            headers = _build_headers(
                api_key=self.api_key,
                anthropic_version=self.anthropic_version,
            )
            requester = self._requester or self._request_anthropic
            try:
                payload = await requester(self.url, headers, body, self.verify_ssl)
            except Exception as exc:
                if not self._should_retry_insecure(exc):
                    raise
                payload = await requester(self.url, headers, body, False)
            return _response_from_payload(payload)
        except Exception as exc:
            message = f"Error calling Anthropic: {exc}"
            retry_after = getattr(exc, "retry_after", None) or self._extract_retry_after(message)
            return LLMResponse(
                content=message,
                finish_reason="error",
                retry_after=retry_after,
                error=message,
            )

    def get_default_model(self) -> str:
        return self.default_model

    async def _request_anthropic(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        verify: bool,
    ) -> dict[str, Any]:
        timeout = httpx.Timeout(self.request_timeout)
        async with httpx.AsyncClient(timeout=timeout, verify=verify) as client:
            response = await client.post(url, headers=headers, json=body)
        if response.status_code >= 400:
            retry_after = self._extract_retry_after_from_headers(response.headers)
            detail = _extract_error_detail(response)
            raise _AnthropicHTTPError(
                f"{response.status_code} {response.reason_phrase}: {detail}",
                retry_after=retry_after,
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Anthropic response must be a JSON object")
        return payload

    def _should_retry_insecure(self, exc: Exception) -> bool:
        if not self.allow_insecure_ssl_fallback or not self.verify_ssl:
            return False
        return "CERTIFICATE_VERIFY_FAILED" in str(exc)


class _AnthropicHTTPError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for idx, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            if isinstance(content, str) and content.strip():
                system_parts.append(content)
            continue

        if role == "user":
            converted.append({
                "role": "user",
                "content": convert_user_content(content),
            })
            continue

        if role == "assistant":
            assistant_blocks: list[dict[str, Any]] = []
            if isinstance(content, str) and content:
                assistant_blocks.append({"type": "text", "text": content})
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                fn = tool_call.get("function") or {}
                name = fn.get("name")
                if not isinstance(name, str) or not name:
                    continue
                raw_arguments = fn.get("arguments")
                parsed_arguments = _parse_tool_arguments(raw_arguments)
                assistant_blocks.append({
                    "type": "tool_use",
                    "id": tool_call.get("id") or f"toolu_{idx}",
                    "name": name,
                    "input": parsed_arguments,
                })
            if assistant_blocks:
                converted.append({"role": "assistant", "content": assistant_blocks})
            continue

        if role == "tool":
            output = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            converted.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id") or f"toolu_{idx}",
                        "content": output,
                    }
                ],
            })

    return "\n\n".join(system_parts), converted


def convert_user_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": ""}]

    blocks: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str):
                blocks.append({"type": "text", "text": text})
        elif item_type == "image_url":
            image_url = item.get("image_url") or {}
            url = image_url.get("url") if isinstance(image_url, dict) else None
            image_block = _convert_image_url(url)
            if image_block is not None:
                blocks.append(image_block)
    return blocks or [{"type": "text", "text": ""}]


def convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = fn.get("parameters")
        converted.append({
            "name": name,
            "description": fn.get("description") or "",
            "input_schema": parameters if isinstance(parameters, dict) else {},
        })
    return converted


def convert_tool_choice(tool_choice: str | dict[str, Any] | None) -> dict[str, Any]:
    if tool_choice is None or tool_choice == "auto":
        return {"type": "auto"}
    if tool_choice == "required":
        return {"type": "any"}
    if tool_choice == "none":
        return {"type": "none"}
    if isinstance(tool_choice, dict):
        tool_type = tool_choice.get("type")
        if tool_type == "function":
            fn = tool_choice.get("function") or {}
            name = fn.get("name")
            if isinstance(name, str) and name:
                return {"type": "tool", "name": name}
        if tool_type in {"auto", "any", "none"}:
            return {"type": tool_type}
        if tool_type == "tool":
            name = tool_choice.get("name")
            if isinstance(name, str) and name:
                return {"type": "tool", "name": name}
    return {"type": "auto"}


def _convert_image_url(url: Any) -> dict[str, Any] | None:
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        mime = "image/png"
        if ";base64" in header:
            mime = header[5:].split(";", 1)[0] or mime
        media_type = _normalize_anthropic_media_type(mime)
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": payload,
            },
        }
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"}:
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": url,
            },
        }
    return None


def _normalize_anthropic_media_type(value: str) -> str:
    if value in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
        return value
    return "image/png"


def _parse_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _response_from_payload(payload: dict[str, Any]) -> LLMResponse:
    content_blocks = payload.get("content")
    text_parts: list[str] = []
    tool_calls: list[ToolCallRequest] = []
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            elif block_type == "tool_use":
                name = block.get("name")
                if not isinstance(name, str) or not name:
                    continue
                tool_calls.append(
                    ToolCallRequest(
                        id=str(block.get("id") or "toolu_0"),
                        name=name,
                        arguments=dict(block.get("input") or {}),
                    )
                )
    usage_payload = payload.get("usage")
    usage: dict[str, int] = {}
    if isinstance(usage_payload, dict):
        for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            value = usage_payload.get(key)
            if isinstance(value, int):
                usage[key] = value
    stop_reason = _map_finish_reason(payload.get("stop_reason"))
    content = "".join(text_parts) if text_parts else None
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=stop_reason,
        usage=usage,
    )


def _map_finish_reason(stop_reason: Any) -> str:
    if not isinstance(stop_reason, str):
        return "stop"
    return _FINISH_REASON_MAP.get(stop_reason, stop_reason)


def _build_headers(*, api_key: str, anthropic_version: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": anthropic_version,
        "content-type": "application/json",
    }


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        return response.text
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
        message = payload.get("message")
        if isinstance(message, str) and message:
            return message
    return response.text
