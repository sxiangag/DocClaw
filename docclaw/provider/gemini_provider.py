"""Google Gemini generateContent provider for DocClaw."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import json
from typing import Any
from urllib.parse import urlparse

import httpx

from docclaw.provider.base import LLMProvider, LLMResponse, ToolCallRequest

DEFAULT_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "error",
    "RECITATION": "error",
    "OTHER": "error",
}


class GeminiProvider(LLMProvider):
    """Use the Gemini generateContent API."""

    def __init__(
        self,
        default_model: str = DEFAULT_GEMINI_MODEL,
        *,
        api_key: str,
        url_template: str = DEFAULT_GEMINI_URL,
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
        self.url_template = url_template
        self.verify_ssl = verify_ssl
        self.allow_insecure_ssl_fallback = allow_insecure_ssl_fallback
        self.request_timeout = request_timeout
        self._requester = requester or self._request_gemini

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
            system_prompt, contents = convert_messages(messages)
            body: dict[str, Any] = {
                "contents": contents,
                "generationConfig": {},
            }
            if max_tokens is not None:
                body["generationConfig"]["maxOutputTokens"] = max_tokens
            if system_prompt:
                body["systemInstruction"] = {
                    "parts": [{"text": system_prompt}],
                }
            if temperature != 0.0:
                body["generationConfig"]["temperature"] = temperature
            if not body["generationConfig"]:
                body.pop("generationConfig")
            if tools:
                body["tools"] = [{"functionDeclarations": convert_tools(tools)}]
                body["toolConfig"] = {
                    "functionCallingConfig": convert_tool_choice(tool_choice, tools),
                }

            url = self.url_template.format(model=resolved_model)
            headers = _build_headers(api_key=self.api_key)
            requester = self._requester or self._request_gemini
            try:
                payload = await requester(url, headers, body, self.verify_ssl)
            except Exception as exc:
                if not self._should_retry_insecure(exc):
                    raise
                payload = await requester(url, headers, body, False)
            return _response_from_payload(payload)
        except Exception as exc:
            message = f"Error calling Gemini: {_format_provider_error(exc)}"
            retry_after = getattr(exc, "retry_after", None) or self._extract_retry_after(message)
            return LLMResponse(
                content=message,
                finish_reason="error",
                retry_after=retry_after,
                error=message,
            )

    def get_default_model(self) -> str:
        return self.default_model

    async def _request_gemini(
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
            raise _GeminiHTTPError(
                f"{response.status_code} {response.reason_phrase}: {detail}",
                retry_after=retry_after,
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Gemini response must be a JSON object")
        return payload

    def _should_retry_insecure(self, exc: Exception) -> bool:
        if not self.allow_insecure_ssl_fallback or not self.verify_ssl:
            return False
        return "CERTIFICATE_VERIFY_FAILED" in str(exc)


class _GeminiHTTPError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _format_provider_error(exc: Exception) -> str:
    detail = str(exc).strip()
    if detail:
        return detail
    return exc.__class__.__name__


def convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for idx, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            if isinstance(content, str) and content.strip():
                system_parts.append(content)
            continue

        if role == "user":
            contents.append({
                "role": "user",
                "parts": convert_user_parts(content),
            })
            continue

        if role == "assistant":
            parts: list[dict[str, Any]] = []
            if isinstance(content, str) and content:
                parts.append({"text": content})
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                fn = tool_call.get("function") or {}
                name = fn.get("name")
                if not isinstance(name, str) or not name:
                    continue
                parts.append({
                    "functionCall": {
                        "id": tool_call.get("id") or f"call_{idx}",
                        "name": name,
                        "args": _parse_tool_arguments(fn.get("arguments")),
                    }
                })
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue

        if role == "tool":
            contents.append({
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "id": message.get("tool_call_id") or f"call_{idx}",
                            "name": message.get("name") or "",
                            "response": _normalize_tool_response_content(content),
                        }
                    }
                ],
            })

    return "\n\n".join(system_parts), contents


def convert_user_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": ""}]

    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str):
                parts.append({"text": text})
        elif item_type == "image_url":
            image_url = item.get("image_url") or {}
            url = image_url.get("url") if isinstance(image_url, dict) else None
            image_part = _convert_image_url(url)
            if image_part is not None:
                parts.append(image_part)
    return parts or [{"text": ""}]


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
            "parameters": _sanitize_gemini_schema(parameters if isinstance(parameters, dict) else {}),
        })
    return converted


def convert_tool_choice(
    tool_choice: str | dict[str, Any] | None,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    allowed_names = [item["function"]["name"] for item in tools if item.get("type") == "function" and isinstance((item.get("function") or {}).get("name"), str)]
    if tool_choice is None or tool_choice == "auto":
        return {"mode": "AUTO"}
    if tool_choice == "required":
        config: dict[str, Any] = {"mode": "ANY"}
        if allowed_names:
            config["allowedFunctionNames"] = allowed_names
        return config
    if tool_choice == "none":
        return {"mode": "NONE"}
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        name = fn.get("name")
        if tool_choice.get("type") == "function" and isinstance(name, str) and name:
            return {"mode": "ANY", "allowedFunctionNames": [name]}
    return {"mode": "AUTO"}


_ALLOWED_SCHEMA_KEYS = {
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "items",
    "properties",
    "required",
}


def _sanitize_gemini_schema(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_sanitize_gemini_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    sanitized: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _ALLOWED_SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            sanitized[key] = {
                prop_name: _sanitize_gemini_schema(prop_schema)
                for prop_name, prop_schema in value.items()
                if isinstance(prop_name, str)
            }
            continue
        if key == "items":
            sanitized[key] = _sanitize_gemini_schema(value)
            continue
        if key == "required" and isinstance(value, list):
            sanitized[key] = [item for item in value if isinstance(item, str)]
            continue
        if key == "enum" and isinstance(value, list):
            sanitized[key] = [item for item in value if isinstance(item, (str, int, float, bool))]
            continue
        sanitized[key] = value

    return sanitized


def _convert_image_url(url: Any) -> dict[str, Any] | None:
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        mime = "image/png"
        if ";base64" in header:
            mime = header[5:].split(";", 1)[0] or mime
        return {
            "inline_data": {
                "mime_type": _normalize_gemini_media_type(mime),
                "data": payload,
            }
        }
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"}:
        return {
            "file_data": {
                "mime_type": "image/png",
                "file_uri": url,
            }
        }
    return None


def _normalize_gemini_media_type(value: str) -> str:
    if value in {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}:
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


def _normalize_tool_response_content(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        return {"content": content}
    return {"content": json.dumps(content, ensure_ascii=False)}


def _response_from_payload(payload: dict[str, Any]) -> LLMResponse:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Gemini response missing candidates")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise RuntimeError("Gemini candidate must be an object")
    content = candidate.get("content") or {}
    parts = content.get("parts") if isinstance(content, dict) else None
    text_parts: list[str] = []
    tool_calls: list[ToolCallRequest] = []
    if isinstance(parts, list):
        for idx, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                text_parts.append(text)
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                name = function_call.get("name")
                if not isinstance(name, str) or not name:
                    continue
                tool_calls.append(
                    ToolCallRequest(
                        id=str(function_call.get("id") or f"call_{idx}"),
                        name=name,
                        arguments=dict(function_call.get("args") or {}),
                    )
                )
    usage: dict[str, int] = {}
    usage_payload = payload.get("usageMetadata")
    if isinstance(usage_payload, dict):
        for source_key, target_key in (
            ("promptTokenCount", "input_tokens"),
            ("candidatesTokenCount", "output_tokens"),
            ("totalTokenCount", "total_tokens"),
        ):
            value = usage_payload.get(source_key)
            if isinstance(value, int):
                usage[target_key] = value
    finish_reason = _map_finish_reason(candidate.get("finishReason"))
    response_text = "".join(text_parts) if text_parts else None
    return LLMResponse(
        content=response_text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
    )


def _map_finish_reason(value: Any) -> str:
    if not isinstance(value, str):
        return "stop"
    return _FINISH_REASON_MAP.get(value, value.lower())


def _build_headers(*, api_key: str) -> dict[str, str]:
    return {
        "x-goog-api-key": api_key,
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
