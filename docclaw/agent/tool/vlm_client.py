"""Shared multimodal VLM helpers for document parsing tools."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any

from docclaw.provider.base import LLMProvider, LLMResponse


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


class VLMClient:
    """Call a multimodal LLM on one image and normalize the response."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.provider = provider
        self.model = model or provider.get_default_model()
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        image_path: Any,
    ) -> tuple[dict[str, Any], LLMResponse]:
        response = await self._chat(
            system_prompt=system_prompt,
            user_payload=user_payload,
            image_path=image_path,
        )
        payload = _parse_json_payload(response.content)
        return payload, response

    async def complete_text(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        image_path: Any,
    ) -> tuple[str, LLMResponse]:
        response = await self._chat(
            system_prompt=system_prompt,
            user_payload=user_payload,
            image_path=image_path,
        )
        return _normalize_text_response(response.content), response

    async def _chat(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        image_path: Any,
    ) -> LLMResponse:
        response = await self.provider.chat(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(user_payload, ensure_ascii=False, indent=2),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url(image_path)},
                        },
                    ],
                },
            ],
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if response.error:
            raise RuntimeError(response.error)
        return response


def image_data_url(path: Any) -> str:
    try:
        from PIL import Image
    except ModuleNotFoundError:  # pragma: no cover - pillow is a runtime dependency here
        Image = None

    if Image is not None and isinstance(path, Image.Image):
        buffer = BytesIO()
        path.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    resolved = Path(path)
    suffix = resolved.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/png")
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _parse_json_payload(content: str | None) -> dict[str, Any]:
    if content is None or not content.strip():
        raise ValueError("empty response")
    candidate = content.strip()
    match = _JSON_BLOCK_RE.search(candidate)
    if match:
        candidate = match.group(1).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end < start:
            raise ValueError("response must contain one JSON object")
        payload = json.loads(candidate[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("response JSON must be an object")
    return payload


def _normalize_text_response(content: str | None) -> str:
    if content is None:
        return ""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    return text
