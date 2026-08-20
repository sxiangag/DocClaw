"""MinerU API-backed OCR candidate recognition."""

from __future__ import annotations

import io
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Sequence
import zipfile

import httpx


class MinerU2_5ProApiRecognizer:
    """Call a running mineru-api service and return plain-text results."""

    DEFAULT_BACKEND = "vlm-auto-engine"
    DEFAULT_PARSE_METHOD = "auto"
    DEFAULT_LANGUAGE = "ch"
    DEFAULT_TIMEOUT_SECONDS = 600.0
    _API_URL_ENVS = ("DOCCLAW_MINERU_API_URL", "MINERU_API_URL")
    _TIMEOUT_ENV = "DOCCLAW_MINERU_API_TIMEOUT"

    def __init__(
        self,
        *,
        api_url: str | None = None,
        backend: str = DEFAULT_BACKEND,
        parse_method: str = DEFAULT_PARSE_METHOD,
        language: str = DEFAULT_LANGUAGE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_url = _normalize_base_url(api_url or _api_url_from_env())
        self.backend = backend
        self.parse_method = parse_method
        self.language = language
        self.timeout_seconds = _timeout_seconds_from_env(timeout_seconds)

    def recognize_batch(self, image_paths: Sequence[Path]) -> list[dict[str, Any]]:
        if not image_paths:
            return []
        if not self.api_url:
            env_names = ", ".join(self._API_URL_ENVS)
            raise RuntimeError(
                f"MinerU API URL is not configured; set one of: {env_names}"
            )

        path_list = [Path(path).expanduser().resolve() for path in image_paths]
        missing_paths = [str(path) for path in path_list if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                f"MinerU API input image(s) do not exist: {', '.join(missing_paths)}"
            )

        form_data = _build_form_data(
            language=self.language,
            backend=self.backend,
            parse_method=self.parse_method,
        )
        upload_names = [f"alt_candidate_{index:04d}{path.suffix or '.png'}" for index, path in enumerate(path_list)]
        timeout = httpx.Timeout(self.timeout_seconds, connect=30.0)

        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            files = []
            handles = []
            try:
                for path, upload_name in zip(path_list, upload_names):
                    mime_type = mimetypes.guess_type(upload_name)[0] or "application/octet-stream"
                    handle = path.open("rb")
                    handles.append(handle)
                    files.append(("files", (upload_name, handle, mime_type)))
                response = client.post(
                    f"{self.api_url}/file_parse",
                    data=form_data,
                    files=files,
                )
            finally:
                for handle in handles:
                    handle.close()

        if response.status_code != 200:
            raise RuntimeError(
                f"MinerU API file_parse failed: {response.status_code} {_response_detail(response)}"
            )

        content_type = response.headers.get("content-type", "").lower()
        if "application/zip" not in content_type and not response.content.startswith(b"PK"):
            raise RuntimeError(
                f"MinerU API file_parse returned unexpected content-type: {content_type or '<missing>'}"
            )

        return _parse_zip_results(response.content, upload_names)


def _api_url_from_env() -> str | None:
    for env_name in MinerU2_5ProApiRecognizer._API_URL_ENVS:
        value = os.environ.get(env_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized.rstrip("/")


def _timeout_seconds_from_env(default: float) -> float:
    raw_value = os.environ.get(MinerU2_5ProApiRecognizer._TIMEOUT_ENV)
    if raw_value is None:
        return default
    try:
        resolved = float(raw_value)
    except ValueError:
        return default
    return resolved if resolved > 0 else default


def _build_form_data(
    *,
    language: str,
    backend: str,
    parse_method: str,
) -> dict[str, str]:
    return {
        "lang_list": language,
        "backend": backend,
        "parse_method": parse_method,
        "formula_enable": "true",
        "table_enable": "true",
        "image_analysis": "false",
        "return_md": "false",
        "return_middle_json": "false",
        "return_model_output": "false",
        "return_content_list": "true",
        "return_images": "false",
        "response_format_zip": "true",
        "return_original_file": "false",
        "client_side_output_generation": "false",
        "start_page_id": "0",
        "end_page_id": "99999",
    }


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text or "<empty response body>"
    if isinstance(payload, dict):
        message = payload.get("message")
        error = payload.get("error")
        if isinstance(message, str) and message.strip():
            if isinstance(error, str) and error.strip():
                return f"{message}: {error}"
            return message
    return json.dumps(payload, ensure_ascii=False)


def _parse_zip_results(zip_bytes: bytes, upload_names: Sequence[str]) -> list[dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = archive.namelist()
        results: list[dict[str, Any]] = []
        for upload_name in upload_names:
            stem = Path(upload_name).stem
            content_list_path = _find_content_list_member(names, stem)
            content_list: Any = []
            if content_list_path is not None:
                with archive.open(content_list_path) as handle:
                    content_list = json.load(handle)
            results.append(
                {
                    "text": _content_list_to_text(content_list),
                    "confidence": None,
                }
            )
        return results


def _find_content_list_member(names: Sequence[str], stem: str) -> str | None:
    preferred_suffixes = (
        f"/{stem}_content_list.json",
        f"/{stem}_content_list_v2.json",
    )
    for suffix in preferred_suffixes:
        for name in names:
            if name.endswith(suffix):
                return name
    return None


def _content_list_to_text(content_list: Any) -> str:
    if not isinstance(content_list, list):
        return ""
    parts: list[str] = []
    for item in content_list:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
            continue
        if item.get("type") == "table":
            table_body = item.get("table_body")
            if isinstance(table_body, str) and table_body.strip():
                parts.append(table_body.strip())
    deduped: list[str] = []
    for part in parts:
        if deduped and deduped[-1] == part:
            continue
        deduped.append(part)
    return "\n".join(deduped)
