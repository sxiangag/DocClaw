"""LLM-backed JSON answer generation."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from docclaw.agent.debug import dump_jsonl_from_env
from docclaw.agent.tool.tool import Tool
from docclaw.agent.tool.answer.llm import (
    _merge_usages,
)
from docclaw.agent.utils import (
    Action,
    ActionType,
    Observation,
    RunState,
    page_id_from_index,
)
from docclaw.provider.base import LLMProvider


class LLMJsonAnswerTool(Tool):
    """Use an LLM provider to synthesize a final JSON object answer."""

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

    @property
    def action_type(self) -> ActionType:
        return "answer_json"

    @property
    def description(self) -> str:
        return (
            "Return a JSON object for structured extraction tasks using the "
            "current page image, OCRed page text, or both depending on mode. "
            "Use this for KIE-style tasks, not for ordinary document QA."
        )

    @property
    def target_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "Explicit page set that should be used for JSON answer generation.",
            "properties": {
                "page_ids": {
                    "type": "array",
                    "description": "Page ids to use as JSON answer context.",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["page_ids"],
            "additionalProperties": False,
        }

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "JSON answer-generation parameters.",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": (
                        "Execution mode: image_only uses only page images; "
                        "page_context uses page images plus OCR/page text; "
                        "all runs both candidates and reconciles disagreements."
                    ),
                    "enum": ["image_only", "page_context", "all"],
                    "default": "all",
                },
            },
            "additionalProperties": False,
        }

    async def execute(self, state: RunState, action: Action) -> Observation:
        try:
            mode = _answer_mode(action)
        except ValueError as exc:
            return self.error(action, str(exc))
        page_context = _build_page_context(
            state,
            action,
            require_text=mode != "image_only",
        )
        if not page_context["pages"]:
            if mode == "image_only":
                return self.error(
                    action,
                    "cannot answer in image_only mode without page image context",
                )
            return self.error(action, "cannot answer without page context; call ocr first")
        dump_jsonl_from_env(
            "DOCCLAW_ANSWER_JSON_DEBUG_PATH",
            {
                "kind": "answer_json_input",
                "model": self.model,
                "document_id": state.document.document_id,
                "action_id": action.action_id,
                "task": state.task.to_dict(),
                "target": action.target,
                "parameters": action.parameters,
                "mode": mode,
                "context": _page_context_without_images(page_context),
                "page_debug": _page_context_debug(page_context),
            },
        )
        stage_runs: list[dict[str, Any]] = []
        stage_failures: list[dict[str, Any]] = []
        has_page_images = any(
            isinstance(page.get("image_data_url"), str) and page["image_data_url"]
            for page in page_context["pages"]
            if isinstance(page, dict)
        )

        image_only_run: dict[str, Any] | None = None
        if mode in {"image_only", "all"} and has_page_images:
            image_only_result = await self._generate_candidate(
                self._build_image_only_messages(
                    state,
                    action,
                    page_context=page_context,
                ),
                temperature=self.temperature,
            )
            image_only_run = {
                "stage": "image_only",
                "temperature": self.temperature,
                **image_only_result,
            }
            if image_only_result["success"]:
                stage_runs.append(image_only_run)
            else:
                stage_failures.append(image_only_run)
        elif mode in {"image_only", "all"}:
            stage_failures.append(
                {
                    "stage": "image_only",
                    "success": False,
                    "error": "skipped because no page image is available",
                    "usage": None,
                    "retry_count": 0,
                }
            )

        page_context_run: dict[str, Any] = {"stage": "page_context", "success": False}
        if mode in {"page_context", "all"}:
            page_context_result = await self._generate_candidate(
                self._build_page_context_messages(
                    state,
                    action,
                    page_context=page_context,
                ),
                temperature=self.temperature,
            )
            page_context_run = {
                "stage": "page_context",
                "temperature": self.temperature,
                **page_context_result,
            }
            if page_context_result["success"]:
                stage_runs.append(page_context_run)
            else:
                stage_failures.append(page_context_run)

        if not stage_runs:
            first_failure = next(
                (
                    failure
                    for failure in stage_failures
                    if failure.get("stage") != "image_only"
                    or failure.get("error") != "skipped because no page image is available"
                ),
                stage_failures[0],
            )
            dump_jsonl_from_env(
                "DOCCLAW_ANSWER_JSON_DEBUG_PATH",
                {
                    "kind": "answer_json_error",
                    "model": self.model,
                    "document_id": state.document.document_id,
                    "action_id": action.action_id,
                    "stage": "candidate_generation",
                    "failures": stage_failures,
                    "mode": mode,
                    "page_debug": _page_context_debug(page_context),
                },
            )
            return self.error(action, first_failure["error"])

        if mode == "image_only":
            selected_candidate = (
                image_only_run if image_only_run is not None else stage_runs[0]
            )
        elif mode == "page_context":
            selected_candidate = page_context_run
        else:
            selected_candidate = (
                page_context_run if page_context_run["success"] else stage_runs[0]
            )
        differing_keys: list[str] = []
        reconcile_run: dict[str, Any] | None = None
        if (
            mode == "all"
            and has_page_images
            and image_only_run is not None
            and image_only_run.get("success")
            and page_context_run.get("success")
        ):
            image_payload = _payload_from_answer(image_only_run["answer"])
            page_payload = _payload_from_answer(page_context_run["answer"])
            differing_keys = _differing_keys(image_payload, page_payload)
            if differing_keys:
                image_diff_payload = _payload_slice(image_payload, differing_keys)
                page_diff_payload = _payload_slice(page_payload, differing_keys)
                reconcile_result = await self._generate_candidate(
                    self._build_reconcile_messages(
                        state,
                        action,
                        page_context=page_context,
                        candidate_a=image_diff_payload,
                        candidate_b=page_diff_payload,
                    ),
                    temperature=0.0,
                )
                reconcile_run = {
                    "stage": "reconcile",
                    "temperature": 0.0,
                    "differing_keys": differing_keys,
                    "candidate_a": image_diff_payload,
                    "candidate_b": page_diff_payload,
                    **reconcile_result,
                }
                if reconcile_result["success"]:
                    reconciled_diff_payload = _payload_from_answer(
                        reconcile_result["answer"]
                    )
                    final_payload = dict(page_payload)
                    for key in differing_keys:
                        if key in reconciled_diff_payload:
                            final_payload[key] = reconciled_diff_payload[key]
                    selected_candidate = {
                        **reconcile_run,
                        "parsed_payload": final_payload,
                        "answer": json.dumps(
                            final_payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "reconciled_diff_payload": reconciled_diff_payload,
                    }
                    stage_runs.append(selected_candidate)
                else:
                    stage_failures.append(reconcile_run)

        answer = selected_candidate["answer"]
        usage = _merge_usages([stage_run.get("usage") for stage_run in stage_runs])
        dump_jsonl_from_env(
            "DOCCLAW_ANSWER_JSON_DEBUG_PATH",
            {
                "kind": "answer_json_output",
                "model": self.model,
                "document_id": state.document.document_id,
                "action_id": action.action_id,
                "stage_runs": stage_runs,
                "stage_failures": stage_failures,
                "differing_keys": differing_keys,
                "selected_stage": selected_candidate.get("stage"),
                "mode": mode,
                "raw_content": selected_candidate["raw_content"],
                "parsed_payload": selected_candidate["parsed_payload"],
                "answer": answer,
                "usage": usage,
                "retry_count": sum(
                    int(stage_run.get("retry_count", 0))
                    for stage_run in stage_runs
                ),
                "page_debug": _page_context_debug(page_context),
            },
        )
        return Observation(
            action_id=action.action_id,
            success=True,
            data={
                "answer": answer,
                "page_ids": [item["page_id"] for item in page_context["pages"]],
                "usage": usage,
                "source": "llm",
            },
            message=(
                f"Prepared JSON answer from {len(page_context['pages'])} page context item(s) "
                f"({len(answer)} chars)."
            ),
        )

    async def _generate_candidate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
    ) -> dict[str, Any]:
        parse_failures: list[dict[str, Any]] = []
        for attempt_index in range(2):
            response = await self.provider.chat(
                messages,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
            )
            if response.error:
                return {
                    "success": False,
                    "error": response.error,
                    "attempts": parse_failures,
                    "usage": response.usage,
                    "retry_count": len(parse_failures),
                }
            try:
                payload = self._parse_payload(response.content)
                answer = self._answer_from_payload(payload)
                return {
                    "success": True,
                    "raw_content": response.content,
                    "parsed_payload": payload,
                    "answer": answer,
                    "usage": response.usage,
                    "retry_count": len(parse_failures),
                }
            except Exception as exc:
                parse_failures.append(
                    {
                        "attempt_index": attempt_index,
                        "raw_content": response.content,
                        "parse_error": str(exc),
                        "usage": response.usage,
                    }
                )
        return {
            "success": False,
            "error": f"invalid JSON answer response: {parse_failures[-1]['parse_error']}",
            "attempts": parse_failures,
            "usage": parse_failures[-1].get("usage"),
            "retry_count": len(parse_failures),
        }

    def _build_image_only_messages(
        self,
        state: RunState,
        action: Action,
        *,
        page_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        system_prompt = (
            "You are the DocClaw JSON answer tool.\n"
            "Your job is to produce the final user-facing JSON object for the structured extraction question.\n"
            "Use only the provided page image to answer.\n"
            "Return only the final JSON object itself. Do not wrap it inside an answer field.\n"
            "JSON rules:\n"
            "- The response must be one valid JSON object.\n"
            "- Use exactly the field names requested by the task prompt.\n"
            "- Do not add extra keys that were not requested.\n"
            "- Preserve the complete value information shown in the document.\n"
        )
        user_payload = json.dumps(
            {"question": state.task.prompt},
            ensure_ascii=False,
            indent=2,
        )
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": _multimodal_user_content(user_payload, page_context),
            },
        ]

    def _build_page_context_messages(
        self,
        state: RunState,
        action: Action,
        *,
        page_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        system_prompt = (
            "You are the DocClaw JSON answer tool.\n"
            "Your job is to produce the final user-facing JSON object for the structured extraction question.\n"
            "Use the provided page image and OCRed page text together, and answer exactly what the task asks.\n"
            "Treat page text as a reading aid, not as ground truth.\n"
            "Use the page image to resolve layout, ambiguous OCR, and visually obvious OCR mistakes.\n"
            "Return only the final JSON object itself. Do not wrap it inside an answer field.\n"
            "JSON rules:\n"
            "- The response must be one valid JSON object.\n"
            "- Use exactly the field names requested by the task prompt.\n"
            "- Do not add extra keys that were not requested.\n"
            "- Preserve the complete value information shown in the document.\n"
        )
        user_payload = json.dumps(
            {
                "question": state.task.prompt,
                "pages": [
                    {
                        "page_id": page.get("page_id"),
                        "text": page.get("text"),
                    }
                    for page in page_context["pages"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": _multimodal_user_content(user_payload, page_context),
            },
        ]

    def _build_reconcile_messages(
        self,
        state: RunState,
        action: Action,
        *,
        page_context: dict[str, Any],
        candidate_a: dict[str, Any],
        candidate_b: dict[str, Any],
    ) -> list[dict[str, Any]]:
        system_prompt = (
            "You are the DocClaw KIE reconciliation tool.\n"
            "The candidates contain extracted values for the same document fields.\n"
            "For each provided key, choose either Candidate A's value or Candidate B's value by reading the page image.\n"
            "Return only one valid JSON object with exactly the provided keys.\n"
            "Do not add keys. Do not invent a third value.\n"
        )
        user_payload = json.dumps(
            {
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
            },
            ensure_ascii=False,
            indent=2,
        )
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": _multimodal_user_content(user_payload, page_context),
            },
        ]

    @staticmethod
    def _parse_payload(content: str | None) -> dict[str, Any]:
        if content is None or not content.strip():
            raise ValueError("empty response")
        candidate = content.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start < 0 or end < start:
                raise ValueError("response must be valid JSON")
            payload = json.loads(candidate[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("response JSON must be an object")
        return payload

    @staticmethod
    def _answer_from_payload(payload: dict[str, Any]) -> str:
        final_payload = payload
        if set(payload).issubset({"answer", "reason"}):
            answer = payload.get("answer")
            if isinstance(answer, dict):
                final_payload = answer
            elif isinstance(answer, str):
                try:
                    parsed_answer = json.loads(answer)
                except json.JSONDecodeError:
                    parsed_answer = None
                if isinstance(parsed_answer, dict):
                    final_payload = parsed_answer
        if not final_payload:
            raise ValueError("response must contain a non-empty JSON object")
        return json.dumps(final_payload, ensure_ascii=False, separators=(",", ":"))

    def update_state(
        self,
        state: RunState,
        action: Action,
        observation: Observation,
    ) -> None:
        answer = observation.data.get("answer")
        if isinstance(answer, str) and answer:
            state.final_answer = answer
            state.status = "completed"


def _answer_mode(action: Action) -> str:
    parameters = action.parameters if isinstance(action.parameters, dict) else {}
    mode = parameters.get("mode", "all")
    if mode not in {"image_only", "page_context", "all"}:
        raise ValueError(f"unsupported JSON answer mode: {mode!r}")
    return mode


def _build_page_context(
    state: RunState,
    action: Action,
    *,
    require_text: bool = True,
) -> dict[str, Any]:
    page_indices = _target_page_indices(state, action)
    pages: list[dict[str, Any]] = []
    for page in state.document.pages:
        if page_indices is not None and page.page_index not in page_indices:
            continue
        text = _page_markdown(page)
        text_format = "markdown"
        if not isinstance(text, str) or not text.strip():
            text = page.ocr_text
            text_format = "plain_text"
        page_item: dict[str, Any] = {
            "page_id": page_id_from_index(page.page_index, document=state.document),
            "page_index": page.page_index,
        }
        if isinstance(text, str) and text.strip():
            page_item["text_format"] = text_format
            page_item["text"] = text
        if isinstance(page.image_path, str) and page.image_path.strip():
            page_item["image_path"] = page.image_path
            image_data_url = _safe_image_data_url(Path(page.image_path))
            if image_data_url is not None:
                page_item["image_data_url"] = image_data_url
        if require_text and "text" not in page_item:
            continue
        if not require_text and "text" not in page_item and "image_data_url" not in page_item:
            continue
        pages.append(page_item)
    return {
        "document": {
            "document_id": state.document.document_id,
            "page_ids": [
                page_id_from_index(page.page_index, document=state.document)
                for page in state.document.pages
            ],
        },
        "pages": pages,
    }


def _target_page_indices(state: RunState, action: Action) -> set[int] | None:
    target = action.target if isinstance(action.target, dict) else {}
    raw_indices = target.get("page_indices")
    if isinstance(raw_indices, list):
        return {
            item
            for item in raw_indices
            if isinstance(item, int) and not isinstance(item, bool)
        }
    raw_index = target.get("page_index")
    if isinstance(raw_index, int) and not isinstance(raw_index, bool):
        return {raw_index}
    raw_ids = target.get("page_ids")
    if isinstance(raw_ids, list):
        values: set[int] = set()
        for page_id in raw_ids:
            if not isinstance(page_id, str):
                continue
            for page in state.document.pages:
                if page_id_from_index(page.page_index, document=state.document) == page_id:
                    values.add(page.page_index)
                    break
        return values
    return None


def _page_context_debug(page_context: dict[str, Any]) -> dict[str, list[int]]:
    pages = page_context.get("pages")
    indices: list[int] = []
    if isinstance(pages, list):
        for page in pages:
            if isinstance(page, dict) and isinstance(page.get("page_index"), int):
                indices.append(page["page_index"])
    return {"context_page_indices": indices}


def _page_context_without_images(page_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "document": page_context.get("document"),
        "pages": [
            {
                key: value
                for key, value in page.items()
                if key != "image_data_url"
            }
            for page in page_context.get("pages", [])
            if isinstance(page, dict)
        ],
    }


def _multimodal_user_content(
    text: str,
    page_context: dict[str, Any],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for page in page_context.get("pages", []):
        if not isinstance(page, dict):
            continue
        image_data_url = page.get("image_data_url")
        if isinstance(image_data_url, str) and image_data_url:
            content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    return content


def _payload_from_answer(answer: str) -> dict[str, Any]:
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _differing_keys(
    left: dict[str, Any],
    right: dict[str, Any],
) -> list[str]:
    keys = sorted(set(left) | set(right))
    return [
        key
        for key in keys
        if _value_signature(left.get(key)) != _value_signature(right.get(key))
    ]


def _payload_slice(payload: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: payload.get(key, "") for key in keys}


def _value_signature(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _page_markdown(page: Any) -> str | None:
    post_layout = page.metadata.get("post_layout")
    if not isinstance(post_layout, dict) or not page.regions:
        return None
    try:
        from docclaw.exporter import export_page_markdown

        markdown = export_page_markdown(page, pretty=False)
    except Exception:
        return None
    return markdown if isinstance(markdown, str) and markdown.strip() else None


def _safe_image_data_url(path: Path) -> str | None:
    try:
        suffix = path.expanduser().suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix, "image/png")
        encoded = path.expanduser().read_bytes()
    except Exception:
        return None
    return f"data:{mime};base64,{base64.b64encode(encoded).decode('ascii')}"
