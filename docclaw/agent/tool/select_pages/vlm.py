"""VLM-backed relevant-page selection implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docclaw.agent.debug import dump_jsonl_from_env
from docclaw.exporter import export_page_markdown
from docclaw.agent.tool.figure.vlm import (
    VLMFigureTool,
    _image_data_url,
)
from docclaw.agent.tool.select_pages.select_pages import SelectPagesTool
from docclaw.agent.utils import Action, RunState, page_id_from_index


COARSE_SELECT_PAGES_CHUNK_SIZE = 50


class VLMSelectPagesTool(SelectPagesTool, VLMFigureTool):
    """Select relevant pages with a multimodal language model."""

    async def select_pages(
        self,
        state: RunState,
        action: Action,
        *,
        targets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        question = action.parameters.get("question")
        mode = str(action.parameters.get("mode") or "").strip().lower()
        resolved_targets = [dict(target) for target in targets]
        page_image_paths = [
            Path(str(target["artifact_path"])).expanduser()
            for target in resolved_targets
        ]
        input_page_debug = _select_pages_input_page_debug(resolved_targets)
        selection_messages, selection_payload = self._build_selection_messages(
            state,
            action,
            resolved_targets=resolved_targets,
            page_image_paths=page_image_paths,
        )
        dump_jsonl_from_env(
            "DOCCLAW_SELECT_PAGES_DEBUG_PATH",
            {
                "kind": "select_pages_input",
                "model": self.model,
                "document_id": state.document.document_id,
                "action_id": action.action_id,
                "task": state.task.to_dict(),
                "question": question,
                "mode": mode,
                "targets": resolved_targets,
                "user_payload": selection_payload,
                "page_debug": input_page_debug,
            },
        )

        try:
            if mode == "coarse" and len(resolved_targets) > 1:
                union_page_ids: list[str] = []
                semantic_results: list[dict[str, Any]] = []
                target_chunks = _chunk_targets(
                    resolved_targets,
                    chunk_size=COARSE_SELECT_PAGES_CHUNK_SIZE,
                )
                for chunk_index, chunk_targets in enumerate(target_chunks):
                    chunk_image_paths = [
                        Path(str(target["artifact_path"])).expanduser()
                        for target in chunk_targets
                    ]
                    chunk_messages, _ = (
                        (selection_messages, selection_payload)
                        if len(target_chunks) == 1
                        else self._build_selection_messages(
                            state,
                            action,
                            resolved_targets=chunk_targets,
                            page_image_paths=chunk_image_paths,
                        )
                    )
                    chunk_normalized, chunk_semantic_results = await self._run_coarse_semantic_union(
                        state,
                        action,
                        chunk_targets=chunk_targets,
                        chunk_messages=chunk_messages,
                        chunk_index=chunk_index,
                        chunk_count=len(target_chunks),
                    )
                    for page_id in chunk_normalized["selected_page_ids"]:
                        if page_id not in union_page_ids:
                            union_page_ids.append(page_id)
                    semantic_results.extend(chunk_semantic_results)
                normalized = {
                    "selected_page_ids": union_page_ids,
                    "reason": "",
                    "confidence": None,
                    "metadata": {
                        "chunk_count": len(target_chunks),
                        "chunk_size": COARSE_SELECT_PAGES_CHUNK_SIZE,
                    },
                    "usage": semantic_results[-1]["usage"] if semantic_results else {},
                    "source": "llm",
                }
                dump_jsonl_from_env(
                    "DOCCLAW_SELECT_PAGES_DEBUG_PATH",
                    {
                        "kind": "select_pages_output",
                        "model": self.model,
                        "document_id": state.document.document_id,
                        "action_id": action.action_id,
                        "mode": mode,
                        "normalized_result": normalized,
                        "page_debug": _select_pages_output_page_debug(
                            normalized,
                            targets=resolved_targets,
                            document=state.document,
                        ),
                        "chunk_count": len(target_chunks),
                        "chunk_size": COARSE_SELECT_PAGES_CHUNK_SIZE,
                        "semantic_attempt_count": len(semantic_results),
                        "semantic_attempts": semantic_results,
                    },
                )
                return {
                    "success": True,
                    "payload": normalized,
                    "artifacts": [],
                    "error": None,
                }

            response, retries, payload = await self._call_json_with_retry(
                messages=selection_messages,
                parse_fn=self._parse_selection_payload,
            )
            normalized = _normalize_selection_payload(
                payload,
                targets=resolved_targets,
                document=state.document,
                mode=mode,
            )
        except Exception as exc:
            dump_jsonl_from_env(
                "DOCCLAW_SELECT_PAGES_DEBUG_PATH",
                {
                    "kind": "select_pages_error",
                    "model": self.model,
                    "document_id": state.document.document_id,
                    "action_id": action.action_id,
                    "mode": mode,
                    "parse_error": str(exc),
                },
            )
            return {"success": False, "error": f"invalid select_pages response: {exc}"}

        normalized["usage"] = response.usage
        normalized["source"] = "llm"
        dump_jsonl_from_env(
            "DOCCLAW_SELECT_PAGES_DEBUG_PATH",
            {
                "kind": "select_pages_output",
                "model": self.model,
                "document_id": state.document.document_id,
                "action_id": action.action_id,
                "mode": mode,
                "raw_content": response.content,
                "parsed_payload": payload,
                "normalized_result": normalized,
                "page_debug": _select_pages_output_page_debug(
                    normalized,
                    targets=resolved_targets,
                    document=state.document,
                ),
                "usage": response.usage,
                "retry_count": len(retries),
            },
        )
        return {
            "success": True,
            "payload": normalized,
            "artifacts": [],
            "error": None,
        }

    async def _run_coarse_semantic_union(
        self,
        state: RunState,
        action: Action,
        *,
        chunk_targets: list[dict[str, Any]],
        chunk_messages: list[dict[str, Any]],
        chunk_index: int,
        chunk_count: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        semantic_temperatures = [
            0.6,
            0.6,
        ]
        union_page_ids: list[str] = []
        semantic_results: list[dict[str, Any]] = []
        for semantic_attempt_index, temperature in enumerate(semantic_temperatures):
            response, retries, payload = await self._call_json_with_retry(
                messages=chunk_messages,
                parse_fn=self._parse_selection_payload,
                temperature=temperature,
            )
            normalized = _normalize_selection_payload(
                payload,
                targets=chunk_targets,
                document=state.document,
                mode="coarse",
            )
            for page_id in normalized["selected_page_ids"]:
                if page_id not in union_page_ids:
                    union_page_ids.append(page_id)
            semantic_results.append(
                {
                    "chunk_index": chunk_index,
                    "chunk_count": chunk_count,
                    "chunk_target_count": len(chunk_targets),
                    "semantic_attempt_index": semantic_attempt_index,
                    "temperature": temperature,
                    "raw_content": response.content,
                    "parsed_payload": payload,
                    "normalized_result": normalized,
                    "page_debug": _select_pages_output_page_debug(
                        normalized,
                        targets=chunk_targets,
                        document=state.document,
                    ),
                    "usage": response.usage,
                    "retry_count": len(retries),
                }
            )
        return {
            "selected_page_ids": union_page_ids,
            "reason": "",
            "confidence": None,
            "metadata": {},
        }, semantic_results

    def _build_selection_messages(
        self,
        state: RunState,
        action: Action,
        *,
        resolved_targets: list[dict[str, Any]],
        page_image_paths: list[Path],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        question = action.parameters.get("question")
        mode = str(action.parameters.get("mode") or "").strip().lower()
        candidate_page_ids = [
            page_id_from_index(page_index, document=state.document)
            for page_index in (
                target.get("page_index")
                for target in resolved_targets
                if isinstance(target.get("page_index"), int)
            )
        ]
        if mode == "refine":
            system_prompt = (
                "You are given a question and a set of candidate document pages.\n"
                "Select which candidate pages should still be kept as relevant to answering the question.\n"
                "Use both the page image and the provided page text when available.\n"
                "This is a higher-precision filtering step over pages that were already kept by an earlier coarse pass.\n"
                "Keep every page that still looks relevant after considering both the visual content and the provided page text.\n"
                "If the relevant content may continue across pages, keep the continuation pages too, not just the page where the topic or section first appears.\n"
                "For numeric or aggregation questions, keep pages that may contain required calculation inputs or comparison values.\n"
                "Do not answer the question itself, and use only the provided candidate pages.\n"
                "A selected_page_ids list must contain at least one page id and may contain multiple page ids.\n"
                "Return only a JSON object, with no prose or markdown before or after it. Ensure all string values are valid JSON strings.\n"
                "Return JSON with this shape:\n"
                "{\n"
                '  "selected_page_ids": ["page_A","page_B"],\n'
                '  "reason": "short explanation of why these pages should be kept"\n'
                "}\n"
            )
        else:
            system_prompt = (
                "You are given a question and a set of candidate document pages.\n"
                "Select which candidate pages should be kept as relevant to answering the question.\n"
                "This is a coarse high-recall filtering step.\n"
                "Keep every candidate page that is visually relevant or plausibly relevant to the question.\n"
                "If the relevant content may continue across pages, keep the continuation pages too, not just the page where the topic or section first appears.\n"
                "For numeric or aggregation questions, keep pages that may contain required calculation inputs or comparison values.\n"
                "When resolving mentioned page number(s), use only page labels in the page header, footer, or crop. "
                "If no reliable page number is visible, use the order of candidate_page_ids in the input as a fallback.\n"
                "Do not answer the question itself, and use only the provided candidate pages.\n"
                "A selected_page_ids list must contain at least one page id and may contain multiple page ids.\n"
                "Return only a JSON object, with no prose or markdown before or after it. Ensure all string values are valid JSON strings.\n"
                "Return JSON with this shape:\n"
                "{\n"
                '  "selected_page_ids": ["page_A","page_B"],\n'
                '  "reason": "short explanation of why these pages should be kept"\n'
                "}\n"
            )
        user_payload = {
            "question": question,
            "mode": mode,
            "candidate_page_ids": candidate_page_ids,
        }
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ]
        for target, image_path in zip(resolved_targets, page_image_paths, strict=True):
            page_index = target.get("page_index")
            assert isinstance(page_index, int)
            page = state.get_page(page_index)
            user_content.append(
                {
                    "type": "text",
                    "text": f"Candidate page_id {page_id_from_index(page_index, document=state.document)}:",
                }
            )
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image_path)},
                }
            )
            if mode == "refine":
                text, text_format = _best_available_page_text(page)
                if text is not None:
                    user_content.append(
                        {
                            "type": "text",
                            "text": (
                                f"Candidate page_id {page_id_from_index(page_index, document=state.document)} {text_format}:\n"
                                f"{text}"
                            ),
                        }
                    )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ], user_payload

    @classmethod
    def _parse_selection_payload(cls, content: str | None) -> dict[str, Any]:
        payload = cls._parse_payload(content)
        selected_page_ids = payload.get("selected_page_ids")
        if not isinstance(selected_page_ids, list):
            raise ValueError("selected_page_ids must be a list")
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        return payload


def _normalize_selection_payload(
    payload: dict[str, Any],
    *,
    targets: list[dict[str, Any]],
    document: Any,
    mode: str | None = None,
) -> dict[str, Any]:
    allowed_page_ids = {
        page_id_from_index(target["page_index"], document=document)
        for target in targets
        if isinstance(target.get("page_index"), int)
    }
    selected_page_ids: list[str] = []
    for item in payload.get("selected_page_ids") or []:
        if not isinstance(item, str):
            continue
        page_id = item.strip()
        if page_id not in allowed_page_ids or page_id in selected_page_ids:
            continue
        selected_page_ids.append(page_id)
    if not selected_page_ids and mode == "refine" and len(allowed_page_ids) == 1:
        selected_page_ids = list(allowed_page_ids)

    reason = str(payload.get("reason") or "").strip()
    if not reason:
        raise ValueError("response JSON must contain a non-empty reason")

    confidence = payload.get("confidence")
    normalized_confidence = None
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        normalized_confidence = float(confidence)
    elif isinstance(confidence, str) and confidence.strip():
        normalized_confidence = float(confidence)

    metadata = payload.get("metadata") or {}
    return {
        "selected_page_ids": selected_page_ids,
        "reason": reason,
        "confidence": normalized_confidence,
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


def _chunk_targets(
    targets: list[dict[str, Any]],
    *,
    chunk_size: int,
) -> list[list[dict[str, Any]]]:
    if chunk_size <= 0 or len(targets) <= chunk_size:
        return [targets]
    return [
        targets[start : start + chunk_size]
        for start in range(0, len(targets), chunk_size)
    ]


def _select_pages_input_page_debug(targets: list[dict[str, Any]]) -> dict[str, list[int]]:
    try:
        candidate_page_indices = _page_indices_from_targets(targets)
        return {
            "candidate_page_indices": candidate_page_indices,
            "target_page_indices": candidate_page_indices,
        }
    except Exception:
        return {
            "candidate_page_indices": [],
            "target_page_indices": [],
        }


def _select_pages_output_page_debug(
    payload: dict[str, Any],
    *,
    targets: list[dict[str, Any]],
    document: Any,
) -> dict[str, list[int]]:
    try:
        return {
            **_select_pages_input_page_debug(targets),
            "selected_page_indices": _selected_page_indices_from_payload(
                payload,
                targets=targets,
                document=document,
            ),
        }
    except Exception:
        return {
            **_select_pages_input_page_debug(targets),
            "selected_page_indices": [],
        }


def _page_indices_from_targets(targets: list[dict[str, Any]]) -> list[int]:
    page_indices: list[int] = []
    seen: set[int] = set()
    for target in targets:
        page_index = target.get("page_index")
        if not isinstance(page_index, int) or page_index in seen:
            continue
        page_indices.append(page_index)
        seen.add(page_index)
    return page_indices


def _selected_page_indices_from_payload(
    payload: dict[str, Any],
    *,
    targets: list[dict[str, Any]],
    document: Any,
) -> list[int]:
    page_index_by_id = {
        page_id_from_index(target["page_index"], document=document): target["page_index"]
        for target in targets
        if isinstance(target.get("page_index"), int)
    }
    selected_page_indices: list[int] = []
    seen: set[int] = set()
    selected_page_ids = payload.get("selected_page_ids")
    if not isinstance(selected_page_ids, list):
        return selected_page_indices
    for item in selected_page_ids:
        if not isinstance(item, str):
            continue
        page_index = page_index_by_id.get(item.strip())
        if page_index is None or page_index in seen:
            continue
        selected_page_indices.append(page_index)
        seen.add(page_index)
    return selected_page_indices


def _best_available_page_text(page) -> tuple[str | None, str]:
    if page is None:
        return None, "text"
    post_layout = page.metadata.get("post_layout")
    if isinstance(post_layout, dict) and page.regions:
        try:
            markdown = export_page_markdown(page, pretty=False)
        except Exception:
            markdown = None
        if isinstance(markdown, str) and markdown.strip():
            return markdown, "markdown"
    text = getattr(page, "ocr_text", None)
    if isinstance(text, str) and text.strip():
        return text, "plain_text"
    return None, "text"
