"""LLM-backed evidence extraction."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import re
from typing import Any

from docclaw.agent.debug import dump_jsonl_from_env
from docclaw.agent.tool.evidence.evidence import EvidenceTool, build_evidence
from docclaw.agent.utils import (
    Action,
    Evidence,
    RunState,
    new_id,
    page_id_from_index,
    page_index_from_id,
    page_index_from_number,
    plannerize_page_refs,
)
from docclaw.provider.base import LLMProvider


class LLMEvidenceTool(EvidenceTool):
    """Extract structured evidence with an LLM provider."""

    _JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

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
    def description(self) -> str:
        return (
            "Use an LLM to assess whether the current run state contains enough "
            "task-relevant evidence, extract grounded evidence items, and report "
            "what information is still missing. Any answerability judgment from this "
            "tool applies only to the current focused state and current evidence "
            "scope; answerability_status='inconclusive' is a refinement signal, not an automatic final "
            "Not answerable decision. Successful calls add grounded evidence plus a "
            "task-level evidence assessment record for later planner decisions."
        )

    def extract_evidence(self, state: RunState, action: Action) -> list[Evidence]:
        raise RuntimeError("LLMEvidenceTool.extract_evidence is async-only; call execute")

    async def execute(self, state: RunState, action: Action):
        mode = _normalize_evidence_mode(action.parameters.get("mode"))
        context = _build_evidence_context(
            state,
            target=action.target,
            mode=mode,
        )
        messages, user_payload = self._build_messages(
            state,
            action,
            context=context,
            mode=mode,
        )
        dump_jsonl_from_env(
            "DOCCLAW_EVIDENCE_DEBUG_PATH",
            {
                "kind": "evidence_input",
                "model": self.model,
                "document_id": state.document.document_id,
                "action_id": action.action_id,
                "task": state.task.to_dict(),
                "target": action.target,
                "parameters": action.parameters,
                "context": context,
                "user_payload": user_payload,
                "page_debug": _evidence_input_page_debug(
                    state,
                    target=action.target,
                    context=context,
                ),
            },
        )
        semantic_attempts = 3
        semantic_temperatures = [
            self.temperature,
            0.6,
            0.6,
        ]
        semantic_failures: list[dict[str, Any]] = []
        chosen_payload: dict[str, Any] | None = None
        chosen_evidence: list[Evidence] | None = None
        chosen_response = None
        chosen_parse_failures: list[dict[str, Any]] = []
        chosen_attempt_index = semantic_attempts - 1

        for semantic_attempt_index in range(semantic_attempts):
            parse_failures: list[dict[str, Any]] = []
            payload: dict[str, Any] | None = None
            evidence: list[Evidence] | None = None
            response = None
            temperature = semantic_temperatures[semantic_attempt_index]
            for format_attempt_index in range(2):
                response = await self.provider.chat(
                    messages,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                )
                if response.error:
                    return self.error(action, response.error)

                try:
                    payload = self._parse_payload(response.content)
                    evidence = self._evidence_from_payload(state, action, payload)
                    break
                except Exception as exc:
                    parse_failures.append(
                        {
                            "format_attempt_index": format_attempt_index,
                            "raw_content": response.content,
                            "parse_error": str(exc),
                            "usage": response.usage,
                        }
                    )
                    if format_attempt_index == 1:
                        dump_jsonl_from_env(
                            "DOCCLAW_EVIDENCE_DEBUG_PATH",
                            {
                                "kind": "evidence_error",
                                "model": self.model,
                                "document_id": state.document.document_id,
                                "action_id": action.action_id,
                                "raw_content": response.content,
                                "parse_error": str(exc),
                                "usage": response.usage,
                                "retry_attempted": True,
                                "attempts": parse_failures,
                                "semantic_attempt_index": semantic_attempt_index,
                                "temperature": temperature,
                            },
                        )
                        return self.error(action, f"invalid evidence response: {exc}")
            assert response is not None
            assert payload is not None
            assert evidence is not None

            semantic_failures.append(
                {
                    "semantic_attempt_index": semantic_attempt_index,
                    "temperature": temperature,
                    "answerability_status": str(payload.get("answerability_status") or "").strip(),
                    "missing_information": payload.get("missing_information"),
                    "evidence_count": len(evidence),
                    "usage": response.usage,
                }
            )

            chosen_payload = payload
            chosen_evidence = evidence
            chosen_response = response
            chosen_parse_failures = parse_failures
            chosen_attempt_index = semantic_attempt_index

            if _is_answerable_payload(payload, evidence):
                break
            if semantic_attempt_index == 0:
                continue

        assert chosen_response is not None
        assert chosen_payload is not None
        assert chosen_evidence is not None
        dump_jsonl_from_env(
            "DOCCLAW_EVIDENCE_DEBUG_PATH",
            {
                "kind": "evidence_output",
                "model": self.model,
                "document_id": state.document.document_id,
                "action_id": action.action_id,
                "raw_content": chosen_response.content,
                "parsed_payload": chosen_payload,
                "evidence": [item.to_dict() for item in chosen_evidence],
                "usage": chosen_response.usage,
                "retry_count": len(chosen_parse_failures),
                "semantic_attempt_count": len(semantic_failures),
                "semantic_attempts": semantic_failures,
                "chosen_semantic_attempt_index": chosen_attempt_index,
                "page_debug": _evidence_output_page_debug(
                    state,
                    target=action.target,
                    context=context,
                    evidence=chosen_evidence,
                ),
            },
        )
        return self._observation(
            state,
            action,
            chosen_evidence,
            chosen_payload,
            chosen_response.usage,
        )

    def _build_messages(
        self,
        state: RunState,
        action: Action,
        *,
        context: dict[str, Any],
        mode: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if mode == "not_answerable_recheck":
            return self._build_recheck_messages(
                state,
                action,
                context=context,
            )
        with_page_images = _normalize_with_page_images(
            action.parameters.get("with_page_images")
        )
        return self._build_default_messages(
            state,
            action,
            context=context,
            with_page_images=with_page_images,
        )

    def _build_default_messages(
        self,
        state: RunState,
        action: Action,
        *,
        context: dict[str, Any],
        with_page_images: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        system_prompt = (
            "You are the DocClaw evidence extraction tool.\n"
            "Your job is to assess whether the current question can already be answered "
            "from the provided context, and to extract only grounded evidence.\n"
            "Context Rules:\n"
            "The provided context is organized as:\n"
            "- document: document identity and page scope.\n"
            "- page_context: page-level text and page-level figure insights.\n"
            "- regions: region-level OCR or region-level context.\n"
            "- exploration_summary: what has already been explored so far.\n"
            "- focused page images may also be provided when page_ids are in scope.\n"
            "Evidence Interpretation Rules:\n"
            "- Treat page-level and region-level context as different layers.\n"
            "- page_context provides page-level context such as page text, summaries, interpretations, or page-level judgments.\n"
            "- regions provides region-level local context such as region text, region content, or other region-specific details.\n"
            "- Treat page_context and region-level context as containers of evidence, not automatic answer units.\n"
            "- Items in page_context may summarize, interpret, or otherwise cover multiple region-level regions or visual units on the page.\n"
            "- A region-level item may be only part of what a page-level item covers, so do not count or compare them as if they were at the same level.\n"
            "- When the provided context contains complementary evidence, make the grounded comparisons, mappings, joins, and cross-page connections needed to answer the question.\n"
            "- The evidence you extract should be specific and complete enough for the answer step to answer the question directly.\n"
            "Visual Reasoning Rules:\n"
            "- Visual reasoning is needed when the question depends on the meaning of images, charts, figures, tables, or other visual content.\n"
            "- When page_context contains page-level figure insights, treat those as the primary visual reasoning context for those pages.\n"
            "- Use region-level context to support, or verify page-level visual understanding when needed.\n"
            "- When the task depends on image, figure, or chart semantics, do not set answerability_status='answerable' based only on text.\n"
            "Task Contract Rules:\n"
            "- Before setting answerability_status='answerable', infer the question's answer contract from the task wording.\n"
            "- lookup: the task asks for a local fact, label, value, definition, name, or span. It requires one direct, grounded, non-conflicting answer-bearing span or value.\n"
            "- count: the task asks how many items, stages, tables, examples, or samples there are. It requires an explicit total or a complete, grounded set of countable items sufficient to derive the total.\n"
            "- comparison: the task asks which item is larger, smaller, highest, lowest, most, least, or asks for a cross-item comparison. It requires the compared items, the comparison basis, and enough complete evidence to support the requested winner, ordering, or difference.\n"
            "- calculation: the task asks for a ratio, rate, percent, percentage, share, fraction, average, difference, subtraction, sum, total, derived metric, formula result, or wording such as 'among all' or 'out of all'. It requires all operands, the calculation relationship, and shared scope; partial inputs, mismatched scopes, or unclear calculation basis mean inconclusive.\n"
            "- enumeration: the task asks to list, enumerate, name all uses, categories, activities, or members of a set. It requires the complete set, not examples or a partial subset.\n"
            "- Aggregation questions such as sum, total, combined, at least, or more require all needed operands, categories, and value-to-label mappings.\n"
            "- For counting, comparison, grouping, or aggregation, do not answer until the logical answer units are identified clearly enough for the requested conclusion.\n"
            "- Do not treat vague statements such as 'several', 'many', 'these questions', or 'key questions' as sufficient numeric evidence unless the total is explicit or the full set is enumerated.\n"
            "- For numeric questions, extract the exact numbers, units, labels or visual positions needed to support the final value.\n"
            "- For percentage, difference, or other derived-value questions, extract the operands, operation, scope, and unit.\n"
            "- For difference or gap questions, preserve whether the evidence supports an absolute difference, a signed difference, or a percentage-point difference.\n"
            "- For count questions, extract either the explicit total or the complete countable set.\n"
            "Answerability Rules:\n"
            "- Set answerability_status='inconclusive' when the current focused state is not enough to answer yet.\n"
            "- When answerability_status='inconclusive', missing_information must describe the specific missing fact, "
            "value, text, label, page content, or region content. Describe what is missing from the current "
            "evidence scope, not what action to take next.\n"
            "- Base answerability and missing_information only on the current provided context. Do not make document-level "
            "absence claims.\n"
            "- Be very careful before setting answerability_status='answerable'. Treat "
            "answerability as specific completeness, not merely relevance. If there is reasonable "
            "uncertainty about completeness, missing candidates, partial coverage, or whether the current "
            "focused state is only a subset of the required evidence, return answerability_status='inconclusive' rather than "
            "'answerable'. Do not use answerability_status='answerable' to mean 'probably enough' or 'likely the answer'.\n"
            "- When answerability_status='answerable', set missing_information to null and extract all grounded "
            "supporting information needed to finalize the answer, not just a partial subset when multiple evidence items are jointly required.\n"
            '- <answerability_status> must be exactly "answerable" or "inconclusive".\n'
            "Return only JSON with this shape:\n"
            "{\n"
            '  "answerability_status": "<answerability_status>",\n'
            '  "missing_information": "optional short note or null",\n'
            '  "evidence": [\n'
            "    {\n"
            '      "content": "evidence text",\n'
            '      "trust_level": "trusted",\n'
            '      "reference": null,\n'
            '      "page_id": "page_a",\n'
            '      "region_id": "optional region id",\n'
            '      "confidence": 0.0,\n'
            '      "metadata": {"reason": "short grounding note"}\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Do not invent facts that are not present in the context."
        )
        user_payload = plannerize_page_refs({
            "task": state.task.to_dict(),
            "target": action.target,
            "parameters": action.parameters,
            "context": context,
        }, document=state.document)
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps(user_payload, ensure_ascii=False, indent=2),
            }
        ]
        if with_page_images:
            for item in _default_candidate_pages_for_images(state, action):
                page_id = item.get("page_id")
                text = item.get("text")
                text_format = item.get("text_format") or "none"
                image_path = item.get("image_path")
                if page_id is not None:
                    user_content.append(
                        {
                            "type": "text",
                            "text": f"Focused page_id {page_id} ({text_format}):",
                        }
                    )
                if isinstance(text, str) and text.strip():
                    user_content.append({"type": "text", "text": text})
                if isinstance(image_path, str) and image_path.strip():
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _image_data_url(Path(image_path)),
                            },
                        }
                    )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ], user_payload

    def _build_recheck_messages(
        self,
        state: RunState,
        action: Action,
        *,
        context: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        system_prompt = (
            "You are the DocClaw evidence extraction tool.\n"
            "This call is a final reassessment immediately before concluding "
            "'Not answerable'. Re-check only the provided candidate pages.\n"
            "Use the candidate page images together with their available page text "
            "(OCR markdown when available, otherwise page OCR text) to decide whether "
            "the question is answerable from these pages.\n"
            "When the provided context contains complementary evidence, make the "
            "grounded comparisons, mappings, joins, and cross-page connections "
            "needed to answer the question.\n"
            "For numeric questions, extract the exact numbers, units, labels, scope, "
            "and calculation relationship needed to support the final answer; if "
            "any required operand or unit is missing, keep the status inconclusive.\n"
            "If the question is answerable from these pages, set answerability_status='answerable' "
            "and extract grounded evidence. The evidence you extract should be specific and "
            "complete enough for the answer step to answer the question directly.\n"
            "If it is still not answerable from these pages, "
            "set answerability_status='inconclusive' and state exactly what is still missing.\n"
            '- <answerability_status> must be exactly "answerable" or "inconclusive".\n'
            "Return only JSON with this shape:\n"
            "{\n"
            '  "answerability_status": "<answerability_status>",\n'
            '  "missing_information": "optional short note or null",\n'
            '  "evidence": [\n'
            "    {\n"
            '      "content": "evidence text",\n'
            '      "trust_level": "trusted",\n'
            '      "reference": null,\n'
            '      "page_id": "page_a",\n'
            '      "region_id": "optional region id",\n'
            '      "confidence": 0.0,\n'
            '      "metadata": {"reason": "short grounding note"}\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Do not invent facts that are not present in the candidate pages."
        )
        user_payload = plannerize_page_refs({
            "task": state.task.to_dict(),
            "target": action.target,
            "parameters": action.parameters,
            "context": context,
        }, document=state.document)
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps(user_payload, ensure_ascii=False, indent=2),
            }
        ]
        for item in context.get("candidate_pages") or []:
            if not isinstance(item, dict):
                continue
            page_id = item.get("page_id")
            text = item.get("text")
            text_format = item.get("text_format") or "none"
            image_path = item.get("image_path")
            if page_id is not None:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"Candidate page_id {page_id} ({text_format}):",
                    }
                )
            if isinstance(text, str) and text.strip():
                user_content.append({"type": "text", "text": text})
            if isinstance(image_path, str) and image_path.strip():
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _image_data_url(Path(image_path)),
                        },
                    }
                )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ], user_payload

    @classmethod
    def _parse_payload(cls, content: str | None) -> dict[str, Any]:
        if content is None or not content.strip():
            raise ValueError("empty response")
        candidate = content.strip()
        match = cls._JSON_BLOCK_RE.search(candidate)
        if match:
            candidate = match.group(1).strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start < 0 or end < start:
                raise ValueError("response must contain a JSON object")
            payload = json.loads(candidate[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("response JSON must be an object")
        return payload

    def _evidence_from_payload(
        self,
        state: RunState,
        action: Action,
        payload: dict[str, Any],
    ) -> list[Evidence]:
        items = payload.get("evidence", [])
        if not isinstance(items, list):
            raise ValueError("evidence must be a list")
        max_items = _optional_int(action.parameters.get("max_evidence"))
        if max_items is not None:
            items = items[:max_items]

        evidence: list[Evidence] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "")
            if not content.strip():
                continue
            page_index = _optional_page_id_to_index(
                item.get("page_id"),
                document=state.document,
            )
            if page_index is None:
                page_index = _optional_page_number_to_index(item.get("page_number"))
            if page_index is None:
                page_index = _optional_int(item.get("page_index"))
            region_id = item.get("region_id")
            confidence = item.get("confidence")
            metadata = item.get("metadata")
            evidence.append(
                build_evidence(
                    state,
                    action,
                    content=content,
                    page_index=_optional_int(page_index),
                    region_id=str(region_id) if region_id else None,
                    confidence=_optional_float(confidence),
                    metadata=dict(metadata) if isinstance(metadata, dict) else {},
                )
            )
        return evidence

    def _observation(
        self,
        state: RunState,
        action: Action,
        evidence: list[Evidence],
        payload: dict[str, Any],
        usage: dict[str, int],
    ):
        from docclaw.agent.utils import Observation

        page_ids = sorted({
            page_id_from_index(item.page_index, document=state.document)
            for item in evidence
            if item.page_index is not None
        })
        page_scope = f" across page_ids {','.join(page_ids)}" if page_ids else ""
        answerability_status = str(payload.get("answerability_status") or "").strip()
        if answerability_status not in {"answerable", "inconclusive"}:
            raise ValueError("answerability_status must be 'answerable' or 'inconclusive'")
        missing_information = payload.get("missing_information")
        if not isinstance(missing_information, str) or not missing_information.strip():
            missing_information = None
        else:
            missing_information = missing_information.strip()

        if evidence:
            message = f"Extracted {len(evidence)} LLM evidence item(s){page_scope}."
        else:
            message = "Extracted no grounded evidence from the current focused state."
        message += (
            " Current focused state is sufficient to answer."
            if answerability_status == "answerable"
            else " Current focused state is not yet sufficient to answer."
        )
        if missing_information:
            message += f" Missing from current focused state: {missing_information}"

        return Observation(
            action_id=action.action_id,
            success=True,
            data={
                "assessment_id": new_id("assessment"),
                "evidence": [item.to_dict() for item in evidence],
                "answerability_status": answerability_status,
                "missing_information": missing_information,
                "usage": usage,
                "source": "llm",
            },
            message=message,
        )


def _build_evidence_context(
    state: RunState,
    *,
    target: dict[str, Any],
    mode: str = "default",
) -> dict[str, Any]:
    if mode == "not_answerable_recheck":
        return _build_recheck_context(state, target=target)
    region_ids = _normalize_region_ids(target.get("region_ids"))
    page_indices = _normalize_page_indices(target.get("page_indices"))

    focused_page_indexes: set[int] = set(page_indices)
    if region_ids:
        for region_id in region_ids:
            region = state.get_region(region_id)
            if region is not None:
                focused_page_indexes.add(region.page_index)

    regions: list[dict[str, Any]] = []
    for page in state.document.pages:
        if focused_page_indexes and page.page_index not in focused_page_indexes:
            continue
        for region in page.regions:
            if region_ids and region.region_id not in region_ids:
                continue
            if not region_ids and focused_page_indexes and page.page_index not in focused_page_indexes:
                continue
            text = region.text
            item = {
                "page_id": page_id_from_index(page.page_index, document=state.document),
                "region_id": region.region_id,
                "inspected": region.region_id in state.inspected_regions,
                "text": text,
            }
            regions.append(item)

    page_context: list[dict[str, Any]] = []
    for page in state.document.pages:
        if focused_page_indexes and page.page_index not in focused_page_indexes:
            continue
        item: dict[str, Any] = {
            "page_id": page_id_from_index(page.page_index, document=state.document),
        }
        markdown = _page_markdown_for_evidence(page)
        if isinstance(markdown, str) and markdown.strip():
            item["text"] = markdown
            item["text_format"] = "markdown"
        else:
            text = page.ocr_text
            if isinstance(text, str) and text.strip():
                item["text"] = text
        page_figure_insights: list[dict[str, Any]] = []
        for insight_key in state.figure_insights.ordered_insight_keys:
            insight = state.figure_insights.get_insight(insight_key)
            if insight is None or insight.page_index != page.page_index:
                continue
            insight_payload: dict[str, Any] = {
                "question": insight.question,
            }
            if isinstance(insight.answer, str) and insight.answer.strip():
                insight_payload["answer"] = insight.answer.strip()
            if isinstance(insight.reason, str) and insight.reason.strip():
                insight_payload["reason"] = insight.reason.strip()
            if len(insight_payload) > 1:
                page_figure_insights.append(insight_payload)
        if page_figure_insights:
            item["figure_insights"] = page_figure_insights
        if len(item) == 1:
            continue
        page_context.append(item)

    return {
        "document": {
            "document_id": state.document.document_id,
            "page_ids": [page_id_from_index(page.page_index, document=state.document) for page in state.document.pages],
        },
        "regions": regions,
        "page_context": page_context,
        "exploration_summary": state.build_exploration_summary(),
    }


def _build_recheck_context(
    state: RunState,
    *,
    target: dict[str, Any],
) -> dict[str, Any]:
    page_indices = _normalize_page_indices(target.get("page_indices"))
    candidate_pages: list[dict[str, Any]] = []
    for page_index in page_indices:
        page = state.get_page(page_index)
        if page is None:
            continue
        text = None
        text_format = "none"
        markdown = _page_markdown_for_evidence(page)
        if isinstance(markdown, str) and markdown.strip():
            text = markdown
            text_format = "markdown"
        elif isinstance(page.ocr_text, str) and page.ocr_text.strip():
            text = page.ocr_text
            text_format = "plain_text"
        item: dict[str, Any] = {
            "page_id": page_id_from_index(page.page_index, document=state.document),
            "text_format": text_format,
        }
        if text is not None:
            item["text"] = text
        if isinstance(page.image_path, str) and page.image_path.strip():
            item["image_path"] = page.image_path
        candidate_pages.append(item)
    return {
        "document": {
            "document_id": state.document.document_id,
            "page_ids": [page_id_from_index(page.page_index, document=state.document) for page in state.document.pages],
        },
        "candidate_pages": candidate_pages,
    }


def _page_markdown_for_evidence(page) -> str | None:
    post_layout = page.metadata.get("post_layout")
    if not isinstance(post_layout, dict) or not page.regions:
        return None
    try:
        from docclaw.exporter import export_page_markdown

        markdown = export_page_markdown(page, pretty=False)
    except Exception:
        return None
    if not isinstance(markdown, str) or not markdown.strip():
        return None
    return markdown


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _normalize_evidence_mode(value: Any) -> str:
    mode = str(value or "").strip()
    if not mode:
        return "default"
    if mode in {"default", "not_answerable_recheck"}:
        return mode
    return "default"


def _normalize_with_page_images(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"false", "0", "no", "off"}:
        return False
    if normalized in {"true", "1", "yes", "on"}:
        return True
    return True


def _normalize_region_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    region_ids: list[str] = []
    seen: set[str] = set()
    for raw_region_id in value:
        region_id = str(raw_region_id).strip()
        if not region_id or region_id in seen:
            continue
        region_ids.append(region_id)
        seen.add(region_id)
    return region_ids


def _normalize_page_indices(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    page_indices: list[int] = []
    seen: set[int] = set()
    for raw_page_index in value:
        try:
            page_index = int(raw_page_index)
        except (TypeError, ValueError):
            continue
        if page_index in seen:
            continue
        page_indices.append(page_index)
        seen.add(page_index)
    return page_indices


def _evidence_input_page_debug(
    state: RunState,
    *,
    target: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, list[int]]:
    try:
        return {
            "target_page_indices": _normalize_page_indices(target.get("page_indices")),
            "context_page_indices": _context_page_indices(state, context=context),
        }
    except Exception:
        return {
            "target_page_indices": [],
            "context_page_indices": [],
        }


def _evidence_output_page_debug(
    state: RunState,
    *,
    target: dict[str, Any],
    context: dict[str, Any],
    evidence: list[Evidence],
) -> dict[str, list[int]]:
    try:
        return {
            **_evidence_input_page_debug(state, target=target, context=context),
            "evidence_page_indices": _dedupe_page_indices(
                item.page_index for item in evidence if item.page_index is not None
            ),
        }
    except Exception:
        return {
            **_evidence_input_page_debug(state, target=target, context=context),
            "evidence_page_indices": [],
        }


def _context_page_indices(state: RunState, *, context: dict[str, Any]) -> list[int]:
    page_indices: list[int] = []
    for key in ("regions", "page_context", "candidate_pages"):
        items = context.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            page_index = _optional_int(item.get("page_index"))
            if page_index is not None:
                page_indices.append(page_index)
                continue
            page_id = item.get("page_id")
            if isinstance(page_id, str) and page_id.strip():
                try:
                    page_indices.append(page_index_from_id(page_id, document=state.document))
                except (TypeError, ValueError):
                    pass
    return _dedupe_page_indices(page_indices)


def _dedupe_page_indices(values) -> list[int]:
    page_indices: list[int] = []
    seen: set[int] = set()
    for value in values:
        if not isinstance(value, int) or value in seen:
            continue
        page_indices.append(value)
        seen.add(value)
    return page_indices


def _default_candidate_pages_for_images(
    state: RunState,
    action: Action,
) -> list[dict[str, Any]]:
    page_indices = _normalize_page_indices(action.target.get("page_indices"))
    if not page_indices:
        return []
    candidate_pages: list[dict[str, Any]] = []
    for page_index in page_indices:
        page = state.get_page(page_index)
        if page is None:
            continue
        item: dict[str, Any] = {
            "page_id": page_id_from_index(page.page_index, document=state.document),
            "text_format": "none",
        }
        markdown = _page_markdown_for_evidence(page)
        if isinstance(markdown, str) and markdown.strip():
            item["text"] = markdown
            item["text_format"] = "markdown"
        elif isinstance(page.ocr_text, str) and page.ocr_text.strip():
            item["text"] = page.ocr_text
            item["text_format"] = "plain_text"
        if isinstance(page.image_path, str) and page.image_path.strip():
            item["image_path"] = page.image_path
        candidate_pages.append(item)
    return candidate_pages


def _optional_page_number_to_index(value: Any) -> int | None:
    maybe = _optional_int(value)
    if maybe is None:
        return None
    try:
        return page_index_from_number(maybe)
    except ValueError:
        return None


def _optional_page_id_to_index(value: Any, *, document: Any) -> int | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return page_index_from_id(value.strip(), document=document)
    except ValueError:
        return None


def _is_answerable_payload(payload: dict[str, Any], evidence: list[Evidence]) -> bool:
    answerability_status = str(payload.get("answerability_status") or "").strip()
    if answerability_status != "answerable":
        return False
    missing_information = payload.get("missing_information")
    if missing_information is not None and str(missing_information).strip():
        return False
    if not evidence:
        return False
    return True


def _normalize_search_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    search_ids: list[str] = []
    seen: set[str] = set()
    for raw_search_id in value:
        search_id = str(raw_search_id).strip()
        if not search_id or search_id in seen:
            continue
        search_ids.append(search_id)
        seen.add(search_id)
    return search_ids


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(value)
    return int(str(value))


def _optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float | str):
        return float(value)
    return float(str(value))
