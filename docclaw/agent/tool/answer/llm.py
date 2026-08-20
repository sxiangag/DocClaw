"""LLM-backed answer generation."""

from __future__ import annotations

import json
import re
from typing import Any

from docclaw.agent.debug import dump_jsonl_from_env
from docclaw.agent.tool.answer.answer import AnswerTool
from docclaw.agent.utils import Action, Evidence, Observation, RunState, page_id_from_index, plannerize_page_refs
from docclaw.provider.base import LLMProvider


class LLMAnswerTool(AnswerTool):
    """Use an LLM provider to synthesize the final answer."""

    _JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    _EVIDENCE_IDS_LINE_RE = re.compile(r'^"?evidence_ids"?\s*:', re.IGNORECASE)

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

    async def generate_answer(self, state: RunState, action: Action) -> Observation:
        if not state.evidence:
            return self.error(
                action,
                "cannot answer without evidence; call extract_evidence first",
            )
        try:
            assessment_id, evidence = _resolve_answer_evidence(state)
        except Exception as exc:
            return self.error(action, str(exc))

        messages = self._build_messages(
            state,
            action,
            assessment_id=assessment_id,
            evidence=evidence,
        )
        dump_jsonl_from_env(
            "DOCCLAW_ANSWER_DEBUG_PATH",
            {
                "kind": "answer_input",
                "model": self.model,
                "document_id": state.document.document_id,
                "action_id": action.action_id,
                "task": state.task.to_dict(),
                "target": action.target,
                "parameters": action.parameters,
                "assessment_id": assessment_id,
                "context": _build_answer_context(state, assessment_id=assessment_id, evidence=evidence),
                "page_debug": _answer_page_debug(evidence),
            },
        )
        candidate_temperatures = [
            0.0,
            0.4,
        ]
        candidate_runs: list[dict[str, Any]] = []
        candidate_failures: list[dict[str, Any]] = []
        for candidate_index, temperature in enumerate(candidate_temperatures):
            candidate_result = await self._generate_candidate(
                messages,
                temperature=temperature,
            )
            if candidate_result["success"]:
                candidate_runs.append(
                    {
                        "candidate_index": candidate_index,
                        "temperature": temperature,
                        **candidate_result,
                    }
                )
                continue
            candidate_failures.append(
                {
                    "candidate_index": candidate_index,
                    "temperature": temperature,
                    **candidate_result,
                }
            )

        if not candidate_runs:
            first_failure = candidate_failures[0]
            dump_jsonl_from_env(
                "DOCCLAW_ANSWER_DEBUG_PATH",
                {
                    "kind": "answer_error",
                    "model": self.model,
                    "document_id": state.document.document_id,
                    "action_id": action.action_id,
                    "stage": "candidate_generation",
                    "failures": candidate_failures,
                    "page_debug": _answer_page_debug(evidence),
                },
            )
            return self.error(action, first_failure["error"])

        selected_candidate = candidate_runs[0]
        selection_result: dict[str, Any] | None = None
        if len(candidate_runs) >= 2:
            first_candidate = candidate_runs[0]
            second_candidate = candidate_runs[1]
            if (
                first_candidate["answer"] != second_candidate["answer"]
                or first_candidate["reason"] != second_candidate["reason"]
            ):
                selector_messages = self._build_selection_messages(
                    state,
                    action,
                    assessment_id=assessment_id,
                    evidence=evidence,
                    candidate_a=first_candidate,
                    candidate_b=second_candidate,
                )
                selection_result = await self._select_candidate(selector_messages)
                if selection_result["success"]:
                    selected_candidate = (
                        first_candidate
                        if selection_result["selected_candidate"] == "A"
                        else second_candidate
                    )
                else:
                    selection_result = {
                        **selection_result,
                        "selected_candidate": "A",
                        "fallback_used": True,
                    }

        answer = selected_candidate["answer"]
        reason = selected_candidate["reason"]
        evidence_ids = [item.evidence_id for item in evidence]
        usage = _merge_usages(
            [
                candidate_run.get("usage")
                for candidate_run in candidate_runs
            ]
            + ([selection_result.get("usage")] if selection_result else [])
        )
        dump_jsonl_from_env(
            "DOCCLAW_ANSWER_DEBUG_PATH",
            {
                "kind": "answer_output",
                "model": self.model,
                "document_id": state.document.document_id,
                "action_id": action.action_id,
                "candidate_runs": candidate_runs,
                "candidate_failures": candidate_failures,
                "selection_result": selection_result,
                "raw_content": selected_candidate["raw_content"],
                "parsed_payload": selected_candidate["parsed_payload"],
                "answer": answer,
                "reason": reason,
                "evidence_ids": evidence_ids,
                "usage": usage,
                "retry_count": sum(
                    int(candidate_run.get("retry_count", 0))
                    for candidate_run in candidate_runs
                ),
                "page_debug": _answer_page_debug(evidence),
            },
        )

        return Observation(
            action_id=action.action_id,
            success=True,
            data={
                "answer": answer,
                "reason": reason,
                "evidence_ids": evidence_ids,
                "usage": usage,
                "source": "llm",
            },
            message=(
                f"Prepared answer candidate from {len(evidence_ids)} evidence item(s) "
                f"({len(answer)} chars)."
            ),
        )

    async def _generate_candidate(
        self,
        messages: list[dict[str, str]],
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
                reason = self._reason_from_payload(payload)
                return {
                    "success": True,
                    "raw_content": response.content,
                    "parsed_payload": payload,
                    "answer": answer,
                    "reason": reason,
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
            "error": f"invalid answer response: {parse_failures[-1]['parse_error']}",
            "attempts": parse_failures,
            "usage": parse_failures[-1].get("usage"),
            "retry_count": len(parse_failures),
        }

    def _build_selection_messages(
        self,
        state: RunState,
        action: Action,
        *,
        assessment_id: str,
        evidence: list[Evidence],
        candidate_a: dict[str, Any],
        candidate_b: dict[str, Any],
    ) -> list[dict[str, str]]:
        system_prompt = (
            "You are the DocClaw answer selector.\n"
            "Choose which candidate answer is better supported by the provided evidence.\n"
            "Return only JSON with this shape:\n"
            "{\n"
            '  "selected_candidate": "A" or "B",\n'
            '  "reason": "<short reason>"\n'
            "}\n"
            "Selection rules:\n"
            "- Prefer the candidate that is better supported by the evidence.\n"
            "- If both are supported, prefer the candidate that is complete, direct, easy to extract, and closest to the evidence wording.\n"
            "- Do not synthesize a new answer. Set selected_candidate to either A or B.\n"
        )
        user_prompt = json.dumps(
            plannerize_page_refs(
                {
                    "task": state.task.to_dict(),
                    "target": action.target,
                    "parameters": action.parameters,
                    "context": _build_answer_context(state, assessment_id=assessment_id, evidence=evidence),
                    "candidate_a": {
                        "answer": candidate_a["answer"],
                        "reason": candidate_a["reason"],
                    },
                    "candidate_b": {
                        "answer": candidate_b["answer"],
                        "reason": candidate_b["reason"],
                    },
                },
                document=state.document,
            ),
            ensure_ascii=False,
            indent=2,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def _select_candidate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        parse_failures: list[dict[str, Any]] = []
        for attempt_index in range(2):
            response = await self.provider.chat(
                messages,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            if response.error:
                return {
                    "success": False,
                    "error": response.error,
                    "attempts": parse_failures,
                    "usage": response.usage,
                }
            try:
                payload = self._parse_selection_payload(response.content)
                return {
                    "success": True,
                    "raw_content": response.content,
                    "parsed_payload": payload,
                    "selected_candidate": payload["selected_candidate"],
                    "reason": payload["reason"],
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
            "error": f"invalid answer selection response: {parse_failures[-1]['parse_error']}",
            "attempts": parse_failures,
            "usage": parse_failures[-1].get("usage"),
            "retry_count": len(parse_failures),
        }

    def _build_messages(
        self,
        state: RunState,
        action: Action,
        *,
        assessment_id: str,
        evidence: list[Evidence],
    ) -> list[dict[str, str]]:
        context = _build_answer_context(state, assessment_id=assessment_id, evidence=evidence)
        system_prompt = (
            "You are the DocClaw answer tool.\n"
            "Your job is to produce the user-facing answer for the current question.\n"
            "Use only the provided evidence, keep the answer concise and clear, and answer exactly what the question asks.\n"
            "The provided evidence set has already been selected upstream. Do not narrow, or drop evidence items on your own.\n"
            "Return only JSON with this shape:\n"
            "{\n"
            '  "answer": "<short clear answer>",\n'
            '  "reason": "<short supporting explanation>"\n'
            "}\n"
            "Answer rules:\n"
            "- The answer field must be a non-empty string.\n"
            "- Keep the answer very short, direct, and clear.\n"
            "- Do not include long reasoning, bullet lists, markdown formatting, caveats, or meta commentary.\n"
            "- Do not mention evidence ids anywhere in the answer text.\n"
            "- Base the answer strictly on the provided evidence.\n"
            "- The provided evidence is sufficient to answer the question. Answer the question based on it.\n"
            "- For counting or enumeration questions, first identify the logical unit being counted from the question. Then derive the total from the provided evidence.\n"
            "- For numeric answers, use the operation requested by the question and preserve the unit and scale.\n"
            "- For discrete counts, return a whole number.\n"
            "- Return the complete label or phrase needed by the question. Do not shorten it to a partial word.\n"
            "- Avoid paraphrasing when the evidence contains a direct answer string.\n"
            "Reason rules:\n"
            "- The reason field should briefly explain why the answer follows from the evidence.\n"
            "- Keep the reason short: one or two short sentences at most.\n"
            "- Do not mention evidence ids anywhere in the reason text.\n"
            "- Do not invent facts not grounded in the evidence.\n"
            "Example:\n"
            '{\n'
            '  "answer": "$42.00",\n'
            '  "reason": "The table reports a total of $42.00."\n'
            '}'
        )
        user_prompt = json.dumps(
            plannerize_page_refs(
                {
                    "task": state.task.to_dict(),
                    "target": action.target,
                    "parameters": action.parameters,
                    "context": context,
                },
                document=state.document,
            ),
            ensure_ascii=False,
            indent=2,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

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
                raise ValueError("response must be valid JSON")
            payload = json.loads(candidate[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("response JSON must be an object")
        return payload

    def _answer_from_payload(self, payload: dict[str, Any]) -> str:
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("response must contain a non-empty answer")
        lines = self._sanitize_text_lines(answer)
        if not lines:
            raise ValueError("response must contain a non-empty answer")
        return " ".join(lines)

    @classmethod
    def _reason_from_payload(cls, payload: dict[str, Any]) -> str:
        reason = payload.get("reason")
        if isinstance(reason, str) and reason.strip():
            lines = cls._sanitize_text_lines(reason)
            return " ".join(lines)
        return ""

    @classmethod
    def _parse_selection_payload(cls, content: str | None) -> dict[str, str]:
        payload = cls._parse_payload(content)
        selected_candidate = payload.get("selected_candidate")
        if selected_candidate not in {"A", "B"}:
            raise ValueError("response must contain selected_candidate='A' or 'B'")
        reason = payload.get("reason")
        if isinstance(reason, str) and reason.strip():
            reason_text = " ".join(cls._sanitize_text_lines(reason))
        else:
            reason_text = ""
        return {
            "selected_candidate": selected_candidate,
            "reason": reason_text,
        }

    @classmethod
    def _sanitize_text_lines(cls, text: str) -> list[str]:
        cleaned: list[str] = []
        for raw_line in text.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if cls._is_evidence_ids_line(line):
                continue
            if cls._is_junk_line(line):
                continue
            cleaned.append(line)
        return cleaned

    @classmethod
    def _is_evidence_ids_line(cls, line: str) -> bool:
        return bool(cls._EVIDENCE_IDS_LINE_RE.match(line))

    @classmethod
    def _is_junk_line(cls, line: str) -> bool:
        return line in {"{", "}", "[", "]"}

def _build_answer_context(
    state: RunState,
    *,
    assessment_id: str,
    evidence: list[Evidence],
) -> dict[str, Any]:
    items = [
        _serialize_evidence(state, item)
        for item in evidence
    ]
    return {
        "document": {
            "document_id": state.document.document_id,
            "page_ids": [page_id_from_index(page.page_index, document=state.document) for page in state.document.pages],
        },
        "assessment_id": assessment_id,
        "evidence": items,
    }


def _serialize_evidence(
    state: RunState,
    evidence: Evidence,
) -> dict[str, Any]:
    return {
        "evidence_id": evidence.evidence_id,
        "content": evidence.content,
        "page_id": (
            page_id_from_index(evidence.page_index, document=state.document)
            if evidence.page_index is not None
            else None
        ),
        "region_id": evidence.region_id,
        "confidence": evidence.confidence,
        "metadata": _to_jsonable(evidence.metadata),
    }


def _answer_page_debug(evidence: list[Evidence]) -> dict[str, list[int]]:
    try:
        return {
            "evidence_page_indices": _dedupe_page_indices(
                item.page_index for item in evidence if item.page_index is not None
            ),
        }
    except Exception:
        return {
            "evidence_page_indices": [],
        }


def _dedupe_page_indices(values) -> list[int]:
    page_indices: list[int] = []
    seen: set[int] = set()
    for value in values:
        if not isinstance(value, int) or value in seen:
            continue
        page_indices.append(value)
        seen.add(value)
    return page_indices


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _merge_usages(usages: list[dict[str, Any] | None]) -> dict[str, int] | None:
    merged: dict[str, int] = {}
    saw_usage = False
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                merged[key] = merged.get(key, 0) + value
                saw_usage = True
    return merged if saw_usage else None


def _resolve_answer_evidence(state: RunState) -> tuple[str, list[Evidence]]:
    assessment_ids = state.evidence_assessment_history.ordered_assessment_ids
    if not assessment_ids:
        raise ValueError(
            "cannot answer without an evidence assessment; call extract_evidence before answer_from_evidence"
        )
    assessment_id = assessment_ids[-1]
    assessment = state.evidence_assessment_history.get_assessment(assessment_id)
    if assessment is None:
        raise ValueError(
            f"latest evidence assessment is missing: {assessment_id}; call extract_evidence again before answer_from_evidence"
        )
    if assessment.answerability_status != "answerable":
        raise ValueError(
            "latest evidence assessment marked the current focused state as not yet sufficient to answer; "
            "refine scope and call extract_evidence again before answer_from_evidence"
        )
    if not assessment.evidence_ids:
        raise ValueError(
            "latest evidence assessment did not select any evidence_ids; "
            "call extract_evidence again before answer_from_evidence"
        )

    evidence_by_id = {
        item.evidence_id: item
        for item in state.evidence
    }
    missing_ids = [
        evidence_id
        for evidence_id in assessment.evidence_ids
        if evidence_id not in evidence_by_id
    ]
    if missing_ids:
        raise ValueError(
            "latest evidence assessment references unknown evidence_ids: "
            + ", ".join(missing_ids)
            + "; call extract_evidence again before answer_from_evidence"
        )

    return (
        assessment.assessment_id,
        [evidence_by_id[evidence_id] for evidence_id in assessment.evidence_ids],
    )
