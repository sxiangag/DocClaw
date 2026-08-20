"""VLM-backed figure understanding implementation."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import re
from typing import Any

from docclaw.agent.debug import dump_jsonl_from_env
from docclaw.agent.tool.figure.figure import (
    FigureTool,
    _batch_observation_message,
    _resolve_figure_targets,
    _single_observation_message,
)
from docclaw.agent.tool.layout.layout import LayoutTool
from docclaw.agent.utils import Action, Observation, RunState, page_id_from_index, plannerize_page_refs
from docclaw.provider.base import LLMProvider


_MAIN_VISUAL_LABELS = {"chart", "image", "table", "seal", "header_image", "footer_image"}
_SUPPORTING_VISUAL_LABELS = {"figure_title", "chart_title", "vision_footnote"}


class VLMFigureTool(FigureTool):
    """Understand figures with a multimodal language model."""

    _JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        artifact_dir: str | Path | None = None,
        layout_tool: LayoutTool | None = None,
    ) -> None:
        self.provider = provider
        self.model = model or provider.get_default_model()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.layout_tool = layout_tool

    def understand_figures(
        self,
        state: RunState,
        target: dict[str, Any],
        action: Action,
    ) -> dict[str, Any]:
        raise RuntimeError("VLMFigureTool.understand_figures is async-only; call execute")

    async def execute(self, state: RunState, action: Action) -> Observation:
        targets, error = _resolve_figure_targets(state, action)
        if error is not None:
            return self.error(action, error)
        assert targets is not None

        mode = str(action.parameters.get("mode") or "").strip().lower()
        if mode == "enumeration" and len(targets) > 1:
            return await self._execute_enumeration_batch(
                state,
                action,
                targets=targets,
            )

        results: list[dict[str, Any]] = []
        artifacts: list[str] = []
        for target in targets:
            result, target_artifacts, result_error = await self._execute_single_target(
                state,
                action,
                target=target,
            )
            if result_error is not None:
                return self.error(action, result_error)
            assert result is not None
            results.append(result)
            artifacts.extend(target_artifacts)

        message = (
            _single_observation_message(results[0])
            if len(results) == 1
            else _batch_observation_message(results)
        )
        return Observation(
            action_id=action.action_id,
            success=True,
            data={
                "results": results,
                "source": "batch" if len(results) > 1 else results[0].get("source", "unknown"),
            },
            message=message,
            artifacts=artifacts,
        )

    async def _execute_enumeration_batch(
        self,
        state: RunState,
        action: Action,
        *,
        targets: list[dict[str, Any]],
    ) -> Observation:
        phase1_results: list[dict[str, Any]] = []
        artifacts: list[str] = []
        for target in targets:
            result, target_artifacts, result_error = await self._execute_single_target(
                state,
                action,
                target=target,
            )
            if result_error is not None:
                return self.error(action, result_error)
            assert result is not None
            phase1_results.append(result)
            artifacts.extend(target_artifacts)

        suspect_reasons, verification_artifacts, verification_error = await self._verify_enumeration_results(
            state,
            action,
            targets=targets,
            results=phase1_results,
        )
        if verification_error is not None:
            return self.error(action, verification_error)
        artifacts.extend(verification_artifacts)

        final_results = list(phase1_results)
        result_index_by_page = {
            result["page_index"]: index
            for index, result in enumerate(final_results)
            if isinstance(result.get("page_index"), int)
        }
        target_by_page = {
            target["page_index"]: target
            for target in targets
            if isinstance(target.get("page_index"), int)
        }
        for page_index, verification_reason in suspect_reasons.items():
            target = target_by_page.get(page_index)
            if target is None:
                continue
            previous_result = final_results[result_index_by_page[page_index]]
            rerun_result, rerun_artifacts, rerun_error = await self._execute_single_target(
                state,
                action,
                target=target,
                enumeration_rerun_reason=verification_reason,
                previous_page_answer=_optional_text(previous_result.get("answer")),
            )
            if rerun_error is not None:
                return self.error(action, rerun_error)
            assert rerun_result is not None
            rerun_metadata = dict(rerun_result.get("metadata") or {})
            rerun_metadata.update(
                {
                    "enumeration_rerun": True,
                    "enumeration_rerun_reason": verification_reason,
                    "enumeration_previous_answer": previous_result.get("answer"),
                }
            )
            rerun_result["metadata"] = rerun_metadata
            final_results[result_index_by_page[page_index]] = rerun_result
            artifacts.extend(rerun_artifacts)

        for result in final_results:
            page_index = result.get("page_index")
            if not isinstance(page_index, int):
                continue
            metadata = dict(result.get("metadata") or {})
            if page_index in suspect_reasons:
                metadata.setdefault("enumeration_suspect_reason", suspect_reasons[page_index])
            result["metadata"] = metadata

        message = _batch_observation_message(final_results)
        deduped_artifacts = list(dict.fromkeys(artifacts))
        return Observation(
            action_id=action.action_id,
            success=True,
            data={
                "results": final_results,
                "source": "batch",
            },
            message=message,
            artifacts=deduped_artifacts,
        )

    async def _execute_single_target(
        self,
        state: RunState,
        action: Action,
        *,
        target: dict[str, Any],
        enumeration_rerun_reason: str | None = None,
        previous_page_answer: str | None = None,
    ) -> tuple[dict[str, Any] | None, list[str], str | None]:
        mode = str(action.parameters.get("mode") or "").strip().lower()
        resolved_target, error = _resolve_target_with_artifact(
            state,
            action,
            target=target,
            default_artifact_dir=self.artifact_dir,
        )
        if error is not None:
            return None, [], error
        assert resolved_target is not None

        image_path = Path(str(resolved_target["artifact_path"])).expanduser()
        if not image_path.exists():
            return None, [], f"artifact_path does not exist: {image_path}"

        try:
            question = action.parameters.get("question")
        except Exception as exc:
            return None, [], str(exc)

        focus_groups: list[dict[str, Any]] = []
        focus_group_targets: list[dict[str, Any]] = []
        focus_artifacts: list[str] = []
        focus_retries: list[dict[str, Any]] = []
        focus_usage: dict[str, int] = {}
        if mode == "inspection":
            layout_error = await self._ensure_layout_for_page(
                state,
                action,
                page_index=resolved_target["page_index"],
            )
            if layout_error is not None:
                return None, [], layout_error
            try:
                focus_messages, focus_payload = self._build_focus_messages(
                    state,
                    resolved_target,
                    action,
                    image_path,
                )
                dump_jsonl_from_env(
                    "DOCCLAW_FIGURE_DEBUG_PATH",
                    {
                        "kind": "figure_input",
                        "model": self.model,
                        "document_id": state.document.document_id,
                        "action_id": action.action_id,
                        "task": state.task.to_dict(),
                        "target": resolved_target,
                        "mode": mode,
                        "question": question,
                        "artifact_path": str(image_path),
                        "user_payload": {"focus_stage": focus_payload},
                        "page_debug": _figure_target_page_debug(resolved_target),
                    },
                )
            except Exception as exc:
                return None, [], str(exc)

            try:
                focus_response, focus_retries, focus_payload_parsed = await self._call_json_with_retry(
                    messages=focus_messages,
                    parse_fn=self._parse_focus_payload,
                )
            except Exception as exc:
                dump_jsonl_from_env(
                    "DOCCLAW_FIGURE_DEBUG_PATH",
                    {
                        "kind": "figure_error",
                        "model": self.model,
                        "document_id": state.document.document_id,
                        "action_id": action.action_id,
                        "stage": "focus",
                        "target": resolved_target,
                        "mode": mode,
                        "parse_error": str(exc),
                        "page_debug": _figure_target_page_debug(resolved_target),
                    },
                )
                return None, [], f"invalid figure response: {exc}"

            page = state.get_page(resolved_target["page_index"])
            assert page is not None
            focus_groups = _normalize_focus_groups(
                focus_payload_parsed.get("focus_groups"),
                page=page,
            )
            focus_usage = dict(focus_response.usage)
            dump_jsonl_from_env(
                "DOCCLAW_FIGURE_DEBUG_PATH",
                {
                    "kind": "figure_focus_output",
                    "model": self.model,
                    "document_id": state.document.document_id,
                    "action_id": action.action_id,
                    "target": resolved_target,
                    "mode": mode,
                    "raw_content": focus_response.content,
                    "parsed_payload": focus_payload_parsed,
                    "focus_groups": focus_groups,
                    "usage": focus_response.usage,
                    "retry_count": len(focus_retries),
                    "page_debug": _figure_target_page_debug(resolved_target),
                },
            )

            focus_group_targets, focus_artifacts, focus_error = _build_focus_group_targets(
                state,
                action,
                resolved_target=resolved_target,
                focus_groups=focus_groups,
                default_artifact_dir=self.artifact_dir,
            )
            if focus_error is not None:
                return None, [], focus_error
            if not focus_group_targets:
                normalized = _no_focus_result(question=question)
                normalized["usage"] = focus_usage
                normalized["source"] = "llm"
                normalized["metadata"] = {
                    **dict(normalized.get("metadata") or {}),
                    "focus_groups": _to_jsonable(focus_groups),
                }
                dump_jsonl_from_env(
                    "DOCCLAW_FIGURE_DEBUG_PATH",
                    {
                        "kind": "figure_answer_skipped",
                        "model": self.model,
                        "document_id": state.document.document_id,
                        "action_id": action.action_id,
                        "target": resolved_target,
                        "mode": mode,
                        "question": question,
                        "reason": normalized["reason"],
                        "answer_context_mode": "crop_only",
                        "focus_groups": _to_jsonable(focus_groups),
                        "focus_crop_artifact_paths": [],
                        "page_debug": _figure_target_page_debug(resolved_target),
                    },
                )
                dump_jsonl_from_env(
                    "DOCCLAW_FIGURE_DEBUG_PATH",
                    {
                        "kind": "figure_output",
                        "model": self.model,
                        "document_id": state.document.document_id,
                        "action_id": action.action_id,
                        "target": resolved_target,
                        "mode": mode,
                        "raw_content": None,
                        "parsed_payload": None,
                        "normalized_result": normalized,
                        "usage": focus_usage,
                        "focus_retry_count": len(focus_retries),
                        "answer_retry_count": 0,
                        "short_circuit": "no_focus_groups",
                        "page_debug": _figure_output_page_debug(resolved_target, normalized),
                    },
                )
                target_artifacts = [str(image_path)] if resolved_target.get("generated_artifact") else []
                target_artifacts.extend(focus_artifacts)
                return (
                    {**resolved_target, **normalized},
                    target_artifacts,
                    None,
                )
        else:
            dump_jsonl_from_env(
                "DOCCLAW_FIGURE_DEBUG_PATH",
                {
                    "kind": "figure_input",
                    "model": self.model,
                    "document_id": state.document.document_id,
                    "action_id": action.action_id,
                    "task": state.task.to_dict(),
                    "target": resolved_target,
                    "mode": mode,
                    "question": question,
                    "artifact_path": str(image_path),
                    "user_payload": {"page_stage": {"question": question}},
                    "page_debug": _figure_target_page_debug(resolved_target),
                },
            )

        answer_messages, answer_payload = self._build_answer_messages(
            state,
            action,
            mode=mode,
            focus_groups=focus_groups,
            focus_group_targets=focus_group_targets,
            page_image_path=image_path,
            enumeration_rerun_reason=enumeration_rerun_reason,
            previous_page_answer=previous_page_answer,
        )
        dump_jsonl_from_env(
            "DOCCLAW_FIGURE_DEBUG_PATH",
            {
                "kind": "figure_answer_input",
                "model": self.model,
                "document_id": state.document.document_id,
                "action_id": action.action_id,
                "task": state.task.to_dict(),
                "target": resolved_target,
                "mode": mode,
                "question": question,
                "artifact_path": str(image_path),
                "user_payload": answer_payload,
                "answer_context_mode": "page_only" if mode == "enumeration" else "crop_only",
                "focus_crop_artifact_paths": [
                    str(item.get("artifact_path"))
                    for item in focus_group_targets
                    if item.get("artifact_path")
                ],
                "page_debug": _figure_target_page_debug(resolved_target),
            },
        )

        try:
            response, answer_retries, payload = await self._call_json_with_retry(
                messages=answer_messages,
                parse_fn=self._parse_payload,
            )
            normalized = _normalize_payload(payload, question=question)
        except Exception as exc:
            dump_jsonl_from_env(
                "DOCCLAW_FIGURE_DEBUG_PATH",
                {
                    "kind": "figure_error",
                    "model": self.model,
                    "document_id": state.document.document_id,
                    "action_id": action.action_id,
                    "stage": "answer",
                    "target": resolved_target,
                    "mode": mode,
                    "parse_error": str(exc),
                    "page_debug": _figure_target_page_debug(resolved_target),
                },
            )
            return None, [], f"invalid figure response: {exc}"

        normalized["usage"] = response.usage
        normalized["source"] = "llm"
        normalized["metadata"] = {
            **dict(normalized.get("metadata") or {}),
            "focus_groups": _to_jsonable(focus_groups),
        }
        dump_jsonl_from_env(
            "DOCCLAW_FIGURE_DEBUG_PATH",
            {
                "kind": "figure_output",
                "model": self.model,
                "document_id": state.document.document_id,
                "action_id": action.action_id,
                "target": resolved_target,
                "mode": mode,
                "raw_content": response.content,
                "parsed_payload": payload,
                "normalized_result": normalized,
                "usage": response.usage,
                "focus_retry_count": len(focus_retries),
                "answer_retry_count": len(answer_retries),
                "page_debug": _figure_output_page_debug(resolved_target, normalized),
            },
        )
        target_artifacts = [str(image_path)] if resolved_target.get("generated_artifact") else []
        target_artifacts.extend(focus_artifacts)
        return (
            {**resolved_target, **normalized},
            target_artifacts,
            None,
        )

    async def _call_json_with_retry(
        self,
        *,
        messages: list[dict[str, Any]],
        parse_fn,
        temperature: float | None = None,
    ) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
        parse_failures: list[dict[str, Any]] = []
        response = None
        payload = None
        for attempt_index in range(2):
            try:
                response = await self.provider.chat(
                    messages,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature if temperature is None else temperature,
                )
            except Exception as exc:
                raise RuntimeError(str(exc))
            if response.error:
                raise RuntimeError(response.error)
            try:
                payload = parse_fn(response.content)
                break
            except Exception as exc:
                parse_failures.append(
                    {
                        "attempt_index": attempt_index,
                        "raw_content": response.content,
                        "parse_error": str(exc),
                        "usage": response.usage,
                    }
                )
                if attempt_index == 1:
                    raise ValueError(str(exc))
        assert response is not None
        assert payload is not None
        return response, parse_failures, payload

    async def _ensure_layout_for_page(
        self,
        state: RunState,
        action: Action,
        *,
        page_index: int,
    ) -> str | None:
        page = state.get_page(page_index)
        if page is None:
            return f"unknown page_index: {page_index}"
        if page.regions:
            return None
        if self.layout_tool is None:
            return "understand_figures requires layout proposals but no layout tool is configured"
        layout_action = Action(
            action_type="parse_layout",
            target={"page_indices": [page_index]},
            parameters={},
            rationale="Auto-layout for understand_figures focus-group proposals.",
        )
        observation = await self.layout_tool.execute(state, layout_action)
        if not observation.success:
            return observation.message or "layout auto-preparation failed"
        self.layout_tool.update_state(state, layout_action, observation)
        return None

    def _build_focus_messages(
        self,
        state: RunState,
        target: dict[str, Any],
        action: Action,
        image_path: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        question = action.parameters.get("question")
        page = state.get_page(target["page_index"])
        assert page is not None
        overlay_path, candidates = _build_layout_overlay(
            state,
            action,
            page=page,
            image_path=image_path,
            default_artifact_dir=self.artifact_dir,
        )
        system_prompt = (
            "You are the DocClaw figure focus-group finder.\n"
            "Task:\n"
            "- Find the focus groups on the provided page that are needed to answer the question.\n"
            "- Use the page overlay and the original page image together.\n"
            "- user_question is the user's original full task; question is the planner's local page-level instruction for this tool call. Use user_question to preserve the intended meaning while answering the local question.\n"
            "- Return only a JSON object, with no prose or markdown before or after it. Ensure all string values are valid JSON strings.\n"
            "Scope:\n"
            "- Reason at page scope.\n"
            "- The page may contain multiple figures, charts, tables, and surrounding text.\n"
            "Candidate Region Rules:\n"
            "- The overlay image shows layout regions labeled by full region_id.\n"
            "- Visual regions are chart, image, table, seal, header_image, and footer_image.\n"
            "- Supporting regions are figure_title, chart_title, and vision_footnote.\n"
            "- Other regions are shown for context but should not be selected.\n"
            "Merge Rules:\n"
            "- Each focus group will become one merged crop.\n"
            "- Each focus group should reconstruct one logical visual unit that may have been split into multiple layout regions.\n"
            "- Each region_id may appear in at most one focus group.\n"
            "- Merge regions when they belong to the same figure, chart, table, panel set, image, or supporting title/vision_footnote for that same unit and should be inspected together in one crop.\n"
            "- Use figure_title, chart_title, and vision_footnote to decide whether a supporting region belongs with the same visual unit.\n"
            "- If the question is about a specific figure, chart, table, panel, line, node, bar, or photo, select the group containing that visual content itself.\n"
            "- Do not create a group from paragraph text or from a sentence that only mentions the figure.\n"
            "- Do not merge unrelated body text into the same group.\n"
            "- You can return multiple focus groups when one merged crop is not enough.\n"
            "Target Figure Rules:\n"
            "- If the question names a specific figure, table, panel, or numbered item such as Figure 5 or Table 3, localize that exact target on the page.\n"
            "- If the target is not present on the page, return an empty focus_groups list.\n"
            "Return JSON with this shape:\n"
            "{\n"
            '  "question": "repeated input question",\n'
            '  "focus_groups": [\n'
            "    {\n"
            '      "region_ids": ["p7_image_1", "p7_figure_title_1"],\n'
            '      "reason": "short explanation of why these visual regions should be read together"\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )
        user_payload = plannerize_page_refs({
            "user_question": state.task.prompt.strip(),
            "question": question,
            "candidates": candidates,
        }, document=state.document)
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False, indent=2)},
                    {"type": "text", "text": "Page overlay with labeled layout regions:"},
                    {"type": "image_url", "image_url": {"url": _image_data_url(overlay_path)}},
                    {"type": "text", "text": "Original page image:"},
                    {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                ],
            },
        ], user_payload

    def _build_answer_messages(
        self,
        state: RunState,
        action: Action,
        *,
        mode: str,
        focus_groups: list[dict[str, Any]],
        focus_group_targets: list[dict[str, Any]],
        page_image_path: Path,
        enumeration_rerun_reason: str | None = None,
        previous_page_answer: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        question = action.parameters.get("question")
        if mode == "enumeration":
            system_prompt = (
                "You are the DocClaw figure understanding tool.\n"
                "Task:\n"
                "- Answer the required question for the provided page only.\n"
                "- Use the full page image to determine this page's contribution to the final enumeration.\n"
                "- user_question is the user's original full task; question is the planner's local page-level instruction for this tool call. Use user_question to preserve the intended meaning while answering the local question.\n"
                "- Return only a JSON object, with no prose or markdown before or after it. Ensure all string values are valid JSON strings.\n"
                "Scope:\n"
                "- The full page image is the primary input.\n"
                "- Reason at page scope rather than crop scope.\n"
                "Enumeration Rules:\n"
                "- First identify the logical item being counted on this page.\n"
                "- Then determine how many items on this page directly satisfy the condition stated in the question.\n"
                "- Inspect carefully. If this page does not contain any matching item, return `0` rather than forcing a match.\n"
                "- Count only items that directly satisfy the stated condition.\n"
                "- Do not rewrite the condition into its opposite, a negated opposite, or an exclusivity shortcut.\n"
                "- If the condition is about missing opinions, unavailable opinions, or no-opinion entries for a group, count only charts that directly show that condition.\n"
                "- Do not infer a match merely because an item includes multiple groups or is not exclusive to some other group.\n"
                "- When a previous_page_answer and a verification_reason are provided, treat them as a re-check hint. Recount from the page image itself, and pay attention to the verification_reason during the recount.\n"
                "Output Rules:\n"
                "- State the counting unit explicitly before giving the final count.\n"
                "- The answer should directly answer the question for this page.\n"
                "- The answer must be this page's contribution as `0`, `1`, or `N` matching items on this page.\n"
                "- The reason should be short and grounded in the full page.\n"
                "Return JSON with this shape:\n"
                "{\n"
                '  "counting_unit": "logical unit being counted on this page",\n'
                '  "answer": "0, 1, or N for this page",\n'
                '  "reason": "short explanation",\n'
                '  "question": "repeated input question"\n'
                "}\n"
            )
            user_payload = {
                "user_question": state.task.prompt.strip(),
                "question": question,
                "mode": mode,
            }
            if previous_page_answer is not None:
                user_payload["previous_page_answer"] = previous_page_answer
            if enumeration_rerun_reason is not None:
                user_payload["verification_reason"] = enumeration_rerun_reason
            user_content: list[dict[str, Any]] = [
                {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False, indent=2)},
            ]
            if previous_page_answer is not None or enumeration_rerun_reason is not None:
                verification_note_lines = ["Re-check this page carefully."]
                if previous_page_answer is not None:
                    verification_note_lines.append(
                        f"Previous page contribution: {previous_page_answer}"
                    )
                if enumeration_rerun_reason is not None:
                    verification_note_lines.append(
                        f"Why this page is suspicious: {enumeration_rerun_reason}"
                    )
                user_content.append(
                    {
                        "type": "text",
                        "text": "\n".join(verification_note_lines),
                    }
                )
            user_content.extend(
                [
                    {"type": "text", "text": "Full page context:"},
                    {"type": "image_url", "image_url": {"url": _image_data_url(page_image_path)}},
                ]
            )
            return [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_content,
                },
            ], user_payload

        system_prompt = (
            "You are the DocClaw figure understanding tool.\n"
            "Task:\n"
            "- Answer the required question for the provided page only.\n"
            "- Use the provided focus-area crops for detailed inspection of the most relevant visual content.\n"
            "- user_question is the user's original full task; question is the planner's local page-level instruction for this tool call. Use user_question to preserve the intended meaning while answering the local question.\n"
            "- Return only a JSON object, with no prose or markdown before or after it. Ensure all string values are valid JSON strings.\n"
            "Scope:\n"
            "- The focus-area crops come from one page and were chosen because they are relevant to the question.\n"
            "Inspection Rules:\n"
            "- If the question depends on exact visual properties, or exact items, inspect the relevant elements one by one before answering.\n"
            "- For example, identify the relevant lines, nodes, markers, labels, or highlighted sets first, then compare them, then answer from that comparison.\n"
            "- Do not assume one focus-area equals one answer unit. A focus-area may contain zero, one, or multiple logical answer units.\n"
            "- Do not answer from a coarse visual impression when the question requires exact item-level reading.\n"
            "Output Rules:\n"
            "- The answer should directly answer the question for this page.\n"
            "- The answer must be for the page as a whole.\n"
            "- The reason should be short and grounded in the provided focus areas.\n"
            "Return JSON with this shape:\n"
            "{\n"
            '  "answer": "direct answer for this page",\n'
            '  "reason": "short explanation",\n'
            '  "question": "repeated input question"\n'
            "}\n"
        )
        user_payload = {
            "user_question": state.task.prompt.strip(),
            "question": question,
            "mode": mode,
            "focus_groups": [
                {
                    "region_ids": item["region_ids"],
                    "reason": item.get("reason"),
                }
                for item in focus_groups
            ],
        }
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ]
        for index, item in enumerate(focus_group_targets, start=1):
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        f"Focus group {index}: "
                        f"region_ids={item.get('region_ids') or []}, "
                        f"reason={item.get('reason') or 'n/a'}"
                    ),
                }
            )
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(Path(str(item['artifact_path'])))},
                }
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ], user_payload

    async def _verify_enumeration_results(
        self,
        state: RunState,
        action: Action,
        *,
        targets: list[dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> tuple[dict[int, str], list[str], str | None]:
        messages, payload = self._build_enumeration_verification_messages(
            state,
            action,
            targets=targets,
            results=results,
        )
        dump_jsonl_from_env(
            "DOCCLAW_FIGURE_DEBUG_PATH",
            {
                "kind": "figure_enumeration_verification_input",
                "model": self.model,
                "document_id": state.document.document_id,
                "action_id": action.action_id,
                "task": state.task.to_dict(),
                "user_payload": payload,
                "page_debug": _figure_verification_input_page_debug(
                    targets=targets,
                    results=results,
                ),
            },
        )
        try:
            response, retries, payload = await self._call_json_with_retry(
                messages=messages,
                parse_fn=self._parse_enumeration_verification_payload,
            )
            suspect_reasons = _normalize_enumeration_verification_payload(
                payload,
                targets=targets,
                document=state.document,
            )
        except Exception as exc:
            dump_jsonl_from_env(
                "DOCCLAW_FIGURE_DEBUG_PATH",
                {
                    "kind": "figure_enumeration_verification_error",
                    "model": self.model,
                    "document_id": state.document.document_id,
                    "action_id": action.action_id,
                    "parse_error": str(exc),
                },
            )
            return {}, [], f"invalid enumeration verification response: {exc}"

        dump_jsonl_from_env(
            "DOCCLAW_FIGURE_DEBUG_PATH",
            {
                "kind": "figure_enumeration_verification_output",
                "model": self.model,
                "document_id": state.document.document_id,
                "action_id": action.action_id,
                "raw_content": response.content,
                "parsed_payload": payload,
                "suspect_reasons": {
                    page_id_from_index(page_index, document=state.document): reason
                    for page_index, reason in suspect_reasons.items()
                },
                "usage": response.usage,
                "retry_count": len(retries),
                "page_debug": _figure_verification_output_page_debug(
                    targets=targets,
                    results=results,
                    suspect_page_indices=list(suspect_reasons.keys()),
                ),
            },
        )
        return suspect_reasons, [], None

    def _build_enumeration_verification_messages(
        self,
        state: RunState,
        action: Action,
        *,
        targets: list[dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        question = action.parameters.get("question")
        page_contributions = []
        for result in results:
            page_index = result.get("page_index")
            if not isinstance(page_index, int):
                continue
            page_contributions.append(
                {
                    "page_id": page_id_from_index(page_index, document=state.document),
                    "answer": _optional_text(result.get("answer")),
                    "reason": _optional_text(result.get("reason")),
                    "counting_unit": _optional_text((result.get("metadata") or {}).get("counting_unit")),
                }
            )
        system_prompt = (
            "You are the DocClaw enumeration verification tool.\n"
            "You are given an original question, a local page-level question, the candidate pages, and each page's current contribution.\n"
            "Your job is to identify pages whose current contribution may be suspicious or wrong.\n"
            "user_question is the user's original full task; question is the planner's local page-level instruction for this tool call. Use user_question to preserve the intended meaning while checking the local page contributions.\n"
            "Mark a page as suspicious only when there is a clear reason it may have missed matching items, overcounted, or used the wrong counting unit.\n"
            "If a page contribution looks plausible, leave it out of suspect_pages.\n"
            "suspect_pages may contain zero, one, or multiple pages.\n"
            "Return only JSON with this shape:\n"
            "{\n"
            '  "suspect_pages": [\n'
            "    {\n"
            '      "page_id": "page_x",\n'
            '      "reason": "short reason why this page should be re-checked"\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )
        user_payload = {
            "user_question": state.task.prompt.strip(),
            "question": question,
            "mode": "enumeration_verification",
            "page_contributions": page_contributions,
        }
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ]
        for target, result in zip(targets, results, strict=True):
            page_index = target.get("page_index")
            assert isinstance(page_index, int)
            page_id = page_id_from_index(page_index, document=state.document)
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        f"Candidate {page_id}: current contribution={_optional_text(result.get('answer')) or 'n/a'}; "
                        f"reason={_optional_text(result.get('reason')) or 'n/a'}"
                    ),
                }
            )
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(Path(str(target['artifact_path'])).expanduser())},
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
            candidate = candidate[start : end + 1]
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                payload = json.loads(_ensure_json_format(candidate))
        if not isinstance(payload, dict):
            raise ValueError("response JSON must be an object")
        return payload

    @classmethod
    def _parse_focus_payload(cls, content: str | None) -> dict[str, Any]:
        payload = cls._parse_payload(content)
        focus_groups = payload.get("focus_groups")
        if not isinstance(focus_groups, list):
            raise ValueError("focus_groups must be a list")
        return payload

    @classmethod
    def _parse_enumeration_verification_payload(cls, content: str | None) -> dict[str, Any]:
        payload = cls._parse_payload(content)
        suspect_pages = payload.get("suspect_pages")
        if not isinstance(suspect_pages, list):
            raise ValueError("suspect_pages must be a list")
        return payload


def _normalize_focus_groups(
    value: Any,
    *,
    page,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    known_region_ids = {region.region_id for region in page.regions}
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        region_ids_value = item.get("region_ids")
        if not isinstance(region_ids_value, list):
            continue
        region_ids: list[str] = []
        for region_id in region_ids_value:
            if not isinstance(region_id, str):
                continue
            cleaned = region_id.strip()
            if not cleaned or cleaned not in known_region_ids or cleaned in region_ids:
                continue
            region_ids.append(cleaned)
        if not region_ids:
            continue
        normalized.append(
            {
                "region_ids": region_ids,
                "reason": _optional_text(item.get("reason")),
            }
        )
    return normalized


def _build_focus_group_targets(
    state: RunState,
    action: Action,
    *,
    resolved_target: dict[str, Any],
    focus_groups: list[dict[str, Any]],
    default_artifact_dir: Path | None,
) -> tuple[list[dict[str, Any]], list[str], str | None]:
    targets: list[dict[str, Any]] = []
    artifacts: list[str] = []
    page = state.get_page(resolved_target["page_index"])
    assert page is not None
    for index, focus_group in enumerate(focus_groups, start=1):
        boxes: list[tuple[float, float, float, float]] = []
        for region_id in focus_group["region_ids"]:
            region = state.get_region(region_id)
            if region is None:
                continue
            boxes.append(tuple(float(v) for v in region.bbox))
        if not boxes:
            continue
        target = {
            "page_index": resolved_target["page_index"],
            "bbox": list(_union_bboxes(boxes, page_width=page.width, page_height=page.height)),
            "coordinate_space": "pixel",
            "region_id": f"focus_{index}",
            "reason": focus_group.get("reason"),
            "region_ids": list(focus_group["region_ids"]),
        }
        resolved_focus_target, error = _resolve_target_with_artifact(
            state,
            action,
            target=target,
            default_artifact_dir=default_artifact_dir,
        )
        if error is not None:
            return [], [], error
        assert resolved_focus_target is not None
        targets.append(resolved_focus_target)
        if resolved_focus_target.get("generated_artifact"):
            artifacts.append(str(resolved_focus_target["artifact_path"]))
    return targets, artifacts, None


def _build_layout_overlay(
    state: RunState,
    action: Action,
    *,
    page,
    image_path: Path,
    default_artifact_dir: Path | None,
) -> tuple[Path, list[dict[str, Any]]]:
    artifact_dir = _resolve_artifact_dir(state, action, default_artifact_dir)
    if artifact_dir is None:
        raise ValueError("understand_figures overlay requires artifact_dir")
    overlay_path = artifact_dir / f"{action.action_id}_page_{page.page_number}_layout_overlay.png"
    candidates: list[dict[str, Any]] = []
    _render_layout_overlay(
        page=page,
        image_path=image_path,
        output_path=overlay_path,
        candidates=candidates,
    )
    return overlay_path, candidates


def _render_layout_overlay(
    *,
    page,
    image_path: Path,
    output_path: Path,
    candidates: list[dict[str, Any]],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    main_color = (220, 38, 38)
    supporting_color = (37, 99, 235)
    other_color = (107, 114, 128)

    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        for region in page.regions:
            role = _focus_role_for_label(region.label or region.raw_type or "")
            color = {
                "main_visual": main_color,
                "supporting": supporting_color,
                "other": other_color,
            }[role]
            x0, y0, x1, y1 = (int(round(v)) for v in region.bbox)
            draw.rectangle((x0, y0, x1, y1), outline=color, width=3 if role != "other" else 2)
            text = str(region.region_id)
            tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), text, font=font)
            text_width = tx1 - tx0
            text_height = ty1 - ty0
            label_x = max(0, min(canvas.width - text_width - 4, x0))
            label_y = y0 - text_height - 6
            if label_y < 0:
                label_y = min(canvas.height - text_height - 4, y1 + 4)
            draw.rectangle(
                (label_x - 2, label_y - 2, label_x + text_width + 2, label_y + text_height + 2),
                fill=(255, 255, 255),
            )
            draw.text((label_x, label_y), text, fill=color, font=font)
            candidates.append(
                {
                    "region_id": region.region_id,
                    "label": str(region.label or region.raw_type or ""),
                    "role": role,
                    "bbox": [
                        region.bbox[0] / page.width,
                        region.bbox[1] / page.height,
                        region.bbox[2] / page.width,
                        region.bbox[3] / page.height,
                    ],
                }
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)


def _focus_role_for_label(label: str) -> str:
    normalized = str(label).strip().lower()
    if normalized in _MAIN_VISUAL_LABELS:
        return "main_visual"
    if normalized in _SUPPORTING_VISUAL_LABELS:
        return "supporting"
    return "other"


def _union_bboxes(
    boxes: list[tuple[float, float, float, float]],
    *,
    page_width: int,
    page_height: int,
) -> tuple[int, int, int, int]:
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    return clamp_bbox(
        (
            int(round(x0)),
            int(round(y0)),
            int(round(x1)),
            int(round(y1)),
        ),
        page_width,
        page_height,
    )

def _ensure_json_format(candidate: str) -> str:
    repaired = (
        candidate
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )

    chars: list[str] = []
    in_string = False
    escape = False
    for index, char in enumerate(repaired):
        if escape:
            chars.append(char)
            escape = False
            continue
        if char == "\\":
            chars.append(char)
            escape = True
            continue
        if char != '"':
            chars.append(char)
            continue
        if not in_string:
            chars.append(char)
            in_string = True
            continue

        if _looks_like_string_terminator(repaired, index + 1):
            chars.append(char)
            in_string = False
        else:
            chars.append('\\"')
    return "".join(chars)


def _next_significant_char(text: str, start: int) -> str | None:
    for index in range(start, len(text)):
        char = text[index]
        if not char.isspace():
            return char
    return None


def _looks_like_string_terminator(text: str, start: int) -> bool:
    next_significant = _next_significant_char(text, start)
    if next_significant is None:
        return True
    if next_significant in {"}", "]", ":"}:
        return True
    if next_significant != ",":
        return False

    comma_index = start
    while comma_index < len(text) and text[comma_index].isspace():
        comma_index += 1
    if comma_index >= len(text) or text[comma_index] != ",":
        return False

    after_comma = _next_significant_char(text, comma_index + 1)
    return after_comma in {'"', "}", "]"}


def _resolve_target_with_artifact(
    state: RunState,
    action: Action,
    *,
    target: dict[str, Any],
    default_artifact_dir: Path | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if "artifact_path" in target:
        return target, None

    page_index = target.get("page_index")
    if not isinstance(page_index, int):
        return None, "figure target must include page_index"
    page = state.get_page(page_index)
    if page is None:
        return None, f"unknown page_index: {page_index}"
    if not page.image_path:
        return None, f"page {page.page_index} has no image_path"

    image_path = Path(page.image_path).expanduser()
    if not image_path.exists():
        return None, f"page image_path does not exist: {image_path}"

    bbox = target.get("bbox")
    coordinate_space = target.get("coordinate_space", "pixel")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None, "figure target bbox must contain four coordinates"

    pixel_bbox, bbox_error = _to_pixel_bbox(
        tuple(float(value) for value in bbox),
        str(coordinate_space),
        page.width,
        page.height,
    )
    if bbox_error is not None:
        return None, bbox_error

    artifact_dir = _resolve_artifact_dir(state, action, default_artifact_dir)
    if artifact_dir is None:
        return None, "figure region target requires artifact_dir"
    region_fragment = str(target.get("region_id") or f"page_{page_index}").strip()
    safe_region_fragment = re.sub(r"[^A-Za-z0-9._-]+", "_", region_fragment) or f"page_{page_index}"
    output_path = artifact_dir / f"{action.action_id}_{safe_region_fragment}_figure.png"
    _render_crop(image_path, output_path, pixel_bbox)
    return {
        **target,
        "pixel_bbox": list(pixel_bbox),
        "artifact_path": str(output_path),
        "generated_artifact": True,
    }, None


def _render_crop(
    source_path: Path,
    output_path: Path,
    bbox: tuple[int, int, int, int],
) -> None:
    from PIL import Image

    with Image.open(source_path) as image:
        width, height = image.size
        x0, y0, x1, y1 = clamp_bbox(bbox, width, height)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("bbox is outside the page image")
        crop = image.crop((x0, y0, x1, y1))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path)


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


def _normalize_payload(payload: dict[str, Any], *, question: Any) -> dict[str, Any]:
    answer = _optional_text(payload.get("answer"))
    reason = _optional_text(payload.get("reason"))
    counting_unit = _optional_text(payload.get("counting_unit"))
    normalized_question = _optional_text(payload.get("question")) or _optional_text(question)
    if answer is None:
        raise ValueError("response JSON must contain a non-empty answer")

    metadata = _to_jsonable(payload.get("metadata") or {})
    if counting_unit is not None:
        metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}
        metadata = {
            **metadata_dict,
            "counting_unit": counting_unit,
        }

    return {
        "answer": answer,
        "reason": reason,
        "question": normalized_question,
        "confidence": _optional_float(payload.get("confidence")),
        "metadata": metadata,
    }


def _normalize_enumeration_verification_payload(
    payload: dict[str, Any],
    *,
    targets: list[dict[str, Any]],
    document: Any,
) -> dict[int, str]:
    allowed_page_ids = {
        page_id_from_index(target["page_index"], document=document): target["page_index"]
        for target in targets
        if isinstance(target.get("page_index"), int)
    }
    normalized: dict[int, str] = {}
    for item in payload.get("suspect_pages") or []:
        if not isinstance(item, dict):
            continue
        page_id = _optional_text(item.get("page_id"))
        reason = _optional_text(item.get("reason"))
        if page_id is None or reason is None:
            continue
        page_index = allowed_page_ids.get(page_id)
        if page_index is None:
            continue
        normalized[page_index] = reason
    return normalized


def _no_focus_result(*, question: Any) -> dict[str, Any]:
    normalized_question = _optional_text(question)
    return {
        "answer": "No relevant figure content for this question was found on this page.",
        "reason": (
            "The focus-selection stage did not identify any chart, figure, or supporting "
            "visual region relevant to the question on this page."
        ),
        "question": normalized_question,
        "confidence": None,
        "metadata": {},
    }


def _normalize_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = _optional_text(item)
        if text:
            items.append(text)
    return items


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _figure_target_page_debug(target: dict[str, Any]) -> dict[str, list[int]]:
    try:
        page_indices = _page_indices_from_target(target)
        return {
            "target_page_indices": page_indices,
        }
    except Exception:
        return {
            "target_page_indices": [],
        }


def _figure_output_page_debug(
    target: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, list[int]]:
    try:
        output_page_indices = _page_indices_from_target(result)
        if not output_page_indices:
            output_page_indices = _page_indices_from_target(target)
        return {
            **_figure_target_page_debug(target),
            "output_page_indices": output_page_indices,
        }
    except Exception:
        return {
            **_figure_target_page_debug(target),
            "output_page_indices": [],
        }


def _figure_verification_input_page_debug(
    *,
    targets: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, list[int]]:
    try:
        return {
            "target_page_indices": _page_indices_from_targets(targets),
            "output_page_indices": _page_indices_from_targets(results),
        }
    except Exception:
        return {
            "target_page_indices": [],
            "output_page_indices": [],
        }


def _figure_verification_output_page_debug(
    *,
    targets: list[dict[str, Any]],
    results: list[dict[str, Any]],
    suspect_page_indices: list[int],
) -> dict[str, list[int]]:
    try:
        return {
            **_figure_verification_input_page_debug(targets=targets, results=results),
            "suspect_page_indices": _dedupe_page_indices(suspect_page_indices),
        }
    except Exception:
        return {
            **_figure_verification_input_page_debug(targets=targets, results=results),
            "suspect_page_indices": [],
        }


def _page_indices_from_target(target: dict[str, Any]) -> list[int]:
    page_index = target.get("page_index")
    if isinstance(page_index, int):
        return [page_index]
    page_indices = target.get("page_indices")
    if isinstance(page_indices, list):
        return _dedupe_page_indices(page_indices)
    return []


def _page_indices_from_targets(targets: list[dict[str, Any]]) -> list[int]:
    values: list[int] = []
    for target in targets:
        values.extend(_page_indices_from_target(target))
    return _dedupe_page_indices(values)


def _dedupe_page_indices(values: list[int]) -> list[int]:
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
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _to_pixel_bbox(
    bbox: tuple[float, float, float, float],
    coordinate_space: str,
    page_width: int | None,
    page_height: int | None,
) -> tuple[tuple[int, int, int, int], str | None]:
    if len(bbox) != 4:
        return (0, 0, 0, 0), "bbox must contain four coordinates"

    if coordinate_space == "relative":
        if page_width is None or page_height is None:
            return (0, 0, 0, 0), "relative bbox requires page width and height"
        values = (
            float(bbox[0]) * page_width,
            float(bbox[1]) * page_height,
            float(bbox[2]) * page_width,
            float(bbox[3]) * page_height,
        )
    elif coordinate_space == "pixel":
        values = tuple(float(value) for value in bbox)
    else:
        return (0, 0, 0, 0), f"unsupported coordinate_space: {coordinate_space}"

    x0, y0, x1, y1 = (int(round(value)) for value in values)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0), "bbox must have positive width and height"

    if page_width is not None and page_height is not None:
        x0, y0, x1, y1 = clamp_bbox((x0, y0, x1, y1), page_width, page_height)
        if x1 <= x0 or y1 <= y0:
            return (0, 0, 0, 0), "bbox is outside the page bounds"

    return (x0, y0, x1, y1), None


def clamp_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        max(0, min(width, x0)),
        max(0, min(height, y0)),
        max(0, min(width, x1)),
        max(0, min(height, y1)),
    )


def _resolve_artifact_dir(
    state: RunState,
    action: Action,
    default_artifact_dir: Path | None,
) -> Path | None:
    value = action.parameters.get("artifact_dir")
    if value is None:
        value = state.metadata.get("artifact_dir")
    if value is None:
        value = state.document.metadata.get("artifact_dir")
    if value is None:
        return default_artifact_dir
    return Path(str(value)).expanduser()
