"""PaddleOCR-VL-backed OCR implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docclaw.agent.tool.ocr.ocr import (
    OcrTool,
    PixelBBox,
    _batch_ocr_message,
    _existing_ocr_metadata,
    _existing_target_ocr,
    _ocr_message,
    _page_ocr_blocks_from_raw,
    _resolve_ocr_targets,
    clamp_bbox,
)
from docclaw.agent.tool.quiet import suppress_vendor_init_output
from docclaw.agent.utils import Action, Observation, RunState, has_searchable_text


class PaddleOCRVLTool(OcrTool):
    """Recognize region content with PaddleOCR-VL prompts."""

    def __init__(
        self,
        *,
        pipeline: Any | None = None,
        pipeline_kwargs: dict[str, Any] | None = None,
        predict_kwargs: dict[str, Any] | None = None,
        artifact_dir: str | Path | None = None,
        allow_chart_region_ocr: bool = False,
    ) -> None:
        super().__init__(artifact_dir=artifact_dir)
        self._pipeline = pipeline
        self.allow_chart_region_ocr = allow_chart_region_ocr
        self.pipeline_kwargs = {
            "device": "gpu:0",
            **dict(pipeline_kwargs or {}),
        }
        self.predict_kwargs = {
            "use_layout_detection": False,
            "prompt_label": "ocr",
            **dict(predict_kwargs or {}),
        }

    @property
    def source_name(self) -> str:
        return "paddleocr_vl"

    async def execute(self, state: RunState, action: Action) -> Observation:
        force = bool(action.parameters.get("force", False))
        targets, error = _resolve_ocr_targets(state, action)
        if error is not None:
            return self.error(action, error)
        assert targets is not None

        results: list[dict[str, Any] | None] = [None] * len(targets)
        artifacts: list[str] = []
        pending_batches: dict[str, dict[str, Any]] = {}

        for target_index, target in enumerate(targets):
            result, target_artifacts, error = self._prepare_single_target_result(
                state,
                action,
                target,
                force=force,
            )
            if error is not None:
                return self.error(action, error)
            artifacts.extend(target_artifacts)
            if result is not None:
                results[target_index] = result
                continue

            source_path, target_artifacts, error = self._prepare_ocr_image(state, action, target)
            if error is not None:
                return self.error(action, error)
            assert source_path is not None
            artifacts.extend(target_artifacts)

            predict_kwargs = self._predict_kwargs_for_target(state, target)
            if not _is_batchable_region_target(target):
                result, error = self._recognize_prepared_target(
                    target,
                    source_path=source_path,
                    predict_kwargs=predict_kwargs,
                )
                if error is not None:
                    return self.error(action, error)
                results[target_index] = result
                continue

            batch_key = _predict_kwargs_key(predict_kwargs)
            batch = pending_batches.setdefault(
                batch_key,
                {
                    "predict_kwargs": predict_kwargs,
                    "items": [],
                },
            )
            batch["items"].append(
                {
                    "target_index": target_index,
                    "target": target,
                    "source_path": source_path,
                }
            )

        for batch in pending_batches.values():
            batch_error = self._run_region_batch(batch["items"], batch["predict_kwargs"], results)
            if batch_error is not None:
                return self.error(action, batch_error)

        completed_results = [result for result in results if result is not None]
        source_counts: dict[str, int] = {}
        for result in completed_results:
            source = str(result.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1

        if len(completed_results) == 1:
            message = _ocr_message(completed_results[0], source_counts)
        else:
            message = _batch_ocr_message(completed_results, source_counts)

        return Observation(
            action_id=action.action_id,
            success=True,
            data={
                "results": completed_results,
                "sources": source_counts,
            },
            message=message,
            artifacts=artifacts,
        )

    def recognize_text(self, image_path: Path) -> dict[str, Any]:
        return self._recognize_with_predict_kwargs(image_path, self.predict_kwargs)

    def _recognize_with_predict_kwargs(
        self,
        image_path: Path,
        predict_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        with suppress_vendor_init_output():
            result = self._get_pipeline().predict(str(image_path), **predict_kwargs)
        result_item = _unwrap_result_item(result)
        return _normalize_paddleocr_vl_result(result_item)

    def _recognize_batch_with_predict_kwargs(
        self,
        image_paths: list[Path],
        predict_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        with suppress_vendor_init_output():
            result = self._get_pipeline().predict(
                [str(image_path) for image_path in image_paths],
                **predict_kwargs,
            )
        if not isinstance(result, list):
            result = list(result)
        if len(result) != len(image_paths):
            raise ValueError(
                "batched PaddleOCR-VL result count mismatch: "
                f"expected {len(image_paths)}, got {len(result)}"
            )
        return [_normalize_paddleocr_vl_result(item) for item in result]

    def _prepare_single_target_result(
        self,
        state: RunState,
        action: Action,
        target: dict[str, Any],
        *,
        force: bool,
    ) -> tuple[dict[str, Any] | None, list[str], str | None]:
        if _should_skip_image_like_region_target(
            state,
            target,
            allow_chart_region_ocr=self.allow_chart_region_ocr,
        ):
            skipped = _build_skipped_image_like_result(target)
            if skipped is not None:
                return skipped, [], None

        existing_payload = _existing_target_ocr(state, target)
        if not force and existing_payload is not None:
            return (
                {
                    **target["data"],
                    "text": existing_payload["text"],
                    "source": existing_payload.get("source", "state"),
                    "confidence": existing_payload["confidence"],
                    **_existing_ocr_metadata(existing_payload),
                },
                [],
                None,
            )
        return None, [], None

    def _recognize_prepared_target(
        self,
        target: dict[str, Any],
        *,
        source_path: Path,
        predict_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        try:
            result = self._recognize_with_predict_kwargs(source_path, predict_kwargs)
        except Exception as exc:
            return {}, f"ocr failed: {exc}"
        return self._result_data_for_target(target, result), None

    def _run_region_batch(
        self,
        batch_items: list[dict[str, Any]],
        predict_kwargs: dict[str, Any],
        results: list[dict[str, Any] | None],
    ) -> str | None:
        if not batch_items:
            return None
        if len(batch_items) == 1:
            item = batch_items[0]
            result, error = self._recognize_prepared_target(
                item["target"],
                source_path=item["source_path"],
                predict_kwargs=predict_kwargs,
            )
            if error is not None:
                return error
            results[int(item["target_index"])] = result
            return None

        source_paths = [item["source_path"] for item in batch_items]
        try:
            batch_results = self._recognize_batch_with_predict_kwargs(
                source_paths,
                predict_kwargs,
            )
        except Exception:
            batch_results = []
            for source_path in source_paths:
                try:
                    batch_results.append(
                        self._recognize_with_predict_kwargs(source_path, predict_kwargs)
                    )
                except Exception as exc:
                    return f"ocr failed: {exc}"

        for item, result in zip(batch_items, batch_results):
            results[int(item["target_index"])] = self._result_data_for_target(
                item["target"],
                result,
            )
        return None

    def _result_data_for_target(
        self,
        target: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        text = result.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        confidence = result.get("confidence")
        if isinstance(target["data"].get("region_id"), str):
            confidence = None

        data = {
            **target["data"],
            "text": text,
            "source": self.source_name,
            "confidence": confidence,
        }
        if not isinstance(target["data"].get("region_id"), str):
            page_ocr_markdown = _page_ocr_markdown_from_raw(result.get("raw"))
            if page_ocr_markdown is not None:
                data["page_ocr_markdown"] = page_ocr_markdown
            page_ocr_blocks = _page_ocr_blocks_from_raw(
                (result.get("raw") or {}).get("parsing_res_list")
                if isinstance(result.get("raw"), dict)
                else None
            )
            if page_ocr_blocks:
                data["page_ocr_blocks"] = page_ocr_blocks
        return data

    def _execute_single_target(
        self,
        state: RunState,
        action: Action,
        target: dict[str, Any],
        *,
        force: bool,
    ) -> tuple[dict[str, Any] | None, list[str], str | None]:
        prepared_result, artifacts, error = self._prepare_single_target_result(
            state,
            action,
            target,
            force=force,
        )
        if error is not None or prepared_result is not None:
            return prepared_result, artifacts, error

        source_path, artifacts, error = self._prepare_ocr_image(state, action, target)
        if error is not None:
            return None, [], error
        assert source_path is not None

        predict_kwargs = self._predict_kwargs_for_target(state, target)
        result, error = self._recognize_prepared_target(
            target,
            source_path=source_path,
            predict_kwargs=predict_kwargs,
        )
        if error is not None:
            return None, [], error
        return result, artifacts, None

    def _predict_kwargs_for_target(
        self,
        state: RunState,
        target: dict[str, Any],
    ) -> dict[str, Any]:
        data = target.get("data")
        base_kwargs = dict(self.predict_kwargs)
        if not isinstance(data, dict):
            return base_kwargs

        region_id = data.get("region_id")
        if isinstance(region_id, str):
            region = state.get_region(region_id)
            return {
                **base_kwargs,
                "use_layout_detection": False,
                "prompt_label": _prompt_label_for_region_type(region.type if region is not None else None),
            }

        page_index = data.get("page_index")
        if isinstance(page_index, int):
            page_kwargs = dict(base_kwargs)
            page_kwargs["use_layout_detection"] = True
            page_kwargs.pop("prompt_label", None)
            return page_kwargs

        return base_kwargs

    def render_crop(
        self,
        source_path: Path,
        output_path: Path,
        bbox: PixelBBox,
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

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            from paddleocr import PaddleOCRVL

            with suppress_vendor_init_output():
                self._pipeline = PaddleOCRVL(**self.pipeline_kwargs)
        return self._pipeline


def _is_batchable_region_target(target: dict[str, Any]) -> bool:
    data = target.get("data")
    return isinstance(data, dict) and isinstance(data.get("region_id"), str)


def _predict_kwargs_key(predict_kwargs: dict[str, Any]) -> str:
    return json.dumps(predict_kwargs, sort_keys=True, default=str)


def _normalize_paddleocr_vl_result(result_item: Any) -> dict[str, Any]:
    result_data = _to_jsonable(_result_to_dict(result_item))
    markdown_data = _extract_markdown_data(result_item)
    if markdown_data is not None:
        result_data["markdown"] = _to_jsonable(markdown_data)
    text = _extract_text(result_item, result_data)
    scores = _extract_scores(result_data)
    confidence = sum(scores) / len(scores) if scores else None
    return {
        "text": text,
        "confidence": confidence,
        "raw": result_data,
    }


def _page_ocr_markdown_from_raw(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    markdown = raw.get("markdown")
    if not isinstance(markdown, dict):
        return None
    for key in ("markdown_texts", "markdown_text", "text"):
        value = markdown.get(key)
        if has_searchable_text(value):
            assert isinstance(value, str)
            return value
    return None


def _unwrap_result_item(result: Any) -> Any:
    if isinstance(result, list):
        if not result:
            return {}
        return result[0]
    return result


def _result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result

    json_attr = getattr(result, "json", None)
    if isinstance(json_attr, dict):
        maybe_res = json_attr.get("res")
        if isinstance(maybe_res, dict):
            return maybe_res
        return json_attr

    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, dict):
            return data

    raise TypeError(f"unsupported PaddleOCRVL result type: {type(result).__name__}")


def _extract_markdown_data(result: Any) -> dict[str, Any] | None:
    markdown = getattr(result, "markdown", None)
    if isinstance(markdown, dict):
        return markdown
    return None


def _extract_text(result_item: Any, result_data: Any) -> str:
    lines = _extract_parsing_res_list_lines(result_item)
    if not lines:
        lines = _extract_parsing_res_list_lines(result_data)
    return "\n".join(lines)


def _extract_parsing_res_list_lines(value: Any) -> list[str]:
    lines: list[str] = []
    _append_parsing_res_list_lines(value, lines)
    deduped: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        if deduped and deduped[-1] == line:
            continue
        deduped.append(line)
    return deduped


def _append_parsing_res_list_lines(value: Any, lines: list[str]) -> None:
    if isinstance(value, dict):
        parsing_res_list = value.get("parsing_res_list")
        if isinstance(parsing_res_list, list):
            for item in parsing_res_list:
                _append_parsing_res_list_lines(item, lines)
            return

        block_content = value.get("block_content")
        if isinstance(block_content, str) and block_content.strip():
            lines.append(block_content.strip())
            return

        for nested in value.values():
            _append_parsing_res_list_lines(nested, lines)
        return

    if isinstance(value, list):
        for item in value:
            _append_parsing_res_list_lines(item, lines)
        return

    content = getattr(value, "content", None)
    if isinstance(content, str) and content.strip():
        lines.append(content.strip())


def _extract_scores(value: Any) -> list[float]:
    scores: list[float] = []
    _append_scores(value, scores)
    return scores


def _append_scores(value: Any, scores: list[float]) -> None:
    if isinstance(value, dict):
        rec_scores = value.get("rec_scores")
        if isinstance(rec_scores, list):
            for item in rec_scores:
                score = _to_float(item)
                if score is not None:
                    scores.append(score)
        score = _to_float(value.get("score"))
        if score is not None:
            scores.append(score)
        confidence = _to_float(value.get("confidence"))
        if confidence is not None:
            scores.append(confidence)
        for nested in value.values():
            _append_scores(nested, scores)
        return

    if isinstance(value, list):
        for item in value:
            _append_scores(item, scores)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _prompt_label_for_region_type(region_type: Any) -> str:
    normalized = str(region_type or "").strip().lower()
    if normalized in {"table", "formula", "chart"}:
        return normalized
    return "ocr"


def _should_skip_image_like_region_target(
    state: RunState,
    target: dict[str, Any],
    *,
    allow_chart_region_ocr: bool,
) -> bool:
    data = target.get("data")
    if not isinstance(data, dict):
        return False
    region_id = data.get("region_id")
    if not isinstance(region_id, str):
        return False
    region = state.get_region(region_id)
    if region is None:
        return False
    if allow_chart_region_ocr and str(region.type or "").strip().lower() == "chart":
        return False
    return _region_is_image_like_for_broad_ocr(region)

def _region_is_image_like_for_broad_ocr(region: Any) -> bool:
    render = region.metadata.get("render")
    if isinstance(render, dict) and isinstance(render.get("image_like"), bool):
        return bool(render.get("image_like"))
    return False


def _build_skipped_image_like_result(target: dict[str, Any]) -> dict[str, Any] | None:
    data = target.get("data")
    if not isinstance(data, dict):
        return None
    return {
        **data,
        "text": "",
        "source": "skipped_image_like",
        "confidence": None,
        "skipped": True,
    }
