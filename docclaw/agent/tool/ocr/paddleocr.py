"""PaddleOCR-backed OCR implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docclaw.agent.tool.ocr.ocr import OcrTool, PixelBBox, clamp_bbox
from docclaw.agent.tool.quiet import suppress_vendor_init_output


class PaddleOCRTool(OcrTool):
    """Recognize text with PaddleOCR."""

    def __init__(
        self,
        *,
        pipeline: Any | None = None,
        pipeline_kwargs: dict[str, Any] | None = None,
        predict_kwargs: dict[str, Any] | None = None,
        artifact_dir: str | Path | None = None,
    ) -> None:
        super().__init__(artifact_dir=artifact_dir)
        self._pipeline = pipeline
        self.pipeline_kwargs = {
            "device": "gpu:0",
            "enable_mkldnn": False,
            **dict(pipeline_kwargs or {}),
        }
        self.predict_kwargs = dict(predict_kwargs or {})

    @property
    def source_name(self) -> str:
        return "paddleocr"

    def recognize_text(self, image_path: Path) -> dict[str, Any]:
        with suppress_vendor_init_output():
            result = self._get_pipeline().predict(str(image_path), **self.predict_kwargs)
        result_data = _to_jsonable(_result_to_dict(result))
        lines = _lines_from_result(result_data)
        text = "\n".join(line["text"] for line in lines if line.get("text"))
        scores = [
            float(line["score"])
            for line in lines
            if isinstance(line.get("score"), (int, float))
        ]
        confidence = sum(scores) / len(scores) if scores else None
        return {
            "text": text,
            "confidence": confidence,
            "lines": lines,
            "raw": result_data,
        }

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
            from paddleocr import PaddleOCR

            with suppress_vendor_init_output():
                self._pipeline = PaddleOCR(**self.pipeline_kwargs)
        return self._pipeline


def _result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, list):
        if not result:
            return {}
        return _result_to_dict(result[0])
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

    raise TypeError(f"unsupported PaddleOCR result type: {type(result).__name__}")


def _lines_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    texts = _to_sequence(result.get("rec_texts"))
    scores = _to_sequence(result.get("rec_scores"))
    boxes = _first_present_sequence(result.get("rec_boxes"), result.get("rec_polys"))

    lines: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            text = str(text)
        line: dict[str, Any] = {
            "text": text,
        }
        if index < len(scores):
            score = _to_float(scores[index])
            if score is not None:
                line["score"] = score
        if index < len(boxes):
            box = _to_serializable_box(boxes[index])
            if box is not None:
                line["box"] = box
        lines.append(line)
    return lines


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_serializable_box(value: Any) -> list[Any] | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return value
    return None


def _first_present_sequence(*values: Any) -> list[Any]:
    for value in values:
        sequence = _to_sequence(value)
        if sequence:
            return sequence
    return []


def _to_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return value
    return []


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
