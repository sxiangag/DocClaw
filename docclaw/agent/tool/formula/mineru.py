"""MinerU-backed formula implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from docclaw.agent.tool.formula.formula import FormulaTool
from docclaw.agent.utils import Action, PageState, Region, RunState, has_searchable_text


class MinerUFormulaTool(FormulaTool):
    """Parse formulas with MinerU2.5-Pro via mineru-vl-utils."""

    DEFAULT_MODEL_NAME = "opendatalab/MinerU2.5-Pro-2604-1.2B"

    def __init__(
        self,
        *,
        client: Any | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        llm_kwargs: dict[str, Any] | None = None,
        vllm_llm: Any | None = None,
    ) -> None:
        self._client = client
        self.model_name = model_name
        self.llm_kwargs = dict(llm_kwargs or {})
        self._vllm_llm = vllm_llm

    @property
    def description(self) -> str:
        return (
            "Parse document formulas from a page set or region set with MinerU2.5-Pro. "
            "This backend runs page-level extraction, normalizes equation blocks into "
            "formula entries, writes them back into page state, and uses region targets "
            "as a filtering mode over the page result."
        )

    def parse_formulas(
        self,
        state: RunState,
        pages: list[PageState],
        regions: list[Region] | None,
        action: Action,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for page in pages:
            page_regions = (
                [region for region in regions if region.page_index == page.page_index]
                if regions is not None
                else None
            )
            if regions is not None and not page_regions:
                continue

            cached_formulas = _cached_formulas_for_page(page, regions=page_regions)
            if cached_formulas is not None:
                results.extend(cached_formulas)
                continue

            if not page.image_path:
                raise ValueError(f"page {page.page_index} has no image_path")
            image_path = Path(page.image_path).expanduser()
            if not image_path.exists():
                raise ValueError(f"page image_path does not exist: {image_path}")

            result = self._extract_page(image_path)
            blocks = _result_to_blocks(result)
            formulas = _formulas_from_result(page, blocks)
            _write_formulas_to_regions(page, formulas)
            selected_formulas = _filter_formulas(formulas, regions=page_regions)
            results.extend(selected_formulas)

        return results

    def _extract_page(self, image_path: Path) -> Any:
        with Image.open(image_path) as image:
            page_image = image.convert("RGB")
        return self._get_client().two_step_extract(page_image)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from mineru_vl_utils import MinerUClient, MinerULogitsProcessor
                from vllm import LLM
            except ImportError as exc:
                raise RuntimeError(
                    "MinerUFormulaTool requires local MinerU vllm dependencies "
                    "(install mineru-vl-utils[vllm] and vllm)."
                ) from exc
            if self._vllm_llm is None:
                self._vllm_llm = LLM(
                    model=self.model_name,
                    logits_processors=[MinerULogitsProcessor],
                    **self.llm_kwargs,
                )
            self._client = MinerUClient(
                backend="vllm-engine",
                vllm_llm=self._vllm_llm,
            )
        return self._client

def _cached_formulas_for_page(
    page: PageState,
    *,
    regions: list[Region] | None,
) -> list[dict[str, Any]] | None:
    selected_regions = regions if regions is not None else [
        region for region in page.regions
        if _is_formula_region(region)
    ]
    if not selected_regions:
        return None
    if not all(region.type == "formula" and has_searchable_text(region.text) for region in selected_regions):
        return None
    return [_region_to_formula(region) for region in selected_regions]


def _filter_formulas(
    formulas: list[dict[str, Any]],
    *,
    regions: list[Region] | None,
) -> list[dict[str, Any]]:
    if not regions:
        return [dict(formula) for formula in formulas]

    filtered: list[dict[str, Any]] = []
    for formula in formulas:
        bbox = _extract_bbox(formula.get("bbox"))
        if bbox is None or any(_bboxes_overlap(bbox, region.bbox) for region in regions):
            filtered.append(dict(formula))
    return filtered


def _write_formulas_to_regions(page: PageState, formulas: list[dict[str, Any]]) -> None:
    for formula in formulas:
        bbox = _extract_bbox(formula.get("bbox"))
        if bbox is None:
            continue
        latex = str(formula.get("text") or "").strip()
        if not latex:
            continue
        confidence = formula.get("confidence")
        normalized_confidence = float(confidence) if isinstance(confidence, (int, float)) else None
        for region in page.regions:
            if not _is_formula_region(region):
                continue
            if _bboxes_overlap(bbox, region.bbox):
                region.text = latex
                region.confidence = normalized_confidence


def _region_to_formula(region: Region) -> dict[str, Any]:
    return {
        "page_index": region.page_index,
        "region_id": region.region_id,
        "text": region.text or "",
        "source": "mineru",
        "confidence": region.confidence,
    }


def _is_formula_region(region: Region) -> bool:
    return str(region.label or "").strip().lower() in _FORMULA_BLOCK_TYPES


def _formulas_from_result(
    page: PageState,
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    formulas: list[dict[str, Any]] = []
    formula_index = 0
    for raw_block in blocks:
        block = _normalize_block_dict(raw_block)
        if block is None:
            continue
        block_type = str(block.get("type") or "").strip().lower()
        if block_type not in _FORMULA_BLOCK_TYPES:
            continue

        latex = _extract_formula_content(block)
        if not latex:
            continue

        bbox = _normalize_bbox(block.get("bbox"), page=page)
        sub_type = _optional_text(block.get("sub_type"))
        confidence = _extract_confidence(block)

        entry: dict[str, Any] = {
            "page_index": page.page_index,
            "region_id": None,
            "text": latex,
            "source": "mineru",
            "confidence": confidence,
            "bbox": list(bbox) if bbox is not None else None,
        }
        formulas.append(entry)
        formula_index += 1
    return formulas


_FORMULA_BLOCK_TYPES = {
    "equation",
    "formula",
    "interline_equation",
    "inline_equation",
    "equation_inline",
    "equation_isolated",
}


def _formula_kind(block_type: str, sub_type: str | None) -> str:
    lowered_sub_type = (sub_type or "").strip().lower()
    if block_type in {"equation_isolated", "interline_equation"}:
        return "display"
    if block_type in {"equation_inline", "inline_equation"}:
        return "inline"
    if "display" in lowered_sub_type or "isolated" in lowered_sub_type or "interline" in lowered_sub_type:
        return "display"
    if "inline" in lowered_sub_type:
        return "inline"
    return "unknown"


def _extract_formula_content(block: dict[str, Any]) -> str | None:
    for key in ("content", "latex", "text"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _extract_confidence(block: dict[str, Any]) -> float | None:
    for key in ("confidence", "score", "prob"):
        value = block.get(key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _normalize_bbox(
    value: Any,
    *,
    page: PageState,
) -> tuple[float, float, float, float] | None:
    bbox = _extract_bbox(value)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    if page.width and page.height and x1 <= 1.0 and y1 <= 1.0:
        return (x0 * page.width, y0 * page.height, x1 * page.width, y1 * page.height)
    return bbox


def _extract_bbox(value: Any) -> tuple[float, float, float, float] | None:
    values = _to_list(value)
    if values is None or len(values) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in values)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _bboxes_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    return lx0 < rx1 and rx0 < lx1 and ly0 < ry1 and ry0 < ly1


def _result_to_blocks(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        blocks: list[dict[str, Any]] = []
        for item in result:
            block = _normalize_block_dict(item)
            if block is not None:
                blocks.append(block)
        return blocks
    if isinstance(result, dict):
        for key in ("content_list", "blocks", "data"):
            maybe_blocks = result.get(key)
            if isinstance(maybe_blocks, list):
                return _result_to_blocks(maybe_blocks)
    maybe_blocks = getattr(result, "content_list", None)
    if isinstance(maybe_blocks, list):
        return _result_to_blocks(maybe_blocks)
    raise TypeError(f"unsupported MinerU result type: {type(result).__name__}")


def _normalize_block_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return _to_jsonable(value)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        data = model_dump()
        if isinstance(data, dict):
            return _to_jsonable(data)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, dict):
            return _to_jsonable(data)

    keys = ("type", "bbox", "angle", "content", "sub_type", "confidence", "score", "prob")
    data: dict[str, Any] = {}
    seen = False
    for key in keys:
        if hasattr(value, key):
            data[key] = getattr(value, key)
            seen = True
    if seen:
        return _to_jsonable(data)
    return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return value
    return None


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
