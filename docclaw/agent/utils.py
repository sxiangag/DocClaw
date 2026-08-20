"""Core data entities for DocClaw.

The models in this module intentionally avoid OCR, layout, and LLM dependencies.
They define the stable vocabulary used by later planner, executor, and trace
layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import random
from typing import Any, Literal
from uuid import uuid4


ActionType = Literal[
    "ocr_enhancement",
    "inspect_ocr",
    "select_ocr",
    "select_pages",
    "crop",
    "rotate",
    "parse_chart",
    "parse_formula",
    "parse_layout",
    "parse_table",
    "zoom",
    "ocr",
    "transcribe",
    "internal_search",
    "understand_figures",
    "extract_evidence",
    "answer_from_evidence",
    "answer_json",
    "stop",
]
CoordinateSpace = Literal["pixel", "relative"]
RunStatus = Literal["running", "completed", "stopped", "failed", "max_steps"]
TaskType = Literal[
    "question_answering",
    "information_extraction",
    "table_qa",
    "verification",
    "other",
]
TrustLevel = Literal["trusted", "untrusted"]

ACTION_TYPES = {
    "ocr_enhancement",
    "inspect_ocr",
    "select_ocr",
    "select_pages",
    "crop",
    "rotate",
    "parse_chart",
    "parse_formula",
    "parse_layout",
    "parse_table",
    "zoom",
    "ocr",
    "transcribe",
    "internal_search",
    "understand_figures",
    "extract_evidence",
    "answer_from_evidence",
    "answer_json",
    "stop",
}
COORDINATE_SPACES = {"pixel", "relative"}
RUN_STATUSES = {"running", "completed", "stopped", "failed", "max_steps"}
TASK_TYPES = {
    "question_answering",
    "information_extraction",
    "table_qa",
    "verification",
    "other",
}
TRUST_LEVELS = {"trusted", "untrusted"}


###############################################################################
# Generic Helpers
###############################################################################

def new_id(prefix: str) -> str:
    """Return a short stable-looking identifier for trace objects."""
    return f"{prefix}_{uuid4().hex[:12]}"


def utc_now_iso() -> str:
    """Return an ISO timestamp with UTC timezone."""
    return datetime.now(timezone.utc).isoformat()


def text_preview(text: str | None, *, limit: int = 160) -> str | None:
    """Return a compact single-line preview of text."""
    if text is None:
        return None
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def has_searchable_text(text: object | None) -> bool:
    """Return whether the value is a non-empty searchable string."""
    return isinstance(text, str) and bool(text.strip())


def page_number_from_index(page_index: int) -> int:
    """Convert a zero-based page index to a 1-based page number."""
    if page_index < 0:
        raise ValueError("page_index must be non-negative")
    return page_index + 1


def page_index_from_number(page_number: int) -> int:
    """Convert a 1-based page number to a zero-based page index."""
    if page_number <= 0:
        raise ValueError("page_number must be positive")
    return page_number - 1


PLANNER_PAGE_ID_MAP_METADATA_KEY = "planner_page_id_by_index"


def _planner_page_id_candidates(*, document_id: str) -> list[str]:
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    tokens = [f"{c1}{v}{c2}" for c1 in consonants for v in vowels for c2 in consonants]
    seed = int(hashlib.sha1(document_id.encode("utf-8")).hexdigest()[:16], 16)
    random.Random(seed).shuffle(tokens)
    return [f"page_{token}" for token in tokens]


def ensure_planner_page_id_map(document: Any) -> dict[int, str]:
    if document is None:
        raise ValueError("document is required for planner page id mapping")
    raw = document.metadata.get(PLANNER_PAGE_ID_MAP_METADATA_KEY)
    page_id_by_index: dict[int, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                page_index = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, str) and value.strip():
                page_id_by_index[page_index] = value.strip()

    used_ids = set(page_id_by_index.values())
    candidates = _planner_page_id_candidates(document_id=document.document_id)
    candidate_iter = iter(candidates)
    for page in sorted(document.pages, key=lambda item: item.page_index):
        if page.page_index in page_id_by_index:
            continue
        candidate = next(candidate_iter, None)
        while candidate is not None and candidate in used_ids:
            candidate = next(candidate_iter, None)
        if candidate is None:
            raise ValueError("exhausted planner page-id candidates")
        page_id_by_index[page.page_index] = candidate
        used_ids.add(candidate)

    document.metadata[PLANNER_PAGE_ID_MAP_METADATA_KEY] = {
        str(page_index): page_id
        for page_index, page_id in sorted(page_id_by_index.items())
    }
    return page_id_by_index


def page_id_from_index(page_index: int, *, document: Any | None = None) -> str:
    """Convert a zero-based page index to a planner-facing page id."""
    if page_index < 0:
        raise ValueError("page_index must be non-negative")
    if document is None:
        return f"page_{page_number_from_index(page_index):03d}"
    page_id_by_index = ensure_planner_page_id_map(document)
    try:
        return page_id_by_index[page_index]
    except KeyError as exc:
        raise ValueError(f"unknown page_index: {page_index}") from exc


def page_index_from_id(page_id: str, *, document: Any | None = None) -> int:
    """Convert a planner-facing page id back to a zero-based page index."""
    if not isinstance(page_id, str):
        raise ValueError("page_id must be a string")
    normalized = page_id.strip()
    if document is None:
        if not normalized.startswith("page_"):
            raise ValueError(f"invalid page_id: {page_id}")
        suffix = normalized[len("page_") :]
        if not suffix.isdigit():
            raise ValueError(f"invalid page_id: {page_id}")
        return page_index_from_number(int(suffix))
    for page_index, candidate in ensure_planner_page_id_map(document).items():
        if candidate == normalized:
            return page_index
    raise ValueError(f"invalid page_id: {page_id}")


def is_page_level_synthetic_region_id(region_id: str | None) -> bool:
    """Return whether the id is a synthetic page-level anchor such as page_5."""
    if not isinstance(region_id, str):
        return False
    if not region_id.startswith("page_"):
        return False
    suffix = region_id[len("page_") :].strip()
    return bool(suffix)


def plannerize_page_refs(value: Any, *, document: Any | None = None) -> Any:
    """Convert page_index/page_indices keys and values to planner-facing page id(s)."""
    if isinstance(value, dict):
        converted: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                planner_key = _planner_page_key(key)
                converted[planner_key] = _plannerize_page_value(key, item, document=document)
            else:
                converted[key] = plannerize_page_refs(item, document=document)
        return converted
    if isinstance(value, list):
        return [plannerize_page_refs(item, document=document) for item in value]
    return value


def _planner_page_key(key: str) -> str:
    if key == "hit_page_ids":
        return "hit_page_ids"
    if key == "searched_pages":
        return "searched_page_ids"
    if "page_indices" in key:
        return key.replace("page_indices", "page_ids")
    if "page_indexes" in key:
        return key.replace("page_indexes", "page_ids")
    if "page_index" in key:
        return key.replace("page_index", "page_id")
    return key


def _plannerize_page_value(key: str, value: Any, *, document: Any | None = None) -> Any:
    if value is None:
        return None
    if key.endswith("_by_page") and isinstance(value, dict):
        converted: dict[Any, Any] = {}
        for page_key, item in value.items():
            if isinstance(page_key, int):
                converted[page_id_from_index(page_key, document=document)] = plannerize_page_refs(item, document=document)
            else:
                converted[page_key] = plannerize_page_refs(item, document=document)
        return converted
    if key == "hit_page_ids":
        if isinstance(value, list):
            return [
                page_id_from_index(int(item), document=document)
                for item in value
                if isinstance(item, int)
            ]
        return value
    if key == "searched_pages":
        if isinstance(value, list):
            return [
                page_id_from_index(int(item), document=document)
                for item in value
                if isinstance(item, int)
            ]
        return value
    if "page_indices" in key or "page_indexes" in key:
        if isinstance(value, list):
            return [
                page_id_from_index(int(item), document=document)
                for item in value
                if isinstance(item, int)
            ]
        return plannerize_page_refs(value, document=document)
    if "page_index" in key:
        if isinstance(value, int):
            return page_id_from_index(value, document=document)
        return plannerize_page_refs(value, document=document)
    return plannerize_page_refs(value, document=document)


def synthetic_ocr_region_id(page_index: int, bbox: tuple[int, int, int, int]) -> str:
    """Return a stable synthetic region id for OCR-only page crops."""
    x0, y0, x1, y1 = bbox
    return f"ocr_region_p{page_number_from_index(page_index)}_{x0}_{y0}_{x1}_{y1}"


###############################################################################
# Document Geometry and Page State
###############################################################################

@dataclass(slots=True)
class Region:
    """A rectangular document region on one page."""

    page_index: int
    bbox: tuple[float, float, float, float]
    region_id: str = field(default_factory=lambda: new_id("region"))
    label: str | None = None
    raw_type: str | None = None
    type: str | None = None
    text: str | None = None
    confidence: float | None = None
    coordinate_space: CoordinateSpace = "pixel"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.coordinate_space not in COORDINATE_SPACES:
            raise ValueError(f"unsupported coordinate_space: {self.coordinate_space}")
        if self.page_index < 0:
            raise ValueError("page_index must be non-negative")
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain four coordinates")
        x0, y0, x1, y1 = self.bbox
        if x1 < x0 or y1 < y0:
            raise ValueError("bbox must be ordered as (x0, y0, x1, y1)")
        if self.coordinate_space == "relative":
            for value in self.bbox:
                if value < 0.0 or value > 1.0:
                    raise ValueError("relative bbox coordinates must be in [0, 1]")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "page_index": self.page_index,
            "page_number": self.page_number,
            "bbox": list(self.bbox),
            "label": self.label,
            "raw_type": self.raw_type,
            "type": self.type,
            "text": self.text,
            "confidence": self.confidence,
            "coordinate_space": self.coordinate_space,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Region:
        return cls(
            region_id=str(data["region_id"]),
            page_index=int(data["page_index"]),
            bbox=tuple(float(v) for v in data["bbox"]),  # type: ignore[arg-type]
            label=data.get("label"),
            raw_type=data.get("raw_type"),
            type=data.get("type"),
            text=data.get("text"),
            confidence=data.get("confidence"),
            coordinate_space=data.get("coordinate_space", "pixel"),
            metadata=dict(data.get("metadata") or {}),
        )

    @property
    def page_number(self) -> int:
        return page_number_from_index(self.page_index)


@dataclass(slots=True)
class PageState:
    """State known about a single document page."""

    page_index: int
    width: int | None = None
    height: int | None = None
    image_path: str | None = None
    ocr_text: str | None = None
    ocr_confidence: float | None = None
    regions: list[Region] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.page_index < 0:
            raise ValueError("page_index must be non-negative")
        if self.width is not None and self.width <= 0:
            raise ValueError("width must be positive")
        if self.height is not None and self.height <= 0:
            raise ValueError("height must be positive")
        if self.ocr_confidence is not None and not 0.0 <= self.ocr_confidence <= 1.0:
            raise ValueError("ocr_confidence must be in [0, 1]")
        seen_region_ids: set[str] = set()
        for region in self.regions:
            if region.page_index != self.page_index:
                raise ValueError("region.page_index does not match page.page_index")
            if region.region_id in seen_region_ids:
                raise ValueError(f"duplicate region_id on page: {region.region_id}")
            seen_region_ids.add(region.region_id)

    def add_region(self, region: Region) -> None:
        if region.page_index != self.page_index:
            raise ValueError("region.page_index does not match page.page_index")
        if self.get_region(region.region_id) is not None:
            raise ValueError(f"duplicate region_id on page: {region.region_id}")
        self.regions.append(region)

    def get_region(self, region_id: str) -> Region | None:
        return next((region for region in self.regions if region.region_id == region_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "image_path": self.image_path,
            "ocr_text": self.ocr_text,
            "ocr_confidence": self.ocr_confidence,
            "regions": [region.to_dict() for region in self.regions],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageState:
        return cls(
            page_index=int(data["page_index"]),
            width=data.get("width"),
            height=data.get("height"),
            image_path=data.get("image_path"),
            ocr_text=data.get("ocr_text"),
            ocr_confidence=data.get("ocr_confidence"),
            regions=[Region.from_dict(item) for item in data.get("regions", [])],
            metadata=dict(data.get("metadata") or {}),
        )

    @property
    def page_number(self) -> int:
        return page_number_from_index(self.page_index)


###############################################################################
# Planner Actions and Executor Observations
# Action -> Execution -> Observation -> State Update -> Action -> Observation -> ...
###############################################################################

@dataclass(slots=True)
class Action:
    """A planner-selected operation over the document."""

    action_type: ActionType
    action_id: str = field(default_factory=lambda: new_id("action"))
    target: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    rationale: str | None = None

    def __post_init__(self) -> None:
        if self.action_type not in ACTION_TYPES:
            raise ValueError(f"unsupported action_type: {self.action_type}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "target": self.target,
            "parameters": self.parameters,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        return cls(
            action_id=str(data["action_id"]),
            action_type=data["action_type"],
            target=dict(data.get("target") or {}),
            parameters=dict(data.get("parameters") or {}),
            rationale=data.get("rationale"),
        )


@dataclass(slots=True)
class Observation:
    """The result of executing one action."""

    action_id: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    message: str | None = None
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "error": self.error,
            "artifacts": list(self.artifacts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        return cls(
            action_id=str(data["action_id"]),
            success=bool(data["success"]),
            data=dict(data.get("data") or {}),
            message=data.get("message"),
            error=data.get("error"),
            artifacts=list(data.get("artifacts") or []),
        )


###############################################################################
# Task, Evidence, and Execution Trace
###############################################################################

@dataclass(slots=True)
class Task:
    """A user objective to execute against a document."""

    prompt: str
    task_id: str = field(default_factory=lambda: new_id("task"))
    task_type: TaskType = "question_answering"
    expected_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"unsupported task_type: {self.task_type}")
        if not self.prompt.strip():
            raise ValueError("task prompt must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "task_type": self.task_type,
            "expected_output": self.expected_output,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        return cls(
            task_id=str(data["task_id"]),
            prompt=str(data["prompt"]),
            task_type=data.get("task_type", "question_answering"),
            expected_output=data.get("expected_output"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class Evidence:
    """A piece of accumulated evidence and its source."""

    content: str
    evidence_id: str = field(default_factory=lambda: new_id("evidence"))
    trust_level: TrustLevel = "trusted"
    reference: str | None = None
    page_index: int | None = None
    region_id: str | None = None
    action_id: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("evidence content must not be empty")
        if self.trust_level not in TRUST_LEVELS:
            raise ValueError(f"unsupported trust_level: {self.trust_level}")
        if self.reference is not None and not self.reference.strip():
            raise ValueError("reference must not be empty when provided")
        if self.page_index is not None and self.page_index < 0:
            raise ValueError("page_index must be non-negative")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "content": self.content,
            "trust_level": self.trust_level,
            "reference": self.reference,
            "page_index": self.page_index,
            "region_id": self.region_id,
            "action_id": self.action_id,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        metadata = dict(data.get("metadata") or {})
        raw_trust_level = data.get("trust_level")
        if isinstance(raw_trust_level, str) and raw_trust_level.strip():
            trust_level = raw_trust_level.strip()
        else:
            trust_level = "trusted"
            if isinstance(metadata.get("source_kind"), str):
                trust_level = "untrusted"

        raw_reference = data.get("reference")
        if isinstance(raw_reference, str) and raw_reference.strip():
            reference = raw_reference.strip()
        else:
            reference = None
            raw_source_ref = data.get("source_ref")
            if isinstance(raw_source_ref, str) and raw_source_ref.strip():
                reference = raw_source_ref.strip()
            for key in ("resource_id", "search_id"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    reference = value.strip()
                    break

        return cls(
            evidence_id=str(data["evidence_id"]),
            content=str(data["content"]),
            trust_level=trust_level,  # type: ignore[arg-type]
            reference=reference,
            page_index=data.get("page_index"),
            region_id=data.get("region_id"),
            action_id=data.get("action_id"),
            confidence=data.get("confidence"),
            metadata=metadata,
        )


@dataclass(slots=True)
class ZoomRegionView:
    """One zoomed region artifact available for later OCR refinement."""

    region_id: str
    page_index: int
    artifact_path: str
    target_long_side_px: int
    bbox: tuple[float, float, float, float]
    pixel_bbox: tuple[int, int, int, int]
    coordinate_space: CoordinateSpace = "pixel"
    artifact_width: int | None = None
    artifact_height: int | None = None
    source_image_path: str | None = None

    def __post_init__(self) -> None:
        if not self.region_id.strip():
            raise ValueError("region_id must not be empty")
        if self.page_index < 0:
            raise ValueError("page_index must be non-negative")
        if not self.artifact_path.strip():
            raise ValueError("artifact_path must not be empty")
        if self.target_long_side_px <= 0:
            raise ValueError("target_long_side_px must be positive")
        if self.coordinate_space not in COORDINATE_SPACES:
            raise ValueError(f"unsupported coordinate_space: {self.coordinate_space}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "page_index": self.page_index,
            "page_number": self.page_number,
            "artifact_path": self.artifact_path,
            "target_long_side_px": self.target_long_side_px,
            "bbox": list(self.bbox),
            "pixel_bbox": list(self.pixel_bbox),
            "coordinate_space": self.coordinate_space,
            "artifact_width": self.artifact_width,
            "artifact_height": self.artifact_height,
            "source_image_path": self.source_image_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ZoomRegionView:
        return cls(
            region_id=str(data["region_id"]),
            page_index=int(data["page_index"]),
            artifact_path=str(data["artifact_path"]),
            target_long_side_px=int(data["target_long_side_px"]),
            bbox=tuple(float(value) for value in data["bbox"]),  # type: ignore[arg-type]
            pixel_bbox=tuple(int(value) for value in data["pixel_bbox"]),  # type: ignore[arg-type]
            coordinate_space=data.get("coordinate_space", "pixel"),
            artifact_width=(
                int(data["artifact_width"])
                if data.get("artifact_width") is not None
                else None
            ),
            artifact_height=(
                int(data["artifact_height"])
                if data.get("artifact_height") is not None
                else None
            ),
            source_image_path=(
                str(data["source_image_path"])
                if data.get("source_image_path") is not None
                else None
            ),
        )

    @property
    def page_number(self) -> int:
        return page_number_from_index(self.page_index)


@dataclass(slots=True)
class ZoomRegionState:
    """Run-scoped zoomed region artifacts."""

    views_by_region_id: dict[str, ZoomRegionView] = field(default_factory=dict)
    ordered_region_ids: list[str] = field(default_factory=list)

    def add_view(self, view: ZoomRegionView) -> None:
        self.views_by_region_id[view.region_id] = view
        if view.region_id not in self.ordered_region_ids:
            self.ordered_region_ids.append(view.region_id)

    def get_view(self, region_id: str) -> ZoomRegionView | None:
        return self.views_by_region_id.get(region_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "views_by_region_id": {
                key: value.to_dict()
                for key, value in self.views_by_region_id.items()
            },
            "ordered_region_ids": list(self.ordered_region_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ZoomRegionState:
        raw_views = data.get("views_by_region_id") or {}
        views_by_region_id = {
            str(key): ZoomRegionView.from_dict(value)
            for key, value in raw_views.items()
            if isinstance(value, dict)
        } if isinstance(raw_views, dict) else {}
        ordered_region_ids = [str(item) for item in data.get("ordered_region_ids", [])]
        return cls(
            views_by_region_id=views_by_region_id,
            ordered_region_ids=ordered_region_ids,
        )


@dataclass(slots=True)
class CropRegionView:
    """One crop-expanded region artifact available for later OCR refinement."""

    region_id: str
    page_index: int
    artifact_path: str
    left_px: int
    right_px: int
    top_px: int
    bottom_px: int
    bbox: tuple[float, float, float, float]
    pixel_bbox: tuple[int, int, int, int]
    coordinate_space: CoordinateSpace = "pixel"
    artifact_width: int | None = None
    artifact_height: int | None = None
    source_image_path: str | None = None

    def __post_init__(self) -> None:
        if not self.region_id.strip():
            raise ValueError("region_id must not be empty")
        if self.page_index < 0:
            raise ValueError("page_index must be non-negative")
        if not self.artifact_path.strip():
            raise ValueError("artifact_path must not be empty")
        if self.coordinate_space not in COORDINATE_SPACES:
            raise ValueError(f"unsupported coordinate_space: {self.coordinate_space}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "page_index": self.page_index,
            "page_number": self.page_number,
            "artifact_path": self.artifact_path,
            "left_px": self.left_px,
            "right_px": self.right_px,
            "top_px": self.top_px,
            "bottom_px": self.bottom_px,
            "bbox": list(self.bbox),
            "pixel_bbox": list(self.pixel_bbox),
            "coordinate_space": self.coordinate_space,
            "artifact_width": self.artifact_width,
            "artifact_height": self.artifact_height,
            "source_image_path": self.source_image_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CropRegionView:
        return cls(
            region_id=str(data["region_id"]),
            page_index=int(data["page_index"]),
            artifact_path=str(data["artifact_path"]),
            left_px=int(data["left_px"]),
            right_px=int(data["right_px"]),
            top_px=int(data["top_px"]),
            bottom_px=int(data["bottom_px"]),
            bbox=tuple(float(value) for value in data["bbox"]),  # type: ignore[arg-type]
            pixel_bbox=tuple(int(value) for value in data["pixel_bbox"]),  # type: ignore[arg-type]
            coordinate_space=data.get("coordinate_space", "pixel"),
            artifact_width=(
                int(data["artifact_width"])
                if data.get("artifact_width") is not None
                else None
            ),
            artifact_height=(
                int(data["artifact_height"])
                if data.get("artifact_height") is not None
                else None
            ),
            source_image_path=(
                str(data["source_image_path"])
                if data.get("source_image_path") is not None
                else None
            ),
        )

    @property
    def page_number(self) -> int:
        return page_number_from_index(self.page_index)


@dataclass(slots=True)
class CropRegionState:
    """Run-scoped crop-expanded region artifacts."""

    views_by_region_id: dict[str, CropRegionView] = field(default_factory=dict)
    ordered_region_ids: list[str] = field(default_factory=list)

    def add_view(self, view: CropRegionView) -> None:
        self.views_by_region_id[view.region_id] = view
        if view.region_id not in self.ordered_region_ids:
            self.ordered_region_ids.append(view.region_id)

    def get_view(self, region_id: str) -> CropRegionView | None:
        return self.views_by_region_id.get(region_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "views_by_region_id": {
                key: value.to_dict()
                for key, value in self.views_by_region_id.items()
            },
            "ordered_region_ids": list(self.ordered_region_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CropRegionState:
        raw_views = data.get("views_by_region_id") or {}
        views_by_region_id = {
            str(key): CropRegionView.from_dict(value)
            for key, value in raw_views.items()
            if isinstance(value, dict)
        } if isinstance(raw_views, dict) else {}
        ordered_region_ids = [str(item) for item in data.get("ordered_region_ids", [])]
        return cls(
            views_by_region_id=views_by_region_id,
            ordered_region_ids=ordered_region_ids,
        )


@dataclass(slots=True)
class RotateRegionView:
    """One rotated region artifact available for later OCR refinement."""

    region_id: str
    page_index: int
    artifact_path: str
    angle_degree: float
    bbox: tuple[float, float, float, float]
    pixel_bbox: tuple[int, int, int, int]
    coordinate_space: CoordinateSpace = "pixel"
    artifact_width: int | None = None
    artifact_height: int | None = None
    source_image_path: str | None = None

    def __post_init__(self) -> None:
        if not self.region_id.strip():
            raise ValueError("region_id must not be empty")
        if self.page_index < 0:
            raise ValueError("page_index must be non-negative")
        if not self.artifact_path.strip():
            raise ValueError("artifact_path must not be empty")
        if self.coordinate_space not in COORDINATE_SPACES:
            raise ValueError(f"unsupported coordinate_space: {self.coordinate_space}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "page_index": self.page_index,
            "page_number": self.page_number,
            "artifact_path": self.artifact_path,
            "angle_degree": self.angle_degree,
            "bbox": list(self.bbox),
            "pixel_bbox": list(self.pixel_bbox),
            "coordinate_space": self.coordinate_space,
            "artifact_width": self.artifact_width,
            "artifact_height": self.artifact_height,
            "source_image_path": self.source_image_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RotateRegionView:
        return cls(
            region_id=str(data["region_id"]),
            page_index=int(data["page_index"]),
            artifact_path=str(data["artifact_path"]),
            angle_degree=float(data["angle_degree"]),
            bbox=tuple(float(value) for value in data["bbox"]),  # type: ignore[arg-type]
            pixel_bbox=tuple(int(value) for value in data["pixel_bbox"]),  # type: ignore[arg-type]
            coordinate_space=data.get("coordinate_space", "pixel"),
            artifact_width=(
                int(data["artifact_width"])
                if data.get("artifact_width") is not None
                else None
            ),
            artifact_height=(
                int(data["artifact_height"])
                if data.get("artifact_height") is not None
                else None
            ),
            source_image_path=(
                str(data["source_image_path"])
                if data.get("source_image_path") is not None
                else None
            ),
        )

    @property
    def page_number(self) -> int:
        return page_number_from_index(self.page_index)


@dataclass(slots=True)
class RotateRegionState:
    """Run-scoped rotated region artifacts."""

    views_by_region_id: dict[str, RotateRegionView] = field(default_factory=dict)
    ordered_region_ids: list[str] = field(default_factory=list)

    def add_view(self, view: RotateRegionView) -> None:
        self.views_by_region_id[view.region_id] = view
        if view.region_id not in self.ordered_region_ids:
            self.ordered_region_ids.append(view.region_id)

    def get_view(self, region_id: str) -> RotateRegionView | None:
        return self.views_by_region_id.get(region_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "views_by_region_id": {
                key: value.to_dict()
                for key, value in self.views_by_region_id.items()
            },
            "ordered_region_ids": list(self.ordered_region_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RotateRegionState:
        raw_views = data.get("views_by_region_id") or {}
        views_by_region_id = {
            str(key): RotateRegionView.from_dict(value)
            for key, value in raw_views.items()
            if isinstance(value, dict)
        } if isinstance(raw_views, dict) else {}
        ordered_region_ids = [str(item) for item in data.get("ordered_region_ids", [])]
        return cls(
            views_by_region_id=views_by_region_id,
            ordered_region_ids=ordered_region_ids,
        )


@dataclass(slots=True)
class TraceStep:
    """One transition in a DocClaw run."""

    step_index: int
    action: Action
    observation: Observation
    state_summary: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if self.action.action_id != self.observation.action_id:
            raise ValueError("action and observation must refer to the same action_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "action": self.action.to_dict(),
            "observation": self.observation.to_dict(),
            "state_summary": self.state_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceStep:
        return cls(
            step_index=int(data["step_index"]),
            timestamp=str(data["timestamp"]),
            action=Action.from_dict(data["action"]),
            observation=Observation.from_dict(data["observation"]),
            state_summary=dict(data.get("state_summary") or {}),
        )


@dataclass(slots=True)
class ActiveSkill:
    """Selected task skill attached to one run."""

    name: str
    reason: str | None = None
    source: str | None = None
    path: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("active skill name must not be empty")
        if self.reason is not None and not self.reason.strip():
            self.reason = None
        if self.source is not None and not self.source.strip():
            self.source = None
        if self.path is not None and not self.path.strip():
            self.path = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reason": self.reason,
            "source": self.source,
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActiveSkill:
        return cls(
            name=str(data["name"]),
            reason=data.get("reason"),
            source=data.get("source"),
            path=data.get("path"),
        )


###############################################################################
# Task-Level Tool Memory
###############################################################################


@dataclass(slots=True)
class SearchHint:
    """One task-scoped internal search hint."""

    page_index: int | None = None
    region_id: str | None = None
    score: float | None = None
    matched_queries: list[str] = field(default_factory=list)
    snippet_preview: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "region_id": self.region_id,
            "score": self.score,
            "matched_queries": list(self.matched_queries),
            "snippet_preview": self.snippet_preview,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchHint:
        return cls(
            page_index=int(data["page_index"]) if data.get("page_index") is not None else None,
            region_id=(
                str(data["region_id"]) if data.get("region_id") is not None else None
            ),
            score=float(data["score"]) if data.get("score") is not None else None,
            matched_queries=[str(item) for item in data.get("matched_queries", [])],
            snippet_preview=(
                str(data["snippet_preview"])
                if data.get("snippet_preview") is not None
                else None
            ),
        )


@dataclass(slots=True)
class SearchHistoryEntry:
    """One task-scoped internal search execution."""

    search_id: str
    action_id: str | None = None
    queries: list[str] = field(default_factory=list)
    mode: str | None = None
    searched_pages: list[int] = field(default_factory=list)
    hit_page_ids: list[int] = field(default_factory=list)
    hit_region_ids: list[str] = field(default_factory=list)
    hit_count: int = 0
    hints: list[SearchHint] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.search_id.strip():
            raise ValueError("search_id must not be empty")
        if self.hit_count < 0:
            raise ValueError("hit_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "search_id": self.search_id,
            "action_id": self.action_id,
            "queries": list(self.queries),
            "mode": self.mode,
            "searched_pages": list(self.searched_pages),
            "hit_page_ids": list(self.hit_page_ids),
            "hit_region_ids": list(self.hit_region_ids),
            "hit_count": self.hit_count,
            "hints": [item.to_dict() for item in self.hints],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchHistoryEntry:
        return cls(
            search_id=str(data["search_id"]),
            action_id=str(data["action_id"]) if data.get("action_id") is not None else None,
            queries=[str(item) for item in data.get("queries", [])],
            mode=str(data["mode"]) if data.get("mode") is not None else None,
            searched_pages=[int(item) for item in data.get("searched_pages", [])],
            hit_page_ids=[int(item) for item in data.get("hit_page_ids", [])],
            hit_region_ids=[str(item) for item in data.get("hit_region_ids", [])],
            hit_count=int(data.get("hit_count", 0)),
            hints=[
                SearchHint.from_dict(item)
                for item in data.get("hints", [])
                if isinstance(item, dict)
            ],
        )


@dataclass(slots=True)
class SearchHistoryState:
    """Task-scoped internal search history."""

    entries_by_id: dict[str, SearchHistoryEntry] = field(default_factory=dict)
    ordered_search_ids: list[str] = field(default_factory=list)

    def add_entry(self, entry: SearchHistoryEntry) -> None:
        self.entries_by_id[entry.search_id] = entry
        if entry.search_id not in self.ordered_search_ids:
            self.ordered_search_ids.append(entry.search_id)

    def get_entry(self, search_id: str) -> SearchHistoryEntry | None:
        return self.entries_by_id.get(search_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries_by_id": {
                key: value.to_dict()
                for key, value in self.entries_by_id.items()
            },
            "ordered_search_ids": list(self.ordered_search_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchHistoryState:
        raw_entries = data.get("entries_by_id") or {}
        entries_by_id = {
            str(key): SearchHistoryEntry.from_dict(value)
            for key, value in raw_entries.items()
            if isinstance(value, dict)
        } if isinstance(raw_entries, dict) else {}
        ordered_search_ids = [str(item) for item in data.get("ordered_search_ids", [])]
        return cls(
            entries_by_id=entries_by_id,
            ordered_search_ids=ordered_search_ids,
        )


@dataclass(slots=True)
class FigureInsight:
    """One question-conditioned figure understanding result."""

    insight_key: str
    question: str
    page_index: int
    answer: str | None = None
    reason: str | None = None
    artifact_path: str | None = None

    def __post_init__(self) -> None:
        if not self.insight_key.strip():
            raise ValueError("insight_key must not be empty")
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if self.page_index < 0:
            raise ValueError("page_index must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "insight_key": self.insight_key,
            "question": self.question,
            "page_index": self.page_index,
            "answer": self.answer,
            "reason": self.reason,
            "artifact_path": self.artifact_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FigureInsight:
        return cls(
            insight_key=str(data["insight_key"]),
            question=str(data["question"]),
            page_index=int(data["page_index"]),
            answer=str(data["answer"]) if data.get("answer") is not None else None,
            reason=str(data["reason"]) if data.get("reason") is not None else None,
            artifact_path=(
                str(data["artifact_path"])
                if data.get("artifact_path") is not None
                else None
            ),
        )


@dataclass(slots=True)
class FigureInsightState:
    """Task-scoped figure understanding memory."""

    insights_by_key: dict[str, FigureInsight] = field(default_factory=dict)
    ordered_insight_keys: list[str] = field(default_factory=list)

    def add_insight(self, insight: FigureInsight) -> None:
        self.insights_by_key[insight.insight_key] = insight
        if insight.insight_key not in self.ordered_insight_keys:
            self.ordered_insight_keys.append(insight.insight_key)

    def get_insight(self, insight_key: str) -> FigureInsight | None:
        return self.insights_by_key.get(insight_key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "insights_by_key": {
                key: value.to_dict()
                for key, value in self.insights_by_key.items()
            },
            "ordered_insight_keys": list(self.ordered_insight_keys),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FigureInsightState:
        raw_insights = data.get("insights_by_key") or {}
        insights_by_key = {
            str(key): FigureInsight.from_dict(value)
            for key, value in raw_insights.items()
            if isinstance(value, dict)
        } if isinstance(raw_insights, dict) else {}
        ordered_insight_keys = [str(item) for item in data.get("ordered_insight_keys", [])]
        return cls(
            insights_by_key=insights_by_key,
            ordered_insight_keys=ordered_insight_keys,
        )


@dataclass(slots=True)
class EvidenceAssessment:
    """One task-scoped evidence sufficiency assessment."""

    assessment_id: str
    action_id: str | None = None
    page_indices: list[int] = field(default_factory=list)
    region_ids: list[str] = field(default_factory=list)
    answerability_status: str = "inconclusive"
    missing_information: str | None = None
    evidence_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.assessment_id.strip():
            raise ValueError("assessment_id must not be empty")
        if self.answerability_status not in {"answerable", "inconclusive"}:
            raise ValueError(f"unsupported answerability_status: {self.answerability_status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "action_id": self.action_id,
            "page_indices": list(self.page_indices),
            "region_ids": list(self.region_ids),
            "answerability_status": self.answerability_status,
            "missing_information": self.missing_information,
            "evidence_ids": list(self.evidence_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceAssessment:
        return cls(
            assessment_id=str(data["assessment_id"]),
            action_id=str(data["action_id"]) if data.get("action_id") is not None else None,
            page_indices=[int(item) for item in data.get("page_indices", [])],
            region_ids=[str(item) for item in data.get("region_ids", [])],
            answerability_status=str(data.get("answerability_status")),
            missing_information=(
                str(data["missing_information"])
                if data.get("missing_information") is not None
                else None
            ),
            evidence_ids=[str(item) for item in data.get("evidence_ids", [])],
        )

@dataclass(slots=True)
class EvidenceAssessmentHistoryState:
    """Task-scoped evidence assessment history."""

    assessments_by_id: dict[str, EvidenceAssessment] = field(default_factory=dict)
    ordered_assessment_ids: list[str] = field(default_factory=list)

    def add_assessment(self, assessment: EvidenceAssessment) -> None:
        self.assessments_by_id[assessment.assessment_id] = assessment
        if assessment.assessment_id not in self.ordered_assessment_ids:
            self.ordered_assessment_ids.append(assessment.assessment_id)

    def get_assessment(self, assessment_id: str) -> EvidenceAssessment | None:
        return self.assessments_by_id.get(assessment_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessments_by_id": {
                key: value.to_dict()
                for key, value in self.assessments_by_id.items()
            },
            "ordered_assessment_ids": list(self.ordered_assessment_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceAssessmentHistoryState:
        raw_assessments = data.get("assessments_by_id") or {}
        assessments_by_id = {
            str(key): EvidenceAssessment.from_dict(value)
            for key, value in raw_assessments.items()
            if isinstance(value, dict)
        } if isinstance(raw_assessments, dict) else {}
        ordered_assessment_ids = [
            str(item) for item in data.get("ordered_assessment_ids", [])
        ]
        return cls(
            assessments_by_id=assessments_by_id,
            ordered_assessment_ids=ordered_assessment_ids,
        )


###############################################################################
# Document Environment State
###############################################################################


@dataclass(slots=True)
class DocumentState:
    """Reusable state of a document, independent of any specific task."""

    document_id: str
    pages: list[PageState] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")
        seen_page_indexes: set[int] = set()
        seen_region_ids: set[str] = set()
        for page in self.pages:
            if page.page_index in seen_page_indexes:
                raise ValueError(f"duplicate page_index: {page.page_index}")
            seen_page_indexes.add(page.page_index)
            for region in page.regions:
                if region.region_id in seen_region_ids:
                    raise ValueError(f"duplicate region_id in document: {region.region_id}")
                seen_region_ids.add(region.region_id)
        self.pages.sort(key=lambda item: item.page_index)

    def add_page(self, page: PageState) -> None:
        if self.get_page(page.page_index) is not None:
            raise ValueError(f"page {page.page_index} already exists")
        for region in page.regions:
            if self.get_region(region.region_id) is not None:
                raise ValueError(f"duplicate region_id in document: {region.region_id}")
        self.pages.append(page)
        self.pages.sort(key=lambda item: item.page_index)

    def get_page(self, page_index: int) -> PageState | None:
        return next((page for page in self.pages if page.page_index == page_index), None)

    def get_page_by_number(self, page_number: int) -> PageState | None:
        return self.get_page(page_index_from_number(page_number))

    def require_page(self, page_index: int) -> PageState:
        page = self.get_page(page_index)
        if page is None:
            raise ValueError(f"unknown page_index: {page_index}")
        return page

    def require_page_by_number(self, page_number: int) -> PageState:
        page = self.get_page_by_number(page_number)
        if page is None:
            raise ValueError(f"unknown page_number: {page_number}")
        return page

    def get_region(self, region_id: str) -> Region | None:
        for page in self.pages:
            region = page.get_region(region_id)
            if region is not None:
                return region
        return None

    def require_region(self, region_id: str) -> Region:
        region = self.get_region(region_id)
        if region is None:
            raise ValueError(f"unknown region_id: {region_id}")
        return region

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-friendly document summary."""
        return {
            "document_id": self.document_id,
            "pages": len(self.pages),
            "regions": sum(len(page.regions) for page in self.pages),
            "pages_with_ocr": sum(1 for page in self.pages if has_searchable_text(page.ocr_text)),
            "regions_with_ocr": sum(
                1
                for page in self.pages
                for region in page.regions
                if has_searchable_text(region.text)
            ),
        }

    def planner_summary(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "pages": [page.to_dict() for page in self.pages],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentState:
        return cls(
            document_id=str(data["document_id"]),
            pages=[PageState.from_dict(item) for item in data.get("pages", [])],
            metadata=dict(data.get("metadata") or {}),
        )


###############################################################################
# Task-Specific Run State
###############################################################################


@dataclass(slots=True)
class RunState:
    """Execution state for one task over one document."""

    document: DocumentState
    task: Task
    run_id: str = field(default_factory=lambda: new_id("run"))
    inspected_regions: set[str] = field(default_factory=set)
    zoom_regions: ZoomRegionState = field(default_factory=ZoomRegionState)
    crop_regions: CropRegionState = field(default_factory=CropRegionState)
    rotate_regions: RotateRegionState = field(default_factory=RotateRegionState)
    search_history: SearchHistoryState = field(default_factory=SearchHistoryState)
    figure_insights: FigureInsightState = field(default_factory=FigureInsightState)
    evidence_assessment_history: EvidenceAssessmentHistoryState = field(
        default_factory=EvidenceAssessmentHistoryState
    )
    evidence: list[Evidence] = field(default_factory=list)
    action_trace: list[TraceStep] = field(default_factory=list)
    status: RunStatus = "running"
    final_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    ACTIVE_SKILL_METADATA_KEY = "active_skill"
    PENDING_EVENTS_METADATA_KEY = "pending_events"

    def __post_init__(self) -> None:
        if self.status not in RUN_STATUSES:
            raise ValueError(f"unsupported run status: {self.status}")

    def get_page(self, page_index: int) -> PageState | None:
        return self.document.get_page(page_index)

    def require_page(self, page_index: int) -> PageState:
        return self.document.require_page(page_index)

    def get_region(self, region_id: str) -> Region | None:
        return self.document.get_region(region_id)

    def require_region(self, region_id: str) -> Region:
        return self.document.require_region(region_id)

    def mark_region_inspected(self, region_id: str) -> None:
        if self.get_region(region_id) is None:
            raise ValueError(f"unknown region_id: {region_id}")
        self.inspected_regions.add(region_id)

    def add_zoom_region_view(self, view: ZoomRegionView) -> None:
        if self.get_region(view.region_id) is None:
            raise ValueError(f"unknown zoom source region_id: {view.region_id}")
        self.zoom_regions.add_view(view)

    def get_zoom_region_view(self, region_id: str) -> ZoomRegionView | None:
        return self.zoom_regions.get_view(region_id)

    def add_crop_region_view(self, view: CropRegionView) -> None:
        if self.get_region(view.region_id) is None:
            raise ValueError(f"unknown crop source region_id: {view.region_id}")
        self.crop_regions.add_view(view)

    def get_crop_region_view(self, region_id: str) -> CropRegionView | None:
        return self.crop_regions.get_view(region_id)

    def add_rotate_region_view(self, view: RotateRegionView) -> None:
        if self.get_region(view.region_id) is None:
            raise ValueError(f"unknown rotate source region_id: {view.region_id}")
        self.rotate_regions.add_view(view)

    def get_rotate_region_view(self, region_id: str) -> RotateRegionView | None:
        return self.rotate_regions.get_view(region_id)

    def add_evidence(self, evidence: Evidence) -> None:
        if (
            evidence.region_id
            and self.get_region(evidence.region_id) is None
            and not is_page_level_synthetic_region_id(evidence.region_id)
        ):
            evidence.metadata.setdefault("unresolved_region_id", evidence.region_id)
            evidence.region_id = None
        if evidence.page_index is not None and self.get_page(evidence.page_index) is None:
            raise ValueError(f"unknown evidence source page_index: {evidence.page_index}")
        self.evidence.append(evidence)

    def add_search_history_entry(self, entry: SearchHistoryEntry) -> None:
        self.search_history.add_entry(entry)

    def add_figure_insight(self, insight: FigureInsight) -> None:
        if self.get_page(insight.page_index) is None:
            raise ValueError(f"unknown figure insight page_index: {insight.page_index}")
        self.figure_insights.add_insight(insight)

    def add_evidence_assessment(self, assessment: EvidenceAssessment) -> None:
        for region_id in assessment.region_ids:
            if self.get_region(region_id) is None:
                raise ValueError(f"unknown evidence assessment region_id: {region_id}")
        for page_index in assessment.page_indices:
            if self.get_page(page_index) is None:
                raise ValueError(f"unknown evidence assessment page_index: {page_index}")
        self.evidence_assessment_history.add_assessment(assessment)

    def add_trace_step(self, action: Action, observation: Observation) -> TraceStep:
        step = TraceStep(
            step_index=len(self.action_trace),
            action=action,
            observation=observation,
        )
        self.action_trace.append(step)
        step.state_summary = self.summary()
        return step

    def get_active_skill(self) -> ActiveSkill | None:
        raw = self.metadata.get(self.ACTIVE_SKILL_METADATA_KEY)
        if not isinstance(raw, dict):
            return None
        if "name" not in raw:
            return None
        return ActiveSkill.from_dict(raw)

    def set_active_skill(
        self,
        skill: ActiveSkill | str,
        *,
        reason: str | None = None,
        source: str | None = None,
        path: str | None = None,
    ) -> ActiveSkill:
        active_skill = (
            skill
            if isinstance(skill, ActiveSkill)
            else ActiveSkill(
                name=skill,
                reason=reason,
                source=source,
                path=path,
            )
        )
        self.metadata[self.ACTIVE_SKILL_METADATA_KEY] = active_skill.to_dict()
        return active_skill

    def clear_active_skill(self) -> None:
        self.metadata.pop(self.ACTIVE_SKILL_METADATA_KEY, None)

    def add_pending_event(self, event: str, payload: dict[str, Any]) -> None:
        if not event.strip():
            raise ValueError("pending event name must not be empty")
        pending = self.metadata.get(self.PENDING_EVENTS_METADATA_KEY)
        if not isinstance(pending, list):
            pending = []
            self.metadata[self.PENDING_EVENTS_METADATA_KEY] = pending
        pending.append(
            {
                "event": event,
                "payload": dict(payload),
            }
        )

    def pop_pending_events(self) -> list[tuple[str, dict[str, Any]]]:
        raw = self.metadata.pop(self.PENDING_EVENTS_METADATA_KEY, [])
        if not isinstance(raw, list):
            return []
        events: list[tuple[str, dict[str, Any]]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            event = item.get("event")
            payload = item.get("payload")
            if not isinstance(event, str) or not isinstance(payload, dict):
                continue
            events.append((event, dict(payload)))
        return events

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-friendly state summary for traces and prompts."""
        active_skill = self.get_active_skill()
        return {
            "run_id": self.run_id,
            "document_id": self.document.document_id,
            "task_id": self.task.task_id,
            "task": self.task.prompt,
            "status": self.status,
            "pages": len(self.document.pages),
            "regions": sum(len(page.regions) for page in self.document.pages),
            "pages_with_ocr": sum(
                1 for page in self.document.pages if has_searchable_text(page.ocr_text)
            ),
            "regions_with_ocr": sum(
                1
                for page in self.document.pages
                for region in page.regions
                if has_searchable_text(region.text)
            ),
            "inspected_regions": len(self.inspected_regions),
            "zoomed_regions": len(self.zoom_regions.views_by_region_id),
            "crop_regions": len(self.crop_regions.views_by_region_id),
            "rotated_regions": len(self.rotate_regions.views_by_region_id),
            "search_history": len(self.search_history.entries_by_id),
            "figure_insights": len(self.figure_insights.insights_by_key),
            "evidence_assessments": len(self.evidence_assessment_history.assessments_by_id),
            "evidence": len(self.evidence),
            "actions": len(self.action_trace),
            "has_final_answer": self.final_answer is not None,
            "has_active_skill": active_skill is not None,
            "active_skill_name": active_skill.name if active_skill is not None else None,
        }

    def planner_summary(self) -> dict[str, Any]:
        active_skill = self.get_active_skill()
        return {
            "run_id": self.run_id,
            "document_id": self.document.document_id,
            "task_id": self.task.task_id,
            "task": self.task.prompt,
            "status": self.status,
            "has_final_answer": self.final_answer is not None,
            "has_active_skill": active_skill is not None,
            "active_skill_name": active_skill.name if active_skill is not None else None,
        }

    def build_exploration_summary(self) -> dict[str, Any]:
        page_search_history: list[dict[str, Any]] = []
        region_search_history: list[dict[str, Any]] = []
        inspected_pages: set[int] = set()
        candidate_region_inspected: set[str] = set()
        evidence_attempt_history: list[dict[str, Any]] = []

        for step in self.action_trace:
            if not step.observation.success:
                continue
            summary = _summarize_observation(step.observation, text_limit=160)
            if not isinstance(summary, dict):
                continue

            if step.action.action_type == "internal_search" and summary.get("kind") == "internal_search":
                queries = _normalize_string_list(summary.get("queries"))
                retriever = _optional_non_empty_str(step.action.parameters.get("retriever"))
                top_k = _optional_int_value(step.action.parameters.get("top_k"))
                if summary.get("mode") == "pages":
                    hit_pages = _page_ids_from_indexes(
                        summary.get("hit_page_ids"),
                        document=self.document,
                    )
                    page_search_history.append(
                        {
                            "queries": queries,
                            "retriever": retriever,
                            "top_k": top_k,
                            "hit_page_ids": hit_pages,
                        }
                    )
                elif summary.get("mode") == "regions":
                    region_search_history.append(
                        {
                            "queries": queries,
                            "retriever": retriever,
                            "top_k": top_k,
                            "page_ids": _target_page_ids(
                                self,
                                step.action.target,
                                fallback_page_indexes=summary.get("page_indices"),
                            ),
                            "hit_region_ids": _normalize_string_list(summary.get("hit_region_ids")),
                        }
                    )

            if step.action.action_type in {
                "extract_evidence",
                "understand_figures",
                "parse_table",
                "parse_chart",
                "parse_formula",
            }:
                inspected_pages.update(
                    _target_page_ids(
                        self,
                        step.action.target,
                        fallback_page_indexes=_observation_page_indexes(summary),
                    )
                )

            if step.action.action_type == "extract_evidence" and summary.get("kind") == "evidence":
                candidate_region_inspected.update(
                    _normalize_string_list(step.action.target.get("region_ids"))
                )
                evidence_attempt_history.append(
                    {
                        "page_ids": _target_page_ids(
                            self,
                            step.action.target,
                            fallback_page_indexes=summary.get("page_indexes"),
                        ),
                        "region_ids": _normalize_string_list(step.action.target.get("region_ids")),
                        "answerability_status": _optional_non_empty_str(
                            summary.get("answerability_status")
                        ) or "inconclusive",
                        "missing_information": _optional_non_empty_str(summary.get("missing_information")),
                    }
                )

        return {
            "page_search_history": page_search_history,
            "region_search_history": region_search_history,
            "candidate_page_inspected": sorted(inspected_pages),
            "candidate_region_inspected": sorted(candidate_region_inspected),
            "evidence_attempt_history": evidence_attempt_history,
        }

    def build_planner_context(
        self,
        *,
        text_limit: int = 160,
        document_memory: dict[str, Any] | None = None,
        document_overview: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the minimal planner-facing state for the current question."""
        memory = document_memory if document_memory is not None else document_overview
        return {
            "document_memory": _compact_mapping(
                dict(
                    memory
                    or {
                        "document_id": self.document.document_id,
                        "page_ids": [page_id_from_index(page.page_index, document=self.document) for page in self.document.pages],
                    }
                )
            ),
            "task_memory": {
                "query": self.task.prompt,
                "action_history": [
                    {
                        "step_index": step.step_index,
                        "result": _planner_trace_result(step, text_limit=text_limit, document=self.document),
                    }
                    for step in self.action_trace
                ],
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "document": self.document.to_dict(),
            "task": self.task.to_dict(),
            "inspected_regions": sorted(self.inspected_regions),
            "zoom_regions": self.zoom_regions.to_dict(),
            "crop_regions": self.crop_regions.to_dict(),
            "rotate_regions": self.rotate_regions.to_dict(),
            "search_history": self.search_history.to_dict(),
            "figure_insights": self.figure_insights.to_dict(),
            "evidence_assessment_history": self.evidence_assessment_history.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "action_trace": [step.to_dict() for step in self.action_trace],
            "status": self.status,
            "final_answer": self.final_answer,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunState:
        return cls(
            run_id=str(data["run_id"]),
            document=DocumentState.from_dict(data["document"]),
            task=Task.from_dict(data["task"]),
            inspected_regions=set(data.get("inspected_regions") or []),
            zoom_regions=ZoomRegionState.from_dict(
                dict(data.get("zoom_regions") or {})
            ),
            crop_regions=CropRegionState.from_dict(
                dict(data.get("crop_regions") or {})
            ),
            rotate_regions=RotateRegionState.from_dict(
                dict(data.get("rotate_regions") or {})
            ),
            search_history=SearchHistoryState.from_dict(
                dict(data.get("search_history") or {})
            ),
            figure_insights=FigureInsightState.from_dict(
                dict(data.get("figure_insights") or {})
            ),
            evidence_assessment_history=EvidenceAssessmentHistoryState.from_dict(
                dict(data.get("evidence_assessment_history") or {})
            ),
            evidence=[Evidence.from_dict(item) for item in data.get("evidence", [])],
            action_trace=[TraceStep.from_dict(item) for item in data.get("action_trace", [])],
            status=data.get("status", "running"),
            final_answer=data.get("final_answer"),
            metadata=dict(data.get("metadata") or {}),
        )


###############################################################################
# Run Output
###############################################################################


@dataclass(slots=True)
class RunResult:
    """Final result returned by a DocClaw run."""

    state: RunState
    status: RunStatus
    answer: str | None = None
    reason: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in RUN_STATUSES:
            raise ValueError(f"unsupported run status: {self.status}")

    @property
    def evidence(self) -> list[Evidence]:
        return self.state.evidence

    @property
    def trace(self) -> list[TraceStep]:
        return self.state.action_trace

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "answer": self.answer,
            "reason": self.reason,
            "error": self.error,
            "state": self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        return cls(
            status=data.get("status", "running"),
            answer=data.get("answer"),
            reason=data.get("reason"),
            error=data.get("error"),
            state=RunState.from_dict(data["state"]),
        )


def _summarize_observation(
    observation: Observation,
    *,
    text_limit: int,
) -> dict[str, Any]:
    data = observation.data
    if not isinstance(data, dict):
        return {}

    if "results" in data and isinstance(data.get("results"), list):
        results = [item for item in data["results"] if isinstance(item, dict)]
        if results and all(
            any(key in item for key in ("answer", "reason", "question"))
            for item in results
        ):
            page_indexes = sorted({
                item.get("page_index")
                for item in results
                if item.get("page_index") is not None
            })
            return {
                "kind": "figure_insight",
                "page_indexes": page_indexes,
            }
        if data.get("mode") in {"zoom", "crop"} and isinstance(data.get("view_results"), list):
            page_indexes = sorted({
                item.get("page_index")
                for item in results
                if item.get("page_index") is not None
            })
            region_ids = sorted({
                item.get("region_id")
                for item in results
                if item.get("region_id")
            })
            return {
                "kind": "ocr_enhancement",
                "mode": data.get("mode"),
                "page_indexes": page_indexes,
                "region_ids": region_ids,
                "sources": sorted(str(key) for key in dict(data.get("sources") or {}).keys()),
            }
        if results and all("artifact_path" in item for item in results):
            page_indexes = sorted({
                item.get("page_index")
                for item in results
                if item.get("page_index") is not None
            })
            region_ids = sorted({
                item.get("region_id")
                for item in results
                if item.get("region_id")
            })
            artifact_kind = str(results[0].get("artifact_kind") or "")
            target_long_side_px = [
                int(value)
                for item in results
                if isinstance((value := item.get("target_long_side_px")), (int, float))
            ]
            crop_offsets = [
                {
                    "left_px": int(item.get("left_px") or 0),
                    "right_px": int(item.get("right_px") or 0),
                    "top_px": int(item.get("top_px") or 0),
                    "bottom_px": int(item.get("bottom_px") or 0),
                }
                for item in results
                if any(
                    item.get(key) is not None
                    for key in ("left_px", "right_px", "top_px", "bottom_px")
                )
            ]
            angle_degrees = [
                float(value)
                for item in results
                if isinstance((value := item.get("angle_degree")), (int, float))
            ]
            if artifact_kind == "crop_view":
                return {
                    "kind": "crop",
                    "page_indexes": page_indexes,
                    "region_ids": region_ids,
                    "crop_offsets": crop_offsets,
                }
            if artifact_kind == "rotate_view":
                return {
                    "kind": "rotate",
                    "page_indexes": page_indexes,
                    "region_ids": region_ids,
                    "angle_degrees": angle_degrees,
                }
            return {
                "kind": "zoom",
                "page_indexes": page_indexes,
                "region_ids": region_ids,
                "target_long_side_px": target_long_side_px,
            }
        page_indexes = sorted({
            item.get("page_index")
            for item in results
            if item.get("page_index") is not None
        })
        region_ids = sorted({
            item.get("region_id")
            for item in results
            if item.get("region_id")
        })
        return {
            "kind": "ocr",
            "page_indexes": page_indexes,
            "region_ids": region_ids,
            "sources": sorted(str(key) for key in dict(data.get("sources") or {}).keys()),
        }

    if "refinement_actions" in data:
        refinement_actions = [
            item
            for item in data.get("refinement_actions", [])
            if isinstance(item, dict)
        ] if isinstance(data.get("refinement_actions"), list) else []
        return {
            "kind": "inspect_ocr",
            "source": data.get("source"),
            "refinement_actions": refinement_actions[:3],
        }

    if "updated_region_ids" in data or "kept_original_region_ids" in data:
        updated_region_ids = [
            item
            for item in data.get("updated_region_ids", [])
            if isinstance(item, str) and item.strip()
        ] if isinstance(data.get("updated_region_ids"), list) else []
        kept_original_region_ids = [
            item
            for item in data.get("kept_original_region_ids", [])
            if isinstance(item, str) and item.strip()
        ] if isinstance(data.get("kept_original_region_ids"), list) else []
        return {
            "kind": "candidate_selection",
            "source": data.get("source"),
            "updated_region_ids": updated_region_ids[:5],
            "kept_original_region_ids": kept_original_region_ids[:5],
        }

    if "pages" in data and isinstance(data.get("pages"), list):
        pages = data["pages"]
        if any(isinstance(page, dict) and "regions" in page for page in pages):
            page_indexes = [
                page.get("page_index") for page in pages if isinstance(page, dict)
            ]
            return {
                "kind": "layout",
                "page_indexes": page_indexes,
                "region_ids": [
                    region.get("region_id")
                    for page in pages
                    if isinstance(page, dict) and isinstance(page.get("regions"), list)
                    for region in page.get("regions", [])
                    if isinstance(region, dict) and region.get("region_id") is not None
                ],
            }
        return {
            "kind": "layout",
            "page_indexes": [page.get("page_index") for page in pages if isinstance(page, dict)],
            "reused_page_indexes": [
                page.get("page_index")
                for page in pages
                if isinstance(page, dict) and page.get("skipped") and page.get("page_index") is not None
            ],
        }

    if "hits" in data and isinstance(data.get("hits"), list) and "searched_pages" in data:
        hits = [item for item in data["hits"] if isinstance(item, dict)]
        return {
            "kind": "internal_search",
            "search_id": data.get("search_id"),
            "queries": list(data.get("queries") or []),
            "mode": data.get("mode"),
            "page_indices": list(data.get("page_indices") or []),
            "searched_pages": list(data.get("searched_pages") or []),
            "hit_page_ids": list(data.get("hit_page_ids") or []),
            "hit_region_ids": list(data.get("hit_region_ids") or []),
            "auto_recovery": [
                {
                    "action_type": item.get("action_type"),
                    "target": item.get("target"),
                    "success": item.get("success"),
                }
                for item in data.get("auto_recovery", [])
                if isinstance(item, dict)
            ],
        }

    if "artifact_path" in data:
        kind = str(data.get("artifact_kind") or "zoom_view")
        return {
            "kind": (
                "crop"
                if kind == "crop_view"
                else "rotate"
                if kind == "rotate_view"
                else "zoom"
            ),
            "page_index": data.get("page_index"),
            "region_id": data.get("region_id"),
            "artifact_path": data.get("artifact_path"),
            "artifact_width": data.get("artifact_width"),
            "artifact_height": data.get("artifact_height"),
            "target_long_side_px": data.get("target_long_side_px"),
            "left_px": data.get("left_px"),
            "right_px": data.get("right_px"),
            "top_px": data.get("top_px"),
            "bottom_px": data.get("bottom_px"),
            "angle_degree": data.get("angle_degree"),
        }

    if "evidence" in data and isinstance(data.get("evidence"), list):
        evidence = data["evidence"]
        return {
            "kind": "evidence",
            "assessment_id": data.get("assessment_id"),
            "source": data.get("source"),
            "answerability_status": data.get("answerability_status"),
            "missing_information": data.get("missing_information"),
            "evidence_ids": [
                item.get("evidence_id")
                for item in evidence
                if isinstance(item, dict) and item.get("evidence_id") is not None
            ],
            "page_indexes": sorted({
                item.get("page_index")
                for item in evidence
                if isinstance(item, dict) and item.get("page_index") is not None
            }),
            "region_ids": sorted({
                item.get("region_id")
                for item in evidence
                if isinstance(item, dict) and item.get("region_id")
            }),
        }

    if "answer" in data:
        answer = data.get("answer")
        return {
            "kind": "answer",
            "source": data.get("source"),
            "answer_preview": text_preview(answer if isinstance(answer, str) else None, limit=text_limit),
            "evidence_ids": list(data.get("evidence_ids") or []),
        }

    if "reason" in data:
        return {
            "kind": "stop",
            "reason": data.get("reason"),
            "has_answer": bool(data.get("answer")),
        }

    return {
        "kind": "generic",
        "data_keys": sorted(str(key) for key in data.keys()),
    }


def _planner_trace_result(
    step: TraceStep,
    *,
    text_limit: int,
    document: Any | None = None,
) -> dict[str, Any]:
    return _compact_mapping(
        {
            "operation": step.action.action_type,
            "status": "ok" if step.observation.success else "error",
            "input": _planner_trace_input(step, document=document),
            "output": _planner_trace_output(step, text_limit=text_limit, document=document),
        }
    )


def _planner_trace_input(step: TraceStep, *, document: Any | None = None) -> dict[str, Any]:
    action = step.action
    payload: dict[str, Any] = {}
    if isinstance(action.target, dict):
        payload.update(action.target)
    if isinstance(action.parameters, dict):
        payload.update(action.parameters)
    data = step.observation.data if isinstance(step.observation.data, dict) else {}
    if action.action_type == "internal_search" and "mode" not in payload:
        mode = _optional_non_empty_str(data.get("mode"))
        if mode is not None:
            payload["mode"] = mode
    if action.action_type == "internal_search" and "page_indices" not in payload and "page_ids" not in payload:
        page_indices = data.get("page_indices")
        if isinstance(page_indices, list):
            payload["page_indices"] = page_indices
    return _compact_mapping(plannerize_page_refs(payload, document=document))


def _planner_trace_output(
    step: TraceStep,
    *,
    text_limit: int,
    document: Any | None = None,
) -> dict[str, Any]:
    if not step.observation.success:
        return _compact_mapping(
            {
                "error": text_preview(step.observation.error, limit=text_limit),
            }
        )

    action_type = step.action.action_type
    data = step.observation.data if isinstance(step.observation.data, dict) else {}

    if action_type == "internal_search":
        return _planner_trace_output_for_internal_search(data, document=document)
    if action_type == "parse_layout":
        return _planner_trace_output_for_layout(data, document=document)
    if action_type == "ocr":
        return _planner_trace_output_for_ocr(data, document=document)
    if action_type == "ocr_enhancement":
        return _planner_trace_output_for_ocr_enhancement(data, document=document)
    if action_type == "understand_figures":
        return _planner_trace_output_for_figure(
            data,
            mode=_optional_non_empty_str(step.action.parameters.get("mode"))
            if isinstance(step.action.parameters, dict)
            else None,
            document=document,
        )
    if action_type == "select_pages":
        return _planner_trace_output_for_select_pages(data)
    if action_type == "extract_evidence":
        return _planner_trace_output_for_evidence(data, text_limit=text_limit, document=document)
    if action_type in {"answer_from_evidence", "answer_json"}:
        return _planner_trace_output_for_answer(data, text_limit=text_limit)
    if action_type == "stop":
        return _planner_trace_output_for_stop(data, text_limit=text_limit)
    if action_type == "select_ocr":
        return _planner_trace_output_for_candidate_selection(data)
    if action_type in {"zoom", "crop", "rotate"}:
        return _planner_trace_output_for_region_artifacts(data, document=document)
    if action_type in {"parse_table", "parse_chart", "parse_formula"}:
        return _planner_trace_output_for_parser_results(data, action_type=action_type, document=document)
    if action_type == "inspect_ocr":
        return _planner_trace_output_for_inspect_ocr(data)
    if action_type == "transcribe":
        return _planner_trace_output_for_transcription(data, document=document)

    summary = _summarize_observation(step.observation, text_limit=text_limit)
    return _compact_mapping(plannerize_page_refs(summary, document=document))


def _planner_trace_output_for_internal_search(data: dict[str, Any], *, document: Any | None = None) -> dict[str, Any]:
    auto_recovery_steps: list[dict[str, Any]] = []
    raw_auto_recovery = data.get("auto_recovery")
    if isinstance(raw_auto_recovery, list):
        for item in raw_auto_recovery:
            if not isinstance(item, dict):
                continue
            target = item.get("target")
            parameters = item.get("parameters")
            input_payload: dict[str, Any] = {}
            if isinstance(target, dict):
                input_payload.update(target)
            if isinstance(parameters, dict):
                input_payload.update(parameters)
            auto_recovery_steps.append(
                _compact_mapping(
                    {
                        "operation": item.get("action_type"),
                        "input": plannerize_page_refs(input_payload, document=document),
                    }
                )
            )
    return _compact_mapping(
        plannerize_page_refs(
            {
                "hit_page_ids": list(data.get("hit_page_ids") or []),
                "hit_region_ids": list(data.get("hit_region_ids") or []),
                "auto_recovery": auto_recovery_steps,
            },
            document=document,
        )
    )


def _planner_trace_output_for_layout(data: dict[str, Any], *, document: Any | None = None) -> dict[str, Any]:
    pages = data.get("pages")
    page_indexes: list[int] = []
    region_ids: list[str] = []
    if isinstance(pages, list):
        for page in pages:
            if not isinstance(page, dict):
                continue
            page_index = page.get("page_index")
            if isinstance(page_index, int):
                page_indexes.append(page_index)
            regions = page.get("regions")
            if isinstance(regions, list):
                for region in sorted(regions, key=_planner_layout_region_sort_key):
                    if not isinstance(region, dict):
                        continue
                    region_id = region.get("region_id")
                    if isinstance(region_id, str) and region_id.strip():
                        region_ids.append(region_id)
    return _compact_mapping(
        plannerize_page_refs(
            {
                "page_indices": _dedupe_ints(page_indexes),
                "region_ids": _normalize_string_list(region_ids),
            },
            document=document,
        )
    )


def _planner_layout_region_sort_key(region: Any) -> tuple[int, str]:
    if not isinstance(region, dict):
        return (10**9, "")
    metadata = region.get("metadata")
    layout = metadata.get("layout") if isinstance(metadata, dict) else None
    raw_sequence_order = layout.get("sequence_order") if isinstance(layout, dict) else None
    sequence_order = (
        int(raw_sequence_order)
        if isinstance(raw_sequence_order, (int, float))
        else 10**9
    )
    region_id = region.get("region_id")
    return (
        sequence_order,
        str(region_id).strip() if isinstance(region_id, str) else "",
    )


def _planner_trace_output_for_ocr(data: dict[str, Any], *, document: Any | None = None) -> dict[str, Any]:
    results = data.get("results")
    page_indexes: list[int] = []
    region_ids: list[str] = []
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            page_index = item.get("page_index")
            if isinstance(page_index, int):
                page_indexes.append(page_index)
            region_id = item.get("region_id")
            if isinstance(region_id, str) and region_id.strip():
                region_ids.append(region_id)
    return _compact_mapping(
        plannerize_page_refs(
            {
                "page_indices": _dedupe_ints(page_indexes),
                "region_ids": _normalize_string_list(region_ids),
            },
            document=document,
        )
    )


def _planner_trace_output_for_figure(
    data: dict[str, Any],
    *,
    mode: str | None,
    document: Any | None = None,
) -> dict[str, Any]:
    results = data.get("results")
    page_indexes: list[int] = []
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            page_index = item.get("page_index")
            if isinstance(page_index, int):
                page_indexes.append(page_index)
    payload: dict[str, Any] = {
        "page_indices": _dedupe_ints(page_indexes),
    }
    return _compact_mapping(plannerize_page_refs(payload, document=document))


def _planner_trace_output_for_select_pages(data: dict[str, Any]) -> dict[str, Any]:
    results = data.get("results")
    selected_page_ids: list[str] = []
    selection_reason: str | None = None
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            raw_selected_page_ids = item.get("selected_page_ids")
            if isinstance(raw_selected_page_ids, list):
                for raw_page_id in raw_selected_page_ids:
                    if not isinstance(raw_page_id, str):
                        continue
                    if raw_page_id in selected_page_ids:
                        continue
                    selected_page_ids.append(raw_page_id)
                if selection_reason is None:
                    selection_reason = _optional_non_empty_str(item.get("reason"))
    return _compact_mapping(
        {
            "selected_page_ids": selected_page_ids,
            "reason": selection_reason,
        }
    )


def _planner_trace_output_for_evidence(
    data: dict[str, Any],
    *,
    text_limit: int,
    document: Any | None = None,
) -> dict[str, Any]:
    evidence = data.get("evidence")
    page_indexes: list[int] = []
    region_ids: list[str] = []
    evidence_ids: list[str] = []
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            page_index = item.get("page_index")
            if isinstance(page_index, int):
                page_indexes.append(page_index)
            region_id = item.get("region_id")
            if isinstance(region_id, str) and region_id.strip():
                region_ids.append(region_id)
            evidence_id = item.get("evidence_id")
            if isinstance(evidence_id, str) and evidence_id.strip():
                evidence_ids.append(evidence_id)
    return _compact_mapping(
        plannerize_page_refs(
            {
                "page_indices": _dedupe_ints(page_indexes),
                "region_ids": _normalize_string_list(region_ids),
                "evidence_ids": _normalize_string_list(evidence_ids),
                "answerability_status": _optional_non_empty_str(data.get("answerability_status")),
                "missing_information": _optional_non_empty_str(data.get("missing_information")),
            },
            document=document,
        )
    )


def _planner_trace_output_for_answer(
    data: dict[str, Any],
    *,
    text_limit: int,
) -> dict[str, Any]:
    answer = _optional_non_empty_str(data.get("answer"))
    return _compact_mapping(
        {
            "answer": text_preview(answer, limit=text_limit * 2),
        }
    )


def _planner_trace_output_for_stop(
    data: dict[str, Any],
    *,
    text_limit: int,
) -> dict[str, Any]:
    return _compact_mapping(
        {
            "answer": _optional_non_empty_str(data.get("answer")),
            "reason": text_preview(
                _optional_non_empty_str(data.get("reason")),
                limit=text_limit,
            ),
        }
    )


def _planner_trace_output_for_candidate_selection(data: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {}
    selections = data.get("selections")
    if isinstance(selections, list):
        for item in selections:
            if not isinstance(item, dict):
                continue
            candidate = _optional_non_empty_str(item.get("selected_candidate"))
            region_id = _optional_non_empty_str(item.get("region_id"))
            if candidate is None or region_id is None:
                continue
            grouped.setdefault(candidate, []).append(region_id)
    return _compact_mapping(
        {
            "selections": {
                candidate: _normalize_string_list(region_ids)
                for candidate, region_ids in grouped.items()
            }
        }
    )


def _planner_trace_output_for_ocr_enhancement(
    data: dict[str, Any],
    *,
    document: Any | None = None,
) -> dict[str, Any]:
    results = data.get("results")
    page_indexes: list[int] = []
    region_ids: list[str] = []
    candidate_kinds: dict[str, list[str]] = {}
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            page_index = item.get("page_index")
            if isinstance(page_index, int):
                page_indexes.append(page_index)
            region_id = _optional_non_empty_str(item.get("region_id"))
            if region_id is None:
                continue
            region_ids.append(region_id)
            candidate_kind = _optional_non_empty_str(item.get("candidate_kind"))
            if candidate_kind is not None:
                candidate_kinds.setdefault(candidate_kind, []).append(region_id)
    return _compact_mapping(
        plannerize_page_refs(
            {
                "page_indices": _dedupe_ints(page_indexes),
                "refined_region_ids": _normalize_string_list(region_ids),
                "candidate_kinds": {
                    candidate: _normalize_string_list(items)
                    for candidate, items in candidate_kinds.items()
                },
            },
            document=document,
        )
    )


def _planner_trace_output_for_region_artifacts(data: dict[str, Any], *, document: Any | None = None) -> dict[str, Any]:
    results = data.get("results")
    page_indexes: list[int] = []
    region_ids: list[str] = []
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            page_index = item.get("page_index")
            if isinstance(page_index, int):
                page_indexes.append(page_index)
            region_id = item.get("region_id")
            if isinstance(region_id, str) and region_id.strip():
                region_ids.append(region_id)
    return _compact_mapping(
        plannerize_page_refs(
            {
                "page_indices": _dedupe_ints(page_indexes),
                "region_ids": _normalize_string_list(region_ids),
            },
            document=document,
        )
    )


def _planner_trace_output_for_parser_results(
    data: dict[str, Any],
    *,
    action_type: str,
    document: Any | None = None,
) -> dict[str, Any]:
    results = data.get("results")
    page_indexes: list[int] = []
    region_ids: list[str] = []
    count = 0
    if isinstance(results, list):
        count = len(results)
        for item in results:
            if not isinstance(item, dict):
                continue
            page_index = item.get("page_index")
            if isinstance(page_index, int):
                page_indexes.append(page_index)
            region_id = item.get("region_id")
            if isinstance(region_id, str) and region_id.strip():
                region_ids.append(region_id)
    count_key = {
        "parse_table": "table_count",
        "parse_chart": "chart_count",
        "parse_formula": "formula_count",
    }.get(action_type, "result_count")
    return _compact_mapping(
        plannerize_page_refs(
            {
                "page_indices": _dedupe_ints(page_indexes),
                "region_ids": _normalize_string_list(region_ids),
                count_key: count,
            },
            document=document,
        )
    )


def _planner_trace_output_for_inspect_ocr(data: dict[str, Any]) -> dict[str, Any]:
    refinement_actions = [
        item
        for item in data.get("refinement_actions", [])
        if isinstance(item, dict)
    ] if isinstance(data.get("refinement_actions"), list) else []
    return _compact_mapping(
        {
            "inspected_region_ids": _normalize_string_list(data.get("inspected_region_ids")),
            "refinement_actions": refinement_actions,
        }
    )


def _planner_trace_output_for_transcription(data: dict[str, Any], *, document: Any | None = None) -> dict[str, Any]:
    pages = data.get("pages")
    page_indexes: list[int] = []
    if isinstance(pages, list):
        for item in pages:
            if not isinstance(item, dict):
                continue
            page_index = item.get("page_index")
            if isinstance(page_index, int):
                page_indexes.append(page_index)
    return _compact_mapping(
        plannerize_page_refs(
            {
                "page_indices": _dedupe_ints(page_indexes),
            },
            document=document,
        )
    )


def _dedupe_ints(values: list[int]) -> list[int]:
    items: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        items.append(value)
        seen.add(value)
    return items


def _compact_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        compacted: dict[Any, Any] = {}
        for key, item in value.items():
            reduced = _compact_mapping(item)
            if reduced is None:
                continue
            if isinstance(reduced, str) and not reduced:
                continue
            if isinstance(reduced, (list, dict)) and not reduced:
                continue
            compacted[key] = reduced
        return compacted
    if isinstance(value, list):
        compacted_items: list[Any] = []
        for item in value:
            reduced = _compact_mapping(item)
            if reduced is None:
                continue
            if isinstance(reduced, str) and not reduced:
                continue
            if isinstance(reduced, (list, dict)) and not reduced:
                continue
            compacted_items.append(reduced)
        return compacted_items
    return value


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for raw_item in value:
        item = str(raw_item).strip()
        if not item or item in seen:
            continue
        items.append(item)
        seen.add(item)
    return items


def _optional_non_empty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _optional_int_value(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _page_ids_from_indexes(value: Any, *, document: Any | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    page_ids: list[str] = []
    seen: set[str] = set()
    for raw_page_index in value:
        if not isinstance(raw_page_index, int):
            continue
        page_id = page_id_from_index(raw_page_index, document=document)
        if page_id in seen:
            continue
        page_ids.append(page_id)
        seen.add(page_id)
    return page_ids


def _observation_page_indexes(summary: dict[str, Any]) -> list[int] | None:
    raw_page_indexes = summary.get("page_indexes")
    if not isinstance(raw_page_indexes, list):
        return None
    return [item for item in raw_page_indexes if isinstance(item, int)]


def _target_page_ids(
    state: RunState,
    target: dict[str, Any],
    *,
    fallback_page_indexes: Any = None,
) -> list[str]:
    page_ids: list[str] = []
    seen: set[str] = set()

    raw_page_ids = target.get("page_ids")
    if isinstance(raw_page_ids, list):
        for raw_page_id in raw_page_ids:
            if not isinstance(raw_page_id, str) or not raw_page_id.strip():
                continue
            page_id = raw_page_id.strip()
            if page_id in seen:
                continue
            page_ids.append(page_id)
            seen.add(page_id)

    raw_page_indexes = target.get("page_indices")
    if isinstance(raw_page_indexes, list):
        for raw_page_index in raw_page_indexes:
            if not isinstance(raw_page_index, int):
                continue
            page_id = page_id_from_index(raw_page_index, document=state.document)
            if page_id in seen:
                continue
            page_ids.append(page_id)
            seen.add(page_id)

    raw_region_ids = target.get("region_ids")
    if isinstance(raw_region_ids, list):
        for raw_region_id in raw_region_ids:
            if not isinstance(raw_region_id, str) or not raw_region_id.strip():
                continue
            region = state.get_region(raw_region_id)
            if region is None:
                continue
            page_id = page_id_from_index(region.page_index, document=state.document)
            if page_id in seen:
                continue
            page_ids.append(page_id)
            seen.add(page_id)

    if page_ids:
        return page_ids

    return _page_ids_from_indexes(fallback_page_indexes, document=state.document)


def _build_action_rollup(
    trace: list[TraceStep],
    *,
    text_limit: int,
) -> list[dict[str, Any]]:
    rollup: dict[str, dict[str, Any]] = {}
    for step in trace:
        key = _action_signature(step.action)
        item = rollup.get(key)
        if item is None:
            item = {
                "action_type": step.action.action_type,
                "target": step.action.target,
                "last_success": False,
                "last_message": None,
                "last_error": None,
                "latest_observation_summary": {},
            }
            rollup[key] = item
        item["last_success"] = step.observation.success
        item["last_message"] = text_preview(step.observation.message, limit=text_limit)
        item["last_error"] = text_preview(step.observation.error, limit=text_limit)
        item["latest_observation_summary"] = _summarize_observation(
            step.observation,
            text_limit=text_limit,
        )
    return list(rollup.values())


def _action_signature(action: Action) -> str:
    return json.dumps(
        {
            "action_type": action.action_type,
            "target": action.target,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
