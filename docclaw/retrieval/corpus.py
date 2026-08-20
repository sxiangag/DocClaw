"""Retrieval-facing data structures built on top of document state."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal
import unicodedata

from docclaw.agent.utils import DocumentState, PageState, Region, has_searchable_text


SearchNodeType = Literal["page", "region"]

SEARCH_NODE_TYPES = {"page", "region"}


def normalize_search_text(text: str) -> str:
    """Return a normalized lexical-search text representation."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


@dataclass(slots=True)
class SearchNode:
    """One document unit that may or may not currently have searchable text."""

    node_id: str
    node_type: SearchNodeType
    page_index: int
    text: str | None
    normalized_text: str | None
    source: str
    image_path: str | None = None
    region_id: str | None = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.node_type not in SEARCH_NODE_TYPES:
            raise ValueError(f"unsupported node_type: {self.node_type}")
        if self.page_index < 0:
            raise ValueError("page_index must be non-negative")
        if not self.node_id.strip():
            raise ValueError("node_id must not be empty")
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if self.text is None:
            if self.normalized_text is not None:
                raise ValueError("normalized_text must be None when text is None")
        else:
            if not self.text.strip():
                raise ValueError("text must not be empty when provided")
            if self.normalized_text is None or not self.normalized_text.strip():
                raise ValueError("normalized_text must not be empty when text is provided")
        if self.image_path is not None and not self.image_path.strip():
            raise ValueError("image_path must not be empty when provided")

    @property
    def has_text(self) -> bool:
        return bool(self.text and self.normalized_text)

    @classmethod
    def from_text(
        cls,
        *,
        node_id: str,
        node_type: SearchNodeType,
        page_index: int,
        text: str,
        source: str,
        image_path: str | None = None,
        region_id: str | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SearchNode:
        return cls(
            node_id=node_id,
            node_type=node_type,
            page_index=page_index,
            text=text,
            normalized_text=normalize_search_text(text),
            source=source,
            image_path=image_path,
            region_id=region_id,
            label=label,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "page_index": self.page_index,
            "region_id": self.region_id,
            "label": self.label,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "source": self.source,
            "image_path": self.image_path,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SearchHit:
    """One ranked retrieval result."""

    rank: int
    score: float
    node: SearchNode
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "node_id": self.node.node_id,
            "node_type": self.node.node_type,
            "page_index": self.node.page_index,
            "region_id": self.node.region_id,
            "label": self.node.label,
            "source": self.node.source,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SkippedUnit:
    """A document unit skipped during corpus construction or retrieval."""

    unit_type: str
    page_index: int | None = None
    region_id: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_type": self.unit_type,
            "page_index": self.page_index,
            "region_id": self.region_id,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SearchCorpus:
    """A retrieval-ready view derived from a document state."""

    document_id: str
    nodes: list[SearchNode] = field(default_factory=list)
    skipped_units: list[SkippedUnit] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")
        seen_node_ids: set[str] = set()
        for node in self.nodes:
            if node.node_id in seen_node_ids:
                raise ValueError(f"duplicate node_id: {node.node_id}")
            seen_node_ids.add(node.node_id)

    @classmethod
    def from_document_state(cls, document: DocumentState) -> SearchCorpus:
        """Create a retrieval wrapper from document state."""

        nodes: list[SearchNode] = []
        skipped_units: list[SkippedUnit] = []

        for page in document.pages:
            nodes.append(_page_node(page))
            if not has_searchable_text(page.ocr_text):
                skipped_units.append(
                    SkippedUnit(
                        unit_type="page",
                        page_index=page.page_index,
                        reason="no searchable page text",
                    )
                )

            for region in page.regions:
                nodes.append(_region_node(region))
                if not has_searchable_text(region.text):
                    skipped_units.append(
                        SkippedUnit(
                            unit_type="region",
                            page_index=region.page_index,
                            region_id=region.region_id,
                            reason="no searchable region text",
                            metadata={"label": region.label},
                        )
                    )

        return cls(
            document_id=document.document_id,
            nodes=nodes,
            skipped_units=skipped_units,
            metadata={
                "page_count": len(document.pages),
                "region_count": sum(len(page.regions) for page in document.pages),
                "pages_with_ocr": sum(1 for page in document.pages if has_searchable_text(page.ocr_text)),
                "regions_with_ocr": sum(
                    1
                    for page in document.pages
                    for region in page.regions
                    if has_searchable_text(region.text)
                ),
                "page_node_count": sum(1 for node in nodes if node.node_type == "page"),
                "region_node_count": sum(1 for node in nodes if node.node_type == "region"),
                "searchable_node_count": sum(1 for node in nodes if node.has_text),
            },
        )

    def summary(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "node_count": len(self.nodes),
            "node_types": sorted({node.node_type for node in self.nodes}),
            "searchable_node_count": sum(1 for node in self.nodes if node.has_text),
            "skipped_units": len(self.skipped_units),
            **self.metadata,
        }


def _page_node(page: PageState) -> SearchNode:
    text = page.ocr_text if has_searchable_text(page.ocr_text) else None
    return SearchNode(
        node_id=f"page:{page.page_index}",
        node_type="page",
        page_index=page.page_index,
        text=text,
        normalized_text=normalize_search_text(text) if text is not None else None,
        source="page_ocr" if text is not None else "page_image",
        image_path=page.image_path,
    )


def _region_node(region: Region) -> SearchNode:
    text = region.text if has_searchable_text(region.text) else None
    return SearchNode(
        node_id=region.region_id,
        node_type="region",
        page_index=region.page_index,
        region_id=region.region_id,
        text=text,
        normalized_text=normalize_search_text(text) if text is not None else None,
        source="region_ocr" if text is not None else "layout_region",
        label=region.label,
        metadata={
            "bbox": list(region.bbox),
            "coordinate_space": region.coordinate_space,
            "content_type": region.type,
        },
    )
