"""Planner-facing internal search tool with retriever routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docclaw.agent.tool.internal_search.common import (
    matches_mode,
    matches_skipped_target,
    matches_target,
    merge_query_hits,
    normalize_queries,
    ordered_hit_page_ids,
    ordered_hit_region_ids,
)
from docclaw.agent.tool.internal_search.image import ImageSemanticRetriever
from docclaw.agent.tool.internal_search.internal_search import (
    InternalSearchToolBase,
    normalize_internal_search_top_k,
    normalize_search_target,
)
from docclaw.agent.tool.internal_search.keyword import KeywordRetriever
from docclaw.agent.tool.internal_search.text import TextSemanticRetriever
from docclaw.agent.tool.tool import Tool
from docclaw.retrieval.corpus import SearchCorpus, SearchNode
from docclaw.agent.utils import page_number_from_index


INTERNAL_SEARCH_RETRIEVERS = ("keyword", "text", "visual")
INTERNAL_SEARCH_RETRIEVER_ALIASES = {
    "lexical": "keyword",
    "semantic": "text",
}
IMAGE_LIKE_REGION_LABELS = {
    "chart",
    "image",
    "header_image",
    "footer_image",
    "seal",
}


@dataclass(slots=True)
class InternalSearchTool(InternalSearchToolBase):
    """Planner-facing internal search tool backed by multiple retrievers."""

    keyword_retriever: KeywordRetriever = field(default_factory=KeywordRetriever)
    text_retriever: TextSemanticRetriever = field(default_factory=TextSemanticRetriever)
    image_retriever: ImageSemanticRetriever = field(default_factory=ImageSemanticRetriever)
    layout_tool: Tool | None = None
    ocr_tool: Tool | None = None

    @property
    def parameters_schema(self) -> dict[str, Any]:
        schema = dict(InternalSearchToolBase.parameters_schema.fget(self))
        properties = dict(schema["properties"])
        properties["retriever"] = {
            "type": "string",
            "description": "Retrieval backend to use for this search.",
            "enum": list(INTERNAL_SEARCH_RETRIEVERS),
        }
        schema["properties"] = properties
        schema["required"] = ["retriever", "queries", "top_k"]
        return schema

    def search_internal(
        self,
        state,
        action,
        *,
        enforce_page_text_coverage: bool = True,
    ) -> dict[str, Any]:
        mode, explicit_page_indices = normalize_search_target(
            action.target,
            document=state.document,
        )
        retriever_name = self._normalize_retriever(action.parameters.get("retriever"))
        searcher = self._searcher(retriever_name)
        top_k = normalize_internal_search_top_k(int(action.parameters["top_k"]), mode=mode)
        include_snippets = bool(action.parameters.get("include_snippets", True))
        max_snippets_per_hit = None
        page_filter_set = set(explicit_page_indices) if explicit_page_indices is not None else None

        corpus = SearchCorpus.from_document_state(state.document)
        _validate_explicit_text_coverage(
            corpus,
            mode=mode,
            retriever_name=retriever_name,
            explicit_page_indices=explicit_page_indices,
            enforce_page_text_coverage=enforce_page_text_coverage,
        )
        queries = normalize_queries(action.parameters.get("queries"))
        aggregated_hits = merge_query_hits(
            [
                searcher.search(
                    corpus,
                    query,
                    top_k=top_k,
                    mode=mode,
                    page_indices=explicit_page_indices,
                    include_snippets=include_snippets,
                    max_snippets_per_hit=max_snippets_per_hit,
                )
                for query in queries
            ],
            queries=queries,
            top_k=top_k,
        )

        searched_nodes = [
            node
            for node in corpus.nodes
            if self._node_is_searchable(node, retriever_name)
            and matches_mode(node, mode)
            and matches_target(node, page_filter_set)
        ]
        searched_pages = sorted({node.page_index for node in searched_nodes})
        searched_node_types = sorted({node.node_type for node in searched_nodes})
        skipped_units = (
            []
            if retriever_name == "visual"
            else [
                skipped.to_dict()
                for skipped in corpus.skipped_units
                if matches_skipped_target(skipped, mode, page_filter_set)
            ]
        )
        hit_page_ids = ordered_hit_page_ids(aggregated_hits)
        hit_region_ids = ordered_hit_region_ids(aggregated_hits)

        return {
            "queries": queries,
            "mode": mode,
            "retriever_used": retriever_name,
            "top_k": top_k,
            "page_indices": explicit_page_indices,
            "include_snippets": include_snippets,
            "searched_pages": searched_pages,
            "searched_node_count": len(searched_nodes),
            "searched_node_types": searched_node_types,
            "skipped_units": skipped_units,
            "hit_page_ids": hit_page_ids,
            "hit_region_ids": hit_region_ids,
            "hits": aggregated_hits,
        }

    def _searcher(self, retriever_name: str):
        if retriever_name == "keyword":
            return self.keyword_retriever
        if retriever_name == "text":
            return self.text_retriever
        if retriever_name == "visual":
            return self.image_retriever
        raise ValueError(f"unsupported internal_search retriever: {retriever_name}")

    def _normalize_retriever(self, value: Any) -> str:
        retriever = str(value).strip().lower() if value is not None else ""
        retriever = INTERNAL_SEARCH_RETRIEVER_ALIASES.get(retriever, retriever)
        if retriever not in INTERNAL_SEARCH_RETRIEVERS:
            raise ValueError(f"unsupported internal_search retriever: {retriever}")
        return retriever

    def _node_is_searchable(self, node: SearchNode, retriever_name: str) -> bool:
        if retriever_name in {"keyword", "text"}:
            return node.has_text
        return node.node_type == "page" and _node_has_image(node)


def _node_has_image(node: SearchNode) -> bool:
    value = node.image_path
    if not isinstance(value, str) or not value.strip():
        return False
    return Path(value).expanduser().exists()


def _validate_explicit_text_coverage(
    corpus: SearchCorpus,
    *,
    mode: str,
    retriever_name: str,
    explicit_page_indices: list[int] | None,
    enforce_page_text_coverage: bool,
) -> None:
    if retriever_name not in {"keyword", "text"}:
        return
    page_filter = set(explicit_page_indices) if explicit_page_indices is not None else None
    if mode == "pages":
        if not enforce_page_text_coverage:
            return
        missing_pages = sorted(
            {
                skipped.page_index
                for skipped in corpus.skipped_units
                if skipped.unit_type == "page"
                and (page_filter is None or skipped.page_index in page_filter)
            }
        )
        if missing_pages:
            missing_physical_pages = [
                page_number_from_index(page_index)
                for page_index in missing_pages
            ]
            scope_description = (
                "every target page"
                if page_filter is not None
                else "the current search scope"
            )
            raise ValueError(
                "internal_search page-mode "
                f"retriever='{retriever_name}' requires searchable page text across "
                f"{scope_description}; missing page OCR/text on pages_ids: {missing_physical_pages}. "
                "Run page-level `ocr` on those pages first."
            )
        return

    if mode == "regions":
        skipped_regions = [
            skipped
            for skipped in corpus.skipped_units
            if skipped.unit_type == "region"
            and (page_filter is None or skipped.page_index in page_filter)
            and not _is_image_like_skipped_region(skipped)
        ]
        if skipped_regions:
            missing_pages = sorted(
                {
                    skipped.page_index
                    for skipped in skipped_regions
                    if isinstance(skipped.page_index, int)
                }
            )
            missing_physical_pages = [
                page_number_from_index(page_index)
                for page_index in missing_pages
            ]
            scope_description = (
                "every region within the explicit target pages"
                if page_filter is not None
                else "the current search scope"
            )
            raise ValueError(
                "internal_search region-mode "
                f"retriever='{retriever_name}' requires searchable region text across "
                f"{scope_description}; missing region OCR/text on "
                f"physical pages={missing_physical_pages}. Run layout analysis and region-level "
                "`ocr` on those pages/regions first."
            )


def _is_image_like_skipped_region(skipped: Any) -> bool:
    if not hasattr(skipped, "metadata"):
        return False
    metadata = skipped.metadata
    if not isinstance(metadata, dict):
        return False
    label = str(metadata.get("label") or "").strip().lower()
    return label in IMAGE_LIKE_REGION_LABELS
