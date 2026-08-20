"""Keyword retrieval implementation based on bm25s."""

from __future__ import annotations

from dataclasses import dataclass, field

import bm25s
import numpy as np

from docclaw.agent.tool.internal_search.common import build_snippets, matches_mode, matches_target
from docclaw.agent.tool.internal_search.internal_search import SEARCH_MODES
from docclaw.retrieval.corpus import SearchCorpus, SearchHit, SearchNode, normalize_search_text


@dataclass(slots=True)
class KeywordRetriever:
    """BM25-based keyword retrieval over search corpus nodes."""

    top_k_default: int = 5
    stopwords: str | list[str] | None = "english"
    _retriever: bm25s.BM25 | None = field(default=None, init=False, repr=False)
    _nodes: list[SearchNode] = field(default_factory=list, init=False, repr=False)
    _corpus_payload: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def build(self, corpus: SearchCorpus) -> None:
        """Build or rebuild the BM25 index for the given corpus."""
        self._nodes = [node for node in corpus.nodes if node.has_text]
        self._corpus_payload = [node.to_dict() for node in self._nodes]

        if not self._nodes:
            self._retriever = None
            return

        tokenized = bm25s.tokenize(
            [node.normalized_text for node in self._nodes],
            lower=False,
            stopwords=self.stopwords if self.stopwords is not None else [],
            show_progress=False,
        )
        retriever = bm25s.BM25(corpus=self._corpus_payload)
        retriever.index(tokenized, show_progress=False)
        self._retriever = retriever

    def search(
        self,
        corpus: SearchCorpus,
        query: str,
        *,
        top_k: int | None = None,
        mode: str = "pages",
        page_index: int | None = None,
        page_indices: list[int] | None = None,
        include_snippets: bool = True,
        max_snippets_per_hit: int | None = None,
    ) -> list[SearchHit]:
        """Return ranked keyword hits for the given query."""
        if not query.strip():
            return []
        if mode not in SEARCH_MODES:
            raise ValueError(f"unsupported search mode: {mode}")

        self.build(corpus)
        if self._retriever is None or not self._nodes:
            return []

        page_filter: set[int] | None = None
        if page_indices is not None:
            page_filter = set(page_indices)
        elif page_index is not None:
            page_filter = {page_index}

        candidate_indices = [
            index
            for index, node in enumerate(self._nodes)
            if matches_mode(node, mode) and matches_target(node, page_filter)
        ]
        if not candidate_indices:
            return []

        k = min(max(1, top_k or self.top_k_default), len(self._nodes))
        weight_mask = np.zeros(len(self._nodes), dtype="float32")
        for index in candidate_indices:
            weight_mask[index] = 1.0

        query_tokens = bm25s.tokenize(
            normalize_search_text(query),
            lower=False,
            stopwords=self.stopwords if self.stopwords is not None else [],
            show_progress=False,
        )
        results, scores = self._retriever.retrieve(
            query_tokens,
            corpus=self._corpus_payload,
            k=k,
            show_progress=False,
            weight_mask=weight_mask,
        )

        documents = results[0]
        document_scores = scores[0]
        hits: list[SearchHit] = []
        for document_payload, score in zip(documents, document_scores, strict=False):
            numeric_score = float(score)
            if numeric_score <= 0.0:
                continue
            node = self._node_from_payload(document_payload)
            snippets = (
                build_snippets(
                    node,
                    query,
                    max_snippets=max_snippets_per_hit,
                )
                if include_snippets
                else []
            )
            hits.append(
                SearchHit(
                    rank=len(hits) + 1,
                    score=numeric_score,
                    node=node,
                    metadata={"snippets": snippets} if snippets else {},
                )
            )
            if len(hits) >= (top_k or self.top_k_default):
                break
        return hits

    def _node_from_payload(self, payload: dict[str, Any]) -> SearchNode:
        return SearchNode(
            node_id=str(payload["node_id"]),
            node_type=payload["node_type"],
            page_index=int(payload["page_index"]),
            image_path=payload.get("image_path"),
            region_id=payload.get("region_id"),
            label=payload.get("label"),
            text=payload.get("text"),
            normalized_text=payload.get("normalized_text"),
            source=str(payload["source"]),
            metadata=dict(payload.get("metadata") or {}),
        )
