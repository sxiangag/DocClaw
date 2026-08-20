"""Text internal search built around ColBERT-style late interaction."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Protocol

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from docclaw.agent.tool.internal_search.common import build_snippets, matches_mode, matches_target
from docclaw.agent.tool.internal_search.internal_search import SEARCH_MODES
from docclaw.retrieval.corpus import SearchCorpus, SearchHit, SearchNode


class TextEmbedder(Protocol):
    """Protocol for late-interaction text embedders."""

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]:
        """Return one token-embedding matrix per document."""

    def encode_query(self, query: str) -> np.ndarray:
        """Return one token-embedding matrix for a query."""

    def score(self, query_embedding: np.ndarray, document_embedding: np.ndarray) -> float:
        """Return a similarity score for one query/document pair."""


@dataclass(slots=True)
class ColBERTTextEmbedder:
    """ColBERT-style text encoder using token-level late interaction."""

    model_name: str = "colbert-ir/colbertv2.0"
    device: str = "auto"
    max_length: int = 512
    batch_size: int = 8
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)
    _resolved_device: str | None = field(default=None, init=False, repr=False)

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]:
        return self._encode(texts)

    def encode_query(self, query: str) -> np.ndarray:
        embeddings = self._encode([query])
        return embeddings[0] if embeddings else np.zeros((0, 1), dtype="float32")

    def score(self, query_embedding: np.ndarray, document_embedding: np.ndarray) -> float:
        if query_embedding.size == 0 or document_embedding.size == 0:
            return 0.0
        similarities = query_embedding @ document_embedding.T
        return float(np.max(similarities, axis=1).sum() / max(1, query_embedding.shape[0]))

    def _encode(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        encoded_batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                output = self._model(**inputs)
                token_embeddings = output.last_hidden_state
                token_embeddings = self._torch.nn.functional.normalize(token_embeddings, p=2, dim=-1)
            attention_mask = inputs["attention_mask"].bool().detach().cpu().numpy()
            vectors = token_embeddings.detach().cpu().float().numpy()
            for row_index, mask in enumerate(attention_mask):
                encoded_batches.append(vectors[row_index][mask].astype("float32", copy=False))
        return encoded_batches

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        self._torch = torch
        self._resolved_device = _resolve_torch_device(torch, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self._resolved_device)
        self._model.eval()


@dataclass(slots=True)
class TextSemanticRetriever:
    """Text semantic retrieval with ColBERT-style late interaction."""

    top_k_default: int = 5
    embedder: TextEmbedder = field(default_factory=ColBERTTextEmbedder)
    min_score: float = 0.05
    _document_id: str | None = field(default=None, init=False, repr=False)
    _nodes: list[SearchNode] = field(default_factory=list, init=False, repr=False)
    _node_embeddings: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _node_text_hashes: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _backend: str = field(default="colbert", init=False, repr=False)

    def build(self, corpus: SearchCorpus) -> None:
        """Build or incrementally refresh the text semantic index."""
        if self._document_id != corpus.document_id:
            self._document_id = corpus.document_id
            self._node_embeddings.clear()
            self._node_text_hashes.clear()

        self._nodes = [node for node in corpus.nodes if node.has_text]
        current_node_ids = {node.node_id for node in self._nodes}
        stale_node_ids = set(self._node_embeddings) - current_node_ids
        for node_id in stale_node_ids:
            self._node_embeddings.pop(node_id, None)
            self._node_text_hashes.pop(node_id, None)

        if not self._nodes:
            return

        nodes_to_encode: list[SearchNode] = []
        texts_to_encode: list[str] = []
        for node in self._nodes:
            assert node.normalized_text is not None
            text_hash = _text_hash(node.normalized_text)
            if self._node_text_hashes.get(node.node_id) == text_hash:
                continue
            nodes_to_encode.append(node)
            texts_to_encode.append(node.normalized_text)

        if not nodes_to_encode:
            return

        embeddings = _encode_unique_text_embeddings(
            self.embedder,
            texts_to_encode,
        )
        for node, embedding in zip(nodes_to_encode, embeddings, strict=False):
            self._node_embeddings[node.node_id] = embedding
            self._node_text_hashes[node.node_id] = _text_hash(node.normalized_text)
        self._backend = "colbert"

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
        """Return ranked text-semantic hits for the given query."""
        if not query.strip():
            return []
        if mode not in SEARCH_MODES:
            raise ValueError(f"unsupported search mode: {mode}")

        self.build(corpus)
        if not self._nodes:
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

        ranked = self._rank_candidates(candidate_indices, query)
        limit = top_k or self.top_k_default
        hits: list[SearchHit] = []
        for node_index, score in ranked:
            numeric_score = float(score)
            if numeric_score < self.min_score:
                continue
            node = self._nodes[node_index]
            snippets = (
                build_snippets(
                    node,
                    query,
                    max_snippets=max_snippets_per_hit,
                )
                if include_snippets
                else []
            )
            metadata: dict[str, Any] = {
                "retriever_used": "text",
                "semantic_backend": self._backend,
            }
            if snippets:
                metadata["snippets"] = snippets
            hits.append(
                SearchHit(
                    rank=len(hits) + 1,
                    score=numeric_score,
                    node=node,
                    metadata=metadata,
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def _rank_candidates(
        self,
        candidate_indices: list[int],
        query: str,
    ) -> list[tuple[int, float]]:
        query_embedding = self.embedder.encode_query(query)
        if query_embedding.size == 0:
            return []
        scores: list[tuple[int, float]] = []
        for index in candidate_indices:
            node = self._nodes[index]
            embedding = self._node_embeddings.get(node.node_id)
            if embedding is None:
                continue
            scores.append((index, self.embedder.score(query_embedding, embedding)))
        return sorted(
            scores,
            key=lambda item: (-float(item[1]), self._nodes[item[0]].page_index, self._nodes[item[0]].node_id),
        )


def _resolve_torch_device(torch: Any, device: str) -> str:
    normalized = device.strip().lower()
    if normalized in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("gpu:"):
        return f"cuda:{normalized.split(':', 1)[1]}"
    return normalized


def _encode_unique_text_embeddings(
    embedder: TextEmbedder,
    texts: list[str],
) -> list[np.ndarray]:
    unique_texts: list[str] = []
    for text in texts:
        if text not in unique_texts:
            unique_texts.append(text)
    embeddings = embedder.encode_documents(unique_texts)
    by_text = {text: embedding for text, embedding in zip(unique_texts, embeddings, strict=False)}
    return [by_text[text] for text in texts]


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
