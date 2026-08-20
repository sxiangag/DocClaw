"""Visual internal search built around ColPali page retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from typing import Any, Protocol

from docclaw.agent.tool.internal_search.common import (
    build_snippets,
    matches_mode,
    matches_target,
)
from docclaw.agent.tool.internal_search.internal_search import SEARCH_MODES
from docclaw.retrieval.corpus import SearchCorpus, SearchHit, SearchNode


class ImageEmbedder(Protocol):
    """Protocol for page-visual retrievers."""

    def encode_documents(self, image_paths: list[str]) -> list[Any | None]:
        """Return one embedding per image path."""

    def encode_query(self, query: str) -> Any:
        """Return a query embedding."""

    def score(self, query_embedding: Any, document_embedding: Any) -> float:
        """Return a similarity score for one query/document pair."""


@dataclass(slots=True)
class ColPaliImageEmbedder:
    """ColPali visual page encoder for query-to-page-image retrieval."""

    model_name: str = "vidore/colpali-v1.3-hf"
    device: str = "auto"
    batch_size: int = 8
    _processor: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)
    _resolved_device: str | None = field(default=None, init=False, repr=False)

    def encode_documents(self, image_paths: list[str]) -> list[Any | None]:
        if not image_paths:
            return []
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None

        from PIL import Image

        embeddings: list[Any | None] = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            images = []
            valid_offsets: list[int] = []
            for offset, image_path in enumerate(batch_paths):
                path = Path(image_path).expanduser()
                if not path.exists():
                    continue
                images.append(Image.open(path).convert("RGB"))
                valid_offsets.append(offset)
            if not images:
                embeddings.extend([None] * len(batch_paths))
                continue
            try:
                inputs = _colpali_image_inputs(self._processor, images, self._model.device)
                with self._torch.inference_mode():
                    batch_output = self._model(**inputs)
                    batch_embeddings = getattr(batch_output, "embeddings", batch_output)
            finally:
                for image in images:
                    image.close()
            by_offset = {
                offset: embedding
                for offset, embedding in zip(valid_offsets, batch_embeddings, strict=False)
            }
            for offset in range(len(batch_paths)):
                embeddings.append(by_offset.get(offset))
        return embeddings

    def encode_query(self, query: str) -> Any:
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        inputs = _colpali_query_inputs(self._processor, query, self._model.device)
        with self._torch.inference_mode():
            output = self._model(**inputs)
            embeddings = getattr(output, "embeddings", output)
            return embeddings[0]

    def score(self, query_embedding: Any, document_embedding: Any) -> float:
        if document_embedding is None:
            return 0.0
        assert self._processor is not None
        scores = _colpali_scores(self._processor, query_embedding, document_embedding)
        value = scores[0][0] if hasattr(scores[0], "__getitem__") else scores[0]
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import ColPaliForRetrieval, ColPaliProcessor
        except ImportError as exc:
            raise RuntimeError(
                "ColPali image search requires `torch`, `transformers`, and `Pillow`."
            ) from exc

        self._torch = torch
        self._resolved_device = _resolve_torch_device(torch, self.device)
        torch_dtype = torch.bfloat16 if self._resolved_device.startswith("cuda") else torch.float32
        self._model = ColPaliForRetrieval.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
        ).to(self._resolved_device)
        self._model.eval()
        self._processor = ColPaliProcessor.from_pretrained(self.model_name)


@dataclass(slots=True)
class ImageSemanticRetriever:
    """Page-level visual retrieval with ColPali embeddings."""

    top_k_default: int = 5
    embedder: ImageEmbedder = field(default_factory=ColPaliImageEmbedder)
    min_score: float = 0.05
    _document_id: str | None = field(default=None, init=False, repr=False)
    _nodes: list[SearchNode] = field(default_factory=list, init=False, repr=False)
    _node_embeddings: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _node_image_signatures: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _backend: str = field(default="colpali", init=False, repr=False)

    def build(self, corpus: SearchCorpus) -> None:
        """Build or incrementally refresh the image index."""
        if self._document_id != corpus.document_id:
            self._document_id = corpus.document_id
            self._node_embeddings.clear()
            self._node_image_signatures.clear()

        self._nodes = list(corpus.nodes)
        current_node_ids = {node.node_id for node in self._nodes}
        stale_node_ids = set(self._node_embeddings) - current_node_ids
        for node_id in stale_node_ids:
            self._node_embeddings.pop(node_id, None)
            self._node_image_signatures.pop(node_id, None)

        nodes_to_encode: list[SearchNode] = []
        image_paths_to_encode: list[str] = []
        image_signatures: list[str] = []
        for node in self._nodes:
            image_path = _node_image_path(node)
            if image_path is None or node.node_type != "page":
                self._node_embeddings.pop(node.node_id, None)
                self._node_image_signatures.pop(node.node_id, None)
                continue
            image_signature = _image_signature(image_path)
            if image_signature is None:
                self._node_embeddings.pop(node.node_id, None)
                self._node_image_signatures.pop(node.node_id, None)
                continue
            if self._node_image_signatures.get(node.node_id) == image_signature:
                continue
            nodes_to_encode.append(node)
            image_paths_to_encode.append(image_path)
            image_signatures.append(image_signature)

        if not nodes_to_encode:
            return

        embeddings = _encode_unique_image_embeddings(self.embedder, image_paths_to_encode)
        for node, image_signature, embedding in zip(
            nodes_to_encode,
            image_signatures,
            embeddings,
            strict=False,
        ):
            if embedding is None:
                self._node_embeddings.pop(node.node_id, None)
                self._node_image_signatures.pop(node.node_id, None)
                continue
            self._node_embeddings[node.node_id] = embedding
            self._node_image_signatures[node.node_id] = image_signature
        self._backend = "colpali"

    def search(
        self,
        corpus: SearchCorpus,
        query: str,
        *,
        top_k: int | None = None,
        mode: str = "pages",
        page_index: int | None = None,
        page_indices: list[int] | None = None,
        region_ids: list[str] | None = None,
        include_snippets: bool = True,
        max_snippets_per_hit: int | None = None,
    ) -> list[SearchHit]:
        """Return ranked image-semantic hits for the given query."""
        if not query.strip():
            return []
        if mode not in SEARCH_MODES:
            raise ValueError(f"unsupported search mode: {mode}")
        if mode != "pages":
            raise ValueError("visual internal search only supports target.mode=pages")
        if region_ids is not None:
            raise ValueError("visual internal search does not support region_ids")

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
            if matches_mode(node, mode)
            and matches_target(node, page_filter)
            and node.node_id in self._node_embeddings
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
                if include_snippets and node.has_text
                else []
            )
            metadata: dict[str, Any] = {
                "retriever_used": "visual",
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


def _colpali_image_inputs(processor: Any, images: list[Any], device: Any) -> Any:
    if hasattr(processor, "process_images"):
        return processor.process_images(images).to(device)
    return processor(images=images, return_tensors="pt").to(device)


def _colpali_query_inputs(processor: Any, query: str, device: Any) -> Any:
    if hasattr(processor, "process_queries"):
        return processor.process_queries([query]).to(device)
    return processor(text=[query], return_tensors="pt").to(device)


def _colpali_scores(processor: Any, query_embedding: Any, document_embedding: Any) -> Any:
    if hasattr(processor, "score_multi_vector"):
        return processor.score_multi_vector([query_embedding], [document_embedding])
    return processor.score_retrieval(query_embedding.unsqueeze(0), document_embedding.unsqueeze(0))


def _node_image_path(node: SearchNode) -> str | None:
    value = node.image_path
    if not isinstance(value, str) or not value.strip():
        return None
    expanded = os.path.expanduser(value)
    return expanded if os.path.exists(expanded) else None


def _image_signature(image_path: str) -> str | None:
    path = Path(image_path).expanduser()
    if not path.exists():
        return None
    stat = path.stat()
    payload = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _encode_unique_image_embeddings(
    embedder: ImageEmbedder,
    image_paths: list[str],
) -> list[Any | None]:
    unique_paths: list[str] = []
    for path in image_paths:
        if path not in unique_paths:
            unique_paths.append(path)
    encoded = embedder.encode_documents(unique_paths)
    by_path = {path: embedding for path, embedding in zip(unique_paths, encoded, strict=False)}
    return [by_path.get(path) for path in image_paths]
