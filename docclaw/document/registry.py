"""Loader registry for document ingestion."""

from __future__ import annotations

from pathlib import Path

from docclaw.agent.utils import DocumentState
from docclaw.document.base import DocumentLoader
from docclaw.document.image import IMAGE_SUFFIXES, PillowDocumentLoader
from docclaw.document.pdf import PDF_SUFFIXES, PdfDocumentLoader


class DocumentLoaderRegistry:
    """Map input formats to document loaders."""

    def __init__(self) -> None:
        self._loaders: dict[str, DocumentLoader] = {}

    def register_suffixes(
        self,
        suffixes: set[str] | list[str] | tuple[str, ...],
        loader: DocumentLoader,
    ) -> None:
        for suffix in suffixes:
            self._loaders[_normalize_suffix(suffix)] = loader

    def get_for_path(self, path: str | Path) -> DocumentLoader | None:
        suffix = _normalize_suffix(Path(path).suffix)
        return self._loaders.get(suffix)

    def require_for_path(self, path: str | Path) -> DocumentLoader:
        loader = self.get_for_path(path)
        if loader is None:
            suffix = Path(path).suffix or "<none>"
            raise ValueError(f"unsupported document format: {suffix}")
        return loader


def build_default_registry() -> DocumentLoaderRegistry:
    registry = DocumentLoaderRegistry()
    registry.register_suffixes(IMAGE_SUFFIXES, PillowDocumentLoader())
    registry.register_suffixes(PDF_SUFFIXES, PdfDocumentLoader())
    return registry


def load_document(
    path: str | Path,
    *,
    artifact_dir: str | Path | None = None,
    document_id: str | None = None,
    loader: DocumentLoader | None = None,
    registry: DocumentLoaderRegistry | None = None,
) -> DocumentState:
    resolved_loader = loader
    if resolved_loader is None:
        resolved_loader = (registry or build_default_registry()).require_for_path(path)
    return resolved_loader.load(
        path,
        artifact_dir=artifact_dir,
        document_id=document_id,
    )


def _normalize_suffix(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        return normalized
    if not normalized.startswith("."):
        normalized = "." + normalized
    return normalized
