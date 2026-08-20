"""Base interfaces for document ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
from pathlib import Path

from docclaw.agent.utils import DocumentState


class DocumentLoader(ABC):
    """Load an input document into DocClaw's DocumentState."""

    @abstractmethod
    def load(
        self,
        path: str | Path,
        *,
        artifact_dir: str | Path | None = None,
        document_id: str | None = None,
    ) -> DocumentState:
        """Return a document state for the given input path."""


def default_document_id_from_path(path: Path) -> str:
    """Build a stable document id from a source path."""
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in path.stem).strip("._")
    safe_stem = stem or "document"
    return f"{safe_stem}_{digest}"
