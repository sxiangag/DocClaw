"""PDF-backed document loader implementations."""

from __future__ import annotations

from pathlib import Path

from docclaw.agent.utils import DocumentState, PageState
from docclaw.document.base import DocumentLoader, default_document_id_from_path

PDF_SUFFIXES = {".pdf"}


class PdfDocumentLoader(DocumentLoader):
    """Load PDF documents by rendering each page into a page-image artifact."""

    def __init__(self, *, dpi: int = 144) -> None:
        # TODO: Add hybrid PDF ingestion later: keep page-image rendering for
        # visual tools, but also extract native PDF text/metadata when useful.
        self.dpi = dpi

    def load(
        self,
        path: str | Path,
        *,
        artifact_dir: str | Path | None = None,
        document_id: str | None = None,
    ) -> DocumentState:
        source_path = Path(path).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"document not found: {source_path}")
        if not source_path.is_file():
            raise ValueError(f"document path is not a file: {source_path}")
        if source_path.suffix.lower() not in PDF_SUFFIXES:
            raise ValueError(f"unsupported document format: {source_path.suffix or '<none>'}")

        if artifact_dir is None:
            raise ValueError("artifact_dir is required for PDF inputs")

        artifact_root = Path(artifact_dir).expanduser().resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        page_dir = artifact_root / "pages"
        page_dir.mkdir(parents=True, exist_ok=True)
        resolved_document_id = document_id or default_document_id_from_path(source_path)

        import pymupdf

        pages: list[PageState] = []
        with pymupdf.open(source_path) as document:
            for page_index, page in enumerate(document):
                pixmap = page.get_pixmap(dpi=self.dpi, alpha=False)
                page_path = page_dir / f"page_{page_index:04d}.png"
                pixmap.save(page_path)
                pages.append(
                    PageState(
                        page_index=page_index,
                        width=pixmap.width,
                        height=pixmap.height,
                        image_path=str(page_path),
                    )
                )

        return DocumentState(
            document_id=resolved_document_id,
            pages=pages,
            metadata={
                "source_path": str(source_path),
                "source_format": "pdf",
                "page_dir": str(page_dir),
                "render_dpi": self.dpi,
            },
        )
