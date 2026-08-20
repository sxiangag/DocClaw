"""Image-backed document loader implementations."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageSequence

from docclaw.agent.utils import DocumentState, PageState
from docclaw.document.base import DocumentLoader, default_document_id_from_path

IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class PillowDocumentLoader(DocumentLoader):
    """Load raster documents with Pillow.

    Single-frame images reuse the original file path. Multi-frame images are
    expanded into per-page PNG files inside ``artifact_dir``.
    """

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
        suffix = source_path.suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported document format: {source_path.suffix or '<none>'}")

        resolved_document_id = document_id or default_document_id_from_path(source_path)
        artifact_root = (
            Path(artifact_dir).expanduser().resolve()
            if artifact_dir is not None
            else None
        )
        if artifact_root is not None:
            artifact_root.mkdir(parents=True, exist_ok=True)

        with Image.open(source_path) as image:
            frame_count = getattr(image, "n_frames", 1)
            if frame_count <= 1:
                return DocumentState(
                    document_id=resolved_document_id,
                    pages=[
                        PageState(
                            page_index=0,
                            width=image.width,
                            height=image.height,
                            image_path=str(source_path),
                        )
                    ],
                    metadata={
                        "source_path": str(source_path),
                        "source_format": suffix.lstrip("."),
                    },
                )

            if artifact_root is None:
                raise ValueError("artifact_dir is required for multi-page image inputs")

            page_dir = artifact_root / "pages"
            page_dir.mkdir(parents=True, exist_ok=True)
            pages: list[PageState] = []
            for page_index, frame in enumerate(ImageSequence.Iterator(image)):
                page_path = page_dir / f"page_{page_index:04d}.png"
                frame.convert("RGB").save(page_path)
                pages.append(
                    PageState(
                        page_index=page_index,
                        width=frame.width,
                        height=frame.height,
                        image_path=str(page_path),
                    )
                )

        return DocumentState(
            document_id=resolved_document_id,
            pages=pages,
            metadata={
                "source_path": str(source_path),
                "source_format": suffix.lstrip("."),
                "page_dir": str(page_dir),
            },
        )
