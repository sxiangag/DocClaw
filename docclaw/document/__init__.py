"""Document loading abstractions and implementations."""

from docclaw.document.base import DocumentLoader
from docclaw.document.image import PillowDocumentLoader
from docclaw.document.pdf import PdfDocumentLoader
from docclaw.document.registry import (
    DocumentLoaderRegistry,
    build_default_registry,
    load_document,
)

__all__ = [
    "DocumentLoader",
    "DocumentLoaderRegistry",
    "PillowDocumentLoader",
    "PdfDocumentLoader",
    "build_default_registry",
    "load_document",
]
