"""Render document pages to images for the ground-truth model."""

from __future__ import annotations

import base64
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".jfif"}


class RenderError(Exception):
    """Raised when a document's pages cannot be rendered."""


def page_count(path: Path) -> int:
    """Number of pages in a document. Images are one page."""
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return 1
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RenderError("PyMuPDF is required to read PDF pages. Install it with: uv sync --extra runners") from e
    with fitz.open(path) as doc:
        return int(doc.page_count)


def render_pages(path: Path, dpi: int = 150, max_pages: int | None = None) -> list[bytes]:
    """Render a document's pages to PNG bytes.

    :param path: Document path (PDF or image).
    :param dpi: Render resolution.
    :param max_pages: Stop after this many pages.
    :raises RenderError: If the document cannot be rendered.
    """
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path.read_bytes()]

    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RenderError("PyMuPDF is required to render PDF pages. Install it with: uv sync --extra runners") from e

    images: list[bytes] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        with fitz.open(path) as doc:
            limit = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
            for index in range(limit):
                pixmap = doc.load_page(index).get_pixmap(matrix=matrix)
                images.append(pixmap.tobytes("png"))
    except Exception as e:
        raise RenderError(f"Failed to render {path}: {e}") from e
    return images


def truncate_pdf(source: Path, dest: Path, max_pages: int) -> bool:
    """Write the first *max_pages* pages of *source* to *dest*.

    A truncated copy keeps the evaluated document and the ground truth covering
    the same pages. Without it, a parser would be penalised for faithfully
    transcribing pages the ground truth never described.

    :return: True if a truncated copy was written, False if PyMuPDF is missing.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return False
    with fitz.open(source) as doc:
        out = fitz.open()
        out.insert_pdf(doc, from_page=0, to_page=min(max_pages, doc.page_count) - 1)
        out.save(dest)
        out.close()
    return True


def to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    """Encode image bytes as a data URL for an OpenAI-compatible vision call."""
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
