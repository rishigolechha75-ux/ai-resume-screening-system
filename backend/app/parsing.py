"""Extract plain text from uploaded resume files (PDF, DOCX, TXT)."""

from __future__ import annotations

import io
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class UnsupportedFileError(ValueError):
    """Raised when a file type cannot be parsed."""


def extract_text(filename: str, content: bytes) -> str:
    """Return the text of a resume file given its name and raw bytes."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _pdf_text(content)
    if suffix == ".docx":
        return _docx_text(content)
    if suffix == ".txt":
        return content.decode("utf-8", errors="ignore")
    raise UnsupportedFileError(
        f"Unsupported file type '{suffix or filename}'. Use one of: "
        + ", ".join(sorted(SUPPORTED_EXTENSIONS))
    )


def _pdf_text(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _docx_text(content: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(content))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)
