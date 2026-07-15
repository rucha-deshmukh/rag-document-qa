"""Document parsing and ingestion pipeline."""

from __future__ import annotations

import io

from app.chunking import chunk_text
from app.config import get_settings
from app.llm import embed
from app.store import VectorStore


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from an uploaded file based on its extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if lower.endswith((".txt", ".md", ".markdown")):
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {filename}")


def ingest_document(store: VectorStore, filename: str, data: bytes) -> int:
    """Parse, chunk, embed, and store a document. Returns chunks ingested."""
    settings = get_settings()
    text = extract_text(filename, data)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        return 0
    embeddings = [embed(c) for c in chunks]
    return store.add(filename, chunks, embeddings)
