"""Vector store backed by Postgres + pgvector.

Provides a thin, testable interface. The in-memory implementation is used by
tests and local runs without a database; :class:`PgVectorStore` is the
production path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Chunk:
    chunk_id: int
    document: str
    text: str
    embedding: list[float]


@dataclass
class Match:
    chunk_id: int
    document: str
    text: str
    score: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class VectorStore(Protocol):
    def add(self, document: str, chunks: list[str], embeddings: list[list[float]]) -> int: ...

    def search(self, embedding: list[float], top_k: int) -> list[Match]: ...


class InMemoryStore:
    """Simple in-memory vector store for tests and local development."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._next_id = 0

    def add(self, document: str, chunks: list[str], embeddings: list[list[float]]) -> int:
        for text, emb in zip(chunks, embeddings):
            self._chunks.append(Chunk(self._next_id, document, text, emb))
            self._next_id += 1
        return len(chunks)

    def search(self, embedding: list[float], top_k: int) -> list[Match]:
        scored = [
            Match(c.chunk_id, c.document, c.text, cosine_similarity(embedding, c.embedding))
            for c in self._chunks
        ]
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]
