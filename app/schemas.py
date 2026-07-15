"""Pydantic request/response models."""

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document: str
    chunks_ingested: int


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class Citation(BaseModel):
    document: str
    chunk_id: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
