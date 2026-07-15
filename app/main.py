"""FastAPI application entrypoint."""

from __future__ import annotations

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import get_settings
from app.ingest import ingest_document
from app.llm import answer, embed
from app.logging_config import configure_logging
from app.schemas import ChatRequest, ChatResponse, Citation, IngestResponse
from app.store import InMemoryStore

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="rag-document-qa", version="0.1.0")

# A process-local store keeps the reference implementation dependency-free.
# Swap for PgVectorStore(get_settings().database_url) in production.
store = InMemoryStore()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        count = ingest_document(store, file.filename or "upload", data)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    logger.info("ingested document=%s chunks=%d", file.filename, count)
    return IngestResponse(document=file.filename or "upload", chunks_ingested=count)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()
    query_embedding = embed(request.question)
    matches = store.search(query_embedding, settings.top_k)
    if not matches:
        raise HTTPException(status_code=404, detail="No documents ingested yet")

    context = "\n\n".join(m.text for m in matches)
    text = answer(request.question, context)
    citations = [
        Citation(document=m.document, chunk_id=m.chunk_id, score=round(m.score, 4))
        for m in matches
    ]
    return ChatResponse(answer=text, citations=citations)
