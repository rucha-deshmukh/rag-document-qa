# rag-document-qa

Production-ready **RAG (Retrieval-Augmented Generation) document Q&A** service. Upload documents, then ask questions and get answers grounded in your documents **with citations**.

## Features

- **Document ingestion** — PDF / TXT / Markdown, chunked with overlap and embedded.
- **Vector retrieval** — semantic search over chunks using Postgres + `pgvector`.
- **Grounded answers with citations** — the LLM answers only from retrieved context and returns the source chunks it used.
- **Production concerns baked in** — request validation, structured logging, timeouts, retries with backoff on LLM/embedding calls, health checks, Docker, and CI.

## Architecture

```
            ┌────────────┐     ┌──────────────┐     ┌─────────────┐
  Upload →  │  Ingest    │ →   │  Chunk +     │ →   │  pgvector   │
            │  endpoint  │     │  Embed       │     │  store      │
            └────────────┘     └──────────────┘     └─────────────┘
                                                          │
            ┌────────────┐     ┌──────────────┐     ┌─────▼───────┐
  Ask   →   │  Chat      │ →   │  Retrieve    │ ←   │  similarity │
            │  endpoint  │     │  top-k       │     │  search     │
            └─────┬──────┘     └──────┬───────┘     └─────────────┘
                  │                   │
                  │            ┌──────▼───────┐
                  └──────────→ │  LLM answer  │  → answer + citations
                               │  (grounded)  │
                               └──────────────┘
```

## Tech stack

- **API**: FastAPI + Uvicorn
- **Vector store**: PostgreSQL + `pgvector`
- **LLM / embeddings**: Anthropic Claude (answers) — pluggable client
- **Tests**: pytest
- **CI**: GitHub Actions
- **Packaging**: Docker + docker-compose

## Quickstart

```bash
# 1. Configure
cp .env.example .env      # then fill in ANTHROPIC_API_KEY etc.

# 2. Run Postgres + the API
docker compose up --build

# API is now on http://localhost:8000  (docs at /docs)
```

Or run locally without Docker:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

## Usage

```bash
# Ingest a document
curl -X POST http://localhost:8000/ingest \
  -F "file=@handbook.pdf"

# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

Response:

```json
{
  "answer": "Refunds are available within 30 days ...",
  "citations": [
    {"document": "handbook.pdf", "chunk_id": 12, "score": 0.83}
  ]
}
```

## Configuration

All config is via environment variables — see [`.env.example`](.env.example).

| Variable | Description | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | API key for the LLM | _required_ |
| `DATABASE_URL` | Postgres connection string | `postgresql://postgres:postgres@localhost:5432/rag` |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` |
| `CHAT_MODEL` | Chat/answer model | `claude-sonnet-5` |
| `TOP_K` | Chunks retrieved per query | `4` |
| `CHUNK_SIZE` | Chunk size in characters | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `150` |

## Development

```bash
pip install -e ".[dev]"
pytest              # run tests
ruff check .        # lint
```

## Roadmap

- [ ] Streaming answers (SSE)
- [ ] Per-document access control / multi-tenant namespaces
- [ ] Re-ranking retrieved chunks
- [ ] Web UI

## License

MIT — see [LICENSE](LICENSE).
