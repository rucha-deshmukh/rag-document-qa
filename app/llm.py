"""LLM and embedding client with timeouts and retry/backoff.

Network calls are wrapped with :mod:`tenacity` so transient failures (rate
limits, timeouts) are retried with exponential backoff before the request is
allowed to fail. Both the embedding and the chat provider fall back to an
offline implementation when no API key is configured, so the app, tests, and
``docker compose up`` run with zero external dependencies during development.
"""

from __future__ import annotations

import hashlib
import logging

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings

logger = logging.getLogger(__name__)

_RETRYABLE = (TimeoutError, ConnectionError)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain the answer, say you "
    "don't know. Be concise."
)

# Dimensionality of the offline fallback embedding. OpenAI's
# text-embedding-3-small returns 1536 dims; keep the fallback consistent so a
# store populated in one mode stays comparable.
_FALLBACK_DIM = 1536


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(4),
    reraise=True,
)
def embed(text: str) -> list[float]:
    """Return an embedding vector for ``text``.

    Uses OpenAI embeddings when ``OPENAI_API_KEY`` is configured; otherwise
    falls back to a deterministic local hash embedding so development and tests
    run offline. Swap the provider block for your embedding vendor of choice
    (OpenAI and Voyage AI are both good options).
    """
    settings = get_settings()
    if not settings.openai_api_key:
        return _fallback_embed(settings.embedding_model, text)

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, timeout=30.0)
    response = client.embeddings.create(model=settings.embedding_model, input=text)
    return response.data[0].embedding


def _fallback_embed(model: str, text: str) -> list[float]:
    """Deterministic offline embedding derived from a hash of the input."""
    digest = hashlib.sha256((model + text).encode()).digest()
    raw = (digest * ((_FALLBACK_DIM // len(digest)) + 1))[:_FALLBACK_DIM]
    return [(b / 255.0) for b in raw]


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(4),
    reraise=True,
)
def answer(question: str, context: str) -> str:
    """Generate a grounded answer from ``context`` for ``question``.

    Uses the Anthropic client when an API key is configured; otherwise returns
    an extractive fallback so local/dev/test runs work offline.
    """
    settings = get_settings()
    if not settings.anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY not set; returning extractive fallback answer")
        return _extractive_fallback(context)

    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key, timeout=30.0)
    message = client.messages.create(
        model=settings.chat_model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def _extractive_fallback(context: str) -> str:
    """Return the first non-empty context line as a naive offline answer."""
    for line in context.splitlines():
        if line.strip():
            return line.strip()
    return "I don't know based on the provided documents."
