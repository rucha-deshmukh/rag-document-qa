"""LLM and embedding client with timeouts and retry/backoff.

The embedding function here is intentionally provider-agnostic and easy to
swap. Network calls are wrapped with :mod:`tenacity` so transient failures
(rate limits, timeouts) are retried with exponential backoff before the
request is allowed to fail.
"""

from __future__ import annotations

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


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(4),
    reraise=True,
)
def embed(text: str) -> list[float]:
    """Return an embedding vector for ``text``.

    Replace the body with a real embedding provider call. Kept as a
    deterministic local hash embedding so the app and tests run without any
    external dependency; swap in your provider before production use.
    """
    import hashlib

    settings = get_settings()
    dim = 1536
    digest = hashlib.sha256((settings.embedding_model + text).encode()).digest()
    # Expand the digest deterministically to the target dimensionality.
    raw = (digest * ((dim // len(digest)) + 1))[:dim]
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
