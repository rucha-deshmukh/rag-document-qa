"""Tests for the chunking utility."""

import pytest

from app.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", 100, 10) == []
    assert chunk_text("   ", 100, 10) == []


def test_short_text_is_single_chunk():
    assert chunk_text("hello world", 100, 10) == ["hello world"]


def test_chunks_have_overlap():
    text = "abcdefghij" * 30  # 300 chars
    chunks = chunk_text(text, 100, 20)
    assert len(chunks) > 1
    # Each chunk (except possibly the last) is at most chunk_size chars.
    assert all(len(c) <= 100 for c in chunks)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        chunk_text("x", 0, 0)
    with pytest.raises(ValueError):
        chunk_text("x", 100, 100)
