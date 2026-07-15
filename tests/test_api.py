"""End-to-end API tests using FastAPI's test client."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ingest_then_chat():
    files = {"file": ("notes.txt", b"The refund window is 30 days from purchase.", "text/plain")}
    resp = client.post("/ingest", files=files)
    assert resp.status_code == 200
    assert resp.json()["chunks_ingested"] >= 1

    resp = client.post("/chat", json={"question": "What is the refund window?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert len(body["citations"]) >= 1


def test_chat_with_no_documents_and_empty_question():
    resp = client.post("/chat", json={"question": ""})
    assert resp.status_code == 422  # validation error
