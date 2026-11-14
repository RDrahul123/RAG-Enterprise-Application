from __future__ import annotations

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["indexed"] is True


def test_stats_lists_documents():
    body = client.get("/api/stats").json()
    assert body["documents"] > 0
    assert body["chunks"] > 0


def test_ask_returns_citations():
    body = client.post(
        "/api/ask", json={"question": "What is the minimum password length requirement?"}
    ).json()
    assert "14" in body["answer"]
    assert body["citations"]
    assert body["sources"]


def test_ask_rejects_empty_question():
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_ui_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Enterprise RAG" in response.text
