from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.service import RagService


@pytest.fixture()
def client(service: RagService, monkeypatch):
    monkeypatch.setattr(main_module, "get_service", lambda: service)
    return TestClient(main_module.app)


def test_health_reports_ready(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["index_ready"] is True
    assert body["provider"] == "extractive"


def test_stats_lists_documents(client):
    response = client.get("/api/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["indexed"] is True
    assert body["documents"] == 2


def test_ask_endpoint_returns_citations(client):
    response = client.post(
        "/api/ask",
        json={"question": "How many days of annual leave do employees receive?", "top_k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert "25" in body["answer"]
    assert body["citations"]
    assert body["citations"][0]["page_number"] == 1
    assert body["trace"]


def test_ask_rejects_empty_question(client):
    response = client.post("/api/ask", json={"question": ""})
    assert response.status_code == 422


def test_metrics_endpoint_exposes_counters(client):
    client.post("/api/ask", json={"question": "What is the rate limit?"})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "rag_requests_total" in response.text


def test_ingest_endpoint_rebuilds(client):
    response = client.post("/api/ingest")
    assert response.status_code == 200
    body = response.json()
    assert body["documents"] == 2
    assert body["chunks"] > 0
