from __future__ import annotations

import json

from app.service import RagService


def test_ingest_builds_consistent_index(settings, service: RagService):
    manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    assert manifest["documents"] == 2
    assert manifest["chunks"] > 0
    assert manifest["dim"] > 0
    assert settings.vectors_path.exists()


def test_stats_reports_index(service: RagService):
    stats = service.stats()
    assert stats.indexed is True
    assert stats.documents == 2
    assert stats.chunks > 0
    assert len(stats.doc_list) == 2


def test_retrieval_returns_ranked_chunks(service: RagService):
    results, trace = service.engine.pipeline.retrieve("How many annual leave days?", top_k=4)
    assert results
    assert results[0].rerank_score is not None
    # ranks are 1-based when present
    assert results[0].dense_rank is None or results[0].dense_rank >= 1
    assert any(stage.name == "fusion" for stage in trace)


def test_answer_cites_the_right_document(service: RagService):
    response = service.ask("How many days of annual leave do employees receive?")
    assert "25" in response.answer
    assert response.citations
    assert response.citations[0].source_path == "leave_policy.txt"
    assert response.citations[0].page_number == 1


def test_answer_cites_api_guide_for_api_question(service: RagService):
    response = service.ask("What HTTP status code indicates a rate limit?")
    assert "429" in response.answer
    assert any(c.source_path == "api_guide.md" for c in response.citations)


def test_unrelated_question_still_returns_sources_or_refuses(service: RagService):
    response = service.ask("What is the capital of France?")
    # Either an explicit refusal, or an answer that is not falsely grounded.
    assert response.answer.strip()
    if "don't have enough information" in response.answer.lower():
        assert response.citations == []
