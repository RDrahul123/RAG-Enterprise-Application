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


def test_extractive_provider_matches_plural_forms():
    """Regression: 'password' must match 'passwords' in the no-LLM baseline."""
    from app.generation.rag import extractive_answer
    from app.models import Chunk, ScoredChunk

    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        doc_title="Security Policy",
        source_path="security.pdf",
        page_number=2,
        position=0,
        text="Passwords must be at least 14 characters and are checked against a breach corpus.",
        token_count=14,
    )
    answer = extractive_answer("What is the minimum password length?", [ScoredChunk(chunk=chunk, score=1.0)])
    assert "14 characters" in answer
    assert "[1]" in answer


def test_grounding_score_treats_trailing_citation_as_part_of_sentence():
    from app.generation.rag import grounding_score

    assert grounding_score("Parental leave is 16 weeks fully paid. [1]", 3) == 1.0
    assert grounding_score("This sentence has no citation at all in it.", 3) == 0.0
