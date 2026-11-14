from __future__ import annotations

import answer
import rag


def make_hits(*texts: str) -> list[rag.Hit]:
    return [rag.Hit(chunk=rag.Chunk(text=text, doc="Policy", page=2), score=1.0) for text in texts]


def test_parse_citations_keeps_only_valid_markers():
    hits = make_hits("first passage", "second passage")
    citations = answer.parse_citations("One fact [1]. Another fact [7].", hits)
    assert len(citations) == 1
    assert citations[0]["doc"] == "Policy"
    assert citations[0]["page"] == 2


def test_parse_citations_ignores_duplicates():
    hits = make_hits("a", "b")
    citations = answer.parse_citations("Fact [2] and again [2].", hits)
    assert len(citations) == 1


def test_grounding_score_counts_trailing_citation():
    assert answer.grounding_score("Parental leave is 16 weeks fully paid. [1]", 3) == 1.0


def test_grounding_score_without_citations_is_zero():
    assert answer.grounding_score("This sentence has no citation at all in it.", 3) == 0.0


def test_extractive_answer_matches_plural_forms():
    hits = make_hits(
        "Passwords must be at least 14 characters and are checked against a breach corpus."
    )
    text = answer.extractive_answer("What is the minimum password length?", hits)
    assert "14 characters" in text
    assert "[1]" in text


def test_answer_cites_the_correct_document(store):
    result = answer.answer(store, "How many days of annual leave do employees receive?")
    assert "24" in result["answer"]
    assert result["citations"]
    assert result["citations"][0]["doc"] == "Northwind Employee Handbook"
    assert result["citations"][0]["page"] == 2


def test_answer_refuses_questions_outside_the_corpus(store):
    result = answer.answer(store, "What is the boiling point of water at sea level?")
    assert result["answer"] == answer.REFUSAL
    assert result["citations"] == []
    assert result["grounded"] is False
