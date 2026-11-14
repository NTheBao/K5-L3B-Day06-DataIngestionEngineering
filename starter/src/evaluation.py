"""Evaluate source-and-evidence retrieval and aggregate benchmark metrics."""
import unicodedata
from collections import defaultdict
from typing import Any


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _gold_evidence(question: dict[str, Any]) -> list[dict[str, Any]]:
    alternatives = question.get("gold_evidence")
    if alternatives:
        return alternatives

    pages = question.get("gold_pages", question.get("gold_page"))
    if not isinstance(pages, list):
        pages = [pages]
    return [
        {
            "source_id": question["gold_source_id"],
            "page": page,
            "evidence_quote": question["evidence_quote"],
        }
        for page in pages
    ]


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    gold = _gold_evidence(question)
    evaluated_hits: list[dict[str, Any]] = []

    for hit in hits:
        metadata = hit.get("metadata") or {}
        normalized_text = _normalize(hit.get("text", ""))
        source_page_match = any(
            metadata.get("document_id") == evidence["source_id"]
            and metadata.get("page") == evidence["page"]
            for evidence in gold
        )
        exact_evidence_match = any(
            metadata.get("document_id") == evidence["source_id"]
            and metadata.get("page") == evidence["page"]
            and _normalize(evidence["evidence_quote"]) in normalized_text
            for evidence in gold
            if evidence.get("evidence_quote")
        )
        evaluated_hits.append(
            {
                "chunk_id": hit["chunk_id"],
                "text": hit["text"],
                "score": hit["score"],
                "distance": hit.get("distance"),
                "metadata": metadata,
                "gold_source_page_match": source_page_match,
                "exact_evidence_match": exact_evidence_match,
            }
        )

    hit_at_1 = bool(evaluated_hits and evaluated_hits[0]["exact_evidence_match"])
    evidence_hit_at_3 = any(hit["exact_evidence_match"] for hit in evaluated_hits[:3])
    return {
        "question_id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold_evidence": gold,
        "filter": question.get("filter"),
        "retrieved_chunks": evaluated_hits,
        "hit_at_1": hit_at_1,
        "evidence_hit_at_3": evidence_hit_at_3,
        "equivalent_evidence_review_required": not evidence_hit_at_3
        and any(hit["gold_source_page_match"] for hit in evaluated_hits[:3]),
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[(trace["strategy"], trace["split"])].append(trace)

    summary: dict[str, Any] = {}
    for (strategy, split), rows in sorted(grouped.items()):
        total = len(rows)
        hit1_count = sum(bool(row["hit_at_1"]) for row in rows)
        hit3_count = sum(bool(row["evidence_hit_at_3"]) for row in rows)
        summary.setdefault(strategy, {})[split] = {
            "question_count": total,
            "hit_at_1": {
                "hits": hit1_count,
                "total": total,
                "rate": hit1_count / total if total else 0.0,
            },
            "evidence_hit_at_3": {
                "hits": hit3_count,
                "total": total,
                "rate": hit3_count / total if total else 0.0,
            },
            "exact_evidence_miss_ids": [
                row["question_id"] for row in rows if not row["evidence_hit_at_3"]
            ],
        }
    return summary
