import unittest

from src.evaluation import evaluate_query, summarize


class EvaluateQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.question = {
            "id": "q1",
            "split": "dev",
            "question": "When?",
            "expected_answer": "Tomorrow.",
            "gold_source_id": "doc-a",
            "gold_page": 4,
            "evidence_quote": "Tomorrow is the deadline.",
        }

    def test_exact_quote_matches_after_whitespace_normalization(self) -> None:
        trace = evaluate_query(
            self.question,
            [
                {
                    "chunk_id": "c1",
                    "text": "Tomorrow\n   is the deadline.",
                    "score": 0.9,
                    "metadata": {"document_id": "doc-a", "page": 4},
                }
            ],
        )

        self.assertTrue(trace["hit_at_1"])
        self.assertTrue(trace["evidence_hit_at_3"])

    def test_any_documented_gold_source_is_accepted(self) -> None:
        question = {
            **self.question,
            "gold_evidence": [
                {
                    "source_id": "doc-a",
                    "page": 4,
                    "evidence_quote": "Tomorrow is the deadline.",
                },
                {
                    "source_id": "doc-b",
                    "page": 8,
                    "evidence_quote": "The deadline is tomorrow.",
                },
            ],
        }
        trace = evaluate_query(
            question,
            [
                {
                    "chunk_id": "c1",
                    "text": "The deadline is tomorrow.",
                    "score": 0.7,
                    "metadata": {"document_id": "doc-b", "page": 8},
                }
            ],
        )

        self.assertTrue(trace["hit_at_1"])
        self.assertTrue(trace["evidence_hit_at_3"])

    def test_source_page_without_quote_requires_review_not_hit(self) -> None:
        trace = evaluate_query(
            self.question,
            [
                {
                    "chunk_id": "c1",
                    "text": "A date is mentioned.",
                    "score": 0.9,
                    "metadata": {"document_id": "doc-a", "page": 4},
                }
            ],
        )

        self.assertFalse(trace["hit_at_1"])
        self.assertFalse(trace["evidence_hit_at_3"])
        self.assertTrue(trace["equivalent_evidence_review_required"])


class SummarizeTests(unittest.TestCase):
    def test_reports_counts_and_denominators_per_split(self) -> None:
        summary = summarize(
            [
                {
                    "strategy": "A",
                    "split": "dev",
                    "question_id": "q1",
                    "hit_at_1": True,
                    "evidence_hit_at_3": True,
                },
                {
                    "strategy": "A",
                    "split": "test",
                    "question_id": "q2",
                    "hit_at_1": False,
                    "evidence_hit_at_3": True,
                },
            ]
        )

        self.assertEqual(summary["A"]["dev"]["hit_at_1"], {"hits": 1, "total": 1, "rate": 1.0})
        self.assertEqual(summary["A"]["test"]["evidence_hit_at_3"], {"hits": 1, "total": 1, "rate": 1.0})
