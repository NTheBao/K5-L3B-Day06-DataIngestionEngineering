import unittest

from src.models import DocumentPage
from src.run import _metadata_audit


class MetadataAuditTests(unittest.TestCase):
    def test_filter_compliance_is_only_applicable_to_strategy_c(self) -> None:
        page = DocumentPage(
            document_id="doc-a",
            source="guide.pdf",
            page=1,
            text="Evidence",
            metadata={"section": "Section 1"},
        )
        traces = [
            {
                "strategy": strategy,
                "split": "test",
                "question_id": "q1",
                "filter": {"section": "Section 1"},
                "retrieved_chunks": [
                    {
                        "chunk_id": strategy,
                        "metadata": {
                            "document_id": "doc-a",
                            "source": "guide.pdf",
                            "page": 1,
                            "section": "Section 1",
                        },
                    }
                ],
            }
            for strategy in ("A", "B", "C")
        ]

        audit = _metadata_audit(traces, {1: page}, test_question_count=1)

        for strategy in ("A", "B"):
            filter_audit = audit[strategy]["test_filter_compliance"]
            self.assertFalse(filter_audit["applicable"])
            self.assertIsNone(filter_audit["compliant_questions"])
            self.assertIsNone(filter_audit["rate"])
        self.assertEqual(
            audit["C"]["test_filter_compliance"],
            {
                "applicable": True,
                "compliant_questions": 1,
                "total_filtered_test_questions": 1,
                "failure_ids": [],
                "rate": 1.0,
            },
        )
