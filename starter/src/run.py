"""Ingest the Agribank–ZaloPay PDF into persistent A/B/C Chroma strategies."""
import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .models import Chunk, DocumentPage
from .store import ChromaStore

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "day06_local_embeddings"


def _read_benchmark(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    benchmark = json.loads(raw.decode("utf-8"))
    questions = benchmark.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"Benchmark {path} must contain a questions list.")

    splits = Counter(question.get("split") for question in questions)
    if len(questions) < 20 or splits["dev"] < 15 or splits["test"] < 5:
        raise ValueError("Benchmark must contain at least 20 questions: 15 dev and 5 test.")
    filtered = [question for question in questions if question.get("filter")]
    if len(filtered) < 3 or not any(question["split"] == "test" for question in filtered):
        raise ValueError("Benchmark needs at least 3 filtered questions, including one test question.")
    return benchmark, hashlib.sha256(raw).hexdigest()


def _content_pages(pages: list[DocumentPage]) -> list[DocumentPage]:
    return [
        page
        for page in pages
        if page.text.strip() and isinstance(page.metadata.get("section"), str)
    ]


def _chunk_statistics(chunks: list[Chunk], token_counts: list[int]) -> dict[str, Any]:
    lengths = [len(chunk.text) for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "chunk_characters_with_overlap": sum(lengths),
        "chunk_tokens_with_overlap": sum(token_counts),
        "token_count_definition": "SentenceTransformer attention-mask token count, including special tokens.",
        "chunk_length_chars": {
            "min": min(lengths) if lengths else 0,
            "mean": round(statistics.mean(lengths), 2) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
    }


def _page_issues(pages: list[DocumentPage]) -> tuple[list[int], list[int]]:
    pages_without_text: list[int] = []
    text_without_section: list[int] = []
    for page in pages:
        if not page.text:
            pages_without_text.append(page.page)
        elif not page.metadata.get("section"):
            text_without_section.append(page.page)
    return pages_without_text, text_without_section


def _index_strategy(
    store: ChromaStore,
    embedder: MyEmbedder,
    strategy: str,
    chunks: list[Chunk],
) -> dict[str, Any]:
    token_counts = embedder.count_tokens([chunk.text for chunk in chunks])
    stats = _chunk_statistics(chunks, token_counts)
    before_count = store.count(strategy)

    started = time.perf_counter()
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    embedding_seconds = time.perf_counter() - started

    started = time.perf_counter()
    store.upsert(strategy, chunks, vectors)
    storage_seconds = time.perf_counter() - started

    after_count = store.count(strategy)
    if after_count != len(chunks):
        raise RuntimeError(
            f"Strategy {strategy} indexed {after_count} chunks; expected {len(chunks)}."
        )

    stats.update(
        {
            "index_strategy": strategy,
            "existing_strategy_chunks_before": before_count,
            "indexed_strategy_chunks_after": after_count,
            "reingestion_has_no_duplicates": after_count == len(chunks),
            "embedding_seconds": round(embedding_seconds, 4),
            "storage_seconds": round(storage_seconds, 4),
            "cost_usd": 0.0,
        }
    )
    return stats


def _run_benchmark(
    benchmark: dict[str, Any],
    embedder: MyEmbedder,
    store: ChromaStore,
) -> tuple[list[dict[str, Any]], float, dict[str, float]]:
    traces: list[dict[str, Any]] = []
    query_embedding_seconds = 0.0
    search_seconds = {strategy: 0.0 for strategy in ("A", "B", "C")}
    for question in benchmark["questions"]:
        started = time.perf_counter()
        query_vector = embedder.embed_query(question["question"])
        query_embedding_seconds += time.perf_counter() - started

        for strategy in ("A", "B", "C"):
            filters = question.get("filter") if strategy == "C" else None
            started = time.perf_counter()
            hits = store.search(strategy, query_vector, top_k=3, filters=filters)
            elapsed = time.perf_counter() - started
            search_seconds[strategy] += elapsed
            trace = evaluate_query(question, hits)
            trace["strategy"] = strategy
            trace["search_seconds"] = round(elapsed, 6)
            traces.append(trace)
    return traces, query_embedding_seconds, search_seconds


def _metadata_audit(
    traces: list[dict[str, Any]],
    pages_by_number: dict[int, DocumentPage],
    test_question_count: int,
) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for strategy in ("A", "B", "C"):
        test_traces = [
            trace
            for trace in traces
            if trace["strategy"] == strategy and trace["split"] == "test"
        ]
        correct_records = 0
        errors: list[dict[str, Any]] = []
        filtered_questions = (
            [trace for trace in test_traces if trace.get("filter")]
            if strategy == "C"
            else []
        )
        compliant_questions = 0
        filter_failures: list[str] = []

        for trace in test_traces:
            for rank in range(3):
                if rank >= len(trace["retrieved_chunks"]):
                    errors.append(
                        {
                            "question_id": trace["question_id"],
                            "rank": rank + 1,
                            "errors": ["no result returned for top-3 slot"],
                        }
                    )
                    continue

                hit = trace["retrieved_chunks"][rank]
                metadata = hit["metadata"]
                page_number = metadata.get("page")
                source_page = pages_by_number.get(page_number) if isinstance(page_number, int) else None
                problems: list[str] = []
                if source_page is None or metadata.get("document_id") != source_page.document_id:
                    problems.append("document_id does not match source page")
                if source_page is None or metadata.get("source") != source_page.source:
                    problems.append("source/page is absent or does not match the PDF")
                if source_page is None or metadata.get("section") != source_page.metadata.get("section"):
                    problems.append("section metadata is absent or does not match the PDF heading")
                if not problems:
                    correct_records += 1
                else:
                    errors.append(
                        {
                            "question_id": trace["question_id"],
                            "rank": rank + 1,
                            "chunk_id": hit["chunk_id"],
                            "errors": problems,
                        }
                    )

            if strategy == "C" and trace.get("filter"):
                expected_section = trace["filter"].get("section")
                hits = trace["retrieved_chunks"]
                if hits and all(hit["metadata"].get("section") == expected_section for hit in hits):
                    compliant_questions += 1
                else:
                    filter_failures.append(trace["question_id"])

        expected_records = test_question_count * 3
        audit[strategy] = {
            "top3_metadata": {
                "correct_records": correct_records,
                "total_records": expected_records,
                "errors": errors,
            },
            "test_filter_compliance": {
                "applicable": strategy == "C",
                "compliant_questions": compliant_questions if strategy == "C" else None,
                "total_filtered_test_questions": len(filtered_questions),
                "failure_ids": filter_failures,
                "rate": (
                    compliant_questions / len(filtered_questions)
                    if filtered_questions
                    else None
                ),
            },
        }

    audit["interpretation"] = {
        "useful_field": "`section` supports a user-supplied scope restriction; `source` and `page` make results traceable.",
        "limitation": "A single PDF and seven sections limit filter diversity; section labels depend on extracted headings.",
    }
    return audit


def _manual_review_packet(
    traces: list[dict[str, Any]],
    chunks_by_strategy: dict[str, list[Chunk]],
    sample_pages: list[int],
) -> dict[str, Any]:
    coherence_items: list[dict[str, Any]] = []
    for page_number in sample_pages:
        for strategy in ("A", "B", "C"):
            storage_strategy = "B" if strategy == "C" else strategy
            matching_chunks = sorted(
                (
                    chunk
                    for chunk in chunks_by_strategy[storage_strategy]
                    if chunk.page == page_number
                ),
                key=lambda chunk: chunk.metadata["chunk_index"],
            )
            chosen_chunk = matching_chunks[0] if matching_chunks else None
            coherence_items.append(
                {
                    "strategy": strategy,
                    "page": page_number,
                    "chunk_id": chosen_chunk.chunk_id if chosen_chunk else None,
                    "text": chosen_chunk.text if chosen_chunk else None,
                    "score_0_to_2": None,
                    "reason": "",
                }
            )

    grounding_items = [
        {
            "strategy": trace["strategy"],
            "question_id": trace["question_id"],
            "question": trace["question"],
            "expected_answer": trace["expected_answer"],
            "gold_evidence": trace["gold_evidence"],
            "filter": trace["filter"],
            "retrieved_chunks": trace["retrieved_chunks"],
            "score_0_to_2": None,
            "reason": "",
        }
        for trace in traces
        if trace["split"] == "test"
    ]
    return {
        "status": "pending_author_review",
        "reviewer": None,
        "method": "Apply the GUIDE.md 0-2 rubric to the displayed source chunks; these labels are not inferred from Hit@3.",
        "coherence": {
            "sampling_policy": "The first ten content page IDs in ascending order, selected before retrieval results were produced.",
            "pages": sample_pages,
            "rubric": {
                "0": "Cut mid-idea or missing subject; not understandable.",
                "1": "Partly understandable but missing context.",
                "2": "Self-contained and preserves the complete idea.",
            },
            "items": coherence_items,
        },
        "grounding": {
            "rubric": {
                "0": "No evidence in top-3.",
                "1": "Evidence exists but required condition/context is missing.",
                "2": "Top-3 evidence is sufficient to answer and trace to source.",
            },
            "items": grounding_items,
        },
    }


def _impact_report(
    summary: dict[str, Any],
    strategy_results: dict[str, dict[str, Any]],
    search_seconds: dict[str, float],
    selection_policy: str,
) -> dict[str, Any]:
    def build_seconds(strategy: str) -> float:
        index_strategy = "B" if strategy == "C" else strategy
        result = strategy_results[index_strategy]
        return (
            result["chunking_seconds"]
            + result["embedding_seconds"]
            + result["storage_seconds"]
        )

    dev_score = lambda strategy: summary[strategy]["dev"]
    selected_strategy = max(
        ("A", "B", "C"),
        key=lambda strategy: (
            dev_score(strategy)["evidence_hit_at_3"]["hits"],
            dev_score(strategy)["hit_at_1"]["hits"],
            -build_seconds(strategy),
        ),
    )
    comparisons: dict[str, Any] = {}
    for left, right in (("A", "B"), ("B", "C")):
        split_deltas: dict[str, Any] = {}
        for split in ("dev", "test"):
            left_metrics = summary[left][split]
            right_metrics = summary[right][split]
            split_deltas[split] = {
                "evidence_hit_at_3_delta_hits": (
                    right_metrics["evidence_hit_at_3"]["hits"]
                    - left_metrics["evidence_hit_at_3"]["hits"]
                ),
                "hit_at_1_delta_hits": (
                    right_metrics["hit_at_1"]["hits"]
                    - left_metrics["hit_at_1"]["hits"]
                ),
            }
        comparisons[f"{left}_vs_{right}"] = {
            "split_metric_deltas": split_deltas,
            "left_chunk_count": strategy_results["B" if left == "C" else left]["chunk_count"],
            "right_chunk_count": strategy_results["B" if right == "C" else right]["chunk_count"],
            "left_index_build_seconds": round(build_seconds(left), 4),
            "right_index_build_seconds": round(build_seconds(right), 4),
            "left_query_search_seconds": round(search_seconds[left], 4),
            "right_query_search_seconds": round(search_seconds[right], 4),
            "right_shares_left_index": left == "B" and right == "C",
        }

    return {
        "selection_policy": selection_policy,
        "selected_on_dev_only": selected_strategy,
        "test_used_for_selection": False,
        "comparisons": comparisons,
        "exact_evidence_misses": {
            strategy: {
                split: summary[strategy][split]["exact_evidence_miss_ids"]
                for split in ("dev", "test")
            }
            for strategy in ("A", "B", "C")
        },
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build local A/B/C vector indexes.")
    parser.add_argument("--pdf", type=Path, default=Path("data/HUONG+DAN+LIEN+KET+THE+AGRIBANK+VOI+ZALOPAY.pdf"))
    parser.add_argument("--benchmark", type=Path, default=Path("../benchmark/benchmark.json"))
    parser.add_argument("--results", type=Path, default=Path("../results"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    args = parser.parse_args()

    model_name = args.model or os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
    chroma_path = Path(os.getenv("CHROMA_PATH", "./data/chroma"))
    if not args.pdf.is_file():
        raise FileNotFoundError(f"PDF source not found: {args.pdf}")
    if not args.benchmark.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {args.benchmark}")

    benchmark, benchmark_hash = _read_benchmark(args.benchmark)
    source_metadata = benchmark.get("source")
    if not isinstance(source_metadata, dict):
        raise ValueError("Benchmark must describe its source document.")
    document_id = source_metadata.get("source_id")
    source_title = source_metadata.get("title")
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("Benchmark source must have a non-empty source_id.")
    if not isinstance(source_title, str) or not source_title:
        raise ValueError("Benchmark source must have a non-empty title.")
    if any(question.get("gold_source_id") != document_id for question in benchmark["questions"]):
        raise ValueError("Every benchmark question must use the declared source_id.")

    run_started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    source = args.pdf.as_posix()
    pages = load_pdf(args.pdf, document_id, source)
    content_pages = _content_pages(pages)
    if len(content_pages) < 30:
        raise ValueError(
            f"Only {len(content_pages)} pages have extractable text and a detected section; "
            "the lab requires at least 30 content pages."
        )

    chunking_started = time.perf_counter()
    chunks_a_started = time.perf_counter()
    chunks_a = [
        chunk
        for page in content_pages
        for chunk in fixed_size(page, size=500, overlap=50)
    ]
    chunking_a_seconds = time.perf_counter() - chunks_a_started
    chunks_b_started = time.perf_counter()
    chunks_b = [
        chunk
        for page in content_pages
        for chunk in by_paragraph(page, max_size=300)
    ]
    chunking_b_seconds = time.perf_counter() - chunks_b_started
    chunking_seconds = time.perf_counter() - chunking_started
    if not chunks_a or not chunks_b:
        raise RuntimeError("Chunking produced no chunks; check the extracted PDF text.")

    model_started = time.perf_counter()
    embedder = MyEmbedder(model_name)
    model_initialization_seconds = time.perf_counter() - model_started
    store = ChromaStore(
        path=str(chroma_path),
        collection_name=args.collection,
        dimensions=embedder.dimensions,
        model_name=embedder.model_name,
    )

    strategy_results = {
        "A": _index_strategy(store, embedder, "A", chunks_a),
        "B": _index_strategy(store, embedder, "B", chunks_b),
    }
    strategy_results["A"].update(
        {
            "chunking": "fixed_size",
            "chunk_size_chars": 500,
            "overlap_chars": 50,
            "chunking_seconds": round(chunking_a_seconds, 4),
        }
    )
    strategy_results["B"].update(
        {
            "chunking": "logical_paragraph_and_step_blocks",
            "max_chunk_size_chars": 300,
            "chunking_seconds": round(chunking_b_seconds, 4),
        }
    )
    strategy_results["C"] = {
        **{
            key: value
            for key, value in strategy_results["B"].items()
            if key
            in {
                "chunk_count",
                "chunk_characters_with_overlap",
                "chunk_tokens_with_overlap",
                "token_count_definition",
                "chunk_length_chars",
                "chunking",
                "max_chunk_size_chars",
                "chunking_seconds",
            }
        },
        "index_strategy": "B (shared; identical chunks and vectors)",
        "shares_index_with": "B",
        "filter": "Apply the user's supplied section filter before retrieval; otherwise unfiltered.",
        "additional_embedding_seconds": 0.0,
        "additional_storage_seconds": 0.0,
        "cost_usd": 0.0,
    }

    selection_policy = benchmark.get("selection_policy")
    if not isinstance(selection_policy, str) or not selection_policy:
        raise ValueError("Benchmark must record a selection policy before evaluation.")
    coherence_sample_pages = [
        page.page for page in sorted(content_pages, key=lambda page: page.page)[:10]
    ]
    traces, query_embedding_seconds, search_seconds = _run_benchmark(
        benchmark,
        embedder,
        store,
    )
    if len(traces) != len(benchmark["questions"]) * 3:
        raise RuntimeError("The benchmark did not produce one trace per question and strategy.")
    retrieval_summary = summarize(traces)
    question_counts = Counter(question["split"] for question in benchmark["questions"])
    metadata_audit = _metadata_audit(
        traces,
        {page.page: page for page in pages},
        question_counts["test"],
    )
    manual_review = _manual_review_packet(
        traces,
        {"A": chunks_a, "B": chunks_b, "C": chunks_b},
        coherence_sample_pages,
    )
    strategy_impact = _impact_report(
        retrieval_summary,
        strategy_results,
        search_seconds,
        selection_policy,
    )

    section_counts = Counter(
        str(page.metadata["section"]) for page in content_pages
    )
    filter_values = sorted(
        {
            question["filter"]["section"]
            for question in benchmark["questions"]
            if question.get("filter") and "section" in question["filter"]
        }
    )
    unknown_filter_values = set(filter_values) - set(section_counts)
    if unknown_filter_values:
        raise ValueError(
            "Benchmark contains section filters absent from the source: "
            + ", ".join(sorted(unknown_filter_values))
        )
    pages_without_text, text_without_section = _page_issues(pages)
    source_digest = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    args.results.mkdir(parents=True, exist_ok=True)
    trace_path = args.results / f"traces-{run_id}.json"
    review_path = args.results / f"manual-review-{run_id}.json"
    trace_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "benchmark_sha256": benchmark_hash,
                "top_k": 3,
                "query_embedding_seconds": round(query_embedding_seconds, 4),
                "search_seconds_by_strategy": {
                    strategy: round(seconds, 4)
                    for strategy, seconds in search_seconds.items()
                },
                "summary": retrieval_summary,
                "traces": traces,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    review_path.write_text(
        json.dumps(manual_review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report: dict[str, Any] = {
        "run_id": run_id,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "document_id": document_id,
            "title": source_title,
            "path": source,
            "sha256": source_digest,
            "physical_pages": len(pages),
            "content_pages": len(content_pages),
            "source_characters": sum(len(page.text) for page in content_pages),
            "ocr_used": False,
            "rights_status": "Not verified; confirm permission before redistribution.",
        },
        "normalization": "Unicode NFKC; whitespace normalized while line and blank-line structure is retained.",
        "metadata": {
            "required_fields": ["document_id", "source", "page"],
            "filter_field": "section",
            "filter_field_type": "string",
            "section_values_detected": dict(section_counts),
            "filter_values_in_benchmark": filter_values,
            "section_value_source": "Section headings extracted from this PDF.",
        },
        "benchmark": {
            "path": args.benchmark.as_posix(),
            "sha256": benchmark_hash,
            "question_counts": dict(question_counts),
            "filtered_question_count": sum(bool(question.get("filter")) for question in benchmark["questions"]),
            "question_ids_by_split": {
                split: [question["id"] for question in benchmark["questions"] if question["split"] == split]
                for split in ("dev", "test")
            },
        },
        "embedding": {
            "provider": "Sentence Transformers (local)",
            "model": embedder.model_name,
            "package_version": version("sentence-transformers"),
            "dimensions": embedder.dimensions,
            "device": "cpu",
            "normalize_embeddings": True,
            "document_query_processing": "Same encode method and model; no query/document prefix.",
            "initialization_seconds": round(model_initialization_seconds, 4),
        },
        "vector_store": {
            "provider": "Chroma",
            "version": chromadb.__version__,
            "path": str(chroma_path),
            "collection": args.collection,
            "initialization": "chromadb.PersistentClient with get_or_create_collection",
            "metric": "cosine",
            "search": "Approximate nearest-neighbor using Chroma HNSW.",
        },
        "strategies": strategy_results,
        "retrieval": {
            "top_k": 3,
            "query_embedding_seconds": round(query_embedding_seconds, 4),
            "search_seconds_by_strategy": {
                strategy: round(seconds, 4)
                for strategy, seconds in search_seconds.items()
            },
            "summary": retrieval_summary,
            "trace_path": trace_path.as_posix(),
            "manual_review_path": review_path.as_posix(),
        },
        "metadata_audit": metadata_audit,
        "manual_review_status": manual_review["status"],
        "coherence_sample_pages": coherence_sample_pages,
        "strategy_impact": strategy_impact,
        "execution": {
            "command": "python -m src.run",
            "python_version": platform.python_version(),
            "arguments": sys.argv[1:],
        },
        "chunking_seconds_total": round(chunking_seconds, 4),
        "page_diagnostics": {
            "pages_without_text": pages_without_text,
            "text_pages_without_detected_section": text_without_section,
        },
        "total_run_seconds": round(time.perf_counter() - run_started, 4),
        "cost_usd": 0.0,
    }

    report_path = args.results / f"pipeline-{run_id}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Pipeline complete: {len(content_pages)} content pages, {len(chunks_a)} A chunks, {len(chunks_b)} B/C chunks.")
    print(f"Chroma collection: {args.collection} at {chroma_path}")
    print(f"Report: {report_path}")
    print(f"Top-3 traces: {trace_path}")
    print(f"Manual review sheet: {review_path}")
    for strategy in ("A", "B", "C"):
        split_metrics = retrieval_summary[strategy]
        print(
            f"{strategy}: dev Evidence Hit@3 "
            f"{split_metrics['dev']['evidence_hit_at_3']['hits']}/{split_metrics['dev']['question_count']}; "
            f"test {split_metrics['test']['evidence_hit_at_3']['hits']}/{split_metrics['test']['question_count']}"
        )
    if pages_without_text:
        print(f"Pages without extractable text: {pages_without_text}")


if __name__ == "__main__":
    main()
