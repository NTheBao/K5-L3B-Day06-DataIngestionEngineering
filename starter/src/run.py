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


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build local A/B/C vector indexes.")
    parser.add_argument("--pdf", type=Path, default=Path("data/HUONG+DAN+LIEN+KET+THE+AGRIBANK+VOI+ZALOPAY.pdf"))
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmark.json"))
    parser.add_argument("--results", type=Path, default=Path("results"))
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
    question_counts = Counter(question["split"] for question in benchmark["questions"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
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

    args.results.mkdir(parents=True, exist_ok=True)
    report_path = args.results / f"pipeline-{run_id}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Pipeline complete: {len(content_pages)} content pages, {len(chunks_a)} A chunks, {len(chunks_b)} B/C chunks.")
    print(f"Chroma collection: {args.collection} at {chroma_path}")
    print(f"Report: {report_path}")
    if pages_without_text:
        print(f"Pages without extractable text: {pages_without_text}")


if __name__ == "__main__":
    main()
