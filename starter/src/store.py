"""Persistent Chroma storage and cosine-vector search."""
import math
from pathlib import Path
from typing import Any

import chromadb

from .models import Chunk


class ChromaStore:
    def __init__(
        self,
        path: str,
        collection_name: str,
        dimensions: int,
        model_name: str,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be greater than zero.")

        self.path = Path(path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.dimensions = dimensions
        self.model_name = model_name
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.create_schema()

    def create_schema(self) -> None:
        expected_metadata = {
            "hnsw:space": "cosine",
            "embedding_model": self.model_name,
            "embedding_dimensions": self.dimensions,
        }
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=expected_metadata,
        )
        actual_metadata = self.collection.metadata or {}
        for key, expected_value in expected_metadata.items():
            if actual_metadata.get(key) != expected_value:
                raise ValueError(
                    f"Existing collection {self.collection_name!r} has incompatible "
                    f"{key!r}: expected {expected_value!r}, got {actual_metadata.get(key)!r}."
                )

    def _storage_strategy(self, strategy: str) -> str:
        if strategy == "C":
            return "B"
        if strategy not in {"A", "B"}:
            raise ValueError(f"Unknown strategy {strategy!r}; expected A, B, or C.")
        return strategy

    def _where(
        self,
        strategy: str,
        filters: dict[str, Any] | None = None,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        conditions: list[dict[str, Any]] = [{"strategy": strategy}]
        if document_id is not None:
            conditions.append({"document_id": document_id})
        if filters:
            if "strategy" in filters or "document_id" in filters:
                raise ValueError("Filters cannot override strategy or document_id.")
            for key, value in filters.items():
                if not isinstance(value, (str, int, float, bool)):
                    raise TypeError(f"Chroma filter {key!r} must have a scalar value.")
                conditions.append({key: value})
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def upsert(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        storage_strategy = self._storage_strategy(strategy)
        if strategy == "C":
            raise ValueError("Strategy C reuses the B index; upsert B chunks instead.")
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding vector.")

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str | int | float | bool]] = []
        validated_vectors: list[list[float]] = []
        ids_by_document: dict[str, set[str]] = {}

        for chunk, vector in zip(chunks, vectors):
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Vector for chunk {chunk.chunk_id!r} has {len(vector)} dimensions; "
                    f"expected {self.dimensions}."
                )
            if not all(math.isfinite(float(value)) for value in vector):
                raise ValueError(f"Vector for chunk {chunk.chunk_id!r} contains non-finite values.")

            stored_id = f"{storage_strategy}:{chunk.chunk_id}"
            metadata: dict[str, str | int | float | bool] = {}
            for key, value in chunk.metadata.items():
                if value is None:
                    continue
                if not isinstance(value, (str, int, float, bool)):
                    raise TypeError(f"Chunk metadata {key!r} must be a scalar value.")
                metadata[key] = value
            metadata.update(
                {
                    "strategy": storage_strategy,
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "source": chunk.source,
                    "page": chunk.page,
                }
            )

            ids.append(stored_id)
            documents.append(chunk.text)
            metadatas.append(metadata)
            validated_vectors.append([float(value) for value in vector])
            ids_by_document.setdefault(chunk.document_id, set()).add(stored_id)

        existing_ids: dict[str, set[str]] = {}
        for document_id in ids_by_document:
            existing = self.collection.get(
                where=self._where(storage_strategy, document_id=document_id),
                include=["metadatas"],
            )
            existing_ids[document_id] = set(existing["ids"])

        if ids:
            self.collection.upsert(
                ids=ids,
                embeddings=validated_vectors,
                documents=documents,
                metadatas=metadatas,
            )

        for document_id, old_ids in existing_ids.items():
            stale_ids = old_ids - ids_by_document[document_id]
            if stale_ids:
                self.collection.delete(ids=sorted(stale_ids))

    def count(self, strategy: str) -> int:
        storage_strategy = self._storage_strategy(strategy)
        records = self.collection.get(
            where=self._where(storage_strategy),
            include=["metadatas"],
        )
        return len(records["ids"])

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        storage_strategy = self._storage_strategy(strategy)
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query vector has {len(query_vector)} dimensions; expected {self.dimensions}."
            )
        if not all(math.isfinite(float(value)) for value in query_vector):
            raise ValueError("Query vector contains non-finite values.")

        available = self.count(storage_strategy)
        if available == 0:
            return []

        results = self.collection.query(
            query_embeddings=[[float(value) for value in query_vector]],
            n_results=min(top_k, available),
            where=self._where(storage_strategy, filters=filters),
            include=["documents", "metadatas", "distances"],
        )
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {
                "chunk_id": ids[index],
                "text": documents[index] or "",
                "score": 1.0 - float(distances[index]),
                "distance": float(distances[index]),
                "metadata": metadatas[index] or {},
            }
            for index in range(len(ids))
        ]
