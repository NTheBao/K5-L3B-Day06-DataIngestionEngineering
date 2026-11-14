"""TODO: store and search vectors using Chroma."""
from typing import Any
from .models import Chunk


class ChromaStore:
    def __init__(self, path: str, collection_name: str, dimensions: int) -> None:
        """TODO: initialize a persistent client and cosine collection; validate dimensions."""
        raise NotImplementedError("TODO: initialize store")

    def create_schema(self) -> None:
        """TODO: create or get the collection with cosine distance."""
        raise NotImplementedError("TODO: create_schema")

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """TODO: validate lengths/dimensions and upsert stable IDs, documents and metadata.

        When re-ingesting changed source documents, remove obsolete chunks too.
        """
        raise NotImplementedError("TODO: upsert")

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """TODO: apply metadata filters and return top-k by cosine distance.

        Return chunk ID, text, score and metadata.
        """
        raise NotImplementedError("TODO: search")
