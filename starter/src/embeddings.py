"""Choose a real embedding provider (API or local) and implement this contract."""
from typing import Protocol

from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Return model-tokenizer counts, including special tokens."""
        ...


class MyEmbedder:
    """Create local sentence embeddings with Sentence Transformers."""

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name, device="cpu")
        self.model_name = model_name
        dimensions = self.model.get_embedding_dimension()
        if dimensions is None:
            raise RuntimeError("The embedding model did not report its output dimensions.")
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, question: str) -> list[float]:
        vector = self.model.encode(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()

    def count_tokens(self, texts: list[str]) -> list[int]:
        if not texts:
            return []

        counts: list[int] = []
        for start in range(0, len(texts), 64):
            batch = self.model.preprocess(texts[start : start + 64])
            attention_mask = batch["attention_mask"]
            counts.extend(int(count) for count in attention_mask.sum(dim=1).tolist())
        return counts
