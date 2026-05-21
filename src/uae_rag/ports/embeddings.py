"""Port: text embedding model.

The domain layer (ingestion, retrieval, generation) depends only on this
protocol. Concrete adapters live under ``uae_rag.adapters.*`` and are wired
through ``uae_rag.config``. Per ADR-0002, no domain module imports an adapter
directly.

The protocol also exposes ``count_tokens`` because the tokenizer is intrinsic
to the embedder — coupling them here avoids inventing a third TokenizerPort
whose only consumer would be the chunker.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingsPort(Protocol):
    """Interface for embedding text into dense vectors.

    Attributes:
        model_id: Provider-qualified model identifier (e.g. ``"intfloat/multilingual-e5-large"``).
        dimension: Output vector length.
        passage_prefix: String prepended to documents before encoding (E5 convention; ``""`` elsewhere).
        query_prefix: String prepended to queries before encoding (E5 convention; ``""`` elsewhere).
    """

    model_id: str
    dimension: int
    passage_prefix: str
    query_prefix: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one L2-normalized vector per input text."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Return one L2-normalized vector for a single query."""
        ...

    def count_tokens(self, text: str) -> int:
        """Return the token count this embedder would see for ``text`` (no special tokens)."""
        ...


__all__ = ["EmbeddingsPort"]
