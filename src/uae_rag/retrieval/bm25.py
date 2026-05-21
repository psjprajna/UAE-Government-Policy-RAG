"""BM25 sparse retriever — local lexical leg of the hybrid pipeline.

Per ADR-0004, BM25 is one of the two retrieval legs that the hybrid retriever
fuses via RRF. ``rank_bm25.BM25Okapi`` does the scoring; this module owns the
tokenization (Arabic-aware) and the corpus lifecycle. The corpus is built fresh
in memory at construction; v1 does not persist the index.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from uae_rag.ingestion.chunker import Chunk
from uae_rag.ports.retrieval import RetrievalHit

logger = logging.getLogger(__name__)

# Arabic combining marks (harakat / diacritics). Stripping them on both sides
# of the index gives recall on queries that omit harakat — the common case.
_AR_DIACRITICS_RE = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")

# Unicode word tokens: ASCII word chars + Arabic block (U+0600..U+06FF).
_TOKEN_RE = re.compile(r"[\w؀-ۿ]+")


def tokenize(text: str) -> list[str]:
    """Case-fold, strip Arabic diacritics, split on Unicode word boundaries.

    Length-1 ASCII letters ("a", "I") are dropped as noise; digits are kept
    because article numbers ("1", "29") carry real signal in legal queries.
    """
    folded = _AR_DIACRITICS_RE.sub("", text.casefold())
    tokens = _TOKEN_RE.findall(folded)
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


def _chunk_metadata(chunk: Chunk) -> dict[str, str | int | None]:
    """Pack chunk fields into a JSON-scalar metadata dict (ChromaDB-compatible)."""
    return {
        "source_slug": chunk.source_slug,
        "breadcrumb": chunk.breadcrumb,
        "article_id": chunk.article_id,
        "language": chunk.language,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "mode": chunk.mode,
    }


class BM25Retriever:
    """Owns its corpus; rebuilds the index on construction."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25 corpus is empty")
        self._chunks: list[Chunk] = list(chunks)
        tokenized: list[list[str]] = [tokenize(c.text) for c in self._chunks]
        self._token_sets: list[frozenset[str]] = [frozenset(toks) for toks in tokenized]
        self._bm25 = BM25Okapi(tokenized)
        logger.debug("BM25 index built over %d chunks", len(self._chunks))

    def retrieve(self, query: str, *, top_k: int = 50) -> list[RetrievalHit]:
        """Score every chunk that shares at least one token with ``query``."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_set = frozenset(query_tokens)
        overlapping = [
            idx for idx, doc_set in enumerate(self._token_sets) if doc_set & query_set
        ]
        if not overlapping:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            ((float(scores[idx]), idx) for idx in overlapping),
            key=lambda t: (-t[0], self._chunks[t[1]].chunk_id),
        )

        hits: list[RetrievalHit] = []
        for rank, (score, idx) in enumerate(ranked[:top_k], start=1):
            chunk = self._chunks[idx]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=_chunk_metadata(chunk),
                    score=score,
                    rank=rank,
                    source="bm25",
                )
            )
        return hits


__all__ = ["BM25Retriever", "tokenize"]
