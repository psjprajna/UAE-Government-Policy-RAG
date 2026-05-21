"""Hybrid retriever — fuse BM25 and dense legs via Reciprocal Rank Fusion.

Per ADR-0004, RRF uses ``k=60`` by default and consumes the top-50 hits from
each leg. The fused result is ranked by descending RRF score; ties break on
ascending ``chunk_id`` so the order is fully deterministic. Per ADR-0002 this
module imports only from ``ports/`` and the standard library.
"""

from __future__ import annotations

import logging

from uae_rag.ports.retrieval import RetrievalHit, RetrievalPort

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Two-leg retriever (BM25 + dense) with Reciprocal Rank Fusion."""

    def __init__(
        self,
        *,
        bm25: RetrievalPort,
        dense: RetrievalPort,
        per_leg_top_k: int = 50,
        rrf_k: int = 60,
    ) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._per_leg_top_k = per_leg_top_k
        self._rrf_k = rrf_k

    def retrieve(self, query: str, *, top_k: int = 20) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be ≥ 1")
        if self._per_leg_top_k < top_k:
            logger.debug(
                "per_leg_top_k (%d) < top_k (%d); fused list capped at per_leg_top_k",
                self._per_leg_top_k,
                top_k,
            )

        # TODO(perf): Phase 8 measurements may justify parallel leg execution
        # via ThreadPoolExecutor; today the GIL contention exceeds the savings
        # on a ~285-chunk corpus, so leg calls run sequentially.
        bm25_hits = self._bm25.retrieve(query, top_k=self._per_leg_top_k)
        try:
            dense_hits = self._dense.retrieve(query, top_k=self._per_leg_top_k)
        except Exception:
            logger.exception("dense leg raised; falling back to BM25-only ranking")
            dense_hits = []

        return self._fuse(bm25_hits, dense_hits)[:top_k]

    def _fuse(
        self,
        bm25_hits: list[RetrievalHit],
        dense_hits: list[RetrievalHit],
    ) -> list[RetrievalHit]:
        """Sum RRF contributions per chunk_id and sort (score desc, chunk_id asc)."""
        scores: dict[str, float] = {}
        # First-seen-wins: BM25 hits seed text/metadata, dense fills any gaps.
        carrier: dict[str, RetrievalHit] = {}
        for hit in (*bm25_hits, *dense_hits):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (
                self._rrf_k + hit.rank
            )
            carrier.setdefault(hit.chunk_id, hit)

        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [
            RetrievalHit(
                chunk_id=cid,
                text=carrier[cid].text,
                metadata=dict(carrier[cid].metadata),
                score=score,
                rank=rank,
                source="hybrid",
            )
            for rank, (cid, score) in enumerate(ordered, start=1)
        ]


__all__ = ["HybridRetriever"]
