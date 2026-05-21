"""Rerank wrapper — compose a ``RetrievalPort`` with a ``RerankerPort``.

Per ADR-0004, the input contract is a ``RetrievalPort`` returning hits with
``source="hybrid"`` (BM25 + dense fused via RRF). Per ADR-0007, the local
reranker is a cross-encoder (BGE-v2-m3) that re-scores the top-``candidate_top_k``
hits before they reach the generator.

``RerankRetriever`` itself satisfies ``RetrievalPort`` so Phase 7's ``/query``
endpoint consumes a single port whether or not reranking is enabled, and a
managed-API swap (Cohere Rerank) in Phase 9 replaces the local cross-encoder
without domain changes. Per ADR-0002 this module imports only from ``ports/``
and the standard library.
"""

from __future__ import annotations

import logging

from uae_rag.ports.reranker import RerankerPort
from uae_rag.ports.retrieval import RetrievalHit, RetrievalPort

logger = logging.getLogger(__name__)


class RerankRetriever:
    """Wraps an inner ``RetrievalPort`` with a cross-encoder ``RerankerPort``.

    The inner retriever produces ``candidate_top_k`` candidates; the reranker
    re-scores them and returns the top ``top_k``. If the reranker raises, the
    wrapper logs and falls back to the inner retriever's hits truncated to
    ``top_k`` — ``/query`` always returns *something* for the user.
    """

    def __init__(
        self,
        *,
        retriever: RetrievalPort,
        reranker: RerankerPort,
        candidate_top_k: int = 20,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._candidate_top_k = candidate_top_k

    def retrieve(self, query: str, *, top_k: int = 10) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be ≥ 1")
        if self._candidate_top_k < top_k:
            logger.debug(
                "candidate_top_k (%d) < top_k (%d); reranked list capped at candidate_top_k",
                self._candidate_top_k,
                top_k,
            )

        candidates = self._retriever.retrieve(query, top_k=self._candidate_top_k)
        if not candidates:
            return []

        try:
            return self._reranker.rerank(query, candidates, top_k=top_k)
        except Exception:
            logger.exception("reranker raised; returning unranked candidates")
            return list(candidates)[:top_k]


__all__ = ["RerankRetriever"]
