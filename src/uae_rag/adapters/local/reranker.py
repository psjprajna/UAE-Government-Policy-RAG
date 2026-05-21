"""Local reranker adapter — sentence-transformers CrossEncoder wrapping BGE-v2-m3.

Defaults to ``BAAI/bge-reranker-v2-m3`` (ADR-0007). Model id, revision SHA,
device, and max sequence length are read from env vars so a different
CrossEncoder model can be swapped in without editing code:

    LOCAL_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
    LOCAL_RERANKER_REVISION=<HF revision sha>
    LOCAL_RERANKER_DEVICE=cpu
    LOCAL_RERANKER_MAX_LENGTH=512

The underlying ``CrossEncoder`` is loaded lazily on the first ``rerank`` call
that has non-empty hits. Construction is cheap; an empty ``hits`` list does
not trigger a model load. Mirrors the lazy-load pattern in
``adapters/local/embeddings.SentenceTransformersEmbeddings``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING

from uae_rag.ports.retrieval import RetrievalHit

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
_DEFAULT_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
_DEFAULT_MAX_LENGTH = 8192


class SentenceTransformersReranker:
    """``RerankerPort`` implementation backed by ``sentence_transformers.CrossEncoder``.

    Attributes are populated from env vars at construction; the model is not
    loaded until the first ``rerank`` call with non-empty hits.
    """

    def __init__(self) -> None:
        self.model_id: str = os.environ.get("LOCAL_RERANKER_MODEL", _DEFAULT_MODEL)
        self.revision: str = os.environ.get("LOCAL_RERANKER_REVISION", _DEFAULT_REVISION)
        device_env = os.environ.get("LOCAL_RERANKER_DEVICE")
        # Empty string also defers to CrossEncoder's auto-detection.
        self.device: str | None = device_env if device_env else None
        self.max_length: int = int(
            os.environ.get("LOCAL_RERANKER_MAX_LENGTH", str(_DEFAULT_MAX_LENGTH))
        )

    @cached_property
    def _model(self) -> CrossEncoder:
        from sentence_transformers import CrossEncoder

        logger.info(
            "loading CrossEncoder model %s @ revision %s (device=%s, max_length=%d)",
            self.model_id,
            self.revision,
            self.device or "auto",
            self.max_length,
        )
        return CrossEncoder(
            self.model_id,
            revision=self.revision,
            device=self.device,
            max_length=self.max_length,
        )

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        top_k: int = 10,
    ) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be ≥ 1")
        if not hits:
            return []

        pairs = [(query, h.text) for h in hits]
        scores = self._model.predict(pairs, show_progress_bar=False).tolist()
        ranked = sorted(
            zip(scores, hits, strict=True),
            key=lambda pair: (-pair[0], pair[1].chunk_id),
        )
        return [
            RetrievalHit(
                chunk_id=h.chunk_id,
                text=h.text,
                metadata=dict(h.metadata),
                score=float(score),
                rank=i + 1,
                source="reranked",
            )
            for i, (score, h) in enumerate(ranked[:top_k])
        ]


__all__ = ["SentenceTransformersReranker"]
