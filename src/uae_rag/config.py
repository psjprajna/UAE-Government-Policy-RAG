"""Adapter selection — the single seam where ports meet implementations.

Reads ``ADAPTER_PROFILE`` (``local`` | ``azure``, default ``local``) and
returns wired-up port implementations. Per ADR-0002 this is the *only*
module in the codebase allowed to import from ``uae_rag.adapters.*``;
the fitness test (``tests/fitness/test_layer_boundaries.py``) enforces it.
"""

from __future__ import annotations

import os
from pathlib import Path

from uae_rag.ports import EmbeddingsPort, VectorIndexPort

_DEFAULT_PROFILE = "local"
_DEFAULT_COLLECTION = "uae_policy_chunks"


def _profile() -> str:
    return os.environ.get("ADAPTER_PROFILE", _DEFAULT_PROFILE)


def get_embeddings() -> EmbeddingsPort:
    """Return the embeddings adapter selected by ``ADAPTER_PROFILE``."""
    profile = _profile()
    if profile == "local":
        from uae_rag.adapters.local.embeddings import SentenceTransformersEmbeddings

        return SentenceTransformersEmbeddings()
    if profile == "azure":
        raise NotImplementedError("Azure embeddings adapter ships in Phase 9")
    raise ValueError(f"Unknown ADAPTER_PROFILE: {profile!r}")


def get_vector_index(
    persist_dir: Path,
    *,
    embedder: EmbeddingsPort,
    collection_name: str = _DEFAULT_COLLECTION,
) -> VectorIndexPort:
    """Return the vector index adapter selected by ``ADAPTER_PROFILE``.

    The index is constructed with the embedder's identity so on-disk
    indexes can detect an incompatible model swap and refuse the write.
    """
    profile = _profile()
    if profile == "local":
        from uae_rag.adapters.local.vector_index import ChromaVectorIndex

        return ChromaVectorIndex(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedder_model_id=embedder.model_id,
            embedder_dimension=embedder.dimension,
        )
    if profile == "azure":
        raise NotImplementedError("Azure vector index adapter ships in Phase 9")
    raise ValueError(f"Unknown ADAPTER_PROFILE: {profile!r}")


__all__ = ["get_embeddings", "get_vector_index"]
