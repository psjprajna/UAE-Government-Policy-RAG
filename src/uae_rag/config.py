"""Adapter selection — the single seam where ports meet implementations.

Reads ``ADAPTER_PROFILE`` (``local`` | ``azure``, default ``local``) and
returns wired-up port implementations. Per ADR-0002 this is the *only*
module in the codebase allowed to import from ``uae_rag.adapters.*``;
the fitness test (``tests/fitness/test_layer_boundaries.py``) enforces it.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from uae_rag.ingestion.chunker import Chunk
from uae_rag.ports import (
    EmbeddingsPort,
    LLMPort,
    RerankerPort,
    RetrievalPort,
    VectorIndexPort,
)

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


def get_retriever(
    *,
    chunks: Sequence[Chunk],
    embedder: EmbeddingsPort,
    vector_index: VectorIndexPort,
    per_leg_top_k: int = 50,
    rrf_k: int = 60,
) -> RetrievalPort:
    """Wire the hybrid retriever — profile-agnostic; both legs route through ports.

    Per ADR-0004, the local fusion runs BM25 (in-memory over ``chunks``) and a
    dense leg backed by ``vector_index``, then merges via RRF. Phase 9's Azure
    wiring will replace this with an Azure AI Search hybrid query that
    satisfies the same ``RetrievalPort``.
    """
    from uae_rag.retrieval.bm25 import BM25Retriever
    from uae_rag.retrieval.dense import DenseRetriever
    from uae_rag.retrieval.hybrid import HybridRetriever

    return HybridRetriever(
        bm25=BM25Retriever(chunks),
        dense=DenseRetriever(embedder=embedder, vector_index=vector_index),
        per_leg_top_k=per_leg_top_k,
        rrf_k=rrf_k,
    )


def get_llm() -> LLMPort:
    """Return the LLM adapter selected by ``ADAPTER_PROFILE``.

    Per ADR-0008 the local profile defaults to Ollama serving ``llama3.1:8b``;
    the model and connection parameters are overridable via the ``LOCAL_LLM_*``
    env vars without touching code. Phase 9 will add an Azure OpenAI adapter
    behind the same ``LLMPort``.
    """
    profile = _profile()
    if profile == "local":
        from uae_rag.adapters.local.llm import OllamaLLM

        return OllamaLLM()
    if profile == "azure":
        raise NotImplementedError("Azure OpenAI LLM adapter ships in Phase 9")
    raise ValueError(f"Unknown ADAPTER_PROFILE: {profile!r}")


def get_judge_llm() -> LLMPort:
    """Return the LLM used by the RAGAS judge, selected by ``RAGAS_JUDGE_PROFILE``.

    Default ``local`` reuses the same :class:`OllamaLLM` as :func:`get_llm` —
    self-bias caveat documented in ADR-0009. The ``openai`` profile is reserved
    for the Phase 9 Azure adapter and raises :class:`NotImplementedError` until
    then so callers fail fast at startup rather than mid-evaluation.
    """
    profile = os.environ.get("RAGAS_JUDGE_PROFILE", _DEFAULT_PROFILE)
    if profile == "local":
        return get_llm()
    if profile == "openai":
        raise NotImplementedError("OpenAI judge ships with Phase 9 Azure adapter")
    raise ValueError(f"Unknown RAGAS_JUDGE_PROFILE: {profile!r}")


def get_reranker() -> RerankerPort:
    """Return the reranker adapter selected by ``ADAPTER_PROFILE``.

    Per ADR-0007, the local profile uses ``BAAI/bge-reranker-v2-m3`` via a
    sentence-transformers ``CrossEncoder``. The Phase 9 Azure/Cohere adapter
    will satisfy the same ``RerankerPort``.
    """
    profile = _profile()
    if profile == "local":
        from uae_rag.adapters.local.reranker import SentenceTransformersReranker

        return SentenceTransformersReranker()
    if profile == "azure":
        raise NotImplementedError("Azure/Cohere reranker adapter ships in Phase 9")
    raise ValueError(f"Unknown ADAPTER_PROFILE: {profile!r}")


__all__ = [
    "get_embeddings",
    "get_judge_llm",
    "get_llm",
    "get_reranker",
    "get_retriever",
    "get_vector_index",
]
