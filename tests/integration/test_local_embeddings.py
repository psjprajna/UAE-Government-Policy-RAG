"""Integration tests for the local ``SentenceTransformersEmbeddings`` adapter.

Two sections:

* **Section A — env-driven swap** (fast, no model load). Verifies the four
  ``LOCAL_EMBEDDINGS_*`` env vars override adapter attributes at construction,
  and that the defaults match the documented e5-large contract. These tests
  never touch ``_model`` so no download is triggered.
* **Section B — real model** (``@pytest.mark.slow``). Loads
  ``intfloat/multilingual-e5-large`` once per module and asserts the
  load-bearing facts: dim 1024, L2-normalization, prefix prepending, and a
  pinned ``count_tokens`` baseline so tokenizer drift fires loudly.

Skipped in the default fast lane. Run with::

    uv run pytest -m "slow and adapter_local"
"""

from __future__ import annotations

import math

import pytest

from uae_rag.adapters.local.embeddings import SentenceTransformersEmbeddings

pytestmark = pytest.mark.adapter_local


# --- Section A — env-driven swap (fast, no model load) -----------------------

_ENV_VARS = (
    "LOCAL_EMBEDDINGS_MODEL",
    "LOCAL_EMBEDDINGS_DIMENSION",
    "LOCAL_EMBEDDINGS_PASSAGE_PREFIX",
    "LOCAL_EMBEDDINGS_QUERY_PREFIX",
)


def test_env_overrides_propagate_to_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """All four LOCAL_EMBEDDINGS_* vars override adapter attributes at construction."""
    monkeypatch.setenv("LOCAL_EMBEDDINGS_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("LOCAL_EMBEDDINGS_DIMENSION", "768")
    monkeypatch.setenv("LOCAL_EMBEDDINGS_PASSAGE_PREFIX", "")
    monkeypatch.setenv("LOCAL_EMBEDDINGS_QUERY_PREFIX", "")

    embedder = SentenceTransformersEmbeddings()

    assert embedder.model_id == "BAAI/bge-m3"
    assert embedder.dimension == 768
    assert embedder.passage_prefix == ""
    assert embedder.query_prefix == ""


def test_defaults_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env overrides, defaults match the documented e5-large contract."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    embedder = SentenceTransformersEmbeddings()

    assert embedder.model_id == "intfloat/multilingual-e5-large"
    assert embedder.dimension == 1024
    assert embedder.passage_prefix == "passage: "
    assert embedder.query_prefix == "query: "


# --- Section B — real model (slow) -------------------------------------------


@pytest.fixture(scope="module")
def real_embedder() -> SentenceTransformersEmbeddings:
    """Load the real e5-large model once per module — amortizes the ~3 s warm-cache load."""
    embedder = SentenceTransformersEmbeddings()
    # Force lazy-load up front so first test doesn't pay the full price.
    embedder.embed_query("warmup")
    return embedder


@pytest.mark.slow
def test_embeddings_dimension_is_1024(real_embedder: SentenceTransformersEmbeddings) -> None:
    vectors = real_embedder.embed_documents(["hello"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 1024


@pytest.mark.slow
def test_embeddings_are_l2_normalized(real_embedder: SentenceTransformersEmbeddings) -> None:
    """sentence-transformers ``normalize_embeddings=True`` produces unit-norm vectors."""
    doc_vec = real_embedder.embed_documents(["annual leave entitlement"])[0]
    query_vec = real_embedder.embed_query("annual leave entitlement")

    assert math.sqrt(sum(x * x for x in doc_vec)) == pytest.approx(1.0, abs=1e-5)
    assert math.sqrt(sum(x * x for x in query_vec)) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.slow
def test_passage_and_query_prefixes_produce_different_vectors(
    real_embedder: SentenceTransformersEmbeddings,
) -> None:
    """``embed_documents`` prepends ``passage: ``; ``embed_query`` prepends ``query: ``.

    Same raw text + different prefixes ⇒ different encoded vectors. If the
    prefixes were dropped, the two outputs would be identical.
    """
    text = "annual leave entitlement"
    doc_vec = real_embedder.embed_documents([text])[0]
    query_vec = real_embedder.embed_query(text)

    cosine = sum(a * b for a, b in zip(doc_vec, query_vec, strict=True))
    # Same semantic content but a load-bearing prefix difference → high but not 1.0.
    assert cosine < 0.999, f"prefixes appear to be dropped (cosine={cosine:.4f})"


@pytest.mark.slow
def test_count_tokens_baseline(real_embedder: SentenceTransformersEmbeddings) -> None:
    """Pinned regression sentinel — tokenizer drift surfaces as a sharp failure.

    XLM-RoBERTa sentencepiece on ``"hello world"`` with ``add_special_tokens=False``
    yields 3 tokens under the current model revision. If the tokenizer revision
    or sentence-transformers' wiring changes, this baseline catches it.
    """
    assert real_embedder.count_tokens("hello world") == 3
