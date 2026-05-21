"""Integration tests for the local ChromaDB vector index adapter.

Marked ``adapter_local`` so they can be filtered out when ChromaDB isn't
installed. No model load — fake vectors throughout. Whole module runs
in well under a second.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from uae_rag.adapters.local.vector_index import (
    ChromaVectorIndex,
    DimensionMismatchError,
)
from uae_rag.ports import VectorIndexPort, VectorRecord

pytestmark = pytest.mark.adapter_local


def _vec(i: int, dim: int = 8) -> list[float]:
    """Unit vector with a 1.0 in slot ``i`` — orthogonal to every other slot."""
    out = [0.0] * dim
    out[i] = 1.0
    return out


def _record(i: int, dim: int = 8) -> VectorRecord:
    return VectorRecord(
        id=f"r{i}",
        embedding=_vec(i, dim),
        document=f"doc-{i}",
        metadata={"source_slug": "labour-law-en", "article_id": None, "page_start": i},
    )


@pytest.fixture
def index(tmp_path: Path) -> ChromaVectorIndex:
    return ChromaVectorIndex(
        persist_dir=tmp_path / "chroma",
        collection_name="test_collection",
        embedder_model_id="fake-model",
        embedder_dimension=8,
    )


def test_chroma_vector_index_satisfies_port(index: ChromaVectorIndex) -> None:
    assert isinstance(index, VectorIndexPort)


def test_upsert_then_count(index: ChromaVectorIndex) -> None:
    index.upsert(_record(i) for i in range(5))
    assert index.count() == 5


def test_query_returns_top_k_by_similarity(index: ChromaVectorIndex) -> None:
    index.upsert(_record(i) for i in range(5))

    hits = index.query(_vec(2), top_k=3)

    assert len(hits) == 3
    assert hits[0].id == "r2"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].document == "doc-2"
    assert hits[0].metadata["source_slug"] == "labour-law-en"
    # Scores are monotonically non-increasing (top-k order).
    for a, b in pairwise(hits):
        assert a.score >= b.score


def test_upsert_is_idempotent_on_repeat(index: ChromaVectorIndex) -> None:
    records = [_record(i) for i in range(3)]

    index.upsert(records)
    index.upsert(records)

    assert index.count() == 3


def test_reset_clears_collection(index: ChromaVectorIndex) -> None:
    index.upsert(_record(i) for i in range(3))
    assert index.count() == 3

    index.reset()

    assert index.count() == 0


def test_where_filter_narrows_query(index: ChromaVectorIndex) -> None:
    a = VectorRecord(id="en-1", embedding=_vec(0), document="d-en", metadata={"language": "en"})
    b = VectorRecord(id="ar-1", embedding=_vec(0), document="d-ar", metadata={"language": "ar"})
    index.upsert([a, b])

    hits = index.query(_vec(0), top_k=5, where={"language": "ar"})

    assert [h.id for h in hits] == ["ar-1"]


def test_dimension_mismatch_on_upsert_raises(tmp_path: Path) -> None:
    """An adapter built for dim=8 refuses a 4-dim vector."""
    index = ChromaVectorIndex(
        persist_dir=tmp_path / "chroma",
        collection_name="dim_check",
        embedder_model_id="fake-model",
        embedder_dimension=8,
    )
    bad = VectorRecord(id="x", embedding=[0.1] * 4, document="d", metadata={"k": "v"})

    with pytest.raises(DimensionMismatchError, match=r"8.*4|4.*8"):
        index.upsert([bad])


def test_dimension_mismatch_across_constructors_blocks_reuse(tmp_path: Path) -> None:
    """A second adapter pointing at the same collection with a different dim is refused."""
    first = ChromaVectorIndex(
        persist_dir=tmp_path / "chroma",
        collection_name="reuse_check",
        embedder_model_id="fake-a",
        embedder_dimension=8,
    )
    first.upsert([_record(0)])
    assert first.count() == 1

    with pytest.raises(DimensionMismatchError):
        ChromaVectorIndex(
            persist_dir=tmp_path / "chroma",
            collection_name="reuse_check",
            embedder_model_id="fake-b",
            embedder_dimension=16,  # mismatch
        )


def test_model_id_change_warns_but_allows(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Same dimension, different model id → WARNING but proceeds (vector space tuning)."""
    import logging

    first = ChromaVectorIndex(
        persist_dir=tmp_path / "chroma",
        collection_name="model_swap",
        embedder_model_id="fake-a",
        embedder_dimension=8,
    )
    first.upsert([_record(0)])

    with caplog.at_level(logging.WARNING, logger="uae_rag.adapters.local.vector_index"):
        second = ChromaVectorIndex(
            persist_dir=tmp_path / "chroma",
            collection_name="model_swap",
            embedder_model_id="fake-b",  # different model, same dim
            embedder_dimension=8,
        )

    assert second.count() == 1
    assert any("model" in r.message.lower() for r in caplog.records)


def test_reset_after_dimension_mismatch_recovers(tmp_path: Path) -> None:
    """``reset()`` drops the collection so a new dim/model can take over."""
    first = ChromaVectorIndex(
        persist_dir=tmp_path / "chroma",
        collection_name="recover",
        embedder_model_id="fake-a",
        embedder_dimension=8,
    )
    first.upsert([_record(0)])
    first.reset()

    # Same persist dir, different dim — should succeed after reset.
    second = ChromaVectorIndex(
        persist_dir=tmp_path / "chroma",
        collection_name="recover",
        embedder_model_id="fake-b",
        embedder_dimension=16,
    )
    second.upsert([VectorRecord(id="new", embedding=[0.1] * 16, document="d", metadata={"k": "v"})])
    assert second.count() == 1
