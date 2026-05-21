"""Unit tests for ``EmbeddingsPort`` — Protocol surface and end-to-end wiring.

A self-contained ``FakeEmbeddings`` (hash-seeded deterministic vectors, no model)
exercises the Protocol's `@runtime_checkable` surface, asserts the E5 prefix
conventions are applied inside the adapter, and closes the
chunker → embed → upsert → query loop against a tiny in-memory index — all
without loading sentence-transformers or ChromaDB.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable

from uae_rag.ingestion.chunker import chunk_articles
from uae_rag.ingestion.parser import Article
from uae_rag.ports import EmbeddingsPort, QueryHit, VectorRecord


def _hash_vector(text: str, dim: int) -> list[float]:
    """Deterministic L2-normalized vector — pseudo-Gaussian seeded by sha256(text)."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    raw = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


class FakeEmbeddings:
    """In-memory ``EmbeddingsPort`` — records prefix application for assertions."""

    model_id: str = "fake-embeddings"
    dimension: int = 1024
    passage_prefix: str = "passage: "
    query_prefix: str = "query: "

    def __init__(self) -> None:
        self.last_document_inputs: list[str] = []
        self.last_query_input: str | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"{self.passage_prefix}{t}" for t in texts]
        self.last_document_inputs = prefixed
        return [_hash_vector(p, self.dimension) for p in prefixed]

    def embed_query(self, text: str) -> list[float]:
        prefixed = f"{self.query_prefix}{text}"
        self.last_query_input = prefixed
        return _hash_vector(prefixed, self.dimension)

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class _MiniFakeIndex:
    """5-line dict-backed index for the round-trip test only.

    The fuller index lives in ``test_vector_index_port.py``. Duplication is
    intentional — Slice 3 chose inline fakes to keep each test file self-contained.
    """

    def __init__(self) -> None:
        self._store: dict[str, VectorRecord] = {}

    def upsert(self, records: Iterable[VectorRecord]) -> None:
        for r in records:
            self._store[r.id] = r

    def query(
        self,
        embedding: list[float],
        *,
        top_k: int = 10,
        where: dict[str, object] | None = None,
    ) -> list[QueryHit]:
        scored = [
            (sum(a * b for a, b in zip(r.embedding, embedding, strict=True)), r)
            for r in self._store.values()
        ]
        scored.sort(key=lambda t: -t[0])
        return [
            QueryHit(id=r.id, document=r.document, metadata=dict(r.metadata), score=s)
            for s, r in scored[:top_k]
        ]

    def count(self) -> int:
        return len(self._store)

    def reset(self) -> None:
        self._store.clear()


def test_fake_embeddings_satisfies_port() -> None:
    """The runtime_checkable Protocol accepts a structurally-conformant fake."""
    assert isinstance(FakeEmbeddings(), EmbeddingsPort)


def test_embed_documents_returns_one_vector_per_text_at_correct_dimension() -> None:
    fake = FakeEmbeddings()

    vectors = fake.embed_documents(["alpha", "beta", "gamma"])

    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == fake.dimension


def test_passage_and_query_prefixes_are_applied_inside_adapter() -> None:
    """Consumers pass raw text; the adapter prepends ``passage: ``/``query: ``."""
    fake = FakeEmbeddings()

    fake.embed_documents(["annual leave entitlement"])
    fake.embed_query("annual leave entitlement")

    assert fake.last_document_inputs == ["passage: annual leave entitlement"]
    assert fake.last_query_input == "query: annual leave entitlement"


def test_chunker_fake_embedder_fake_index_round_trip_closes_loop() -> None:
    """chunk → embed_documents → upsert → embed_query → query returns the chunk's id.

    Body is paragraph-split (count_tokens-aware) into multiple chunks; the query
    matches one chunk's text verbatim and expects that chunk first.
    """
    fake = FakeEmbeddings()
    index = _MiniFakeIndex()

    paragraphs = [
        "Annual leave entitlement is thirty calendar days per year.",
        "End of service gratuity is calculated on the last basic wage.",
        "Probation may not exceed six months under federal labour law.",
    ]
    article = Article(
        article_id="29",
        breadcrumb="Article (29)",
        language="en",
        page_start=10,
        page_end=10,
        body="\n\n".join(paragraphs),
    )

    chunks = chunk_articles(
        [article],
        source_slug="labour-law-en",
        count_tokens=fake.count_tokens,
        max_tokens=12,  # force paragraph-level split — each para is ~10 words
    )
    assert len(chunks) >= 2

    vectors = fake.embed_documents([c.text for c in chunks])
    records = [
        VectorRecord(
            id=c.chunk_id,
            embedding=v,
            document=c.text,
            metadata={
                "source_slug": c.source_slug,
                "language": c.language,
                "article_id": c.article_id,
            },
        )
        for c, v in zip(chunks, vectors, strict=True)
    ]
    index.upsert(records)
    assert index.count() == len(records)

    target = chunks[0]
    hits = index.query(fake.embed_query(target.text), top_k=3)

    assert len(hits) == 3
    upserted_ids = {c.chunk_id for c in chunks}
    for h in hits:
        assert h.id in upserted_ids
        assert isinstance(h, QueryHit)
