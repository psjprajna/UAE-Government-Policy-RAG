"""Local embeddings adapter — sentence-transformers wrapping a multilingual model.

Defaults to ``intfloat/multilingual-e5-large`` (ADR-0001). Model id, dimension,
and the E5 ``passage:`` / ``query:`` prefixes are read from env vars so a
different sentence-transformers model can be swapped in without editing code:

    LOCAL_EMBEDDINGS_MODEL=BAAI/bge-m3
    LOCAL_EMBEDDINGS_DIMENSION=1024
    LOCAL_EMBEDDINGS_PASSAGE_PREFIX=
    LOCAL_EMBEDDINGS_QUERY_PREFIX=

The underlying ``SentenceTransformer`` instance is loaded lazily on the first
``embed_documents`` / ``embed_query`` / ``count_tokens`` call. Until then the
adapter is cheap to construct and import — important so that ``--dry-run``
build runs and unit tests don't pay the ~1.4 GB model download cost.
"""

from __future__ import annotations

import logging
import os
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "intfloat/multilingual-e5-large"
_DEFAULT_DIMENSION = 1024
_DEFAULT_PASSAGE_PREFIX = "passage: "
_DEFAULT_QUERY_PREFIX = "query: "


class SentenceTransformersEmbeddings:
    """``EmbeddingsPort`` implementation backed by sentence-transformers.

    Attributes are populated from env vars at construction time. The model is
    not loaded until the first call that needs it.
    """

    def __init__(self) -> None:
        self.model_id: str = os.environ.get("LOCAL_EMBEDDINGS_MODEL", _DEFAULT_MODEL)
        self.dimension: int = int(
            os.environ.get("LOCAL_EMBEDDINGS_DIMENSION", str(_DEFAULT_DIMENSION))
        )
        self.passage_prefix: str = os.environ.get(
            "LOCAL_EMBEDDINGS_PASSAGE_PREFIX", _DEFAULT_PASSAGE_PREFIX
        )
        self.query_prefix: str = os.environ.get(
            "LOCAL_EMBEDDINGS_QUERY_PREFIX", _DEFAULT_QUERY_PREFIX
        )

    @cached_property
    def _model(self) -> SentenceTransformer:
        from sentence_transformers import SentenceTransformer

        logger.info("loading sentence-transformer model %s", self.model_id)
        return SentenceTransformer(self.model_id)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"{self.passage_prefix}{t}" for t in texts]
        vectors = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            f"{self.query_prefix}{text}",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [float(x) for x in vector]

    def count_tokens(self, text: str) -> int:
        tokenizer = self._model.tokenizer
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return len(encoded)


__all__ = ["SentenceTransformersEmbeddings"]
