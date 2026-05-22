"""Multi-query retriever — recall augmentation via LLM-generated query rephrasings.

Per ADR-0004's Phase 4 deferral, this wrapper composes a ``RetrievalPort`` with
an ``LLMPort`` to expand one user query into ``n_variations`` distinct
rephrasings, retrieves each variation (plus the original) independently from
the inner retriever, then fuses the per-query hit lists with the same
Reciprocal Rank Fusion (``k=60``) algebra ``HybridRetriever`` uses. The wrapper
itself satisfies ``RetrievalPort`` so Phase 7's ``/query`` can compose
``MultiQuery → Hybrid → Rerank → Generator`` against a single stable seam.

The variation prompt is English-only by design (ADR-0008): the LLM rephrases
for retrieval recall, not for the user, and most documented query-expansion
prompting research is English. Arabic queries pass through to the inner
retriever directly via the unchanged original query; the variations are an
additional recall layer.

LLM failure during variation generation degrades cleanly to a single
inner-retriever call with the original query — ``/query`` never falls over
because the LLM is down. Per ADR-0002 this module imports only from ``ports/``
and the standard library.
"""

from __future__ import annotations

import logging
import re

from uae_rag.ports.llm import LLMPort
from uae_rag.ports.retrieval import RetrievalHit, RetrievalPort

logger = logging.getLogger(__name__)

_VARIATION_PROMPT = """\
You generate alternative phrasings of a search query to improve retrieval recall over a corpus of UAE government policy documents.

Original query: {query}

Produce {n} distinct alternative phrasings, one per line, numbered 1.-{n}.
Keep each variation short (<= 15 words) and avoid changing the meaning.

Variations:
"""

# Strips leading numeric/bullet prefixes such as ``1.``, ``2)``, ``3 -``, ``4 ``, ``- ``.
_PREFIX_RE = re.compile(r"^\s*(?:\d+\s*[.\):\-]?|[-*•])\s*")


class MultiQueryRetriever:
    """Composes ``RetrievalPort`` + ``LLMPort`` for query expansion with RRF fusion."""

    def __init__(
        self,
        *,
        retriever: RetrievalPort,
        llm: LLMPort,
        n_variations: int = 3,
        rrf_k: int = 60,
        max_variation_tokens: int = 64,
    ) -> None:
        if n_variations < 1:
            raise ValueError("n_variations must be >= 1")
        self._retriever = retriever
        self._llm = llm
        self._n_variations = n_variations
        self._rrf_k = rrf_k
        self._max_variation_tokens = max_variation_tokens

    def retrieve(self, query: str, *, top_k: int = 20) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not query.strip():
            return self._retriever.retrieve(query, top_k=top_k)

        try:
            raw = self._llm.generate(
                _VARIATION_PROMPT.format(query=query, n=self._n_variations),
                max_output_tokens=self._max_variation_tokens,
                temperature=0.0,
            )
        except Exception:
            logger.exception(
                "LLM variation generation failed; falling back to single-query retrieval"
            )
            return self._retriever.retrieve(query, top_k=top_k)

        variations = _parse_variations(raw, self._n_variations)
        # ``dict.fromkeys`` preserves insertion order while collapsing duplicates,
        # including the case where the LLM echoes the original query verbatim.
        queries = list(dict.fromkeys([query, *variations]))

        # TODO(perf): Phase 8 may justify ThreadPoolExecutor across queries; sequential
        # for now to mirror HybridRetriever's per-leg policy.
        hit_lists = [self._retriever.retrieve(q, top_k=top_k) for q in queries]
        return self._fuse(hit_lists)[:top_k]

    def _fuse(self, hit_lists: list[list[RetrievalHit]]) -> list[RetrievalHit]:
        """Sum RRF contributions per ``chunk_id``; sort score-desc with chunk_id tie-break."""
        scores: dict[str, float] = {}
        carrier: dict[str, RetrievalHit] = {}
        for hits in hit_lists:
            for hit in hits:
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
                source=carrier[cid].source,
            )
            for rank, (cid, score) in enumerate(ordered, start=1)
        ]


def _parse_variations(raw: str, n: int) -> list[str]:
    """Extract up to ``n`` distinct variations from the LLM's line-delimited output."""
    seen: set[str] = set()
    out: list[str] = []
    for line in raw.splitlines():
        candidate = _PREFIX_RE.sub("", line).strip()
        if not candidate:
            continue
        if candidate in seen:
            logger.debug("dedup dropped duplicate variation: %s", candidate)
            continue
        seen.add(candidate)
        out.append(candidate)
        if len(out) >= n:
            break
    return out


__all__ = ["MultiQueryRetriever"]
