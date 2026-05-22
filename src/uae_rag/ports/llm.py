"""Port: synchronous LLM completion behind a single ``generate`` call.

The domain layer depends on this protocol; concrete LLM adapters under
``uae_rag.adapters.*`` implement it. Per ADR-0002 no domain module imports an
adapter directly; per ADR-0001 the local profile defaults to Ollama serving
``llama3.1:8b``; ADR-0008 locks the model, decoding defaults, and the sync
port shape (vs. async — explicitly deferred so ``MultiQueryRetriever`` can
satisfy the sync ``RetrievalPort``).

A transport-level failure (daemon unreachable, timeout, non-200 response)
surfaces as ``LLMUnavailableError`` so callers can decide between a 503
response, a fallback, or propagation without parsing provider-specific
exception types.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class LLMUnavailableError(RuntimeError):
    """The LLM endpoint could not be reached or returned an unrecoverable error.

    The underlying provider exception is preserved on ``__cause__`` for
    debugging; the message itself avoids leaking the configured host into
    logs (per ADR-0008's privacy note).
    """


@runtime_checkable
class LLMPort(Protocol):
    """Interface for synchronous prompt → completion calls.

    Attributes:
        model_id: Provider-qualified model identifier (e.g. ``"llama3.1:8b"``).
    """

    model_id: str

    def generate(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Return the model's completion for ``prompt``.

        Implementations must raise ``ValueError`` for ``max_output_tokens < 1``
        before contacting the provider, and ``LLMUnavailableError`` for any
        transport-level failure (connection error, timeout, non-200, malformed
        response). Greedy decoding at ``temperature=0.0`` is required to be
        deterministic on the same prompt + model build.
        """
        ...


__all__ = ["LLMPort", "LLMUnavailableError"]
