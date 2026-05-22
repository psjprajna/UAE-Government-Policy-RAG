"""Language detection — Arabic vs. English routing for the generator.

A pure-function domain utility, not a port. Only one library (``lingua``) is
in scope and there is no Azure swap target; introducing an adapter layer here
would be over-engineering. The detector is built once at module-import time
(build cost is ~0.1 ms; first detect call is ~7 ms warmup; subsequent calls
are sub-millisecond).

Empty / whitespace-only / digit-only input → ``"en"`` (lingua returns
``None`` for these; we default to English). This is documented behavior, not
a guess: the only place it surfaces is the empty-hits short-circuit in
``Generator``, where the refusal phrase is what callers care about.
"""

from __future__ import annotations

from typing import Literal

from lingua import Language, LanguageDetectorBuilder

_DETECTOR = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.ARABIC).build()
_DEFAULT: Literal["en"] = "en"


def detect_language(text: str) -> Literal["en", "ar"]:
    """Return ``"en"`` or ``"ar"`` for ``text``; defaults to ``"en"`` on ambiguity.

    The detector is restricted to English and Arabic, so any other input
    (French, digits, single ASCII character) is forced into one of the two
    or, when lingua returns ``None``, falls back to English.
    """
    if not text.strip():
        return _DEFAULT
    lang = _DETECTOR.detect_language_of(text)
    if lang is None:
        return _DEFAULT
    return "ar" if lang == Language.ARABIC else "en"


__all__ = ["detect_language"]
