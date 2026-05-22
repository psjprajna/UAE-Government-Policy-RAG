"""Generation package — language detection, prompt assembly, citation rendering, and the Generator.

Per ADR-0002 these modules import only from ``uae_rag.ports`` and the standard
library (plus the ``lingua`` utility for language detection). Concrete LLM
adapters live under ``uae_rag.adapters.*`` and are wired in ``config.py``.
"""
