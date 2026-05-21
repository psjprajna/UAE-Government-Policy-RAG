# UAE Government Policy RAG

A retrieval-augmented question-answering service over public UAE government policy documents — UAE Labour Law (Federal Law No. 33 of 2021), MOHRE regulations, and UAE Visa regulations. Accepts questions in Arabic or English and returns grounded answers with article-level citations.

> **Current capability (through Phase 1):** walking skeleton + reproducible corpus. `POST /query` still returns a deterministic stubbed answer (the real pipeline lands in Phase 7). A one-command fetcher downloads the four UAE government policy PDFs into `data/raw/` and verifies them against a committed SHA-256 registry at `data/registry.json`. Sources behind a JavaScript challenge (uaelegislation.gov.ae, MOHRE) are fetched through a headless Playwright + chromium adapter; direct-PDF sources (ICP) go through stdlib `urllib`. Re-runs are no-ops; tampered or stale files surface as a clear hash-mismatch error rather than a silent drift.

## Why this exists

UAE Labour Law and the surrounding policy stack are dense, multilingual, and frequently consulted by HR practitioners. Generic chat models hallucinate articles. This service retrieves the actual government text first, grounds every claim in a cited chunk, and reports its own quality via RAGAS scores on a 50-question golden set.

## Stack

- **Web**: FastAPI + Uvicorn, Python 3.11
- **Package management**: `uv`
- **Architecture**: Hexagonal (Ports & Adapters) — `ports/` define interfaces, `adapters/local/` provides free local impls (ChromaDB, Ollama, sentence-transformers, BGE reranker), `adapters/azure/` is the swap-in for Azure OpenAI + Azure AI Search

The full architecture and decision history live alongside the project but outside this repo (see "Repository layout" below).

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Run the dev server
uv run uvicorn uae_rag.api.main:app --reload

# 3. In another terminal, hit the endpoint
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the annual leave entitlement?"}'
```

You should receive a JSON response of shape:

```json
{
  "answer": "...",
  "citations": [
    {"source": "UAE Labour Law", "article": "..."}
  ],
  "language": "en"
}
```

Swagger UI is auto-served at `http://localhost:8000/docs`.

## Corpus fetch

The four indexed PDFs are downloaded from their official UAE government sources and verified by SHA-256 against `data/registry.json` (committed). The PDFs themselves live in `data/raw/` and are gitignored.

### One-time setup (browser-fetched sources)

`labour-law-en`, `labour-law-ar`, and `mohre-resolutions` are served from portals (`uaelegislation.gov.ae`, `mohre.gov.ae`) that gate their downloads behind a JavaScript challenge that plain HTTP clients cannot defeat. They go through a Playwright + headless chromium adapter (`src/uae_rag/adapters/local/browser_fetcher.py`). Install once:

```bash
uv sync --extra fetch --extra dev          # adds the playwright Python package
uv run playwright install chromium         # downloads the chromium binary (~150 MB, one-time per machine)
```

`visa-regulations` is served directly by ICP and uses stdlib `urllib` — no browser needed.

### Running the fetcher

```bash
uv run python scripts/fetch_corpus.py                     # download anything missing and verify
uv run python scripts/fetch_corpus.py --skip-download     # verify only what's on disk; never hit the network
uv run python scripts/fetch_corpus.py --source SLUG       # operate on one source (repeatable); SLUGs: labour-law-en, labour-law-ar, mohre-resolutions, visa-regulations
uv run python scripts/fetch_corpus.py --force             # accept hash drift after confirming the upstream document genuinely updated
```

Exit codes: `0` if every selected source verifies; `1` if any source fails to download or its hash drifts; `2` on bad CLI usage.

### Manual-drop fallback

If a portal upgrades its bot detection or the chromium install is unavailable on a machine, you can drop the file manually:

1. Open the source URL in a real browser and save the PDF (URLs are in `src/uae_rag/ingestion/registry.py:SOURCES`).
2. Place it at `data/raw/<local_filename>.pdf`.
3. Run `uv run python scripts/fetch_corpus.py --source <slug> --skip-download` to register its SHA.

The same procedure applies to any source that becomes harder to fetch automatically in the future.

## Tests

```bash
uv run pytest                     # full suite
uv run pytest tests/fitness/      # architectural boundary tests only
uv run pytest tests/e2e/          # API end-to-end smoke
```

## Repository layout

Only `src/`, `tests/`, `.gitignore`, `pyproject.toml`, and this README are version-controlled here. The development governance (architecture notes, decision records, specifications, status) lives in a sibling workspace folder that is intentionally not committed — it's the engineer's working memory, not part of the shipped artifact.

```
src/uae_rag/
├── api/          # FastAPI routers (entrypoint)
├── ports/        # Interface definitions (Embeddings, VectorIndex, LLM, Reranker, Parser)
├── adapters/
│   ├── local/    # ChromaDB, Ollama, sentence-transformers, BGE — free to run
│   └── azure/    # Azure OpenAI, Azure AI Search, Form Recognizer (Phase 9)
├── ingestion/    # PDF parser, heading-aware chunker, document registry
├── retrieval/    # Hybrid BM25 + dense, multi-query, rerank pipeline
├── generation/   # Language detection, prompt assembly, citation injection
├── evals/        # RAGAS harness + golden set
└── config.py     # Reads ADAPTER_PROFILE; wires ports to adapter implementations
```

## License

Project code is for educational and portfolio use. The indexed source documents are public UAE government regulations and remain the property of their respective publishers.
