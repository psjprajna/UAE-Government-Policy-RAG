# UAE Government Policy RAG

A retrieval-augmented question-answering service over public UAE government policy documents — UAE Labour Law (Federal Law No. 33 of 2021), MOHRE regulations, and UAE Visa regulations. Accepts questions in Arabic or English and returns grounded answers with article-level citations.

> **Current capability (Phase 0):** walking skeleton. A FastAPI service exposes `POST /query` and returns a deterministic stubbed answer with one citation. Real ingestion, retrieval, generation, and evaluation are added in subsequent phases.

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
