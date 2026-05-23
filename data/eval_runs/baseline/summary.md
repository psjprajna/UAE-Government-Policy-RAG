# Eval run 2026-05-23T17-02-23Z

- **Adapter profile:** local
- **Judge profile:** local
- **Answerer:** llama3.1:8b · **Judge:** llama3.1:8b
- **Embedder:** intfloat/multilingual-e5-large · **Reranker:** bge-reranker-v2-m3
- **RAGAS:** 0.4.3 · **Git:** 528162c
- **Counts:** 50 total · 50 ok · 0 errored

## Overall (mean ± std)

| Metric | Mean | Std | N |
|---|---:|---:|---:|
| faithfulness | 0.826 | 0.259 | 27 |
| answer_relevancy | 0.845 | 0.072 | 40 |
| llm_context_precision_with_reference | 0.947 | 0.125 | 31 |
| context_recall | 0.883 | 0.249 | 46 |
| answer_correctness | 0.665 | 0.137 | 27 |
| domain_quality | 1.000 | 0.000 | 1 |

## By language

### EN

| Metric | Mean | Std | N |
|---|---:|---:|---:|
| faithfulness | 0.830 | 0.187 | 13 |
| answer_relevancy | 0.893 | 0.042 | 23 |
| llm_context_precision_with_reference | 0.925 | 0.161 | 17 |
| context_recall | 0.881 | 0.245 | 27 |
| answer_correctness | 0.685 | 0.148 | 17 |
| domain_quality | 1.000 | 0.000 | 1 |

### AR

| Metric | Mean | Std | N |
|---|---:|---:|---:|
| faithfulness | 0.821 | 0.312 | 14 |
| answer_relevancy | 0.781 | 0.051 | 17 |
| llm_context_precision_with_reference | 0.973 | 0.046 | 14 |
| context_recall | 0.886 | 0.254 | 19 |
| answer_correctness | 0.631 | 0.108 | 10 |
| domain_quality | — | — | 0 |

## By topic

### labour-law

| Metric | Mean | Std | N |
|---|---:|---:|---:|
| faithfulness | 0.838 | 0.272 | 20 |
| answer_relevancy | 0.821 | 0.072 | 26 |
| llm_context_precision_with_reference | 0.962 | 0.075 | 22 |
| context_recall | 0.908 | 0.226 | 29 |
| answer_correctness | 0.637 | 0.117 | 18 |
| domain_quality | 1.000 | 0.000 | 1 |

### mohre

| Metric | Mean | Std | N |
|---|---:|---:|---:|
| faithfulness | 0.840 | 0.193 | 6 |
| answer_relevancy | 0.902 | 0.030 | 11 |
| llm_context_precision_with_reference | 0.867 | 0.227 | 6 |
| context_recall | 0.885 | 0.286 | 12 |
| answer_correctness | 0.723 | 0.164 | 8 |
| domain_quality | — | — | 0 |

### visa

| Metric | Mean | Std | N |
|---|---:|---:|---:|
| faithfulness | 0.500 | 0.000 | 1 |
| answer_relevancy | 0.846 | 0.063 | 3 |
| llm_context_precision_with_reference | 1.000 | 0.000 | 3 |
| context_recall | 0.733 | 0.226 | 5 |
| answer_correctness | 0.717 | 0.000 | 1 |
| domain_quality | — | — | 0 |
