# Retrieval pipeline

```text
                              +-> BM25 ------------------+
query -> explicit corpus scope                            +-> RRF -> ranked evidence
                              +-> BGE-M3 -> pgvector ----+
```

## Reproducible corpus scope

Every search must specify the 64-character ingestion `pipeline_signature` and the
chunking strategy. Optional language, act type, document, and effective-date filters are
applied identically to lexical and dense retrieval. This prevents an experiment from
silently comparing systems over different document fragments.

## Lexical baseline

The BM25 implementation is dependency-free and uses the standard Okapi formula with
`k1=1.5` and `b=0.75`. Its Unicode tokenizer supports Russian, Kazakh, English, and
hyphenated legal numbers. It intentionally performs no stemming or stop-word removal;
future morphology variants must be evaluated as separately named baselines.

## Dense baseline

- Model: `BAAI/bge-m3`.
- Pinned revision: `5617a9f61b028005a4858fdac845db406aefb181`.
- Output: 1024-dimensional, L2-normalized dense vectors.
- Distance: exact cosine distance in pgvector.
- Runtime: Sentence Transformers on CUDA when available.

The embedding row records the model name, exact revision, vector dimensions,
normalization flag, adapter version, and encoding configuration. Repeated indexing only
embeds missing `(chunk, model, revision)` combinations.

## Hybrid ranking

BM25 and dense retrieval each return four times the requested result count. Reciprocal
Rank Fusion combines the lists with `k=60`; raw BM25 and cosine scores are not mixed
because they are on incompatible scales.

## Evaluation

`RetrievalEvaluator` computes binary-relevance Recall@k, reciprocal rank, and nDCG@k.
Benchmark relevance labels must be reviewed manually and stored under
`data/evaluation/reviewed/`. Version-controlled candidates in `data/evaluation/drafts/`
require explicit `--allow-draft` for pilot runs and are not treated as expert ground truth.
The command `regagent evaluate` produces per-query and language-level reports;
see [evaluation guide](evaluation.md) for the complete protocol.
