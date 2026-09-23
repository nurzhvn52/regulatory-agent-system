# ADR 0002: Store metadata and vectors in PostgreSQL with pgvector

- Status: accepted
- Date: 2026-09-20

## Decision

Use PostgreSQL as the system of record for documents, versions, sections, experiment
runs, and citations. Use pgvector for dense embeddings. Keep the BM25 research baseline
behind a separate retrieval adapter.

The development container is exposed on host port `55432` because the development
machine already has a separate PostgreSQL instance listening on the default port `5432`.

## Rationale

Regulatory analysis requires relational version metadata as well as vector similarity.
Keeping both in PostgreSQL reduces operational complexity and enables consistent filters
by language, act type, jurisdiction, and effective date.

## Consequences

- Exact cosine search is the default for the current small evaluation corpus, so the
  retrieval baseline does not lose recall because of approximate indexing.
- A model-specific partial HNSW index will be introduced when corpus size makes exact
  search too slow; exact search remains the reference during index tuning.
- The base embedding column accepts different dimensions. Model-specific partial HNSW
  indexes will be introduced with explicit casts after each experimental model is fixed.
- The BM25 implementation is measured separately rather than conflated with PostgreSQL
  full-text ranking.
