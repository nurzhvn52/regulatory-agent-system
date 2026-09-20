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

- HNSW is the default approximate vector index.
- Exact search remains available for recall checks on small evaluation corpora.
- The base embedding column accepts different dimensions. Model-specific partial HNSW
  indexes will be introduced with explicit casts after each experimental model is fixed.
- The BM25 implementation is measured separately rather than conflated with PostgreSQL
  full-text ranking.
