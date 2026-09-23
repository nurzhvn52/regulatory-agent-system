# Regulatory Agent System

Research platform for generating evidence-grounded intelligent agents that analyse
regulatory documents with lexical and vector retrieval plus large language models.

The repository is intentionally organised as a modular monolith. Domain and research
logic do not depend on a specific LLM provider, embedding model, vector store, or agent
runtime.

## Initial scope

- Public regulatory legal acts of the Republic of Kazakhstan.
- Russian first, with Kazakh represented throughout the data model.
- Three agent tasks: cited question answering, requirement extraction, and comparison
  of document versions.
- Four retrieval baselines: BM25, dense, hybrid, and hybrid with reranking.

## Prerequisites

- Python 3.11
- Poetry 2.x
- Docker Desktop with Linux containers

## Local setup

```powershell
Copy-Item .env.example .env
poetry env use 3.11
poetry install
docker compose up -d db
poetry run alembic upgrade head
poetry run uvicorn regagent.api.app:create_app --factory --reload
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation.

## Quality checks

```powershell
poetry run ruff check .
poetry run mypy src
poetry run pytest
```

## Current API

- `GET /health` reports application status.
- `POST /v1/agent-plans/compile` compiles a declarative `AgentSpec` into a deterministic
  workflow plan. A LangGraph runtime adapter will execute these plans in a later stage.

Architecture decisions are recorded under `docs/adr` so that implementation choices can
be cited and reproduced in the dissertation.

The embedding column intentionally accepts multiple dimensions. Once a model is selected
for a concrete experiment, a model-specific partial HNSW index can be added without
preventing comparison with embeddings of another dimensionality.

## Ingest a document

The development database is exposed on port `55432` to avoid collisions with a local
PostgreSQL installation. An ingestion run stores both the fixed-window baseline and the
legal-structure chunks:

```powershell
poetry run regagent ingest data/raw/document.html `
  --source-url "https://official.example/document" `
  --publisher "Official publisher" `
  --issuer "Issuing authority" `
  --language ru `
  --act-type law `
  --official-number "ACT-001"
```

Running the same command again is idempotent: an unchanged content hash returns
`duplicate: true`. A changed source is stored as another document version.

Supported inputs are HTML, DOCX, text-based PDF, and UTF-8 TXT. An image-only PDF fails
with an explicit OCR-required error; OCR will be implemented as a separate adapter so it
can be evaluated independently.

## Index and search

Retrieval always uses an explicit preprocessing snapshot. Copy the `pipeline_signature`
returned by ingestion, then create BGE-M3 embeddings for the desired corpus:

```powershell
poetry run regagent index `
  --pipeline-signature "<64-character-signature>" `
  --chunking-strategy legal_structure `
  --language ru `
  --language kk
```

Run the lexical, dense, or hybrid baseline with the same scope:

```powershell
poetry run regagent search "кто проводит правовой мониторинг" `
  --strategy hybrid `
  --pipeline-signature "<64-character-signature>" `
  --chunking-strategy legal_structure `
  --language ru `
  --top-k 5
```

The current research corpus uses exact cosine search. This preserves perfect vector
recall while the corpus is small; HNSW will be added and measured separately when exact
search becomes a bottleneck. Retrieval architecture and metric definitions are recorded
in `docs/architecture/retrieval.md`.
