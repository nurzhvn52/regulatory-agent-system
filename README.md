# Regulatory Agent System

[![CI](https://github.com/nurzhvn52/regulatory-agent-system/actions/workflows/ci.yml/badge.svg)](https://github.com/nurzhvn52/regulatory-agent-system/actions/workflows/ci.yml)

Ask a question about Kazakhstan's laws and get an answer that cites the exact article it
came from, or a refusal when the documents do not support one. This is the codebase of my
master's thesis at Astana IT University: it loads official legal acts in Russian and
Kazakh, splits them along their legal structure, searches them with BM25, BGE-M3
embeddings or both, and checks every citation in a generated answer against the source
text.

**Status:** research in progress. Retrieval and the cited-QA agent run end to end; the
60-question pilot uses AI-drafted questions that are still waiting for expert review, so
its numbers are not results yet.

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
  workflow plan. The first cited-QA plan can now be executed from the CLI; other task
  plans and reranking remain research work.

See [cited QA runtime](docs/architecture/cited-qa-runtime.md) for local LLM setup,
version pinning, citation checks, and known limitations.

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

## Evaluate retrieval

A pilot dataset contains 60 AI-authored, unreviewed questions (30 RU/KK pairs).
Run all three baselines on the same pinned corpus:

```powershell
poetry run regagent evaluate data/evaluation/drafts/legal_acts_ru_kk.yaml `
  --pipeline-signature 8cc110d69566d224ebebde20f022187eb19cead5992ced909fcb088ed697a88d `
  --allow-draft `
  --output-dir artifacts/my-pilot-run
```

The new output directory contains JSON results, a CSV summary by language, provenance,
and a `review.md` packet with questions and source evidence. Draft labels are refused
unless `--allow-draft` is supplied. These pilot measurements are not dissertation
findings until the relevance labels and experimental design have been reviewed.

See [evaluation guide](docs/architecture/evaluation.md) for metric definitions,
review workflow, version pinning, and limitations of article-overlap labels.

## Evaluate cited QA end to end

`regagent evaluate-agent` runs the pinned RU/KK questions through retrieval,
generation, citation verification, and refusal handling. It writes durable
per-case traces and a human-review worksheet, but leaves legal answer accuracy
ungraded until independent review. See the [agent evaluation guide](docs/architecture/agent-evaluation.md).
The [full 60-case technical pilot](docs/research/agent-pilot-2026-09-24.md)
records the first end-to-end baseline and its limitations.
