# Cited QA runtime (first executable agent)

The runtime accepts only `cited_qa` + `cited_answer_v1`. It executes the generated
retrieve → answer → verify → finish plan. `hybrid_rerank` plans are explicitly rejected
until a real reranker is implemented; no reranking is silently substituted.

## Local run

Use an OpenAI-compatible Chat Completions server with a model that can produce JSON.
The server URL is explicit: the application does not send regulatory text to a hosted
provider unless you configure such an endpoint. For example, with a local server at
`http://127.0.0.1:1234/v1`:

```powershell
$env:REGAGENT_LLM_PROVIDER = "openai_compatible"
$env:REGAGENT_LLM_MODEL = "<model-name-served-locally>"
$env:REGAGENT_LLM_BASE_URL = "http://127.0.0.1:1234/v1"
poetry run regagent answer "Какой срок хранения документов?" `
  --pipeline-signature "<64-character-signature>" `
  --version-id "<UUID-from-search-output>" `
  --effective-on 2026-01-01 `
  --language ru
```

The strict MVP specification requires both a pinned version ID and the date on which
it must be effective. The version must have a known `effective_from` in the database;
unknown dates are excluded. Import a dated document version via the `ingest` command's
`--effective-from` argument. If no evidence qualifies, the agent refuses to answer.
Use `regagent search` to inspect version IDs first. The response contains the run ID,
source URL, document ID, version ID, chunk ID, article, exact quote, and model usage.
The PostgreSQL `agent_specs` and `agent_runs` tables retain the spec snapshot, input,
output, status, and node trace. Set `--experiment-id` to group runs.

The existing research corpus has no recorded `effective_from` values. For a technical
pilot on that corpus, use `--spec configs/agents/cited_qa_pilot.yaml` and still provide
an explicit `--version-id`, but omit `--effective-on`. This only tests retrieval and
answer grounding for that exact snapshot. It **does not establish** that the version
was legally effective on any date; do not use these outputs as current-law advice or
as final dissertation results.

## Verification boundary

The verifier checks that every cited chunk was retrieved in scope and that each quote
is present verbatim apart from whitespace. It then asks the configured LLM whether
each claim follows from its quotes, failing closed on a negative or malformed verdict.
This second-model-call check is **not** a proof of legal or logical entailment. It can
share the generator's biases; expert evaluation and adversarial tests are required.
The returned answer is assembled from claim texts and citation markers so that no
uncited free-text summary is appended. All source text is presented as untrusted data.

The current strict MVP requires a working LLM server, PostgreSQL, ingested dated
versions, and their embeddings for hybrid retrieval. If embeddings are not yet indexed,
use a separate `AgentSpec` with `retrieval_strategy: bm25` and `reranker_enabled: false`
for a lexical smoke test; keep its spec snapshot in the run record.
