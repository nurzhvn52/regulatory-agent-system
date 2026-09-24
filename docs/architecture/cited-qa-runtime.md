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

### Verified local pilot (2026-09-24)

On the development machine (RTX 4060 Laptop, 8 GiB VRAM), a portable Ollama
`v0.34.4` instance served `qwen3:4b-instruct-2507-q4_K_M` locally. The model digest
reported by `/api/tags` was
`0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`.
Ollama's OpenAI-compatible endpoint supports JSON Schema response formatting; the
adapter passes the Pydantic schema, and the application still validates every field,
chunk ID, quote, and claim after generation.

Install Ollama from the [official Windows instructions](https://docs.ollama.com/windows)
or use its standalone archive, then run
`ollama pull qwen3:4b-instruct-2507-q4_K_M`. The portable binary and downloaded model
used for this pilot are under the Git-ignored `models/` directory, not in the repo.
With Ollama running and this model pulled, execute:

```powershell
$env:REGAGENT_LLM_PROVIDER = "openai_compatible"
$env:REGAGENT_LLM_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
$env:REGAGENT_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
$env:HF_HOME = "D:\hf"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
poetry run regagent answer "Кто осуществляет правовой мониторинг нормативных правовых актов?" `
  --spec configs/agents/cited_qa_pilot.yaml `
  --pipeline-signature 8cc110d69566d224ebebde20f022187eb19cead5992ced909fcb088ed697a88d `
  --version-id 3ad9a4af-b595-436b-9ac9-dbd89db6ba05 `
  --language ru `
  --experiment-id pilot-qwen3-2026-09-24
```

The local pilot run `0b203328-11d4-4931-8c25-cfb48eb2746f` was `answered`, with
an exact quote from article 50. Three earlier exploratory runs were `refused` for
invalid citations, malformed structure, or an unsupported extra claim. These four
runs are development observations, **not** a statistical quality estimate. Database
UUIDs in the example belong to this local corpus and may differ after re-ingestion.
The pilot config retrieves one chunk and the current answer schema allows one claim;
multi-evidence answers remain a later research extension.

As a negative control, the question about the amount of a fine for failing to perform
legal monitoring returned `refused` in run `f64181c9-5aca-4bf1-9d09-fe996e619fc0`:
the retrieved fragment did not state any fine. This is a single observed case, not a
measured refusal rate.

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
