"""Reproducible end-to-end cited-QA evaluation with human-review artifacts."""

import csv
import hashlib
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

import yaml

from regagent.application.agent_generation.evaluation import (
    AgentEvaluationRecord,
    summarize_agent_records,
    technical_checks,
)
from regagent.application.agent_generation.runtime import CitedAnswer, CitedQAAgent
from regagent.application.ports import EmbeddingProvider, LLMProvider, Retriever
from regagent.application.retrieval.benchmark import (
    BenchmarkCase,
    BenchmarkDataset,
    ResolvedBenchmark,
    resolve_benchmark,
)
from regagent.application.retrieval.models import RetrievalEvaluationCase
from regagent.application.retrieval.ports import RetrievalRepository
from regagent.application.retrieval.services import DenseRetriever, ReciprocalRankFusionRetriever
from regagent.domain.agents import AgentSpec, TaskType
from regagent.domain.retrieval import RetrievalFilters, RetrievalStrategy
from regagent.infrastructure.benchmark_runner import SnapshotBM25Retriever, _code_provenance


class AgentRunStore(Protocol):
    async def start(
        self,
        spec: AgentSpec,
        question: str,
        filters: RetrievalFilters,
        experiment_id: str | None = None,
    ) -> UUID: ...

    async def finish(self, run_id: UUID, result: CitedAnswer) -> None: ...

    async def fail(self, run_id: UUID, error: Exception) -> None: ...


def _selected_cases(
    dataset: BenchmarkDataset, resolved: ResolvedBenchmark, max_pairs: int | None
) -> tuple[tuple[BenchmarkCase, RetrievalEvaluationCase], ...]:
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("--max-pairs must be positive")
    pair_ids = list(dict.fromkeys(case.pair_id for case in dataset.cases))
    chosen = set(pair_ids[:max_pairs])
    return tuple(
        (case, resolved_case)
        for case, resolved_case in zip(dataset.cases, resolved.cases, strict=True)
        if case.pair_id in chosen
    )


def _retriever(
    spec: AgentSpec,
    resolved: ResolvedBenchmark,
    repository: RetrievalRepository,
    provider: EmbeddingProvider | None,
) -> Retriever:
    lexical = SnapshotBM25Retriever(resolved)
    if spec.retrieval_strategy is RetrievalStrategy.BM25:
        return lexical
    if provider is None:
        raise ValueError("Dense and hybrid retrieval require an embedding provider")
    dense = DenseRetriever(repository, provider)
    if spec.retrieval_strategy is RetrievalStrategy.DENSE:
        return dense
    return ReciprocalRankFusionRetriever((lexical, dense))


async def run_agent_evaluation(
    dataset_path: Path,
    output_dir: Path,
    scope: RetrievalFilters,
    spec: AgentSpec,
    repository: RetrievalRepository,
    provider: EmbeddingProvider | None,
    llm: LLMProvider,
    runs: AgentRunStore,
    *,
    allow_draft: bool = False,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("Output directory already exists; choose a new run directory")
    if spec.task_type is not TaskType.CITED_QA or spec.output_schema != "cited_answer_v1":
        raise ValueError("Agent evaluation supports only cited_qa with cited_answer_v1")
    if spec.reranker_enabled or spec.retrieval_strategy is RetrievalStrategy.HYBRID_RERANK:
        raise ValueError("Reranking is not implemented in the cited QA runtime")
    if spec.verification.require_effective_version and scope.effective_on is None:
        raise ValueError("This agent requires an effective date; current pilot sources lack one")
    raw = dataset_path.read_bytes()
    dataset = BenchmarkDataset.model_validate(yaml.safe_load(raw.decode("utf-8")))
    resolved = resolve_benchmark(
        dataset, await repository.list_chunks(scope), scope, allow_draft=allow_draft
    )
    selected = _selected_cases(dataset, resolved, max_pairs)
    if any(case.language not in spec.languages for case, _ in selected):
        raise ValueError("Dataset language is not enabled by the agent specification")
    if spec.retrieval_strategy is not RetrievalStrategy.BM25:
        if provider is None:
            raise ValueError("Dense and hybrid retrieval require an embedding provider")
        for filters in {resolved_case.filters for _, resolved_case in selected}:
            missing = await repository.list_unembedded_chunks(filters, provider.model_spec)
            if missing:
                raise ValueError(f"Dense index incomplete: {len(missing)} missing chunks")
    agent = CitedQAAgent(_retriever(spec, resolved, repository, provider), llm)
    experiment_id = f"agent-eval-{uuid4()}"
    manifest: dict[str, Any] = {
        "experiment_id": experiment_id,
        "status": "running",
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "corpus_sha256": resolved.corpus_sha256,
        "retrieval_labels": "pilot_unreviewed" if any(
            case.review_status == "draft" for case, _ in selected
        ) else "reviewed",
        "answer_labels": "not_provided",
        "created_at": datetime.now(UTC).isoformat(),
        "case_count": len(selected),
        "pair_count": len({case.pair_id for case, _ in selected}),
        "corpus_chunk_count": len(resolved.chunks),
        "scope": scope.model_dump(mode="json"),
        "agent_spec": spec.model_dump(mode="json"),
        "llm_model": llm.model_name,
        "embedding_model": provider.model_spec.model_dump(mode="json") if provider else None,
        "code": _code_provenance(),
        "notes": [
            "Article overlap is a coarse retrieval proxy, not answer correctness.",
            "Citation integrity verifies exact text and provenance, not legal entailment.",
            "Human answer accuracy remains unavailable until expert review.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "dataset.yaml").write_bytes(raw)
    (output_dir / "resolved_cases.json").write_text(
        json.dumps(
            [case.model_dump(mode="json") for _, case in selected],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    records: list[AgentEvaluationRecord] = []
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for case, resolved_case in selected:
            run_id = await runs.start(spec, case.query, resolved_case.filters, experiment_id)
            started = time.perf_counter()
            try:
                result = await agent.answer(spec, case.query, resolved_case.filters)
                elapsed = (time.perf_counter() - started) * 1000
                await runs.finish(run_id, result)
                retrieved, cited, integrity = technical_checks(
                    result, resolved_case.relevant_chunk_ids, resolved.chunks
                )
                record = AgentEvaluationRecord(
                    case_id=case.case_id,
                    pair_id=case.pair_id,
                    language=case.language,
                    question=case.query,
                    review_status=case.review_status,
                    run_id=run_id,
                    latency_ms=elapsed,
                    result=result,
                    retrieved_article_overlap=retrieved,
                    cited_article_overlap=cited,
                    citation_integrity=integrity,
                )
            except Exception as error:
                elapsed = (time.perf_counter() - started) * 1000
                await runs.fail(run_id, error)
                record = AgentEvaluationRecord(
                    case_id=case.case_id,
                    pair_id=case.pair_id,
                    language=case.language,
                    question=case.query,
                    review_status=case.review_status,
                    run_id=run_id,
                    latency_ms=elapsed,
                    error=f"{type(error).__name__}: {error}",
                )
            records.append(record)
            stream.write(record.model_dump_json() + "\n")
            stream.flush()
    after = resolve_benchmark(
        dataset, await repository.list_chunks(scope), scope, allow_draft=allow_draft
    )
    if after.corpus_sha256 != resolved.corpus_sha256:
        manifest["status"] = "invalid_corpus_changed"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise ValueError("Corpus changed during agent evaluation; results are invalid")
    summary = summarize_agent_records(records)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    _write_review_artifacts(output_dir, records, resolved, selected)
    manifest["results_sha256"] = hashlib.sha256(
        (output_dir / "results.jsonl").read_bytes()
    ).hexdigest()
    manifest["status"] = (
        "complete_pilot_ungraded"
        if manifest["retrieval_labels"] == "pilot_unreviewed"
        else "complete_ungraded"
    )
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_dir": str(output_dir.resolve()),
        "status": manifest["status"],
        "experiment_id": experiment_id,
        "summary": summary,
    }


def _write_review_artifacts(
    output_dir: Path,
    records: Sequence[AgentEvaluationRecord],
    resolved: ResolvedBenchmark,
    selected: Sequence[tuple[BenchmarkCase, RetrievalEvaluationCase]],
) -> None:
    columns = [
        "case_id", "pair_id", "language", "question", "agent_status", "agent_answer",
        "expected_answerable", "answer_correct", "citation_supports", "version_correct",
        "reviewer", "reviewed_at", "notes",
    ]
    with (output_dir / "human_review.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "case_id": record.case_id,
                    "pair_id": record.pair_id,
                    "language": record.language.value,
                    "question": record.question,
                    "agent_status": record.status,
                    "agent_answer": (record.result.answer or "") if record.result else "",
                }
            )
    by_id = {chunk.chunk_id: chunk for chunk in resolved.chunks}
    by_case = {case.case_id: (case, resolved_case) for case, resolved_case in selected}
    lines = [
        "# Проверка ответов агента", "",
        "Это пилотный набор. Статья из черновой разметки — ориентир, а не эталон ответа.",
        "Заполните human_review.csv после чтения исходной версии акта и найденного фрагмента.",
        "Поля expected_answerable, answer_correct, citation_supports, version_correct: true/false.",
        "Для отказа answer_correct и citation_supports оставьте пустыми;",
        "укажите expected_answerable.",
        "Укажите reviewer, reviewed_at (YYYY-MM-DD), notes. Автоматическая проверка цитаты",
        "не подтверждает юридическую верность вывода.", "",
    ]
    for record in records:
        case, resolved_case = by_case[record.case_id]
        lines.extend(
            [f"## {record.case_id} ({record.language.value})", "", f"Вопрос: {record.question}", ""]
        )
        lines.append(
            "Черновая разметка: "
            + ", ".join(
                f"{label.source_id}, статья {label.article}" for label in case.relevant_articles
            )
        )
        lines.append(f"Статус: {record.status}; run_id: {record.run_id}")
        if record.result:
            lines.append(f"Ответ: {record.result.answer or record.result.reason or '—'}")
            lines.append(f"Цитаты: {len(record.result.sources)}")
            for source in record.result.sources:
                lines.extend(
                    [
                        "", f"Источник: {source.source_url}",
                        f"Версия: {source.version_id}; статья: {source.article or '—'}",
                        f"> {source.quote}",
                    ]
                )
            for chunk_id in record.result.retrieved_chunk_ids:
                chunk = by_id.get(chunk_id)
                if chunk:
                    lines.extend(["", f"Найденный фрагмент {chunk_id}:", ""])
                    lines.extend(f"> {line}" for line in chunk.text.splitlines())
        elif record.error:
            lines.append(f"Ошибка выполнения: {record.error}")
        lines.extend(["", "Фрагменты размеченной статьи (черновой ориентир):", ""])
        for chunk_id in sorted(resolved_case.relevant_chunk_ids, key=str):
            chunk = by_id[chunk_id]
            lines.extend(
                [
                    f"Источник: {chunk.source_url}",
                    f"Версия: {chunk.version_id}; фрагмент: {chunk_id}",
                    *[f"> {line}" for line in chunk.text.splitlines()],
                    "",
                ]
            )
        lines.append("")
    (output_dir / "review.md").write_text("\n".join(lines), encoding="utf-8")
