"""CLI composition and UTF-8 artifacts for reproducible retrieval experiments."""

import csv
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

from regagent.application.ports import EmbeddingProvider, Retriever
from regagent.application.retrieval.benchmark import (
    BenchmarkDataset,
    ResolvedBenchmark,
    resolve_benchmark,
)
from regagent.application.retrieval.bm25 import BM25Index
from regagent.application.retrieval.evaluation import RetrievalEvaluator
from regagent.application.retrieval.models import AggregateEvaluation
from regagent.application.retrieval.ports import RetrievalRepository
from regagent.application.retrieval.services import DenseRetriever, ReciprocalRankFusionRetriever
from regagent.domain.retrieval import RetrievalFilters, RetrievalHit, RetrievalQuery


class _SnapshotBM25:
    def __init__(self, resolved: ResolvedBenchmark) -> None:
        self._indexes = {
            filters: BM25Index(
                [chunk for chunk in resolved.chunks if chunk.version_id in filters.version_ids]
            )
            for filters in {case.filters for case in resolved.cases}
        }

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalHit]:
        return [
            chunk.to_hit(score=score, rank=rank)
            for rank, (chunk, score) in enumerate(
                self._indexes[query.filters].search(query.text, query.top_k), start=1
            )
        ]


async def run_benchmark(
    dataset_path: Path,
    output_dir: Path,
    scope: RetrievalFilters,
    repository: RetrievalRepository,
    provider: EmbeddingProvider,
    *,
    strategies: tuple[str, ...] = ("bm25", "dense", "hybrid"),
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    allow_draft: bool = False,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("Output directory already exists; choose a new run directory")
    if not strategies or len(set(strategies)) != len(strategies):
        raise ValueError("Choose one or more distinct retrieval strategies")
    if set(strategies) - {"bm25", "dense", "hybrid"}:
        raise ValueError("Unsupported retrieval strategy")
    if not k_values or any(k < 1 or k > 100 for k in k_values):
        raise ValueError("k must be between 1 and 100")
    raw = dataset_path.read_bytes()
    dataset = BenchmarkDataset.model_validate(yaml.safe_load(raw.decode("utf-8")))
    resolved = resolve_benchmark(
        dataset,
        await repository.list_chunks(scope),
        scope,
        allow_draft=allow_draft,
    )
    # Fail before any measurement if a dense channel would search a smaller corpus.
    if set(strategies) & {"dense", "hybrid"}:
        for filters in {case.filters for case in resolved.cases}:
            missing = await repository.list_unembedded_chunks(filters, provider.model_spec)
            if missing:
                raise ValueError(f"Dense index incomplete: {len(missing)} missing chunks")
    lexical = _SnapshotBM25(resolved)
    dense = DenseRetriever(repository, provider)
    retrievers: dict[str, Retriever] = {
        "bm25": lexical,
        "dense": dense,
        "hybrid": ReciprocalRankFusionRetriever((lexical, dense)),
    }
    evaluations = {}
    for strategy in strategies:
        evaluations[strategy] = await RetrievalEvaluator(
            retrievers[strategy],
            k_values=k_values,
        ).evaluate(resolved.cases)
    # Refuse to publish measurements if ingestion changed the corpus during the run.
    after = resolve_benchmark(
        dataset,
        await repository.list_chunks(scope),
        scope,
        allow_draft=allow_draft,
    )
    if after.corpus_sha256 != resolved.corpus_sha256:
        raise ValueError("Corpus changed during evaluation; rerun on a stable snapshot")
    summary = _summaries(dataset, evaluations, k_values)
    manifest = {
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "corpus_sha256": resolved.corpus_sha256,
        "status": "pilot_unreviewed" if resolved.contains_drafts else "reviewed",
        "created_at": datetime.now(UTC).isoformat(),
        "case_count": len(resolved.cases),
        "chunk_count": len(resolved.chunks),
        "scope": scope.model_dump(mode="json"),
        "embedding_model": provider.model_spec.model_dump(mode="json"),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in (
                "sentence-transformers",
                "torch",
                "pgvector",
                "sqlalchemy",
            )
        },
        "code": _code_provenance(),
        "k_values": k_values,
        "bm25": {"k1": 1.5, "b": 0.75, "tokenizer": "unicode_no_stemming_v1"},
        "hybrid": {"rrf_constant": 60, "candidate_multiplier": 4, "candidate_cap": 100},
        "mrr_cutoff": max(k_values),
        "relevance": "binary_chunk_overlap_with_labeled_article",
        "stable_chunk_keys": {str(key): value for key, value in resolved.stable_chunk_keys.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "dataset.yaml").write_bytes(raw)
    (output_dir / "resolved_cases.json").write_text(
        json.dumps(
            [case.model_dump(mode="json") for case in resolved.cases], ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "results.json").write_text(
        json.dumps(
            {name: result.model_dump(mode="json") for name, result in evaluations.items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    _write_review(output_dir / "review.md", dataset, resolved)
    return {
        "output_dir": str(output_dir.resolve()),
        "status": manifest["status"],
        "cases": len(resolved.cases),
        "chunks": len(resolved.chunks),
        "summary": summary,
    }


def _summaries(
    dataset: BenchmarkDataset,
    evaluations: dict[str, AggregateEvaluation],
    k_values: tuple[int, ...],
) -> list[dict[str, Any]]:
    languages = {case.case_id: case.language.value for case in dataset.cases}
    rows = []
    for strategy, result in evaluations.items():
        for language in ("all", *sorted(set(languages.values()))):
            cases = [
                case
                for case in result.cases
                if language == "all" or languages[case.case_id] == language
            ]
            row: dict[str, Any] = {
                "strategy": strategy,
                "language": language,
                "cases": len(cases),
                f"mrr@{max(k_values)}": fmean(c.metrics.reciprocal_rank for c in cases),
            }
            for k in k_values:
                row[f"recall@{k}"] = fmean(c.metrics.recall_at_k[k] for c in cases)
                row[f"ndcg@{k}"] = fmean(c.metrics.ndcg_at_k[k] for c in cases)
            rows.append(row)
    return rows


def _write_review(path: Path, dataset: BenchmarkDataset, resolved: ResolvedBenchmark) -> None:
    lines = [
        f"# Review: {dataset.dataset_id}",
        "",
        "Draft relevance labels require human review against the pinned source version.",
        "All overlapping chunks count as relevant; this is article-level coarse labeling.",
        "Correct query/labels in dataset YAML, then set reviewer, reviewed_at and review_status.",
        "Keep translation pairs together when making train/dev/test splits.",
        "",
    ]
    cases = {case.case_id: case for case in resolved.cases}
    for case in dataset.cases:
        lines.extend(
            [f"## {case.case_id} ({case.language}, {case.review_status})", "", case.query, ""]
        )
        lines.extend(
            f"- {label.source_id}: article {label.article}" for label in case.relevant_articles
        )
        for chunk in resolved.chunks:
            if chunk.chunk_id in cases[case.case_id].relevant_chunk_ids:
                lines.extend(
                    [
                        "",
                        f"Source: {chunk.source_url}",
                        f"Version hash: {chunk.source_content_hash}",
                        f"Chunk key: {resolved.stable_chunk_keys[chunk.chunk_id]}",
                        "",
                        *[f"> {line}" for line in chunk.text.splitlines()],
                        "",
                    ]
                )
    path.write_text("\n".join(lines), encoding="utf-8")


def _code_provenance() -> dict[str, str | bool]:
    project_root = Path(__file__).resolve().parents[3]
    try:

        def git(*args: str) -> bytes:
            return subprocess.run(
                ["git", *args],
                cwd=project_root,
                capture_output=True,
                check=True,
            ).stdout

        files = [*sorted((project_root / "src").rglob("*.py")), project_root / "poetry.lock"]
        fingerprint = hashlib.sha256()
        for path in files:
            fingerprint.update(path.relative_to(project_root).as_posix().encode())
            fingerprint.update(path.read_bytes())
        return {
            "commit": git("rev-parse", "HEAD").decode().strip(),
            "dirty": bool(git("status", "--porcelain")),
            "source_sha256": fingerprint.hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable"}
