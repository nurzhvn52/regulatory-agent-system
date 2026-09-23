import json
from pathlib import Path
from uuid import UUID

import pytest
import yaml
from pydantic import ValidationError

from regagent.application.retrieval.benchmark import (
    ArticleLabel,
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkSource,
    resolve_benchmark,
)
from regagent.application.retrieval.evaluation import evaluate_ranking
from regagent.application.retrieval.models import CorpusChunk, EmbeddingModelSpec
from regagent.domain.documents import ActType, Language
from regagent.domain.retrieval import RetrievalFilters, RetrievalHit
from regagent.infrastructure.benchmark_runner import run_benchmark

SCOPE = RetrievalFilters(pipeline_signature="a" * 64)


def _chunk() -> CorpusChunk:
    return CorpusChunk(
        chunk_id=UUID(int=1),
        document_id=UUID(int=2),
        version_id=UUID(int=3),
        text="Срок действия нормативного акта",
        title="Test source",
        language=Language.RU,
        act_type=ActType.LAW,
        ordinal=0,
        source_url="https://example.test/law",
        source_content_hash="b" * 64,
        article="44",
        articles=("44", "45"),
    )


def _dataset() -> BenchmarkDataset:
    return BenchmarkDataset(
        dataset_id="test",
        description="test",
        sources=(
            BenchmarkSource(
                source_id="law",
                language=Language.RU,
                source_url=_chunk().source_url,
                content_hash="b" * 64,
            ),
        ),
        cases=(
            BenchmarkCase(
                case_id="q1",
                pair_id="p1",
                query="срок действия",
                language=Language.RU,
                relevant_articles=(ArticleLabel(source_id="law", article="45"),),
            ),
        ),
    )


def test_draft_requires_explicit_pilot_flag() -> None:
    with pytest.raises(ValueError, match="unreviewed"):
        resolve_benchmark(_dataset(), [_chunk()], SCOPE)


def test_qrels_match_all_source_sections_and_pin_exact_version() -> None:
    # Article 45 is not the anchor: fixed windows may cover several articles.
    resolved = resolve_benchmark(_dataset(), [_chunk()], SCOPE, allow_draft=True)
    assert resolved.cases[0].relevant_chunk_ids == frozenset([UUID(int=1)])
    assert resolved.cases[0].filters.version_ids == (UUID(int=3),)


def test_portable_qrels_and_snapshot_hash_ignore_database_uuids() -> None:
    first = resolve_benchmark(_dataset(), [_chunk()], SCOPE, allow_draft=True)
    recreated = _chunk().model_copy(
        update={
            "chunk_id": UUID(int=101),
            "document_id": UUID(int=102),
            "version_id": UUID(int=103),
        }
    )
    second = resolve_benchmark(_dataset(), [recreated], SCOPE, allow_draft=True)
    assert second.corpus_sha256 == first.corpus_sha256
    assert second.cases[0].relevant_chunk_ids == frozenset([UUID(int=101)])


def test_changed_text_changes_snapshot_hash() -> None:
    first = resolve_benchmark(_dataset(), [_chunk()], SCOPE, allow_draft=True)
    second = resolve_benchmark(
        _dataset(),
        [_chunk().model_copy(update={"text": "Changed content"})],
        SCOPE,
        allow_draft=True,
    )
    assert second.corpus_sha256 != first.corpus_sha256


@pytest.mark.parametrize(
    "update, message",
    [
        ({"source_content_hash": "c" * 64}, "Missing pinned"),
        ({"articles": ("10",)}, "Unresolved article"),
    ],
)
def test_missing_source_or_article_is_not_silently_dropped(
    update: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_benchmark(_dataset(), [_chunk().model_copy(update=update)], SCOPE, allow_draft=True)


def test_reviewed_labels_require_review_metadata() -> None:
    data = _dataset().cases[0].model_dump()
    data["review_status"] = "reviewed"
    with pytest.raises(ValidationError, match="reviewer"):
        BenchmarkCase.model_validate(data)


def test_duplicate_cases_are_rejected() -> None:
    data = _dataset().model_dump()
    data["cases"] = [data["cases"][0], data["cases"][0]]
    with pytest.raises(ValidationError, match="Case IDs"):
        BenchmarkDataset.model_validate(data)


def test_duplicate_hits_cannot_inflate_ndcg() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_ranking([UUID(int=1), UUID(int=1)], [UUID(int=1)])


def test_perfect_and_empty_rankings() -> None:
    assert evaluate_ranking([UUID(int=1)], [UUID(int=1)]).ndcg_at_k[10] == 1.0
    empty = evaluate_ranking([], [UUID(int=1)])
    assert empty.reciprocal_rank == 0.0
    assert empty.recall_at_k[10] == 0.0


def test_shipped_pilot_has_sixty_unreviewed_paired_cases() -> None:
    path = Path(__file__).parents[2] / "data/evaluation/drafts/legal_acts_ru_kk.yaml"
    dataset = BenchmarkDataset.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert len(dataset.cases) == 60
    assert all(case.review_status == "draft" for case in dataset.cases)
    for pair in {case.pair_id for case in dataset.cases}:
        assert {c.language for c in dataset.cases if c.pair_id == pair} == {
            Language.RU,
            Language.KK,
        }


class _Provider:
    model_spec = EmbeddingModelSpec(name="fake", revision="test")
    dimensions = 2

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class _Repository:
    def __init__(self, *, missing: bool = False) -> None:
        self.missing = missing

    async def list_chunks(self, filters: RetrievalFilters) -> list[CorpusChunk]:
        return [_chunk()]

    async def list_unembedded_chunks(
        self,
        filters: RetrievalFilters,
        model: EmbeddingModelSpec,
    ) -> list[CorpusChunk]:
        return [_chunk()] if self.missing else []

    async def dense_search(
        self,
        filters: RetrievalFilters,
        model: EmbeddingModelSpec,
        query_vector: list[float],
        top_k: int,
    ) -> list[RetrievalHit]:
        assert filters.version_ids == (UUID(int=3),)
        return [_chunk().to_hit(score=1, rank=1)]


@pytest.mark.asyncio
async def test_runner_writes_three_strategies_and_prevents_overwrite(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(_dataset().model_dump_json(), encoding="utf-8")
    output = tmp_path / "run"
    result = await run_benchmark(
        dataset_path,
        output,
        SCOPE,
        _Repository(),
        _Provider(),
        allow_draft=True,
    )
    assert result["status"] == "pilot_unreviewed"
    assert {row["strategy"] for row in result["summary"]} == {"bm25", "dense", "hybrid"}
    assert all(row["recall@1"] == 1 for row in result["summary"])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mrr_cutoff"] == 10
    assert "Срок" in (output / "review.md").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        await run_benchmark(
            dataset_path, output, SCOPE, _Repository(), _Provider(), allow_draft=True
        )


@pytest.mark.asyncio
async def test_runner_refuses_incomplete_dense_index(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(_dataset().model_dump_json(), encoding="utf-8")
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="incomplete"):
        await run_benchmark(
            dataset_path,
            output,
            SCOPE,
            _Repository(missing=True),
            _Provider(),
            allow_draft=True,
        )
    assert not output.exists()
