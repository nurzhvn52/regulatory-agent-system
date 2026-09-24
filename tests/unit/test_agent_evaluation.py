import csv
import json
from datetime import date
from pathlib import Path
from uuid import UUID

import pytest

from regagent.application.agent_generation.evaluation import AgentEvaluationRecord, technical_checks
from regagent.application.agent_generation.human_review import HumanJudgment, score_judgments
from regagent.application.agent_generation.runtime import (
    AnswerClaim,
    AnswerStatus,
    CitedAnswer,
    CitedSource,
    EvidenceQuote,
)
from regagent.application.ports import LLMRequest, LLMResponse
from regagent.application.retrieval.benchmark import (
    ArticleLabel,
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkSource,
)
from regagent.application.retrieval.models import CorpusChunk
from regagent.domain.agents import AgentSpec, TaskType, VerificationPolicy
from regagent.domain.documents import ActType, Language
from regagent.domain.retrieval import RetrievalFilters, RetrievalStrategy
from regagent.infrastructure.agent_evaluation_runner import run_agent_evaluation
from regagent.infrastructure.agent_review_scoring import score_agent_review


def _chunk() -> CorpusChunk:
    return CorpusChunk(
        chunk_id=UUID(int=1),
        document_id=UUID(int=2),
        version_id=UUID(int=3),
        text="Статья 5. Организация обязана хранить документы в течение пяти лет.",
        title="Тестовый акт",
        language=Language.RU,
        act_type=ActType.LAW,
        article="5",
        articles=("5",),
        source_url="https://example.test/act",
        ordinal=0,
        source_content_hash="b" * 64,
    )


def _dataset() -> BenchmarkDataset:
    return BenchmarkDataset(
        dataset_id="qa-test",
        description="pilot",
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
                query="Сколько лет организация хранит документы?",
                language=Language.RU,
                relevant_articles=(ArticleLabel(source_id="law", article="5"),),
            ),
        ),
    )


def _spec() -> AgentSpec:
    return AgentSpec(
        name="test-qa",
        task_type=TaskType.CITED_QA,
        retrieval_strategy=RetrievalStrategy.BM25,
        reranker_enabled=False,
        top_k=1,
        verification=VerificationPolicy(require_effective_version=False),
        output_schema="cited_answer_v1",
    )


class _Repository:
    async def list_chunks(self, filters: RetrievalFilters) -> list[CorpusChunk]:
        return [_chunk()]


class _Runs:
    def __init__(self) -> None:
        self.finished: list[UUID] = []
        self.failed: list[UUID] = []

    async def start(
        self,
        spec: AgentSpec,
        question: str,
        filters: RetrievalFilters,
        experiment_id: str | None = None,
    ) -> UUID:
        assert filters.version_ids == (UUID(int=3),)
        assert experiment_id is not None
        return UUID(int=9)

    async def finish(self, run_id: UUID, result: CitedAnswer) -> None:
        self.finished.append(run_id)

    async def fail(self, run_id: UUID, error: Exception) -> None:
        self.failed.append(run_id)


class _LLM:
    model_name = "fake-model"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if "claim" in request.user_prompt:
            payload = {"supported": True, "reason": "direct quotation"}
        else:
            payload = {
                "claims": [
                    {
                        "text": "Организация хранит документы пять лет.",
                        "citations": [
                            {
                                "chunk_id": str(_chunk().chunk_id),
                                "quote": "хранить документы в течение пяти лет",
                            }
                        ],
                    }
                ],
                "refusal_reason": None,
            }
        return LLMResponse(
            text=json.dumps(payload, ensure_ascii=False),
            model=self.model_name,
            input_tokens=10,
            output_tokens=5,
        )


@pytest.mark.asyncio
async def test_agent_runner_writes_ungraded_review_packet(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text(_dataset().model_dump_json(), encoding="utf-8")
    output = tmp_path / "run"
    runs = _Runs()
    result = await run_agent_evaluation(
        dataset_path,
        output,
        RetrievalFilters(pipeline_signature="a" * 64),
        _spec(),
        _Repository(),
        None,
        _LLM(),
        runs,
        allow_draft=True,
    )
    assert result["status"] == "complete_pilot_ungraded"
    assert runs.finished == [UUID(int=9)]
    assert result["summary"][0]["citation_integrity_rate_answered"] == 1.0
    assert result["summary"][0]["human_answer_accuracy"] is None
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["answer_labels"] == "not_provided"
    assert manifest["case_count"] == 1
    with (output / "human_review.csv").open(encoding="utf-8-sig", newline="") as stream:
        review = list(csv.DictReader(stream))
    assert review[0]["expected_answerable"] == ""
    assert review[0]["reviewer"] == ""
    assert "хранить документы" in (output / "review.md").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        await run_agent_evaluation(
            dataset_path,
            output,
            RetrievalFilters(pipeline_signature="a" * 64),
            _spec(),
            _Repository(),
            None,
            _LLM(),
            runs,
            allow_draft=True,
        )


@pytest.mark.asyncio
async def test_runner_requires_explicit_draft_permission(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text(_dataset().model_dump_json(), encoding="utf-8")
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="unreviewed"):
        await run_agent_evaluation(
            dataset_path,
            output,
            RetrievalFilters(pipeline_signature="a" * 64),
            _spec(),
            _Repository(),
            None,
            _LLM(),
            _Runs(),
        )
    assert not output.exists()


@pytest.mark.asyncio
async def test_human_review_scoring_requires_complete_signed_labels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text(_dataset().model_dump_json(), encoding="utf-8")
    output = tmp_path / "run"
    await run_agent_evaluation(
        dataset_path,
        output,
        RetrievalFilters(pipeline_signature="a" * 64),
        _spec(),
        _Repository(),
        None,
        _LLM(),
        _Runs(),
        allow_draft=True,
    )
    score_path = tmp_path / "human_score.json"
    with pytest.raises(ValueError, match="unreviewed labels"):
        score_agent_review(output, score_path)
    with pytest.raises(ValueError, match="reviewer is required"):
        score_agent_review(output, score_path, allow_draft=True)
    review_path = output / "human_review.csv"
    with review_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0].update(
        {
            "expected_answerable": "true",
            "answer_correct": "true",
            "citation_supports": "true",
            "version_correct": "true",
            "reviewer": "unit-test-fixture",
            "reviewed_at": "2026-09-24",
        }
    )
    with review_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = score_agent_review(output, score_path, allow_draft=True)
    assert report["status"] == "pilot_reviewed_answers_unreviewed_questions"
    assert report["summary"][0]["system_success_rate"] == 1.0
    assert score_path.exists()
    with pytest.raises(ValueError, match="already exists"):
        score_agent_review(output, score_path, allow_draft=True)


def test_citation_integrity_rechecks_quote_not_just_agent_status() -> None:
    source = CitedSource(
        chunk_id=_chunk().chunk_id,
        document_id=_chunk().document_id,
        version_id=_chunk().version_id,
        title=_chunk().title,
        article="5",
        paragraph=None,
        source_url=_chunk().source_url,
        quote="несуществующая цитата из документа",
    )
    answer = CitedAnswer(
        status=AnswerStatus.ANSWERED,
        answer="Срок пять лет.",
        claims=(
            AnswerClaim(
                text="Срок пять лет.",
                citations=(EvidenceQuote(chunk_id=_chunk().chunk_id, quote=source.quote),),
            ),
        ),
        sources=(source,),
        retrieved_chunk_ids=(_chunk().chunk_id,),
    )
    retrieved, cited, integrity = technical_checks(
        answer, frozenset({_chunk().chunk_id}), [_chunk()]
    )
    assert retrieved and cited
    assert not integrity


def test_human_score_distinguishes_appropriate_and_false_refusal() -> None:
    records = [
        AgentEvaluationRecord(
            case_id=f"q{index}",
            pair_id=f"p{index}",
            language=Language.RU,
            question="Вопрос без ответа?",
            review_status="draft",
            run_id=UUID(int=index),
            latency_ms=1,
            result=CitedAnswer(status=AnswerStatus.REFUSED, reason="Нет доказательств"),
        )
        for index in (1, 2)
    ]
    judgments = [
        HumanJudgment(
            case_id=f"q{index}",
            expected_answerable=answerable,
            answer_correct=None,
            citation_supports=None,
            version_correct=None,
            reviewer="unit-test-fixture",
            reviewed_at=date(2026, 9, 24),
            notes="",
        )
        for index, answerable in ((1, False), (2, True))
    ]
    summary = score_judgments(records, judgments)[0]
    assert summary["system_success_rate"] == 0.5
    assert summary["appropriate_refusal_rate_among_refusals"] == 0.5
    assert summary["false_refusal_rate_among_answerable"] == 1.0
