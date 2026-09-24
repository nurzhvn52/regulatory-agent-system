import json
from datetime import date
from uuid import UUID

import pytest

from regagent.application.agent_generation.runtime import AnswerStatus, CitedQAAgent
from regagent.application.ports import LLMRequest, LLMResponse
from regagent.domain.agents import AgentSpec, TaskType, VerificationPolicy
from regagent.domain.documents import Language
from regagent.domain.retrieval import (
    RetrievalFilters,
    RetrievalHit,
    RetrievalQuery,
    RetrievalStrategy,
)

SIGNATURE = "a" * 64
TEXT = "Статья 5. Организация обязана хранить документы в течение пяти лет."
CHUNK_ID = UUID(int=1)


class FakeRetriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.calls: list[RetrievalQuery] = []

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalHit]:
        self.calls.append(query)
        return self.hits


class FakeLLM:
    model_name = "fake-test-model"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            text=self.outputs.pop(0), model=self.model_name, input_tokens=10, output_tokens=5
        )


def _hit() -> RetrievalHit:
    return RetrievalHit(
        chunk_id=UUID(int=1),
        document_id=UUID(int=2),
        version_id=UUID(int=3),
        text=TEXT,
        score=1.0,
        rank=1,
        article="5",
        title="Тестовый акт",
        language=Language.RU,
        source_url="https://example.test/act",
    )


def _spec(*, entailment: bool = True, effective: bool = True) -> AgentSpec:
    return AgentSpec(
        name="test-qa",
        task_type=TaskType.CITED_QA,
        retrieval_strategy=RetrievalStrategy.BM25,
        reranker_enabled=False,
        top_k=3,
        verification=VerificationPolicy(
            entailment_check=entailment, require_effective_version=effective
        ),
        output_schema="cited_answer_v1",
    )


def _filters(*, effective: bool = True) -> RetrievalFilters:
    return RetrievalFilters(
        pipeline_signature=SIGNATURE,
        languages=(Language.RU,),
        effective_on=date(2026, 1, 1) if effective else None,
        version_ids=(UUID(int=3),),
    )


def _draft(
    chunk_id: UUID = CHUNK_ID, quote: str = "хранить документы в течение пяти лет"
) -> str:
    return json.dumps(
        {
            "claims": [
                {
                    "text": "Организация обязана хранить документы пять лет.",
                    "citations": [{"chunk_id": str(chunk_id), "quote": quote}],
                }
            ],
            "refusal_reason": None,
        },
        ensure_ascii=False,
    )


@pytest.mark.asyncio
async def test_answer_contains_verified_quote_and_provenance() -> None:
    retriever = FakeRetriever([_hit()])
    llm = FakeLLM([_draft(), '{"supported": true, "reason": "quoted duty"}'])

    result = await CitedQAAgent(retriever, llm).answer(_spec(), "Каков срок хранения?", _filters())

    assert result.status is AnswerStatus.ANSWERED
    assert result.sources[0].source_url == "https://example.test/act"
    assert result.sources[0].version_id == UUID(int=3)
    assert str(UUID(int=1)) in (result.answer or "")
    assert result.input_tokens == 20
    assert retriever.calls[0].top_k == 3


@pytest.mark.asyncio
async def test_refuses_when_retrieval_is_empty_without_calling_llm() -> None:
    llm = FakeLLM([])
    result = await CitedQAAgent(FakeRetriever([]), llm).answer(
        _spec(), "Вопрос", _filters()
    )
    assert result.status is AnswerStatus.REFUSED
    assert result.answer is None
    assert llm.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "draft",
    [_draft(UUID(int=99)), _draft(quote="Это предложение отсутствует в источнике")],
)
async def test_refuses_hallucinated_citations(draft: str) -> None:
    llm = FakeLLM([draft])
    result = await CitedQAAgent(FakeRetriever([_hit()]), llm).answer(
        _spec(), "Вопрос", _filters()
    )
    assert result.status is AnswerStatus.REFUSED
    assert result.sources == ()
    assert len(llm.requests) == 1


@pytest.mark.asyncio
async def test_refuses_when_entailment_check_fails() -> None:
    llm = FakeLLM([_draft(), '{"supported": false, "reason": "not established"}'])
    result = await CitedQAAgent(FakeRetriever([_hit()]), llm).answer(
        _spec(), "Вопрос", _filters()
    )
    assert result.status is AnswerStatus.REFUSED
    assert result.sources == ()


@pytest.mark.asyncio
async def test_refuses_malformed_generation() -> None:
    result = await CitedQAAgent(FakeRetriever([_hit()]), FakeLLM(["not JSON"])).answer(
        _spec(), "Вопрос", _filters()
    )
    assert result.status is AnswerStatus.REFUSED


@pytest.mark.asyncio
async def test_refuses_extra_claims_outside_pilot_schema() -> None:
    payload = json.loads(_draft())
    payload["claims"].append(payload["claims"][0])
    result = await CitedQAAgent(
        FakeRetriever([_hit()]), FakeLLM([json.dumps(payload, ensure_ascii=False)])
    ).answer(_spec(), "Вопрос", _filters())
    assert result.status is AnswerStatus.REFUSED


@pytest.mark.asyncio
async def test_rejects_missing_date_before_retrieval() -> None:
    retriever = FakeRetriever([_hit()])
    with pytest.raises(ValueError, match="effective-on"):
        await CitedQAAgent(retriever, FakeLLM([])).answer(
            _spec(), "Вопрос", _filters(effective=False)
        )
    assert retriever.calls == []


@pytest.mark.asyncio
async def test_rejects_unpinned_version_before_retrieval() -> None:
    retriever = FakeRetriever([_hit()])
    with pytest.raises(ValueError, match="version-id"):
        await CitedQAAgent(retriever, FakeLLM([])).answer(
            _spec(), "Вопрос", _filters().model_copy(update={"version_ids": ()})
        )
    assert retriever.calls == []


@pytest.mark.asyncio
async def test_pilot_allows_unknown_effective_date_but_only_pinned_version() -> None:
    retriever = FakeRetriever([_hit()])
    llm = FakeLLM([_draft(), '{"supported": true, "reason": "quoted duty"}'])
    result = await CitedQAAgent(retriever, llm).answer(
        _spec(effective=False), "Вопрос", _filters(effective=False)
    )
    assert result.status is AnswerStatus.ANSWERED
    assert retriever.calls[0].filters.version_ids == (UUID(int=3),)


@pytest.mark.asyncio
async def test_rejects_unimplemented_reranking() -> None:
    spec = AgentSpec(name="old", task_type=TaskType.CITED_QA, output_schema="cited_answer_v1")
    with pytest.raises(ValueError, match="Reranking"):
        await CitedQAAgent(FakeRetriever([]), FakeLLM([])).answer(spec, "Вопрос", _filters())
