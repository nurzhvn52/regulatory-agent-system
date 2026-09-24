"""Technical QA evaluation, deliberately separate from expert legal grading."""

import re
from collections.abc import Sequence
from statistics import fmean
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from regagent.application.agent_generation.runtime import AnswerStatus, CitedAnswer
from regagent.application.retrieval.models import CorpusChunk
from regagent.domain.documents import Language


class AgentEvaluationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    pair_id: str
    language: Language
    question: str
    review_status: str
    run_id: UUID
    latency_ms: float = Field(ge=0)
    result: CitedAnswer | None = None
    error: str | None = None
    retrieved_article_overlap: bool | None = None
    cited_article_overlap: bool | None = None
    citation_integrity: bool | None = None

    @model_validator(mode="after")
    def check_outcome(self) -> "AgentEvaluationRecord":
        if (self.result is None) == (self.error is None):
            raise ValueError("Exactly one of result or error is required")
        return self

    @property
    def status(self) -> str:
        if self.result is None:
            return "failed"
        return self.result.status.value


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def technical_checks(
    result: CitedAnswer,
    relevant_chunk_ids: frozenset[UUID],
    chunks: Sequence[CorpusChunk],
) -> tuple[bool, bool | None, bool | None]:
    """Return article-overlap proxies and independently rechecked citation integrity."""
    retrieved_overlap = bool(set(result.retrieved_chunk_ids) & relevant_chunk_ids)
    if result.status is AnswerStatus.REFUSED:
        return retrieved_overlap, None, None
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    cited_ids = {source.chunk_id for source in result.sources}
    cited_overlap = bool(cited_ids & relevant_chunk_ids)
    integrity = bool(result.claims and result.sources and result.answer)
    cited_in_claims = {
        (citation.chunk_id, citation.quote)
        for claim in result.claims
        for citation in claim.citations
    }
    integrity = integrity and {
        (source.chunk_id, source.quote) for source in result.sources
    } == cited_in_claims
    for source in result.sources:
        chunk = by_id.get(source.chunk_id)
        integrity = integrity and (
            chunk is not None
            and source.chunk_id in result.retrieved_chunk_ids
            and source.document_id == chunk.document_id
            and source.version_id == chunk.version_id
            and source.source_url == chunk.source_url
            and source.title == chunk.title
            and source.article == chunk.article
            and source.paragraph == chunk.paragraph
            and _normalized(source.quote) in _normalized(chunk.text)
        )
    return retrieved_overlap, cited_overlap, integrity


def summarize_agent_records(records: Sequence[AgentEvaluationRecord]) -> list[dict[str, object]]:
    if not records:
        raise ValueError("At least one case is required")
    rows: list[dict[str, object]] = []
    for language in ("all", *sorted({record.language.value for record in records})):
        selected = [
            record for record in records if language == "all" or record.language.value == language
        ]
        answered = [record for record in selected if record.status == "answered"]
        refused = [record for record in selected if record.status == "refused"]
        failed = [record for record in selected if record.status == "failed"]
        successful = [record for record in selected if record.result is not None]
        rows.append(
            {
                "language": language,
                "cases": len(selected),
                "answered": len(answered),
                "refused": len(refused),
                "failed": len(failed),
                "answer_rate": len(answered) / len(selected),
                "retrieved_article_overlap_rate_proxy": fmean(
                    1.0 if record.retrieved_article_overlap else 0.0 for record in successful
                )
                if successful
                else None,
                "cited_article_overlap_rate_proxy_answered": fmean(
                    1.0 if record.cited_article_overlap else 0.0 for record in answered
                )
                if answered
                else None,
                "citation_integrity_rate_answered": fmean(
                    1.0 if record.citation_integrity else 0.0 for record in answered
                )
                if answered
                else None,
                "mean_latency_ms": fmean(record.latency_ms for record in selected),
                "mean_input_tokens_successful": fmean(
                    record.result.input_tokens for record in successful if record.result is not None
                )
                if successful
                else None,
                "mean_output_tokens_successful": fmean(
                    record.result.output_tokens
                    for record in successful
                    if record.result is not None
                )
                if successful
                else None,
                "human_answer_accuracy": None,
            }
        )
    return rows
