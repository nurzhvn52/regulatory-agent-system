"""Internal models for a reproducible retrieval corpus and embedding index."""

from datetime import date
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from regagent.domain.documents import ActType, Language
from regagent.domain.retrieval import RetrievalFilters, RetrievalHit


class CorpusChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: UUID
    document_id: UUID
    version_id: UUID
    text: str
    title: str
    language: Language
    act_type: ActType
    article: str | None = None
    paragraph: str | None = None
    source_url: str
    ordinal: int = Field(ge=0)
    source_content_hash: str = ""
    articles: tuple[str, ...] = ()
    chunking_config: dict[str, Any] = Field(default_factory=dict)

    def to_hit(self, *, score: float, rank: int) -> RetrievalHit:
        return RetrievalHit(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            version_id=self.version_id,
            text=self.text,
            score=score,
            rank=rank,
            article=self.article,
            paragraph=self.paragraph,
            title=self.title,
            language=self.language,
            source_url=self.source_url,
        )


class EmbeddingModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    dimensions: int | None = Field(default=None, ge=1)
    normalized: bool = True
    config: dict[str, Any] = Field(default_factory=dict)


class EmbeddingVector(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: UUID
    values: list[float] = Field(min_length=1)


class IndexingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: EmbeddingModelSpec
    dimensions: int = Field(ge=1)
    discovered_chunks: int = Field(ge=0)
    indexed_chunks: int = Field(ge=0)
    skipped_chunks: int = Field(ge=0)


class RetrievalEvaluationCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_chunk_ids: frozenset[UUID] = Field(min_length=1)
    filters: RetrievalFilters


class RetrievalMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    recall_at_k: dict[int, float]
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: dict[int, float]


class EvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    metrics: RetrievalMetrics
    hits: tuple[RetrievalHit, ...]


class AggregateEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_count: int = Field(ge=0)
    mean_recall_at_k: dict[int, float]
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    mean_ndcg_at_k: dict[int, float]
    evaluated_at: date = Field(default_factory=date.today)
    cases: tuple[EvaluationResult, ...]
