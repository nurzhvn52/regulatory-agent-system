"""Models exchanged by lexical, vector, hybrid, and reranking components."""

from datetime import date
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from regagent.domain.documents import ActType, Language


class RetrievalStrategy(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class RetrievalFilters(BaseModel):
    model_config = ConfigDict(frozen=True)

    languages: tuple[Language, ...] = ()
    act_types: tuple[ActType, ...] = ()
    effective_on: date | None = None
    document_ids: tuple[UUID, ...] = ()


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    top_k: int = Field(default=15, ge=1, le=100)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)


class RetrievalHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: UUID
    document_id: UUID
    version_id: UUID
    text: str
    score: float
    rank: int = Field(ge=1)
    article: str | None = None
    paragraph: str | None = None
    source_url: str

