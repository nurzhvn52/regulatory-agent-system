"""Data exchanged by parsers, structure analysis, chunkers, and persistence."""

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from regagent.domain.documents import ActType, Document, DocumentVersion, Language


class RawBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    ordinal: int = Field(ge=0)
    text: str
    style: str | None = None
    page: int | None = Field(default=None, ge=1)


class ParsedSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    title: str
    media_type: str
    blocks: tuple[RawBlock, ...]
    parser_name: str
    parser_version: str


class NormalizedSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    title: str
    media_type: str
    blocks: tuple[RawBlock, ...]
    parser_name: str
    parser_version: str
    content_hash: str = Field(min_length=64, max_length=64)


class SectionKind(StrEnum):
    HEADING = "heading"
    PART = "part"
    CHAPTER = "chapter"
    ARTICLE = "article"
    PARAGRAPH = "paragraph"
    SUBPARAGRAPH = "subparagraph"
    BODY = "body"


class StructuredSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    parent_id: UUID | None = None
    ordinal: int = Field(ge=0)
    kind: SectionKind
    label: str | None = None
    heading: str | None = None
    article: str | None = None
    paragraph: str | None = None
    hierarchy_path: tuple[str, ...] = ()
    text: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)


class ChunkingStrategy(StrEnum):
    FIXED_WINDOW = "fixed_window"
    LEGAL_STRUCTURE = "legal_structure"


class ChunkDraft(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    ordinal: int = Field(ge=0)
    text: str = Field(min_length=1)
    content_hash: str = Field(min_length=64, max_length=64)
    token_count: int = Field(ge=1)
    hierarchy_path: tuple[str, ...] = ()
    source_section_ids: tuple[UUID, ...] = Field(min_length=1)
    strategy: ChunkingStrategy
    config: dict[str, Any] = Field(default_factory=dict)


class IngestionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_path: Path
    language: Language
    act_type: ActType
    issuer: str = Field(min_length=1)
    source_url: HttpUrl
    source_publisher: str = Field(min_length=1)
    retrieved_at: date
    title: str | None = None
    jurisdiction: str = "KZ"
    official_number: str | None = None
    source_identifier: str | None = None
    adopted_at: date | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    version_label: str | None = None


class IngestionPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    document: Document
    version: DocumentVersion
    pipeline_signature: str = Field(min_length=64, max_length=64)
    sections: tuple[StructuredSection, ...]
    chunks: tuple[ChunkDraft, ...]


class IngestionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: UUID
    version_id: UUID
    pipeline_signature: str = Field(min_length=64, max_length=64)
    section_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    chunking_strategies: tuple[ChunkingStrategy, ...]
    duplicate: bool
