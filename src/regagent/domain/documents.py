"""Provider-independent models for regulatory documents and their provenance."""

from datetime import date
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Language(StrEnum):
    RU = "ru"
    KK = "kk"
    EN = "en"


class ActType(StrEnum):
    LAW = "law"
    CODE = "code"
    DECREE = "decree"
    ORDER = "order"
    REGULATION = "regulation"
    STANDARD = "standard"
    OTHER = "other"


class SourceReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: HttpUrl
    retrieved_at: date
    publisher: str
    source_identifier: str | None = None


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    title: str = Field(min_length=1)
    act_type: ActType
    issuer: str = Field(min_length=1)
    jurisdiction: str = "KZ"
    language: Language
    official_number: str | None = None


class DocumentVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    adopted_at: date | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    version_label: str | None = None
    content_hash: str = Field(min_length=16)
    source: SourceReference

    @model_validator(mode="after")
    def validate_effective_interval(self) -> "DocumentVersion":
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_to < self.effective_from
        ):
            raise ValueError("effective_to cannot be earlier than effective_from")
        return self


class DocumentSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    version_id: UUID
    parent_id: UUID | None = None
    ordinal: int = Field(ge=0)
    heading: str | None = None
    article: str | None = None
    paragraph: str | None = None
    hierarchy_path: tuple[str, ...] = ()
    text: str = Field(min_length=1)


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    section_id: UUID
    ordinal: int = Field(ge=0)
    text: str = Field(min_length=1)
    content_hash: str = Field(min_length=16)
    token_count: int = Field(ge=1)
    hierarchy_path: tuple[str, ...] = ()

