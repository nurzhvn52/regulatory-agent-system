"""Persistence models for documents, versions, embeddings, and experiment runs."""

from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from regagent.infrastructure.database.base import Base


class DocumentRecord(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_language", "language"),
        Index("ix_documents_act_type", "act_type"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(Text)
    act_type: Mapped[str] = mapped_column(String(50))
    issuer: Mapped[str] = mapped_column(Text)
    jurisdiction: Mapped[str] = mapped_column(String(20), default="KZ")
    language: Mapped[str] = mapped_column(String(10))
    official_number: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DocumentVersionRecord(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "content_hash", name="uq_version_document_hash"),
        CheckConstraint(
            "effective_to IS NULL OR effective_from IS NULL OR effective_to >= effective_from",
            name="ck_version_effective_interval",
        ),
        Index(
            "ix_versions_document_effective",
            "document_id",
            "effective_from",
            "effective_to",
        ),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE")
    )
    adopted_at: Mapped[date | None] = mapped_column(Date)
    effective_from: Mapped[date | None] = mapped_column(Date)
    effective_to: Mapped[date | None] = mapped_column(Date)
    version_label: Mapped[str | None] = mapped_column(String(200))
    content_hash: Mapped[str] = mapped_column(String(128))
    source_url: Mapped[str] = mapped_column(Text)
    source_publisher: Mapped[str] = mapped_column(Text)
    source_identifier: Mapped[str | None] = mapped_column(String(200))
    retrieved_at: Mapped[date] = mapped_column(Date)


class DocumentSectionRecord(Base):
    __tablename__ = "document_sections"
    __table_args__ = (
        UniqueConstraint(
            "version_id",
            "pipeline_signature",
            "ordinal",
            name="uq_section_version_pipeline_ordinal",
        ),
        CheckConstraint("page IS NULL OR page > 0", name="ck_section_positive_page"),
        Index("ix_sections_version", "version_id"),
        Index("ix_sections_version_pipeline", "version_id", "pipeline_signature"),
        Index("ix_sections_article", "article"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    version_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE")
    )
    pipeline_signature: Mapped[str] = mapped_column(String(64))
    parent_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("document_sections.id", ondelete="CASCADE")
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(50))
    label: Mapped[str | None] = mapped_column(String(100))
    heading: Mapped[str | None] = mapped_column(Text)
    article: Mapped[str | None] = mapped_column(String(100))
    paragraph: Mapped[str | None] = mapped_column(String(100))
    hierarchy_path: Mapped[list[str]] = mapped_column(JSONB, default=list)
    text: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer)


class ChunkRecord(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint(
            "version_id",
            "pipeline_signature",
            "chunking_strategy",
            "ordinal",
            name="uq_chunk_version_pipeline_strategy_ordinal",
        ),
        CheckConstraint("token_count > 0", name="ck_chunk_positive_token_count"),
        Index("ix_chunks_version", "version_id"),
        Index("ix_chunks_version_pipeline", "version_id", "pipeline_signature"),
        Index("ix_chunks_section", "section_id"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    version_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE")
    )
    pipeline_signature: Mapped[str] = mapped_column(String(64))
    section_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("document_sections.id", ondelete="CASCADE")
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(128))
    token_count: Mapped[int] = mapped_column(Integer)
    hierarchy_path: Mapped[list[str]] = mapped_column(JSONB, default=list)
    source_section_ids: Mapped[list[str]] = mapped_column(JSONB)
    chunking_strategy: Mapped[str] = mapped_column(String(100))
    chunking_config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class EmbeddingRecord(Base):
    __tablename__ = "embeddings"
    __table_args__ = (
        UniqueConstraint(
            "chunk_id",
            "model_name",
            "model_revision",
            name="uq_embedding_chunk_model_revision",
        ),
        CheckConstraint("dimensions > 0", name="ck_embedding_positive_dimensions"),
        Index("ix_embeddings_model", "model_name"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    chunk_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE")
    )
    model_name: Mapped[str] = mapped_column(String(300))
    model_revision: Mapped[str] = mapped_column(String(200))
    dimensions: Mapped[int] = mapped_column(Integer)
    normalized: Mapped[bool] = mapped_column(Boolean, default=True)
    embedding_config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    embedding: Mapped[list[float]] = mapped_column(Vector())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AgentSpecRecord(Base):
    __tablename__ = "agent_specs"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(200))
    schema_version: Mapped[str] = mapped_column(String(50), default="1")
    spec: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AgentRunRecord(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        Index("ix_agent_runs_experiment", "experiment_id"),
        Index("ix_agent_runs_status", "status"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_spec_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("agent_specs.id", ondelete="RESTRICT")
    )
    status: Mapped[str] = mapped_column(String(30))
    llm_model: Mapped[str | None] = mapped_column(String(300))
    experiment_id: Mapped[str | None] = mapped_column(String(200))
    input: Mapped[dict[str, Any]] = mapped_column(JSONB)
    output: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
