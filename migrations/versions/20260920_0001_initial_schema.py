"""Create the initial regulatory-document research schema.

Revision ID: 20260920_0001
Revises:
Create Date: 2026-09-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "20260920_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("act_type", sa.String(length=50), nullable=False),
        sa.Column("issuer", sa.Text(), nullable=False),
        sa.Column("jurisdiction", sa.String(length=20), nullable=False),
        sa.Column("language", sa.String(length=10), nullable=False),
        sa.Column("official_number", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_language", "documents", ["language"])
    op.create_index("ix_documents_act_type", "documents", ["act_type"])

    op.create_table(
        "document_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("adopted_at", sa.Date(), nullable=True),
        sa.Column("effective_from", sa.Date(), nullable=True),
        sa.Column("effective_to", sa.Date(), nullable=True),
        sa.Column("version_label", sa.String(length=200), nullable=True),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("source_publisher", sa.Text(), nullable=False),
        sa.Column("source_identifier", sa.String(length=200), nullable=True),
        sa.Column("retrieved_at", sa.Date(), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "content_hash", name="uq_version_document_hash"),
        sa.CheckConstraint(
            "effective_to IS NULL OR effective_from IS NULL OR effective_to >= effective_from",
            name="ck_version_effective_interval",
        ),
    )
    op.create_index(
        "ix_versions_document_effective",
        "document_versions",
        ["document_id", "effective_from", "effective_to"],
    )

    op.create_table(
        "document_sections",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("parent_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("heading", sa.Text(), nullable=True),
        sa.Column("article", sa.String(length=100), nullable=True),
        sa.Column("paragraph", sa.String(length=100), nullable=True),
        sa.Column("hierarchy_path", postgresql.JSONB(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["parent_id"], ["document_sections.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version_id", "ordinal", name="uq_section_version_ordinal"),
    )
    op.create_index("ix_sections_version", "document_sections", ["version_id"])
    op.create_index("ix_sections_article", "document_sections", ["article"])

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("section_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("hierarchy_path", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(["section_id"], ["document_sections.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("section_id", "ordinal", name="uq_chunk_section_ordinal"),
        sa.CheckConstraint("token_count > 0", name="ck_chunk_positive_token_count"),
    )
    op.create_index("ix_chunks_version", "chunks", ["version_id"])
    op.create_index("ix_chunks_section", "chunks", ["section_id"])

    op.create_table(
        "embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_name", sa.String(length=300), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chunk_id", "model_name", name="uq_embedding_chunk_model"),
        sa.CheckConstraint("dimensions > 0", name="ck_embedding_positive_dimensions"),
    )
    op.create_index("ix_embeddings_model", "embeddings", ["model_name"])

    op.create_table(
        "agent_specs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("schema_version", sa.String(length=50), nullable=False),
        sa.Column("spec", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "agent_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_spec_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("llm_model", sa.String(length=300), nullable=True),
        sa.Column("experiment_id", sa.String(length=200), nullable=True),
        sa.Column("input", postgresql.JSONB(), nullable=False),
        sa.Column("output", postgresql.JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["agent_spec_id"], ["agent_specs.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_runs_experiment", "agent_runs", ["experiment_id"])
    op.create_index("ix_agent_runs_status", "agent_runs", ["status"])


def downgrade() -> None:
    op.drop_table("agent_runs")
    op.drop_table("agent_specs")
    op.drop_table("embeddings")
    op.drop_table("chunks")
    op.drop_table("document_sections")
    op.drop_table("document_versions")
    op.drop_table("documents")
