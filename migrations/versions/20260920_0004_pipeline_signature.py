"""Version section and chunk outputs by processing pipeline.

Revision ID: 20260920_0004
Revises: 20260920_0003
Create Date: 2026-09-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260920_0004"
down_revision: str | None = "20260920_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "document_sections",
        sa.Column(
            "pipeline_signature",
            sa.String(length=64),
            server_default="legacy",
            nullable=False,
        ),
    )
    op.drop_constraint("uq_section_version_ordinal", "document_sections", type_="unique")
    op.create_unique_constraint(
        "uq_section_version_pipeline_ordinal",
        "document_sections",
        ["version_id", "pipeline_signature", "ordinal"],
    )
    op.create_index(
        "ix_sections_version_pipeline",
        "document_sections",
        ["version_id", "pipeline_signature"],
    )
    op.alter_column("document_sections", "pipeline_signature", server_default=None)

    op.add_column(
        "chunks",
        sa.Column(
            "pipeline_signature",
            sa.String(length=64),
            server_default="legacy",
            nullable=False,
        ),
    )
    op.drop_constraint("uq_chunk_version_strategy_ordinal", "chunks", type_="unique")
    op.create_unique_constraint(
        "uq_chunk_version_pipeline_strategy_ordinal",
        "chunks",
        ["version_id", "pipeline_signature", "chunking_strategy", "ordinal"],
    )
    op.create_index(
        "ix_chunks_version_pipeline",
        "chunks",
        ["version_id", "pipeline_signature"],
    )
    op.alter_column("chunks", "pipeline_signature", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_chunks_version_pipeline", table_name="chunks")
    op.drop_constraint(
        "uq_chunk_version_pipeline_strategy_ordinal",
        "chunks",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_chunk_version_strategy_ordinal",
        "chunks",
        ["version_id", "chunking_strategy", "ordinal"],
    )
    op.drop_column("chunks", "pipeline_signature")

    op.drop_index("ix_sections_version_pipeline", table_name="document_sections")
    op.drop_constraint(
        "uq_section_version_pipeline_ordinal",
        "document_sections",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_section_version_ordinal",
        "document_sections",
        ["version_id", "ordinal"],
    )
    op.drop_column("document_sections", "pipeline_signature")

