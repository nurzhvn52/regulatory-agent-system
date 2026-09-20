"""Add chunking strategy and multi-section provenance.

Revision ID: 20260920_0002
Revises: 20260920_0001
Create Date: 2026-09-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260920_0002"
down_revision: str | None = "20260920_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column(
            "source_section_ids",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "chunks",
        sa.Column(
            "chunking_strategy",
            sa.String(length=100),
            server_default="legacy",
            nullable=False,
        ),
    )
    op.add_column(
        "chunks",
        sa.Column(
            "chunking_config",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.execute(
        "UPDATE chunks SET source_section_ids = jsonb_build_array(section_id::text) "
        "WHERE source_section_ids = '[]'::jsonb"
    )
    op.drop_constraint("uq_chunk_section_ordinal", "chunks", type_="unique")
    op.create_unique_constraint(
        "uq_chunk_version_strategy_ordinal",
        "chunks",
        ["version_id", "chunking_strategy", "ordinal"],
    )
    op.alter_column("chunks", "chunking_strategy", server_default=None)
    op.alter_column("chunks", "source_section_ids", server_default=None)
    op.alter_column("chunks", "chunking_config", server_default=None)


def downgrade() -> None:
    op.drop_constraint("uq_chunk_version_strategy_ordinal", "chunks", type_="unique")
    op.create_unique_constraint(
        "uq_chunk_section_ordinal",
        "chunks",
        ["section_id", "ordinal"],
    )
    op.drop_column("chunks", "chunking_config")
    op.drop_column("chunks", "chunking_strategy")
    op.drop_column("chunks", "source_section_ids")

