"""Record the exact embedding model revision and encoding configuration.

Revision ID: 20260920_0005
Revises: 20260920_0004
Create Date: 2026-09-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260920_0005"
down_revision: str | None = "20260920_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "embeddings",
        sa.Column(
            "model_revision",
            sa.String(length=200),
            server_default="legacy",
            nullable=False,
        ),
    )
    op.add_column(
        "embeddings",
        sa.Column(
            "normalized",
            sa.Boolean(),
            server_default=sa.true(),
            nullable=False,
        ),
    )
    op.add_column(
        "embeddings",
        sa.Column(
            "embedding_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.drop_constraint("uq_embedding_chunk_model", "embeddings", type_="unique")
    op.create_unique_constraint(
        "uq_embedding_chunk_model_revision",
        "embeddings",
        ["chunk_id", "model_name", "model_revision"],
    )
    op.alter_column("embeddings", "model_revision", server_default=None)
    op.alter_column("embeddings", "normalized", server_default=None)
    op.alter_column("embeddings", "embedding_config", server_default=None)


def downgrade() -> None:
    op.drop_constraint(
        "uq_embedding_chunk_model_revision",
        "embeddings",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_embedding_chunk_model",
        "embeddings",
        ["chunk_id", "model_name"],
    )
    op.drop_column("embeddings", "embedding_config")
    op.drop_column("embeddings", "normalized")
    op.drop_column("embeddings", "model_revision")
