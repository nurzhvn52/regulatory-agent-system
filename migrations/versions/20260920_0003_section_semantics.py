"""Persist detected legal section semantics.

Revision ID: 20260920_0003
Revises: 20260920_0002
Create Date: 2026-09-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260920_0003"
down_revision: str | None = "20260920_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "document_sections",
        sa.Column("kind", sa.String(length=50), server_default="legacy", nullable=False),
    )
    op.add_column(
        "document_sections",
        sa.Column("label", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "document_sections",
        sa.Column("page", sa.Integer(), nullable=True),
    )
    op.create_check_constraint(
        "ck_section_positive_page",
        "document_sections",
        "page IS NULL OR page > 0",
    )
    op.alter_column("document_sections", "kind", server_default=None)


def downgrade() -> None:
    op.drop_constraint(
        "ck_section_positive_page",
        "document_sections",
        type_="check",
    )
    op.drop_column("document_sections", "page")
    op.drop_column("document_sections", "label")
    op.drop_column("document_sections", "kind")

