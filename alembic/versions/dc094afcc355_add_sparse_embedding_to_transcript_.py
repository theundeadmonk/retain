"""add sparse_embedding to transcript_chunks for hybrid search

Revision ID: dc094afcc355
Revises: 99177f82447e
Create Date: 2026-05-30 09:20:38.794276

"""
from typing import Sequence, Union

from alembic import op
import pgvector
import sqlalchemy as sa


revision: str = 'dc094afcc355'
down_revision: Union[str, Sequence[str], None] = '99177f82447e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'transcript_chunks',
        sa.Column(
            'sparse_embedding',
            pgvector.sqlalchemy.sparsevec.SPARSEVEC(dim=30522),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column('transcript_chunks', 'sparse_embedding')
