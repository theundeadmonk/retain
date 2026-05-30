"""halfvec embedding + sparse HNSW index

Revision ID: f927966412d5
Revises: dc094afcc355
Create Date: 2026-05-30 09:36:32.748511

"""
from typing import Sequence, Union

from alembic import op
import pgvector
import sqlalchemy as sa


revision: str = 'f927966412d5'
down_revision: Union[str, Sequence[str], None] = 'dc094afcc355'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index(
        'ix_transcript_chunks_embedding_hnsw',
        table_name='transcript_chunks',
    )
    op.alter_column(
        'transcript_chunks', 'embedding',
        existing_type=pgvector.sqlalchemy.vector.VECTOR(dim=1024),
        type_=pgvector.sqlalchemy.halfvec.HALFVEC(dim=1024),
        existing_nullable=True,
    )
    op.create_index(
        'ix_transcript_chunks_embedding_hnsw',
        'transcript_chunks',
        ['embedding'],
        unique=False,
        postgresql_using='hnsw',
        postgresql_ops={'embedding': 'halfvec_cosine_ops'},
        postgresql_with={'m': '16', 'ef_construction': '200'},
    )
    op.create_index(
        'ix_transcript_chunks_sparse_embedding_hnsw',
        'transcript_chunks',
        ['sparse_embedding'],
        unique=False,
        postgresql_using='hnsw',
        postgresql_ops={'sparse_embedding': 'sparsevec_l2_ops'},
    )


def downgrade() -> None:
    op.drop_index(
        'ix_transcript_chunks_sparse_embedding_hnsw',
        table_name='transcript_chunks',
    )
    op.drop_index(
        'ix_transcript_chunks_embedding_hnsw',
        table_name='transcript_chunks',
    )
    op.alter_column(
        'transcript_chunks', 'embedding',
        existing_type=pgvector.sqlalchemy.halfvec.HALFVEC(dim=1024),
        type_=pgvector.sqlalchemy.vector.VECTOR(dim=1024),
        existing_nullable=True,
    )
    op.create_index(
        'ix_transcript_chunks_embedding_hnsw',
        'transcript_chunks',
        ['embedding'],
        unique=False,
        postgresql_using='hnsw',
        postgresql_ops={'embedding': 'vector_cosine_ops'},
        postgresql_with={'m': '16', 'ef_construction': '200'},
    )
