"""add transcript_chunks for semantic search

Revision ID: 2d346a80ac0e
Revises: 10821374dcd2
Create Date: 2026-05-29 22:31:04.412108

"""
from typing import Sequence, Union

from alembic import op
import pgvector
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2d346a80ac0e'
down_revision: Union[str, Sequence[str], None] = '10821374dcd2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('transcript_chunks',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('transcript_id', sa.String(length=32), nullable=False),
    sa.Column('entity_type', sa.Text(), nullable=False),
    sa.Column('entity_id', sa.Text(), nullable=False),
    sa.Column('chunk_index', sa.Integer(), nullable=False),
    sa.Column('chunk_text', sa.Text(), nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=1024), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_transcript_chunks_entity_id'), 'transcript_chunks', ['entity_id'], unique=False)
    op.create_index(op.f('ix_transcript_chunks_entity_type'), 'transcript_chunks', ['entity_type'], unique=False)
    op.create_index(op.f('ix_transcript_chunks_transcript_id'), 'transcript_chunks', ['transcript_id'], unique=False)
    op.execute(
        "CREATE INDEX ix_transcript_chunks_embedding_hnsw "
        "ON transcript_chunks "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 200)"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_transcript_chunks_embedding_hnsw")
    op.drop_index(op.f('ix_transcript_chunks_transcript_id'), table_name='transcript_chunks')
    op.drop_index(op.f('ix_transcript_chunks_entity_type'), table_name='transcript_chunks')
    op.drop_index(op.f('ix_transcript_chunks_entity_id'), table_name='transcript_chunks')
    op.drop_table('transcript_chunks')
