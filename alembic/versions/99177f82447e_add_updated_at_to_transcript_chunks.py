"""add updated_at to transcript_chunks

Revision ID: 99177f82447e
Revises: 2d346a80ac0e
Create Date: 2026-05-30 08:30:49.802472

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '99177f82447e'
down_revision: Union[str, Sequence[str], None] = '2d346a80ac0e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'transcript_chunks',
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('transcript_chunks', 'updated_at')
