"""add reid feedback table

Revision ID: 8f47df0a815f
Revises: 4021e4e63a60
Create Date: 2024-06-26 07:52:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '8f47df0a815f'
down_revision: Union[str, Sequence[str], None] = '4021e4e63a60'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'reid_feedback_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('event_timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('global_person_id', sa.String(length=100), nullable=False),
        sa.Column('candidate_person_id', sa.String(length=100), nullable=True),
        sa.Column('match_id', sa.String(length=100), nullable=True),
        sa.Column('camera_id', sa.String(length=50), nullable=False),
        sa.Column('environment_id', sa.String(length=50), nullable=False),
        sa.Column('frame_number', sa.Integer(), nullable=True),
        sa.Column('session_id', sa.String(length=100), nullable=True),
        sa.Column('decision', sa.String(length=20), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('feedback_source', sa.String(length=50), server_default=sa.text("'frontend'"), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('feedback_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_reid_feedback_person_time', 'reid_feedback_events', ['global_person_id', 'event_timestamp'], unique=False)
    op.create_index('idx_reid_feedback_camera_time', 'reid_feedback_events', ['camera_id', 'event_timestamp'], unique=False)
    op.create_index('idx_reid_feedback_decision', 'reid_feedback_events', ['decision'], unique=False)
    op.create_index('idx_reid_feedback_match', 'reid_feedback_events', ['match_id'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_reid_feedback_match', table_name='reid_feedback_events')
    op.drop_index('idx_reid_feedback_decision', table_name='reid_feedback_events')
    op.drop_index('idx_reid_feedback_camera_time', table_name='reid_feedback_events')
    op.drop_index('idx_reid_feedback_person_time', table_name='reid_feedback_events')
    op.drop_table('reid_feedback_events')
