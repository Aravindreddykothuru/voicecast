"""Record when a worker last proved it was still working on a project.

A run that no worker picks up (workers down, wrong queue) or whose worker
dies mid-stage used to sit in "queued"/"processing" forever: nothing ever
changed the row again, and the UI had no way to tell a slow CPU stage from a
dead one. Workers now write heartbeat_at every few seconds while a task runs
(app/pipeline/liveness.py), and the API reports a run as stalled once it goes
quiet for longer than STALL_AFTER_SECONDS.

Nullable, no backfill: projects that never ran have no heartbeat.

Revision ID: 0007_project_heartbeat
Revises: 0006_timestamps_not_null
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_project_heartbeat"
down_revision = "0006_timestamps_not_null"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("projects", sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True))
    # Estimated once per speaker from their own audio (median F0) so an
    # engine with male and female voices gives each speaker a matching one;
    # "unknown" when there was too little voiced speech to tell.
    op.add_column("speakers", sa.Column("voice_gender", sa.String(16), nullable=True))


def downgrade() -> None:
    op.drop_column("speakers", "voice_gender")
    op.drop_column("projects", "heartbeat_at")
