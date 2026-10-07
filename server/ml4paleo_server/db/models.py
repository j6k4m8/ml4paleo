"""
Database tables. Each build step adds its tables here, with an Alembic
migration in `ml4paleo_server/migrations/versions`.
"""

import datetime
import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    SmallInteger,
    String,
    Text,
    false,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, BYTEA, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin, uuid7


class SiteSetting(TimestampMixin, Base):
    """
    Runtime settings that admins change from the web UI (for example whether
    signup is open). Deploy-time settings live in environment variables.
    """

    __tablename__ = "site_settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[Any] = mapped_column(JSONB, nullable=False)


class User(TimestampMixin, Base):
    """
    A person with an account. Usernames and emails are stored lowercase.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    username: Mapped[str] = mapped_column(String(32), unique=True)
    email: Mapped[str | None] = mapped_column(String(254), unique=True)
    email_verified_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    password_hash: Mapped[str | None] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(default=False, server_default=false())
    # "active" or "disabled".
    status: Mapped[str] = mapped_column(String(16), default="active")
    must_change_password: Mapped[bool] = mapped_column(
        default=False, server_default=false()
    )
    # TOTP secrets are encrypted with a key derived from the server secret.
    totp_secret_enc: Mapped[str | None] = mapped_column(String(255))
    totp_pending_enc: Mapped[str | None] = mapped_column(String(255))
    # The time step of the last accepted code, so a code can't be reused.
    totp_last_step: Mapped[int | None] = mapped_column(BigInteger)
    last_login_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    # Per-user limits that replace the deploy defaults, for example
    # {"storage_gb": 100}. A null value means unlimited.
    quota_override: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )

    __table_args__ = (
        CheckConstraint("status IN ('active', 'disabled')", name="status"),
    )


class UserSession(Base):
    """
    A signed-in browser. Only a hash of the session token is stored.
    """

    __tablename__ = "sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_seen_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )
    ip: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(512))


class AuthToken(Base):
    """
    A single-use token sent by email or shared as a link: email verification,
    password reset, or a signup invite. Only a hash of the token is stored.
    """

    __tablename__ = "auth_tokens"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    email: Mapped[str | None] = mapped_column(String(254))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("kind IN ('verify', 'reset', 'invite')", name="kind"),
    )


class RateLimit(Base):
    """
    Fixed-window request counters. The table is unlogged: losing it in a
    crash only resets the counters.
    """

    __tablename__ = "rate_limits"
    __table_args__ = {"prefixes": ["UNLOGGED"]}

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    window_start: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    count: Mapped[int]


class EmailOutbox(Base):
    """
    Email waiting to be sent by the housekeeper.
    """

    __tablename__ = "email_outbox"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    to_address: Mapped[str] = mapped_column(String(254))
    subject: Mapped[str] = mapped_column(String(255))
    # The message text, sealed (`ml4paleo_server.sealing`): it holds
    # single-use links that must not sit readable in the database.
    body_sealed: Mapped[str] = mapped_column(Text)
    # "queued", "sent", or "failed".
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(default=0)
    # Failed sends wait longer before each retry.
    next_attempt_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    sent_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))


class Project(TimestampMixin, Base):
    """
    One scan and everything made from it. The owner and collaborators all
    have full access; only the owner can delete the project. Storage used by
    the project counts against the owner's quota.
    """

    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    name: Mapped[str] = mapped_column(String(100))
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), index=True
    )
    settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    # The v1 job this project was imported from, if any.
    v1_job_id: Mapped[str | None] = mapped_column(String(6), unique=True)
    deleted_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )


class ProjectMember(Base):
    """
    A person with access to a project. The owner has a row too.
    """

    __tablename__ = "project_members"

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    added_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AuditEvent(Base):
    """
    Who did what, for actions on accounts and projects.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    ip: Mapped[str | None] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64))
    target_type: Mapped[str] = mapped_column(String(32))
    target_id: Mapped[str] = mapped_column(String(64))
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )

    __table_args__ = (Index("ix_audit_log_target", "target_type", "target_id"),)


class UserUsage(Base):
    """
    What each user currently uses against their quota. Change it only through
    `ml4paleo_server.quotas`, which reserves usage before work starts and
    releases it when the work fails or its output is deleted.
    """

    __tablename__ = "user_usage"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    storage_bytes: Mapped[int] = mapped_column(
        BigInteger, default=0, server_default=text("0")
    )
    trained_models: Mapped[int] = mapped_column(default=0, server_default=text("0"))


class QuotaRequest(Base):
    """
    A user's request for higher limits, for an admin to grant or decline.
    """

    __tablename__ = "quota_requests"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    message: Mapped[str] = mapped_column(Text)
    # "open", "granted", or "declined".
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    resolved_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    resolved_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    __table_args__ = (
        CheckConstraint("status IN ('open', 'granted', 'declined')", name="status"),
    )


class Worker(TimestampMixin, Base):
    """
    A credential that job workers use to pull work. Several worker processes
    may share one (for example replicas of one container); each process sends
    its capabilities with every claim, and `caps` keeps the last ones seen.
    Whether a worker is online is derived from `last_seen_at`.
    """

    __tablename__ = "workers"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    # "local" (on this machine), "remote" (another machine), or "burst"
    # (a cloud machine started for a backlog).
    pool: Mapped[str] = mapped_column(String(16))
    # SHA-256 of the bearer token; the token itself is shown once.
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    # "active" or "revoked".
    status: Mapped[str] = mapped_column(String(16), default="active")
    caps: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    last_seen_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    expires_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )

    __table_args__ = (
        CheckConstraint("pool IN ('local', 'remote', 'burst')", name="pool"),
        CheckConstraint("status IN ('active', 'revoked')", name="status"),
    )


JOB_STATUSES = ("blocked", "queued", "leased", "succeeded", "failed", "cancelled")


class Job(TimestampMixin, Base):
    """
    One unit of work for a worker. Jobs form pipelines: every job has a
    `root_id` (the first job of its pipeline, or itself), and `job_deps` says
    which jobs must succeed before a job can run (until then it is
    "blocked"). See `ml4paleo_server.jobs`.
    """

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    root_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    # The job that created this one, for jobs that fan out more jobs.
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    kind: Mapped[str] = mapped_column(String(64))
    tier: Mapped[int] = mapped_column(SmallInteger)
    status: Mapped[str] = mapped_column(String(16))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    # Labels a worker must have (for example "gpu"), and the GPU memory it needs.
    required_labels: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)), default=list, server_default=text("'{}'")
    )
    min_vram_gb: Mapped[float] = mapped_column(
        Float, default=0, server_default=text("0")
    )
    # Pipeline progress is the weighted mean of its jobs' progress.
    weight: Mapped[float] = mapped_column(Float, default=1, server_default=text("1"))
    progress: Mapped[float] = mapped_column(Float, default=0, server_default=text("0"))
    message: Mapped[str | None] = mapped_column(String(200))
    attempts: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    max_attempts: Mapped[int] = mapped_column(default=3, server_default=text("3"))
    # Retries wait until this time.
    not_before: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # When the pipeline was submitted; jobs in a tier run in this order.
    submitted_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    # Whether a backlog of this job may start new (burst) machines.
    scale_trigger: Mapped[bool] = mapped_column(default=True)
    lease_worker_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("workers.id", ondelete="SET NULL")
    )
    lease_token_hash: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    cancel_requested: Mapped[bool] = mapped_column(default=False)
    # Enqueuing again with the same key returns the existing job.
    idempotency_key: Mapped[str | None] = mapped_column(String(200), unique=True)
    # Storage the job may use: [{"path": "projects/<id>/...", "access": "r"}].
    # Paths are relative to project storage; the claim turns each one into a
    # StorageGrant for the worker (see ml4paleo_server.broker).
    grants: Mapped[list[dict[str, str]]] = mapped_column(
        JSONB, default=list, server_default=text("'[]'::jsonb")
    )
    started_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    finished_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ({})".format(", ".join(f"'{s}'" for s in JOB_STATUSES)),
            name="status",
        ),
        CheckConstraint("progress >= 0 AND progress <= 1", name="progress"),
        # The claim query walks this index in priority order.
        Index(
            "ix_jobs_claim_order",
            "tier",
            "submitted_at",
            "id",
            postgresql_where=text("status = 'queued'"),
        ),
        Index(
            "ix_jobs_lease_expiry",
            "lease_expires_at",
            postgresql_where=text("status = 'leased'"),
        ),
    )


class JobDep(Base):
    """
    `job_id` runs only after `depends_on` succeeds.
    """

    __tablename__ = "job_deps"

    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True
    )
    depends_on: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True, index=True
    )


class JobAttempt(Base):
    """
    One lease of a job by a worker, and how it ended.
    """

    __tablename__ = "job_attempts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    attempt: Mapped[int]
    worker_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("workers.id", ondelete="SET NULL")
    )
    started_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    ended_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    # "succeeded", "failed", "expired" (the worker went silent), "cancelled",
    # or "released" (given back, not counted); None while running.
    outcome: Mapped[str | None] = mapped_column(String(16))
    error: Mapped[str | None] = mapped_column(Text)


ARTIFACT_STATES = (
    "staging",
    "committed",
    "superseded",
    "failed",
    "deleting",
    "deleted",
)


class Artifact(Base):
    """
    Something a job made and stored: an image pyramid, predictions, a model,
    meshes, an export. Its files live under `projects/<project>/artifacts/<id>/`
    and never change once committed.

    - `staging`: jobs are writing it.
    - `committed`: complete. The job that produces it commits it when it
      succeeds, after finding `_MANIFEST.json` (written last); its size then
      counts against the project owner's storage quota.
    - `superseded`: a newer artifact took its place as a head.
    - `failed`: its job failed or it was abandoned.
    - `deleting`: garbage collection is removing its files.
    - `deleted`: garbage collection removed its files.
    """

    __tablename__ = "artifacts"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(32))
    state: Mapped[str] = mapped_column(String(16), default="staging")
    # The bytes it counts against the quota, measured at commit.
    bytes: Mapped[int] = mapped_column(BigInteger, default=0, server_default=text("0"))
    # What it was made from (artifact ids and parameters).
    inputs: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    manifest: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    # The job whose success commits it.
    produced_by_job: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="SET NULL"), index=True
    )
    # The head slot it takes when committed (for example "image").
    head_slot: Mapped[str | None] = mapped_column(String(32))
    # For caches such as exports: the same key gives the same artifact.
    cache_key: Mapped[str | None] = mapped_column(String(200))
    expires_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # When it entered its current state.
    state_changed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "state IN ({})".format(", ".join(f"'{s}'" for s in ARTIFACT_STATES)),
            name="state",
        ),
        Index("ix_artifacts_state", "state", "state_changed_at"),
        Index(
            "ix_artifacts_cache_key",
            "project_id",
            "cache_key",
            unique=True,
            postgresql_where=text("cache_key IS NOT NULL AND state = 'committed'"),
        ),
    )


class ArtifactHead(Base):
    """
    The current artifact for each slot of a project ("image", "prediction",
    and so on). Moving a head is how a new result replaces the old one.
    """

    __tablename__ = "artifact_heads"

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    slot: Mapped[str] = mapped_column(String(32), primary_key=True)
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("artifacts.id", ondelete="RESTRICT"), index=True
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


UPLOAD_STATES = (
    "uploading",
    "completing",
    "complete",
    "aborted",
    "deleting",
    "deleted",
)


class Upload(Base):
    """
    A file a person uploads into a project, for example a scan to ingest.

    Browsers send it straight to object storage as an S3 multipart upload,
    with presigned URLs for each part, so an upload can resume after a broken
    connection. Its declared size counts against the project owner's storage
    quota from the start. The file lives at `projects/<project>/uploads/<id>/data`.

    - `uploading`: parts are arriving; it is aborted if not finished by
      `expires_at`.
    - `completing`: recorded just before asking storage to assemble the parts,
      so a crash at any point can be finished (or cleaned up) later.
    - `complete`: the whole file is stored; it is deleted after `expires_at`
      unless a job is still using it.
    - `aborted`: given up; nothing was kept.
    - `deleting`, `deleted`: garbage collection is removing it, or has.
    """

    __tablename__ = "uploads"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    # The name the person gave the file (its extension says what it is).
    filename: Mapped[str] = mapped_column(String(255))
    size: Mapped[int] = mapped_column(BigInteger)
    part_size: Mapped[int] = mapped_column(BigInteger)
    # The storage service's id for the multipart upload.
    multipart_id: Mapped[str | None] = mapped_column(String(1024))
    state: Mapped[str] = mapped_column(String(16), default="uploading")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    expires_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )

    __table_args__ = (
        CheckConstraint(
            "state IN ({})".format(", ".join(f"'{s}'" for s in UPLOAD_STATES)),
            name="state",
        ),
        CheckConstraint("size > 0 AND part_size > 0", name="sizes"),
    )


class LabelClass(Base):
    """
    One class in a project's label set. Values 0 (unlabeled) and 1
    (background) are built in; classes take 2 to 254 and a value is never
    reused, so old labels can't change meaning.
    """

    __tablename__ = "label_classes"

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    value: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    color: Mapped[str] = mapped_column(String(7))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    deleted_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    __table_args__ = (CheckConstraint("value BETWEEN 2 AND 254", name="value"),)


class LabelChunk(Base):
    """
    The current state of one 64-cubed label chunk: content hashes of its class
    and source arrays (None when all zero), stored as blobs under
    `projects/<project>/labels/blobs/`. Every edit makes a new version.
    """

    __tablename__ = "label_chunks"

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    cz: Mapped[int] = mapped_column(primary_key=True)
    cy: Mapped[int] = mapped_column(primary_key=True)
    cx: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    class_sha: Mapped[str | None] = mapped_column(String(64))
    source_sha: Mapped[str | None] = mapped_column(String(64))
    labeled_voxels: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    # Voxels per class value, as {"2": 1234, ...}.
    class_counts: Mapped[dict[str, int]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class LabelOp(Base):
    """
    One label edit (a stroke, a polygon, a fill, an accepted proposal), or
    the undo or redo of one. `seq` orders a project's ops; an edit is `live`
    until undone. The label state is the overlay of the live edits' claims
    in `seq` order (see `ml4paleo.labels.deltas`).
    """

    __tablename__ = "label_ops"

    seq: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    # The browser's id for the op, so a retried request applies it once.
    client_op_id: Mapped[uuid.UUID]
    # "edit", "undo", or "redo".
    kind: Mapped[str] = mapped_column(String(8))
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="SET NULL")
    )
    # ml4paleo.labels.Source of what an edit wrote.
    source: Mapped[int] = mapped_column(SmallInteger)
    # How the edit was made (tool, brush size, polygon points, ...).
    tool: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    # Global box (z0, y0, x0, z1, y1, x1) of everything the op touched.
    bbox: Mapped[list[int]] = mapped_column(ARRAY(BigInteger))
    # For undo and redo: the edit they act on.
    target_seq: Mapped[int | None] = mapped_column(
        ForeignKey("label_ops.seq", ondelete="CASCADE")
    )
    live: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("kind IN ('edit', 'undo', 'redo')", name="kind"),
        Index("ix_label_ops_client_op", "project_id", "client_op_id", unique=True),
    )


class LabelOpChunk(Base):
    """
    What an op did to one chunk: the chunk versions before and after, and
    (for edits) its claim, from which undo and redo recompute the chunk.
    """

    __tablename__ = "label_op_chunks"

    seq: Mapped[int] = mapped_column(
        ForeignKey("label_ops.seq", ondelete="CASCADE"), primary_key=True
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE")
    )
    cz: Mapped[int] = mapped_column(primary_key=True)
    cy: Mapped[int] = mapped_column(primary_key=True)
    cx: Mapped[int] = mapped_column(primary_key=True)
    base_version: Mapped[int]
    new_version: Mapped[int]
    class_sha: Mapped[str | None] = mapped_column(String(64))
    claim_box: Mapped[list[int] | None] = mapped_column(ARRAY(SmallInteger))
    claim_mask: Mapped[bytes | None] = mapped_column(BYTEA)
    claim_values: Mapped[bytes | None] = mapped_column(BYTEA)

    __table_args__ = (
        # Undo and redo read every claim on a chunk, in op order.
        Index("ix_label_op_chunks_chunk", "project_id", "cz", "cy", "cx", "seq"),
    )


ROI_STATUSES = ("open", "complete", "skipped")


class Roi(TimestampMixin, Base):
    """
    A region of interest: a cube or slab where someone labels for training.
    Unlabeled voxels count only inside ROIs marked complete (as background).
    """

    __tablename__ = "rois"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    # Global box (z0, y0, x0, z1, y1, x1), half-open.
    bbox: Mapped[list[int]] = mapped_column(ARRAY(BigInteger))
    # "cube" or "slice".
    kind: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(16), default="open")
    # "train" or "val" (validation, held out of training).
    split: Mapped[str] = mapped_column(String(8), default="train")
    # "user", "suggested", "explore" (a random place, to see how a model does
    # somewhere new), or "v1" (an annotation sample brought over from v1).
    origin: Mapped[str] = mapped_column(String(16), default="user")
    score: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        CheckConstraint(
            "status IN ({})".format(", ".join(f"'{s}'" for s in ROI_STATUSES)),
            name="status",
        ),
        CheckConstraint("kind IN ('cube', 'slice')", name="kind"),
        CheckConstraint("split IN ('train', 'val')", name="split"),
    )


class TrainingSet(Base):
    """
    What a model trained on, pinned: the image, the label chunks (by content
    hash) at one moment, the ROIs, and the classes. Its id is the SHA-256 of
    its manifest, which lives in project storage (`training/<id>/`); the same
    labels and ROIs give the same training set.
    """

    __tablename__ = "training_sets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    # Counts and ids from the manifest, for showing without reading it.
    summary: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TrainedModel(Base):
    """
    A segmentation model of a project. Its state follows its job and
    artifact: training while the job runs, ready once the artifact commits,
    failed if the job fails, and gone once deleted (which frees the owner's
    trained-model slot).
    """

    __tablename__ = "models"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(100))
    plugin: Mapped[str] = mapped_column(String(32))
    plugin_version: Mapped[str | None] = mapped_column(String(32))
    params: Mapped[dict[str, Any]] = mapped_column(JSONB)
    training_set_id: Mapped[str] = mapped_column(
        ForeignKey("training_sets.id", ondelete="CASCADE"), index=True
    )
    # The project's class values, in the model's order (1..K).
    class_values: Mapped[list[int]] = mapped_column(ARRAY(SmallInteger))
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="SET NULL")
    )
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="SET NULL")
    )
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    # Whether it counts against the project owner's trained-model quota.
    holds_slot: Mapped[bool] = mapped_column(default=False, server_default=false())
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    deleted_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
