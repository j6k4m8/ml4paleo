"""
The worker protocol: messages between the API server and job workers.

Workers pull work over HTTPS from `/api/worker/v1`, authenticating with a
bearer token (`m4pw_...`). A worker claims a job, sends heartbeats while it
runs (which renew its lease and tell it if the job was cancelled), and then
reports the job complete or failed. Every report carries the lease token from
the claim; a report on a lease the worker no longer holds gets HTTP 409, and
the worker must throw its output away.

Both the server and the worker import these models, so they can't drift apart.
"""

import datetime
import enum
import json
import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .storage import StorageGrant

PROTOCOL_VERSION = 1
WORKER_TOKEN_PREFIX = "m4pw_"
# Workers may wait this long in one claim request for a job to turn up.
MAX_CLAIM_WAIT_SECONDS = 25
MAX_RESULT_BYTES = 64 * 1024
MAX_ERROR_CHARS = 4000


class Tier(enum.IntEnum):
    """
    Job priority. Lower runs first; jobs in one tier run in the order their
    pipelines were submitted.
    """

    INTERACTIVE = 0
    NORMAL = 1
    # Runs only on capacity that is idle anyway; never starts new machines.
    BACKGROUND = 2


class WorkerCaps(BaseModel):
    """
    What a worker process can run. Workers send this with every claim, so
    processes that share a token (for example several replicas of one
    container) may differ.
    """

    version: str = Field(max_length=32)
    kinds: list[str] = Field(max_length=100)
    labels: list[str] = Field(default=[], max_length=100)
    vram_gb: float = Field(default=0, ge=0)
    cpus: int = Field(default=1, ge=1)
    memory_gb: float = Field(default=0, ge=0)
    slots: int = Field(default=1, ge=1)


class HelloIn(BaseModel):
    caps: WorkerCaps
    protocol_version: int = PROTOCOL_VERSION


class HelloOut(BaseModel):
    worker_id: uuid.UUID
    name: str
    heartbeat_seconds: float
    lease_seconds: float


class ClaimIn(BaseModel):
    caps: WorkerCaps
    wait_seconds: float = Field(default=MAX_CLAIM_WAIT_SECONDS, ge=0)


class JobLease(BaseModel):
    job_id: uuid.UUID
    kind: str
    payload: dict[str, Any]
    lease_token: str
    lease_expires_at: datetime.datetime
    attempt: int
    # Storage the job may use. Credentials appear only in JSON.
    grants: list[StorageGrant] = []


class ClaimOut(BaseModel):
    job: JobLease | None


class HeartbeatIn(BaseModel):
    lease_token: str = Field(max_length=128)
    # The fraction of the job done so far, if the worker knows it.
    progress: float | None = Field(default=None, ge=0, le=1)
    message: str | None = Field(default=None, max_length=200)


class HeartbeatOut(BaseModel):
    lease_expires_at: datetime.datetime
    # The job was cancelled: stop, and report it failed.
    cancel: bool
    grants: list[StorageGrant] = []


class CompleteIn(BaseModel):
    lease_token: str = Field(max_length=128)
    # A small JSON summary. Large outputs go to storage, not here.
    result: dict[str, Any] = {}

    @field_validator("result")
    @classmethod
    def _small(cls, result: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(result)) > MAX_RESULT_BYTES:
            raise ValueError(f"result is over {MAX_RESULT_BYTES} bytes")
        return result


class FailIn(BaseModel):
    lease_token: str = Field(max_length=128)
    error: str = Field(max_length=MAX_ERROR_CHARS)
    # False for errors that a retry can't fix (bad input, cancellation).
    retryable: bool = True


class ReleaseIn(BaseModel):
    """
    Give a job back without counting the attempt, for example when the worker
    is shutting down.
    """

    lease_token: str = Field(max_length=128)


class LabelOpIn(BaseModel):
    """
    A label edit from a job, applied to the job's project as an edit from the
    annotator is (`POST /api/projects/{id}/labels/ops`). Only some kinds of
    job may send them.
    """

    lease_token: str = Field(max_length=128)
    # Sending the same id again applies the edit once.
    client_op_id: uuid.UUID
    # As the labels API takes them: masks and values base64-encoded.
    deltas: list[dict[str, Any]] = Field(min_length=1, max_length=512)
    tool: dict[str, Any] = {}
