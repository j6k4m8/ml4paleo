"""
A project's labels: the class list, edits with undo and redo, the change
feed collaborators follow, and the labels as a zarr group for viewers.

    GET    /api/projects/{id}/labels/classes
    POST   /api/projects/{id}/labels/classes             {name, color}
    PATCH  /api/projects/{id}/labels/classes/{value}     {name?, color?}
    DELETE /api/projects/{id}/labels/classes/{value}
    POST   /api/projects/{id}/labels/ops                 apply an edit
    POST   /api/projects/{id}/labels/ops/{seq}/undo      {client_op_id}
    POST   /api/projects/{id}/labels/ops/{seq}/redo      {client_op_id}
    GET    /api/projects/{id}/labels/ops                 history, newest first
    GET    /api/projects/{id}/labels/changes?after=seq   what changed since
    GET    /api/projects/{id}/labels/events?after=seq    the same, as SSE
    GET    /api/projects/{id}/labels/zarr/{key}          the labels as zarr

The zarr group has two uint8 arrays shaped like the image: `class` (label
values) and `source` (who made each label, `ml4paleo.labels.Source`), in
64-cubed chunks. A chunk's ETag is its content hash and `X-Chunk-Version` is
its version (send it back as `base_version`); chunks that are all zero are
404, which zarr reads as zeros.
"""

import asyncio
import base64
import binascii
import datetime
import json
import re
import uuid
from collections.abc import Sequence
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, HTTPException, Path, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, select
from starlette.concurrency import run_in_threadpool

from ml4paleo.labels import BACKGROUND, LABEL_CHUNK_ZYX, MAX_CLASS, UNLABELED, Source
from ml4paleo.labels.codec import ZARR_CODECS, blob_key
from ml4paleo.labels.deltas import ChunkDelta, unpack_mask, unpack_values
from ml4paleo.segmentation.predict import open_prediction
from ml4paleo.storage import get_bytes

from .. import artifacts, audit, labels, streams
from ..auth.deps import CurrentAuth, DbSession, SettingsDep
from ..db import (
    Artifact,
    LabelChunk,
    LabelClass,
    LabelOp,
    Project,
    ProjectMember,
    Roi,
    UserSession,
)
from ..storage import project_storage
from .projects import MemberProject

router = APIRouter(prefix="/api/projects/{project_id}/labels", tags=["labels"])

MAX_DELTAS = 512
MAX_TOOL_BYTES = 16 * 1024
FIRST_CLASS = BACKGROUND + 1
COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
EVENT_INTERVAL_SECONDS = 1.0
STREAM_FOR = datetime.timedelta(minutes=30)
# Op seqs are bigints; larger numbers never name an op.
MAX_SEQ = 2**63 - 1
Seq = Annotated[int, Path(ge=1, le=MAX_SEQ)]


# --- classes ---------------------------------------------------------------


class ClassIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    color: str = Field(pattern=COLOR.pattern)


class ClassPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    color: str | None = Field(default=None, pattern=COLOR.pattern)


class ClassOut(BaseModel):
    value: int
    name: str
    color: str


@router.get("/classes")
async def list_classes(project: MemberProject, db: DbSession) -> list[ClassOut]:
    rows = (
        await db.scalars(
            select(LabelClass)
            .where(LabelClass.project_id == project.id, LabelClass.deleted_at.is_(None))
            .order_by(LabelClass.value)
        )
    ).all()
    return [ClassOut(value=r.value, name=r.name, color=r.color) for r in rows]


@router.post("/classes", status_code=201)
async def add_class(
    body: ClassIn,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
) -> ClassOut:
    """
    Add a class. It gets the next value never used in this project.
    """
    # Lock the project so two new classes can't take the same value.
    await db.scalar(
        select(Project.id)
        .where(Project.id == project.id)
        .with_for_update(key_share=True)
    )
    highest = await db.scalar(
        select(func.max(LabelClass.value)).where(LabelClass.project_id == project.id)
    )
    value = max(FIRST_CLASS, (highest or 0) + 1)
    if value > MAX_CLASS:
        raise HTTPException(
            status_code=409, detail="This project has used every class value."
        )
    db.add(
        LabelClass(project_id=project.id, value=value, name=body.name, color=body.color)
    )
    audit.record(
        db,
        actor_id=auth.user.id,
        action="labels.class.add",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"value": value, "name": body.name},
    )
    await db.commit()
    return ClassOut(value=value, name=body.name, color=body.color)


async def _class(db, project, value: int) -> LabelClass:
    row = await db.scalar(
        select(LabelClass).where(
            LabelClass.project_id == project.id,
            LabelClass.value == value,
            LabelClass.deleted_at.is_(None),
        )
    )
    if row is None:
        raise HTTPException(status_code=404, detail="No such class.")
    return row


@router.patch("/classes/{value}")
async def edit_class(
    value: int, body: ClassPatch, project: MemberProject, db: DbSession
) -> ClassOut:
    row = await _class(db, project, value)
    if body.name is not None:
        row.name = body.name
    if body.color is not None:
        row.color = body.color
    await db.commit()
    return ClassOut(value=row.value, name=row.name, color=row.color)


@router.delete("/classes/{value}", status_code=204)
async def remove_class(
    value: int,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
) -> None:
    """
    Retire a class: no new labels can use it, and its value is never reused.
    """
    row = await _class(db, project, value)
    row.deleted_at = datetime.datetime.now(datetime.UTC)
    audit.record(
        db,
        actor_id=auth.user.id,
        action="labels.class.remove",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"value": value},
    )
    await db.commit()


# --- ops ---------------------------------------------------------------------


def _b64(data: str) -> bytes:
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("not valid base64") from None


class DeltaIn(BaseModel):
    """
    One chunk's part of an edit (see `ml4paleo.labels.deltas.ChunkDelta`),
    with `mask` and `values` base64-encoded.
    """

    key: tuple[int, int, int]
    base_version: int = 0
    box: tuple[int, int, int, int, int, int]
    mask: str
    value: int | None = None
    values: str | None = None
    only_if: str = "any"

    def to_delta(self) -> ChunkDelta:
        return ChunkDelta(
            key=self.key,
            base_version=self.base_version,
            box=self.box,
            mask=_b64(self.mask),
            value=self.value,
            values=_b64(self.values) if self.values is not None else None,
            only_if=self.only_if,
        )


class OpIn(BaseModel):
    # Edits here are always people's own (`Source.HUMAN`); accepting a
    # prediction has its own endpoint, which checks it.
    model_config = ConfigDict(extra="forbid")

    client_op_id: uuid.UUID
    deltas: list[DeltaIn] = Field(min_length=1, max_length=MAX_DELTAS)
    # Refuse the edit if any chunk changed since the client read it.
    strict: bool = False
    tool: dict[str, Any] = {}

    @field_validator("tool")
    @classmethod
    def _small(cls, tool: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(tool)) > MAX_TOOL_BYTES:
            raise ValueError("tool is too large")
        return tool


class ChunkOut(BaseModel):
    key: tuple[int, int, int]
    version: int
    sha: str | None


class OpOut(BaseModel):
    seq: int
    chunks: list[ChunkOut]


def _op_out(result: labels.OpResult) -> OpOut:
    return OpOut(
        seq=result.seq,
        chunks=[
            ChunkOut(key=c.key, version=c.version, sha=c.class_sha)
            for c in result.chunks
        ],
    )


async def allowed_values(db, project_id: uuid.UUID) -> set[int]:
    classes = await db.scalars(
        select(LabelClass.value).where(
            LabelClass.project_id == project_id, LabelClass.deleted_at.is_(None)
        )
    )
    return {UNLABELED, BACKGROUND, *classes}


def check_values(deltas: list[ChunkDelta], allowed: set[int]) -> None:
    for delta in deltas:
        if delta.value is not None:
            used = {delta.value}
        else:
            used = set(
                np.unique(unpack_values(delta.values or b"", delta.box_shape)).tolist()
            )
        if not used <= allowed:
            raise ValueError(
                f"label values {sorted(used - allowed)} are not classes here"
            )


@router.post("/ops", status_code=201)
async def apply_op(
    body: OpIn,
    project: MemberProject,
    auth: CurrentAuth,
    db: DbSession,
    settings: SettingsDep,
) -> OpOut:
    """
    Apply one edit. Sending the same `client_op_id` again returns the first
    result. A strict edit gets 409 with the chunks that changed meanwhile.
    """
    # A retry returns the first result, even if a class it used is gone now.
    if done := await labels.existing(db, project.id, body.client_op_id):
        return _op_out(done)
    try:
        deltas = [delta.to_delta() for delta in body.deltas]
        allowed = await allowed_values(db, project.id)
        await run_in_threadpool(check_values, deltas, allowed)
        result = await labels.apply_edit(
            db,
            settings,
            project.id,
            client_op_id=body.client_op_id,
            deltas=deltas,
            source=Source.HUMAN,
            tool=body.tool,
            strict=body.strict,
            user_id=auth.user.id,
        )
    except labels.NoImage:
        raise HTTPException(
            status_code=409, detail="This project has no image yet."
        ) from None
    except labels.Conflict as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": "Some chunks changed; reload them.", "chunks": exc.keys},
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    await db.commit()
    return _op_out(result)


# The most of a prediction one accept may read (16 MiB of label values).
MAX_ACCEPT_VOXELS = 256**3


class AcceptIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_op_id: uuid.UUID
    prediction_artifact_id: uuid.UUID
    roi_id: uuid.UUID
    # One predicted value per delta (an op touches a chunk once), written
    # only into unlabeled voxels.
    deltas: list[DeltaIn] = Field(min_length=1, max_length=MAX_DELTAS)


def _check_against_prediction(
    deltas: list[ChunkDelta], predicted: np.ndarray, origin: Sequence[int]
) -> None:
    """Every voxel a delta selects must hold the value it writes."""
    for delta in deltas:
        start = [
            k * c + b - o
            for k, c, b, o in zip(
                delta.key, LABEL_CHUNK_ZYX, delta.box[:3], origin, strict=True
            )
        ]
        shape = delta.box_shape
        region = predicted[
            tuple(slice(a, a + n) for a, n in zip(start, shape, strict=True))
        ]
        mask = unpack_mask(delta.mask, shape)
        if (region[mask] != delta.value).any():
            raise ValueError("Those labels don't match the prediction")


@router.post("/accept", status_code=201)
async def accept_prediction(
    body: AcceptIn,
    project: MemberProject,
    auth: CurrentAuth,
    db: DbSession,
    settings: SettingsDep,
) -> OpOut:
    """
    Accept part of a model's prediction as labels, recorded as
    model-verified with the prediction, its model, and the ROI.

    The server checks the claim: each delta writes one predicted value (not
    0) into only unlabeled voxels inside the ROI, and the stored prediction
    has exactly that value at every voxel it selects. Undo and redo work as
    for any edit.
    """
    if done := await labels.existing(db, project.id, body.client_op_id):
        return _op_out(done)
    prediction = await db.scalar(
        select(Artifact).where(
            Artifact.id == body.prediction_artifact_id,
            Artifact.project_id == project.id,
            Artifact.kind == "prediction",
            Artifact.state.in_(("committed", "superseded")),
        )
    )
    roi = await db.scalar(
        select(Roi).where(Roi.id == body.roi_id, Roi.project_id == project.id)
    )
    if prediction is None or roi is None:
        raise HTTPException(status_code=404, detail="No such prediction or ROI.")
    try:
        deltas = [delta.to_delta() for delta in body.deltas]
        for delta in deltas:
            if (
                delta.values is not None
                or not delta.value
                or delta.only_if != "unlabeled"
            ):
                raise ValueError(
                    "An accepted prediction writes one predicted value per chunk, "
                    "only into unlabeled voxels"
                )
        allowed = await allowed_values(db, project.id)
        await run_in_threadpool(check_values, deltas, allowed)
        box = labels.global_box(deltas)
        if any(box[a] < roi.bbox[a] or box[a + 3] > roi.bbox[a + 3] for a in range(3)):
            raise ValueError("Those labels reach outside the ROI")
        if np.prod([box[a + 3] - box[a] for a in range(3)]) > MAX_ACCEPT_VOXELS:
            raise ValueError("That's too much to accept at once; use a smaller ROI")
        grant = project_storage(settings).child(artifacts.artifact_path(prediction))
        region = tuple(slice(box[a], box[a + 3]) for a in range(3))
        predicted = await run_in_threadpool(
            lambda: np.asarray(open_prediction(grant)["class"][region])  # type: ignore[index]
        )
        await run_in_threadpool(_check_against_prediction, deltas, predicted, box[:3])
        result = await labels.apply_edit(
            db,
            settings,
            project.id,
            client_op_id=body.client_op_id,
            deltas=deltas,
            source=Source.MODEL_VERIFIED,
            tool={
                "name": "accept-prediction",
                "prediction": str(prediction.id),
                "model": prediction.inputs.get("model_id"),
                "roi": str(roi.id),
            },
            user_id=auth.user.id,
        )
    except labels.NoImage:
        raise HTTPException(
            status_code=409, detail="This project has no image yet."
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    await db.commit()
    return _op_out(result)


class ToggleIn(BaseModel):
    client_op_id: uuid.UUID


async def _toggle(seq, body, project, auth, db, settings, live: bool) -> OpOut:
    try:
        result = await labels.set_live(
            db,
            settings,
            project.id,
            target_seq=seq,
            live=live,
            client_op_id=body.client_op_id,
            user_id=auth.user.id,
        )
    except labels.NotFound:
        raise HTTPException(status_code=404, detail="No such edit.") from None
    except labels.AlreadyDone:
        raise HTTPException(
            status_code=409,
            detail="That edit is already " + ("live." if live else "undone."),
        ) from None
    await db.commit()
    return _op_out(result)


@router.post("/ops/{seq}/undo", status_code=201)
async def undo(
    seq: Seq,
    body: ToggleIn,
    project: MemberProject,
    auth: CurrentAuth,
    db: DbSession,
    settings: SettingsDep,
) -> OpOut:
    return await _toggle(seq, body, project, auth, db, settings, live=False)


@router.post("/ops/{seq}/redo", status_code=201)
async def redo(
    seq: Seq,
    body: ToggleIn,
    project: MemberProject,
    auth: CurrentAuth,
    db: DbSession,
    settings: SettingsDep,
) -> OpOut:
    return await _toggle(seq, body, project, auth, db, settings, live=True)


class HistoryOut(BaseModel):
    seq: int
    kind: str
    user_id: uuid.UUID | None
    source: int
    tool: dict[str, Any]
    bbox: list[int]
    target_seq: int | None
    live: bool
    created_at: datetime.datetime


def _history_out(op: LabelOp) -> HistoryOut:
    return HistoryOut(
        seq=op.seq,
        kind=op.kind,
        user_id=op.user_id,
        source=op.source,
        tool=op.tool,
        bbox=op.bbox,
        target_seq=op.target_seq,
        live=op.live,
        created_at=op.created_at,
    )


@router.get("/ops")
async def history(
    project: MemberProject,
    db: DbSession,
    before: Annotated[int | None, Query(ge=1, le=MAX_SEQ)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[HistoryOut]:
    query = select(LabelOp).where(LabelOp.project_id == project.id)
    if before is not None:
        query = query.where(LabelOp.seq < before)
    ops = (await db.scalars(query.order_by(LabelOp.seq.desc()).limit(limit))).all()
    return [_history_out(op) for op in ops]


class ChangeOut(BaseModel):
    op: HistoryOut
    chunks: list[ChunkOut]


async def _changes(db, project_id, after: int) -> list[ChangeOut]:
    return [
        ChangeOut(
            op=_history_out(op),
            chunks=[
                ChunkOut(key=c.key, version=c.version, sha=c.class_sha) for c in chunks
            ],
        )
        for op, chunks in await labels.changes_since(db, project_id, after)
    ]


@router.get("/changes")
async def changes(
    project: MemberProject,
    db: DbSession,
    after: Annotated[int, Query(ge=0, le=MAX_SEQ)] = 0,
) -> list[ChangeOut]:
    """
    Edits, undos, and redos after op `after`, oldest first (at most 200), with
    the chunk versions they made; refetch those chunks.
    """
    return await _changes(db, project.id, after)


@router.get("/events")
async def events(
    project: MemberProject,
    auth: CurrentAuth,
    db: DbSession,
    request: Request,
    after: Annotated[int, Query(ge=0, le=MAX_SEQ)] = 0,
) -> StreamingResponse:
    """
    Server-sent events: a `change` event (as in /changes) for each op after
    `after`, as they happen, while the viewer stays signed in and a member.
    A reconnecting browser resumes after the last event it got.
    """
    streams.check(auth.user.id)
    last_event_id = request.headers.get("last-event-id", "")
    if last_event_id.isdigit() and len(last_event_id) < 20:
        after = max(after, min(int(last_event_id), MAX_SEQ))
    user_id, session_hash, project_id = (
        auth.user.id,
        auth.session.token_hash,
        project.id,
    )
    await db.rollback()
    sessionmaker = request.app.state.sessionmaker

    async def still_allowed(session) -> bool:
        member = await session.scalar(
            select(ProjectMember.user_id)
            .join(Project, Project.id == ProjectMember.project_id)
            .where(
                ProjectMember.project_id == project_id,
                ProjectMember.user_id == user_id,
                Project.deleted_at.is_(None),
            )
        )
        signed_in = await session.scalar(
            select(UserSession.token_hash).where(
                UserSession.token_hash == session_hash,
                UserSession.expires_at > datetime.datetime.now(datetime.UTC),
            )
        )
        return member is not None and signed_in is not None

    async def stream():
        last = after
        deadline = datetime.datetime.now(datetime.UTC) + STREAM_FOR
        while datetime.datetime.now(datetime.UTC) < deadline:
            async with sessionmaker() as session:
                if not await still_allowed(session):
                    return
                batch = await _changes(session, project_id, last)
            for change in batch:
                last = change.op.seq
                yield f"id: {last}\nevent: change\ndata: {change.model_dump_json()}\n\n"
            if await request.is_disconnected():
                return
            if not batch:
                await asyncio.sleep(EVENT_INTERVAL_SECONDS)

    return StreamingResponse(
        streams.counted(user_id, stream()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


# --- the labels as zarr ----------------------------------------------------

ARRAYS = ("class", "source")
_CHUNK_KEY = re.compile(r"^(class|source)/c/(\d{1,9})/(\d{1,9})/(\d{1,9})$")


def _array_metadata(shape: tuple[int, int, int]) -> dict:
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(shape),
        "data_type": "uint8",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(LABEL_CHUNK_ZYX)},
        },
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": ZARR_CODECS,
        "dimension_names": ["z", "y", "x"],
        "attributes": {},
    }


@router.get("/zarr/{key:path}")
async def label_zarr(
    key: str,
    project: MemberProject,
    db: DbSession,
    settings: SettingsDep,
    request: Request,
) -> Response:
    revalidate = {"Cache-Control": "private, no-cache"}
    try:
        shape = await labels.volume_shape(db, project.id)
    except labels.NoImage:
        raise HTTPException(
            status_code=404, detail="This project has no image yet."
        ) from None
    if key == "zarr.json":
        body = {"zarr_format": 3, "node_type": "group", "attributes": {}}
        return Response(
            json.dumps(body), media_type="application/json", headers=revalidate
        )
    if key in (f"{name}/zarr.json" for name in ARRAYS):
        return Response(
            json.dumps(_array_metadata(shape)),
            media_type="application/json",
            headers=revalidate,
        )
    match = _CHUNK_KEY.match(key)
    if match is None:
        raise HTTPException(status_code=404, detail="No such zarr key.")
    array, cz, cy, cx = (
        match.group(1),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
    )
    row = await db.scalar(
        select(LabelChunk).where(
            LabelChunk.project_id == project.id,
            LabelChunk.cz == cz,
            LabelChunk.cy == cy,
            LabelChunk.cx == cx,
        )
    )
    version = row.version if row is not None else 0
    sha = (
        None if row is None else (row.class_sha if array == "class" else row.source_sha)
    )
    headers = {**revalidate, "X-Chunk-Version": str(version)}
    root = labels.labels_root(settings, project.id)
    # Give the connection back before reading storage.
    await db.rollback()
    if sha is None:
        return Response(status_code=404, headers=headers)
    headers["ETag"] = f'"{sha}"'
    if request.headers.get("if-none-match") == headers["ETag"]:
        return Response(status_code=304, headers=headers)
    data = await run_in_threadpool(get_bytes, root, blob_key(sha))
    if data is None:
        raise HTTPException(
            status_code=500, detail="A label chunk is missing from storage."
        )
    return Response(data, media_type="application/octet-stream", headers=headers)
