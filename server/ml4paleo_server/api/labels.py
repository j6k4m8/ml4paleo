"""
A project's labels: the class list, edits with undo and redo, the change
feed collaborators follow, and the labels as a zarr group for viewers.

    GET    /api/projects/{id}/labels/classes
    POST   /api/projects/{id}/labels/classes             {name, color}
    PATCH  /api/projects/{id}/labels/classes/{value}     {name?, color?}
    DELETE /api/projects/{id}/labels/classes/{value}
    GET    /api/projects/{id}/labels/counts              voxels labeled with each value
    POST   /api/projects/{id}/labels/ops                 apply an edit
    POST   /api/projects/{id}/labels/ops/{seq}/undo      {client_op_id}
    POST   /api/projects/{id}/labels/ops/{seq}/redo      {client_op_id}
    POST   /api/projects/{id}/labels/accept              accept part of a prediction
    GET    /api/projects/{id}/labels/ops                 history, newest first
    GET    /api/projects/{id}/labels/changes?after=seq   what changed since
    GET    /api/projects/{id}/labels/events?after=seq    the same, as SSE
    GET    /api/projects/{id}/labels/zarr/{key}          the labels as zarr

Accepting a prediction is how a model's labels become the project's:
`{client_op_id, prediction_artifact_id, deltas}` and where they go, exactly
one of `roi_id` (an ROI) or `box` (`[z0, y0, x0, z1, y1, x1]`, whole voxels,
half-open, not empty, inside the image, as an ROI's box is; the annotator
sends the part of a slice a view shows). Either way the deltas must stay
inside it and the labels they write span at most 256^3 voxels (a box may
hold no more either; an ROI may be bigger), each delta writes one predicted
value (never 0) into only unlabeled voxels, and the server reads the stored
prediction and refuses the op unless it holds that value at every voxel a
delta selects, and is of the project's current image if it says which image
it was made from (409). The op is `Source.MODEL_VERIFIED`, and its tool
record names the prediction, its model, and the ROI or the box; garbage
collection keeps a prediction such an op names, undone or not.

The zarr group has two uint8 arrays shaped like the image: `class` (label
values) and `source` (who made each label, `ml4paleo.labels.Source`), in
64-cubed chunks. A chunk's ETag is its content hash and `X-Chunk-Version` is
its version (send it back as `base_version`); chunks that are all zero are
404, which zarr reads as zeros.

For zoomed-out views there are also `class_1`, `class_2`, ... : the labels at
the image's pyramid levels (see `label_pyramid`), made when asked for, to look
at only. Their `X-Pyramid-Version` (not `X-Chunk-Version`, which is for edits)
and ETag change when any chunk under them changes, and finding that out takes
one pass over the label rows under the chunk, which for the top chunk is every
row of the project. Their ETags are only unique to the URL: two chunks can have
the same one, so don't tell chunks apart by it.

The group's `zarr.json` lists these arrays twice. `attributes.ome.multiscales`
is standard OME-Zarr 0.5 (axes z, y, x, and a dataset per level with the image's
voxel size times the level's factors as its scale), which is what makes
Neuroglancer open the group as one volume and pick a level by zoom, in the
image's coordinates. `attributes.ml4paleo.label_levels` is the web viewer's: each
array's name, shape, and `factor_zyx`.

A chunk too big to make in one request is a 503 with `Retry-After`, and what
was done is kept, so asking again gets further. A client must ask again, as
zarr readers don't (they take a 503 for an error and show nothing): wait
`Retry-After` seconds and a little more at random, so chunks asked for together
don't all come back together, and keep asking while the chunk is wanted.
"""

import asyncio
import base64
import binascii
import datetime
import json
import math
import re
import threading
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Path, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import func, select, text
from starlette.concurrency import run_in_threadpool

from ml4paleo.labels import BACKGROUND, LABEL_CHUNK_ZYX, MAX_CLASS, UNLABELED, Source
from ml4paleo.labels.codec import ZARR_CODECS, blob_key
from ml4paleo.labels.deltas import ChunkDelta, unpack_mask, unpack_values
from ml4paleo.ome import LevelSpec
from ml4paleo.protocol import json_text
from ml4paleo.segmentation.predict import open_prediction
from ml4paleo.storage import StorageGrant, get_bytes, object_store

from .. import artifacts, audit, label_pyramid, labels, streams
from ..auth.deps import CurrentAuth, DbSession, SettingsDep
from ..db import (
    Artifact,
    Job,
    LabelChunk,
    LabelClass,
    LabelOp,
    Project,
    ProjectMember,
    Roi,
    TrainedModel,
    User,
    UserSession,
)
from ..storage import project_storage
from .projects import MemberProject

router = APIRouter(prefix="/api/projects/{project_id}/labels", tags=["labels"])

MAX_DELTAS = 512
MAX_TOOL_BYTES = 16 * 1024
# Prediction chunks are read on one pool of threads for the whole process, this
# many: reading one takes a round trip or two to the object store (a sharded
# array's index, then the chunk), so a store a way off is waited on this many
# at once, not each in turn, and these are all the process ever has reading,
# however many accepts are under way.
PREDICTION_READERS = 32
# One accept has at most this many reads under way (half the pool), so another
# gets going at once, and a big one doesn't make the others wait for all its
# chunks, as it would if it queued them all.
ACCEPT_READS_AT_ONCE = 16
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


@router.get("/counts")
async def count_labels(project: MemberProject, db: DbSession) -> dict[str, int]:
    """
    Voxels labeled with each value, for background (1) and every class the
    project has: what there is to train on. A value nobody painted says 0.
    """
    live = {
        BACKGROUND,
        *(
            await db.scalars(
                select(LabelClass.value).where(
                    LabelClass.project_id == project.id,
                    LabelClass.deleted_at.is_(None),
                )
            )
        ),
    }
    rows = await db.execute(
        text(
            "SELECT counts.key, sum(counts.value::bigint) "
            "FROM label_chunks CROSS JOIN LATERAL "
            "jsonb_each_text(label_chunks.class_counts) AS counts "
            "WHERE label_chunks.project_id = :project GROUP BY counts.key"
        ),
        {"project": project.id},
    )
    painted = {int(key): int(total) for key, total in rows}
    return {str(value): painted.get(value, 0) for value in sorted(live)}


async def new_class_values(db, project_id: uuid.UUID, count: int) -> list[int]:
    """
    The next `count` class values never used in a project, for new classes.
    """
    # Lock the project (FOR NO KEY UPDATE, which conflicts with itself) so two
    # new classes, or these and the v1 import's, can't take the same value.
    await db.scalar(
        select(Project.id)
        .where(Project.id == project_id)
        .with_for_update(key_share=True)
    )
    highest = await db.scalar(
        select(func.max(LabelClass.value)).where(LabelClass.project_id == project_id)
    )
    first = max(FIRST_CLASS, (highest or 0) + 1)
    room = max(0, MAX_CLASS - first + 1)
    if count > room:
        detail = "This project has used every class value."
        if room:
            classes = "class" if room == 1 else "classes"
            detail = f"This project has room for only {room} more {classes}."
        raise HTTPException(status_code=409, detail=detail)
    return list(range(first, first + count))


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
    [value] = await new_class_values(db, project.id, 1)
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
        if len(json_text(tool, "tool")) > MAX_TOOL_BYTES:
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


# The most one accept may span, as a box or as the labels it sends (16 MiB of
# label values): what a page takes in at once. What the server reads doesn't
# depend on it (see `_check_against_prediction`).
MAX_ACCEPT_VOXELS = 256**3


class AcceptIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_op_id: uuid.UUID
    prediction_artifact_id: uuid.UUID
    # Where the labels go: an ROI, or a box (z0, y0, x0, z1, y1, x1) of whole
    # voxels, half-open, inside the image. Exactly one.
    roi_id: uuid.UUID | None = None
    box: tuple[int, int, int, int, int, int] | None = None
    # One predicted value per delta (an op touches a chunk once), written
    # only into unlabeled voxels.
    deltas: list[DeltaIn] = Field(min_length=1, max_length=MAX_DELTAS)

    @model_validator(mode="after")
    def _one_place(self) -> "AcceptIn":
        if (self.roi_id is None) == (self.box is None):
            raise ValueError("Give exactly one of roi_id and box")
        return self


def _check_size(box: Sequence[int], place: str) -> None:
    """Raise ValueError if `box` (z0, y0, x0, z1, y1, x1) holds too much to accept."""
    if math.prod(box[a + 3] - box[a] for a in range(3)) > MAX_ACCEPT_VOXELS:
        raise ValueError(f"That's too much to accept at once; use a smaller {place}")


async def _check_current_image(db, project_id: uuid.UUID, prediction: Artifact) -> None:
    """
    Refuse a prediction made from an image that has been replaced since, which
    doesn't fit the labels (one that doesn't name its image isn't checked).
    """
    named = prediction.inputs.get("image_artifact_id")
    if named is None:
        return
    image = await artifacts.head(db, project_id, "image")
    if image is None:
        raise labels.NoImage
    if named != str(image.id):
        raise HTTPException(
            status_code=409, detail="That prediction is of an image that was replaced."
        )


_readers: ThreadPoolExecutor | None = None
_readers_made = threading.Lock()


def _prediction_readers() -> ThreadPoolExecutor:
    """
    The process's pool of `PREDICTION_READERS` threads for reading prediction
    chunks, made when first needed (not at import, so each process a server
    forks makes its own). Its threads start as work needs them, are idle
    between accepts, and are stopped, idle, when the process exits.
    """
    global _readers
    with _readers_made:
        if _readers is None:
            _readers = ThreadPoolExecutor(
                PREDICTION_READERS, thread_name_prefix="accept-read"
            )
        return _readers


def _check_against_prediction(grant: StorageGrant, deltas: list[ChunkDelta]) -> None:
    """
    Every voxel a delta selects must hold the value it writes, in the
    prediction at `grant`, and the prediction must reach as far as the labels.

    Only the chunks the labels are in are read, each once. A delta lies in one
    chunk of the labels and so in one of the prediction's (both are 64 voxels
    a side), so that is a chunk for each delta (`MAX_DELTAS` at most) however
    far apart they lie, and one read serves the deltas a request holds for a
    chunk. The reads run on the process's reader threads, `ACCEPT_READS_AT_ONCE`
    of them at most for this accept and `PREDICTION_READERS` in all for every
    accept at once (zarr's synchronous reads are safe from several threads).
    """
    classes: Any = open_prediction(grant)["class"]
    in_chunk: dict[tuple[int, ...], list[tuple[ChunkDelta, tuple[slice, ...]]]] = {}
    for delta in deltas:
        start = [
            k * c + b
            for k, c, b in zip(delta.key, LABEL_CHUNK_ZYX, delta.box[:3], strict=True)
        ]
        region = tuple(
            slice(a, a + n) for a, n in zip(start, delta.box_shape, strict=True)
        )
        in_chunk.setdefault(delta.key, []).append((delta, region))
    # Before reading any of it: a read past the end of an array is cut short.
    if any(
        part.stop > n
        for group in in_chunk.values()
        for _, region in group
        for part, n in zip(region, classes.shape, strict=True)
    ):
        raise ValueError("That prediction doesn't cover those labels")
    _read_chunks(classes, list(in_chunk.values()))


def _read_chunks(
    classes: Any, groups: list[list[tuple[ChunkDelta, tuple[slice, ...]]]]
) -> None:
    """
    Check each chunk's deltas (`_check_chunk`) on the reader threads, with at
    most `ACCEPT_READS_AT_ONCE` of this accept's reads queued or running, so
    its chunks take turns with other accepts' in the pool's queue, one as
    each of its own finishes.

    The first to fail stops this accept's reads that haven't begun, and this
    waits for those that have, so none of them is running when it returns.
    Other accepts' reads are not touched.
    """
    readers = _prediction_readers()
    waiting = iter(groups)
    running: set[Future[None]] = set()
    try:
        while True:
            while (
                len(running) < ACCEPT_READS_AT_ONCE
                and (group := next(waiting, None)) is not None
            ):
                running.add(readers.submit(_check_chunk, classes, group))
            if not running:
                return
            done, running = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
    except BaseException:
        for future in running:
            future.cancel()
        wait(running)
        raise


def _check_chunk(
    classes: Any, group: list[tuple[ChunkDelta, tuple[slice, ...]]]
) -> None:
    """Read the part of a chunk that its deltas are in, and check each against it."""
    low = [min(region[a].start for _, region in group) for a in range(3)]
    high = [max(region[a].stop for _, region in group) for a in range(3)]
    read = np.asarray(
        classes[tuple(slice(lo, hi) for lo, hi in zip(low, high, strict=True))]
    )
    for delta, region in group:
        part = read[
            tuple(
                slice(r.start - lo, r.stop - lo)
                for r, lo in zip(region, low, strict=True)
            )
        ]
        if (part[unpack_mask(delta.mask, delta.box_shape)] != delta.value).any():
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
    model-verified with the prediction, its model, and where they went: the
    ROI, or the box.

    The server checks the claim: each delta writes one predicted value (not
    0) into only unlabeled voxels inside the ROI or box, and the stored
    prediction has exactly that value at every voxel it selects. A box is
    checked as an ROI's is (whole voxels, not empty, inside the image). The
    labels sent may span at most `MAX_ACCEPT_VOXELS` (and so may a box). Only
    the chunks of the prediction the labels are in are read: a chunk for each
    delta, at most `MAX_DELTAS`, whatever the box or ROI is like, on one pool of
    `PREDICTION_READERS` threads shared by every accept in the process, an
    accept having `ACCEPT_READS_AT_ONCE` of its reads under way at most. Undo
    and redo work as for any edit.
    """
    if done := await labels.existing(db, project.id, body.client_op_id):
        return _op_out(done)
    # Share-locked until the op commits: garbage collection locks the
    # artifact and then looks for ops accepted from it, so it either waits
    # and finds this one, or has already started deleting it and this
    # finds nothing.
    prediction = await db.scalar(
        select(Artifact)
        .where(
            Artifact.id == body.prediction_artifact_id,
            Artifact.project_id == project.id,
            Artifact.kind == "prediction",
            Artifact.state.in_(("committed", "superseded")),
        )
        .with_for_update(read=True)
    )
    roi = None
    if body.roi_id is not None:
        roi = await db.scalar(
            select(Roi).where(Roi.id == body.roi_id, Roi.project_id == project.id)
        )
    if prediction is None or (body.roi_id is not None and roi is None):
        what = "prediction or ROI" if body.roi_id is not None else "prediction"
        raise HTTPException(status_code=404, detail=f"No such {what}.")
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
        if roi is not None:
            place, limit, named_for = "ROI", list(roi.bbox), {"roi": str(roi.id)}
        else:
            assert body.box is not None
            place, limit, named_for = "box", list(body.box), {"box": list(body.box)}
            labels.check_box(limit, await labels.volume_shape(db, project.id))
            _check_size(limit, place)
        box = labels.global_box(deltas)
        if any(box[a] < limit[a] or box[a + 3] > limit[a + 3] for a in range(3)):
            raise ValueError(f"Those labels reach outside the {place}")
        _check_size(box, place)
        await _check_current_image(db, project.id, prediction)
        grant = project_storage(settings).child(artifacts.artifact_path(prediction))
        await run_in_threadpool(_check_against_prediction, grant, deltas)
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
                **named_for,
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


class AcceptedOut(BaseModel):
    """
    Where accepted labels came from: a prediction of the whole image or one
    person's proposal, the model that made it (none for a prediction brought
    over from v1, whose job is named instead), and the ROI they went into.
    """

    kind: Literal["prediction", "proposal"]
    model_id: uuid.UUID | None
    model_name: str | None
    v1_job_id: str | None
    roi_id: uuid.UUID | None


class HistoryEntryOut(HistoryOut):
    """
    An op as the history shows it: who made it (a member, or the kind of job
    for edits a job made, such as an import), where accepted labels came
    from, and for an undo or redo, the edit it acts on, as that is now.
    """

    username: str | None
    job_kind: str | None
    accepted: AcceptedOut | None
    target: "HistoryEntryOut | None" = None


def _uuid(value: Any) -> uuid.UUID | None:
    """A UUID written as a string, or None for anything else."""
    try:
        return uuid.UUID(value) if isinstance(value, str) else None
    except ValueError:
        return None


async def _lookup(db, key, value, ids: set[Any], *where) -> dict[Any, Any]:
    """
    `value` by `key` for the rows whose key is in `ids` (and that match
    `where`), in one query.
    """
    ids.discard(None)
    if not ids:
        return {}
    return dict(
        (await db.execute(select(key, value).where(key.in_(ids), *where))).all()
    )


async def _accepted(
    db, project_id: uuid.UUID, ops: Sequence[LabelOp]
) -> dict[int, AcceptedOut]:
    """
    Where the accepted labels among `ops` came from, by seq. Only the server
    makes model-verified edits, with the tool record `accept_prediction`
    writes; anyone's own edits can say anything there, so they don't count.
    """
    accepted = [
        op for op in ops if op.kind == "edit" and op.source == Source.MODEL_VERIFIED
    ]
    if not accepted:
        return {}
    named = {_uuid(op.tool.get("prediction")) for op in accepted} - {None}
    predictions = {
        row.id: row
        for row in await db.execute(
            select(Artifact.id, Artifact.head_slot, Artifact.inputs).where(
                Artifact.project_id == project_id, Artifact.id.in_(named)
            )
        )
    }
    models = await _lookup(
        db,
        TrainedModel.id,
        TrainedModel.name,
        {_uuid(op.tool.get("model")) for op in accepted},
        TrainedModel.project_id == project_id,
    )
    out = {}
    for op in accepted:
        prediction = predictions.get(_uuid(op.tool.get("prediction")))
        slot = (prediction.head_slot or "") if prediction else ""
        v1_job_id = prediction.inputs.get("v1_job_id") if prediction else None
        model_id = _uuid(op.tool.get("model"))
        out[op.seq] = AcceptedOut(
            kind=(
                "proposal"
                if slot.startswith(artifacts.PROPOSAL_SLOTS)
                else "prediction"
            ),
            model_id=model_id,
            model_name=models.get(model_id),
            v1_job_id=v1_job_id if isinstance(v1_job_id, str) else None,
            roi_id=_uuid(op.tool.get("roi")),
        )
    return out


async def _entries(
    db, project_id: uuid.UUID, ops: Sequence[LabelOp]
) -> list[HistoryEntryOut]:
    """`ops` as the history shows them, with every name looked up at once."""
    targets: dict[int, LabelOp] = {}
    if wanted := {op.target_seq for op in ops if op.target_seq is not None}:
        found = await db.scalars(
            select(LabelOp).where(
                LabelOp.project_id == project_id, LabelOp.seq.in_(wanted)
            )
        )
        targets = {op.seq: op for op in found}
    every = [*ops, *targets.values()]
    usernames = await _lookup(db, User.id, User.username, {op.user_id for op in every})
    job_kinds = await _lookup(db, Job.id, Job.kind, {op.job_id for op in every})
    accepted = await _accepted(db, project_id, every)

    def entry(op: LabelOp, target: HistoryEntryOut | None = None) -> HistoryEntryOut:
        return HistoryEntryOut(
            **_history_out(op).model_dump(),
            username=usernames.get(op.user_id),
            job_kind=job_kinds.get(op.job_id),
            accepted=accepted.get(op.seq),
            target=target,
        )

    return [
        entry(op, entry(targets[op.target_seq]) if op.target_seq in targets else None)
        for op in ops
    ]


@router.get("/ops")
async def history(
    project: MemberProject,
    db: DbSession,
    before: Annotated[int | None, Query(ge=1, le=MAX_SEQ)] = None,
    after: Annotated[int | None, Query(ge=0, le=MAX_SEQ)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    user_id: uuid.UUID | None = None,
    source: Source | None = None,
) -> list[HistoryEntryOut]:
    """
    The project's ops, newest first, `limit` at a time, between ops `after`
    and `before` (neither included); only `user_id`'s, or only those of one
    `source` (an undo or redo has its edit's), if asked. Each says who made
    it and, for accepted labels, which model's prediction or proposal they
    came from; an undo or redo comes with its edit.
    """
    query = select(LabelOp).where(LabelOp.project_id == project.id)
    if before is not None:
        query = query.where(LabelOp.seq < before)
    if after is not None:
        query = query.where(LabelOp.seq > after)
    if user_id is not None:
        query = query.where(LabelOp.user_id == user_id)
    if source is not None:
        query = query.where(LabelOp.source == int(source))
    ops = (await db.scalars(query.order_by(LabelOp.seq.desc()).limit(limit))).all()
    return await _entries(db, project.id, ops)


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

    return streams.response(streams.reserve(user_id), stream())


# --- the labels as zarr ----------------------------------------------------


_ARRAY_KEY = re.compile(r"^(class|source|class_([1-9]\d?))/zarr\.json$")
_CHUNK_KEY = re.compile(
    r"^(class|source|class_([1-9]\d?))/c/(\d{1,9})/(\d{1,9})/(\d{1,9})$"
)


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


def _group_metadata(manifest: Mapping, levels: Sequence[LevelSpec]) -> dict:
    # The arrays of the labels' pyramid, which is the image's, listed in
    # OME-Zarr's way (`ome`, for viewers that read it, such as Neuroglancer,
    # which draws the arrays as one volume with the image's voxel size, and
    # picks a level by zoom) and in ours (`ml4paleo`, which the web viewer reads).
    listed = [
        {
            "array": label_pyramid.array_name(level),
            "shape": list(spec.shape_zyx),
            "factor_zyx": list(spec.factor_zyx),
        }
        for level, spec in enumerate(levels)
    ]
    return {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": label_pyramid.multiscales(manifest, levels),
            "ml4paleo": {"label_levels": listed},
        },
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
    project_id = project.id
    try:
        image_id, manifest = await labels.volume_image(db, project_id)
    except labels.NoImage:
        raise HTTPException(
            status_code=404, detail="This project has no image yet."
        ) from None
    levels = label_pyramid.levels_of(manifest)
    if key == "zarr.json":
        return Response(
            json.dumps(_group_metadata(manifest, levels)),
            media_type="application/json",
            headers=revalidate,
        )
    if found := _ARRAY_KEY.match(key):
        level = int(found.group(2) or 0)
        if level >= len(levels):
            raise HTTPException(status_code=404, detail="No such zarr key.")
        return Response(
            json.dumps(_array_metadata(levels[level].shape_zyx)),
            media_type="application/json",
            headers=revalidate,
        )
    match = _CHUNK_KEY.match(key)
    if match is None:
        raise HTTPException(status_code=404, detail="No such zarr key.")
    array, level, cz, cy, cx = (
        match.group(1),
        int(match.group(2) or 0),
        int(match.group(3)),
        int(match.group(4)),
        int(match.group(5)),
    )
    if level >= len(levels):
        raise HTTPException(status_code=404, detail="No such zarr key.")
    if level:
        return await _coarse_chunk(
            request, db, settings, project_id, image_id, levels, level, (cz, cy, cx)
        )
    row = await db.scalar(
        select(LabelChunk).where(
            LabelChunk.project_id == project_id,
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
    root = labels.labels_root(settings, project_id)
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


async def _coarse_chunk(
    request: Request,
    db,
    settings,
    project_id: uuid.UUID,
    image_id: uuid.UUID,
    levels: Sequence[LevelSpec],
    level: int,
    key: tuple[int, int, int],
) -> Response:
    headers = {"Cache-Control": "private, no-cache"}
    retired: frozenset[int] = frozenset()
    if any(k >= n for k, n in zip(key, label_pyramid.grid(levels[level]), strict=True)):
        state = label_pyramid.Fingerprint(count=0, versions=0)
    else:
        # Which retired classes the chunk has is part of the state.
        retired = await label_pyramid.retired_classes(db, project_id)
        state = await label_pyramid.fingerprint(
            db, project_id, levels, level, key, retired
        )
    headers["X-Pyramid-Version"] = str(state.versions)
    if state.count == 0:
        await db.rollback()
        return Response(status_code=404, headers=headers)
    plan = label_pyramid.Plan(image_id, list(levels), retired)
    etag = plan.etag(level, state)
    if request.headers.get("if-none-match") == etag:
        await db.rollback()
        return Response(status_code=304, headers={**headers, "ETag": etag})
    pyramid: label_pyramid.LabelPyramid = request.app.state.label_pyramid
    store = object_store(labels.labels_root(settings, project_id))
    try:
        data = await pyramid.chunk(db, store, project_id, plan, level, key, state)
    except label_pyramid.Busy:
        raise HTTPException(
            status_code=503,
            detail="These labels take a moment to work out; ask again.",
            headers={
                **headers,
                "Retry-After": str(label_pyramid.retry_after(level, key)),
            },
        ) from None
    except label_pyramid.MissingBlob:
        raise HTTPException(
            status_code=500, detail="A label chunk is missing from storage."
        ) from None
    if data is None:
        # Nothing under it shows a label: all erased since the fingerprint, or
        # all of retired classes.
        return Response(status_code=404, headers=headers)
    return Response(
        data, media_type="application/octet-stream", headers={**headers, "ETag": etag}
    )
