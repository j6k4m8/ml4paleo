"""
The label writer: applies label edits ("ops") to a project's label chunks,
and undoes and redoes them.

Each op is one transaction:

1. An op the client already sent (the same `client_op_id`) returns the
   result it had, so retried requests apply once. The check runs again once
   the op's locks are held, so a retry that raced the first attempt waits
   for it and returns its result.
2. The op's chunk rows are locked in key order, so concurrent ops can't
   deadlock.
3. A strict op (a polygon fill, an accepted proposal) whose chunks changed
   since the client read them (`base_version`) is refused, naming the chunks;
   other ops (brush strokes) apply to the current state, voxel by voxel.
4. Each chunk's new class and source arrays are stored as content-addressed
   blobs (`projects/<project>/labels/blobs/`), so history never copies data.
5. The op, its claims, and the new chunk versions are recorded, and a
   notification tells other viewers which chunks changed. Ops take their
   `seq` under a per-project lock held until commit, so seqs commit in
   order and readers paging by seq (the change feed) never skip one.

Undo and redo flip an edit between live and undone and recompute its voxels
as the overlay of the live edits' claims (`ml4paleo.labels.deltas`), so the
result never depends on the order of undos and never disturbs voxels a later
edit also wrote.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import func, select, text, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from ml4paleo.labels import LABEL_CHUNK_ZYX, Source
from ml4paleo.labels.codec import blob_key, content_hash, decode_chunk, encode_chunk
from ml4paleo.labels.deltas import ChunkDelta, Claim, apply_delta, recompute
from ml4paleo.ome import LevelSpec
from ml4paleo.storage import StorageGrant, get_bytes, put_bytes

from . import artifacts, label_pyramid
from .db import LabelChunk, LabelOp, LabelOpChunk
from .settings import Settings
from .storage import project_storage

CHANNEL = "m4p_labels"
ChunkKey = tuple[int, int, int]


class NoImage(Exception):
    """
    The project has no image yet, so its labels have no shape.
    """


class Conflict(Exception):
    """
    A strict op's chunks changed since the client read them.
    """

    def __init__(self, keys: list[ChunkKey]):
        super().__init__(f"chunks {keys} changed")
        self.keys = keys


class NotFound(Exception):
    pass


class AlreadyDone(Exception):
    """
    Undoing an edit that is already undone, or redoing a live one.
    """


@dataclass(frozen=True)
class ChunkState:
    key: ChunkKey
    version: int
    class_sha: str | None


@dataclass(frozen=True)
class OpResult:
    seq: int
    chunks: list[ChunkState]


def labels_root(settings: Settings, project_id: uuid.UUID) -> StorageGrant:
    return project_storage(settings).child(f"projects/{project_id}/labels")


async def volume_shape(db: AsyncSession, project_id: uuid.UUID) -> tuple[int, int, int]:
    """
    The label volume's shape (z, y, x): the project image's.
    """
    image = await artifacts.head(db, project_id, "image")
    if image is None or not image.manifest:
        raise NoImage
    _, z, y, x = image.manifest["shape_czyx"]
    return (z, y, x)


async def volume_levels(db: AsyncSession, project_id: uuid.UUID) -> list[LevelSpec]:
    """
    The levels of the labels as zarr: the project image's, whose level 0 is
    the label volume itself.
    """
    image = await artifacts.head(db, project_id, "image")
    if image is None or not image.manifest:
        raise NoImage
    return label_pyramid.levels_of(image.manifest)


async def _read_chunk(grant: StorageGrant, sha: str | None) -> np.ndarray:
    if sha is None:
        return np.zeros(LABEL_CHUNK_ZYX, dtype=np.uint8)
    data = await run_in_threadpool(get_bytes, grant, blob_key(sha))
    if data is None:
        raise RuntimeError(f"Label blob {sha} is missing")
    return await run_in_threadpool(decode_chunk, data)


def _encode(chunk: np.ndarray) -> tuple[str | None, bytes | None]:
    sha = content_hash(chunk)
    return sha, encode_chunk(chunk) if sha is not None else None


async def _write_chunk(grant: StorageGrant, chunk: np.ndarray) -> str | None:
    # Hashing and compressing take milliseconds per chunk; keep them off the
    # event loop.
    sha, data = await run_in_threadpool(_encode, chunk)
    if sha is not None and data is not None:
        await run_in_threadpool(put_bytes, grant, blob_key(sha), data)
    return sha


def _counts(class_chunk: np.ndarray) -> tuple[int, dict[str, int]]:
    values, counts = np.unique(class_chunk, return_counts=True)
    by_value = {str(int(v)): int(c) for v, c in zip(values, counts, strict=True) if v}
    return sum(by_value.values()), by_value


async def _lock_chunks(
    db: AsyncSession, project_id: uuid.UUID, keys: Sequence[ChunkKey]
) -> dict[ChunkKey, LabelChunk]:
    ordered = sorted(set(keys))
    await db.execute(
        insert(LabelChunk)
        .values(
            [
                {"project_id": project_id, "cz": z, "cy": y, "cx": x}
                for z, y, x in ordered
            ]
        )
        .on_conflict_do_nothing()
    )
    rows = (
        await db.scalars(
            select(LabelChunk)
            .where(
                LabelChunk.project_id == project_id,
                tuple_(LabelChunk.cz, LabelChunk.cy, LabelChunk.cx).in_(ordered),
            )
            .order_by(LabelChunk.cz, LabelChunk.cy, LabelChunk.cx)
            .with_for_update(key_share=True)
            .execution_options(populate_existing=True)
        )
    ).all()
    return {(row.cz, row.cy, row.cx): row for row in rows}


async def _store_chunk(
    grant: StorageGrant,
    row: LabelChunk,
    class_chunk: np.ndarray,
    source_chunk: np.ndarray,
) -> None:
    row.class_sha = await _write_chunk(grant, class_chunk)
    row.source_sha = await _write_chunk(grant, source_chunk)
    row.labeled_voxels, row.class_counts = await run_in_threadpool(_counts, class_chunk)
    row.version += 1


async def notify(db: AsyncSession, project_id: uuid.UUID) -> None:
    await db.execute(
        text("SELECT pg_notify(:channel, :project)"),
        {"channel": CHANNEL, "project": str(project_id)},
    )


async def existing(
    db: AsyncSession, project_id: uuid.UUID, client_op_id: uuid.UUID
) -> OpResult | None:
    """
    The result of the op the client sent as `client_op_id`, if it applied.
    """
    op = await db.scalar(
        select(LabelOp).where(
            LabelOp.project_id == project_id, LabelOp.client_op_id == client_op_id
        )
    )
    return await result_of(db, op) if op is not None else None


async def result_of(db: AsyncSession, op: LabelOp) -> OpResult:
    rows = (
        await db.scalars(
            select(LabelOpChunk)
            .where(LabelOpChunk.seq == op.seq)
            .order_by(LabelOpChunk.cz, LabelOpChunk.cy, LabelOpChunk.cx)
        )
    ).all()
    return OpResult(
        seq=op.seq,
        chunks=[
            ChunkState((r.cz, r.cy, r.cx), r.new_version, r.class_sha) for r in rows
        ],
    )


def global_box(deltas: Sequence[ChunkDelta]) -> list[int]:
    """The level-0 box (z0, y0, x0, z1, y1, x1) the deltas cover."""
    starts, stops = [], []
    for delta in deltas:
        origin = [k * s for k, s in zip(delta.key, LABEL_CHUNK_ZYX, strict=True)]
        starts.append([o + b for o, b in zip(origin, delta.box[:3], strict=True)])
        stops.append([o + b for o, b in zip(origin, delta.box[3:], strict=True)])
    return [min(s[i] for s in starts) for i in range(3)] + [
        max(s[i] for s in stops) for i in range(3)
    ]


async def apply_edit(
    db: AsyncSession,
    settings: Settings,
    project_id: uuid.UUID,
    *,
    client_op_id: uuid.UUID,
    deltas: Sequence[ChunkDelta],
    source: Source = Source.HUMAN,
    tool: dict[str, Any] | None = None,
    strict: bool = False,
    user_id: uuid.UUID | None = None,
    job_id: uuid.UUID | None = None,
) -> OpResult:
    """
    Apply one edit (see the module docstring); the caller commits.
    """
    if done := await existing(db, project_id, client_op_id):
        return done
    if not deltas:
        raise ValueError("An edit needs at least one chunk")
    keys = [delta.key for delta in deltas]
    if len(set(keys)) != len(keys):
        raise ValueError("Send one delta per chunk")
    shape = await volume_shape(db, project_id)
    for delta in deltas:
        delta.check_within(shape)
    chunks = await _lock_chunks(db, project_id, keys)
    # A retry of this op that got the locks first has committed by now.
    if done := await existing(db, project_id, client_op_id):
        return done
    if strict:
        stale = sorted(d.key for d in deltas if chunks[d.key].version != d.base_version)
        if stale:
            raise Conflict(stale)
    grant = labels_root(settings, project_id)
    claims = []
    for delta in sorted(deltas, key=lambda d: d.key):
        row = chunks[delta.key]
        class_chunk = await _read_chunk(grant, row.class_sha)
        source_chunk = await _read_chunk(grant, row.source_sha)
        applied = await run_in_threadpool(
            apply_delta, class_chunk, source_chunk, delta, source
        )
        base = row.version
        await _store_chunk(grant, row, applied.class_chunk, applied.source_chunk)
        claims.append(
            LabelOpChunk(
                project_id=project_id,
                cz=delta.key[0],
                cy=delta.key[1],
                cx=delta.key[2],
                base_version=base,
                new_version=row.version,
                class_sha=row.class_sha,
                claim_box=list(applied.claim.box),
                claim_mask=applied.claim.mask,
                claim_values=applied.claim.values,
            )
        )
    op = LabelOp(
        project_id=project_id,
        client_op_id=client_op_id,
        kind="edit",
        user_id=user_id,
        job_id=job_id,
        source=int(source),
        tool=tool or {},
        bbox=global_box(deltas),
        live=True,
    )
    return await _record(db, op, claims)


async def _record(
    db: AsyncSession, op: LabelOp, chunks: list[LabelOpChunk]
) -> OpResult:
    """
    Log an op and the chunk versions it made, and tell viewers.

    The op takes its seq under a per-project lock that its transaction holds
    until it commits, so ops commit in seq order: a reader that sees op N
    already sees every op before it. Ops that share a chunk also take seqs in
    the order they applied, because they hold that chunk's lock here.
    """
    await db.execute(
        select(
            func.pg_advisory_xact_lock(
                func.hashtextextended(f"{CHANNEL}:{op.project_id}", 0)
            )
        )
    )
    # Ops with no chunks in common don't wait for each other's locks, so
    # the same client_op_id sent twice with different chunks lands here.
    if await existing(db, op.project_id, op.client_op_id):
        raise ValueError("That client_op_id was already used for another op")
    db.add(op)
    await db.flush()
    for chunk in chunks:
        chunk.seq = op.seq
        db.add(chunk)
    await db.flush()
    await notify(db, op.project_id)
    return await result_of(db, op)


async def set_live(
    db: AsyncSession,
    settings: Settings,
    project_id: uuid.UUID,
    *,
    target_seq: int,
    live: bool,
    client_op_id: uuid.UUID,
    user_id: uuid.UUID | None = None,
) -> OpResult:
    """
    Undo (`live=False`) or redo (`live=True`) an edit; the caller commits.
    """
    if done := await existing(db, project_id, client_op_id):
        return done
    target = await db.scalar(
        select(LabelOp)
        .where(
            LabelOp.seq == target_seq,
            LabelOp.project_id == project_id,
            LabelOp.kind == "edit",
        )
        .with_for_update(key_share=True)
        .execution_options(populate_existing=True)
    )
    if target is None:
        raise NotFound
    # A retry of this undo that got the edit's lock first has committed.
    if done := await existing(db, project_id, client_op_id):
        return done
    if target.live == live:
        raise AlreadyDone
    claims = (
        await db.scalars(
            select(LabelOpChunk)
            .where(LabelOpChunk.seq == target.seq)
            .order_by(LabelOpChunk.cz, LabelOpChunk.cy, LabelOpChunk.cx)
        )
    ).all()
    chunks = await _lock_chunks(db, project_id, [(c.cz, c.cy, c.cx) for c in claims])
    target.live = live
    grant = labels_root(settings, project_id)
    made = []
    for claim_row in claims:
        key = (claim_row.cz, claim_row.cy, claim_row.cx)
        region = _claim(claim_row, Source(target.source))
        live_claims = [
            _claim(row, Source(source))
            for row, source in (
                await db.execute(
                    select(LabelOpChunk, LabelOp.source)
                    .join(LabelOp, LabelOp.seq == LabelOpChunk.seq)
                    .where(
                        LabelOpChunk.project_id == project_id,
                        LabelOpChunk.cz == key[0],
                        LabelOpChunk.cy == key[1],
                        LabelOpChunk.cx == key[2],
                        LabelOp.kind == "edit",
                        LabelOp.live,
                    )
                    .order_by(LabelOpChunk.seq)
                )
            ).all()
        ]
        row = chunks[key]
        class_chunk = await _read_chunk(grant, row.class_sha)
        source_chunk = await _read_chunk(grant, row.source_sha)
        recomputed = await run_in_threadpool(
            recompute, class_chunk, source_chunk, region, live_claims
        )
        base = row.version
        await _store_chunk(grant, row, recomputed.class_chunk, recomputed.source_chunk)
        made.append(
            LabelOpChunk(
                project_id=project_id,
                cz=key[0],
                cy=key[1],
                cx=key[2],
                base_version=base,
                new_version=row.version,
                class_sha=row.class_sha,
            )
        )
    op = LabelOp(
        project_id=project_id,
        client_op_id=client_op_id,
        kind="redo" if live else "undo",
        user_id=user_id,
        source=target.source,
        tool={},
        bbox=target.bbox,
        target_seq=target.seq,
        live=True,
    )
    return await _record(db, op, made)


def _claim(row: LabelOpChunk, source: Source) -> Claim:
    assert row.claim_box is not None and row.claim_mask is not None
    assert row.claim_values is not None
    return Claim(
        key=(row.cz, row.cy, row.cx),
        box=tuple(row.claim_box),  # type: ignore[arg-type]
        mask=row.claim_mask,
        values=row.claim_values,
        source=source,
    )


async def changes_since(
    db: AsyncSession, project_id: uuid.UUID, after_seq: int, limit: int = 200
) -> list[tuple[LabelOp, list[ChunkState]]]:
    """
    Ops after `after_seq`, oldest first, with the chunk versions they made.
    Ops commit in seq order (see `_record`), so paging by seq skips none.
    """
    ops = (
        await db.scalars(
            select(LabelOp)
            .where(LabelOp.project_id == project_id, LabelOp.seq > after_seq)
            .order_by(LabelOp.seq)
            .limit(limit)
        )
    ).all()
    return [(op, (await result_of(db, op)).chunks) for op in ops]


__all__ = [
    "CHANNEL",
    "AlreadyDone",
    "ChunkState",
    "Conflict",
    "NoImage",
    "NotFound",
    "OpResult",
    "apply_edit",
    "changes_since",
    "existing",
    "global_box",
    "labels_root",
    "result_of",
    "set_live",
    "volume_levels",
    "volume_shape",
]
