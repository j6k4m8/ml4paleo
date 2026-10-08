"""
The coarse levels of a project's labels, made for viewers as they ask.

A viewer zoomed out far enough to see the whole volume can't fetch every
full-resolution label chunk, so the label zarr also offers `class_1`,
`class_2`, ... : the labels at the image's pyramid levels 1, 2, ... (`class` is
level 0). They have the image's level shapes, and a voxel at level k covers the
same block of the volume as the image's voxel there. Chunks stay 64-cubed.

These levels exist only to be looked at. Nothing stores them, edits never
touch them, and training, exports, accepts, and segmentation read the full
resolution labels. How a level shrinks the one below it (`downsample_labels`)
is in `ml4paleo.labels.pyramid`.

A chunk of level k is made from the chunks of level k - 1 under it (level 1,
from the full resolution chunks), when someone asks for it:

- What it was made from is known without reading any pixels. A level-k chunk
  covers a box of full-resolution chunks (`footprint`), and every edit of one
  of those chunks (an undo or redo too) raises its version. So the sum of the
  versions of the chunks in the box says exactly which state of the labels the
  chunk shows: it goes up whenever anything in the box changes. One query on
  the label chunk rows gives it, with how many of the chunks have labels. The
  sum is the chunk's `X-Chunk-Version`, and its `ETag` has it too, so a
  viewer's revalidation costs one query. A chunk with no labeled chunk under
  it is simply missing (unlabeled), as at level 0. The `ETag` also names what
  else the pixels follow: the image whose levels these are (replacing it can
  change them, labels unchanged), and `RULE_VERSION`. It is only unique to its
  URL: two chunks can have the same one.
- A computed chunk is kept, as the bytes a viewer gets, in a cache that holds
  the most recently used ones under a memory limit. Its key includes that sum,
  so an edit leaves the stale copy behind rather than finding it. An edit
  therefore costs recomputing the chunk above it at each level, each from one
  new chunk and seven cached ones, and an untouched chunk is never recomputed.
- Nothing is cached below level 1, so the first request for a chunk high in
  the pyramid has to build everything under it, which can mean reading every
  labeled chunk below. One request works on that for about a second and a half,
  or until it has read 512 stored chunks (`seconds`, `blobs`), and always
  finishes the chunk it is on. Then it gives up with `Busy`, having cached what
  it made. The viewer asks again and carries on from there, so no request holds
  the server for long however big the volume, and each is cheap once the levels
  below it are cached.
- What a build finished is kept until the chunk above it is made. A request
  that gives up has only its finished chunks (the ones under a chunk not yet
  made) to show for it, and the next request, perhaps in another process's
  turn, needs them. So they are pinned: they go last when the cache is full,
  and are released when the chunk above them is made, or after two minutes if
  nobody asks again, so a build nobody comes back to can't hold memory.
- Requests share the work, and the memory it takes is bounded. A chunk is
  built by one request at a time: another that wants it waits for that build
  (and gets `Busy` if it gives up). A chunk of level 1 is read and shrunk, or a
  chunk of a higher level combined, only when one of a few slots is free, and a
  request that would queue behind too many others gets `Busy` at once. The
  slots are never held while waiting for another chunk, so they can't deadlock;
  a request holds no database connection while it waits for one.

Each process has its own cache.
"""

import asyncio
import contextlib
import functools
import hashlib
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import obstore
from sqlalchemy import Integer, and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from ml4paleo.labels import LABEL_CHUNK_ZYX
from ml4paleo.labels.codec import blob_key, decode_chunk, encode_chunk
from ml4paleo.labels.pyramid import downsample_labels
from ml4paleo.ome import DEFAULT_CHUNK_ZYX, LevelSpec, plan_levels

from .db import LabelChunk

log = logging.getLogger(__name__)

ChunkKey = tuple[int, int, int]
Box = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

# Bump when `downsample_labels` or the chunk codec changes the bytes a chunk
# is made of, so viewers drop the chunks they kept. A test pins both.
RULE_VERSION = 2
# Memory for cached chunks, in each API process.
CACHE_BYTES = 64 * 1024 * 1024
# How long one request works on a chunk, and how many stored chunks it reads,
# before it gives up and asks the viewer to come back. Shrinking a chunk takes
# from under a millisecond (a few classes in big regions) to 5 ms (many classes,
# mixed everywhere), so a request makes some hundreds.
SECONDS = 1.5
BLOBS = 512
# Pieces of work (the chunks under a chunk of level 1 read and shrunk, or a
# chunk of a higher level combined) running at once in each process, and how
# many more may wait for a slot before a request is turned away.
BUILD_SLOTS = 2
BUILD_QUEUE = 8
# How much longer than its own budget a request waits for another's build.
WAIT_SECONDS = 5
# How long chunks a build finished stay pinned without anyone using them, and
# the least memory (more, if the cache is bigger) they may take in all. A build
# has at most seven chunks per level pinned, so this is room for several.
PIN_SECONDS = 120
PIN_BYTES = 16 * 1024 * 1024
# What a cached chunk costs besides its bytes (its key, and the cache's).
ENTRY_OVERHEAD = 256


class Busy(Exception):
    """
    The chunk takes more work than one request does. What was done is kept:
    ask again.
    """


class MissingBlob(Exception):
    """
    A label chunk the database lists isn't in storage.
    """


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """
    Which state of the labels a chunk shows: how many full-resolution chunks
    under it have labels, and the sum of the versions of all the chunks under
    it, erased ones too (which only goes up).
    """

    count: int
    versions: int


@dataclass(frozen=True)
class Plan:
    """
    What a project's coarse levels follow besides its labels: the image whose
    levels they share.
    """

    image: uuid.UUID
    levels: list[LevelSpec]

    @functools.cached_property
    def tag(self) -> str:
        return hashlib.blake2s(self.image.bytes, digest_size=6).hexdigest()

    def etag(self, level: int, state: Fingerprint) -> str:
        z, y, x = self.levels[level].factor_zyx
        return f'"p{RULE_VERSION}.{self.tag}.{z}.{y}.{x}.{state.versions}"'


def array_name(level: int) -> str:
    """
    The label zarr's array for a level of the pyramid.
    """
    return "class" if level == 0 else f"class_{level}"


def levels_of(manifest: Mapping) -> list[LevelSpec]:
    """
    The levels of a project's labels, which are its image's. Images don't
    record them, but a pyramid follows from the shape and voxel size, so
    this plans it as ingest did. If the image has a different number of levels
    than that plan, it was made some other way, and only level 0 is offered.
    """
    z, y, x = manifest["shape_czyx"][1:]
    try:
        levels = plan_levels(
            (z, y, x), manifest.get("voxel_size_zyx"), DEFAULT_CHUNK_ZYX
        )
    except (TypeError, ValueError) as exc:
        _warn(f"An image's levels can't be planned: {exc}")
        return [LevelSpec("0", (z, y, x), (1, 1, 1))]
    declared = manifest.get("levels")
    if isinstance(declared, int) and declared != len(levels):
        _warn(f"An image has {declared} levels, not the {len(levels)} planned")
        return levels[:1]
    return levels


@functools.cache
def _warn(message: str) -> None:
    # Once per message: this runs for every request.
    log.warning(message)


def grid(level: LevelSpec) -> ChunkKey:
    """
    How many chunks a level has along each axis.
    """
    z, y, x = (
        -(-n // c) for n, c in zip(level.shape_zyx, LABEL_CHUNK_ZYX, strict=True)
    )
    return (z, y, x)


def footprint(level: LevelSpec, key: ChunkKey) -> Box:
    """
    The full-resolution chunks under a chunk of `level`, as half-open ranges
    of chunk positions. A chunk covers 64 voxels of its level along each axis,
    which are 64 times the level's factor full-resolution voxels.
    """
    z, y, x = ((c * f, (c + 1) * f) for c, f in zip(key, level.factor_zyx, strict=True))
    return (z, y, x)


def _step(levels: Sequence[LevelSpec], level: int) -> ChunkKey:
    """
    How many voxels of `level - 1` make one voxel of `level`, along each axis.
    """
    below = levels[level - 1].factor_zyx
    z, y, x = (a // b for a, b in zip(levels[level].factor_zyx, below, strict=True))
    return (z, y, x)


def _within(project_id: uuid.UUID, box: Box):
    (z0, z1), (y0, y1), (x0, x1) = box
    return and_(
        LabelChunk.project_id == project_id,
        LabelChunk.cz >= z0,
        LabelChunk.cz < z1,
        LabelChunk.cy >= y0,
        LabelChunk.cy < y1,
        LabelChunk.cx >= x0,
        LabelChunk.cx < x1,
    )


# Chunks that have labels, and the versions of every chunk in the box.
_COUNT = func.count(LabelChunk.class_sha)
_VERSIONS = func.coalesce(func.sum(LabelChunk.version), 0)


async def fingerprint(
    db: AsyncSession,
    project_id: uuid.UUID,
    levels: Sequence[LevelSpec],
    level: int,
    key: ChunkKey,
) -> Fingerprint:
    """
    Which state of the labels a chunk of `level` shows (one query).
    """
    box = footprint(levels[level], key)
    row = (
        await db.execute(select(_COUNT, _VERSIONS).where(_within(project_id, box)))
    ).one()
    return Fingerprint(count=int(row[0]), versions=int(row[1]))


async def _children(
    db: AsyncSession,
    project_id: uuid.UUID,
    levels: Sequence[LevelSpec],
    level: int,
    key: ChunkKey,
) -> dict[ChunkKey, Fingerprint]:
    """
    The chunks of `level - 1` under a chunk of `level` that have labels, and
    their fingerprints, in one query (grouping the full-resolution chunks by
    the chunk of `level - 1` they are in).
    """
    fz, fy, fx = levels[level - 1].factor_zyx
    gz = LabelChunk.cz.op("/", return_type=Integer)(fz).label("gz")
    gy = LabelChunk.cy.op("/", return_type=Integer)(fy).label("gy")
    gx = LabelChunk.cx.op("/", return_type=Integer)(fx).label("gx")
    rows = await db.execute(
        select(gz, gy, gx, _COUNT, _VERSIONS)
        .where(_within(project_id, footprint(levels[level], key)))
        .group_by("gz", "gy", "gx")
    )
    return {
        (z, y, x): Fingerprint(count=int(count), versions=int(versions))
        for z, y, x, count, versions in rows
        if count
    }


async def _leaves(
    db: AsyncSession,
    project_id: uuid.UUID,
    levels: Sequence[LevelSpec],
    key: ChunkKey,
) -> list[tuple[ChunkKey, str]]:
    """
    The full-resolution chunks with labels under a chunk of level 1, and their
    blobs.
    """
    rows = await db.execute(
        select(LabelChunk.cz, LabelChunk.cy, LabelChunk.cx, LabelChunk.class_sha)
        .where(
            _within(project_id, footprint(levels[1], key)),
            LabelChunk.class_sha.is_not(None),
        )
        .order_by(LabelChunk.cz, LabelChunk.cy, LabelChunk.cx)
    )
    return [((z, y, x), sha) for z, y, x, sha in rows if sha is not None]


async def _read(store, sha: str) -> bytes:
    try:
        result = await obstore.get_async(store, blob_key(sha))
    except FileNotFoundError:
        raise MissingBlob(sha) from None
    return bytes(await result.bytes_async())


def _combine(parts: Sequence[tuple[ChunkKey, bytes]], step: ChunkKey) -> bytes | None:
    """
    Make the chunk covering `parts`, which are chunks (as stored) at the given
    positions among the `step` chunks along each axis that it covers.
    """
    made = np.zeros(LABEL_CHUNK_ZYX, dtype=np.uint8)
    size = [n // s for n, s in zip(LABEL_CHUNK_ZYX, step, strict=True)]
    for position, data in parts:
        start = [p * n for p, n in zip(position, size, strict=True)]
        place = tuple(slice(a, a + n) for a, n in zip(start, size, strict=True))
        made[place] = downsample_labels(decode_chunk(data), step)
    return encode_chunk(made) if made.any() else None


class _Cache:
    """
    Byte strings under a memory limit, least recently used first to go. Some
    can be pinned, which exempts them from the limit (they go last, and only
    when the pinned ones themselves exceed `pin_limit`, oldest first) until
    they are unpinned or go unused for `pin_seconds`.
    """

    def __init__(
        self,
        limit: int,
        pin_limit: int | None = None,
        pin_seconds: float = PIN_SECONDS,
        clock=time.monotonic,
    ):
        self.limit = limit
        self.pin_limit = max(limit, PIN_BYTES) if pin_limit is None else pin_limit
        self.pin_seconds = pin_seconds
        self.size = 0
        self.pinned_size = 0
        self._clock = clock
        self._items = OrderedDict[Hashable, bytes]()
        # Pinned entries, in the order their pins end.
        self._pinned = OrderedDict[Hashable, tuple[bytes, float]]()

    def get(self, key: Hashable) -> bytes | None:
        if (data := self._items.get(key)) is not None:
            self._items.move_to_end(key)
            return data
        if (held := self._pinned.get(key)) is not None:
            # Still wanted: the pin starts over.
            self._pinned[key] = (held[0], self._clock() + self.pin_seconds)
            self._pinned.move_to_end(key)
            return held[0]
        return None

    def put(self, key: Hashable, data: bytes, pin: bool = False) -> None:
        cost = len(data) + ENTRY_OVERHEAD
        if cost > (self.pin_limit if pin else self.limit):
            return
        self._drop(key)
        if pin:
            self._pinned[key] = (data, self._clock() + self.pin_seconds)
            self.pinned_size += cost
        else:
            self._items[key] = data
            self.size += cost
        self._trim()

    def unpin(self, key: Hashable) -> None:
        """
        Let go of a pin: the entry stays, as the most recently used.
        """
        if (held := self._pinned.pop(key, None)) is not None:
            cost = len(held[0]) + ENTRY_OVERHEAD
            self.pinned_size -= cost
            self.size += cost
            self._items[key] = held[0]
            self._trim()

    def _drop(self, key: Hashable) -> None:
        if (old := self._items.pop(key, None)) is not None:
            self.size -= len(old) + ENTRY_OVERHEAD
        if (held := self._pinned.pop(key, None)) is not None:
            self.pinned_size -= len(held[0]) + ENTRY_OVERHEAD

    def _release(self, key: Hashable, first: bool) -> None:
        data, _ = self._pinned.pop(key)
        cost = len(data) + ENTRY_OVERHEAD
        self.pinned_size -= cost
        self.size += cost
        self._items[key] = data
        if first:
            self._items.move_to_end(key, last=False)

    def _trim(self) -> None:
        # Pins that nobody came back for end; the oldest go if there are too many.
        now = self._clock()
        while self._pinned:
            key, (_, until) = next(iter(self._pinned.items()))
            if until > now and self.pinned_size <= self.pin_limit:
                break
            self._release(key, first=True)
        while self.size > self.limit:
            _, gone = self._items.popitem(last=False)
            self.size -= len(gone) + ENTRY_OVERHEAD

    @property
    def total(self) -> int:
        return self.size + self.pinned_size


class _Budget:
    """
    How long a request may keep working, and how many stored chunks it may
    read, checked as it goes from chunk to chunk. The first chunk is always
    done, so every request gets somewhere.
    """

    def __init__(self, seconds: float, blobs: int):
        self.deadline = time.monotonic() + seconds
        self.blobs = blobs
        self.started = False

    def spend(self, blobs: int = 0) -> None:
        self.blobs -= blobs
        if self.started and (self.blobs < 0 or time.monotonic() > self.deadline):
            raise Busy
        self.started = True


class _Gate:
    """
    Lets a few pieces of work run at once, and a few more wait for a turn; one
    that would wait behind more is turned away (`Busy`) instead of held.
    """

    def __init__(self, slots: int, queue: int):
        self._slots = asyncio.Semaphore(slots)
        self._queue = queue
        self._waiting = 0

    @contextlib.asynccontextmanager
    async def __call__(self):
        if self._slots.locked() and self._waiting >= self._queue:
            raise Busy
        self._waiting += 1
        try:
            await self._slots.acquire()
        finally:
            self._waiting -= 1
        try:
            yield
        finally:
            self._slots.release()


class LabelPyramid:
    """
    Computes, and keeps, chunks of a project's coarse label levels. Use one
    per process, from one event loop.
    """

    def __init__(
        self,
        cache_bytes: int = CACHE_BYTES,
        seconds: float = SECONDS,
        blobs: int = BLOBS,
        slots: int = BUILD_SLOTS,
        queue: int = BUILD_QUEUE,
    ):
        self.seconds = seconds
        self.blobs = blobs
        self._cache = _Cache(cache_bytes)
        self._gate = _Gate(slots, queue)
        # The chunks being built now, by cache key, and what each will be.
        self._building: dict[Hashable, asyncio.Future[bytes | None]] = {}

    async def chunk(
        self,
        db: AsyncSession,
        store,
        project_id: uuid.UUID,
        plan: Plan,
        level: int,
        key: ChunkKey,
        state: Fingerprint,
    ) -> bytes | None:
        """
        A chunk of `level` (1 or more), as zarr chunk bytes; None if nothing
        under it shows a label. `state` says which state to compute, from
        `fingerprint`. Raises `Busy` if that takes more than the budget. The
        session's connection is given back, as storage and computing are slow.
        """
        return await self._build(
            db,
            store,
            project_id,
            plan,
            level,
            key,
            state,
            _Budget(self.seconds, self.blobs),
            pin=False,
        )

    async def _build(
        self,
        db: AsyncSession,
        store,
        project_id: uuid.UUID,
        plan: Plan,
        level: int,
        key: ChunkKey,
        state: Fingerprint,
        budget: _Budget,
        pin: bool,
    ) -> bytes | None:
        cached = _cache_key(project_id, plan, level, key, state)
        if (data := self._cache.get(cached)) is not None:
            return data
        if (building := self._building.get(cached)) is not None:
            # Someone is already making it: wait for that, not holding a
            # connection, and share whatever comes of it.
            await db.rollback()
            return await self._wait(building)
        made: asyncio.Future[bytes | None] = asyncio.get_running_loop().create_future()
        self._building[cached] = made
        try:
            data = await self._make(
                db, store, project_id, plan, level, key, cached, budget, pin
            )
        except BaseException as exc:
            # Anyone waiting can only ask again, whatever stopped this.
            made.set_exception(exc if isinstance(exc, Exception) else Busy())
            made.exception()
            raise
        else:
            made.set_result(data)
            return data
        finally:
            del self._building[cached]

    async def _wait(self, building: asyncio.Future[bytes | None]) -> bytes | None:
        try:
            return await asyncio.wait_for(
                asyncio.shield(building), self.seconds + WAIT_SECONDS
            )
        except TimeoutError:
            raise Busy from None

    async def _make(
        self,
        db: AsyncSession,
        store,
        project_id: uuid.UUID,
        plan: Plan,
        level: int,
        key: ChunkKey,
        cached: Hashable,
        budget: _Budget,
        pin: bool,
    ) -> bytes | None:
        levels = plan.levels
        step = _step(levels, level)
        origin = (key[0] * step[0], key[1] * step[1], key[2] * step[2])
        kept: list[Hashable] = []
        if level == 1:
            leaves = await _leaves(db, project_id, levels, key)
            budget.spend(len(leaves))
            await db.rollback()
            async with self._gate():
                blobs = await asyncio.gather(*(_read(store, sha) for _, sha in leaves))
                parts = [
                    (_position(at, origin), blob)
                    for (at, _), blob in zip(leaves, blobs, strict=True)
                ]
                data = await run_in_threadpool(_combine, parts, step)
        else:
            parts = []
            below = await _children(db, project_id, levels, level, key)
            for child, child_state in sorted(below.items()):
                # What is made of a chunk under this one is kept (pinned)
                # until this one is made, so a request that gives up before
                # then leaves its work for the next.
                made = await self._build(
                    db,
                    store,
                    project_id,
                    plan,
                    level - 1,
                    child,
                    child_state,
                    budget,
                    pin=True,
                )
                if made is not None:
                    parts.append((_position(child, origin), made))
                    kept.append(
                        _cache_key(project_id, plan, level - 1, child, child_state)
                    )
            budget.spend()
            await db.rollback()
            async with self._gate():
                data = await run_in_threadpool(_combine, parts, step)
        if data is not None:
            self._cache.put(cached, data, pin=pin)
        for child_key in kept:
            self._cache.unpin(child_key)
        return data


def _cache_key(
    project_id: uuid.UUID, plan: Plan, level: int, key: ChunkKey, state: Fingerprint
) -> Hashable:
    return (project_id, plan.tag, level, key, state)


def _position(key: ChunkKey, origin: ChunkKey) -> ChunkKey:
    """
    Where a chunk is among the ones under a chunk starting at `origin`.
    """
    z, y, x = (k - o for k, o in zip(key, origin, strict=True))
    return (z, y, x)
