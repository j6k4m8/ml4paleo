"""
Label import jobs (see the server's `pipelines/labelimport.py`): labels made
elsewhere, from a file the person uploaded (`ml4paleo.labelimport`).

    labels.probe    check the file is the image's size, and count each
                    value's voxels
    labels.import   send the labels, chunk by chunk, as edits through the
                    server's label writer

Both get the upload as their only grant (read-only).
"""

import base64
import contextlib
import functools
import io
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ml4paleo.ingest import IngestError, SliceLimits
from ml4paleo.labelimport import (
    ImagePlanes,
    Lookup,
    Planes,
    ZipPlanes,
    chunk_blocks,
    chunk_delta,
    count_values,
    file_kind,
)
from ml4paleo.labels.deltas import ChunkDelta
from ml4paleo.storage import open_object

from ..client import with_retries
from ..context import JobContext, PermanentError

UPLOAD_KEY = "data"
COPY_BYTES = 16 * 1024**2
# The namespace of the import's edit ids, one per upload and chunk.
IMPORT_OPS = uuid.UUID("5b0c8f8e-3f43-4a0e-9d7e-4b2f0f6a51c3")
# How long one edit may keep failing on the network or the server before
# the job gives up (and is tried again later).
EDIT_PATIENCE_SECONDS = 120


@contextlib.contextmanager
def _planes(ctx: JobContext) -> Iterator[tuple[str, Planes]]:
    """
    The uploaded file's slices. A TIFF is copied to local disk first: reading
    one of its compressed pages from anything else reads the whole file.
    """
    limits = SliceLimits.for_memory(ctx.memory_budget_bytes)
    with open_object(ctx.grants[0], UPLOAD_KEY) as source:
        try:
            kind = file_kind(source)
            if kind == "zip":
                yield kind, ZipPlanes(source, limits)
                return
            if kind == "png":
                with ImagePlanes(source, limits) as planes:
                    yield kind, planes
                return
            with tempfile.TemporaryDirectory(prefix="m4p-labels-") as scratch:
                size = source.seek(0, io.SEEK_END)
                source.seek(0)
                free = shutil.disk_usage(scratch).free
                if size > free:
                    # Another worker may have room.
                    raise RuntimeError(
                        f"This worker has {free / 1024**3:.1f} GB of disk free, too "
                        f"little to read a {size / 1024**3:.1f} GB TIFF."
                    )
                path = Path(scratch) / "labels.tif"
                with open(path, "wb") as copy:
                    while chunk := source.read(COPY_BYTES):
                        copy.write(chunk)
                        ctx.check()
                with ImagePlanes(path, limits) as planes:
                    yield kind, planes
        except IngestError as exc:
            raise PermanentError(str(exc)) from exc


def probe(ctx: JobContext) -> dict[str, Any]:
    shape = [int(n) for n in ctx.payload["image_shape_zyx"]]

    def progress(fraction: float) -> None:
        ctx.progress(fraction)
        ctx.check()

    with _planes(ctx) as (kind, planes):
        counts = count_values(planes, shape, progress=progress)
    return {
        "format": kind,
        "shape_zyx": shape,
        "values": [[value, counts[value]] for value in sorted(counts)],
    }


def _wire(delta: ChunkDelta) -> dict[str, Any]:
    """A delta as the labels API takes it."""
    wire: dict[str, Any] = {
        "key": list(delta.key),
        "box": list(delta.box),
        "mask": base64.b64encode(delta.mask).decode(),
        "only_if": delta.only_if,
    }
    if delta.values is not None:
        wire["values"] = base64.b64encode(delta.values).decode()
    else:
        wire["value"] = delta.value
    return wire


def run(ctx: JobContext) -> dict[str, Any]:
    """
    Send each chunk's labels as one edit, named by the upload and chunk, so a
    retried job sends each chunk once.
    """
    payload = ctx.payload
    shape = [int(n) for n in payload["shape_zyx"]]
    lookup = Lookup(payload["lookup"])
    only_if = "any" if payload.get("overwrite") else "unlabeled"
    upload_id = payload["upload_id"]
    tool = {
        "name": "label-import",
        "import": payload["import_id"],
        "file": payload["filename"],
    }
    chunks = voxels = 0
    with _planes(ctx) as (_, planes):
        blocks = chunk_blocks(
            planes,
            lookup,
            shape,
            # The rest goes to a slice as it's read (see SliceLimits).
            max_bytes=ctx.memory_budget_bytes // 4,
            progress=ctx.progress,
            check=ctx.check,
        )
        for key, origin, block in blocks:
            op = {
                "client_op_id": str(
                    uuid.uuid5(IMPORT_OPS, "{}/{}/{}/{}".format(upload_id, *key))
                ),
                "deltas": [_wire(chunk_delta(block, origin, only_if))],
                "tool": tool,
            }
            try:
                with_retries(
                    functools.partial(ctx.apply_label_op, op),
                    give_up_after=EDIT_PATIENCE_SECONDS,
                )
            except ValueError as exc:
                raise PermanentError(
                    f"The server refused the labels at {origin}: {exc}"
                ) from exc
            chunks += 1
            voxels += int((block != 0).sum())
    return {"chunks": chunks, "voxels": voxels}
