"""
Importing labels from a file (a TIFF stack, or a zip of TIFF or PNG slices;
see `ml4paleo.labelimport`), as two pipelines of one job each:

    labels.probe    read the file once: check it's the image's size and
                    count each value's voxels
    labels.import   bring the values into the project's labels, chunk by
                    chunk, as edits (`Source.IMPORTED`)

In between, the person says what each value the check found becomes (a
class, background, or nothing), so the import's payload carries a lookup
from the file's values to label values. The check's id names the import.

An upload is checked once and imported once. Its edits are named by the
upload and chunk, so a retried import job applies each once; they fill only
voxels nobody labeled, unless the person asked to overwrite. The upload counts
against the owner's storage like any other until it goes: two days after the
check if it's never imported, and as soon as the import ends however it ends
(garbage collection deletes it once no job uses it).
"""

import datetime
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ml4paleo.labelimport import MAX_VALUES
from ml4paleo.labels import BACKGROUND, MAX_CLASS

from .. import jobs
from ..db import Job, Upload
from ..uploads import now, upload_path

PROBE = "labels.probe"
IMPORT = "labels.import"
# An upload that was checked but never imported goes after this long.
KEEP_UNIMPORTED = datetime.timedelta(days=2)
MAX_SHAPE = 2**31


async def start_check(
    db: AsyncSession,
    upload: Upload,
    image_shape_zyx: list[int],
    created_by: uuid.UUID,
) -> Job:
    """
    Check a finished upload against the image (its shape, `z, y, x`), or
    return the upload's check if it has one. The caller holds a lock on the
    upload.
    """
    upload.expires_at = min(upload.expires_at, now() + KEEP_UNIMPORTED)
    return await jobs.enqueue(
        db,
        PROBE,
        {
            "upload_id": str(upload.id),
            "filename": upload.filename,
            "image_shape_zyx": list(image_shape_zyx),
        },
        project_id=upload.project_id,
        created_by=created_by,
        grants=[{"path": upload_path(upload), "access": "r"}],
        idempotency_key=f"{PROBE}:{upload.id}",
    )


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def check_probe_result(result: dict[str, Any]) -> None:
    """
    Refuse a check's result that an import can't be built from.
    """
    if result.get("format") not in ("tiff", "png", "zip"):
        raise ValueError("format must be tiff, png, or zip")
    shape = result.get("shape_zyx")
    if not (
        isinstance(shape, list)
        and len(shape) == 3
        and all(_count(n) and 0 < n < MAX_SHAPE for n in shape)
    ):
        raise ValueError("shape_zyx must be three positive integers")
    values = result.get("values")
    if not isinstance(values, list) or len(values) > MAX_VALUES + 1:
        raise ValueError(f"values must be a list of up to {MAX_VALUES + 1} pairs")
    for pair in values:
        if not (
            isinstance(pair, list)
            and len(pair) == 2
            and isinstance(pair[0], int)
            and not isinstance(pair[0], bool)
            and _count(pair[1])
            and pair[1] > 0
        ):
            raise ValueError("each value must be [value, voxels]")
    found = [pair[0] for pair in values]
    if found != sorted(set(found)):
        raise ValueError("values must be distinct and in order")


async def start_import(
    db: AsyncSession,
    probe: Job,
    upload: Upload,
    lookup: list[tuple[int, int]],
    *,
    overwrite: bool,
    created_by: uuid.UUID,
) -> Job:
    """
    Import a checked file's labels through `lookup` (the file's values and
    the label values they become). The upload goes as soon as the import
    ends.
    """
    assert probe.result is not None
    if any(not BACKGROUND <= label <= MAX_CLASS for _, label in lookup):
        raise ValueError("Labels must be background or a class")
    job = await jobs.enqueue(
        db,
        IMPORT,
        {
            "import_id": str(probe.id),
            "upload_id": str(upload.id),
            "filename": upload.filename,
            "shape_zyx": probe.result["shape_zyx"],
            "lookup": [list(pair) for pair in sorted(lookup)],
            "overwrite": overwrite,
        },
        project_id=probe.project_id,
        created_by=created_by,
        grants=[{"path": upload_path(upload), "access": "r"}],
        idempotency_key=f"{IMPORT}:{probe.id}",
    )
    upload.expires_at = now()
    return job


def check_import_result(result: dict[str, Any]) -> None:
    for key in ("chunks", "voxels"):
        if not _count(result.get(key)):
            raise ValueError(f"{key} must be a count")


async def checks(db: AsyncSession, project_id: uuid.UUID, limit: int = 10) -> list[Job]:
    """A project's newest checks (each names an import), newest first."""
    return list(
        (
            await db.scalars(
                select(Job)
                .where(Job.project_id == project_id, Job.kind == PROBE)
                .order_by(Job.created_at.desc(), Job.id.desc())
                .limit(limit)
            )
        ).all()
    )


async def import_of(db: AsyncSession, probe: Job) -> Job | None:
    """The import started from a check, if it has been."""
    return await db.scalar(
        select(Job).where(Job.idempotency_key == f"{IMPORT}:{probe.id}")
    )
