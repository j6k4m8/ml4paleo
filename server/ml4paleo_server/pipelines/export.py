"""
Exports: a head artifact as one zip archive to download, kept for a week.

    export.files    every file: the image or a label volume as a zarr group
                    at the archive's root, or the meshes, named by class
    export.images   one TIFF or PNG per slice, as the stack was uploaded

An export is an "export" artifact with no head slot. Its `cache_key` names
the source artifact and the format, so asking again while it's kept gives
the same archive (and keeps it another week). Garbage collection deletes it
once it expires; until then it counts against the owner's storage.
"""

import datetime
import re
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from .. import artifacts, jobs
from ..db import Artifact, Job

# What each head can be exported as.
FORMATS: dict[str, tuple[str, ...]] = {
    "image": ("zarr", "tiff", "png"),
    "prediction": ("zarr", "tiff", "png"),
    "segmentation": ("zarr", "tiff", "png"),
    "meshes": ("zip",),
}
KEEP = datetime.timedelta(days=7)
# Part of the cache key: bump it when archives change, so old ones aren't
# handed out again.
LAYOUT = 1


def cache_key(source: Artifact, fmt: str) -> str:
    return f"{source.id}/{fmt}/{LAYOUT}"


def filename(project_name: str, slot: str, fmt: str) -> str:
    """
    The archive's name, such as `skull-image.ome.zarr.zip`.
    """
    stem = re.sub(r"[^a-z0-9]+", "-", project_name.lower()).strip("-") or "project"
    stem = f"{stem}-{slot}"
    if fmt == "zarr":
        return f"{stem}.ome.zarr.zip" if slot == "image" else f"{stem}.zarr.zip"
    if fmt == "zip":
        return f"{stem}.zip"
    return f"{stem}-{fmt}.zip"


async def start(
    db: AsyncSession,
    *,
    source: Artifact,
    slot: str,
    fmt: str,
    project_name: str,
    created_by: uuid.UUID,
) -> tuple[Job, Artifact]:
    name = filename(project_name, slot, fmt)
    export = await artifacts.create_staging(
        db,
        project_id=source.project_id,
        kind="export",
        inputs={
            "source": slot,
            "source_artifact_id": str(source.id),
            "format": fmt,
            "filename": name,
        },
        cache_key=cache_key(source, fmt),
        expires_at=artifacts.now() + KEEP,
    )
    # A zarr group sits at the archive's root (zarr reads it there); other
    # files go in a folder named like the archive.
    folder = "" if fmt == "zarr" else name.removesuffix(".zip") + "/"
    job = await jobs.enqueue(
        db,
        "export.images" if fmt in ("tiff", "png") else "export.files",
        {"source": slot, "format": fmt, "folder": folder},
        project_id=source.project_id,
        created_by=created_by,
        grants=[artifacts.grant_for(source, "r"), artifacts.grant_for(export)],
    )
    export.produced_by_job = job.id
    await db.flush()
    return job, export
