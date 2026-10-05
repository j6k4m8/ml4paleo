"""
The ingest pipeline: an upload becomes the project's image.

    ingest.probe -> ingest.slab x N -> pyramid.level 1 .. L-1 -> artifact.finalize

`start` creates the image artifact (staging) and the probe job. When the
probe succeeds, `after_probe` adds the rest from what it found: one slab job
per shard-deep slab (they run in parallel), one job per pyramid level, and
the finalize job, whose success commits the artifact as the project's image.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from .. import artifacts, jobs
from ..db import Artifact, Job, Upload
from ..uploads import upload_path

# How much of the pipeline's progress bar each part is.
WEIGHTS = {"probe": 2.0, "slabs": 70.0, "pyramid": 20.0, "finalize": 3.0}


async def start(
    db: AsyncSession, upload: Upload, created_by: uuid.UUID
) -> tuple[Job, Artifact]:
    artifact = await artifacts.create_staging(
        db,
        project_id=upload.project_id,
        kind="image",
        head_slot="image",
        inputs={"upload_id": str(upload.id), "filename": upload.filename},
    )
    grants = [
        artifacts.grant_for(artifact),
        {"path": upload_path(upload), "access": "r"},
    ]
    probe = await jobs.enqueue(
        db,
        "ingest.probe",
        {
            "artifact_id": str(artifact.id),
            "upload_id": str(upload.id),
            "filename": upload.filename,
        },
        project_id=upload.project_id,
        created_by=created_by,
        grants=grants,
        weight=WEIGHTS["probe"],
    )
    return probe, artifact


def check_probe_result(result: dict) -> None:
    """
    Refuse a probe result the rest of the pipeline can't be built from.
    """
    shape = result.get("shape_zyx")
    if not (
        isinstance(shape, list)
        and len(shape) == 3
        and all(isinstance(n, int) and n > 0 for n in shape)
    ):
        raise ValueError("shape_zyx must be three positive integers")
    levels = result.get("levels")
    if not (isinstance(levels, int) and 1 <= levels <= 32):
        raise ValueError("levels must be between 1 and 32")
    slabs = result.get("slabs")
    if not isinstance(slabs, list) or not 0 < len(slabs) <= 10_000:
        raise ValueError("slabs must be a list of up to 10000 ranges")
    position = 0
    for slab in slabs:
        if not (
            isinstance(slab, list)
            and len(slab) == 2
            and slab[0] == position
            and isinstance(slab[1], int)
            and slab[1] > slab[0]
        ):
            raise ValueError("slabs must cover the volume in order")
        position = slab[1]
    if position != shape[0]:
        raise ValueError("slabs must cover the whole depth")
    if result.get("kind") not in ("images", "dicom"):
        raise ValueError("kind must be images or dicom")


async def after_probe(db: AsyncSession, probe: Job) -> None:
    result = probe.result or {}
    image_grant = probe.grants[:1]
    depth = result["shape_zyx"][0]
    common = {"pipeline": probe, "created_by": probe.created_by}
    previous = [
        await jobs.enqueue(
            db,
            "ingest.slab",
            {"z_range": z_range},
            depends_on=[probe],
            grants=probe.grants,
            weight=WEIGHTS["slabs"] * (z_range[1] - z_range[0]) / depth,
            **common,
        )
        for z_range in result["slabs"]
    ]
    levels = int(result["levels"])
    # Each level has about an eighth of the work of the one before.
    shares = [8.0**-level for level in range(1, levels)]
    for level, share in zip(range(1, levels), shares, strict=True):
        previous = [
            await jobs.enqueue(
                db,
                "pyramid.level",
                {"level": level},
                depends_on=previous,
                grants=image_grant,
                weight=WEIGHTS["pyramid"] * share / sum(shares),
                **common,
            )
        ]
    finalize = await jobs.enqueue(
        db,
        "artifact.finalize",
        {
            "source": {
                "upload_id": probe.payload["upload_id"],
                "filename": probe.payload["filename"],
                "kind": result["kind"],
                "slices": result["slices"],
            }
        },
        depends_on=previous,
        grants=image_grant,
        weight=WEIGHTS["finalize"],
        **common,
    )
    artifact = await db.get(Artifact, uuid.UUID(probe.payload["artifact_id"]))
    assert artifact is not None
    artifact.produced_by_job = finalize.id
    await db.flush()
