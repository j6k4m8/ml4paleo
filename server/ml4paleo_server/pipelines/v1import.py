"""
Importing a v1 job into a new project (see `ml4paleo.v1import` for what v1
kept). The v1.* jobs run on a worker with the v1 volume (label "v1-volume"),
which runs nothing else; the pyramid and finalize jobs run on any worker.

    v1.probe -> v1.slab x N -> pyramid.level 1 .. L-1 -> artifact.finalize
             -> v1.labels       (if the job has annotation samples to place)
             -> v1.prediction   (if it has a finished segmentation)

`start` creates the image artifact and the probe. When the probe succeeds,
`after_probe` checks that the image fits in the owner's storage, then adds a
"Foreground" class (v1 had one) and the rest; labels and the prediction wait
for the image to commit, since edits are checked against it. When
`v1.labels` succeeds, `after_labels` adds a complete slice ROI for each
sample, so training treats the samples as fully labeled slices.
"""

import math
import uuid
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ml4paleo.labels import FIRST_CLASS, MAX_CLASS
from ml4paleo.v1import import SEGMENTATION_NAME

from .. import artifacts, jobs, quotas
from ..db import Artifact, Job, LabelClass, Project, Roi, User
from ..settings import Settings
from .ingest import WEIGHTS, check_volume

V1 = ["v1-volume"]
FOREGROUND_NAME = "Foreground"
FOREGROUND_COLOR = "#f2c14e"
# Weights beyond the ingest's: the samples and the segmentation.
LABEL_WEIGHT = 10.0
PREDICTION_WEIGHT = 15.0
MAX_ROIS = 100_000
# An image with its pyramid takes about this much more than its full
# resolution alone (each level has an eighth of the voxels of the last).
PYRAMID = 1.15


async def start(
    db: AsyncSession, project: Project, job_id: str, created_by: uuid.UUID
) -> tuple[Job, Artifact]:
    artifact = await artifacts.create_staging(
        db,
        project_id=project.id,
        kind="image",
        head_slot="image",
        inputs={"v1_job_id": job_id},
    )
    probe = await jobs.enqueue(
        db,
        "v1.probe",
        {"job_id": job_id, "artifact_id": str(artifact.id)},
        project_id=project.id,
        created_by=created_by,
        grants=[artifacts.grant_for(artifact)],
        required_labels=V1,
        weight=WEIGHTS["probe"],
    )
    return probe, artifact


def check_probe_result(result: dict[str, Any]) -> None:
    check_volume(result)
    if result.get("kind") != "v1":
        raise ValueError("kind must be v1")
    for key in ("annotations", "skipped_annotations"):
        if not (isinstance(result.get(key), int) and result[key] >= 0):
            raise ValueError(f"{key} must be a count")
    dtype = result.get("dtype")
    try:
        numeric = isinstance(dtype, str) and np.dtype(dtype).kind in "buif"
    except TypeError:
        numeric = False
    if not numeric:
        raise ValueError("dtype must be a numeric dtype")
    found = result.get("segmentation")
    if found is not None and not (
        isinstance(found, str)
        and len(found) <= 64
        and SEGMENTATION_NAME.fullmatch(found)
    ):
        raise ValueError("segmentation must be a v1 segmentation's folder name")


def _size(nbytes: float) -> str:
    if nbytes < 1024**3:
        return f"{max(1, round(nbytes / 1024**2))} MB"
    return f"{nbytes / 1024**3:.1f} GB"


async def _check_room(
    db: AsyncSession, settings: Settings, project_id: uuid.UUID, result: dict
) -> None:
    """
    Refuse an import whose image won't fit in the owner's storage, before
    any of it is copied. It's stored compressed, so this errs on the large
    side.
    """
    owner = await db.scalar(
        select(User)
        .join(Project, Project.owner_id == User.id)
        .where(Project.id == project_id)
    )
    assert owner is not None
    limit = (await quotas.limits_for(db, owner, settings)).storage_bytes
    if limit is None:
        return
    itemsize = np.dtype(result["dtype"]).itemsize
    size = math.prod(result["shape_zyx"]) * itemsize * PYRAMID
    left = max(0, limit - (await quotas.usage_for(db, owner.id)).storage_bytes)
    if size > left:
        raise jobs.Rejected(
            f"This scan takes about {_size(size)}, and you have {_size(left)} of "
            "storage left. Free some up or ask for more on the account page, then "
            "import the job again.",
            retryable=False,
        )


async def _foreground(db: AsyncSession, project_id: uuid.UUID) -> int:
    """
    The value of the class v1's foreground becomes: "Foreground", added
    unless an earlier try at the import added it.
    """
    # As adding a class does: lock the project (FOR NO KEY UPDATE) so values
    # stay unique.
    await db.scalar(
        select(Project.id)
        .where(Project.id == project_id)
        .with_for_update(key_share=True)
    )
    added = await db.scalar(
        select(LabelClass.value)
        .where(
            LabelClass.project_id == project_id,
            LabelClass.name == FOREGROUND_NAME,
            LabelClass.deleted_at.is_(None),
        )
        .order_by(LabelClass.value)
        .limit(1)
    )
    if added is not None:
        return added
    highest = await db.scalar(
        select(func.max(LabelClass.value)).where(LabelClass.project_id == project_id)
    )
    value = max(FIRST_CLASS, (highest or 0) + 1)
    if value > MAX_CLASS:
        raise jobs.Rejected("The project has used every class value.", retryable=False)
    db.add(
        LabelClass(
            project_id=project_id,
            value=value,
            name=FOREGROUND_NAME,
            color=FOREGROUND_COLOR,
        )
    )
    await db.flush()
    return value


async def after_probe(db: AsyncSession, settings: Settings, probe: Job) -> None:
    result = probe.result or {}
    assert probe.project_id is not None
    await _check_room(db, settings, probe.project_id, result)
    job_id = probe.payload["job_id"]
    shape = result["shape_zyx"]
    image_grant = probe.grants[:1]
    common = {"pipeline": probe, "created_by": probe.created_by}
    previous = [
        await jobs.enqueue(
            db,
            "v1.slab",
            {"job_id": job_id, "z_range": z_range},
            depends_on=[probe],
            grants=image_grant,
            required_labels=V1,
            weight=WEIGHTS["slabs"] * (z_range[1] - z_range[0]) / shape[0],
            **common,
        )
        for z_range in result["slabs"]
    ]
    levels = int(result["levels"])
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
        {"source": {"kind": "v1", "v1_job_id": job_id}},
        depends_on=previous,
        grants=image_grant,
        weight=WEIGHTS["finalize"],
        **common,
    )
    image = await db.get(Artifact, uuid.UUID(probe.payload["artifact_id"]))
    assert image is not None
    image.produced_by_job = finalize.id
    foreground = await _foreground(db, probe.project_id)
    if result["annotations"]:
        await jobs.enqueue(
            db,
            "v1.labels",
            {"job_id": job_id, "shape_zyx": shape, "foreground": foreground},
            depends_on=[finalize],
            required_labels=V1,
            weight=LABEL_WEIGHT,
            **common,
        )
    if found := result.get("segmentation"):
        prediction = await artifacts.create_staging(
            db,
            project_id=probe.project_id,
            kind="prediction",
            head_slot="prediction",
            inputs={
                "model_id": None,
                "image_artifact_id": str(image.id),
                "v1_job_id": job_id,
                "v1_segmentation": found,
            },
        )
        job = await jobs.enqueue(
            db,
            "v1.prediction",
            {
                "job_id": job_id,
                "segmentation": found,
                "shape_zyx": shape,
                "foreground": foreground,
            },
            depends_on=[finalize],
            grants=[artifacts.grant_for(prediction)],
            required_labels=V1,
            weight=PREDICTION_WEIGHT,
            **common,
        )
        prediction.produced_by_job = job.id
    await db.flush()


def check_labels_result(result: dict[str, Any]) -> None:
    rois = result.get("rois")
    if not isinstance(rois, list) or len(rois) > MAX_ROIS:
        raise ValueError(f"rois must be a list of up to {MAX_ROIS} boxes")
    for box in rois:
        if not (
            isinstance(box, list)
            and len(box) == 6
            and all(isinstance(n, int) and n >= 0 for n in box)
            and box[3] == box[0] + 1
            and box[4] > box[1]
            and box[5] > box[2]
        ):
            raise ValueError(
                "each roi must be a one-slice box (z0, y0, x0, z1, y1, x1)"
            )


async def after_labels(db: AsyncSession, settings: Settings, job: Job) -> None:
    """Each placed sample was a fully labeled slice: a complete slice ROI."""
    assert job.project_id is not None
    seen = set()
    for box in (job.result or {})["rois"]:
        if tuple(box) in seen:
            continue
        seen.add(tuple(box))
        db.add(
            Roi(
                project_id=job.project_id,
                created_by=job.created_by,
                bbox=box,
                kind="slice",
                status="complete",
                split="train",
                origin="v1",
            )
        )
    await db.flush()
