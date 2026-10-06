"""
Importing a v1 job into a new project (see `ml4paleo.v1import` for what v1
kept). The v1.* jobs run on a worker with the v1 volume (label "v1-volume"),
which runs nothing else; the pyramid and finalize jobs run on any worker.

An import is up to three pipelines. The first brings over the image:

    v1.probe -> v1.slab x N -> pyramid.level 1 .. L-1 -> artifact.finalize

Once the image is in (edits are checked against it), each of these starts as
a pipeline of its own, so one failing doesn't stop the other:

    v1.labels       (if the job has annotation samples to place)
    v1.prediction   (if it has a finished segmentation)

`start` creates the image artifact and the probe. When the probe succeeds,
`after_probe` checks that the image fits in the owner's storage, then adds a
"Foreground" class (v1 had one) and the rest of the image's pipeline. When
the image commits, `after_image` starts the other two. When `v1.labels`
succeeds, `after_labels` adds a complete slice ROI for each sample, so
training treats the samples as fully labeled slices.

A part is done when a job of it has succeeded. Claiming the job again starts
whichever parts aren't done or running (`resume`), and each can run again:
the image is made anew, the labels' edits are named by sample (so the ones
that landed aren't applied twice) and their ROIs added once, and a new
prediction replaces the one before.
"""

import math
import uuid
from typing import Any

import numpy as np
from sqlalchemy import exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from ml4paleo.labels import FIRST_CLASS, MAX_CLASS
from ml4paleo.v1import import SEGMENTATION_NAME

from .. import artifacts, jobs, quotas
from ..db import Artifact, Job, LabelClass, Project, Roi, User
from ..settings import Settings
from .ingest import WEIGHTS, check_volume

V1 = ["v1-volume"]
FOREGROUND_NAME = "Foreground"
FOREGROUND_COLOR = "#f2c14e"
MAX_ROIS = 100_000
UNFINISHED = ("blocked", "queued", "leased")
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


class NoRoom(Exception):
    """An import, or part of one, won't fit in its owner's storage."""


def estimate(result: dict[str, Any]) -> tuple[int, int]:
    """
    The most the image (with its pyramid) and the prediction that a probe
    found take, in bytes. They're stored compressed, so this errs on the
    large side.
    """
    voxels = math.prod(result["shape_zyx"])
    image = math.ceil(voxels * np.dtype(result["dtype"]).itemsize * PYRAMID)
    # A prediction holds a one-byte class per voxel.
    return image, voxels if result.get("segmentation") else 0


def _size(nbytes: float) -> str:
    if nbytes >= 1024**3:
        return f"{nbytes / 1024**3:.1f} GB"
    if 0 < nbytes < 1024**2 / 2:
        return "under 1 MB"
    return f"{round(nbytes / 1024**2)} MB"


async def _coming(
    db: AsyncSession, owner_id: uuid.UUID, exclude: uuid.UUID | None
) -> int:
    """
    What the owner's imports still running will add to their storage: an
    image and prediction for each image still coming over (but `exclude`'s),
    and each prediction.
    """
    other = aliased(Job)
    images = (
        select(Job.result)
        .join(Project, Project.id == Job.project_id)
        .where(
            Project.owner_id == owner_id,
            Job.kind == "v1.probe",
            Job.status == "succeeded",
            exists().where(other.root_id == Job.id, other.status.in_(UNFINISHED)),
        )
    )
    if exclude is not None:
        images = images.where(Job.id != exclude)
    predictions = (
        select(Job.payload)
        .join(Project, Project.id == Job.project_id)
        .where(
            Project.owner_id == owner_id,
            Job.kind == "v1.prediction",
            Job.status.in_(UNFINISHED),
        )
    )
    total = 0
    for result in await db.scalars(images):
        if result is not None:
            total += sum(estimate(result))
    for payload in await db.scalars(predictions):
        total += math.prod(payload["shape_zyx"])
    return total


async def _check_room(
    db: AsyncSession,
    settings: Settings,
    project_id: uuid.UUID,
    need: int,
    what: str,
    exclude: uuid.UUID | None = None,
) -> None:
    """
    Raise `NoRoom` if `need` bytes won't fit in the storage the project's
    owner has left once their other imports still running are in.
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
    used = (await quotas.usage_for(db, owner.id)).storage_bytes
    left = max(0, limit - used - await _coming(db, owner.id, exclude))
    if need > left:
        raise NoRoom(
            f"{what} takes about {_size(need)}, and you have {_size(left)} of "
            "storage left for it. Free some up or ask for more on the account "
            "page, then import the job again."
        )


async def _foreground(db: AsyncSession, project_id: uuid.UUID) -> int:
    """
    The value of the class v1's foreground becomes: "Foreground", added
    unless the probe or an earlier try at the import added it.
    """
    added = (
        select(LabelClass.value)
        .where(
            LabelClass.project_id == project_id,
            LabelClass.name == FOREGROUND_NAME,
            LabelClass.deleted_at.is_(None),
        )
        .order_by(LabelClass.value)
        .limit(1)
    )
    # Found, it needs no lock. The image's finalize gets here holding its
    # owner's usage (it just counted the image), which deleting the project
    # can wait for while holding the project.
    if (value := await db.scalar(added)) is not None:
        return value
    # As adding a class does: lock the project (FOR NO KEY UPDATE) so values
    # stay unique, and look again.
    await db.scalar(
        select(Project.id)
        .where(Project.id == project_id)
        .with_for_update(key_share=True)
    )
    if (value := await db.scalar(added)) is not None:
        return value
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
    try:
        # The prediction too: it comes once the image is in.
        await _check_room(
            db,
            settings,
            probe.project_id,
            sum(estimate(result)),
            "This job",
            exclude=probe.id,
        )
    except NoRoom as exc:
        raise jobs.Rejected(str(exc), retryable=False) from None
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
    await _foreground(db, probe.project_id)
    await db.flush()


async def after_image(db: AsyncSession, settings: Settings, finalize: Job) -> None:
    """The image is in: start what else the probe found to bring over."""
    # Share-locked: stopping the project (`train.stop_project`) cancels this
    # pipeline under a lock, then looks for pipelines started meanwhile. So
    # either it has, and this sees it, or it waits and finds what this starts.
    probe = await db.scalar(
        select(Job)
        .where(Job.id == finalize.root_id)
        .with_for_update(read=True)
        .execution_options(populate_existing=True)
    )
    assert probe is not None and probe.created_by is not None
    deleted = await db.scalar(
        select(Project.deleted_at).where(Project.id == probe.project_id)
    )
    if probe.cancel_requested or deleted is not None:
        return
    result = probe.result or {}
    await follow_ups(
        db,
        probe,
        labels=bool(result["annotations"]),
        prediction=bool(result.get("segmentation")),
        created_by=probe.created_by,
    )


async def follow_ups(
    db: AsyncSession,
    probe: Job,
    *,
    labels: bool,
    prediction: bool,
    created_by: uuid.UUID,
    again: bool = False,
) -> list[Job]:
    """
    Start the labels and the prediction (as `probe` found them), each as a
    pipeline of its own, and return their jobs. Labels brought over `again`
    only fill voxels nobody has labeled, so they keep edits made since.
    """
    assert probe.project_id is not None
    result = probe.result or {}
    job_id = probe.payload["job_id"]
    shape = result["shape_zyx"]
    foreground = await _foreground(db, probe.project_id)
    common = {
        "project_id": probe.project_id,
        "created_by": created_by,
        "required_labels": V1,
    }
    started = []
    if labels:
        started.append(
            await jobs.enqueue(
                db,
                "v1.labels",
                {
                    "job_id": job_id,
                    "shape_zyx": shape,
                    "foreground": foreground,
                    "only_unlabeled": again,
                },
                **common,
            )
        )
    if prediction:
        found = result["segmentation"]
        artifact = await artifacts.create_staging(
            db,
            project_id=probe.project_id,
            kind="prediction",
            head_slot="prediction",
            inputs={
                "model_id": None,
                "image_artifact_id": probe.payload["artifact_id"],
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
            grants=[artifacts.grant_for(artifact)],
            **common,
        )
        artifact.produced_by_job = job.id
        started.append(job)
    await db.flush()
    return started


async def resume(
    db: AsyncSession, settings: Settings, project: Project, created_by: uuid.UUID
) -> list[Job]:
    """
    Start again whichever parts of the project's import aren't done or
    running, and return the jobs that start them. A prediction is left out
    if the project has one since (from a model of its own) or it won't fit
    in the owner's storage (the image's probe checks the image itself).
    """
    assert project.v1_job_id is not None
    # Running first: the image's finalize commits the image and ends its
    # pipeline at once, so looking the other way round could miss both.
    root = aliased(Job)
    image_running = await db.scalar(
        select(Job.id)
        .join(root, root.id == Job.root_id)
        .where(
            Job.project_id == project.id,
            Job.status.in_(UNFINISHED),
            root.kind == "v1.probe",
        )
        .limit(1)
    )
    if image_running is not None:
        return []
    image = await artifacts.head(db, project.id, "image")
    if image is None:
        probe, _ = await start(db, project, project.v1_job_id, created_by)
        return [probe]
    probe = await db.scalar(
        select(Job)
        .where(
            Job.project_id == project.id,
            Job.kind == "v1.probe",
            Job.status == "succeeded",
        )
        .order_by(Job.created_at.desc(), Job.id.desc())
        .limit(1)
    )
    if probe is None or probe.payload.get("artifact_id") != str(image.id):
        # Someone made another image the project's; the rest of the import
        # wouldn't fit it.
        return []
    result = probe.result or {}

    async def missing(kind: str) -> bool:
        done_or_running = await db.scalar(
            select(Job.id)
            .where(
                Job.project_id == project.id,
                Job.kind == kind,
                Job.status.in_(("succeeded", *UNFINISHED)),
            )
            .limit(1)
        )
        return done_or_running is None

    labels = bool(result.get("annotations")) and await missing("v1.labels")
    prediction = (
        bool(result.get("segmentation"))
        and await missing("v1.prediction")
        and await artifacts.head(db, project.id, "prediction") is None
    )
    if prediction:
        try:
            await _check_room(
                db, settings, project.id, estimate(result)[1], "The segmentation"
            )
        except NoRoom:
            # Its failed pipeline already says why; claiming the job again
            # once there's room brings it over.
            prediction = False
    if not (labels or prediction):
        return []
    return await follow_ups(
        db,
        probe,
        labels=labels,
        prediction=prediction,
        created_by=created_by,
        again=True,
    )


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
    """
    Each placed sample was a fully labeled slice: a complete slice ROI, added
    once however often the labels are brought over.
    """
    assert job.project_id is not None
    seen = {
        tuple(bbox)
        for bbox in await db.scalars(
            select(Roi.bbox).where(Roi.project_id == job.project_id, Roi.origin == "v1")
        )
    }
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
