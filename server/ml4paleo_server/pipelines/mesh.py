"""
Meshes of the final segmentation, one per class, into a "meshes" artifact
that becomes the project's "meshes" head.

    mesh.block x N -> mesh.join x K -> mesh.finalize

Each `mesh.block` meshes one 256³ block (with one voxel of overlap), in
sub-boxes when its surface is too large to mesh at once, and writes each
class's pieces under `scratch/`; each `mesh.join` welds one class's pieces
as it streams them into `<value>.stl`, `.obj`, and `.glb`, and sums them up
in `<value>.json`; `mesh.finalize` writes `mesh_info.json` (axis order,
units, voxel size, files per class), cleans up `scratch/`, and writes the
manifest.
"""

import uuid
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from ml4paleo.meshing.blocks import mesh_blocks

from .. import artifacts, jobs
from ..db import Artifact, Job

BLOCK = 256
WEIGHTS = {"blocks": 80.0, "joins": 18.0, "finalize": 2.0}


async def source_image(db: AsyncSession, segmentation: Artifact) -> Artifact | None:
    """
    The image the segmentation's prediction was made from, or the project's
    current image if that one is gone.
    """
    image = None
    prediction_id = (segmentation.inputs or {}).get("prediction_artifact_id")
    prediction = (
        await db.get(Artifact, uuid.UUID(prediction_id)) if prediction_id else None
    )
    if prediction is not None:
        image_id = (prediction.inputs or {}).get("image_artifact_id")
        image = await db.get(Artifact, uuid.UUID(image_id)) if image_id else None
    if image is None or not image.manifest:
        image = await artifacts.head(db, segmentation.project_id, "image")
    return image


async def start(
    db: AsyncSession,
    *,
    segmentation: Artifact,
    classes: list[dict],
    downsample: int,
    method: Literal["any", "majority"],
    simplify: float,
    created_by: uuid.UUID,
) -> tuple[Job, Artifact]:
    assert segmentation.manifest is not None
    shape = [int(n) for n in segmentation.manifest["shape_zyx"]]
    image = await source_image(db, segmentation)
    manifest = (image.manifest if image is not None else None) or {}
    voxel_size = manifest.get("voxel_size_zyx")
    meshes = await artifacts.create_staging(
        db,
        project_id=segmentation.project_id,
        kind="meshes",
        head_slot="meshes",
        inputs={
            "segmentation_artifact_id": str(segmentation.id),
            "downsample": downsample,
            "method": method,
            "simplify": simplify,
        },
    )
    grants = [artifacts.grant_for(segmentation, "r"), artifacts.grant_for(meshes)]
    common = {
        "project_id": segmentation.project_id,
        "created_by": created_by,
        "grants": grants,
    }
    boxes = mesh_blocks(shape, BLOCK)
    payload = {
        "shape_zyx": shape,
        "values": [c["value"] for c in classes],
        "downsample": downsample,
        "method": method,
        "simplify": simplify,
        "blocks": len(boxes),
        "block_size": BLOCK,
        # Physical size of a voxel, (x, y, z); voxels if the scan didn't say.
        "voxel_size_xyz": list(reversed(voxel_size)) if voxel_size else [1.0, 1.0, 1.0],
        "unit": manifest.get("unit") if voxel_size else "voxels",
    }
    first = None
    blocks = []
    for index, box in enumerate(boxes):
        job = await jobs.enqueue(
            db,
            "mesh.block",
            {**payload, "block": index, "box": list(box)},
            weight=WEIGHTS["blocks"] / len(boxes),
            pipeline=first,
            **common,
        )
        first = first or job
        blocks.append(job)
    assert first is not None
    joins = [
        await jobs.enqueue(
            db,
            "mesh.join",
            {**payload, "value": c["value"], "name": c["name"]},
            pipeline=first,
            depends_on=blocks,
            weight=WEIGHTS["joins"] / max(1, len(classes)),
            **common,
        )
        for c in classes
    ]
    finalize = await jobs.enqueue(
        db,
        "mesh.finalize",
        {**payload, "classes": classes},
        pipeline=first,
        depends_on=joins or blocks,
        weight=WEIGHTS["finalize"],
        **common,
    )
    meshes.produced_by_job = finalize.id
    await db.flush()
    return first, meshes
