"""
Ingest jobs: from an uploaded archive to an OME-Zarr image artifact.

    ingest.probe -> ingest.slab (one per shard-deep slab, in parallel)
                 -> pyramid.level (1, 2, ...) -> artifact.finalize

Every job gets the image artifact as its first grant (read-write); the probe
and slab jobs also get the upload as their second (read-only). The server
adds the slab, pyramid, and finalize jobs when the probe succeeds, from its
result, and commits the artifact when finalize succeeds.
"""

from typing import Any

import numpy as np
from PIL import Image

from ml4paleo.ingest import (
    IngestError,
    SliceLimits,
    SourceIndex,
    intensity_summary,
    slab_provider,
)
from ml4paleo.ingest import probe as probe_archive
from ml4paleo.ome import OmeImage, downsample_level, write_from_provider
from ml4paleo.storage import (
    delete_object,
    get_bytes,
    open_object,
    put_bytes,
    write_manifest,
)

from ..context import JobContext, PermanentError

UPLOAD_KEY = "data"
INDEX_KEY = "ingest/source.json"
# Pillow's own limit is meant for web images. Ingest checks each slice's
# decoded size against the job's memory budget before decoding instead.
Image.MAX_IMAGE_PIXELS = 2**30


def probe(ctx: JobContext) -> dict[str, Any]:
    """
    Work out the volume in the upload, create the empty image, and record the
    slices in stacking order for the slab jobs.
    """
    image_grant, upload_grant = ctx.grants
    try:
        index = probe_archive(
            # Small reads: the probe looks at the first bytes of many members.
            open_object(upload_grant, UPLOAD_KEY, buffer_size=64 * 1024),
            SliceLimits.for_memory(ctx.memory_budget_bytes),
        )
    except IngestError as exc:
        raise PermanentError(str(exc)) from exc
    x, y, z = index.shape_xyz
    # Creating the image replaces whatever is there, so make sure this job
    # still holds its lease (the storage proxy also refuses writes after a
    # lease ends).
    ctx.check()
    image = OmeImage.create(
        image_grant,
        shape_czyx=(1, z, y, x),
        dtype=np.dtype(index.dtype),
        voxel_size_zyx=index.voxel_size_zyx,
        unit=index.unit,
        name=ctx.payload.get("filename") or "image",
        overwrite=True,
    )
    put_bytes(image_grant, INDEX_KEY, index.to_json())
    shards = image.array(0).shards
    assert shards is not None  # ml4paleo images are always sharded
    depth = shards[1]
    return {
        "kind": index.kind,
        "slices": len(index.members),
        "shape_zyx": [z, y, x],
        "dtype": index.dtype,
        "levels": image.num_levels,
        # Shard-deep slabs, so parallel jobs never write the same shard.
        "slabs": [[start, min(start + depth, z)] for start in range(0, z, depth)],
    }


def slab(ctx: JobContext) -> dict[str, Any]:
    image_grant, upload_grant = ctx.grants
    index_bytes = get_bytes(image_grant, INDEX_KEY)
    if index_bytes is None:
        raise PermanentError("The probe's slice index is missing.")
    index = SourceIndex.from_json(index_bytes)
    z0, z1 = ctx.payload["z_range"]

    def progress(done: int, total: int) -> None:
        ctx.progress(done / total)
        ctx.check()

    try:
        provider = slab_provider(
            open_object(upload_grant, UPLOAD_KEY),
            index,
            SliceLimits.for_memory(ctx.memory_budget_bytes),
        )
        write_from_provider(
            provider, OmeImage.open(image_grant), z_range=(z0, z1), progress=progress
        )
    except IngestError as exc:
        raise PermanentError(str(exc)) from exc
    except ValueError as exc:
        # The providers report mismatched or unreadable slices this way.
        raise PermanentError(str(exc)) from exc
    return {"z_range": [z0, z1]}


def pyramid(ctx: JobContext) -> dict[str, Any]:
    level = int(ctx.payload["level"])
    downsample_level(OmeImage.open(ctx.grants[0]), level - 1, method="mean")
    return {"level": level}


def finalize(ctx: JobContext) -> dict[str, Any]:
    """
    Summarize the image and write the manifest, so the server can commit the
    artifact, then remove ingest's scratch files.
    """
    grant = ctx.grants[0]
    image = OmeImage.open(grant)
    coarsest = np.asarray(image.array(image.num_levels - 1)[0])
    index_bytes = get_bytes(grant, INDEX_KEY)
    index = SourceIndex.from_json(index_bytes) if index_bytes else None
    manifest = {
        "kind": "image",
        "shape_czyx": list(image.shape_czyx),
        "dtype": np.dtype(image.dtype).str,
        "levels": image.num_levels,
        "voxel_size_zyx": list(image.voxel_size_zyx) if image.voxel_size_zyx else None,
        "unit": image.unit,
        "source": ctx.payload.get("source", {}),
        # Files in the upload that weren't slices, which ingest left out.
        "skipped": index.skipped if index else [],
        "skipped_count": index.skipped_count if index else 0,
        "notes": index.notes if index else [],
        **intensity_summary(coarsest),
    }
    write_manifest(grant, manifest)
    # Only now, so a retry after a failed manifest write still has the index.
    delete_object(grant, INDEX_KEY)
    return {"levels": image.num_levels}
