"""
Export jobs (see the server's `pipelines/export.py`).

Grants, in order: the source artifact (read) and the export artifact
(write). A job writes the archive's parts, then a manifest listing their
sizes, which the API uses to serve them as one file.
"""

import json
import re
from typing import Any

import numpy as np
import obstore

from ml4paleo.export import (
    PNG_DTYPES,
    Parts,
    add_entry,
    encode_slice,
    open_archive,
    part_key,
    slab_depth,
    slice_names,
    slices,
)
from ml4paleo.ome import OmeImage
from ml4paleo.segmentation.predict import open_prediction
from ml4paleo.storage import (
    MANIFEST_KEY,
    StorageGrant,
    get_bytes,
    object_store,
    put_bytes,
    write_manifest,
)

from ..context import JobContext, PermanentError

# Files of ours that aren't part of what's exported.
SKIP = (MANIFEST_KEY, "inputs.json")
STREAM_CHUNK = 8 * 1024**2
MESH_INFO = "mesh_info.json"


def files(ctx: JobContext) -> dict[str, Any]:
    """
    Every file of the source: a zarr group at the archive's root, or the
    meshes in a folder, named for their classes.
    """
    source, export = ctx.grants
    store = object_store(source)
    listed = sorted(
        (meta["path"], int(meta["size"]))
        for batch in obstore.list(store, chunk_size=1000)
        for meta in batch
        if meta["path"] not in SKIP and not meta["path"].startswith("scratch/")
    )
    meshes = ctx.payload["source"] == "meshes"
    names, info = _mesh_names(source) if meshes else ({}, None)
    folder = ctx.payload["folder"]
    total = sum(size for _, size in listed) or 1
    done = 0
    out = Parts(lambda index, data: put_bytes(export, part_key(index), data))
    with open_archive(out) as archive:
        for key, size in listed:
            name = folder + names.get(key, key)
            if key == MESH_INFO and info is not None:
                add_entry(archive, name, [info], len(info), compress=True)
            else:
                chunks = obstore.get(store, key).stream(min_chunk_size=STREAM_CHUNK)
                # Zarr chunks are compressed already; mesh files aren't.
                add_entry(archive, name, chunks, size, compress=meshes)
            done += size
            ctx.progress(0.99 * done / total)
            ctx.check()
    return _finish(ctx, out, len(listed))


def _mesh_names(source: StorageGrant) -> tuple[dict[str, str], bytes]:
    """
    Names for the mesh files by class (`2.stl` becomes `bone.stl`), and
    `mesh_info.json` naming them so.
    """
    raw = get_bytes(source, MESH_INFO)
    if raw is None:
        raise PermanentError(f"The meshes have no {MESH_INFO}")
    info = json.loads(raw)
    names: dict[str, str] = {}
    taken: set[str] = set()
    for entry in info.get("classes", []):
        stem = re.sub(r"[^a-z0-9]+", "-", str(entry["name"]).lower()).strip("-")
        if not stem or stem in taken:
            stem = f"{stem or 'class'}-{entry['value']}"
        taken.add(stem)
        files = {}
        for extension, key in entry["files"].items():
            names[key] = f"{stem}.{extension}"
            files[extension] = names[key]
        entry["files"] = files
    return names, json.dumps(info, indent=2).encode()


def images(ctx: JobContext) -> dict[str, Any]:
    """
    One TIFF or PNG per z (and channel) of the image, or of a prediction's
    or segmentation's classes.
    """
    source, export = ctx.grants
    fmt = ctx.payload["format"]
    if ctx.payload["source"] == "image":
        array: Any = OmeImage.open(source).array(0)
        shape = tuple(int(n) for n in array.shape)
    else:
        array = open_prediction(source)["class"]
        shape = (1, *(int(n) for n in array.shape))
    if fmt == "png" and np.dtype(array.dtype) not in PNG_DTYPES:
        raise PermanentError(
            f"PNG holds 8- or 16-bit unsigned values, and this image is "
            f"{np.dtype(array.dtype)}: export TIFF instead"
        )
    depth = slab_depth(
        shape,  # type: ignore[arg-type]
        np.dtype(array.dtype).itemsize,
        int(array.chunks[-3]),
        ctx.memory_budget_bytes,
    )
    name = slice_names(shape, fmt, ctx.payload["folder"].rstrip("/"))  # type: ignore[arg-type]
    count = shape[0] * shape[1]
    out = Parts(lambda index, data: put_bytes(export, part_key(index), data))
    with open_archive(out) as archive:
        for done, (c, z, plane) in enumerate(slices(array, depth), start=1):
            data = encode_slice(plane, fmt)
            add_entry(archive, name(c, z), [data], len(data))
            if done % 8 == 0 or done == count:
                ctx.progress(0.99 * done / count)
                ctx.check()
    return _finish(ctx, out, count)


def _finish(ctx: JobContext, out: Parts, entries: int) -> dict[str, Any]:
    sizes = out.finish()
    _, export = ctx.grants
    write_manifest(
        export,
        {
            "kind": "export",
            "format": ctx.payload["format"],
            "source": ctx.payload["source"],
            "entries": entries,
            "size": sum(sizes),
            "parts": sizes,
        },
    )
    return {"bytes": sum(sizes), "parts": len(sizes)}
