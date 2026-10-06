"""
Mesh jobs (see the server's `pipelines/mesh.py`).

Grants, in order: the segmentation artifact (read) and the meshes artifact
(write).
"""

import io
import json
from typing import Any

import numpy as np

from ml4paleo.meshing.blocks import join, mesh_block, read_box, to_glb, to_obj, to_stl
from ml4paleo.segmentation.predict import open_prediction
from ml4paleo.storage import delete_object, get_bytes, put_bytes, write_manifest

from ..context import JobContext

FORMATS = {"stl": to_stl, "obj": to_obj, "glb": to_glb}


def block(ctx: JobContext) -> dict[str, Any]:
    segmentation_grant, meshes_grant = ctx.grants
    box = tuple(int(n) for n in ctx.payload["box"])
    shape = ctx.payload["shape_zyx"]
    d = int(ctx.payload["downsample"])
    rb = read_box(box, shape, d)  # type: ignore[arg-type]
    region = np.asarray(
        open_prediction(segmentation_grant)["class"][
            rb[0] : rb[3], rb[1] : rb[4], rb[2] : rb[5]
        ]  # type: ignore[index]
    )
    ctx.check()
    pieces = mesh_block(
        region,
        box,  # type: ignore[arg-type]
        shape,
        ctx.payload["values"],
        downsample=d,
        method=ctx.payload["method"],
        max_error=float(ctx.payload["simplify"]),
    )
    buffer = io.BytesIO()
    arrays = {}
    for value, (vertices, faces) in pieces.items():
        arrays[f"v{value}"] = vertices
        arrays[f"f{value}"] = faces
    np.savez_compressed(buffer, **arrays)
    put_bytes(meshes_grant, f"scratch/{ctx.payload['block']}.npz", buffer.getvalue())
    return {"classes": sorted(pieces)}


def join_class(ctx: JobContext) -> dict[str, Any]:
    meshes_grant = ctx.grants[1]
    value = int(ctx.payload["value"])
    pieces = []
    for index in range(int(ctx.payload["blocks"])):
        raw = get_bytes(meshes_grant, f"scratch/{index}.npz")
        if raw is None:
            continue
        saved = np.load(io.BytesIO(raw))
        if f"v{value}" in saved:
            pieces.append((saved[f"v{value}"], saved[f"f{value}"]))
        ctx.check()
    vertices, faces = join(pieces, ctx.payload["voxel_size_xyz"])
    if len(faces) == 0:
        return {"value": value, "triangles": 0}
    for extension, write in FORMATS.items():
        put_bytes(meshes_grant, f"{value}.{extension}", write(vertices, faces))
    return {
        "value": value,
        "triangles": int(len(faces)),
        "vertices": int(len(vertices)),
    }


def finalize(ctx: JobContext) -> dict[str, Any]:
    meshes_grant = ctx.grants[1]
    classes = []
    for entry in ctx.payload["classes"]:
        value = int(entry["value"])
        if get_bytes(meshes_grant, f"{value}.stl") is None:
            continue
        classes.append({**entry, "files": {ext: f"{value}.{ext}" for ext in FORMATS}})
    info = {
        "axis_order": "xyz",
        "units": ctx.payload["unit"],
        "voxel_size_xyz": ctx.payload["voxel_size_xyz"],
        "downsample": ctx.payload["downsample"],
        "method": ctx.payload["method"],
        "simplify": ctx.payload["simplify"],
        "classes": classes,
    }
    put_bytes(meshes_grant, "mesh_info.json", json.dumps(info, indent=2).encode())
    for index in range(int(ctx.payload["blocks"])):
        delete_object(meshes_grant, f"scratch/{index}.npz")
    write_manifest(meshes_grant, {"kind": "meshes", **info})
    return {"classes": len(classes)}
