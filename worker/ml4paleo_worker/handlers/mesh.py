"""
Mesh jobs (see the server's `pipelines/mesh.py`).

Grants, in order: the segmentation artifact (read) and the meshes artifact
(write).

Neither kind of job holds more than a piece of a class at a time. A block
meshes each class in pieces sized to the job's memory budget, writes them to
`scratch/<block>/<value>.npz` as they come, and writes its summary (pieces,
triangles, and vertices per class) to `scratch/<block>.json` last. A join
streams one class's pieces, block by block, into local files, uploads them,
and writes `<value>.json` (triangles, vertices, and files) for finalize.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import obstore

from ml4paleo.meshing.blocks import (
    Join,
    PieceWriter,
    TooDetailed,
    face_limit,
    mesh_block,
    read_box,
    read_pieces,
)
from ml4paleo.segmentation.predict import open_prediction
from ml4paleo.storage import (
    PROXY_MAX_OBJECT_BYTES,
    delete_object,
    get_bytes,
    object_store,
    open_object,
    put_bytes,
    put_file,
    write_manifest,
)

from ..context import JobContext, PermanentError

EXTENSIONS = ("stl", "obj", "glb")
# Each file must fit through the storage proxy, in one request (and a GLB
# can't be larger anyway). This also caps how complex a mesh can get.
MAX_FILE_BYTES = PROXY_MAX_OBJECT_BYTES
# Roughly what one seam vertex waiting to be welded takes in memory.
SEAM_BYTES = 200


def block(ctx: JobContext) -> dict[str, Any]:
    segmentation_grant, meshes_grant = ctx.grants
    index = int(ctx.payload["block"])
    box = tuple(int(n) for n in ctx.payload["box"])
    shape = ctx.payload["shape_zyx"]
    d = int(ctx.payload["downsample"])
    simplify = float(ctx.payload["simplify"])
    values = ctx.payload["values"]
    rb = read_box(box, shape, d)  # type: ignore[arg-type]
    region = np.asarray(
        open_prediction(segmentation_grant)["class"][
            rb[0] : rb[3], rb[1] : rb[4], rb[2] : rb[5]
        ]  # type: ignore[index]
    )
    ctx.check()
    limit = face_limit(ctx.memory_budget_bytes, simplify > 0)
    classes: dict[str, dict[str, int]] = {}
    for done, value in enumerate(values):
        counts = {"pieces": 0, "triangles": 0, "vertices": 0}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pieces.npz"
            with PieceWriter(path) as writer:
                pieces = mesh_block(
                    region,
                    box,  # type: ignore[arg-type]
                    shape,
                    [value],
                    downsample=d,
                    method=ctx.payload["method"],
                    max_error=simplify,
                    max_faces=limit,
                )
                try:
                    for _, piece in pieces:
                        writer.add(piece)
                        counts["pieces"] += 1
                        counts["triangles"] += len(piece.faces)
                        counts["vertices"] += len(piece.vertices)
                        ctx.check()
                except TooDetailed as exc:
                    raise PermanentError(
                        f"Block {index} has more surface than this worker can mesh "
                        f"({exc}); mesh at a coarser resolution."
                    ) from None
            if counts["pieces"]:
                put_file(meshes_grant, f"scratch/{index}/{value}.npz", path)
                classes[str(value)] = counts
        ctx.progress((done + 1) / len(values))
    put_bytes(
        meshes_grant, f"scratch/{index}.json", json.dumps({"classes": classes}).encode()
    )
    return {"classes": sorted(int(value) for value in classes)}


def _check_size(name: str, extension: str, size: int) -> None:
    if size > MAX_FILE_BYTES:
        raise PermanentError(
            f"The {name} mesh would be a {size / 2**30:.1f} GiB {extension.upper()} "
            f"file, more than the {MAX_FILE_BYTES / 2**30:.0f} GiB a file can be; "
            "mesh at a coarser resolution or simplify more."
        )


def join_class(ctx: JobContext) -> dict[str, Any]:
    meshes_grant = ctx.grants[1]
    value = int(ctx.payload["value"])
    name = ctx.payload.get("name") or f"class {value}"
    blocks = int(ctx.payload["blocks"])
    found = []
    for index in range(blocks):
        raw = get_bytes(meshes_grant, f"scratch/{index}.json")
        if raw is None:
            raise PermanentError(f"Block {index}'s summary is missing.")
        found.append(json.loads(raw)["classes"].get(str(value)))
    # A binary STL takes 50 bytes a triangle, more than OBJ or GLB as a rule,
    # so a mesh too large for one is refused before any work.
    triangles = sum(counts["triangles"] for counts in found if counts)
    _check_size(name, "stl", 84 + 50 * triangles)
    files: dict[str, str] = {}
    with (
        tempfile.TemporaryDirectory() as tmp,
        Join(
            Path(tmp),
            ctx.payload["shape_zyx"],
            int(ctx.payload["block_size"]),
            ctx.payload["voxel_size_xyz"],
        ) as join,
    ):
        for index, counts in enumerate(found):
            if counts:
                key = f"scratch/{index}/{value}.npz"
                try:
                    file = open_object(meshes_grant, key)
                except FileNotFoundError:
                    raise PermanentError(
                        f"Block {index}'s pieces of the {name} mesh are missing."
                    ) from None
                with file:
                    for piece in read_pieces(file):
                        join.add(piece)
                        if join.pending * SEAM_BYTES > ctx.memory_budget_bytes // 2:
                            raise PermanentError(
                                f"The {name} mesh has too many vertices on block "
                                "seams to weld in this worker's memory; mesh at a "
                                "coarser resolution or simplify more."
                            )
                        ctx.check()
            join.block_done(index)
            ctx.progress(0.7 * (index + 1) / blocks)
        if join.triangles:
            writers = {
                "stl": join.finish,
                "obj": lambda: join.write_obj(Path(tmp) / "mesh.obj"),
                "glb": lambda: join.write_glb(
                    Path(tmp) / "mesh.glb", ctx.payload["unit"]
                ),
            }
            for done, extension in enumerate(EXTENSIONS):
                try:
                    path = writers[extension]()
                except TooDetailed as exc:
                    raise PermanentError(
                        f"The {name} mesh is too large for a {extension.upper()} "
                        f"file ({exc}); mesh at a coarser resolution or simplify more."
                    ) from None
                _check_size(name, extension, path.stat().st_size)
                put_file(meshes_grant, f"{value}.{extension}", path)
                path.unlink()
                files[extension] = f"{value}.{extension}"
                ctx.progress(0.7 + 0.1 * (done + 1))
                ctx.check()
        summary = {
            "value": value,
            "triangles": join.triangles,
            "vertices": join.vertices,
            "files": files,
        }
    put_bytes(meshes_grant, f"{value}.json", json.dumps(summary).encode())
    return {key: summary[key] for key in ("value", "triangles", "vertices")}


def finalize(ctx: JobContext) -> dict[str, Any]:
    meshes_grant = ctx.grants[1]
    classes = []
    for entry in ctx.payload["classes"]:
        raw = get_bytes(meshes_grant, f"{int(entry['value'])}.json")
        if raw is None:
            raise PermanentError(f"The {entry['name']} mesh's summary is missing.")
        joined = json.loads(raw)
        if joined["triangles"]:
            classes.append(
                {
                    **entry,
                    "files": joined["files"],
                    "triangles": joined["triangles"],
                    "vertices": joined["vertices"],
                }
            )
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
    # Whatever is left in scratch/, so a retry finds only what remains.
    for batch in obstore.list(object_store(meshes_grant), prefix="scratch/"):
        for meta in batch:
            delete_object(meshes_grant, meta["path"])
    write_manifest(meshes_grant, {"kind": "meshes", **info})
    return {"classes": len(classes)}
