"""
Meshes, end to end: a worker meshes the final segmentation block by block
into one closed surface per class, in the scan's units, as STL, OBJ, and GLB;
and the mesh jobs stay within a small memory budget on porous classes, or
fail for good saying what to change.
"""

import json
import struct
import subprocess
import sys
import threading
import time
import types
import uuid

import numpy as np
import pytest
from helpers import SECRET_KEY, add_worker, run_db, signup
from ml4paleo_server import artifacts
from ml4paleo_server.db import Artifact, Project, User
from ml4paleo_server.pipelines import mesh as mesh_pipeline
from ml4paleo_server.settings import Settings
from ml4paleo_server.storage import project_storage
from ml4paleo_worker.client import ServerClient
from ml4paleo_worker.context import PermanentError
from ml4paleo_worker.handlers import HANDLERS
from ml4paleo_worker.handlers import mesh as mesh_jobs
from ml4paleo_worker.main import Worker
from sqlalchemy import update

from ml4paleo.meshing.blocks import mesh_block, mesh_blocks
from ml4paleo.protocol import WorkerCaps
from ml4paleo.segmentation.predict import create_prediction
from ml4paleo.storage import StorageGrant, get_bytes

pytest.importorskip("zmesh")

SHAPE = (20, 24, 28)
VOXEL_SIZE_ZYX = (2.0, 0.5, 0.25)
BONE, TOOTH, CLAW = 2, 3, 4


@pytest.fixture
def settings(migrated_database_url, tmp_path, s3_endpoint, s3_bucket):
    """S3 storage, so the worker reaches it as it would in production."""
    return Settings(
        database_url=migrated_database_url,
        secret_key=SECRET_KEY,
        storage={
            "url": f"s3://{s3_bucket}/{tmp_path.name}",
            "endpoint": s3_endpoint,
            "public_endpoint": s3_endpoint,
            "access_key_id": "test",
            "secret_access_key": "test",
            "region": "us-east-1",
        },
    )


def segmented() -> np.ndarray:
    classes = np.ones(SHAPE, dtype=np.uint8)
    classes[4:18, 6:20, 10:22] = BONE  # across block seams on every axis
    classes[0:3, 20:24, 0:4] = TOOTH  # against three faces of the volume
    return classes


async def add_image(db, project: str, voxel_size_zyx, unit, shape_zyx=SHAPE):
    image = await artifacts.create_staging(
        db, project_id=uuid.UUID(project), kind="image", head_slot="image"
    )
    image.state = "committed"
    image.manifest = {
        "shape_czyx": [1, *shape_zyx],
        "window": [0, 1],
        "voxel_size_zyx": list(voxel_size_zyx),
        "unit": unit,
    }
    await artifacts.set_head(db, image)
    return image


def add_segmentation(settings, database_url, project: str):
    """
    A final segmentation, predicted from an image that a later upload (with
    another voxel size) has since replaced.
    """

    async def create(db):
        image = await add_image(db, project, VOXEL_SIZE_ZYX, "millimeter")
        prediction = await artifacts.create_staging(
            db,
            project_id=uuid.UUID(project),
            kind="prediction",
            inputs={"image_artifact_id": str(image.id)},
        )
        prediction.state = "committed"
        prediction.manifest = {"kind": "prediction", "shape_zyx": list(SHAPE)}
        artifact = await artifacts.create_staging(
            db,
            project_id=uuid.UUID(project),
            kind="segmentation",
            head_slot="segmentation",
            inputs={"prediction_artifact_id": str(prediction.id)},
        )
        grant = project_storage(settings).child(artifacts.artifact_path(artifact))
        group = create_prediction(grant, SHAPE, arrays=("class",), kind="segmentation")
        group["class"][:] = segmented()  # type: ignore[index]
        artifact.state = "committed"
        artifact.manifest = {"kind": "segmentation", "shape_zyx": list(SHAPE)}
        await artifacts.set_head(db, artifact)
        await add_image(db, project, (1.0, 1.0, 1.0), "micrometer")

    run_db(database_url, create)


def read_stl(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    (count,) = struct.unpack_from("<I", raw, 80)
    assert len(raw) == 84 + 50 * count
    records = np.frombuffer(
        raw[84:],
        dtype=np.dtype(
            [("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("_", "<u2")]
        ),
    )
    corners = records["corners"].reshape(-1, 3)
    vertices, faces = np.unique(corners, axis=0, return_inverse=True)
    return vertices, faces.reshape(-1, 3)


def read_glb(raw: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    (size,) = struct.unpack_from("<I", raw, 12)
    gltf = json.loads(raw[20 : 20 + size])
    count = gltf["accessors"][0]["count"]
    binary = raw[28 + size :]
    vertices = np.frombuffer(binary[: 12 * count], dtype="<f4").reshape(-1, 3)
    faces = np.frombuffer(binary[12 * count :], dtype="<u4").reshape(-1, 3)
    return vertices, faces.astype(np.int64), gltf


def signed_volume(vertices, faces):
    triangles = vertices[faces].astype(np.float64)
    return (
        np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum()
        / 6
    )


def closed(faces) -> bool:
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return bool((counts == 2).all())


def test_a_worker_meshes_each_class(
    new_browser, settings, migrated_database_url, live_server, monkeypatch
):
    # Small blocks, so the test crosses seams the way a big scan does.
    monkeypatch.setattr(mesh_pipeline, "BLOCK", 16)
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    base = f"/api/projects/{project}/meshes"
    for name in ("bone", "tooth", "claw"):
        ada.post(
            f"/api/projects/{project}/labels/classes",
            json={"name": name, "color": "#ffffff"},
        )
    assert ada.post(base, json={}).status_code == 409  # nothing to mesh yet
    assert ada.get(base).status_code == 404
    add_segmentation(settings, migrated_database_url, project)
    assert ada.post(base, json={"downsample": 3}).status_code == 422

    started = ada.post(base, json={"simplify": 0})
    assert started.status_code == 202, started.text
    again = ada.post(base, json={})
    assert again.status_code == 409 and "already" in again.json()["detail"]
    token = add_worker(migrated_database_url)
    client = ServerClient(token, base_url=live_server)
    worker = Worker(
        client,
        WorkerCaps(version="test", kinds=sorted(HANDLERS)),
        claim_wait_seconds=0.5,
        heartbeat_seconds=0.2,
    )
    thread = threading.Thread(target=worker.run, kwargs={"max_jobs": None})
    thread.start()
    try:
        deadline = time.monotonic() + 120
        pipeline_url = (
            f"/api/projects/{project}/pipelines/{started.json()['pipeline_id']}"
        )
        while time.monotonic() < deadline:
            pipeline = ada.get(pipeline_url).json()
            if pipeline["status"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.3)
    finally:
        worker.stop()
        thread.join(timeout=30)
        client.close()
    assert pipeline["status"] == "succeeded", pipeline
    assert pipeline["kind"] == "meshes"
    # Blocks, the step that waits for them all, a join per class, finalize.
    assert pipeline["jobs"] == 8 + 1 + 3 + 1

    meshes = ada.get(base).json()
    assert meshes["artifact_id"] == started.json()["artifact_id"]
    segmentation = ada.get(f"/api/projects/{project}/segmentation").json()
    assert meshes["segmentation_artifact_id"] == segmentation["artifact_id"]
    info = meshes["info"]
    assert info["axis_order"] == "xyz"
    # The image the segmentation came from, not the newer one.
    assert info["units"] == "millimeter"
    assert info["voxel_size_xyz"] == list(reversed(VOXEL_SIZE_ZYX))

    async def check_image_provenance(db):
        exported = await db.get(Artifact, uuid.UUID(meshes["artifact_id"]))
        source = await db.get(Artifact, uuid.UUID(segmentation["artifact_id"]))
        image = await mesh_pipeline.source_image(db, source)
        current = await artifacts.head(db, uuid.UUID(project), "image")
        assert exported.inputs["image_artifact_id"] == str(image.id)
        assert image.id != current.id

    run_db(migrated_database_url, check_image_provenance)
    # Claw has no voxels, so no mesh.
    assert [(c["value"], c["name"]) for c in info["classes"]] == [
        (BONE, "bone"),
        (TOOTH, "tooth"),
    ]
    files = info["classes"][0]["files"]
    assert files == {"stl": "2.stl", "obj": "2.obj", "glb": "2.glb"}

    # The joined blocks match meshing the whole volume at once.
    volume = segmented()
    for entry in info["classes"]:
        value = entry["value"]
        raw = ada.get(meshes["files_url"] + f"{value}.stl")
        assert raw.status_code == 200
        vertices, faces = read_stl(raw.content)
        assert closed(faces)
        assert len(faces) == entry["triangles"]
        ((_, whole),) = mesh_block(volume, (0, 0, 0, *SHAPE), SHAPE, [value])
        assert signed_volume(vertices, faces) == pytest.approx(
            signed_volume(whole.vertices * list(reversed(VOXEL_SIZE_ZYX)), whole.faces),
            rel=1e-5,
        )
    # In millimeters, (x, y, z), from the voxels' outer corners.
    vertices, _ = read_stl(ada.get(meshes["files_url"] + "2.stl").content)
    assert vertices.min(axis=0) == pytest.approx([10 * 0.25, 6 * 0.5, 4 * 2.0])
    assert vertices.max(axis=0) == pytest.approx([22 * 0.25, 20 * 0.5, 18 * 2.0])

    # The GLB holds the same millimeters and scales its scene to meters.
    glb = ada.get(meshes["files_url"] + "2.glb").content
    magic, version, length = struct.unpack_from("<4sII", glb)
    assert (magic, version, length) == (b"glTF", 2, len(glb))
    glb_vertices, glb_faces, gltf = read_glb(glb)
    assert gltf["nodes"][0]["scale"] == [0.001, 0.001, 0.001]
    assert glb_vertices.min(axis=0) == pytest.approx(vertices.min(axis=0))
    assert closed(glb_faces)
    obj = ada.get(meshes["files_url"] + "3.obj").text
    assert obj.count("\nf ") > 0 and obj.count("\nv ") > 0

    grant = project_storage(settings).child(
        f"projects/{project}/artifacts/{meshes['artifact_id']}"
    )
    assert get_bytes(grant, "scratch/0.json") is None
    assert get_bytes(grant, f"scratch/0/{BONE}.npz") is None
    assert get_bytes(grant, "mesh_info.json") is not None
    assert json.loads(get_bytes(grant, f"{CLAW}.json") or b"")["triangles"] == 0
    assert ada.get(meshes["files_url"] + "4.stl").status_code == 404


def test_meshes_borrow_the_current_images_scale_only_on_its_grid(
    migrated_database_url,
):
    async def check(db):
        user = User(username="ada")
        db.add(user)
        await db.flush()
        project = Project(name="Skull", owner_id=user.id)
        db.add(project)
        await db.flush()
        # A segmentation that doesn't say what it was predicted from.
        segmentation = await artifacts.create_staging(
            db, project_id=project.id, kind="segmentation"
        )
        segmentation.manifest = {"kind": "segmentation", "shape_zyx": list(SHAPE)}
        await add_image(db, str(project.id), (1.0, 1.0, 1.0), "meter", (5, 6, 7))
        assert await mesh_pipeline.source_image(db, segmentation) is None
        image = await add_image(db, str(project.id), VOXEL_SIZE_ZYX, "millimeter")
        found = await mesh_pipeline.source_image(db, segmentation)
        assert found is not None and found.id == image.id

    run_db(migrated_database_url, check)


def test_meshes_wait_for_storage(new_browser, settings, migrated_database_url):
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    ada.post(
        f"/api/projects/{project}/labels/classes",
        json={"name": "bone", "color": "#ffffff"},
    )
    add_segmentation(settings, migrated_database_url, project)

    async def full(db):
        await db.execute(update(User).values(quota_override={"storage_gb": 0}))

    run_db(migrated_database_url, full)
    refused = ada.post(f"/api/projects/{project}/meshes", json={})
    assert refused.status_code == 403
    assert refused.json()["detail"] == "storage_quota_exceeded"


# One mesh job on local storage, in a process of its own, and how far its
# peak memory rose while it ran (with libraries loaded beforehand).
MEASURE = """
import json, resource, sys, types
import zmesh
from ml4paleo.storage import StorageGrant, get_bytes
from ml4paleo_worker.handlers import mesh

kind, root, budget, payload = sys.argv[1], sys.argv[2], int(sys.argv[3]), json.loads(sys.argv[4])
grants = [StorageGrant(url=f"file://{root}/segmentation"), StorageGrant(url=f"file://{root}/meshes", access="rw")]
get_bytes(grants[1], "nothing")
ctx = types.SimpleNamespace(payload=payload, grants=grants, memory_budget_bytes=budget, check=lambda: None, progress=lambda *a: None)
scale = 1 if sys.platform == "darwin" else 1024
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
result = getattr(mesh, kind)(ctx)
grew = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale - before
print(json.dumps({"result": result, "grew": grew}))
"""


def porous(tmp_path, side: int) -> dict:
    """
    A class of random voxels, as porous as a class can be, and the payload
    of meshing it as one block.
    """
    volume = np.where(np.random.default_rng(0).random((side,) * 3) < 0.5, BONE, 1)
    grant = StorageGrant(url=f"file://{tmp_path}/segmentation", access="rw")
    group = create_prediction(
        grant, volume.shape, arrays=("class",), kind="segmentation"
    )
    group["class"][:] = volume.astype(np.uint8)  # type: ignore[index]
    return {
        "shape_zyx": [side] * 3,
        "values": [BONE],
        "downsample": 1,
        "method": "any",
        "simplify": 0.0,
        "blocks": 1,
        "block_size": side,
        "block": 0,
        "box": [0, 0, 0, side, side, side],
        "voxel_size_xyz": [1.0, 1.0, 1.0],
        "unit": "voxels",
        "value": BONE,
        "name": "bone",
    }


def run_job(kind: str, tmp_path, budget: int, payload: dict) -> dict:
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            MEASURE,
            kind,
            str(tmp_path),
            str(budget),
            json.dumps(payload),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_porous_classes_mesh_within_a_small_memory_budget(tmp_path):
    payload = porous(tmp_path, 96)
    budget = 192 * 2**20
    # Meshed in one go, at the 600 or so bytes a voxel face zmesh was
    # measured to take, this block would need several times the budget.
    volume = np.pad(np.random.default_rng(0).random((96,) * 3) < 0.5, 1)
    faces = sum(int(np.count_nonzero(np.diff(volume, axis=a))) for a in range(3))
    assert faces * 600 > 3 * budget

    block = run_job("block", tmp_path, budget, payload)
    assert block["result"] == {"classes": [BONE]}
    assert block["grew"] < budget
    join = run_job("join_class", tmp_path, budget, payload)
    assert join["grew"] < budget
    stl = (tmp_path / "meshes" / f"{BONE}.stl").read_bytes()
    assert struct.unpack_from("<I", stl, 80)[0] == join["result"]["triangles"]


def job(tmp_path, payload: dict, budget: int = 3 * 2**30):
    return types.SimpleNamespace(
        payload=payload,
        grants=[
            StorageGrant(url=f"file://{tmp_path}/segmentation"),
            StorageGrant(url=f"file://{tmp_path}/meshes", access="rw"),
        ],
        memory_budget_bytes=budget,
        check=lambda: None,
        progress=lambda *args: None,
    )


def test_a_block_too_porous_for_its_budget_fails_for_good(tmp_path):
    payload = porous(tmp_path, 32)
    with pytest.raises(PermanentError, match="coarser resolution"):
        mesh_jobs.block(job(tmp_path, payload, budget=2**20))


def test_a_join_refuses_files_too_large_to_store(tmp_path, monkeypatch):
    payload = porous(tmp_path, 32)
    mesh_jobs.block(job(tmp_path, payload))
    monkeypatch.setattr(mesh_jobs, "MAX_FILE_BYTES", 10_000)
    with pytest.raises(PermanentError, match="coarser resolution or simplify more"):
        mesh_jobs.join_class(job(tmp_path, payload))


def test_a_join_fails_for_good_without_a_block(tmp_path):
    payload = porous(tmp_path, 32)
    boxes = mesh_blocks(payload["shape_zyx"], 16)
    payload = {**payload, "block_size": 16, "blocks": len(boxes)}
    for index, box in enumerate(boxes[:-1]):
        mesh_jobs.block(job(tmp_path, {**payload, "block": index, "box": list(box)}))
    with pytest.raises(PermanentError, match=f"Block {len(boxes) - 1}'s summary"):
        mesh_jobs.join_class(job(tmp_path, payload))
