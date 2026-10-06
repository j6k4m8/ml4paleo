"""
Meshes, end to end: a worker meshes the final segmentation block by block
into one closed surface per class, in the scan's units, as STL, OBJ, and GLB;
and the mesh jobs fail for good, saying what to change, when they can't work.
"""

import json
import struct
import threading
import time
import types
import uuid

import numpy as np
import pytest
from helpers import SECRET_KEY, add_worker, run_db, signup
from ml4paleo_server import artifacts
from ml4paleo_server.pipelines import mesh as mesh_pipeline
from ml4paleo_server.settings import Settings
from ml4paleo_server.storage import project_storage
from ml4paleo_worker.client import ServerClient
from ml4paleo_worker.context import PermanentError
from ml4paleo_worker.handlers import HANDLERS
from ml4paleo_worker.handlers import mesh as mesh_jobs
from ml4paleo_worker.main import Worker

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


def add_segmentation(settings, database_url, project: str):
    async def create(db):
        image = await artifacts.create_staging(
            db, project_id=uuid.UUID(project), kind="image", head_slot="image"
        )
        image.state = "committed"
        image.manifest = {
            "shape_czyx": [1, *SHAPE],
            "window": [0, 1],
            "voxel_size_zyx": list(VOXEL_SIZE_ZYX),
            "unit": "millimeter",
        }
        await artifacts.set_head(db, image)
        artifact = await artifacts.create_staging(
            db,
            project_id=uuid.UUID(project),
            kind="segmentation",
            head_slot="segmentation",
        )
        grant = project_storage(settings).child(artifacts.artifact_path(artifact))
        group = create_prediction(grant, SHAPE, arrays=("class",), kind="segmentation")
        group["class"][:] = segmented()  # type: ignore[index]
        artifact.state = "committed"
        artifact.manifest = {"kind": "segmentation", "shape_zyx": list(SHAPE)}
        await artifacts.set_head(db, artifact)

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
    assert pipeline["jobs"] == 8 + 3 + 1  # blocks, a join per class, finalize

    meshes = ada.get(base).json()
    assert meshes["artifact_id"] == started.json()["artifact_id"]
    segmentation = ada.get(f"/api/projects/{project}/segmentation").json()
    assert meshes["segmentation_artifact_id"] == segmentation["artifact_id"]
    info = meshes["info"]
    assert info["axis_order"] == "xyz"
    assert info["units"] == "millimeter"
    assert info["voxel_size_xyz"] == list(reversed(VOXEL_SIZE_ZYX))
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

    glb = ada.get(meshes["files_url"] + "3.glb").content
    magic, version, length = struct.unpack_from("<4sII", glb)
    assert (magic, version, length) == (b"glTF", 2, len(glb))
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
