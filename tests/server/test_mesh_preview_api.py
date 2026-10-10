import gzip
import struct

from helpers import signup
from ml4paleo_server.mesh_preview import MAX_INPUT


def test_preview_requires_membership_and_never_creates_saved_meshes(new_browser):
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/projects", json={"name": "Preview"}).json()["id"]
    endpoint = f"/api/projects/{project}/meshes/preview"
    data = struct.pack("<10I", 4, 4, 4, 1, 0, 0, 0, 4, 4, 4) + bytes([2]) * 64
    headers = {"Content-Type": "application/octet-stream"}
    counts = ada.get(f"/api/projects/{project}/labels/counts").json()
    result = ada.post(endpoint, content=data, headers=headers)
    assert result.status_code == 200, result.text
    assert len(result.content) > 0 and len(result.content) % 40 == 0
    assert result.headers["cache-control"] == "no-store"
    # Chunk-local builds accept typed query options and remain ephemeral.
    chunk = ada.post(
        endpoint + "?chunk=0,0,0&downsample=2", content=data, headers=headers
    )
    assert chunk.status_code == 200, chunk.text
    assert len(chunk.content) > 0 and len(chunk.content) % 40 == 0
    assert (
        ada.post(endpoint + "?chunk=0,0,1", content=data, headers=headers).status_code
        == 422
    )
    assert (
        ada.post(endpoint + "?downsample=3", content=data, headers=headers).status_code
        == 422
    )
    compressed = ada.post(
        endpoint,
        content=gzip.compress(data),
        headers={
            **headers,
            "Content-Encoding": "gzip",
            "Accept-Encoding": "gzip",
        },
    )
    assert compressed.status_code == 200, compressed.text
    assert compressed.content == result.content
    # A tiny mesh may be below the 1 KiB response-compression threshold.
    if len(result.content) >= 1024:
        assert compressed.headers["content-encoding"] == "gzip"
    assert (
        ada.post(
            endpoint,
            content=gzip.compress(b"x" * (MAX_INPUT + 1)),
            headers={**headers, "Content-Encoding": "gzip"},
        ).status_code
        == 413
    )
    assert (
        ada.post(
            endpoint,
            content=b"invalid gzip",
            headers={**headers, "Content-Encoding": "gzip"},
        ).status_code
        == 400
    )
    assert (
        ada.post(
            endpoint, content=data, headers={**headers, "Content-Encoding": "br"}
        ).status_code
        == 415
    )
    assert ada.get(f"/api/projects/{project}/meshes").status_code == 404
    assert ada.get(f"/api/projects/{project}/pipelines").json() == []
    assert ada.get(f"/api/projects/{project}/labels/counts").json() == counts
    assert ada.post(endpoint, content=data[:-1], headers=headers).status_code == 422
    assert ada.post(endpoint, json={}).status_code == 415
    assert (
        ada.post(endpoint, content=b"x" * (MAX_INPUT + 1), headers=headers).status_code
        == 413
    )
    other = new_browser()
    signup(other, "grace")
    assert other.post(endpoint, content=data, headers=headers).status_code == 404
    assert (
        new_browser().post(endpoint, content=data, headers=headers).status_code == 401
    )
