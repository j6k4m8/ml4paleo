"""
The data gateway (viewers read committed artifacts through the API) and the
self-hosted Neuroglancer.
"""

import json
from urllib.parse import unquote

import numpy as np
import pytest
from helpers import run_db, signup
from ml4paleo_server import artifacts
from ml4paleo_server.db import Artifact
from ml4paleo_server.storage import project_storage
from sqlalchemy import update

from ml4paleo.ome import OmeImage


def make_project(browser, name="Skull") -> str:
    return browser.post("/api/projects", json={"name": name}).json()["id"]


def committed_image(settings, database_url, project_id, *, state="committed"):
    """
    A small OME-Zarr image artifact, as ingest would leave it.
    """

    async def create(db):
        artifact = await artifacts.create_staging(
            db, project_id=project_id, kind="image", head_slot="image"
        )
        image = OmeImage.create(
            project_storage(settings).child(artifacts.artifact_path(artifact)),
            shape_czyx=(1, 4, 6, 8),
            dtype=np.uint16,
            chunk_zyx=(2, 2, 2),
            shard_zyx=(2, 4, 4),
        )
        image.array(0)[0] = np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8)
        artifact.state = state
        artifact.manifest = {"window": [10, 100]}
        if state == "committed":
            await artifacts.set_head(db, artifact)
        return artifact.id

    return run_db(database_url, create)


def test_members_read_committed_artifacts(new_browser, settings, migrated_database_url):
    browser = new_browser()
    signup(browser)
    project = make_project(browser)
    artifact_id = committed_image(settings, migrated_database_url, project)
    base = f"/api/projects/{project}/artifacts/{artifact_id}/zarr"

    metadata = browser.get(f"{base}/zarr.json")
    assert metadata.status_code == 200
    assert "ome" in metadata.json()["attributes"]
    assert "immutable" in metadata.headers["cache-control"]
    size = int(browser.request("HEAD", f"{base}/zarr.json").headers["content-length"])
    assert size == len(metadata.content)
    # Ranges, including the suffix ranges sharded zarr uses for shard indexes.
    tail = browser.get(f"{base}/zarr.json", headers={"Range": "bytes=-5"})
    assert (tail.status_code, tail.content) == (206, metadata.content[-5:])
    middle = browser.get(f"{base}/zarr.json", headers={"Range": "bytes=2-6"})
    assert middle.content == metadata.content[2:7]
    # A chunk that was never written is a 404, which zarr reads as empty.
    assert browser.get(f"{base}/0/c/0/9/9/9").status_code == 404
    for bad in ["", "..%2Fsecret", "0/../../x"]:
        assert browser.get(f"{base}/{bad}").status_code in (400, 404)


def test_only_members_and_committed_artifacts(
    new_browser, settings, migrated_database_url
):
    ada = new_browser()
    signup(ada)
    project = make_project(ada)
    other = make_project(ada, "Other")
    artifact_id = committed_image(settings, migrated_database_url, project)
    staging_id = committed_image(
        settings, migrated_database_url, project, state="staging"
    )
    bob = new_browser()
    signup(bob, username="bob")
    file = "zarr.json"
    assert (
        bob.get(
            f"/api/projects/{project}/artifacts/{artifact_id}/zarr/{file}"
        ).status_code
        == 404
    )
    # An artifact of one project can't be read through another.
    assert (
        ada.get(
            f"/api/projects/{other}/artifacts/{artifact_id}/zarr/{file}"
        ).status_code
        == 404
    )
    assert (
        ada.get(
            f"/api/projects/{project}/artifacts/{staging_id}/zarr/{file}"
        ).status_code
        == 404
    )

    async def delete(db):
        await db.execute(
            update(Artifact).where(Artifact.id == artifact_id).values(state="deleted")
        )

    run_db(migrated_database_url, delete)
    assert (
        ada.get(
            f"/api/projects/{project}/artifacts/{artifact_id}/zarr/{file}"
        ).status_code
        == 404
    )


@pytest.fixture
def neuroglancer(tmp_path):
    directory = tmp_path / "neuroglancer"
    directory.mkdir()
    (directory / "index.html").write_text("<html>neuroglancer</html>")
    (directory / "main.bundle.js").write_text("console.log('ng')")
    return directory


def test_neuroglancer_is_served_with_its_own_policy(
    new_browser, settings, migrated_database_url, neuroglancer
):
    with_ng = settings.model_copy(update={"neuroglancer_dir": neuroglancer})
    browser = new_browser(with_ng)
    page = browser.get("/neuroglancer/")
    assert page.text == "<html>neuroglancer</html>"
    policy = page.headers["content-security-policy"]
    # Its decoding worker builds functions at run time and compiles WebAssembly.
    assert "script-src 'self' 'unsafe-eval' 'wasm-unsafe-eval'" in policy
    assert (
        "connect-src 'self'" in policy
        and "'unsafe-inline'" not in policy.split("script-src", 1)[1].split(";")[0]
    )
    assert browser.get("/neuroglancer/main.bundle.js").text == "console.log('ng')"
    # The rest of the site keeps the strict policy.
    assert "eval" not in browser.get("/api/health").headers["content-security-policy"]

    signup(browser)
    project = make_project(browser)
    artifact_id = committed_image(with_ng, migrated_database_url, project)
    image = browser.get(f"/api/projects/{project}/image").json()
    assert image["zarr_url"] == f"/api/projects/{project}/artifacts/{artifact_id}/zarr/"
    state = json.loads(unquote(image["neuroglancer_url"].split("#!", 1)[1]))
    [layer] = state["layers"]
    assert layer["source"] == f"zarr3://{with_ng.public_url}{image['zarr_url']}"
    assert layer["shaderControls"]["normalized"]["range"] == [10, 100]


def test_without_a_neuroglancer_build_there_is_no_link(
    new_browser, settings, migrated_database_url
):
    browser = new_browser()
    signup(browser)
    project = make_project(browser)
    committed_image(settings, migrated_database_url, project)
    assert (
        browser.get(f"/api/projects/{project}/image").json()["neuroglancer_url"] is None
    )
    # The path falls through to the web app instead, under the strict policy.
    page = browser.get("/neuroglancer/")
    assert "has not been built" in page.text
    assert "eval" not in page.headers["content-security-policy"]
