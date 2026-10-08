"""
The coarse levels of the label zarr: the image's pyramid for labels, made
from the full-resolution labels when a viewer asks, for display only.
"""

import asyncio
import base64
import hashlib
import pathlib
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from helpers import run_db, signup
from ml4paleo_server import artifacts, label_pyramid
from ml4paleo_server.app import create_app

from ml4paleo.labels import LABEL_CHUNK_ZYX
from ml4paleo.labels.codec import ZARR_CODECS, decode_chunk
from ml4paleo.labels.deltas import split_into_deltas
from ml4paleo.labels.pyramid import downsample_labels
from ml4paleo.ome import OmeImage, plan_levels
from ml4paleo.storage import StorageGrant

# (z, y, x): levels of 4, with several chunks at the first two.
SHAPE = (140, 150, 270)
ANISOTROPIC = {"voxel_size_zyx": [4.0, 1.0, 1.0], "unit": "millimeter"}
# The output of the rule on a fixed block (see the test that uses it).
RULE_PIN = "0cc2d53744168f83e28faaba9970c0fd20c028bde561ce382a8abeb1dc0fd3a0"


def add_image(database_url, project, shape=SHAPE, **manifest) -> None:
    """
    Make a new image (which has just this manifest) the project's current one.
    """

    async def add(db):
        artifact = await artifacts.create_staging(
            db, project_id=uuid.UUID(project), kind="image", head_slot="image"
        )
        artifact.state = "committed"
        artifact.manifest = {"shape_czyx": [1, *shape], **manifest}
        await artifacts.set_head(db, artifact)

    run_db(database_url, add)


def make_project(browser, database_url, shape=SHAPE, **manifest) -> str:
    project = browser.post("/api/projects", json={"name": "Skull"}).json()["id"]
    add_image(database_url, project, shape, **manifest)
    for name, color in [("bone", "#ffffff"), ("matrix", "#884400")]:
        browser.post(
            f"/api/projects/{project}/labels/classes",
            json={"name": name, "color": color},
        )
    return project


def send(browser, project, deltas, tool) -> dict:
    response = browser.post(
        f"/api/projects/{project}/labels/ops",
        json={
            "client_op_id": str(uuid.uuid4()),
            "deltas": [
                {
                    "key": list(delta.key),
                    "base_version": 0,
                    "box": list(delta.box),
                    "mask": base64.b64encode(delta.mask).decode(),
                    "value": delta.value,
                    "values": base64.b64encode(delta.values).decode()
                    if delta.values
                    else None,
                    "only_if": "any",
                }
                for delta in deltas
            ],
            "tool": {"name": tool},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def paint(browser, project, volume, origin=(0, 0, 0)) -> dict:
    """
    Label every voxel of `volume` that isn't zero, as one op.
    """
    deltas = split_into_deltas(volume != 0, origin, values=volume)
    return send(browser, project, deltas, "brush")


def erase(browser, project, box) -> None:
    """
    Unlabel a (z0, y0, x0, z1, y1, x1) box.
    """
    z0, y0, x0, z1, y1, x1 = box
    mask = np.ones((z1 - z0, y1 - y0, x1 - x0), dtype=bool)
    send(browser, project, split_into_deltas(mask, (z0, y0, x0), value=0), "eraser")


def get(browser, project, level, key, **kwargs):
    array = label_pyramid.array_name(level)
    path = "/".join(str(k) for k in key)
    return browser.get(
        f"/api/projects/{project}/labels/zarr/{array}/c/{path}", **kwargs
    )


def chunk(browser, project, level, key) -> np.ndarray | None:
    response = get(browser, project, level, key)
    if response.status_code == 404:
        return None
    assert response.status_code == 200, response.text
    return decode_chunk(response.content)


def shrink(volume, levels) -> list[np.ndarray]:
    """
    Every level of the pyramid, by shrinking the whole volume level by level.
    """
    out = [volume]
    for below, level in zip(levels, levels[1:], strict=False):
        step = tuple(
            a // b for a, b in zip(level.factor_zyx, below.factor_zyx, strict=True)
        )
        out.append(downsample_labels(out[-1], step))
        assert out[-1].shape == level.shape_zyx
    return out


def expected_chunk(array, key) -> np.ndarray:
    out = np.zeros(LABEL_CHUNK_ZYX, dtype=np.uint8)
    part = array[tuple(slice(c * 64, (c + 1) * 64) for c in key)]
    out[tuple(slice(0, n) for n in part.shape)] = part
    return out


def check_every_level(browser, project, volume, levels, shown=None) -> None:
    """
    Every chunk of every level is the whole volume shrunk to that level, and
    missing exactly where that is all unlabeled. Coarse levels are made of
    `shown` if the volume has voxels (of retired classes) they leave out.
    """
    arrays = shrink(volume if shown is None else shown, levels)
    arrays[0] = volume
    for level, array in enumerate(arrays):
        for key in np.ndindex(*(-(-n // 64) for n in array.shape)):
            expected = expected_chunk(array, key)
            got = chunk(browser, project, level, key)
            if expected.any():
                assert got is not None, (level, key)
                np.testing.assert_array_equal(got, expected, err_msg=f"{level} {key}")
            else:
                assert got is None, (level, key)


def messy_volume(shape=SHAPE) -> np.ndarray:
    """
    Labels of every kind: speckle, big regions, thin ones, and a block of
    mixed classes where blocks have to vote.
    """

    def part(low, high):
        return tuple(
            slice(int(n * a), int(n * b))
            for n, a, b in zip(shape, low, high, strict=True)
        )

    rng = np.random.default_rng(7)
    volume = np.zeros(shape, dtype=np.uint8)
    speckle = rng.random(shape) < 0.002
    volume[speckle] = rng.choice([1, 2, 3], size=int(speckle.sum()))
    volume[part((0.07, 0.13, 0.11), (0.36, 0.6, 0.44))] = 2
    volume[part((0.43, 0.03, 0.74), (0.5, 0.06, 0.96))] = 3
    volume[:, int(shape[1] * 0.51), int(shape[2] * 0.01)] = 3
    mixed = part((0.68, 0.67, 0.55), (0.98, 0.93, 0.74))
    volume[mixed] = rng.integers(0, 4, size=volume[mixed].shape)
    volume[part((0.36, 0.67, 0.48), (0.43, 0.73, 0.52))] = 1
    return volume


@pytest.fixture
def ada(new_browser):
    browser = new_browser()
    signup(browser)
    # However slow the machine, a request finishes what it starts, unless a
    # test says otherwise.
    pyramid = browser.client.app.state.label_pyramid
    pyramid.seconds, pyramid.blobs = 3600, 10**9
    return browser


@pytest.fixture
def project(ada, migrated_database_url):
    return make_project(ada, migrated_database_url)


def test_the_zarr_group_lists_the_levels_of_the_image(ada, project):
    levels = plan_levels(SHAPE)
    assert len(levels) == 4
    group = ada.get(f"/api/projects/{project}/labels/zarr/zarr.json").json()
    assert group["node_type"] == "group"
    listed = group["attributes"]["ml4paleo"]["label_levels"]
    assert listed == [
        {
            "array": "class" if i == 0 else f"class_{i}",
            "shape": list(level.shape_zyx),
            "factor_zyx": list(level.factor_zyx),
        }
        for i, level in enumerate(levels)
    ]
    for i, level in enumerate(levels):
        name = "class" if i == 0 else f"class_{i}"
        metadata = ada.get(f"/api/projects/{project}/labels/zarr/{name}/zarr.json")
        assert metadata.status_code == 200
        array = metadata.json()
        assert array["shape"] == list(level.shape_zyx)
        assert array["data_type"] == "uint8"
        assert array["chunk_grid"]["configuration"]["chunk_shape"] == [64, 64, 64]
        # The same codecs as the full resolution array, so one reader opens all.
        base = ada.get(f"/api/projects/{project}/labels/zarr/class/zarr.json").json()
        assert array["codecs"] == base["codecs"]
    for name in ["class_4", "class_31", "class_0", "class_01", "source_1", "classes"]:
        response = ada.get(f"/api/projects/{project}/labels/zarr/{name}/zarr.json")
        assert response.status_code == 404, name
        response = ada.get(f"/api/projects/{project}/labels/zarr/{name}/c/0/0/0")
        assert response.status_code == 404, name


def test_a_project_without_an_image_has_no_label_zarr(ada):
    project = ada.post("/api/projects", json={"name": "Empty"}).json()["id"]
    for key in ["zarr.json", "class/zarr.json", "class_1/zarr.json", "class_1/c/0/0/0"]:
        response = ada.get(f"/api/projects/{project}/labels/zarr/{key}")
        assert response.status_code == 404, key
        assert "no image" in response.json()["detail"]


def test_every_level_is_the_labels_shrunk(ada, project):
    volume = messy_volume()
    paint(ada, project, volume)
    check_every_level(ada, project, volume, plan_levels(SHAPE))


def test_levels_are_those_of_a_real_image(tmp_path):
    for shape, voxel in [
        ((70, 130, 100), None),
        ((140, 150, 270), None),
        ((70, 200, 260), (4.0, 1.0, 1.0)),
        ((300, 40, 40), (0.5, 1.5, 1.5)),
        ((1, 300, 40), None),
        ((64, 64, 64), None),
        ((65, 64, 64), None),
    ]:
        grant = StorageGrant(url=f"file://{tmp_path}/{shape}{voxel}", access="rw")
        image = OmeImage.create(
            grant, shape_czyx=(1, *shape), dtype="uint8", voxel_size_zyx=voxel
        )
        manifest = {
            "shape_czyx": list(image.shape_czyx),
            "levels": image.num_levels,
            "voxel_size_zyx": list(image.voxel_size_zyx) if voxel else None,
        }
        levels = label_pyramid.levels_of(manifest)
        assert len(levels) == image.num_levels
        base = image.scale_zyx(0)
        for k, level in enumerate(levels):
            assert level.shape_zyx == tuple(image.array(k).shape[1:])
            # How the viewer reads an image's levels (web/src/lib/viewer/image.ts).
            scale = image.scale_zyx(k)
            assert level.factor_zyx == tuple(
                round(s / b) for s, b in zip(scale, base, strict=True)
            )


def test_an_image_made_some_other_way_has_level_zero_only(ada, migrated_database_url):
    project = make_project(ada, migrated_database_url, levels=9)
    group = ada.get(f"/api/projects/{project}/labels/zarr/zarr.json").json()
    assert [a["array"] for a in group["attributes"]["ml4paleo"]["label_levels"]] == [
        "class"
    ]
    assert get(ada, project, 1, (0, 0, 0)).status_code == 404
    # One that says as many levels as the plan has, or none, gets them all.
    for declared in [{"levels": 4}, {}]:
        project = make_project(ada, migrated_database_url, **declared)
        group = ada.get(f"/api/projects/{project}/labels/zarr/zarr.json").json()
        assert len(group["attributes"]["ml4paleo"]["label_levels"]) == 4


@pytest.mark.parametrize("odd", [[1, 0, 1], [1, -2, 1], 5, "big"])
def test_an_images_odd_voxel_sizes_leave_level_zero_alone(
    ada, migrated_database_url, odd
):
    project = make_project(ada, migrated_database_url, voxel_size_zyx=odd)
    metadata = ada.get(f"/api/projects/{project}/labels/zarr/class/zarr.json")
    assert metadata.json()["shape"] == list(SHAPE)
    group = ada.get(f"/api/projects/{project}/labels/zarr/zarr.json").json()
    assert len(group["attributes"]["ml4paleo"]["label_levels"]) == 1


def test_levels_halve_only_the_axes_the_image_halves(ada, migrated_database_url):
    shape = (70, 200, 260)
    project = make_project(ada, migrated_database_url, shape=shape, **ANISOTROPIC)
    levels = plan_levels(shape, (4.0, 1.0, 1.0))
    assert [level.factor_zyx for level in levels] == [
        (1, 1, 1),
        (1, 2, 2),
        (1, 4, 4),
        (2, 8, 8),
    ]
    volume = messy_volume(shape)
    paint(ada, project, volume)
    check_every_level(ada, project, volume, levels)


def test_thin_labels_show_at_every_level(ada, project):
    levels = plan_levels(SHAPE)
    volume = np.zeros(SHAPE, dtype=np.uint8)
    # A class in a sea of background, and a line a voxel wide.
    volume[40:100, 20:130, 30:200] = 1
    volume[101, 117, 233] = 3
    volume[139, 3, :] = 2
    paint(ada, project, volume)
    for level, spec in enumerate(levels):
        factor = spec.factor_zyx
        at = tuple(p // f for p, f in zip((101, 117, 233), factor, strict=True))
        key = tuple(a // 64 for a in at)
        got = chunk(ada, project, level, key)
        assert got is not None
        assert got[tuple(a % 64 for a in at)] == 3
        assert (got == 3).sum() == 1
        line = tuple(p // f for p, f in zip((139, 3), factor[:2], strict=True))
        key = (line[0] // 64, line[1] // 64, 0)
        got = chunk(ada, project, level, key)
        assert got is not None
        assert (got[line[0] % 64, line[1] % 64, :] == 2).sum() == min(
            64, spec.shape_zyx[2]
        )


def test_a_coarse_chunk_is_missing_where_nothing_is_labeled(ada, project):
    levels = plan_levels(SHAPE)
    volume = np.zeros(SHAPE, dtype=np.uint8)
    volume[5:9, 5:9, 5:9] = 2
    paint(ada, project, volume)
    for level in range(1, len(levels)):
        assert chunk(ada, project, level, (0, 0, 0)) is not None
    response = get(ada, project, 1, (1, 1, 2))
    assert (response.status_code, response.headers["x-chunk-version"]) == (404, "0")
    assert "etag" not in response.headers
    # Past the edge of the level.
    assert get(ada, project, 1, (5, 0, 0)).status_code == 404
    assert get(ada, project, 3, (1, 0, 0)).status_code == 404
    # Erased again: nothing is labeled, though the chunks have been edited.
    erase(ada, project, (0, 0, 0, 20, 20, 20))
    for level in range(1, len(levels)):
        response = get(ada, project, level, (0, 0, 0))
        assert response.status_code == 404
        assert int(response.headers["x-chunk-version"]) > 0


def test_a_coarse_chunk_changes_when_anything_under_it_does(ada, project):
    levels = plan_levels(SHAPE)
    volume = messy_volume()
    paint(ada, project, volume)
    keys = {
        level: list(np.ndindex(*(-(-n // 64) for n in spec.shape_zyx)))
        for level, spec in enumerate(levels)
        if level
    }

    def snapshot():
        seen = {}
        for level, grid in keys.items():
            for key in grid:
                response = get(ada, project, level, key)
                seen[level, key] = (
                    response.status_code,
                    response.headers.get("etag"),
                    int(response.headers["x-chunk-version"]),
                    response.content,
                )
        return seen

    before = snapshot()
    # The version of a coarse chunk is the sum of those under it.
    for (level, key), (_, _, version, _) in before.items():
        if level != 1:
            continue
        (z0, z1), (y0, y1), (x0, x1) = label_pyramid.footprint(levels[1], key)
        under = [
            int(get(ada, project, 0, (z, y, x)).headers["x-chunk-version"])
            for z in range(z0, z1)
            for y in range(y0, y1)
            for x in range(x0, x1)
        ]
        assert version == sum(under)

    voxel = (131, 70, 200)
    volume[voxel] = 2
    result = paint(ada, project, volume[131:132, 70:71, 200:201], voxel)
    after = snapshot()
    pyramid = shrink(volume, levels)
    for (level, key), (status, _, _, content) in after.items():
        expected = expected_chunk(pyramid[level], key)
        assert (status == 200) == bool(expected.any())
        if status == 200:
            np.testing.assert_array_equal(decode_chunk(content), expected)

    def over(level):
        return tuple(
            p // (64 * f) for p, f in zip(voxel, levels[level].factor_zyx, strict=True)
        )

    for (level, key), (status, etag, version, content) in after.items():
        old_status, old_etag, old_version, old_content = before[level, key]
        if key == over(level):
            assert etag != old_etag and version > old_version
            assert status == 200
        else:
            assert (status, etag, version, content) == (
                old_status,
                old_etag,
                old_version,
                old_content,
            )
        if etag:
            revalidated = get(ada, project, level, key, headers={"If-None-Match": etag})
            assert revalidated.status_code == 304 and revalidated.content == b""
            assert revalidated.headers["etag"] == etag
            assert revalidated.headers["x-chunk-version"] == str(version)
    # An old ETag gets the new chunk, not a 304.
    stale = get(
        ada, project, 1, over(1), headers={"If-None-Match": before[1, over(1)][1]}
    )
    assert stale.status_code == 200 and stale.headers["etag"] == after[1, over(1)][1]

    # Undoing the edit brings every chunk's pixels back, under new ETags.
    undone = ada.post(
        f"/api/projects/{project}/labels/ops/{result['seq']}/undo",
        json={"client_op_id": str(uuid.uuid4())},
    )
    assert undone.status_code == 201
    final = snapshot()
    for found, (status, etag, version, content) in final.items():
        level, key = found
        old_status, old_etag, old_version, old_content = before[found]
        assert (status, content) == (old_status, old_content)
        if key == over(level):
            assert len({old_etag, after[found][1], etag}) == 3
        else:
            assert (etag, version) == (old_etag, old_version)


@pytest.fixture
def reads(monkeypatch):
    """
    The label blobs read to make coarse chunks, in order.
    """
    seen = []
    real = label_pyramid._read

    async def counting(store, sha):
        seen.append(sha)
        return await real(store, sha)

    monkeypatch.setattr(label_pyramid, "_read", counting)
    return seen


def labeled_chunks(volume) -> int:
    grid = [-(-n // 64) for n in volume.shape]
    return sum(
        1
        for key in np.ndindex(*grid)
        if volume[tuple(slice(c * 64, (c + 1) * 64) for c in key)].any()
    )


def test_a_coarse_chunk_is_made_once_and_an_edit_costs_one_chunk_a_level(
    ada, project, reads
):
    levels = plan_levels(SHAPE)
    top = len(levels) - 1
    volume = messy_volume()
    paint(ada, project, volume)
    first = get(ada, project, top, (0, 0, 0))
    assert first.status_code == 200
    # Every labeled chunk, read once, to make every level under the top one.
    assert len(reads) == labeled_chunks(volume)
    reads.clear()
    again = get(ada, project, top, (0, 0, 0))
    assert again.content == first.content
    for level in range(1, top):
        assert get(ada, project, level, (0, 0, 0)).status_code == 200
    assert reads == []

    voxel = (131, 70, 200)
    volume[voxel] = 3
    paint(ada, project, volume[131:132, 70:71, 200:201], voxel)
    edited = get(ada, project, top, (0, 0, 0))
    assert edited.headers["etag"] != first.headers["etag"]
    under = tuple(
        p // (64 * f) for p, f in zip(voxel, levels[1].factor_zyx, strict=True)
    )
    (z0, z1), (y0, y1), (x0, x1) = label_pyramid.footprint(levels[1], under)
    around = volume[z0 * 64 : z1 * 64, y0 * 64 : y1 * 64, x0 * 64 : x1 * 64]
    # One chunk of level 1 again, from the chunks under it; the rest were kept.
    assert 0 < len(reads) == labeled_chunks(around) <= 8
    expected = shrink(volume, levels)[top]
    np.testing.assert_array_equal(
        decode_chunk(edited.content), expected_chunk(expected, (0, 0, 0))
    )


# Out of time from the start, or allowed only a couple of chunks' reads, a
# request still does the chunk it is on.
@pytest.mark.parametrize("limit, most", [({"seconds": 0}, 8), ({"blobs": 20}, 20)])
def test_a_chunk_too_big_for_one_request_comes_in_several(
    ada, project, reads, limit, most
):
    levels = plan_levels(SHAPE)
    top = len(levels) - 1
    volume = messy_volume()
    paint(ada, project, volume)
    pyramid = ada.client.app.state.label_pyramid
    for name, value in limit.items():
        setattr(pyramid, name, value)
    asked = []
    for _ in range(100):
        before = len(reads)
        response = get(ada, project, top, (0, 0, 0))
        asked.append((response.status_code, len(reads) - before))
        if response.status_code != 503:
            break
        assert response.headers["retry-after"] == "1"
    assert asked[-1][0] == 200
    assert [code for code, _ in asked[:-1]] == [503] * (len(asked) - 1)
    assert len(asked) > 2
    # No request reads more than its share, and none redoes the work of an
    # earlier one: every labeled chunk is read once.
    assert max(n for _, n in asked) <= most
    assert sum(n for _, n in asked) == labeled_chunks(volume) == len(reads)
    expected = shrink(volume, levels)[top]
    np.testing.assert_array_equal(
        decode_chunk(response.content), expected_chunk(expected, (0, 0, 0))
    )
    # Having asked for the top, everything under it is cached.
    reads.clear()
    pyramid.seconds = 0
    for level in range(1, top):
        for key in np.ndindex(*(-(-n // 64) for n in levels[level].shape_zyx)):
            assert get(ada, project, level, key).status_code in (200, 404)
    assert reads == []


class Spy:
    """
    What label chunks were read to make coarse ones, and how many were held at
    once: read, but not yet combined. Reads take a little while, so requests
    that start together overlap.
    """

    def __init__(self, monkeypatch, app, delay=0.2):
        self.engine = app.state.engine.sync_engine
        self.reads = self.combines = self.held = self.most_held = 0
        self.connections = []
        self._lock = threading.Lock()
        real_read, real_combine = label_pyramid._read, label_pyramid._combine

        async def read(store, sha):
            await asyncio.sleep(delay)
            data = await real_read(store, sha)
            with self._lock:
                self.reads += 1
                self.held += 1
                self.most_held = max(self.most_held, self.held)
                # Connections the pool has lent out, which nobody should hold
                # while waiting on storage.
                self.connections.append(self.engine.pool.checkedout())
            return data

        def combine(parts, *args):
            try:
                return real_combine(parts, *args)
            finally:
                with self._lock:
                    self.combines += 1
                    self.held -= len(parts)

        monkeypatch.setattr(label_pyramid, "_read", read)
        monkeypatch.setattr(label_pyramid, "_combine", combine)


def concurrently(count, function):
    with ThreadPoolExecutor(count) as pool:
        return list(pool.map(function, range(count)))


def test_requests_for_one_cold_chunk_share_one_build(ada, project, monkeypatch):
    volume = np.zeros(SHAPE, dtype=np.uint8)
    # Eight labeled chunks under the chunk of level 1 that everyone asks for.
    for z, y, x in np.ndindex(2, 2, 2):
        volume[64 * z + 1, 64 * y + 1, 64 * x + 1] = 2
    paint(ada, project, volume)
    spy = Spy(monkeypatch, ada.client.app)
    answers = concurrently(16, lambda _: get(ada, project, 1, (0, 0, 0)))
    assert [a.status_code for a in answers] == [200] * 16
    assert len({a.content for a in answers}) == 1
    assert len({a.headers["etag"] for a in answers}) == 1
    # One request built it, from its eight chunks, and the rest took the result.
    assert (spy.reads, spy.combines, spy.most_held) == (8, 1, 8)
    # Nobody held a database connection while waiting on storage or a build.
    assert spy.connections == [0] * 8
    expected = shrink(volume, plan_levels(SHAPE))[1]
    np.testing.assert_array_equal(
        decode_chunk(answers[0].content), expected_chunk(expected, (0, 0, 0))
    )


def test_a_waiting_request_gives_up_when_the_build_does(ada, project, monkeypatch):
    paint(ada, project, messy_volume())
    pyramid = ada.client.app.state.label_pyramid
    pyramid.seconds = 0
    Spy(monkeypatch, ada.client.app, delay=0.3)
    top = len(plan_levels(SHAPE)) - 1
    answers = concurrently(6, lambda _: get(ada, project, top, (0, 0, 0)))
    # The one that built it ran out of time, and so did those who waited.
    assert [a.status_code for a in answers] == [503] * 6
    assert {a.headers["retry-after"] for a in answers} == {"1"}


def test_builds_hold_only_what_their_slots_allow_and_a_long_queue_is_turned_away(
    ada, project, monkeypatch
):
    levels = plan_levels(SHAPE)
    volume = np.zeros(SHAPE, dtype=np.uint8)
    for key in np.ndindex(3, 3, 5):
        volume[tuple(64 * k + 1 for k in key)] = 2
    paint(ada, project, volume)
    spy = Spy(monkeypatch, ada.client.app)
    keys = list(np.ndindex(*(-(-n // 64) for n in levels[1].shape_zyx)))
    assert len(keys) == 12
    answers = concurrently(12, lambda i: get(ada, project, 1, keys[i]))
    busy = [a for a in answers if a.status_code == 503]
    # Two build and eight wait; the last two come back another time.
    assert len(busy) == 2
    assert {a.headers["retry-after"] for a in busy} == {"1"}
    assert [a.status_code for a in answers if a not in busy] == [200] * 10
    # At most two chunks of level 1 were being read and made at once.
    assert spy.most_held <= 2 * 8
    for i, answer in enumerate(answers):
        again = answer if answer.status_code == 200 else get(ada, project, 1, keys[i])
        assert again.status_code == 200


def test_a_gate_turns_away_what_would_wait_too_long():
    async def scenario():
        gate = label_pyramid._Gate(slots=1, queue=1)
        order = []

        async def work(name):
            async with gate():
                order.append(name)
                await asyncio.sleep(0.05)

        first = asyncio.create_task(work("first"))
        await asyncio.sleep(0)
        second = asyncio.create_task(work("second"))
        await asyncio.sleep(0)
        with pytest.raises(label_pyramid.Busy):
            await work("third")
        await asyncio.gather(first, second)
        # The slot is free again, and the queue empty.
        await work("fourth")
        assert order == ["first", "second", "fourth"]

    asyncio.run(scenario())


def test_coarse_chunks_of_other_projects_and_people_stay_apart(
    ada, project, new_browser, migrated_database_url
):
    paint(ada, project, messy_volume())
    other = make_project(ada, migrated_database_url)
    assert chunk(ada, project, 1, (0, 0, 0)) is not None
    response = get(ada, other, 1, (0, 0, 0))
    assert (response.status_code, response.headers["x-chunk-version"]) == (404, "0")
    bob = new_browser()
    signup(bob, username="bob")
    assert get(bob, project, 1, (0, 0, 0)).status_code == 404
    assert get(new_browser(), project, 1, (0, 0, 0)).status_code == 401


def test_replacing_the_image_changes_what_coarse_chunks_say(ada, migrated_database_url):
    # These two images plan the same level 2, by way of different levels 1, so
    # the same labels make different pixels there under one factor.
    shape, one, other = (2, 200, 200), (1, 1, 1), (2.5, 1, 1)
    first, second = plan_levels(shape, one), plan_levels(shape, other)
    assert first[2].shape_zyx == second[2].shape_zyx
    assert first[2].factor_zyx == second[2].factor_zyx == (2, 4, 4)
    assert first[1].factor_zyx != second[1].factor_zyx
    project = make_project(ada, migrated_database_url, shape, voxel_size_zyx=one)
    volume = np.random.default_rng(5).integers(0, 4, shape).astype(np.uint8)
    paint(ada, project, volume)
    before = get(ada, project, 2, (0, 0, 0))
    assert before.status_code == 200
    np.testing.assert_array_equal(
        decode_chunk(before.content),
        expected_chunk(shrink(volume, first)[2], (0, 0, 0)),
    )

    add_image(migrated_database_url, project, shape, voxel_size_zyx=other)
    after = get(ada, project, 2, (0, 0, 0))
    assert after.status_code == 200
    # No label changed, but what a viewer kept no longer holds.
    assert after.headers["etag"] != before.headers["etag"]
    assert after.content != before.content
    stale = get(
        ada, project, 2, (0, 0, 0), headers={"If-None-Match": before.headers["etag"]}
    )
    assert stale.status_code == 200
    np.testing.assert_array_equal(
        decode_chunk(stale.content),
        expected_chunk(shrink(volume, second)[2], (0, 0, 0)),
    )


def test_retired_classes_count_as_unlabeled_in_coarse_levels(ada, project):
    levels = plan_levels(SHAPE)
    volume = np.zeros(SHAPE, dtype=np.uint8)
    # Bone, with one voxel of matrix in every block of eight, which bone
    # outvotes; and a patch of bone alone.
    volume[0:64, 0:64, 0:64] = 2
    volume[0:64:2, 0:64:2, 0:64:2] = 3
    volume[100:110, 100:110, 100:110] = 2
    paint(ada, project, volume)
    before = get(ada, project, 1, (0, 0, 0))
    assert (decode_chunk(before.content)[:32, :32, :32] == 2).all()

    removed = ada.request("DELETE", f"/api/projects/{project}/labels/classes/2")
    assert removed.status_code == 204
    # Viewers draw bone as nothing, so it can't hide the matrix: that shows,
    # and where only bone was, there is nothing to show.
    shown = np.where(volume == 2, 0, volume)
    check_every_level(ada, project, volume, levels, shown)
    after = get(ada, project, 1, (0, 0, 0))
    assert after.headers["etag"] != before.headers["etag"]
    assert (decode_chunk(after.content)[:32, :32, :32] == 3).all()
    stale = get(
        ada, project, 1, (0, 0, 0), headers={"If-None-Match": before.headers["etag"]}
    )
    assert stale.status_code == 200
    assert get(ada, project, 1, (1, 1, 1)).status_code == 404


def test_the_rule_and_the_codec_are_pinned_to_the_rule_version():
    # Viewers keep coarse chunks for as long as their ETag holds, which is only
    # right for the rule and codec that made them. If this fails, bump
    # `label_pyramid.RULE_VERSION`, then update what is pinned here.
    labels = np.array([0, 1, 2, 3, 4, 9], dtype=np.uint8)[
        (np.arange(8**3, dtype=np.uint64) * 2654435761 % 4294967296 >> 8) % 6
    ].reshape(8, 8, 8)
    # Some blocks with only background, and some with nothing.
    labels[:2] %= 2
    labels[:2, :, :4] = 0
    shrunk = downsample_labels(labels, (2, 2, 2))
    assert hashlib.sha256(shrunk.tobytes()).hexdigest() == RULE_PIN
    assert ZARR_CODECS == [
        {"name": "bytes"},
        {"name": "zstd", "configuration": {"level": 3, "checksum": False}},
    ]
    assert label_pyramid.RULE_VERSION == 2


def test_a_label_blob_gone_from_storage_is_an_error(ada, project, settings):
    paint(ada, project, messy_volume())
    root = pathlib.Path(settings.storage.url.removeprefix("file://"))
    shutil.rmtree(root / "projects" / project / "labels" / "blobs")
    response = get(ada, project, 1, (0, 0, 0))
    assert response.status_code == 500
    assert "missing from storage" in response.json()["detail"]


def test_the_cache_keeps_the_most_recent_chunks_under_its_limit():
    size = 100 + label_pyramid.ENTRY_OVERHEAD
    cache = label_pyramid._Cache(3 * size)
    for name in "abc":
        cache.put(name, b"x" * 100)
    assert cache.get("a") is not None
    cache.put("d", b"x" * 100)
    # "b" went, as the one used longest ago.
    assert [cache.get(name) is not None for name in "abcd"] == [True, False, True, True]
    cache.put("big", b"x" * 1000)
    assert cache.get("big") is None and cache.get("a") is not None
    cache.put("a", b"y" * 50)
    assert cache.get("a") == b"y" * 50
    assert cache.size == 2 * size + 50 + label_pyramid.ENTRY_OVERHEAD


def test_pinned_chunks_stay_until_let_go_unused_or_too_many():
    size = 100 + label_pyramid.ENTRY_OVERHEAD
    now = [0.0]

    def make(limit, pin_limit=100 * size):
        return label_pyramid._Cache(
            limit * size, pin_limit, pin_seconds=10, clock=lambda: now[0]
        )

    # The limit is for the others: pinned ones are over it, and stay.
    cache = make(limit=2)
    for name in "abc":
        cache.put(name, b"x" * 100, pin=True)
    for name in "def":
        cache.put(name, b"x" * 100)
    assert [cache.get(name) is not None for name in "abcdef"] == [True] * 3 + [
        False,
        True,
        True,
    ]
    assert (cache.size, cache.pinned_size) == (2 * size, 3 * size)
    # Letting one go makes it an ordinary entry, which the limit then counts.
    cache.unpin("a")
    assert cache.get("a") is not None and cache.get("d") is None
    assert (cache.size, cache.pinned_size) == (2 * size, 2 * size)

    # A pin nobody uses for ten seconds ends; using it starts it over.
    cache = make(limit=2)
    now[0] = 0
    for name in "ab":
        cache.put(name, b"x" * 100, pin=True)
    now[0] = 8
    assert cache.get("a") is not None
    now[0] = 12
    cache.put("c", b"x" * 100)
    cache.put("d", b"x" * 100)
    # "b" ended, and went first of the others.
    assert [cache.get(name) is not None for name in "abcd"] == [True, False, True, True]
    assert cache.pinned_size == size

    # Too many pinned: the ones pinned longest ago are let go.
    cache = make(limit=10, pin_limit=2 * size)
    for name in "abc":
        cache.put(name, b"x" * 100, pin=True)
    assert cache.pinned_size == 2 * size and cache.size == size
    assert [cache.get(name) is not None for name in "abc"] == [True] * 3


@pytest.mark.parametrize("room", [0.5, 0.8, 1.2])
@pytest.mark.parametrize("keys", [[(0, 0, 0)], [(0, 0, 0), (0, 0, 1)]])
def test_a_cold_chunk_finishes_however_little_room_the_cache_has(
    ada, project, room, keys
):
    # Dense labels, so every chunk of level 1 has plenty to keep.
    dense = np.random.default_rng(1).integers(0, 4, SHAPE).astype(np.uint8)
    paint(ada, project, dense)
    pyramid = ada.client.app.state.label_pyramid
    for key in keys:
        assert get(ada, project, 2, key).status_code == 200
    level1 = [
        len(data) + label_pyramid.ENTRY_OVERHEAD
        for (_, _, level, *_), data in pyramid._cache._items.items()
        if level == 1
    ]
    # Eight under the first chunk, four more under the second.
    assert len(level1) == (8, 12)[len(keys) - 1]
    # A cache with room for only some of the chunks of level 1 that these
    # chunks of level 2 are made of, and a request that makes just one.
    pyramid._cache = label_pyramid._Cache(int(room * sum(level1)))
    pyramid.blobs = 1
    attempts = {}
    for attempt in range(1, 41):
        for key in keys:
            if key not in attempts and get(ada, project, 2, key).status_code == 200:
                attempts[key] = attempt
    # It takes as many requests as it has chunks under it, and then one.
    assert sorted(attempts) == keys
    assert max(attempts.values()) <= 9 + len(keys) - 1
    # Nothing is left pinned once what they were pinned for is made.
    assert pyramid._cache.pinned_size == 0


def test_the_cache_follows_the_setting(settings):
    app = create_app(settings.model_copy(update={"label_cache_mb": 8}))
    assert app.state.label_pyramid._cache.limit == 8 * 1024 * 1024
    assert create_app(settings).state.label_pyramid._cache.limit == 64 * 1024 * 1024
