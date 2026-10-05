"""
The storage layer gives one code path for zarr arrays on local disk and on
S3-compatible object stores, and grants can never reach outside their prefix.
"""

import numpy as np
import pytest
import zarr

from ml4paleo.storage import StorageGrant, object_store, zarr_store
from ml4paleo.volume_providers import NumpyVolumeProvider, ZarrVolumeProvider
from ml4paleo.volume_providers.io import export_zarr_array

BUCKET = "ml4paleo-test"


@pytest.fixture(scope="module")
def s3_endpoint():
    """
    Run moto's S3 server in-process, so the S3 path is tested without Docker.
    """
    import boto3
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    ).create_bucket(Bucket=BUCKET)
    yield endpoint
    server.stop()


@pytest.fixture(params=["file", "s3"])
def grant(request, tmp_path):
    if request.param == "file":
        return StorageGrant(url=f"file://{tmp_path}/project", access="rw")
    endpoint = request.getfixturevalue("s3_endpoint")
    return StorageGrant(
        url=f"s3://{BUCKET}/projects/{tmp_path.name}",
        access="rw",
        endpoint=endpoint,
        credentials={"access_key_id": "test", "secret_access_key": "test"},
    )


def test_zarr_arrays_round_trip(grant):
    data = np.random.default_rng(0).integers(0, 65535, size=(6, 10, 8), dtype=np.uint16)
    array_grant = grant.child("artifacts/image")
    array = zarr.create_array(
        store=zarr_store(array_grant),
        shape=data.shape,
        chunks=(4, 4, 4),
        dtype=data.dtype,
    )
    array[:] = data
    array.attrs["voxel_size_xyz_mm"] = [0.5, 0.5, 2.0]

    read_back = ZarrVolumeProvider(array_grant.model_copy(update={"access": "r"}))
    np.testing.assert_array_equal(read_back[:, :, :], data)
    assert read_back.voxel_size_xyz_mm == (0.5, 0.5, 2.0)


def test_child_grants_write_under_their_prefix(grant):
    store = object_store(grant.child("artifacts/a"))
    store.put("hello.txt", b"hi")
    parent = object_store(grant)
    keys = [item["path"] for batch in parent.list() for item in batch]
    assert keys == ["artifacts/a/hello.txt"]


def test_read_only_grants_reject_writes(grant):
    array_grant = grant.child("ro")
    zarr.create_array(store=zarr_store(array_grant), shape=(2,), dtype="uint8")
    read_only = zarr_store(array_grant.model_copy(update={"access": "r"}))
    with pytest.raises(ValueError):
        zarr.open_array(store=read_only, mode="r+")[:] = 1


@pytest.mark.parametrize(
    "url",
    [
        "s3://bucket/a/../b",
        "s3://bucket/a/./b",
        "s3://bucket/a//b",
        "file://relative/path",
        "file:///tmp/..",
        "s3:///no-bucket",
        "http://example.com/data",
        "s3://bucket/data?versionId=1",
    ],
)
def test_unsafe_urls_are_rejected(url):
    with pytest.raises(ValueError):
        StorageGrant(url=url)


@pytest.mark.parametrize("relative_path", ["../escape", "a/../../b", "a/./b", "a\\b"])
def test_child_paths_cannot_escape(relative_path):
    with pytest.raises(ValueError):
        StorageGrant(url="s3://bucket/projects/p1").child(relative_path)


def test_unknown_credential_keys_are_rejected():
    grant = StorageGrant(url="s3://bucket/p", credentials={"password": "x"})
    with pytest.raises(ValueError):
        object_store(grant)


def test_v1_zarr_v2_arrays_are_still_readable(tmp_path):
    data = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    export_zarr_array(
        NumpyVolumeProvider(data), tmp_path / "v1.zarr", chunk_size=(2, 2, 2)
    )
    assert (tmp_path / "v1.zarr" / ".zarray").exists()
    np.testing.assert_array_equal(
        ZarrVolumeProvider(tmp_path / "v1.zarr")[:, :, :], data
    )
