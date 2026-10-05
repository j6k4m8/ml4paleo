"""
The storage layer gives one code path for zarr arrays on local disk and on
S3-compatible object stores, and grants can never reach outside their prefix.
"""

import numpy as np
import obstore
import pytest
import zarr

from ml4paleo.storage import (
    StorageGrant,
    delete_object,
    get_bytes,
    object_store,
    put_bytes,
    zarr_store,
)
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
        "s3://user:secret@bucket/data",
        "s3://bucket/a\x00b",
        "file:///tmp/a\nb",
    ],
)
def test_unsafe_urls_are_rejected(url):
    with pytest.raises(ValueError):
        StorageGrant(url=url)


@pytest.mark.parametrize(
    "relative_path",
    [
        "../escape",
        "a/../../b",
        "a/./b",
        "a\\b",
        "%2e%2e/p2",
        "a%2F..%2F..%2Fp2",
        "a?b/c",
        "a#b",
    ],
)
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


def test_object_helpers_round_trip_and_respect_read_only(grant):
    put_bytes(grant, "notes/a.txt", b"hello")
    assert get_bytes(grant, "notes/a.txt") == b"hello"
    assert get_bytes(grant, "notes/missing.txt") is None
    read_only = grant.model_copy(update={"access": "r"})
    assert get_bytes(read_only, "notes/a.txt") == b"hello"
    with pytest.raises(PermissionError):
        put_bytes(read_only, "notes/b.txt", b"nope")
    with pytest.raises(PermissionError):
        delete_object(read_only, "notes/a.txt")
    delete_object(grant, "notes/a.txt")
    assert get_bytes(grant, "notes/a.txt") is None
    with pytest.raises(ValueError):
        put_bytes(grant, "../escape.txt", b"nope")


def test_credentials_are_hidden_from_repr_but_sent_as_json():
    grant = StorageGrant(
        url="s3://bucket/p",
        credentials={"access_key_id": "AKIA", "secret_access_key": "very-secret"},
    )
    assert "very-secret" not in repr(grant)
    assert "very-secret" not in str(grant)
    round_tripped = StorageGrant.model_validate_json(grant.model_dump_json())
    assert round_tripped.secret("secret_access_key") == "very-secret"
    assert grant.child("a/b").secret("secret_access_key") == "very-secret"


def test_refresh_supplies_credentials(s3_endpoint, tmp_path):
    calls = []

    def refresh():
        calls.append(1)
        return StorageGrant(
            url=f"s3://{BUCKET}/refresh/{tmp_path.name}",
            access="rw",
            endpoint=s3_endpoint,
            credentials={"access_key_id": "test", "secret_access_key": "test"},
        )

    grant = StorageGrant(
        url=f"s3://{BUCKET}/refresh/{tmp_path.name}",
        access="rw",
        endpoint=s3_endpoint,
    )
    store = object_store(grant, refresh=refresh)
    obstore.put(store, "x.txt", b"x")
    assert obstore.get(store, "x.txt").bytes().to_bytes() == b"x"
    assert calls
