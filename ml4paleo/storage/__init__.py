"""
One storage layer for local disk, S3-compatible object stores, and Google
Cloud Storage.

Every read or write of volume data goes through a `StorageGrant`: a URL that
names a location, plus the credentials needed to reach it. The URL scheme picks
the backend:

- `file:///absolute/path`: local disk, for tests and single-machine use.
- `s3://bucket/prefix`: AWS S3, or any S3-compatible service (SeaweedFS, R2)
  when `endpoint` is set.
- `gs://bucket/prefix`: Google Cloud Storage.
- `https://server/api/worker/v1/jobs/<job>/storage/<n>` (or `http://` on a
  private network): the API server's storage proxy, for workers that have no
  storage credentials of their own. The only credential is `token`, the job's
  lease token, so access ends when the lease does.

`zarr_store` gives zarr-python a store rooted at the grant's location, and
`get_bytes`, `put_bytes`, `put_file`, and `delete_object` cover plain
objects. Code that reads or writes data therefore has one path for every
backend.

A job that produces an artifact writes `MANIFEST_KEY` last (`write_manifest`);
the server commits the artifact only if it finds one.

Read-only grants are enforced by these helpers and by the zarr store. The raw
obstore handle from `object_store` cannot refuse writes, so read-only grants
should also carry read-only credentials (the credential broker issues those).
"""

import http.client
import io
import json
import os
import pathlib
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal
from urllib.parse import quote, urlsplit

import obstore
import zarr.storage
from obstore.store import GCSStore, HTTPStore, LocalStore, S3Store
from pydantic import (
    BaseModel,
    ConfigDict,
    SecretStr,
    field_serializer,
    field_validator,
)

Scheme = Literal["file", "s3", "gs", "http", "https"]
SUPPORTED_SCHEMES: tuple[Scheme, ...] = ("file", "s3", "gs", "http", "https")
PROXY_SCHEMES = ("http", "https")
# The storage proxy takes each object in one request, of at most this size.
PROXY_MAX_OBJECT_BYTES = 4 * 1024**3
PROXY_PUT_ATTEMPTS = 3
MANIFEST_KEY = "_MANIFEST.json"

# Credential keys a grant may carry, per backend.
S3_CREDENTIAL_KEYS = ("access_key_id", "secret_access_key", "session_token")
GCS_CREDENTIAL_KEYS = ("service_account_key", "token")
PROXY_CREDENTIAL_KEYS = ("token",)

# Characters never allowed in storage paths. Rejecting "%" means a path can't
# smuggle an encoded "..", and "?" and "#" would end the path part of a URL.
FORBIDDEN_PATH_CHARACTERS = frozenset("%?#\\")


class StorageGrant(BaseModel):
    """
    A storage location and the access needed to use it.

    Grants are immutable and serializable, so the server can hand them to
    workers. Use `child` to narrow a grant to a sub-location.

    Credentials are hidden from `repr` and `str`, so a logged grant doesn't
    leak them. JSON serialization does include them, because that is how the
    server sends grants to workers.
    """

    model_config = ConfigDict(frozen=True)

    url: str
    access: Literal["r", "rw"] = "r"
    credentials: dict[str, SecretStr] = {}
    endpoint: str | None = None
    region: str | None = None
    expires_at: datetime | None = None

    @field_validator("url")
    @classmethod
    def _validate_url(cls, url: str) -> str:
        # Check before parsing: urlsplit silently drops tabs and newlines.
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in url):
            raise ValueError("Storage URLs cannot contain control characters")
        parts = urlsplit(url)
        if parts.scheme not in SUPPORTED_SCHEMES:
            raise ValueError(
                f"Unsupported storage URL scheme {parts.scheme!r}; "
                f"expected one of {SUPPORTED_SCHEMES}"
            )
        if parts.query or parts.fragment:
            raise ValueError("Storage URLs cannot have a query or fragment")
        if parts.username is not None or parts.password is not None:
            raise ValueError("Put credentials in the grant, not in the URL")
        if parts.scheme == "file":
            if parts.netloc not in ("", "localhost"):
                raise ValueError("file:// URLs must not name a host")
            if not parts.path.startswith("/"):
                raise ValueError("file:// URLs must use an absolute path")
        elif not parts.netloc:
            raise ValueError(f"{parts.scheme}:// URLs must name a bucket or host")
        _check_path_segments(parts.path.strip("/"))
        return url.rstrip("/") if parts.path not in ("", "/") else url

    @field_serializer("credentials", when_used="json")
    def _reveal_credentials(self, credentials: dict[str, SecretStr]) -> dict[str, str]:
        return {key: value.get_secret_value() for key, value in credentials.items()}

    def secret(self, key: str) -> str | None:
        value = self.credentials.get(key)
        return value.get_secret_value() if value is not None else None

    @property
    def scheme(self) -> Scheme:
        return urlsplit(self.url).scheme  # type: ignore[return-value]

    @property
    def bucket(self) -> str | None:
        """
        The bucket name, or None for local disk and the storage proxy.
        """
        if self.scheme == "file" or self.scheme in PROXY_SCHEMES:
            return None
        return urlsplit(self.url).netloc

    @property
    def path(self) -> str:
        """
        The path within the bucket (or the absolute path on local disk).
        """
        path = urlsplit(self.url).path
        return path if self.scheme == "file" else path.strip("/")

    def child(self, relative_path: str) -> "StorageGrant":
        """
        Return a grant for `relative_path` under this grant's location.

        The path must be relative and must not contain `.` or `..` segments
        (or `%`, so they can't be encoded), so a child grant can never point
        outside its parent.
        """
        relative_path = relative_path.strip("/")
        _check_path_segments(relative_path)
        if not relative_path:
            return self
        # Validate the new URL as a whole, rather than copying the model.
        return StorageGrant.model_validate(
            {**self.model_dump(), "url": f"{self.url.rstrip('/')}/{relative_path}"}
        )


Refresh = Callable[[], StorageGrant]


def object_store(
    grant: StorageGrant, refresh: Refresh | None = None
) -> LocalStore | S3Store | GCSStore | HTTPStore:
    """
    Return an obstore store rooted at the grant's location.

    For long jobs whose temporary credentials expire, pass `refresh`: a
    function that returns a fresh grant for the same location. The store calls
    it whenever its credentials are about to expire.

    The storage proxy's store can't do multipart uploads: write through
    `put_bytes`, `put_file`, or `zarr_store`, which send each object in one
    request.
    """
    if grant.scheme == "file":
        return LocalStore(grant.path, mkdir=grant.access == "rw")
    if grant.scheme == "s3":
        return _s3_store(grant, refresh)
    if grant.scheme in PROXY_SCHEMES:
        return _proxy_store(grant)
    return _gcs_store(grant, refresh)


def zarr_store(
    grant: StorageGrant, refresh: Refresh | None = None
) -> zarr.storage.ObjectStore:
    """
    Return a zarr store for the grant's location. Read-only grants give a
    read-only store.
    """
    store_class = (
        _SinglePutZarrStore
        if grant.scheme in PROXY_SCHEMES
        else zarr.storage.ObjectStore
    )
    return store_class(object_store(grant, refresh), read_only=grant.access == "r")


class _SinglePutZarrStore(zarr.storage.ObjectStore):
    """
    A zarr store that sends every object in one request, for the storage
    proxy (obstore's HTTP store has no multipart uploads).
    """

    async def set(self, key: str, value) -> None:
        self._check_writable()
        await obstore.put_async(
            self.store, key, value.as_buffer_like(), use_multipart=False
        )

    async def set_if_not_exists(self, key: str, value) -> None:
        self._check_writable()
        if not await self.exists(key):
            await self.set(key, value)


def get_bytes(grant: StorageGrant, key: str) -> bytes | None:
    """
    Read one object under the grant, or return None if it doesn't exist.
    """
    _check_path_segments(key)
    try:
        return obstore.get(object_store(grant), key).bytes().to_bytes()
    except FileNotFoundError:
        return None


class _RangeReader(io.RawIOBase):
    """
    A seekable, read-only file over one stored object, fetching byte ranges
    as they are read. Wrap it in `io.BufferedReader` to fetch in large blocks.
    """

    def __init__(self, store, key: str, size: int):
        self._store = store
        self._key = key
        self._size = size
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self._size}
        self._position = max(0, base[whence] + offset)
        return self._position

    def readinto(self, buffer) -> int:
        start = self._position
        length = min(len(buffer), self._size - start)
        if length <= 0:
            return 0
        data = obstore.get_range(self._store, self._key, start=start, length=length)
        view = memoryview(data)
        buffer[: len(view)] = view
        self._position += len(view)
        return len(view)


def open_object(
    grant: StorageGrant, key: str, buffer_size: int = 4 * 1024 * 1024
) -> io.BufferedReader:
    """
    Open one stored object as a seekable binary file, without downloading it:
    reads fetch byte ranges of `buffer_size` (so, for example, `zipfile` can
    read single members of a huge archive).
    """
    _check_path_segments(key)
    store = object_store(grant)
    size = obstore.head(store, key)["size"]
    return io.BufferedReader(_RangeReader(store, key, size), buffer_size=buffer_size)


def put_bytes(grant: StorageGrant, key: str, data: bytes) -> None:
    """
    Write one object under the grant. Refuses read-only grants.
    """
    _check_writable(grant)
    _check_path_segments(key)
    obstore.put(
        object_store(grant),
        key,
        data,
        use_multipart=False if grant.scheme in PROXY_SCHEMES else None,
    )


def put_file(grant: StorageGrant, key: str, path: str | os.PathLike[str]) -> None:
    """
    Write one object under the grant from a local file, without reading the
    file into memory: obstore sends it in parts, and the storage proxy
    (which takes each object in one request, of at most
    `PROXY_MAX_OBJECT_BYTES`) gets it as a stream. Refuses read-only grants.
    """
    _check_writable(grant)
    _check_path_segments(key)
    path = pathlib.Path(path)
    if grant.scheme in PROXY_SCHEMES:
        _put_to_proxy(grant, key, path)
    else:
        obstore.put(object_store(grant), key, path)


def _put_to_proxy(grant: StorageGrant, key: str, path: pathlib.Path) -> None:
    # obstore's HTTP store would hold the whole file in memory (twice) to
    # send it in one request.
    size = path.stat().st_size
    if size > PROXY_MAX_OBJECT_BYTES:
        raise ValueError(
            f"{key} is {size} bytes; the storage proxy takes objects of up to "
            f"{PROXY_MAX_OBJECT_BYTES}"
        )
    url = f"{grant.url}/{quote(key)}"
    headers = {"Content-Length": str(size), "Content-Type": "application/octet-stream"}
    if token := grant.secret("token"):
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(1, PROXY_PUT_ATTEMPTS + 1):
        try:
            status, detail = _stream_put(url, headers, path)
        except (OSError, http.client.HTTPException):
            # The connection failed; try again, as obstore would.
            if attempt == PROXY_PUT_ATTEMPTS:
                raise
        else:
            if status < 300:
                return
            if status < 500 or attempt == PROXY_PUT_ATTEMPTS:
                raise OSError(f"The storage proxy refused {key}: {status} {detail}")
        time.sleep(2**attempt)


def _stream_put(
    url: str, headers: dict[str, str], path: pathlib.Path
) -> tuple[int, str]:
    # urllib sends a file body as it reads it, and goes through whatever
    # proxy the environment names, as obstore's own client does.
    with path.open("rb") as body:
        request = urllib.request.Request(url, data=body, headers=headers, method="PUT")
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return response.status, ""
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(500).decode(errors="replace")


def write_manifest(grant: StorageGrant, manifest: dict[str, Any]) -> None:
    """
    Write an artifact's manifest. Write it last: its presence tells the
    server that everything else is in place.
    """
    put_bytes(grant, MANIFEST_KEY, json.dumps(manifest, sort_keys=True).encode())


def delete_object(grant: StorageGrant, key: str) -> None:
    """
    Delete one object under the grant. Refuses read-only grants.

    An object that isn't there counts as deleted, as it does on S3 and
    through the storage proxy, so a retried job can delete what its last
    attempt already did.
    """
    _check_writable(grant)
    _check_path_segments(key)
    try:
        obstore.delete(object_store(grant), key)
    except FileNotFoundError:
        pass


def _check_writable(grant: StorageGrant) -> None:
    if grant.access != "rw":
        raise PermissionError(f"The grant for {grant.url} is read-only")


def _s3_store(grant: StorageGrant, refresh: Refresh | None) -> S3Store:
    _check_credential_keys(grant, S3_CREDENTIAL_KEYS)
    config: dict[str, Any] = {}
    kwargs: dict[str, Any] = {}
    if refresh is not None:

        def provide_credentials():
            fresh = refresh()
            return {
                "access_key_id": fresh.secret("access_key_id") or "",
                "secret_access_key": fresh.secret("secret_access_key") or "",
                "token": fresh.secret("session_token"),
                "expires_at": fresh.expires_at,
            }

        kwargs["credential_provider"] = provide_credentials
    else:
        config.update(
            {
                key: value.get_secret_value()
                for key, value in grant.credentials.items()
                if value.get_secret_value()
            }
        )
    client_options: dict[str, Any] = {}
    if grant.endpoint is not None:
        config["endpoint"] = grant.endpoint
        # S3-compatible services usually only support path-style requests.
        config["virtual_hosted_style_request"] = False
        if grant.endpoint.startswith("http://"):
            client_options["allow_http"] = True
    region = grant.region or ("us-east-1" if grant.endpoint else None)
    if region is not None:
        config["region"] = region
    return S3Store(
        grant.bucket,
        prefix=grant.path or None,
        client_options=client_options or None,  # type: ignore[arg-type]
        **kwargs,
        **config,
    )


def _proxy_store(grant: StorageGrant) -> HTTPStore:
    _check_credential_keys(grant, PROXY_CREDENTIAL_KEYS)
    client_options: dict[str, Any] = {"timeout": "600s"}
    if token := grant.secret("token"):
        client_options["default_headers"] = {"Authorization": f"Bearer {token}"}
    if grant.scheme == "http":
        client_options["allow_http"] = True
    return HTTPStore(grant.url, client_options=client_options)  # type: ignore[arg-type]


def _gcs_store(grant: StorageGrant, refresh: Refresh | None) -> GCSStore:
    _check_credential_keys(grant, GCS_CREDENTIAL_KEYS)
    kwargs: dict[str, Any] = {}
    if service_account_key := grant.secret("service_account_key"):
        kwargs["service_account_key"] = service_account_key
    if refresh is not None or grant.secret("token"):

        def provide_token():
            fresh = refresh() if refresh is not None else grant
            return {
                "token": fresh.secret("token") or "",
                "expires_at": fresh.expires_at,
            }

        kwargs["credential_provider"] = provide_token
    return GCSStore(grant.bucket, prefix=grant.path or None, **kwargs)


def _check_credential_keys(grant: StorageGrant, allowed: tuple[str, ...]) -> None:
    unknown = set(grant.credentials) - set(allowed)
    if unknown:
        raise ValueError(
            f"Unknown {grant.scheme} credential keys {sorted(unknown)}; "
            f"expected some of {list(allowed)}"
        )


def _check_path_segments(path: str) -> None:
    if not path:
        return
    if forbidden := FORBIDDEN_PATH_CHARACTERS.intersection(path):
        raise ValueError(f"Storage paths cannot contain {sorted(forbidden)}: {path!r}")
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in path):
        raise ValueError(f"Storage paths cannot contain control characters: {path!r}")
    # Split by hand: PurePosixPath would silently drop "." and empty segments.
    for segment in path.split("/"):
        if segment in ("", ".", ".."):
            raise ValueError(
                f"Storage paths cannot contain empty, '.', or '..' segments: {path!r}"
            )


__all__ = [
    "MANIFEST_KEY",
    "PROXY_MAX_OBJECT_BYTES",
    "SUPPORTED_SCHEMES",
    "StorageGrant",
    "delete_object",
    "get_bytes",
    "object_store",
    "open_object",
    "put_bytes",
    "put_file",
    "write_manifest",
    "zarr_store",
]
