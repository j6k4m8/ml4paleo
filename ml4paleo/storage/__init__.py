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

`zarr_store` gives zarr-python a store rooted at the grant's location, and
`get_bytes`, `put_bytes`, and `delete_object` cover plain objects. Code that
reads or writes data therefore has one path for every backend.

Read-only grants are enforced by these helpers and by the zarr store. The raw
obstore handle from `object_store` cannot refuse writes, so read-only grants
should also carry read-only credentials (the credential broker issues those).
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlsplit

import obstore
import zarr.storage
from obstore.store import GCSStore, LocalStore, S3Store
from pydantic import (
    BaseModel,
    ConfigDict,
    SecretStr,
    field_serializer,
    field_validator,
)

Scheme = Literal["file", "s3", "gs"]
SUPPORTED_SCHEMES: tuple[Scheme, ...] = ("file", "s3", "gs")

# Credential keys a grant may carry, per backend.
S3_CREDENTIAL_KEYS = ("access_key_id", "secret_access_key", "session_token")
GCS_CREDENTIAL_KEYS = ("service_account_key", "token")

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
            raise ValueError(f"{parts.scheme}:// URLs must name a bucket")
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
        The bucket name, or None for local disk.
        """
        return None if self.scheme == "file" else urlsplit(self.url).netloc

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
) -> LocalStore | S3Store | GCSStore:
    """
    Return an obstore store rooted at the grant's location.

    For long jobs whose temporary credentials expire, pass `refresh`: a
    function that returns a fresh grant for the same location. The store calls
    it whenever its credentials are about to expire.
    """
    if grant.scheme == "file":
        return LocalStore(grant.path, mkdir=grant.access == "rw")
    if grant.scheme == "s3":
        return _s3_store(grant, refresh)
    return _gcs_store(grant, refresh)


def zarr_store(
    grant: StorageGrant, refresh: Refresh | None = None
) -> zarr.storage.ObjectStore:
    """
    Return a zarr store for the grant's location. Read-only grants give a
    read-only store.
    """
    return zarr.storage.ObjectStore(
        object_store(grant, refresh), read_only=grant.access == "r"
    )


def get_bytes(grant: StorageGrant, key: str) -> bytes | None:
    """
    Read one object under the grant, or return None if it doesn't exist.
    """
    _check_path_segments(key)
    try:
        return obstore.get(object_store(grant), key).bytes().to_bytes()
    except FileNotFoundError:
        return None


def put_bytes(grant: StorageGrant, key: str, data: bytes) -> None:
    """
    Write one object under the grant. Refuses read-only grants.
    """
    _check_writable(grant)
    _check_path_segments(key)
    obstore.put(object_store(grant), key, data)


def delete_object(grant: StorageGrant, key: str) -> None:
    """
    Delete one object under the grant. Refuses read-only grants.
    """
    _check_writable(grant)
    _check_path_segments(key)
    obstore.delete(object_store(grant), key)


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
    "SUPPORTED_SCHEMES",
    "StorageGrant",
    "delete_object",
    "get_bytes",
    "object_store",
    "put_bytes",
    "zarr_store",
]
