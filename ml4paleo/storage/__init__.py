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

`object_store` returns an obstore store rooted at the grant's location, and
`zarr_store` wraps that store for zarr-python. Code that reads or writes arrays
therefore has one path for every backend.
"""

from datetime import datetime
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

import zarr.storage
from obstore.store import GCSStore, LocalStore, S3Store
from pydantic import BaseModel, ConfigDict, field_validator

Scheme = Literal["file", "s3", "gs"]
SUPPORTED_SCHEMES: tuple[Scheme, ...] = ("file", "s3", "gs")

# Credential keys a grant may carry, per backend.
S3_CREDENTIAL_KEYS = ("access_key_id", "secret_access_key", "session_token")
GCS_CREDENTIAL_KEYS = ("service_account_key", "token")


class StorageGrant(BaseModel):
    """
    A storage location and the access needed to use it.

    Grants are immutable and serializable, so the server can hand them to
    workers. Use `child` to narrow a grant to a sub-location.
    """

    model_config = ConfigDict(frozen=True)

    url: str
    access: Literal["r", "rw"] = "r"
    credentials: dict[str, str] = {}
    endpoint: str | None = None
    region: str | None = None
    expires_at: datetime | None = None

    @field_validator("url")
    @classmethod
    def _validate_url(cls, url: str) -> str:
        parts = urlsplit(url)
        if parts.scheme not in SUPPORTED_SCHEMES:
            raise ValueError(
                f"Unsupported storage URL scheme {parts.scheme!r}; "
                f"expected one of {SUPPORTED_SCHEMES}"
            )
        if parts.query or parts.fragment:
            raise ValueError("Storage URLs cannot have a query or fragment")
        if parts.scheme == "file":
            if parts.netloc not in ("", "localhost"):
                raise ValueError("file:// URLs must not name a host")
            if not parts.path.startswith("/"):
                raise ValueError("file:// URLs must use an absolute path")
        elif not parts.netloc:
            raise ValueError(f"{parts.scheme}:// URLs must name a bucket")
        _check_path_segments(unquote(parts.path).strip("/"))
        return url.rstrip("/") if parts.path not in ("", "/") else url

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
        path = unquote(urlsplit(self.url).path)
        return path if self.scheme == "file" else path.strip("/")

    def child(self, relative_path: str) -> "StorageGrant":
        """
        Return a grant for `relative_path` under this grant's location.

        The path must be relative and must not contain `.` or `..` segments,
        so a child grant can never point outside its parent.
        """
        relative_path = relative_path.strip("/")
        _check_path_segments(relative_path)
        if not relative_path:
            return self
        return self.model_copy(update={"url": f"{self.url.rstrip('/')}/{relative_path}"})


def object_store(grant: StorageGrant) -> LocalStore | S3Store | GCSStore:
    """
    Return an obstore store rooted at the grant's location.
    """
    if grant.scheme == "file":
        return LocalStore(grant.path, mkdir=grant.access == "rw")
    if grant.scheme == "s3":
        return _s3_store(grant)
    return _gcs_store(grant)


def zarr_store(grant: StorageGrant) -> zarr.storage.ObjectStore:
    """
    Return a zarr store for the grant's location. Read-only grants give a
    read-only store.
    """
    return zarr.storage.ObjectStore(object_store(grant), read_only=grant.access == "r")


def _s3_store(grant: StorageGrant) -> S3Store:
    _check_credential_keys(grant, S3_CREDENTIAL_KEYS)
    config: dict[str, Any] = {
        key: value for key, value in grant.credentials.items() if value
    }
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
        **config,
    )


def _gcs_store(grant: StorageGrant) -> GCSStore:
    _check_credential_keys(grant, GCS_CREDENTIAL_KEYS)
    kwargs: dict[str, Any] = {}
    if service_account_key := grant.credentials.get("service_account_key"):
        kwargs["service_account_key"] = service_account_key
    if token := grant.credentials.get("token"):
        expires_at = grant.expires_at

        def provide_token():
            return {"token": token, "expires_at": expires_at}

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
    if "\\" in path:
        raise ValueError(f"Storage paths cannot contain backslashes: {path!r}")
    # Split by hand: PurePosixPath would silently drop "." and empty segments.
    for segment in path.split("/"):
        if segment in ("", ".", ".."):
            raise ValueError(
                f"Storage paths cannot contain empty, '.', or '..' segments: {path!r}"
            )


__all__ = [
    "SUPPORTED_SCHEMES",
    "StorageGrant",
    "object_store",
    "zarr_store",
]
