"""
The storage proxy: workers without storage credentials of their own read
and write their job's files through the API.

`ml4paleo.storage.object_store` talks to it with obstore's HTTP store, which
uses plain HTTP: GET (with Range), HEAD, PUT (one request per object; there
are no multipart uploads), DELETE, and WebDAV PROPFIND for listing. Each
route serves one grant of one job (`/jobs/<job>/storage/<n>/<key>`), and the
bearer token is that job's lease token, so access ends when the lease does.

Requests don't hold a database connection while bytes stream: the lease is
checked in a short session of its own.
"""

import datetime
import hashlib
import secrets
import uuid
from email.utils import format_datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import quote
from xml.sax.saxutils import escape

import obstore
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ml4paleo.storage import StorageGrant, object_store

from ..db import Job
from ..storage import project_storage

if TYPE_CHECKING:
    from obstore import ObjectMeta

router = APIRouter(
    prefix="/api/worker/v1/jobs/{job_id}/storage/{index}",
    tags=["worker"],
    include_in_schema=False,
)

# One object per PUT; the largest shard is well under this.
MAX_PUT_BYTES = 4 * 1024**3
_KEY_CHECK = StorageGrant(url="s3://key-check")


async def _store(request: Request, job_id: uuid.UUID, index: int, write: bool):
    """
    Check the lease token and return an obstore store for the grant.
    """
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    job = None
    if scheme.lower() == "bearer" and token:
        async with request.app.state.sessionmaker() as db:
            job = await db.get(Job, job_id)
    if (
        job is None
        or job.status != "leased"
        or job.lease_token_hash is None
        or job.lease_expires_at is None
        or job.lease_expires_at <= datetime.datetime.now(datetime.UTC)
        or not secrets.compare_digest(
            job.lease_token_hash, hashlib.sha256(token.encode()).hexdigest()
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="This job's lease is not held with that token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not 0 <= index < len(job.grants):
        raise HTTPException(status_code=404, detail="No such grant.")
    spec = job.grants[index]
    if write and spec["access"] != "rw":
        raise HTTPException(status_code=403, detail="This grant is read-only.")
    return object_store(project_storage(request.app.state.settings).child(spec["path"]))


def _checked(key: str) -> str:
    try:
        _KEY_CHECK.child(key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return key.strip("/")


def _parse_range(header: str) -> Any:
    """
    Turn an HTTP Range header (one range) into obstore's range option.
    """
    unit, _, spec = header.partition("=")
    first, sep, last = spec.strip().partition("-")
    try:
        if unit.strip() != "bytes" or not sep or "," in spec:
            raise ValueError
        if not first:
            return {"suffix": int(last)}
        if not last:
            return {"offset": int(first)}
        start, end = int(first), int(last) + 1
        if end <= start:
            raise ValueError
        return (start, end)
    except ValueError:
        raise HTTPException(status_code=416, detail="Unsupported Range.") from None


def _meta_headers(meta: "ObjectMeta") -> dict[str, str]:
    headers = {"Last-Modified": format_datetime(meta["last_modified"], usegmt=True)}
    if e_tag := meta.get("e_tag"):
        headers["ETag"] = e_tag
    return headers


@router.api_route("/{key:path}", methods=["GET", "HEAD"])
async def read(job_id: uuid.UUID, index: int, key: str, request: Request) -> Response:
    store = await _store(request, job_id, index, write=False)
    key = _checked(key)
    if request.method == "HEAD":
        try:
            meta = await obstore.head_async(store, key)
        except FileNotFoundError:
            return Response(status_code=404)
        return Response(
            headers={"Content-Length": str(meta["size"]), **_meta_headers(meta)}
        )
    range_header = request.headers.get("range")
    options: Any = {"range": _parse_range(range_header)} if range_header else {}
    try:
        result = await obstore.get_async(store, key, options=options)
    except FileNotFoundError:
        return Response(status_code=404)
    except Exception as exc:  # noqa: BLE001 - obstore reports bad ranges generically
        if range_header:
            raise HTTPException(
                status_code=416, detail="Range not satisfiable."
            ) from exc
        raise
    start, end = result.range
    size = result.meta["size"]
    headers = {
        "Content-Length": str(end - start),
        "Accept-Ranges": "bytes",
        **_meta_headers(result.meta),
    }
    if range_header:
        headers["Content-Range"] = f"bytes {start}-{end - 1}/{size}"

    async def body():
        async for chunk in result.stream():
            yield memoryview(chunk)

    return StreamingResponse(
        body(),
        status_code=206 if range_header else 200,
        headers=headers,
        media_type="application/octet-stream",
    )


@router.put("/{key:path}", status_code=201)
async def write(job_id: uuid.UUID, index: int, key: str, request: Request) -> Response:
    store = await _store(request, job_id, index, write=True)
    key = _checked(key)
    received = 0

    async def body():
        nonlocal received
        async for chunk in request.stream():
            received += len(chunk)
            if received > MAX_PUT_BYTES:
                raise ValueError("too large")
            if chunk:
                yield chunk

    try:
        await obstore.put_async(store, key, body())
    except Exception:
        if received > MAX_PUT_BYTES:
            raise HTTPException(
                status_code=413, detail=f"Objects are limited to {MAX_PUT_BYTES} bytes."
            ) from None
        raise
    return Response(status_code=201)


@router.delete("/{key:path}", status_code=204)
async def remove(job_id: uuid.UUID, index: int, key: str, request: Request) -> None:
    store = await _store(request, job_id, index, write=True)
    try:
        await obstore.delete_async(store, _checked(key))
    except FileNotFoundError:
        pass


@router.api_route("", methods=["PROPFIND"])
async def list_root(job_id: uuid.UUID, index: int, request: Request) -> Response:
    return await _list(request, job_id, index, None)


@router.api_route("/{key:path}", methods=["PROPFIND"])
async def list_prefix(
    job_id: uuid.UUID, index: int, key: str, request: Request
) -> Response:
    return await _list(request, job_id, index, _checked(key) or None)


async def _list(
    request: Request, job_id: uuid.UUID, index: int, prefix: str | None
) -> Response:
    """
    Answer a WebDAV PROPFIND. Each href is the key relative to the grant
    (with a leading "/"), which is how obstore's HTTP store maps hrefs back
    to keys.
    """
    store = await _store(request, job_id, index, write=False)
    objects: list[ObjectMeta] = []
    prefixes: list[str] = []
    if request.headers.get("depth") == "1":
        listed = await obstore.list_with_delimiter_async(store, prefix=prefix)
        objects, prefixes = list(listed["objects"]), list(listed["common_prefixes"])
    else:
        async for batch in obstore.list(store, prefix=prefix):
            objects.extend(batch)
    if not objects and not prefixes:
        return Response(status_code=404)
    responses = [
        "<D:response><D:href>{}</D:href><D:propstat><D:prop>"
        "<D:getcontentlength>{}</D:getcontentlength>"
        "<D:getlastmodified>{}</D:getlastmodified><D:resourcetype/>"
        "</D:prop><D:status>HTTP/1.1 200 OK</D:status></D:propstat></D:response>".format(
            escape(quote("/" + meta["path"])),
            meta["size"],
            format_datetime(meta["last_modified"], usegmt=True),
        )
        for meta in objects
    ]
    # obstore's parser wants a modification time on "directories" too.
    listed_at = format_datetime(datetime.datetime.now(datetime.UTC), usegmt=True)
    responses += [
        "<D:response><D:href>{}</D:href><D:propstat><D:prop>"
        "<D:getlastmodified>{}</D:getlastmodified>"
        "<D:resourcetype><D:collection/></D:resourcetype></D:prop>"
        "<D:status>HTTP/1.1 200 OK</D:status></D:propstat></D:response>".format(
            escape(quote("/" + common.rstrip("/") + "/")), listed_at
        )
        for common in prefixes
    ]
    xml = (
        '<?xml version="1.0" encoding="utf-8"?><D:multistatus xmlns:D="DAV:">'
        + "".join(responses)
        + "</D:multistatus>"
    )
    return Response(content=xml, status_code=207, media_type="application/xml")
