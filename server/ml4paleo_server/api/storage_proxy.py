"""
The storage proxy: workers without storage credentials of their own read
and write their job's files through the API.

`ml4paleo.storage.object_store` talks to it with obstore's HTTP store, which
uses plain HTTP: GET (with Range), HEAD, PUT (one request per object; there
are no multipart uploads), DELETE, and WebDAV PROPFIND for listing. Each
route serves one grant of one job (`/jobs/<job>/storage/<n>/<key>`), and the
bearer token is that job's lease token, so access ends when the lease does.

Requests don't hold a database connection while bytes stream: the lease is
checked in a short session of its own. A PUT checks the lease again when its
body has arrived and holds a share lock on the job until the object is
stored, so an upload can't land after its job has finished (completing a job
waits for uploads in flight) or after another worker took the job over.
"""

import datetime
import hashlib
import secrets
import uuid
from email.utils import format_datetime
from typing import TYPE_CHECKING
from urllib.parse import quote
from xml.sax.saxutils import escape

import obstore
from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import select

from ml4paleo.storage import StorageGrant, object_store

from .. import objects
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


class LeaseEnded(Exception):
    """
    The lease ended while an upload was arriving.
    """


def _token(request: Request) -> str:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return token if scheme.lower() == "bearer" else ""


def _holds_lease(job: Job | None, token: str) -> bool:
    return (
        job is not None
        and bool(token)
        and job.status == "leased"
        and job.lease_token_hash is not None
        and job.lease_expires_at is not None
        and job.lease_expires_at > datetime.datetime.now(datetime.UTC)
        and secrets.compare_digest(
            job.lease_token_hash, hashlib.sha256(token.encode()).hexdigest()
        )
    )


async def _store(request: Request, job_id: uuid.UUID, index: int, write: bool):
    """
    Check the lease token and return an obstore store for the grant.
    """
    token = _token(request)
    job = None
    if token:
        async with request.app.state.sessionmaker() as db:
            job = await db.get(Job, job_id)
    if job is None or not _holds_lease(job, token):
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
    """
    Check an object key. The grant's root itself is not an object.
    """
    if not key.strip("/"):
        raise HTTPException(status_code=400, detail="Name an object.")
    try:
        _KEY_CHECK.child(key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return key.strip("/")


@router.api_route("/{key:path}", methods=["GET", "HEAD"])
async def read(job_id: uuid.UUID, index: int, key: str, request: Request) -> Response:
    store = await _store(request, job_id, index, write=False)
    return await objects.serve(store, _checked(key), request)


@router.put("/{key:path}", status_code=201)
async def write(job_id: uuid.UUID, index: int, key: str, request: Request) -> Response:
    store = await _store(request, job_id, index, write=True)
    key = _checked(key)
    token = _token(request)
    received = 0
    lease_ended = False
    # Opened lazily: its connection is used only from the end of the body
    # until the object is stored.
    async with request.app.state.sessionmaker() as db:

        async def body():
            nonlocal received, lease_ended
            async for chunk in request.stream():
                received += len(chunk)
                if received > MAX_PUT_BYTES:
                    raise ValueError("too large")
                if chunk:
                    yield chunk
            # The object becomes visible when this generator ends, so check
            # the lease now and keep it locked until the store finishes.
            try:
                await _hold_lease(db, job_id, token)
            except LeaseEnded:
                lease_ended = True
                # Raising aborts the upload, so nothing becomes visible.
                raise

        try:
            await obstore.put_async(store, key, body())
        except Exception:
            if received > MAX_PUT_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Objects are limited to {MAX_PUT_BYTES} bytes.",
                ) from None
            if lease_ended:
                raise HTTPException(
                    status_code=401,
                    detail="The job's lease ended during the upload.",
                    headers={"WWW-Authenticate": "Bearer"},
                ) from None
            raise
        await db.commit()
    return Response(status_code=201)


@router.delete("/{key:path}", status_code=204)
async def remove(job_id: uuid.UUID, index: int, key: str, request: Request) -> None:
    store = await _store(request, job_id, index, write=True)
    key = _checked(key)
    async with request.app.state.sessionmaker() as db:
        # Like a finished upload: hold the lease while the object goes, so a
        # delete can't take effect after the job's artifacts are committed.
        try:
            await _hold_lease(db, job_id, _token(request))
        except LeaseEnded:
            raise HTTPException(
                status_code=401,
                detail="This job's lease is not held with that token.",
                headers={"WWW-Authenticate": "Bearer"},
            ) from None
        try:
            await obstore.delete_async(store, key)
        except FileNotFoundError:
            pass
        await db.commit()


async def _hold_lease(db, job_id: uuid.UUID, token: str) -> None:
    """
    Share-lock the job until `db` commits, and check that `token` still
    holds its lease. Completing (or taking over) the job needs a stronger
    lock, so it waits until the caller is done.
    """
    job = await db.scalar(
        select(Job)
        .where(Job.id == job_id)
        .with_for_update(read=True)
        .execution_options(populate_existing=True)
    )
    if not _holds_lease(job, token):
        raise LeaseEnded


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
