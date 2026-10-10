"""Lossless HTTP compression, without changing stored objects or streaming events."""

import zlib

from fastapi import HTTPException, Request
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MINIMUM_SIZE = 1024
MESH_MEDIA_TYPE = "application/vnd.ml4paleo.mesh-preview"


def _accepts_gzip(value: str) -> bool:
    qualities = {}
    for item in value.lower().split(","):
        coding, *parameters = item.strip().split(";")
        quality = 1.0
        for parameter in parameters:
            name, _, setting = parameter.strip().partition("=")
            if name == "q":
                try:
                    quality = float(setting)
                except ValueError:
                    quality = 0.0
        qualities[coding.strip()] = quality if 0 <= quality <= 1 else 0.0
    return qualities.get("gzip", qualities.get("*", 0)) > 0


def _compressible(message: Message) -> bool:
    headers = Headers(raw=message["headers"])
    media = headers.get("content-type", "").partition(";")[0].strip().lower()
    if (
        message["status"] in (204, 206, 304)
        or "content-encoding" in headers
        or "content-range" in headers
        or "no-transform" in headers.get("cache-control", "").lower()
        or media == "text/event-stream"
    ):
        return False
    if int(headers.get("content-length", str(MINIMUM_SIZE))) < MINIMUM_SIZE:
        return False
    # Binary chunks are already zstd-compressed. Also leave image/video formats,
    # archives, and arbitrary downloads alone; their object bytes must survive.
    return (
        media.startswith("text/")
        or media.endswith("+json")
        or media
        in {
            "application/json",
            "application/javascript",
            "application/wasm",
            "image/svg+xml",
            MESH_MEDIA_TYPE,
        }
    )


class CompressionMiddleware:
    """Use Starlette's streaming gzip at a fast level, only where it helps.

    Keep the original scope for the application. The normalized negotiation
    passed to gzip honors q=0 and wildcard preferences (not substring matching).
    Auth responses can contain secrets and reflected input: never compress them.
    Worker storage traffic and ranges retain their exact object representation.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] == "HEAD"
            or "range" in Headers(scope=scope)
            or scope["path"].startswith(("/api/auth/", "/api/worker/"))
        ):
            await self.app(scope, receive, send)
            return

        async def select(_scope: Scope, receive: Receive, gzip_send: Send) -> None:
            bypass = False

            async def choose(message: Message) -> None:
                nonlocal bypass
                if message["type"] == "http.response.start":
                    bypass = not _compressible(message)
                await (send if bypass else gzip_send)(message)

            await self.app(scope, receive, choose)

        async def encoded_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=message["headers"])
                if headers.get("content-encoding") == "gzip":
                    # Strong validators and byte ranges describe the original
                    # file bytes, not this negotiated representation.
                    etag = headers.get("etag")
                    if etag and not etag.startswith("W/"):
                        headers["ETag"] = "W/" + etag
                    if "accept-ranges" in headers:
                        del headers["accept-ranges"]
            await send(message)

        accepted = _accepts_gzip(Headers(scope=scope).get("accept-encoding", ""))
        headers = [(k, v) for k, v in scope["headers"] if k != b"accept-encoding"]
        headers.append((b"accept-encoding", b"gzip" if accepted else b"identity"))
        await GZipMiddleware(select, minimum_size=MINIMUM_SIZE, compresslevel=3)(
            {**scope, "headers": headers}, receive, encoded_send
        )


def _inflate(data: bytes, limit: int) -> bytes:
    try:
        decoder = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
        raw = decoder.decompress(data, limit + 1)
    except zlib.error:
        raise HTTPException(400, "Invalid gzip body.") from None
    if len(raw) > limit:
        raise HTTPException(413, "3D preview is too large.")
    if not decoder.eof or decoder.unused_data:
        raise HTTPException(400, "Incomplete or trailing gzip data.")
    return raw


async def preview_body(request: Request, limit: int) -> bytes:
    """Bound both wire and inflated bytes; support gzip only on this endpoint."""
    encoding = request.headers.get("content-encoding", "identity").strip().lower()
    if encoding not in ("identity", "gzip"):
        raise HTTPException(415, "Expected identity or gzip Content-Encoding.")
    body = bytearray()
    async for part in request.stream():
        if len(body) + len(part) > limit:
            raise HTTPException(413, "3D preview is too large.")
        body.extend(part)
    raw = bytes(body)
    return await run_in_threadpool(_inflate, raw, limit) if encoding == "gzip" else raw
