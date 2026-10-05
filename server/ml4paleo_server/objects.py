"""
Serving stored objects over HTTP: GET with a byte range, and HEAD.

Both the worker storage proxy and the data gateway use these. They speak what
zarr readers send (obstore's HTTP store, Neuroglancer, zarrita): a single
range per request, as `bytes=a-b`, `bytes=a-` (from an offset), or `bytes=-n`
(the last n bytes, which sharded zarr uses to read a shard's index).
"""

from email.utils import format_datetime
from typing import TYPE_CHECKING, Any

import obstore
from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse

if TYPE_CHECKING:
    from obstore import ObjectMeta


def parse_range(header: str) -> Any:
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


def meta_headers(meta: "ObjectMeta") -> dict[str, str]:
    headers = {"Last-Modified": format_datetime(meta["last_modified"], usegmt=True)}
    if e_tag := meta.get("e_tag"):
        headers["ETag"] = e_tag
    return headers


async def serve(
    store, key: str, request: Request, headers: dict[str, str] | None = None
) -> Response:
    """
    Answer a GET or HEAD for one object; 404 if it doesn't exist (which zarr
    readers take to mean an empty chunk). `headers` are added to the answer.
    """
    extra = headers or {}
    if request.method == "HEAD":
        try:
            meta = await obstore.head_async(store, key)
        except FileNotFoundError:
            return Response(status_code=404, headers=extra)
        return Response(
            headers={
                "Content-Length": str(meta["size"]),
                "Accept-Ranges": "bytes",
                **meta_headers(meta),
                **extra,
            }
        )
    range_header = request.headers.get("range")
    options: Any = {"range": parse_range(range_header)} if range_header else {}
    try:
        result = await obstore.get_async(store, key, options=options)
    except FileNotFoundError:
        return Response(status_code=404, headers=extra)
    except Exception as exc:  # noqa: BLE001 - obstore reports bad ranges generically
        if range_header:
            raise HTTPException(
                status_code=416, detail="Range not satisfiable."
            ) from exc
        raise
    start, end = result.range
    response_headers = {
        "Content-Length": str(end - start),
        "Accept-Ranges": "bytes",
        **meta_headers(result.meta),
        **extra,
    }
    if range_header:
        response_headers["Content-Range"] = (
            f"bytes {start}-{end - 1}/{result.meta['size']}"
        )

    async def body():
        async for chunk in result.stream():
            yield memoryview(chunk)

    return StreamingResponse(
        body(),
        status_code=206 if range_header else 200,
        headers=response_headers,
        media_type="application/octet-stream",
    )
