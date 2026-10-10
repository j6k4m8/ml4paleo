"""Assert the actual ASGI wire bytes, not HTTPX's transparently decoded body."""

import asyncio
import gzip

import pytest
from fastapi import HTTPException, Request
from ml4paleo_server.compression import CompressionMiddleware, preview_body

DATA = b'{"voxels": [0, 0, 2, 2, 0], "name": "bone"}' * 150


def wire(
    *,
    body=DATA,
    media="application/json",
    encoding="gzip",
    path="/api/test",
    method="GET",
    request_headers=(),
    response_headers=(),
    status=200,
    streaming=False,
):
    messages = []
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(b"accept-encoding", encoding.encode()), *request_headers],
    }

    async def send(message):
        messages.append(message)

    async def receive():
        raise AssertionError("Response must not read the request body")

    async def app(original_scope, receive, send):
        assert original_scope == scope
        headers = [(b"content-type", media.encode()), *response_headers]
        if not streaming:
            headers.append((b"content-length", str(len(body)).encode()))
        await send(
            {"type": "http.response.start", "status": status, "headers": headers}
        )
        if streaming:
            await send(
                {"type": "http.response.body", "body": body[:10], "more_body": True}
            )
            if media == "text/event-stream":
                # First event is available now, not when the stream closes.
                assert len(messages) == 2 and messages[1]["body"] == body[:10]
            await send({"type": "http.response.body", "body": body[10:]})
        else:
            await send({"type": "http.response.body", "body": body})

    asyncio.run(CompressionMiddleware(app)(scope, receive, send))
    headers = dict(messages[0]["headers"])
    return headers, b"".join(m.get("body", b"") for m in messages[1:])


@pytest.mark.parametrize(
    "media",
    [
        "application/json",
        "text/javascript",
        "text/css",
        "text/html",
        "image/svg+xml",
        "application/wasm",
        "application/problem+json",
        "application/vnd.ml4paleo.mesh-preview",
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_text_json_meshes_compress_losslessly(media, streaming):
    headers, body = wire(
        media=media,
        streaming=streaming,
        response_headers=[
            (b"vary", b"Origin"),
            (b"etag", b'"123"'),
            (b"accept-ranges", b"bytes"),
        ],
    )
    assert headers[b"content-encoding"] == b"gzip"
    assert gzip.decompress(body) == DATA
    assert len(body) < len(DATA) // 10
    assert headers[b"vary"] == b"Origin, Accept-Encoding"
    assert headers[b"etag"] == b'W/"123"'
    assert b"accept-ranges" not in headers
    if streaming:
        assert b"content-length" not in headers
    else:
        assert int(headers[b"content-length"]) == len(body)


@pytest.mark.parametrize(
    "encoding, accepted",
    [
        ("gzip", True),
        ("GZip; q=0.5", True),
        ("br, gzip;q=1", True),
        ("*", True),
        ("gzip;q=0, *;q=1", False),
        ("*;q=0", False),
        ("gzip;q=wat", False),
        ("gzip;q=NaN", False),
        ("gzip;q=2", False),
        ("notgzip", False),
        ("identity", False),
        ("", False),
    ],
)
def test_content_negotiation(encoding, accepted):
    headers, body = wire(encoding=encoding)
    assert (headers.get(b"content-encoding") == b"gzip") == accepted
    assert (gzip.decompress(body) if accepted else body) == DATA
    assert headers[b"vary"] == b"Accept-Encoding"


@pytest.mark.parametrize(
    "options",
    [
        {"media": "application/octet-stream"},
        {"media": "application/zstd"},
        {"media": "image/png"},
        {"media": "application/zip"},
        {"media": "text/event-stream", "streaming": True},
        {"media": "text/event-stream; charset=utf-8"},
        {"request_headers": [(b"range", b"bytes=0-100")]},
        {"method": "HEAD"},
        {"status": 206},
        {"status": 304},
        {"status": 204},
        {"response_headers": [(b"content-range", b"bytes 0-100/1000")]},
        {"response_headers": [(b"cache-control", b"private, no-transform")]},
        {"path": "/api/auth/session"},
        {"path": "/api/worker/v1/jobs/1/storage/0/zarr.json"},
    ],
)
def test_do_not_transform_chunks_ranges_events_or_secrets(options):
    headers, body = wire(**options)
    assert b"content-encoding" not in headers
    assert body == DATA


def test_small_and_preencoded_responses_stay_unchanged():
    assert wire(body=b"tiny")[1] == b"tiny"
    assert b"content-encoding" not in wire(body=b"tiny")[0]
    compressed = gzip.compress(DATA)
    headers, body = wire(
        body=compressed, response_headers=[(b"content-encoding", b"gzip")]
    )
    assert body == compressed
    assert headers[b"content-encoding"] == b"gzip"


def read_body(body, encoding="gzip", limit=8192):
    parts = [body[i : i + 17] for i in range(0, len(body), 17)] or [b""]

    async def receive():
        return {"type": "http.request", "body": parts.pop(0), "more_body": bool(parts)}

    request = Request(
        {"type": "http", "headers": [(b"content-encoding", encoding.encode())]}, receive
    )
    return asyncio.run(preview_body(request, limit))


def test_request_gzip_decodes_fragmented_input_and_preserves_identity():
    assert read_body(gzip.compress(DATA)) == DATA
    assert read_body(DATA, encoding="identity") == DATA
    assert read_body(gzip.compress(b"x" * 8192)) == b"x" * 8192


@pytest.mark.parametrize(
    "body, encoding, status",
    [
        (gzip.compress(b"x" * 8193), "gzip", 413),
        (b"x" * 8193, "identity", 413),
        (b"x" * 8193, "gzip", 413),
        (b"not gzip", "gzip", 400),
        (gzip.compress(DATA)[:-1], "gzip", 400),
        (gzip.compress(DATA) + b"trailing", "gzip", 400),
        (gzip.compress(DATA) + gzip.compress(DATA), "gzip", 400),
        (DATA, "br", 415),
        (DATA, "gzip, gzip", 415),
    ],
)
def test_request_bounds_and_invalid_encodings(body, encoding, status):
    with pytest.raises(HTTPException) as error:
        read_body(body, encoding)
    assert error.value.status_code == status
