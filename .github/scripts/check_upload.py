"""
Upload a file the way a browser does, against the compose stack: sign up,
create a project, start an upload, PUT its part straight to SeaweedFS through
Caddy with the presigned URL, and complete it. Also check that storage
refuses a part of the wrong length (the URL signs it), that a part URL can't
be turned into a copy of another object, and that requests other than signed
part uploads never reach SeaweedFS. Then upload a zip of image slices and
ingest it: the local worker turns it into the project's image.

    python check_upload.py https://localhost
"""

import http.cookiejar
import io
import json
import secrets
import ssl
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib

# Caddy serves "localhost" with its own certificate authority.
CONTEXT = ssl._create_unverified_context()


def is_neuroglancer_link(value: str | None) -> bool:
    """Allow the bundled viewer's cache-version query without relaxing its route."""
    link = urllib.parse.urlsplit(value or "")
    return (
        not link.scheme
        and not link.netloc
        and link.path == "/neuroglancer/"
        and link.fragment.startswith("!")
    )


def png(width: int, height: int, value: int) -> bytes:
    """
    A grayscale PNG (the runner's Python has no imaging library).
    """

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    rows = b"".join(
        b"\x00" + bytes([(value + x) % 256 for x in range(width)])
        for _ in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def main(origin: str) -> int:
    cookies = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cookies),
        urllib.request.HTTPSHandler(context=CONTEXT),
    )
    csrf = ""

    def api(method: str, path: str, body=None) -> dict:
        nonlocal csrf
        request = urllib.request.Request(
            origin + path,
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Content-Type": "application/json",
                "Origin": origin,
                "X-CSRF-Token": csrf,
            },
        )
        with opener.open(request) as response:
            answer = json.loads(response.read() or b"{}")
        csrf = answer.get("csrf_token", csrf)
        return answer

    def put(url: str, data: bytes, headers=None) -> tuple[int, bytes]:
        request = urllib.request.Request(
            url, data=data, method="PUT", headers=headers or {}
        )
        try:
            with urllib.request.urlopen(request, context=CONTEXT) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    api(
        "POST",
        "/api/auth/signup",
        {
            "username": "uploader",
            "email": "uploader@example.org",
            "password": secrets.token_urlsafe(18),
        },
    )
    project = api("POST", "/api/projects", {"name": "Upload check"})["id"]
    data = secrets.token_bytes(6 * 1024 * 1024)
    upload = api(
        "POST",
        f"/api/projects/{project}/uploads",
        {"filename": "scan.zip", "size": len(data)},
    )
    base = f"/api/projects/{project}/uploads/{upload['id']}"
    url = api("POST", f"{base}/part-urls", {"parts": [1]})["urls"]["1"]
    problems = []
    if not url.startswith(origin + "/"):
        problems.append(f"part URL {url} is not on {origin}")
    status, body = put(url, data[:-1])
    if status != 403:
        problems.append(f"a short part got HTTP {status}, not 403: {body[:200]!r}")
    status, body = put(url, data)
    if status != 200:
        problems.append(f"the part upload got HTTP {status}: {body[:300]!r}")
    if api("GET", base)["stored_parts"] != [1]:
        problems.append("storage doesn't list the uploaded part")
    if api("POST", f"{base}/complete")["state"] != "complete":
        problems.append("the upload didn't complete")
    # A second upload's part URL, with a header that would make storage copy
    # the first file into it instead (the signature doesn't cover it). The
    # body has the signed length, so storage would accept the request: only
    # Caddy refusing it keeps the part empty.
    second = api(
        "POST",
        f"/api/projects/{project}/uploads",
        {"filename": "copy.zip", "size": len(data)},
    )
    second_base = f"/api/projects/{project}/uploads/{second['id']}"
    second_url = api("POST", f"{second_base}/part-urls", {"parts": [1]})["urls"]["1"]
    source = url.split("?")[0].removeprefix(origin + "/")
    status, body = put(
        second_url,
        bytes(len(data)),
        {
            "X-Amz-Copy-Source": source,
            "X-Amz-Copy-Source-Range": f"bytes=0-{len(data) - 1}",
        },
    )
    if status == 200 or api("GET", second_base)["stored_parts"]:
        problems.append(f"a part URL copied another object (HTTP {status})")
    # Unsigned requests to the bucket path go to the API, not to storage.
    unsigned = url.split("?")[0]
    status, body = put(unsigned, b"x")
    if status == 200 or b"<?xml" in body:
        problems.append(f"an unsigned PUT reached storage (HTTP {status})")
    try:
        with urllib.request.urlopen(unsigned, context=CONTEXT) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        body = error.read()
    if data[:64] in body or b"<?xml" in body:
        problems.append("a GET of the uploaded file reached storage")
    # Ingest a small stack of slices end to end.
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as stack:
        for z in range(20):
            stack.writestr(f"stack/slice_{z}.png", png(16, 12, z * 10))
    scan = archive.getvalue()
    upload = api(
        "POST",
        f"/api/projects/{project}/uploads",
        {"filename": "stack.zip", "size": len(scan)},
    )
    base = f"/api/projects/{project}/uploads/{upload['id']}"
    status, body = put(
        api("POST", f"{base}/part-urls", {"parts": [1]})["urls"]["1"], scan
    )
    if status != 200 or api("POST", f"{base}/complete")["state"] != "complete":
        problems.append(f"uploading the stack failed (HTTP {status}): {body[:200]!r}")
    pipeline = api(
        "POST", f"/api/projects/{project}/ingest", {"upload_id": upload["id"]}
    )
    deadline = time.monotonic() + 300
    while pipeline["status"] not in ("succeeded", "failed", "cancelled"):
        if time.monotonic() > deadline:
            break
        time.sleep(2)
        pipeline = api("GET", f"/api/projects/{project}/pipelines/{pipeline['id']}")
    if pipeline["status"] != "succeeded":
        problems.append(f"ingest ended {pipeline['status']}: {pipeline.get('error')}")
    else:
        image = api("GET", f"/api/projects/{project}/image")
        if image["manifest"]["shape_czyx"] != [1, 20, 12, 16]:
            problems.append(f"the image has shape {image['manifest']['shape_czyx']}")
        # Viewers read the image through the data gateway.
        metadata = api("GET", image["zarr_url"] + "zarr.json")
        if "ome" not in metadata.get("attributes", {}):
            problems.append("the gateway didn't serve the image's OME-Zarr metadata")
        if not is_neuroglancer_link(image.get("neuroglancer_url")):
            problems.append("there is no Neuroglancer link for the image")
    # The web app, on any route, with its start script allowed by hash.
    with opener.open(origin + "/p/" + project + "/annotate") as response:
        page = response.read()
        policy = response.headers.get("Content-Security-Policy", "")
    if b"/_app/immutable/" not in page or "'sha256-" not in policy or "eval" in policy:
        problems.append("the web app isn't served, or its start script isn't allowed")
    # Neuroglancer itself, with its own content security policy.
    with opener.open(origin + "/neuroglancer/") as response:
        page = response.read()
        policy = response.headers.get("Content-Security-Policy", "")
    if b"neuroglancer" not in page.lower() or "wasm-unsafe-eval" not in policy:
        problems.append("Neuroglancer isn't served at /neuroglancer/")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
