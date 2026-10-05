"""
Upload a file the way a browser does, against the compose stack: sign up,
create a project, start an upload, PUT its part straight to SeaweedFS through
Caddy with the presigned URL, and complete it. Also check that storage
refuses a part of the wrong length (the URL signs it), and that requests
other than signed part uploads never reach SeaweedFS.

    python check_upload.py https://localhost
"""

import http.cookiejar
import json
import secrets
import ssl
import sys
import urllib.error
import urllib.request

# Caddy serves "localhost" with its own certificate authority.
CONTEXT = ssl._create_unverified_context()


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

    def put(url: str, data: bytes) -> tuple[int, bytes]:
        request = urllib.request.Request(url, data=data, method="PUT")
        try:
            with urllib.request.urlopen(request, context=CONTEXT) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    api(
        "POST",
        "/api/auth/signup",
        {"username": "uploader", "password": secrets.token_urlsafe(18)},
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
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
