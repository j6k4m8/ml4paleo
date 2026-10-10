"""
The API skeleton: health, security headers, SPA serving, and migrations.
"""

import base64
import hashlib
import http.client
import uuid
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient
from helpers import NOT_JSON, strict_json
from ml4paleo_server import migrations
from ml4paleo_server.app import create_app
from ml4paleo_server.db import uuid7
from ml4paleo_server.settings import Settings


def test_health_checks_the_database(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_security_headers_are_set(client):
    headers = client.get("/api/health").headers
    assert "frame-ancestors 'none'" in headers["content-security-policy"]
    assert headers["x-content-type-options"] == "nosniff"
    assert "access-control-allow-origin" not in headers


def test_unknown_api_paths_are_404_not_the_web_app(client):
    for path in ["/api", "/api/nope", "/api/projects/x/y"]:
        response = client.get(path)
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")


def test_placeholder_page_without_a_web_build(client):
    response = client.get("/p/123/annotate")
    assert response.status_code == 200
    assert "has not been built" in response.text


@pytest.fixture
def web_dir(tmp_path):
    web = tmp_path / "web"
    (web / "_app" / "immutable").mkdir(parents=True)
    (web / "index.html").write_text("<html>app shell</html>")
    (web / "favicon.png").write_bytes(b"png")
    (web / "_app" / "immutable" / "start.abc123.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("outside the web dir")
    return web


def test_web_app_serves_files_and_falls_back_to_index(settings, web_dir):
    settings = settings.model_copy(update={"web_dir": web_dir})
    with TestClient(create_app(settings)) as client:
        asset = client.get("/_app/immutable/start.abc123.js")
        assert asset.text == "console.log(1)"
        assert "immutable" in asset.headers["cache-control"]
        assert client.get("/favicon.png").headers["cache-control"] == "no-cache"
        for path in ["/", "/p/123/annotate", "/..%2Fsecret.txt", "/%2E%2E/secret.txt"]:
            response = client.get(path)
            assert response.status_code == 200
            assert response.text == "<html>app shell</html>", path


def test_the_web_apps_inline_script_is_allowed_by_hash(settings, web_dir):
    script = "\n\t\t\t{ start(); }\n\t\t"
    (web_dir / "index.html").write_text(
        f'<html><script src="/x.js"></script><script>{script}</script></html>'
    )
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    settings = settings.model_copy(update={"web_dir": web_dir})
    with TestClient(create_app(settings)) as client:
        policy = client.get("/projects").headers["content-security-policy"]
    script_src = next(d for d in policy.split("; ") if d.startswith("script-src"))
    assert script_src.split() == ["script-src", "'self'", f"'sha256-{digest}'"]


def test_large_web_assets_compress_but_ranges_remain_exact(settings, web_dir):
    data = b"console.log('a large test bundle');\n" * 4000
    (web_dir / "_app" / "immutable" / "large.js").write_bytes(data)
    settings = settings.model_copy(update={"web_dir": web_dir})
    with TestClient(create_app(settings)) as client:
        url = "/_app/immutable/large.js"
        packed = client.get(url, headers={"Accept-Encoding": "gzip"})
        plain = client.get(url, headers={"Accept-Encoding": "identity"})
        assert packed.content == plain.content == data
        assert packed.headers["content-encoding"] == "gzip"
        assert packed.headers["etag"] == "W/" + plain.headers["etag"]
        assert "Accept-Encoding" in packed.headers["vary"]
        assert "immutable" in packed.headers["cache-control"]
        part = client.get(
            url, headers={"Accept-Encoding": "gzip", "Range": "bytes=7-31"}
        )
        assert part.status_code == 206
        assert part.content == data[7:32]
        assert "content-encoding" not in part.headers
        assert part.headers["content-range"] == f"bytes 7-31/{len(data)}"


def test_migrations_round_trip_and_match_the_models(database_url):
    migrations.upgrade(database_url)
    migrations.check(database_url)
    migrations.downgrade(database_url, "base")
    migrations.upgrade(database_url)


def test_settings_read_secrets_from_files(monkeypatch, tmp_path):
    (tmp_path / "key").write_text("from-a-file\n")
    (tmp_path / "s3").write_text("s3-secret")
    monkeypatch.setenv("M4P_SECRET_KEY_FILE", str(tmp_path / "key"))
    monkeypatch.setenv("M4P_STORAGE__SECRET_ACCESS_KEY_FILE", str(tmp_path / "s3"))
    monkeypatch.setenv("M4P_STORAGE__URL", "s3://bucket")
    monkeypatch.setenv("M4P_PUBLIC_URL", "https://example.org/")
    settings = Settings()
    assert settings.secret_key.get_secret_value() == "from-a-file"
    assert settings.storage.secret_access_key.get_secret_value() == "s3-secret"
    assert settings.storage.url == "s3://bucket"
    assert settings.public_url == "https://example.org"
    assert settings.is_https
    # A direct value wins over a file.
    monkeypatch.setenv("M4P_SECRET_KEY", "direct")
    assert Settings().secret_key.get_secret_value() == "direct"


def test_uuid7_sorts_by_time_and_sets_version_bits():
    ids = [uuid7() for _ in range(5)]
    assert all(i.version == 7 and i.variant == uuid.RFC_4122 for i in ids)
    assert [i.int >> 80 for i in ids] == sorted(i.int >> 80 for i in ids)


def test_plain_http_on_a_public_address_is_refused(settings):
    public_http = settings.model_copy(
        update={"public_url": "http://ml4paleo.example.org"}
    )
    with pytest.raises(RuntimeError, match="HTTPS"):
        create_app(public_http)
    create_app(public_http.model_copy(update={"allow_insecure_http": True}))
    create_app(
        settings.model_copy(update={"public_url": "https://ml4paleo.example.org"})
    )


def test_health_fails_when_storage_is_unreachable(settings, tmp_path):
    broken = settings.model_copy(
        update={
            "storage": settings.storage.model_copy(
                update={"url": "s3://missing-bucket", "endpoint": "http://127.0.0.1:9"}
            )
        }
    )
    with TestClient(create_app(broken), raise_server_exceptions=False) as client:
        assert client.get("/api/health").status_code == 500


def test_empty_secret_files_are_ignored(monkeypatch, tmp_path):
    (tmp_path / "empty").write_text("\n")
    monkeypatch.setenv("M4P_STORAGE__SECRET_ACCESS_KEY_FILE", str(tmp_path / "empty"))
    assert Settings().storage.secret_access_key is None


def login_with(literal: str) -> str:
    """A login whose password is `literal`, written into the JSON as it is (NaN, say, unquoted)."""
    return '{"username": "ada", "password": ' + literal + "}"


@pytest.mark.parametrize("literal", NOT_JSON)
def test_a_number_json_cannot_hold_is_a_422_a_browser_can_read(client, literal):
    response = client.post(
        "/api/auth/login",
        content=login_with(literal),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422, response.text
    assert response.headers["content-type"].startswith("application/json")
    [error] = strict_json(response.text)["detail"]
    # The usual error, as the web app reads it (where, and what), with the input written as text.
    assert error["loc"] == ["body", "password"]
    assert error["type"] == "string_type" and "string" in error["msg"]
    assert error["input"] == NOT_JSON[literal]


def test_other_validation_errors_keep_their_usual_shape(client):
    response = client.post("/api/auth/login", json={"username": "ada"})
    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert error["loc"] == ["body", "password"]
    assert (error["type"], error["msg"]) == ("missing", "Field required")
    assert error["input"] == {"username": "ada"}


@pytest.mark.parametrize("literal", NOT_JSON)
def test_a_number_json_cannot_hold_gets_its_422_over_a_real_connection(
    live_server, literal
):
    where = urlsplit(live_server)
    connection = http.client.HTTPConnection(where.hostname, where.port, timeout=10)
    try:
        connection.request(
            "POST",
            "/api/auth/login",
            body=login_with(literal),
            headers={"Content-Type": "application/json"},
        )
        # A dropped connection raises here instead of answering.
        response = connection.getresponse()
        assert response.status == 422
        [error] = strict_json(response.read().decode())["detail"]
        assert (
            error["loc"] == ["body", "password"] and error["input"] == NOT_JSON[literal]
        )
    finally:
        connection.close()
    # And the server goes on.
    connection = http.client.HTTPConnection(where.hostname, where.port, timeout=10)
    try:
        connection.request("GET", "/api/health")
        assert connection.getresponse().status == 200
    finally:
        connection.close()


def can_hold_nan(spec: dict) -> set[tuple[str, str, str]]:
    """
    The request fields (method, path, field) in an OpenAPI spec that could be
    sent a number JSON can't hold, or something with one in it: a number that is
    not bounded on both sides, and anything free-form (an object or list of no
    particular shape).
    """
    schemas = spec["components"]["schemas"]
    found: set[tuple[str, str, str]] = set()

    def walk(schema: dict, where: tuple[str, str], name: str, seen: frozenset) -> None:
        if "$ref" in schema:
            ref = schema["$ref"].split("/")[-1]
            if ref not in seen:
                walk(schemas[ref], where, name, seen | {ref})
            return
        for key in ("anyOf", "oneOf", "allOf"):
            for option in schema.get(key, []):
                walk(option, where, name, seen)
        if any(key in schema for key in ("anyOf", "oneOf", "allOf")):
            return
        kind = schema.get("type")
        if kind == "number":
            bounded = "minimum" in schema and "maximum" in schema
            if not bounded:
                found.add((*where, name))
        elif kind == "object" or "properties" in schema:
            for field, sub in schema.get("properties", {}).items():
                walk(sub, where, f"{name}.{field}", seen)
            extra = schema.get("additionalProperties")
            if not schema.get("properties") and extra in (None, True, {}):
                found.add((*where, name))
            elif isinstance(extra, dict) and extra:
                walk(extra, where, name + "{}", seen)
        elif kind == "array":
            options = [
                *schema.get("prefixItems", []),
                *([schema["items"]] if "items" in schema else []),
            ]
            if not options:
                found.add((*where, name))
            for option in options:
                walk(option, where, name + "[]", seen)
        elif kind is None and not schema.get("enum") and "const" not in schema:
            found.add((*where, name))

    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            where = (method.upper(), path)
            for parameter in operation.get("parameters", []):
                walk(parameter.get("schema", {}), where, parameter["name"], frozenset())
            for content in operation.get("requestBody", {}).get("content", {}).values():
                walk(content.get("schema", {}), where, "body", frozenset())
    return found


# Each of these is checked by a test of its own, which sends it NaN and the
# infinities and expects a 422. Free-form fields hold JSON, which the database
# stores and can't hold them (`ml4paleo.protocol.json_text` refuses them); a
# number with no upper bound would take infinity (so they say `allow_inf_nan=False`).
FREE_FORM_FIELDS = {
    ("POST", "/api/projects/{project_id}/labels/ops", "body.tool"),
    ("POST", "/api/projects/{project_id}/models", "body.params"),
    ("POST", "/api/worker/v1/jobs/{job_id}/complete", "body.result"),
    ("POST", "/api/worker/v1/jobs/{job_id}/label-ops", "body.tool"),
    ("POST", "/api/worker/v1/jobs/{job_id}/label-ops", "body.deltas[]"),
}
UNBOUNDED_NUMBERS = (
    {
        ("PUT", "/api/admin/users/{user_id}/quota", f"body.{limit}")
        for limit in ("storage_gb", "cpu_hours_per_day", "gpu_hours_per_day")
    }
    | {
        (
            "POST",
            "/api/admin/quota-requests/{request_id}",
            f"body.quota_override.{limit}",
        )
        for limit in ("storage_gb", "cpu_hours_per_day", "gpu_hours_per_day")
    }
    | {
        (method, path, name)
        for method, path in (
            ("POST", "/api/worker/v1/hello"),
            ("POST", "/api/worker/v1/claim"),
        )
        for name in ("body.caps.vram_gb", "body.caps.memory_gb")
    }
    | {("POST", "/api/worker/v1/claim", "body.wait_seconds")}
)


def test_a_new_request_field_that_could_hold_nan_is_one_that_is_checked(app):
    found = can_hold_nan(app.openapi())
    new = found - FREE_FORM_FIELDS - UNBOUNDED_NUMBERS
    assert not new, (
        f"{sorted(new)} could be sent NaN or infinity, which the database can't "
        "store (a 500). Refuse them (`json_text` for JSON, `allow_inf_nan=False` "
        "or bounds for a number), test it, and add the field to the lists here."
    )
    assert found == FREE_FORM_FIELDS | UNBOUNDED_NUMBERS
