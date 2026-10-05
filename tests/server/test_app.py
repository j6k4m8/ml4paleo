"""
The API skeleton: health, security headers, SPA serving, and migrations.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
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
