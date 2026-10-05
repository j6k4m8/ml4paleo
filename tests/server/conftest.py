"""
Fixtures for server tests: a Postgres server, a fresh database per test, and
an app client.

Postgres comes from `M4P_TEST_DATABASE_URL` when set (CI uses a service
container). Otherwise a throwaway cluster is started with the local `initdb`
and `pg_ctl`, and tests that need it are skipped if those are not installed.
"""

import os
import shutil
import socket
import subprocess
import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient
from ml4paleo_server import migrations
from ml4paleo_server.app import create_app
from ml4paleo_server.settings import Settings
from sqlalchemy.engine import make_url


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def postgres_server_url(tmp_path_factory):
    if url := os.environ.get("M4P_TEST_DATABASE_URL"):
        yield url
        return
    initdb, pg_ctl = shutil.which("initdb"), shutil.which("pg_ctl")
    if not (initdb and pg_ctl):
        pytest.skip("Postgres is not installed; set M4P_TEST_DATABASE_URL")
    data_dir = tmp_path_factory.mktemp("postgres")
    port = _free_port()
    # Postgres refuses to start on macOS without a valid locale in LC_ALL.
    env = {**os.environ, "LC_ALL": "C"}
    subprocess.run(
        [initdb, "-D", data_dir, "-U", "postgres", "--auth=trust", "-E", "UTF8"],
        check=True,
        capture_output=True,
        env=env,
    )
    subprocess.run(
        [
            pg_ctl,
            "-D",
            data_dir,
            "-o",
            f"-p {port} -c listen_addresses=127.0.0.1 -k ''",
            "-l",
            data_dir / "server.log",
            "-w",
            "start",
        ],
        check=True,
        capture_output=True,
        env=env,
    )
    try:
        yield f"postgresql+psycopg://postgres@127.0.0.1:{port}/postgres"
    finally:
        subprocess.run(
            [pg_ctl, "-D", data_dir, "-m", "immediate", "stop"], capture_output=True
        )


@pytest.fixture
def database_url(postgres_server_url):
    """
    Create an empty database for one test and drop it afterwards.
    """
    server = make_url(postgres_server_url)
    name = f"test_{uuid.uuid4().hex[:12]}"
    admin_dsn = server.set(drivername="postgresql").render_as_string(
        hide_password=False
    )
    with psycopg.connect(admin_dsn, autocommit=True) as connection:
        connection.execute(f'CREATE DATABASE "{name}"')
    try:
        yield server.set(database=name).render_as_string(hide_password=False)
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as connection:
            connection.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
def migrated_database_url(database_url):
    migrations.upgrade(database_url)
    return database_url


@pytest.fixture
def settings(migrated_database_url, tmp_path):
    return Settings(
        database_url=migrated_database_url,
        secret_key="test-secret-key-that-is-long-enough-0123456789",
        storage={"url": f"file://{tmp_path}/data"},
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


class Browser:
    """
    A test client that behaves like the web app: it keeps cookies and sends
    the CSRF token it got from the last session response.
    """

    def __init__(self, client: TestClient):
        self.client = client
        self.csrf_token: str | None = None

    def request(self, method: str, url: str, **kwargs):
        headers = dict(kwargs.pop("headers", None) or {})
        if self.csrf_token and method.upper() not in SAFE_METHODS:
            headers.setdefault("X-CSRF-Token", self.csrf_token)
        response = self.client.request(method, url, headers=headers, **kwargs)
        if response.content and response.headers.get("content-type", "").startswith(
            "application/json"
        ):
            body = response.json()
            if isinstance(body, dict) and "csrf_token" in body:
                self.csrf_token = body["csrf_token"]
        return response

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def put(self, url, **kwargs):
        return self.request("PUT", url, **kwargs)


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
def new_browser(app):
    """
    Return a function that opens a new browser (its own cookies) on the app.
    """
    clients = []

    def open_browser(custom_settings: Settings | None = None) -> Browser:
        """
        Open a browser on the shared app, or on a new app built from
        `custom_settings`.
        """
        client = TestClient(create_app(custom_settings) if custom_settings else app)
        client.__enter__()
        clients.append(client)
        return Browser(client)

    yield open_browser
    for client in clients:
        client.__exit__(None, None, None)
