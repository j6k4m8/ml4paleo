"""
The FastAPI application.

The API lives under `/api`. Every other GET serves the built web app (a
single-page app): existing files are served as-is and any other path gets
`index.html`, so client-side routes work on reload.
"""

import base64
import contextlib
import hashlib
import math
import pathlib
import re
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit

import obstore
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool

from ml4paleo.storage import object_store

from . import __version__
from .api import ROUTERS
from .auth.sessions import cookie_name
from .auth.tokens import csrf_token, tokens_match
from .compression import CompressionMiddleware
from .db import create_engine, create_sessionmaker
from .jobs import JobSignal
from .label_pyramid import LabelPyramid
from .mesh_preview import MeshPreviewPool
from .settings import Settings
from .storage import project_storage
from .viewer import (
    NEUROGLANCER_CONTENT_SECURITY_POLICY,
    NEUROGLANCER_PATH,
    neuroglancer_available,
)

MIN_SECRET_KEY_LENGTH = 32
STORAGE_CHECK_SECONDS = 30.0
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# Requests that may carry a stale session cookie, so they are only checked
# by origin. Logging in with a forged request is still blocked by the
# Origin check.
SESSION_CSRF_EXEMPT = {
    "/api/auth/login",
    "/api/auth/signup",
    "/api/auth/verify-email",
    "/api/auth/password-reset/request",
    "/api/auth/password-reset/confirm",
}

_INLINE_SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.DOTALL)


def content_security_policy(
    web_dir: pathlib.Path | None, upload_origin: str | None = None
) -> str:
    """
    The policy for everything but Neuroglancer. SvelteKit starts the web app
    with an inline script in `index.html`, allowed by its hash; nothing else
    runs inline, and nothing evaluates code (the viewer decodes zstd in plain
    JavaScript). Browsers upload file parts to `upload_origin` when storage
    is on another site (a cloud bucket).
    """
    scripts = ["'self'"]
    index = web_dir / "index.html" if web_dir is not None else None
    if index is not None and index.is_file():
        for body in _INLINE_SCRIPT.findall(index.read_text(encoding="utf-8")):
            digest = hashlib.sha256(body.encode()).digest()
            scripts.append(f"'sha256-{base64.b64encode(digest).decode()}'")
    return "; ".join(
        [
            "default-src 'self'",
            f"script-src {' '.join(scripts)}",
            # Svelte renders some style attributes (the route announcer).
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' blob: data:",
            "worker-src 'self' blob:",
            f"connect-src 'self'{f' {upload_origin}' if upload_origin else ''}",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
        ]
    )


PLACEHOLDER_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>ml4paleo</title></head>
<body><p>The ml4paleo API is running, but the web app has not been built.</p></body>
</html>
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    if not (settings.is_https or settings.is_local or settings.allow_insecure_http):
        raise RuntimeError(
            f"M4P_PUBLIC_URL is {settings.public_url}, which is plain HTTP on a "
            "public address. Serve the app over HTTPS, or set "
            "M4P_ALLOW_INSECURE_HTTP=true if you really mean it."
        )
    secret_key = settings.secret_key.get_secret_value()
    if len(secret_key) < MIN_SECRET_KEY_LENGTH:
        raise RuntimeError(
            f"M4P_SECRET_KEY must be at least {MIN_SECRET_KEY_LENGTH} random characters."
        )
    has_neuroglancer = neuroglancer_available(settings)
    parts = urlsplit(settings.public_url)
    public_origin = f"{parts.scheme}://{parts.netloc}"
    session_cookie = cookie_name(settings)
    app_policy = content_security_policy(settings.web_dir, _upload_origin(settings))

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(settings.database_url.get_secret_value())
        app.state.engine = engine
        app.state.sessionmaker = create_sessionmaker(engine)
        job_signal = JobSignal(settings.database_url.get_secret_value())
        job_signal.start()
        app.state.job_signal = job_signal
        try:
            yield
        finally:
            app.state.mesh_preview.close()
            await job_signal.stop()
            await engine.dispose()

    app = FastAPI(
        title="ml4paleo",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        swagger_ui_oauth2_redirect_url="/api/docs/oauth2-redirect",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.label_pyramid = LabelPyramid(settings.label_cache_mb * 1024 * 1024)
    app.state.mesh_preview = MeshPreviewPool()
    app.add_middleware(CompressionMiddleware)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        request: Request, exc: RequestValidationError
    ) -> Response:
        """
        A 422 that says what is wrong, as FastAPI's own does. Each error
        echoes the input it refused, and a number JSON can't hold (NaN or an
        infinity, which Python's parser takes, as it does a number too big for
        a float, like 1e999) can't be written back: the default answer fails
        to render, and the request gets a 500.
        """
        return JSONResponse(
            status_code=422,
            content={"detail": _writable(jsonable_encoder(exc.errors()))},
        )

    @app.middleware("http")
    async def csrf_protection(request: Request, call_next) -> Response:
        """
        Refuse state-changing API requests from other sites. Browsers send
        `Origin` (and `Sec-Fetch-Site`) on cross-site requests, and a signed-in
        browser must also echo its CSRF token, which other sites can't read.
        """
        path = request.url.path
        if request.method not in SAFE_METHODS and path.startswith("/api/"):
            origin = request.headers.get("origin")
            if (origin is not None and origin != public_origin) or request.headers.get(
                "sec-fetch-site"
            ) == "cross-site":
                return JSONResponse({"detail": "Cross-site request refused."}, 403)
            session_token = request.cookies.get(session_cookie)
            if (
                session_token
                and path not in SESSION_CSRF_EXEMPT
                and not tokens_match(
                    csrf_token(secret_key, session_token),
                    request.headers.get("x-csrf-token"),
                )
            ):
                return JSONResponse({"detail": "Missing or wrong CSRF token."}, 403)
        return await call_next(request)

    @app.middleware("http")
    async def security_headers(request: Request, call_next) -> Response:
        response = await call_next(request)
        policy = (
            NEUROGLANCER_CONTENT_SECURITY_POLICY
            if has_neuroglancer and request.url.path.startswith(NEUROGLANCER_PATH + "/")
            else app_policy
        )
        response.headers.setdefault("Content-Security-Policy", policy)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        if settings.is_https:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
            )
        return response

    # When storage last answered; health checks ask it at most this often,
    # since anyone may call them.
    storage_checked = {"at": -STORAGE_CHECK_SECONDS}

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, str]:
        """
        Check that the database and project storage both answer.
        """
        async with request.app.state.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        if time.monotonic() - storage_checked["at"] >= STORAGE_CHECK_SECONDS:
            await run_in_threadpool(_check_storage, settings)
            storage_checked["at"] = time.monotonic()
        return {"status": "ok", "version": __version__}

    for router in ROUTERS:
        app.include_router(router)

    if has_neuroglancer:
        assert settings.neuroglancer_dir is not None
        app.mount(
            NEUROGLANCER_PATH,
            StaticFiles(directory=settings.neuroglancer_dir, html=True),
            name="neuroglancer",
        )

    @app.get("/{path:path}", include_in_schema=False)
    async def web_app(path: str) -> Response:
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404)
        return _serve_web_app(settings.web_dir, path)

    return app


def _writable(value: Any) -> Any:
    """
    `value` (as `jsonable_encoder` leaves it) with each number JSON can't hold
    written as text instead: "nan", "inf", or "-inf".
    """
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _writable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_writable(item) for item in value]
    return value


def _upload_origin(settings: Settings) -> str | None:
    """
    Where browsers send upload parts, when that's another site than this one.
    """
    endpoint = settings.storage.public_endpoint
    if not endpoint:
        return None
    parts = urlsplit(endpoint)
    origin = f"{parts.scheme}://{parts.netloc}"
    here = urlsplit(settings.public_url)
    return None if origin == f"{here.scheme}://{here.netloc}" else origin


def _check_storage(settings: Settings) -> None:
    """
    List at most one object from project storage, which fails if the bucket
    is missing or the credentials are wrong.
    """
    store = object_store(project_storage(settings))
    for _ in obstore.list(store, chunk_size=1):
        break


def _serve_web_app(web_dir: pathlib.Path | None, path: str) -> Response:
    if web_dir is None or not (web_dir / "index.html").is_file():
        return HTMLResponse(PLACEHOLDER_PAGE)
    root = web_dir.resolve()
    candidate = (root / path).resolve()
    if path and candidate.is_relative_to(root) and candidate.is_file():
        # SvelteKit puts content-hashed files under _app/immutable.
        immutable = candidate.is_relative_to(root / "_app" / "immutable")
        return FileResponse(
            candidate,
            headers={
                "Cache-Control": "public, max-age=31536000, immutable"
                if immutable
                else "no-cache"
            },
        )
    return FileResponse(root / "index.html", headers={"Cache-Control": "no-cache"})
