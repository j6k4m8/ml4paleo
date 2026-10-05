"""
The FastAPI application.

The API lives under `/api`. Every other GET serves the built web app (a
single-page app): existing files are served as-is and any other path gets
`index.html`, so client-side routes work on reload.
"""

import contextlib
import pathlib
from collections.abc import AsyncIterator

import obstore
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool

from ml4paleo.storage import object_store

from . import __version__
from .db import create_engine, create_sessionmaker
from .settings import Settings
from .storage import project_storage

CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "img-src 'self' blob: data:",
        "worker-src 'self' blob:",
        "connect-src 'self'",
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

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(settings.database_url.get_secret_value())
        app.state.engine = engine
        app.state.sessionmaker = create_sessionmaker(engine)
        try:
            yield
        finally:
            await engine.dispose()

    app = FastAPI(
        title="ml4paleo",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings

    @app.middleware("http")
    async def security_headers(request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        if settings.is_https:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
            )
        return response

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, str]:
        """
        Check that the database and project storage both answer.
        """
        async with request.app.state.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        await run_in_threadpool(_check_storage, settings)
        return {"status": "ok", "version": __version__}

    @app.get("/{path:path}", include_in_schema=False)
    async def web_app(path: str) -> Response:
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404)
        return _serve_web_app(settings.web_dir, path)

    return app


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
