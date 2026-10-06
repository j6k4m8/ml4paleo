"""
The self-hosted Neuroglancer, served at /neuroglancer/ from
`settings.neuroglancer_dir` (the server image builds it from a pinned
Neuroglancer release), and links that open an image in it.
"""

import json
from typing import Any
from urllib.parse import quote

from .settings import Settings

NEUROGLANCER_PATH = "/neuroglancer"
# Neuroglancer compiles WebAssembly decoders, builds some functions at run
# time (its chunk decoding worker does so as it starts, so it can't draw
# anything without 'unsafe-eval'), and injects its own styles. It still runs
# only scripts from this origin (no inline scripts) and connects only to this
# origin, so a shared view link can't load data from elsewhere or send any
# there. This applies to /neuroglancer/ only; the rest of the site keeps the
# strict policy.
NEUROGLANCER_CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self' 'unsafe-eval' 'wasm-unsafe-eval'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' blob: data:",
        "worker-src 'self' blob:",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ]
)


def neuroglancer_available(settings: Settings) -> bool:
    directory = settings.neuroglancer_dir
    return directory is not None and (directory / "index.html").is_file()


def neuroglancer_link(public_url: str, zarr_url: str, manifest: dict) -> str:
    """
    A Neuroglancer view of an OME-Zarr image, with its display window.
    """
    layer: dict[str, Any] = {
        "type": "image",
        "name": "image",
        "source": f"zarr3://{public_url}{zarr_url}",
    }
    if window := manifest.get("window"):
        layer["shaderControls"] = {"normalized": {"range": window}}
    state = {"layers": [layer], "layout": "4panel"}
    return f"{NEUROGLANCER_PATH}/#!" + quote(json.dumps(state, separators=(",", ":")))
