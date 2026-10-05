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
# Neuroglancer compiles WebAssembly decoders and injects its own styles. It
# still loads data only from this origin (connect-src), so a shared view link
# can't make it send anything elsewhere.
NEUROGLANCER_CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self' 'wasm-unsafe-eval'",
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
