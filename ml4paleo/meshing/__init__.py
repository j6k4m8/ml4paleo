"""
Meshes of segmentations. `blocks` meshes block by block for workers; the v1
web app's `ChunkedMesher` (which needs the `mesh` extra) is imported only
when asked for, so the light parts stay importable without it.
"""

from typing import Any

# Written next to the meshes to record their coordinate system. Meshes without
# this file predate x, y, z vertex order and are stored in (z, y, x) order.
MESH_INFO_FILENAME = "mesh_info.json"


def __getattr__(name: str) -> Any:
    if name in ("ChunkedMesher", "write_obj"):
        from . import chunked

        return getattr(chunked, name)
    raise AttributeError(name)
