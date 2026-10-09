"""
The self-hosted Neuroglancer, served at /neuroglancer/ from
`settings.neuroglancer_dir` (the server image builds it from a pinned
Neuroglancer release), and links that open a project's data in it.
"""

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from ml4paleo.labels import FIRST_CLASS, MAX_CLASS

from .settings import Settings

NEUROGLANCER_PATH = "/neuroglancer"
# Neuroglancer compiles WebAssembly decoders, builds some functions at run
# time (its chunk decoding worker does so as it starts, so it can't draw
# anything without 'unsafe-eval'), and injects its own styles. It still runs
# only scripts from this origin (no inline scripts) and connects only to this
# origin, so a shared view link can't load data from elsewhere or send any
# there. This applies to /neuroglancer/ only; the rest of the site keeps the
# strict policy. The project's Neuroglancer tab frames it, so pages of this
# origin may (and nobody else, as with the rest of the site).
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
        "frame-ancestors 'self'",
    ]
)

# How opaque each kind of layer's segments are, as in the annotator.
LABELS_ALPHA = 0.5
PREDICTION_ALPHA = 0.35
SEGMENTATION_ALPHA = 0.45
# The views start zoomed so the longest side of the image takes this many
# pixels, about what a panel of a laptop's window is high.
FIT_PIXELS = 400

# What OME-Zarr calls a length, in meters: the metric prefixes (before
# "meter"), and the others Neuroglancer reads.
_METRIC = {
    "yotta": 24,
    "zetta": 21,
    "exa": 18,
    "peta": 15,
    "tera": 12,
    "giga": 9,
    "mega": 6,
    "kilo": 3,
    "hecto": 2,
    "deca": 1,
    "": 0,
    "deci": -1,
    "centi": -2,
    "milli": -3,
    "micro": -6,
    "nano": -9,
    "pico": -12,
    "femto": -15,
    "atto": -18,
    "zepto": -21,
    "yocto": -24,
}
_OTHER_LENGTHS = {
    "angstrom": 1e-10,
    "foot": 0.3048,
    "inch": 0.0254,
    "mile": 1609.34,
    "parsec": 3.0856775814913673e16,
    "yard": 0.9144,
}
_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


def neuroglancer_available(settings: Settings) -> bool:
    directory = settings.neuroglancer_dir
    return directory is not None and (directory / "index.html").is_file()


def neuroglancer_link(
    public_url: str,
    zarr_url: str,
    manifest: Mapping[str, Any],
    *,
    labels_url: str | None = None,
    classes: Sequence[tuple[int, str]] = (),
    prediction_url: str | None = None,
    segmentation_url: str | None = None,
) -> str:
    """
    A Neuroglancer view of a project, as a link to /neuroglancer/ with its
    state in the fragment (JSON, percent-encoded). The layers, from the bottom:

    - `image`: the OME-Zarr image at `zarr_url`, with its display window;
    - `labels`: the label zarr group at `labels_url`, if given, which
      Neuroglancer draws as one multiscale volume in the image's coordinates;
    - `prediction` and `final segmentation`: the `class` arrays of the zarr
      groups at `prediction_url` and `segmentation_url`, if given, hidden
      until someone switches them on.

    The last three are segmentation layers that select only the project's
    `classes` (pairs of a value, 2 or more, and a color, `#rrggbb`), so
    background (1) and unlabeled (0) stay transparent. The first of them
    holds the selection and the colors and the others link to it, so they are
    in the link once and every layer draws a class alike. The view starts in
    Neuroglancer's four panels, in the middle of the image, zoomed to fit it.
    """
    # Voxels of the image, as the layers are told to see them: in meters if
    # the image's voxel size is known, else the voxel itself is the unit.
    shape = _shape(manifest)
    size = _voxel_size(manifest)
    scales = size or (1.0, 1.0, 1.0)
    dimensions = {
        axis: [scale, "m"] if size else [1, ""]
        for axis, scale in zip("xyz", scales, strict=True)
    }

    image: dict[str, Any] = {
        "type": "image",
        "name": "image",
        "source": f"zarr3://{public_url}{zarr_url}",
    }
    if window := manifest.get("window"):
        image["shaderControls"] = {"normalized": {"range": window}}
    layers = [image]

    classes = [(v, c) for v, c in sorted(classes) if FIRST_CLASS <= v <= MAX_CLASS]
    shown: list[tuple[str, str | dict[str, Any], float, bool]] = []
    if labels_url is not None:
        # A group of arrays, which Neuroglancer reads as a multiscale volume.
        shown.append(
            ("labels", f"zarr3://{public_url}{labels_url}", LABELS_ALPHA, True)
        )
    for name, group_url, alpha in [
        ("prediction", prediction_url, PREDICTION_ALPHA),
        ("final segmentation", segmentation_url, SEGMENTATION_ALPHA),
    ]:
        if group_url is not None:
            # An array has no voxel size of its own: say it is the image's.
            source: str | dict[str, Any] = f"zarr3://{public_url}{group_url}class"
            if size:
                source = {
                    "url": source,
                    "transform": {
                        "inputDimensions": dimensions,
                        "outputDimensions": dimensions,
                    },
                }
            shown.append((name, source, alpha, False))
    for name, source, alpha, visible in shown:
        layer: dict[str, Any] = {
            "type": "segmentation",
            "name": name,
            "source": source,
            "selectedAlpha": alpha,
        }
        if not visible:
            layer["visible"] = False
        if not classes:
            # With nothing selected, Neuroglancer would draw every segment.
            layer["ignoreNullVisibleSet"] = False
        if name == shown[0][0]:
            layer["segments"] = [value for value, _ in classes]
            colors = {str(v): c.lower() for v, c in classes if _COLOR.match(c)}
            if colors:
                layer["segmentColors"] = colors
        else:
            layer["linkedSegmentationGroup"] = shown[0][0]
        layers.append(layer)

    state: dict[str, Any] = {"dimensions": dimensions}
    if shape:
        longest = max(n * s for n, s in zip(shape, scales, strict=True)) / min(scales)
        state["position"] = [n // 2 if n % 2 == 0 else n / 2 for n in shape]
        state["crossSectionScale"] = round(longest / FIT_PIXELS, 3)
    state["layers"] = layers
    state["layout"] = "4panel"
    return f"{NEUROGLANCER_PATH}/#!" + quote(
        json.dumps(state, separators=(",", ":")), safe="/:,"
    )


def _shape(manifest: Mapping[str, Any]) -> list[int] | None:
    """The image's shape (x, y, z), or None if the manifest doesn't give one."""
    try:
        z, y, x = (int(n) for n in manifest["shape_czyx"][1:])
    except (KeyError, TypeError, ValueError):
        return None
    return [x, y, z] if min(x, y, z) > 0 else None


def _voxel_size(manifest: Mapping[str, Any]) -> tuple[float, float, float] | None:
    """
    The image's voxel size in meters (x, y, z), or None if it isn't known or
    isn't in a length Neuroglancer knows.
    """
    size, unit = manifest.get("voxel_size_zyx"), manifest.get("unit")
    meters = _meters(unit) if isinstance(unit, str) else None
    if meters is None:
        return None
    try:
        z, y, x = (float(v) * meters for v in size or ())
    except (TypeError, ValueError):
        return None
    return (x, y, z) if all(math.isfinite(v) and v > 0 for v in (x, y, z)) else None


def _meters(unit: str) -> float | None:
    """Meters in one `unit`, as OME-Zarr names it ("millimeter"), if it is a length."""
    if unit in _OTHER_LENGTHS:
        return _OTHER_LENGTHS[unit]
    prefix, meter, rest = unit.partition("meter")
    if meter and not rest and prefix in _METRIC:
        return 10.0 ** _METRIC[prefix]
    return None
