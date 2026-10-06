"""
Reading ml4paleo v1's volume folder, to import its jobs.

v1 (the Flask app in `webapp/`) kept everything under one folder:

    jobs.json                         {job id: record}
    chunks/<JOB>/                     the image: a zarr v2 array, (x, y, z)
    training/<JOB>/img<ts>.png        annotation samples: an 8-bit view,
                   seg<ts>.png        its mask (red above 0 is foreground),
                   meta<ts>.json      and where it came from (April 2026 on)
    models/<JOB>/<ts>.model, .json    random forests and their sidecars
    segmented/<JOB>/<ts>.zarr/        segmentations: zarr v2, (x, y, z), 0 for
                                      background and anything else (usually
                                      255) for foreground

Job ids are six uppercase hex digits. Each annotation sample is a fully
labeled 512² XY slice of a random cutout: red above 0 is foreground (v1
trained on any red) and everything else background. Only samples with
metadata can be placed in the volume; older ones are skipped. PNG rows are
y and columns x.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np

JOB_ID = re.compile(r"[0-9A-F]{6}")
JOBS_FILE = "jobs.json"
# A segmentation's folder: the time its model was trained.
SEGMENTATION_NAME = re.compile(r"\d+\.zarr")
# Sample metadata past this isn't a position in any scan.
MAX_COORDINATE = 2**31
# Statuses of jobs whose upload never became an image (as v1's job page
# decides whether it can be annotated); v1 never used "pending".
UNCONVERTED = {"pending", "uploading", "uploaded", "converting", "convert_error"}
# Statuses that mean the newest segmentation finished. v1 also sets
# "annotated" whenever someone annotates, even after segmenting, so this is
# only a fallback for jobs without sidecars from v1's segment runner (older
# ones, or ones migrated with v1's button) that name their output.
SEGMENTED = {"segmented", "meshing_queued", "meshing", "meshed", "mesh_error"}


def normalize_job_id(text: str) -> str | None:
    """The job id in `text` (any case), or None if it isn't one."""
    job_id = text.strip().upper()
    return job_id if JOB_ID.fullmatch(job_id) else None


def read_jobs(root: Path) -> dict[str, dict[str, Any]]:
    """Every job record in `jobs.json`, by id."""
    try:
        raw = json.loads((root / JOBS_FILE).read_text())
    except FileNotFoundError:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{JOBS_FILE} isn't a JSON object")
    return {
        key: record
        for key, record in raw.items()
        if JOB_ID.fullmatch(key) and isinstance(record, dict)
    }


def status(record: dict[str, Any]) -> str:
    """A job's status, such as "meshed" (stored as "JobStatus.MESHED")."""
    return str(record.get("status", "")).split(".")[-1].lower()


def image_path(root: Path, job_id: str) -> Path:
    return root / "chunks" / job_id


def _is_array(path: Path) -> bool:
    return (path / ".zarray").is_file()


def _is_segmentation(name: Any, folder: Path) -> TypeGuard[str]:
    return (
        isinstance(name, str)
        and SEGMENTATION_NAME.fullmatch(name) is not None
        and _is_array(folder / name)
    )


def _stamp(name: str) -> int:
    digits = name.split(".")[0].split("-")[0]
    return int(digits) if digits.isdigit() else -1


def _from_segment_runner(meta: dict[str, Any]) -> bool:
    """
    Whether v1's segment runner wrote a model's sidecar: it names the run's
    segmentation only once segmenting succeeds. v1's "migrate metadata"
    button names one whenever its folder exists, finished or not.
    """
    return "legacy_metadata_migrated_at" not in meta and (
        "training_samples" in meta or "metrics" in meta
    )


def segmentation(root: Path, job_id: str, record: dict[str, Any]) -> str | None:
    """
    The name of the newest segmentation known to be complete: one a sidecar
    from v1's segment runner names, else the newest one if the job's status
    says segmenting finished. None if there's none.
    """
    folder = root / "segmented" / job_id
    named = []
    for sidecar in (root / "models" / job_id).glob("*.json"):
        try:
            meta = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue
        if not (isinstance(meta, dict) and _from_segment_runner(meta)):
            continue
        name = meta.get("segmentation_id")
        if _is_segmentation(name, folder):
            named.append(name)
    if named:
        return max(named, key=_stamp)
    found = [p.name for p in folder.glob("*.zarr") if _is_segmentation(p.name, folder)]
    if found and status(record) in SEGMENTED:
        return max(found, key=_stamp)
    return None


@dataclass(frozen=True)
class Annotation:
    """One placed annotation sample: a fully labeled part of an XY slice."""

    stamp: str
    mask_path: Path
    z: int
    # The part of the 512² sample inside the volume (rows are y, columns
    # x), and where its first row and column land.
    rows: tuple[int, int]
    cols: tuple[int, int]
    y0: int
    x0: int

    @property
    def box_zyx(self) -> list[int]:
        """Where it lands, (z0, y0, x0, z1, y1, x1), half-open."""
        height = self.rows[1] - self.rows[0]
        width = self.cols[1] - self.cols[0]
        return [self.z, self.y0, self.x0, self.z + 1, self.y0 + height, self.x0 + width]

    def foreground(self) -> np.ndarray:
        """Which voxels of the part are foreground (y, x)."""
        from PIL import Image

        with Image.open(self.mask_path) as image:
            red = np.asarray(image.convert("RGBA"))[..., 0]
        part = red[self.rows[0] : self.rows[1], self.cols[0] : self.cols[1]]
        if part.shape != (self.rows[1] - self.rows[0], self.cols[1] - self.cols[0]):
            raise ValueError(f"{self.mask_path.name} is smaller than its metadata says")
        return part > 0


def annotations(
    root: Path, job_id: str, shape_xyz: tuple[int, int, int]
) -> tuple[list[Annotation], int]:
    """
    The job's annotation samples that can be placed in a volume of
    `shape_xyz`, oldest first, and how many couldn't be (no metadata, or
    metadata that doesn't fit the volume).
    """
    folder = root / "training" / job_id
    if not folder.is_dir():
        return [], 0
    placed: list[Annotation] = []
    skipped = 0
    for image in sorted(folder.glob("img*.png")):
        stamp = image.stem.removeprefix("img")
        mask = folder / f"seg{stamp}.png"
        # v1 ignores samples whose mask is missing, and so does this.
        if not mask.is_file():
            continue
        annotation = _place(folder / f"meta{stamp}.json", mask, stamp, shape_xyz)
        if annotation is None:
            skipped += 1
        else:
            placed.append(annotation)
    placed.sort(key=lambda a: (_stamp(a.stamp), a.stamp))
    return placed, skipped


def _numbers(values: Any, count: int = 3) -> list[int]:
    """
    Whole numbers from sample metadata, read with int() as v1 read them.
    Anything else, or anything infinite or too large to be a position in a
    scan, raises.
    """
    numbers = [int(v) for v in values]
    if len(numbers) != count or any(abs(n) >= MAX_COORDINATE for n in numbers):
        raise ValueError("not a position in a scan")
    return numbers


def _place(
    meta_path: Path, mask: Path, stamp: str, shape_xyz: tuple[int, int, int]
) -> Annotation | None:
    try:
        meta = json.loads(meta_path.read_text())
        origin = _numbers(meta["cutout_origin_xyz"])
        cutout = _numbers(meta["cutout_shape_xyz"])
        padding = _numbers(meta.get("padding_before_xyz", [0, 0, 0]))
        requested = _numbers(meta.get("requested_shape_xyz", [0, 0, 1]))
        # As v1's `annotation_sample_metadata_for_z` does: the labeled slice
        # is the sample's local z (its middle one by default), which is
        # padding if it falls outside the cutout.
        depth = max(1, requested[2])
        [local] = _numbers([meta.get("annotated_local_z_index", depth // 2)], 1)
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError):
        # int() raises OverflowError for an infinity (JSON's 1e400).
        return None
    local = min(max(local, 0), depth - 1)
    if not 0 <= local - padding[2] < cutout[2]:
        return None
    z = origin[2] + local - padding[2]
    # Mask pixel (row r, column c) is voxel (x, y) = (origin - padding + (c,
    # r)) for c - padding_x in [0, cutout_x), r - padding_y in [0, cutout_y);
    # the rest is padding. Clip to the volume too, in case of odd metadata.
    first_x = max(origin[0], 0)
    first_y = max(origin[1], 0)
    last_x = min(origin[0] + cutout[0], shape_xyz[0])
    last_y = min(origin[1] + cutout[1], shape_xyz[1])
    if not (0 <= z < shape_xyz[2] and first_x < last_x and first_y < last_y):
        return None
    cols = (padding[0] + first_x - origin[0], padding[0] + last_x - origin[0])
    rows = (padding[1] + first_y - origin[1], padding[1] + last_y - origin[1])
    return Annotation(
        stamp=stamp, mask_path=mask, z=z, rows=rows, cols=cols, y0=first_y, x0=first_x
    )
