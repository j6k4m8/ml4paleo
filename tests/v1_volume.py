"""
A small v1 volume folder, as the v1 app (`webapp/`) wrote it, for import
tests. Jobs:

- `ABC123`: converted (x, y, z) = (40, 30, 20) uint16 with a voxel size, two
  placed annotation samples (one from a multi-slice polygon submit), one too
  old to place, a model whose sidecar (from v1's segment runner) names its
  segmentation, and a newer segmentation no sidecar names (an unfinished run).
- `FEED01`: converted, annotated after segmenting, with no sidecars naming
  segmentations, so nothing counts as finished.
- `DEAD00`: uploaded but never converted.
- `DEAD01`: its conversion failed partway, leaving part of an array.
"""

import json
from pathlib import Path

import numcodecs
import numpy as np
import zarr
from PIL import Image

SHAPE_XYZ = (40, 30, 20)
VOXEL_SIZE_XYZ_MM = (0.25, 0.5, 2.0)
# The samples' cutout: the whole x and y extent (the volume is smaller than
# 512²), centred in the sample, and 11 slices from z = 4.
CUTOUT_ORIGIN_XYZ = (0, 0, 4)
PADDING_XYZ = ((512 - 40) // 2, (512 - 30) // 2, 0)
# Foreground in each placed sample, as volume (y, x) boxes.
FOREGROUND = {"1745400000": (10, 20, 5, 25), "1745400100-z07": (2, 6, 30, 38)}
SEGMENTED_BOX_XYZ = (slice(8, 30), slice(5, 20), slice(2, 12))


def image() -> np.ndarray:
    x, y, z = np.indices(SHAPE_XYZ)
    return (x * 1000 + y * 30 + z).astype(np.uint16)


def segmentation() -> np.ndarray:
    values = np.zeros(SHAPE_XYZ, dtype=np.uint64)
    values[SEGMENTED_BOX_XYZ] = 255
    return values


def _array(path: Path, data: np.ndarray, chunks, attrs=None) -> None:
    array = zarr.create_array(
        str(path),
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        zarr_format=2,
        compressors=numcodecs.Blosc(cname="lz4", clevel=5, shuffle=1),
        chunk_key_encoding={"name": "v2", "separator": "."},
        fill_value=0,
        config={"write_empty_chunks": False},
    )
    array[:] = data
    if attrs:
        array.attrs.update(attrs)


def runner_sidecar(model_id: str, segmented: bool = True) -> dict:
    """
    A model's sidecar as v1's segment runner wrote it: on training, then
    naming the segmentation once segmenting succeeded.
    """
    sidecar = {
        "model_id": model_id,
        "job_id": "ABC123",
        "annotation_count": 2,
        "training_samples": [{"sample_id": "1745400000"}],
        "metrics": {"train_foreground_dice": 0.9},
    }
    if segmented:
        sidecar["segmentation_id"] = f"{model_id}.zarr"
    return sidecar


def _sample(folder: Path, stamp: str, local_z: int | None, foreground, meta=True):
    view = np.zeros((512, 512, 4), dtype=np.uint8)
    view[..., 3] = 255
    Image.fromarray(view).save(folder / f"img{stamp}.png")
    mask = np.zeros((512, 512, 4), dtype=np.uint8)
    y0, y1, x0, x1 = foreground
    py, px = PADDING_XYZ[1], PADDING_XYZ[0]
    mask[py + y0 : py + y1, px + x0 : px + x1, 0] = 255
    mask[py + y0 : py + y1, px + x0 : px + x1, 3] = 255
    # Red in the padding, which isn't part of the volume.
    mask[:5, :5, 0] = 255
    Image.fromarray(mask).save(folder / f"seg{stamp}.png")
    if not meta:
        return
    record = {
        "cutout_origin_xyz": list(CUTOUT_ORIGIN_XYZ),
        "cutout_shape_xyz": [SHAPE_XYZ[0], SHAPE_XYZ[1], 11],
        "requested_shape_xyz": [512, 512, 11],
        "padding_before_xyz": list(PADDING_XYZ),
        "padding_after_xyz": [
            512 - SHAPE_XYZ[0] - PADDING_XYZ[0],
            512 - SHAPE_XYZ[1] - PADDING_XYZ[1],
            0,
        ],
        "intensity_window": {
            "source_min": 0,
            "source_max": 65535,
            "window_min": 0,
            "window_max": 4000,
        },
        "timestamp": stamp,
        "job_id": "ABC123",
        "annotation_source": "polygon_v2",
    }
    if local_z is not None:
        record["annotated_local_z_index"] = local_z
        record["annotated_global_z_index"] = CUTOUT_ORIGIN_XYZ[2] + local_z
    (folder / f"meta{stamp}.json").write_text(json.dumps(record))


def make(root: Path) -> Path:
    def record(job_id, status, name):
        return {
            "status": f"JobStatus.{status}",
            "name": name,
            "id": job_id,
            "source_type": "dicom",
            "created_at": "2024-07-05T16:24:13.446237",
            "last_updated_at": "2024-07-05T16:35:11.017184",
            "current_job_progress": 0.97,
            "shape": list(SHAPE_XYZ),
        }

    jobs = {
        "ABC123": record("ABC123", "MESHED", "Burrow"),
        "FEED01": record("FEED01", "ANNOTATED", ""),
        "DEAD00": record("DEAD00", "UPLOADED", "Never converted"),
        "DEAD01": record("DEAD01", "CONVERT_ERROR", "Failed conversion"),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "jobs.json").write_text(json.dumps(jobs, indent=4))
    attrs = {"voxel_size_xyz_mm": list(VOXEL_SIZE_XYZ_MM)}
    for job_id in ("ABC123", "FEED01"):
        _array(root / "chunks" / job_id, image(), (16, 16, 8), attrs)
    # Its conversion failed after writing the first slab.
    partial = np.zeros(SHAPE_XYZ, dtype=np.uint16)
    partial[:, :, :8] = image()[:, :, :8]
    _array(root / "chunks" / "DEAD01", partial, (16, 16, 8))

    training = root / "training" / "ABC123"
    training.mkdir(parents=True)
    # The legacy page always labeled the middle slice (local z 5, z 9).
    _sample(training, "1745400000", None, FOREGROUND["1745400000"])
    # A polygon submit of several slices; this file is local z 7 (z 11).
    _sample(training, "1745400100-z07", 7, FOREGROUND["1745400100-z07"])
    # From before samples recorded where they came from.
    _sample(training, "1676000000", None, (0, 4, 0, 4), meta=False)
    # A mask without its image, which v1 ignored.
    Image.fromarray(np.zeros((512, 512, 4), dtype=np.uint8)).save(
        training / "seg1745400200.png"
    )

    models = root / "models" / "ABC123"
    models.mkdir(parents=True)
    (models / "1745400050.json").write_text(
        json.dumps({"rf_kwargs": {}, "model_class": "RandomForest3DSegmenter"})
    )
    (models / "1745400150.json").write_text(json.dumps(runner_sidecar("1745400150")))
    segmented = root / "segmented" / "ABC123"
    _array(segmented / "1745400150.zarr", segmentation(), (16, 16, 16))
    _array(
        segmented / "1745400300.zarr",
        np.zeros(SHAPE_XYZ, dtype=np.uint64),
        (16, 16, 16),
    )
    _array(
        root / "segmented" / "FEED01" / "1745400150.zarr", segmentation(), (16, 16, 16)
    )
    return root
