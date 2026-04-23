"""
This file stores utilities and helper methods that are commonly used by other
files in the web application directory. Methods SHOULD go here if they are
used by more than one file in the web application directory. Methods SHOULD
NOT go here if they are only used by one file in the web application directory
or if they would be useful to other parts of the application (and should
therefore live in the ml4paleo package).

"""

import datetime
import json
import math
import pathlib
from html import escape
from typing import Any, Optional, Union

import numpy as np
from flask import request
from config import CONFIG
from job import UploadJob
from PIL import Image
from ml4paleo.volume_providers import ZarrVolumeProvider

MODEL_METRIC_SECTION_KEYS = (
    "metric",
    "metrics",
    "scores",
    "evaluation",
    "validation",
    "validation_metrics",
    "val_metrics",
    "results",
    "summary",
)
MODEL_METRIC_KEYWORDS = (
    "metric",
    "dice",
    "iou",
    "jaccard",
    "f1",
    "score",
    "accuracy",
    "precision",
    "recall",
    "loss",
)


def _job_id(job_or_id: Union[UploadJob, str]) -> str:
    """
    Normalize an UploadJob or raw job ID to the job ID string.
    """
    return job_or_id.id if isinstance(job_or_id, UploadJob) else str(job_or_id)


def _model_directory(job_or_id: Union[UploadJob, str]) -> pathlib.Path:
    """
    Return the directory that stores trained models for a job.
    """
    return pathlib.Path(CONFIG.model_directory) / _job_id(job_or_id)


def _segmentation_directory(job_or_id: Union[UploadJob, str]) -> pathlib.Path:
    """
    Return the directory that stores segmentation outputs for a job.
    """
    return pathlib.Path(CONFIG.segmented_directory) / _job_id(job_or_id)


def _mesh_directory(job_or_id: Union[UploadJob, str]) -> pathlib.Path:
    """
    Return the directory that stores mesh outputs for a job.
    """
    return pathlib.Path(CONFIG.meshed_directory) / _job_id(job_or_id)


def _annotation_directory(job_or_id: Union[UploadJob, str]) -> pathlib.Path:
    """
    Return the directory that stores annotation files for a job.
    """
    return pathlib.Path(CONFIG.training_directory) / _job_id(job_or_id)


def get_annotation_pairs(
    job_or_id: Union[UploadJob, str],
) -> list[tuple[pathlib.Path, pathlib.Path, Optional[pathlib.Path]]]:
    """
    Return saved annotation image/mask pairs, plus optional metadata files.
    """
    annotation_dir = _annotation_directory(job_or_id)
    if not annotation_dir.exists():
        return []

    pairs = []
    for img_path in sorted(annotation_dir.glob(f"{CONFIG.training_img_prefix}*.png")):
        timestamp = img_path.stem.removeprefix(CONFIG.training_img_prefix)
        seg_path = annotation_dir / f"{CONFIG.training_seg_prefix}{timestamp}.png"
        if not seg_path.exists():
            continue
        meta_path = annotation_dir / f"{CONFIG.training_meta_prefix}{timestamp}.json"
        pairs.append((img_path, seg_path, meta_path if meta_path.exists() else None))
    return pairs


def count_annotation_samples(job_or_id: Union[UploadJob, str]) -> int:
    """
    Count complete annotation samples for a job.
    """
    return len(get_annotation_pairs(job_or_id))


def load_annotation_sample_metadata(
    meta_path: Optional[pathlib.Path],
) -> Optional[dict[str, Any]]:
    """
    Load one saved annotation metadata JSON file if it exists.
    """
    if meta_path is None or not meta_path.exists():
        return None

    with meta_path.open("r") as f:
        metadata = json.load(f)

    return metadata if isinstance(metadata, dict) else None


def load_annotation_source_slice(
    job_or_id: Union[UploadJob, str],
    sample_metadata: dict[str, Any],
) -> np.ndarray:
    """
    Reconstruct the raw annotated XY slice from saved cutout metadata.

    The returned slice uses the native volume XY orientation used by the batch
    segmenter. Browser-displayed annotation PNGs are the transpose of this.
    """
    requested_x, requested_y, _requested_z = tuple(
        int(v) for v in sample_metadata["requested_shape_xyz"]
    )
    actual_x, actual_y, actual_z = tuple(
        int(v) for v in sample_metadata["cutout_shape_xyz"]
    )
    x_start, y_start, z_start = tuple(
        int(v) for v in sample_metadata["cutout_origin_xyz"]
    )
    x_pad_before, y_pad_before, z_pad_before = tuple(
        int(v) for v in sample_metadata["padding_before_xyz"]
    )
    annotated_local_z_index = int(sample_metadata["annotated_local_z_index"])
    source_local_z_index = annotated_local_z_index - z_pad_before

    job_id = job_or_id.id if isinstance(job_or_id, UploadJob) else str(job_or_id)
    volume_provider = ZarrVolumeProvider(pathlib.Path(CONFIG.chunked_directory) / job_id)
    source_slice = np.zeros((requested_x, requested_y), dtype=volume_provider.dtype)

    if 0 <= source_local_z_index < actual_z:
        raw_slice = np.asarray(
            volume_provider[
                x_start : x_start + actual_x,
                y_start : y_start + actual_y,
                z_start + source_local_z_index,
            ]
        )
        source_slice[
            x_pad_before : x_pad_before + actual_x,
            y_pad_before : y_pad_before + actual_y,
        ] = raw_slice

    return source_slice


def normalize_model_id(model_id: str) -> str:
    """
    Normalize model or artifact filenames to the shared model ID stem.
    """
    normalized = pathlib.Path(str(model_id)).name
    for suffix in (".model", ".json", ".zarr"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def model_id_to_segmentation_id(model_id: str) -> str:
    """
    Map a model ID to the corresponding segmentation/mesh artifact ID.
    """
    return f"{normalize_model_id(model_id)}.zarr"


def _model_sort_key(model_id: str) -> tuple[int, Union[int, str], str]:
    """
    Sort numeric timestamp IDs chronologically and fall back to lexical order.
    """
    normalized = normalize_model_id(model_id)
    if normalized.isdigit():
        return (0, int(normalized), normalized)
    return (1, normalized, normalized)


def _coerce_metric_value(value: Any) -> Optional[float]:
    """
    Convert scalar metadata values into finite floats when possible.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        value_f = float(value)
        return value_f if math.isfinite(value_f) else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("%"):
            stripped = stripped[:-1]
        try:
            value_f = float(stripped)
        except ValueError:
            return None
        return value_f if math.isfinite(value_f) else None
    return None


def _normalize_metric_key(metric_key: Optional[str]) -> Optional[str]:
    """
    Normalize metric keys for dict access and comparisons.
    """
    if metric_key is None:
        return None
    normalized = str(metric_key).strip().lower().replace(" ", "_")
    return normalized or None


def _display_metric_label(metric_key: str) -> str:
    """
    Convert a metric key into a short human-readable label.
    """
    return str(metric_key).replace("_", " ").strip().title()


def _looks_like_metric_key(metric_key: str) -> bool:
    """
    Heuristic to keep numeric hyperparameters out of the metrics dict.
    """
    normalized = _normalize_metric_key(metric_key) or ""
    return any(keyword in normalized for keyword in MODEL_METRIC_KEYWORDS)


def _extract_metrics_from_mapping(mapping: dict[str, Any]) -> dict[str, float]:
    """
    Extract metric-like numeric values from a metadata mapping.
    """
    metrics: dict[str, float] = {}
    for name_key, value_key in (
        ("best_metric_name", "best_metric_value"),
        ("metric_name", "metric_value"),
        ("score_name", "score_value"),
        ("name", "value"),
    ):
        metric_name = mapping.get(name_key)
        metric_value = _coerce_metric_value(mapping.get(value_key))
        normalized_metric_name = (
            _normalize_metric_key(metric_name) if isinstance(metric_name, str) else None
        )
        if normalized_metric_name and metric_value is not None:
            metrics[normalized_metric_name] = metric_value

    for key, value in mapping.items():
        normalized_key = _normalize_metric_key(key)
        metric_value = _coerce_metric_value(value)
        if normalized_key and metric_value is not None and _looks_like_metric_key(normalized_key):
            metrics[normalized_key] = metric_value

    for key, value in mapping.items():
        normalized_key = _normalize_metric_key(key)
        if isinstance(value, dict) and (
            normalized_key in MODEL_METRIC_SECTION_KEYS
            or (normalized_key is not None and _looks_like_metric_key(normalized_key))
        ):
            metrics.update(_extract_metrics_from_mapping(value))

    return metrics


def load_model_metadata_sidecar(
    job_or_id: Union[UploadJob, str],
    model_id: str,
) -> Optional[dict[str, Any]]:
    """
    Load a model's JSON metadata sidecar if present and valid.
    """
    metadata_path = _model_directory(job_or_id) / f"{normalize_model_id(model_id)}.json"
    if not metadata_path.exists():
        return None
    try:
        with metadata_path.open("r") as f:
            payload = json.load(f)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _model_metadata_needs_migration(metadata: Optional[dict[str, Any]]) -> bool:
    """
    Return True when a model sidecar is missing core inferred fields.
    """
    if not isinstance(metadata, dict):
        return True
    if not isinstance(metadata.get("model_id"), str) or not metadata.get("model_id"):
        return True
    if not isinstance(metadata.get("job_id"), str) or not metadata.get("job_id"):
        return True
    if not isinstance(metadata.get("created_at"), str) or not metadata.get("created_at"):
        return True
    if not isinstance(metadata.get("annotation_count"), int):
        return True
    return False


def _annotation_timestamp(annotation_img_path: pathlib.Path) -> Optional[int]:
    """
    Parse the timestamp stem from an annotation image filename.
    """
    timestamp = annotation_img_path.stem.removeprefix(CONFIG.training_img_prefix)
    return int(timestamp) if timestamp.isdigit() else None


def _annotation_count_for_model(job_or_id: Union[UploadJob, str], model_id: str) -> int:
    """
    Count annotation samples that would have existed when a timestamped model was trained.
    """
    normalized_model_id = normalize_model_id(model_id)
    if not normalized_model_id.isdigit():
        return count_annotation_samples(job_or_id)

    cutoff = int(normalized_model_id)
    count = 0
    for img_path, _seg_path, _meta_path in get_annotation_pairs(job_or_id):
        annotation_timestamp = _annotation_timestamp(img_path)
        if annotation_timestamp is not None and annotation_timestamp <= cutoff:
            count += 1
    return count


def _infer_model_created_at(model_id: str, model_path: pathlib.Path) -> Optional[str]:
    """
    Infer a creation time for a model from its timestamped ID or file metadata.
    """
    normalized_model_id = normalize_model_id(model_id)
    if normalized_model_id.isdigit():
        try:
            return datetime.datetime.fromtimestamp(int(normalized_model_id)).isoformat()
        except (OverflowError, OSError, ValueError):
            pass
    if model_path.exists():
        try:
            return datetime.datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
        except OSError:
            return None
    return None


def _artifact_status(segmentation_ready: bool, mesh_ready: bool) -> str:
    """
    Return a short status string describing derived artifact availability.
    """
    if mesh_ready:
        return "Model, segmentation, and mesh ready"
    if segmentation_ready:
        return "Model and segmentation ready"
    return "Model ready; derived artifacts unavailable"


def migrate_model_metadata_sidecar(
    job_or_id: Union[UploadJob, str],
    model_id: str,
) -> pathlib.Path:
    """
    Backfill one legacy model sidecar with inferred metadata.
    """
    normalized_model_id = normalize_model_id(model_id)
    model_dir = _model_directory(job_or_id)
    model_path = model_dir / f"{normalized_model_id}.model"
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    metadata_path = model_dir / f"{normalized_model_id}.json"
    existing = load_model_metadata_sidecar(job_or_id, normalized_model_id)
    metadata: dict[str, Any] = existing.copy() if isinstance(existing, dict) else {}
    created_at = _infer_model_created_at(normalized_model_id, model_path)
    annotation_count = _annotation_count_for_model(job_or_id, normalized_model_id)
    segmentation_id = model_id_to_segmentation_id(normalized_model_id)
    segmentation_path = _segmentation_directory(job_or_id) / segmentation_id

    metadata["model_id"] = normalized_model_id
    metadata["job_id"] = _job_id(job_or_id)
    if created_at:
        metadata["created_at"] = created_at
    metadata["annotation_count"] = annotation_count
    if not isinstance(metadata.get("model_class"), str) or not metadata.get("model_class"):
        metadata["model_class"] = "RandomForest3DSegmenter"
    if not isinstance(metadata.get("rf_kwargs"), dict):
        metadata["rf_kwargs"] = {}
    if segmentation_path.is_dir():
        metadata["segmentation_id"] = segmentation_id
    metadata["legacy_metadata_migrated_at"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return metadata_path


def _training_curve_status(metrics: dict[str, float], metric_key: str) -> str:
    """
    Return a short status string for training/metric availability.
    """
    if metric_key in metrics:
        return f"{_display_metric_label(metric_key)} available"
    if len(metrics) > 0:
        return f"{_display_metric_label(metric_key)} unavailable for this model"
    return "Metric unavailable (legacy model)"


def get_model_runs(
    job_or_id: Union[UploadJob, str],
    train_metric_key: str = "train_foreground_dice",
) -> list[dict[str, Any]]:
    """
    Return saved model runs for a job, newest first.

    Each returned dict is intended to be directly consumable by Flask/Jinja.
    """
    model_dir = _model_directory(job_or_id)
    if not model_dir.exists():
        return []

    model_ids = sorted(
        {
            normalize_model_id(model_path.name)
            for model_path in model_dir.glob("*.model")
            if model_path.is_file()
        },
        key=_model_sort_key,
        reverse=True,
    )

    normalized_metric_key = _normalize_metric_key(train_metric_key) or "train_foreground_dice"
    runs: list[dict[str, Any]] = []
    for model_id in model_ids:
        metadata_path = model_dir / f"{model_id}.json"
        model_path = model_dir / f"{model_id}.model"
        metadata = load_model_metadata_sidecar(job_or_id, model_id)
        metrics = _extract_metrics_from_mapping(metadata or {})
        needs_migration = _model_metadata_needs_migration(metadata)
        has_training_metrics = len(metrics) > 0
        annotation_count = None
        metadata_backed_sample_count = None
        training_samples = []
        model_class = None
        rf_kwargs: dict[str, Any] = {}
        if isinstance(metadata, dict):
            annotation_count_value = metadata.get("annotation_count")
            if isinstance(annotation_count_value, int):
                annotation_count = annotation_count_value
            metadata_backed_value = metadata.get("metadata_backed_sample_count")
            if isinstance(metadata_backed_value, int):
                metadata_backed_sample_count = metadata_backed_value
            model_class_value = metadata.get("model_class")
            if isinstance(model_class_value, str) and model_class_value.strip():
                model_class = model_class_value
            rf_kwargs_value = metadata.get("rf_kwargs")
            if isinstance(rf_kwargs_value, dict):
                rf_kwargs = rf_kwargs_value
            training_samples_value = metadata.get("training_samples")
            if isinstance(training_samples_value, list):
                training_samples = training_samples_value

        if annotation_count is None:
            annotation_count = _annotation_count_for_model(job_or_id, model_id)

        segmentation_id = model_id_to_segmentation_id(model_id)
        segmentation_path = _segmentation_directory(job_or_id) / segmentation_id
        mesh_path = _mesh_directory(job_or_id) / segmentation_id
        mesh_stl_files = sorted(
            path for path in mesh_path.glob("*.combined.stl") if path.is_file()
        ) if mesh_path.exists() else []
        mesh_obj_files = sorted(
            path for path in mesh_path.glob("*.combined.obj") if path.is_file()
        ) if mesh_path.exists() else []
        segmentation_ready = segmentation_path.is_dir()
        mesh_ready = len(mesh_stl_files) > 0 or len(mesh_obj_files) > 0
        train_metric_value = metrics.get(normalized_metric_key)
        legacy_summary = None
        if needs_migration:
            legacy_summary = (
                "This older run only has minimal metadata. "
                "Migrate it to save inferred timestamps and annotation counts."
            )
        elif not has_training_metrics:
            legacy_summary = (
                "This older run predates stored training metrics, so it is omitted "
                "from the chart."
            )

        runs.append(
            {
                "model_id": model_id,
                "created_at": _infer_model_created_at(model_id, model_path),
                "annotation_count": annotation_count,
                "metadata_backed_sample_count": metadata_backed_sample_count,
                "training_samples_count": len(training_samples) if len(training_samples) > 0 else None,
                "model_class": model_class or "Unknown model",
                "rf_kwargs": rf_kwargs,
                "metrics": metrics,
                "train_metric_key": normalized_metric_key,
                "train_metric_value": train_metric_value,
                "train_dice_value": metrics.get("train_foreground_dice"),
                "train_loss_value": metrics.get("train_loss"),
                "train_iou_value": metrics.get("train_foreground_iou"),
                "metric_ready": train_metric_value is not None,
                "has_training_metrics": has_training_metrics,
                "needs_migration": needs_migration,
                "is_legacy": needs_migration or not has_training_metrics,
                "has_rf_params": len(rf_kwargs) > 0,
                "legacy_summary": legacy_summary,
                "segmentation_id": segmentation_id,
                "segmentation_path": segmentation_path,
                "segmentation_ready": segmentation_ready,
                "mesh_id": segmentation_id,
                "mesh_path": mesh_path,
                "mesh_ready": mesh_ready,
                "model_path": model_path,
                "metadata_path": metadata_path if metadata_path.exists() else None,
                "metadata": metadata,
                "artifact_status": _artifact_status(segmentation_ready, mesh_ready),
                "training_curve_status": _training_curve_status(
                    metrics, normalized_metric_key
                ),
                "status_text": (
                    f"{_artifact_status(segmentation_ready, mesh_ready)}; "
                    f"{_training_curve_status(metrics, normalized_metric_key)}"
                ),
            }
        )
    return runs


def _empty_metric_chart_svg(
    message: str,
    width: int,
    height: int,
    subtitle: Optional[str] = None,
) -> str:
    """
    Render a simple placeholder SVG when no metric chart can be drawn.
    """
    subtitle_markup = ""
    if subtitle:
        subtitle_markup = (
            f'<text x="{width / 2:.1f}" y="{height / 2 + 18:.1f}" '
            'text-anchor="middle" font-family="system-ui, sans-serif" '
            'font-size="11" fill="#64748b">'
            f"{escape(subtitle)}</text>"
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(message)}">'
        f'<rect width="{width}" height="{height}" rx="12" fill="#f8fafc" stroke="#cbd5e1"/>'
        f'<text x="{width / 2:.1f}" y="{height / 2:.1f}" text-anchor="middle" '
        'font-family="system-ui, sans-serif" font-size="13" fill="#0f172a">'
        f"{escape(message)}</text>"
        f"{subtitle_markup}"
        "</svg>"
    )


def build_model_metric_chart_svg(
    models: list[dict[str, Any]],
    metric_key: str = "train_foreground_dice",
    metric_label: str = "Train Dice",
    width: int = 640,
    height: int = 240,
) -> str:
    """
    Build a simple SVG chart for one metric across models.
    """
    width = max(int(width), 240)
    height = max(int(height), 160)
    if len(models) == 0:
        return _empty_metric_chart_svg("No model runs available", width, height)

    normalized_metric_key = _normalize_metric_key(metric_key) or "train_foreground_dice"
    ordered_models = sorted(
        models, key=lambda model: _model_sort_key(str(model.get("model_id", "")))
    )
    points = []
    for idx, model in enumerate(ordered_models):
        metrics = model.get("metrics", {})
        metric_value = None
        if isinstance(metrics, dict):
            metric_value = _coerce_metric_value(metrics.get(normalized_metric_key))
        if metric_value is None:
            metric_value = _coerce_metric_value(model.get("train_metric_value"))
            model_metric_key = _normalize_metric_key(model.get("train_metric_key"))
            if model_metric_key != normalized_metric_key:
                metric_value = None
        if metric_value is not None:
            points.append((idx, metric_value, str(model.get("model_id", ""))))

    if len(points) == 0:
        return _empty_metric_chart_svg(
            "Metric unavailable",
            width,
            height,
            subtitle="Legacy models do not expose this metric.",
        )

    values = [value for _, value, _ in points]
    value_min = min(values)
    value_max = max(values)
    padding = (value_max - value_min) * 0.1 if value_max != value_min else (abs(value_max) * 0.1 or 1.0)
    plot_min = value_min - padding
    plot_max = value_max + padding

    margin_left = 56
    margin_right = 20
    margin_top = 24
    margin_bottom = 44
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def x_pos(index: int) -> float:
        if len(ordered_models) == 1:
            return margin_left + (plot_width / 2)
        return margin_left + (plot_width * index / (len(ordered_models) - 1))

    def y_pos(value: float) -> float:
        if plot_max == plot_min:
            return margin_top + (plot_height / 2)
        normalized = (value - plot_min) / (plot_max - plot_min)
        return margin_top + plot_height - (normalized * plot_height)

    grid_values = [plot_min, (plot_min + plot_max) / 2, plot_max]
    grid_markup = []
    for grid_value in grid_values:
        y = y_pos(grid_value)
        grid_markup.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        grid_markup.append(
            f'<text x="{margin_left - 8}" y="{y + 4:.2f}" text-anchor="end" '
            'font-family="system-ui, sans-serif" font-size="10" fill="#64748b">'
            f"{grid_value:.4g}</text>"
        )

    tick_indices = list(range(len(ordered_models)))
    if len(tick_indices) > 6:
        tick_indices = sorted(
            {
                0,
                len(ordered_models) // 3,
                (2 * len(ordered_models)) // 3,
                len(ordered_models) - 1,
            }
        )

    x_axis_markup = []
    for idx in tick_indices:
        tick_x = x_pos(idx)
        tick_label = escape(str(ordered_models[idx].get("model_id", "")))
        x_axis_markup.append(
            f'<line x1="{tick_x:.2f}" y1="{height - margin_bottom}" '
            f'x2="{tick_x:.2f}" y2="{height - margin_bottom + 6}" '
            'stroke="#94a3b8" stroke-width="1"/>'
        )
        x_axis_markup.append(
            f'<text x="{tick_x:.2f}" y="{height - 16}" text-anchor="middle" '
            'font-family="system-ui, sans-serif" font-size="10" fill="#64748b">'
            f"{tick_label}</text>"
        )

    polyline_points = []
    point_markup = []
    for idx, value, model_id in points:
        x = x_pos(idx)
        y = y_pos(value)
        polyline_points.append(f"{x:.2f},{y:.2f}")
        point_markup.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="#0f766e">'
            f'<title>{escape(model_id)}: {value:.6g}</title></circle>'
        )

    line_markup = ""
    if len(polyline_points) >= 2:
        line_markup = (
            '<polyline fill="none" stroke="#0f766e" stroke-width="2.5" '
            f'points="{" ".join(polyline_points)}"/>'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(metric_label)} across model runs">'
        f'<rect width="{width}" height="{height}" rx="12" fill="#ffffff" stroke="#cbd5e1"/>'
        f'<text x="{margin_left}" y="16" font-family="system-ui, sans-serif" '
        'font-size="13" font-weight="600" fill="#0f172a">'
        f"{escape(metric_label)}</text>"
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" '
        f'x2="{width - margin_right}" y2="{height - margin_bottom}" '
        'stroke="#94a3b8" stroke-width="1"/>'
        f'{"".join(grid_markup)}'
        f'{"".join(x_axis_markup)}'
        f"{line_markup}"
        f'{"".join(point_markup)}'
        "</svg>"
    )


def get_mesh_directory(job: UploadJob, seg_id: str) -> pathlib.Path:
    """
    Return the directory that stores mesh artifacts for a segmentation.
    """
    return _mesh_directory(job) / seg_id


def get_mesh_files(
    job: UploadJob,
    seg_id: str,
    pattern: str,
) -> list[pathlib.Path]:
    """
    Return sorted mesh artifact files for a segmentation.
    """
    mesh_path = get_mesh_directory(job, seg_id)
    if not mesh_path.exists():
        return []
    return sorted(path for path in mesh_path.glob(pattern) if path.is_file())


def get_latest_segmentation_id(job: UploadJob) -> Optional[str]:
    """
    Get the latest segmentation for the job, or None if it doesn't exist.

    Segmentation is stored as a zarr file in the segmented directory, and it
    is named after the job ID and the time at which it was generated. These
    segmentation files MAY be broken or partial if the segmentation process
    was interrupted, so the latest segmentation may not always be the one
    that should be used.

    Arguments:
        job (UploadJob): The job for which to get the latest segmentation.

    Returns:
        str: The name of the latest segmentation file, like "1234.zarr".
        None: If no segmentation file exists for the job.

    """
    zarr_path = _segmentation_directory(job)
    if not zarr_path.exists():
        return None
    segmentation_paths = sorted(zarr_path.glob("*.zarr"))
    if len(segmentation_paths) == 0:
        return None
    segmentation_path = segmentation_paths[-1]
    return segmentation_path.name


def get_latest_segmentation_model(job: UploadJob) -> Optional[pathlib.Path]:
    """
    Get the latest segmentation model for the job.

    Segmentation models are stored as pickles in the models directory, and
    they are named after the job ID and the time at which they were generated.
    These line up with the segmentation filenames in the segmented directory.

    There is no such thing as a broken or partial segmentation model, so the
    latest model is always the one that should be used, assuming the training
    data hasn't gotten worse.

    The only time this method should return None is if the segmentation model
    directory doesn't exist.

    Arguments:
        job (UploadJob): The job for which to get the latest model.

    Returns:
        pathlib.Path: The path to the latest segmentation model.
        None: if no model has been made yet.

    """
    model_runs = get_model_runs(job)
    if len(model_runs) == 0:
        return None
    return model_runs[0]["model_path"]


def get_latest_mesh_id(job: UploadJob) -> Optional[str]:
    """
    Get the latest meshed segmentation ID for the job, or None if absent.

    Meshes are stored under the meshed directory as one subdirectory per
    segmentation timestamp/ID.
    """
    mesh_path = _mesh_directory(job)
    if not mesh_path.exists():
        return None

    mesh_dirs = sorted(path for path in mesh_path.iterdir() if path.is_dir())
    for mesh_dir in reversed(mesh_dirs):
        if any(mesh_dir.glob("*.combined.stl")):
            return mesh_dir.name
    return None


def get_latest_mesh_obj_path(job: UploadJob) -> Optional[pathlib.Path]:
    """
    Return one representative OBJ mesh for the latest valid mesh output.
    """
    latest_mesh_id = get_latest_mesh_id(job)
    if latest_mesh_id is None:
        return None

    obj_files = get_mesh_files(job, latest_mesh_id, "*.combined.obj")
    if len(obj_files) == 0:
        return None

    for obj_file in obj_files:
        if obj_file.name == "255.combined.obj":
            return obj_file
    return obj_files[0]


def create_neuroglancer_link(job: UploadJob, return_state: bool = False):
    """
    Create a neuroglancer link for the images.

    Arguments:
        job (UploadJob): The job for which to create the link.
        return_state (bool): If True, return the neuroglancer state as well.

    Returns:
        str: The neuroglancer link
        dict: The neuroglancer state, if `return_state` is True.
    """
    # Check for segmentation:

    protocol = "https" # request.url.split(":")[0]
    jsondata = {
        "layers": [
            {
                "type": "image",
                "source": f"zarr://{protocol}://{request.host}/api/job/{job.id}/zarr/",
                "tab": "source",
                "name": "zarr",
            }
        ]
    }
    zarr_path = pathlib.Path(CONFIG.segmented_directory) / job.id
    if zarr_path.exists():
        # Get the latest segmentation (the last one in the list)
        seg_id = get_latest_segmentation_id(job)

        # Create the neuroglancer layer:
        seg_layer = {
            "type": "segmentation",
            "source": f"zarr://{protocol}://{request.host}/api/job/{job.id}/segmentation/{seg_id}/zarr/",
            "tab": "source",
            "name": f"segmentation {seg_id}",
        }
        jsondata["layers"].append(seg_layer)

    mesh_obj_path = get_latest_mesh_obj_path(job)
    if mesh_obj_path is not None:
        mesh_seg_id = mesh_obj_path.parent.name

        mesh_layer = {
            "type": "mesh",
            "source": {
                "url": f"obj://{protocol}://{request.host}/api/job/{job.id}/segmentation/{mesh_seg_id}/obj/{mesh_obj_path.name}",
                "transform": {
                    "matrix": [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
                    "outputDimensions": {
                        "d0": [1, "m"],
                        "d1": [1, "m"],
                        "d2": [1, "m"],
                    },
                    "inputDimensions": {"x": [1, "m"], "y": [1, "m"], "z": [1, "m"]},
                },
            },
            "tab": "source",
            "name": f"mesh",
        }
        jsondata["layers"].append(mesh_layer)
    jsondump = json.dumps(jsondata)

    if return_state:
        return f"https://neuroglancer.bossdb.io/#!{jsondump}", jsondata
    return f"https://neuroglancer.bossdb.io/#!{jsondump}"


def normalize_annotation_volume(
    vol: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
) -> tuple[np.ndarray, dict]:
    """
    Normalize a sampled annotation subvolume into a high-contrast uint8 volume.

    The annotation UI and training round-trip go through a browser canvas, so
    they are effectively limited to 8-bit display data. We therefore choose a
    robust intensity window server-side before serializing to PNG.
    """
    vol = np.asarray(vol)
    finite_values = vol[np.isfinite(vol)]
    if finite_values.size == 0:
        return np.zeros(vol.shape, dtype=np.uint8), {
            "source_min": 0.0,
            "source_max": 0.0,
            "window_min": 0.0,
            "window_max": 0.0,
        }

    source_min = float(finite_values.min())
    source_max = float(finite_values.max())

    # Prefer non-zero voxels when possible so padded borders and sparse black
    # backgrounds don't collapse the display range.
    nonzero_values = finite_values[finite_values != 0]
    window_values = nonzero_values if nonzero_values.size >= 128 else finite_values

    window_min = float(np.percentile(window_values, lower_percentile))
    window_max = float(np.percentile(window_values, upper_percentile))

    if window_max <= window_min:
        window_min = source_min
        window_max = source_max

    if window_max <= window_min:
        fill_value = 0 if source_max <= 0 else 255
        return np.full(vol.shape, fill_value, dtype=np.uint8), {
            "source_min": source_min,
            "source_max": source_max,
            "window_min": window_min,
            "window_max": window_max,
        }

    vol_float = vol.astype(np.float32, copy=False)
    normalized = np.clip((vol_float - window_min) / (window_max - window_min), 0.0, 1.0)
    normalized = (normalized * 255.0).round().astype(np.uint8)
    return normalized, {
        "source_min": source_min,
        "source_max": source_max,
        "window_min": window_min,
        "window_max": window_max,
    }


def get_png_filmstrip(vol: np.ndarray):
    """
    Create an image filmstrip from a volume.

    Note that this assumes that the volume is in ZYX order. The filmstrip will
    concatenate the layers in the Y direction, so the width will be the same
    as the width of the volume, and the height will be the height of the volume
    times the number of layers.

    Arguments:
        vol (np.ndarray): The volume to create the filmstrip from.

    Returns:
        PIL.Image: The filmstrip as an Image

    """
    # Get the number of layers
    num_layers = vol.shape[0]
    # Get the width and height of each layer
    width, height = vol.shape[1:]
    # Create a new image
    filmstrip = Image.fromarray(np.zeros((height * num_layers, width), dtype=vol.dtype))
    # Loop through the layers
    for i in range(num_layers):
        # Get the layer
        layer = vol[i]
        # Get the start and end of the layer in the filmstrip
        start = i * height
        end = (i + 1) * height
        # Paste the layer into the filmstrip
        filmstrip.paste(Image.fromarray(layer), (0, start, width, end))
    return filmstrip
