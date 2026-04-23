"""
This is the main script of the web application. It is an absolute monster of a
file, and will be split up into smaller files in the future.

For now, the best way to find a function or process you're looking for is to
search for the URL that corresponds to the action, and then look at the
function that is decorated by the flask routing for that URL.

"""

import base64
import io
import json
import zipfile
import numpy as np
import logging
import os
import pathlib
import time
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    make_response,
    send_file,
    send_from_directory,
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

from PIL import Image
from job import JSONFileUploadJobManager, JobStatus, UploadJob
from ml4paleo.segmentation.rf import RandomForest3DSegmenter

from ml4paleo.volume_providers import ZarrVolumeProvider
from ml4paleo.volume_providers.io import (
    get_random_tile,
    export_to_img_stack,
    get_random_zyx_subvolume,
)

from config import CONFIG
from apputils import (
    annotation_canvas_size_xy,
    annotation_display_image,
    annotation_mask_to_rgba,
    annotation_polygon_source_size_xy,
    annotation_sample_metadata_for_z,
    build_model_metric_chart_svg,
    count_annotation_samples,
    extract_annotation_image_slice,
    get_annotation_pairs,
    get_job_artifact_freshness,
    get_latest_segmentation_id,
    get_latest_mesh_id,
    get_mesh_directory,
    get_mesh_files,
    get_latest_segmentation_model,
    get_model_runs,
    migrate_model_metadata_sidecar,
    load_annotation_source_slice,
    create_neuroglancer_link,
    get_png_filmstrip,
    normalize_annotation_volume,
    rasterize_annotation_regions,
)
from segmentrunner import train_job

log = logging.getLogger(__name__)

TIMELINE_STEPS = ("Upload", "Convert", "Annotate", "Train", "Segment", "Mesh")
SOURCE_LABELS = {
    "image_stack": "Image stack",
    "dicom": "DICOM series",
}
MODEL_PAGE_METRIC_OPTIONS = {
    "train_foreground_dice": "Train Dice",
    "train_loss": "Training Loss (1 - Dice)",
    "train_foreground_iou": "Train IoU",
}


def _timeline_position_for_status(status: JobStatus) -> tuple[int | None, int | None]:
    """
    Return the current timeline index and optional error index for a job status.
    """
    if status in [JobStatus.PENDING, JobStatus.UPLOADING]:
        return 0, None
    if status in [JobStatus.UPLOADED, JobStatus.CONVERTING]:
        return 1, None
    if status == JobStatus.CONVERT_ERROR:
        return 1, 1
    if status == JobStatus.CONVERTED:
        return 2, None
    if status in [JobStatus.ANNOTATED, JobStatus.TRAINING_QUEUED, JobStatus.TRAINING]:
        return 3, None
    if status in [JobStatus.TRAINED, JobStatus.SEGMENTING]:
        return 4, None
    if status == JobStatus.SEGMENT_ERROR:
        return 4, 4
    if status in [JobStatus.SEGMENTED, JobStatus.MESHING_QUEUED, JobStatus.MESHING]:
        return 5, None
    if status == JobStatus.MESH_ERROR:
        return 5, 5
    if status in [JobStatus.MESHED, JobStatus.DONE]:
        return None, None
    return 0, 0


def build_job_timeline(job: UploadJob) -> list[dict[str, str]]:
    """
    Build the rendered timeline state for the job page.
    """
    current_index, error_index = _timeline_position_for_status(job.status)
    steps: list[dict[str, str]] = []
    for idx, label in enumerate(TIMELINE_STEPS):
        if error_index == idx:
            state = "error"
        elif current_index is None:
            state = "done"
        elif idx < current_index:
            state = "done"
        elif idx == current_index:
            state = "current"
        else:
            state = "upcoming"
        steps.append({"label": label, "state": state})
    return steps


def build_job_timeline_caption(job: UploadJob, num_annotations: int) -> str:
    """
    Return a short status line for the page timeline.
    """
    progress_pct = job.current_job_progress * 100
    if job.status == JobStatus.PENDING:
        return "Waiting for upload to begin."
    if job.status == JobStatus.UPLOADING:
        return "Uploading source files."
    if job.status == JobStatus.UPLOADED:
        return "Upload complete. Conversion is next."
    if job.status == JobStatus.CONVERTING:
        return f"Converting source data into the project volume ({progress_pct:.1f}%)."
    if job.status == JobStatus.CONVERT_ERROR:
        return "Conversion failed."
    if job.status == JobStatus.CONVERTED:
        return "Volume ready for annotation."
    if job.status == JobStatus.ANNOTATED:
        return f"{num_annotations} annotation sample(s) saved. Training is next."
    if job.status == JobStatus.TRAINING_QUEUED:
        return "Training is queued."
    if job.status == JobStatus.TRAINING:
        return "Training the segmentation model."
    if job.status == JobStatus.TRAINED:
        return "Model is trained. Segmentation is next."
    if job.status == JobStatus.SEGMENTING:
        return f"Running segmentation across the full volume ({progress_pct:.1f}%)."
    if job.status == JobStatus.SEGMENT_ERROR:
        return "Segmentation failed."
    if job.status == JobStatus.SEGMENTED:
        return "Segmentation complete. Mesh generation is available."
    if job.status == JobStatus.MESHING_QUEUED:
        return "Meshing is queued."
    if job.status == JobStatus.MESHING:
        return "Generating surface meshes."
    if job.status == JobStatus.MESH_ERROR:
        return "Mesh generation failed."
    if job.status == JobStatus.MESHED:
        return "Meshes are ready."
    if job.status == JobStatus.DONE:
        return "Processing complete."
    return "Project state unavailable."


def _payload_field(payload, snake_name: str, camel_name: str | None = None):
    """
    Return a payload value, accepting either snake_case or camelCase keys.
    """
    if not isinstance(payload, dict):
        return None
    if snake_name in payload:
        return payload.get(snake_name)
    if camel_name is not None and camel_name in payload:
        return payload.get(camel_name)
    return None


def _coerce_annotation_regions(payload) -> tuple[list, list]:
    """
    Extract positive/negative polygon regions from a submit payload.
    """
    polygon_payload = _payload_field(payload, "polygons")
    if isinstance(polygon_payload, dict):
        positive_regions = _payload_field(
            polygon_payload, "positive_regions", "positiveRegions"
        )
        negative_regions = _payload_field(
            polygon_payload, "negative_regions", "negativeRegions"
        )
    else:
        positive_regions = _payload_field(payload, "positive_regions", "positiveRegions")
        negative_regions = _payload_field(payload, "negative_regions", "negativeRegions")

    if not isinstance(positive_regions, list):
        positive_regions = []
    if not isinstance(negative_regions, list):
        negative_regions = []
    return positive_regions, negative_regions


def _submission_uses_polygon_regions(payload) -> bool:
    """
    Return True when a submit payload contains polygon geometry.
    """
    positive_regions, negative_regions = _coerce_annotation_regions(payload)
    polygon_payload = _payload_field(payload, "polygons")
    regions_by_slice = _payload_field(payload, "regions_by_slice", "regionsBySlice")
    return (
        isinstance(polygon_payload, dict)
        or (isinstance(regions_by_slice, dict) and len(regions_by_slice) > 0)
        or "positive_regions" in payload
        or "positiveRegions" in payload
        or "negative_regions" in payload
        or "negativeRegions" in payload
        or len(positive_regions) > 0
        or len(negative_regions) > 0
    )


def _coerce_regions_by_slice(
    payload: dict,
    *,
    default_slice_index: int,
) -> dict[int, dict[str, list]]:
    """
    Normalize one polygon payload into a per-slice region mapping.
    """
    normalized: dict[int, dict[str, list]] = {}
    regions_by_slice = _payload_field(payload, "regions_by_slice", "regionsBySlice")
    if isinstance(regions_by_slice, dict):
        for raw_slice_index, raw_region_payload in regions_by_slice.items():
            try:
                slice_index = int(raw_slice_index)
            except (TypeError, ValueError):
                continue
            if not isinstance(raw_region_payload, dict):
                continue
            positive_regions, negative_regions = _coerce_annotation_regions(
                raw_region_payload
            )
            if positive_regions or negative_regions:
                normalized[slice_index] = {
                    "positive_regions": positive_regions,
                    "negative_regions": negative_regions,
                }
    if normalized:
        return normalized

    positive_regions, negative_regions = _coerce_annotation_regions(payload)
    if positive_regions or negative_regions:
        normalized[int(default_slice_index)] = {
            "positive_regions": positive_regions,
            "negative_regions": negative_regions,
        }
    return normalized


def _updated_sample_metadata(payload) -> dict | None:
    """
    Return normalized sample metadata with optional z/window updates applied.
    """
    sample_metadata = _payload_field(payload, "sample_metadata", "sampleMetadata")
    if not isinstance(sample_metadata, dict):
        return None

    annotated_local_z_index = _payload_field(
        payload,
        "annotated_local_z_index",
        "annotatedLocalZIndex",
    )
    if annotated_local_z_index is None:
        annotated_local_z_index = sample_metadata.get("annotated_local_z_index")

    intensity_window = _payload_field(payload, "intensity_window", "intensityWindow")
    if not isinstance(intensity_window, dict):
        intensity_window = sample_metadata.get("intensity_window")

    return annotation_sample_metadata_for_z(
        sample_metadata,
        annotated_local_z_index=annotated_local_z_index,
        intensity_window=intensity_window if isinstance(intensity_window, dict) else None,
    )


def _decode_data_url_image(data_url: str) -> Image.Image:
    """
    Decode one `data:image/...;base64,...` payload into a PIL image.
    """
    encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def _legacy_rgba_image(img_xy: np.ndarray) -> Image.Image:
    """
    Convert one grayscale display slice into a legacy-style RGBA PNG.
    """
    img_uint8 = np.asarray(img_xy, dtype=np.uint8)
    rgba = np.zeros((img_uint8.shape[0], img_uint8.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0] = img_uint8
    rgba[:, :, 1] = img_uint8
    rgba[:, :, 2] = img_uint8
    rgba[:, :, 3] = 255
    return Image.fromarray(rgba)


def _legacy_rgba_mask(mask_xy: np.ndarray) -> Image.Image:
    """
    Convert one binary mask into the legacy red-plus-alpha PNG style.
    """
    mask_uint8 = np.asarray(mask_xy, dtype=np.uint8)
    rgba = np.zeros((mask_uint8.shape[0], mask_uint8.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0] = mask_uint8
    rgba[:, :, 3] = mask_uint8
    return Image.fromarray(rgba)


def _polygon_source_size_xy(
    payload: dict,
    sample_metadata: dict | None = None,
    fallback_size_xy: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """
    Resolve the XY coordinate space used by polygon points in one payload.
    """
    polygon_payload = _payload_field(payload, "polygons")
    canvas_size_xy = _payload_field(payload, "canvas_size_xy", "canvasSizeXY")
    if canvas_size_xy is None:
        canvas_size_xy = _payload_field(
            payload,
            "displayed_image_shape_xy",
            "displayedImageShapeXY",
        )
    if canvas_size_xy is None and isinstance(polygon_payload, dict):
        canvas_size_xy = _payload_field(
            polygon_payload, "canvas_size_xy", "canvasSizeXY"
        )
    if canvas_size_xy is None and isinstance(polygon_payload, dict):
        canvas_size_xy = _payload_field(
            polygon_payload,
            "displayed_image_shape_xy",
            "displayedImageShapeXY",
        )
    return annotation_polygon_source_size_xy(
        sample_metadata,
        canvas_size_xy=canvas_size_xy,
        fallback_size_xy=fallback_size_xy,
    )


class ML4PaleoWebApplication:
    """
    The main web application.

    """

    def __init__(self, app: Flask):
        job_manager = JSONFileUploadJobManager("volume/jobs.json")
        self.app = app

        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.app.route("/about")
        def about():
            return render_template("about.html")

        @self.app.route("/api/job/new", methods=["POST"])
        def new_job():
            request_json = request.get_json() or {}
            try:
                job = UploadJob(
                    status=JobStatus.UPLOADING,
                    name=request_json.get("name", "Untitled Job"),
                    source_type=request_json.get("source_type", "image_stack"),
                    # These fields will be automatically populated:
                    # id=None,
                    # created_at=None,
                )
            except ValueError as exc:
                return jsonify({"status": "error", "message": str(exc)}), 400
            job_id = job_manager.new_job(job)
            return jsonify({"job_id": job_id})

        ############################
        #
        #  Uploading
        #
        ############################

        # https://codecalamity.com/uploading-large-files-by-chunking-featuring-python-flask-and-dropzone-js/
        @self.app.route("/api/upload", methods=["POST"])
        def upload_file():
            file = request.files["file"]
            if not file.filename:
                return make_response(("No file", 400))
            job_id = request.headers.get("X-Job-ID", None)
            if job_id is None:
                return make_response(("No job ID", 400))
            # Get the job from the job manager:
            if not job_manager.has_job(job_id):
                return make_response(("Invalid job", 400))

            # Create the directory for the job if it doesn't exist:
            job_dir = os.path.join(str(CONFIG.upload_directory), job_id)
            if not os.path.exists(job_dir):
                os.makedirs(job_dir, exist_ok=True)

            save_path = os.path.join(
                str(CONFIG.upload_directory), job_id, secure_filename(file.filename)
            )

            current_chunk = int(request.form["dzchunkindex"])
            # If the file already exists it's ok if we are appending to it,
            # but not if it's new file that would overwrite the existing one
            if os.path.exists(save_path) and current_chunk == 0:
                # 400 and 500s will tell dropzone that an error occurred and show an error
                return make_response(("File already exists", 400))
            try:
                with open(save_path, "ab") as f:
                    f.seek(int(request.form["dzchunkbyteoffset"]))
                    f.write(file.stream.read())
            except OSError:
                # log.exception will include the traceback so we can see what's wrong
                log.exception("Could not write to file")
                return make_response(
                    ("Not sure why, but we couldn't write the file to disk", 500)
                )
            total_chunks = int(request.form["dztotalchunkcount"])
            if current_chunk + 1 == total_chunks:
                # This was the last chunk, the file should be complete and the size we expect
                if os.path.getsize(save_path) != int(request.form["dztotalfilesize"]):
                    log.error(
                        f"File {file.filename} was completed, but has a size mismatch."
                        f"Was {os.path.getsize(save_path)} but we expected {request.form['dztotalfilesize']} "
                    )
                    return make_response(("Size mismatch", 500))
                else:
                    log.info(f"File {file.filename} has been uploaded successfully")
            else:
                log.debug(
                    f"Chunk {current_chunk + 1} of {total_chunks} "
                    f"for file {file.filename} complete"
                )
            return make_response(("Chunk upload successful", 200))

        @self.app.route("/api/job/status", methods=["POST"])
        def get_post_job_status():
            job_id = (request.get_json() or {}).get("job_id", None)
            if job_id is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            job = job_manager.get_job(job_id)
            return jsonify({"job_id": job_id, "status": str(job.status)})

        @self.app.route("/api/job/<job_id>/status", methods=["GET"])
        def get_job_status(job_id):
            if job_id is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            try:
                job = job_manager.get_job(job_id)
            except IndexError:
                return (
                    jsonify({"status": "error", "message": f"Job ID {job_id} not found."}),
                    400,
                )
            return jsonify({"job_id": job_id, "status": str(job.status)})

        @self.app.route("/api/job/<job_id>/rename", methods=["POST"])
        def rename_job(job_id):
            if job_id is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            request_json = request.get_json() or {}
            new_name = (request_json.get("name") or "").strip()
            if not new_name:
                return (
                    jsonify({"status": "error", "message": "name is required"}),
                    400,
                )

            try:
                job_manager.get_job(job_id)
            except IndexError:
                return (
                    jsonify({"status": "error", "message": f"Job ID {job_id} not found."}),
                    400,
                )

            job_manager.update_job(job_id, update={"name": new_name})
            return jsonify({"status": "success", "job_id": job_id, "name": new_name})

        @self.app.route("/api/job/status/upload-complete", methods=["POST"])
        def upload_complete():
            job_id = (request.get_json() or {}).get("job_id", None)
            if job_id is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            job = job_manager.get_job(job_id)
            upload_dir = pathlib.Path(CONFIG.upload_directory) / job_id
            if not upload_dir.exists() or not any(upload_dir.iterdir()):
                return (
                    jsonify({"status": "error", "message": "No uploaded files found"}),
                    400,
                )
            job.complete_upload()
            job_manager.update_job(job_id, job)
            return jsonify({"job_id": job_id, "status": str(job.status)})

        @self.app.route("/job/<job_id>", methods=["GET"])
        def job_page(job_id):
            if job_id is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            job = job_manager.get_job(job_id)
            # Get the number of annotations for this job:
            num_annotations = count_annotation_samples(job)

            voxel_count = (job.shape[0] * job.shape[1] * job.shape[2]) if job.shape else 0
            voxel_count_localized = "Not available yet"
            # Replace voxels with KV, MV, GV, etc.
            if voxel_count:
                voxel_count_localized = voxel_count
                for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
                    if voxel_count_localized < 1024:
                        break
                    voxel_count_localized /= 1024
                voxel_count_localized = f"{voxel_count_localized:.2f} {unit}V"

            artifact_freshness = get_job_artifact_freshness(job)
            latest_segmentation_id = artifact_freshness["latest_segmentation_id"]
            latest_mesh_seg_id = artifact_freshness["latest_mesh_seg_id"]
            segmentation_output_available = latest_segmentation_id is not None
            mesh_output_available = latest_mesh_seg_id is not None
            segmentation_stale = bool(artifact_freshness["segmentation_stale"])
            mesh_stale = bool(artifact_freshness["mesh_stale"])
            has_been_annotated = num_annotations > 0
            annotation_ready = job.status not in [
                JobStatus.CONVERTING,
                JobStatus.UPLOADING,
                JobStatus.UPLOADED,
                JobStatus.CONVERT_ERROR,
            ]
            segmentation_ready = segmentation_output_available and not segmentation_stale
            meshes_ready = mesh_output_available and not mesh_stale
            can_trigger_segmentation = has_been_annotated and job.status not in [
                JobStatus.TRAINING_QUEUED,
                JobStatus.TRAINING,
                JobStatus.SEGMENTING,
            ]
            can_trigger_meshing = latest_segmentation_id is not None and (
                not segmentation_stale
            ) and job.status not in [
                JobStatus.MESHING_QUEUED,
                JobStatus.MESHING,
            ]
            should_offer_mesh_regeneration = can_trigger_meshing and (
                (not mesh_output_available) or mesh_stale
            )
            show_annotation_cta = (
                (annotation_ready and job.status != JobStatus.SEGMENTING)
                or (
                    job.status == JobStatus.CONVERTING
                    and job.current_job_progress > 0.4
                )
            )
            return render_template(
                "job_page.html",
                job=job,
                job_status_str=str(job.status),
                source_label=SOURCE_LABELS.get(
                    job.source_type,
                    job.source_type.replace("_", " ").title(),
                ),
                timeline_steps=build_job_timeline(job),
                timeline_status_text=build_job_timeline_caption(job, num_annotations),
                voxel_count=voxel_count,
                voxel_count_localized=voxel_count_localized,
                num_annotations=num_annotations,
                neuroglancer_link=create_neuroglancer_link(job),
                latest_segmentation_id=latest_segmentation_id,
                latest_mesh_seg_id=latest_mesh_seg_id,
                latest_model_id=artifact_freshness["latest_model_id"],
                show_annotation_cta=show_annotation_cta,
                is_converting=job.status == JobStatus.CONVERTING,
                is_training=job.status in [
                    JobStatus.TRAINING_QUEUED,
                    JobStatus.TRAINING,
                ],
                is_segmenting=job.status == JobStatus.SEGMENTING,
                is_meshing=job.status in [
                    JobStatus.MESHING_QUEUED,
                    JobStatus.MESHING,
                ],
                has_error=job.status in [
                    JobStatus.CONVERT_ERROR,
                    JobStatus.SEGMENT_ERROR,
                    JobStatus.MESH_ERROR,
                    JobStatus.ERROR,
                ],
                # Status breakdown:
                has_been_annotated=has_been_annotated,
                annotation_ready=annotation_ready,
                segmentation_ready=segmentation_ready,
                meshes_ready=meshes_ready,
                segmentation_output_available=segmentation_output_available,
                mesh_output_available=mesh_output_available,
                segmentation_stale=segmentation_stale,
                segmentation_stale_reasons=artifact_freshness["segmentation_stale_reasons"],
                mesh_stale=mesh_stale,
                mesh_stale_reasons=artifact_freshness["mesh_stale_reasons"],
                latest_model_annotation_count=artifact_freshness["latest_model_annotation_count"],
                latest_segmentation_annotation_count=artifact_freshness["latest_segmentation_annotation_count"],
                latest_mesh_annotation_count=artifact_freshness["latest_mesh_annotation_count"],
                can_trigger_segmentation=can_trigger_segmentation,
                can_trigger_meshing=can_trigger_meshing,
                should_offer_mesh_regeneration=should_offer_mesh_regeneration,
            )

        ############################
        #
        # Annotation
        #
        ############################

        @self.app.route("/api/annotate/<job_id>/images/next")
        def get_next_image(job_id):
            """
            Get a random 512x512 slice from the dataset and return it for
            tracing with `send_file`.
            """
            job = job_manager.get_job(job_id)
            if job is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            # Get the zarr file for the dataset:
            zarr_path = pathlib.Path(CONFIG.chunked_directory) / job.id
            if not zarr_path.exists():
                return (
                    jsonify({"status": "error", "message": "zarr file not found"}),
                    400,
                )
            zarrvol = ZarrVolumeProvider(zarr_path)
            # Get a random slice from the dataset:
            vol_zyx, sample_metadata = get_random_zyx_subvolume(
                zarrvol,
                CONFIG.annotation_shape_xyz[::-1],
                return_metadata=True,
            )
            display_vol_zyx, display_stats = normalize_annotation_volume(vol_zyx)
            # Get the slice as a PIL image:
            img = get_png_filmstrip(display_vol_zyx)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            # Return the slice as a file:
            resp = make_response(send_file(img_bytes, mimetype="image/png"))
            resp.headers["zInfo"] = json.dumps(
                {
                    "zInfo": {
                        "min": 0,
                        "max": int(display_vol_zyx.shape[0] - 1),
                        "current": int(display_vol_zyx.shape[0] // 2),
                    },
                    "intensityWindow": display_stats,
                    "sampleMetadata": sample_metadata,
                }
            )
            return resp

        @self.app.route("/api/annotate/<job_id>/data/submit", methods=["POST"])
        def submit_annotation(job_id):
            job = job_manager.get_job(job_id)
            if job is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            # Get the annotation data from the request:
            data = request.get_json() or {}
            uses_polygon_payload = _submission_uses_polygon_regions(data)
            image_b64 = _payload_field(data, "image")
            mask_b64 = _payload_field(data, "mask")
            sample_metadata = _updated_sample_metadata(data)
            if not uses_polygon_payload and (image_b64 is None or mask_b64 is None):
                return (
                    jsonify(
                        {"status": "error", "message": "{mask, image} not provided"}
                    ),
                    400,
                )
            if uses_polygon_payload and image_b64 is None and sample_metadata is None:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "polygon submission requires image or sample_metadata",
                        }
                    ),
                    400,
                )
            default_slice_index = (
                int(sample_metadata["annotated_local_z_index"])
                if isinstance(sample_metadata, dict)
                else int(CONFIG.annotation_shape_xyz[2] // 2)
            )
            polygon_regions_by_slice = _coerce_regions_by_slice(
                data,
                default_slice_index=default_slice_index,
            )
            if uses_polygon_payload and len(polygon_regions_by_slice) == 0:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "polygon submission requires at least one region",
                        }
                    ),
                    400,
                )

            tic = str(int(time.time()))
            _prefix = pathlib.Path(CONFIG.training_directory) / job.id
            _prefix.mkdir(parents=True, exist_ok=True)
            decoded_image = (
                _decode_data_url_image(str(image_b64)) if image_b64 is not None else None
            )
            response_mask_path: pathlib.Path | None = None
            saved_samples: list[dict[str, str | int | None]] = []

            if uses_polygon_payload:
                if sample_metadata is None and len(polygon_regions_by_slice) > 1:
                    keep_slice_index = (
                        default_slice_index
                        if default_slice_index in polygon_regions_by_slice
                        else sorted(polygon_regions_by_slice.keys())[0]
                    )
                    polygon_regions_by_slice = {
                        keep_slice_index: polygon_regions_by_slice[keep_slice_index]
                    }

                for slice_index in sorted(polygon_regions_by_slice):
                    slice_regions = polygon_regions_by_slice[slice_index]
                    sample_metadata_for_slice = (
                        annotation_sample_metadata_for_z(
                            sample_metadata,
                            annotated_local_z_index=slice_index,
                            intensity_window=sample_metadata.get("intensity_window"),
                        )
                        if sample_metadata is not None
                        else None
                    )
                    if sample_metadata_for_slice is not None:
                        training_image = annotation_display_image(
                            job.id,
                            sample_metadata_for_slice,
                        )
                    elif decoded_image is not None:
                        training_image = extract_annotation_image_slice(decoded_image)
                    else:
                        return (
                            jsonify(
                                {
                                    "status": "error",
                                    "message": "image or sample_metadata is required",
                                }
                            ),
                            400,
                        )

                    suffix = (
                        tic
                        if len(polygon_regions_by_slice) == 1
                        else f"{tic}-z{int(slice_index):02d}"
                    )
                    fname = _prefix / (CONFIG.training_img_prefix + suffix + ".png")
                    mfname = _prefix / (CONFIG.training_seg_prefix + suffix + ".png")
                    training_image.save(fname)

                    mask_np = rasterize_annotation_regions(
                        annotation_canvas_size_xy(),
                        positive_regions=slice_regions["positive_regions"],
                        negative_regions=slice_regions["negative_regions"],
                        source_size_xy=_polygon_source_size_xy(
                            data,
                            sample_metadata_for_slice or sample_metadata,
                            fallback_size_xy=training_image.size,
                        ),
                    )
                    annotation_mask_to_rgba(mask_np).save(mfname)
                    if response_mask_path is None or slice_index == default_slice_index:
                        response_mask_path = mfname

                    metadata_to_save = dict(sample_metadata_for_slice or {})
                    metadata_to_save.update(
                        {
                            "timestamp": suffix,
                            "job_id": job.id,
                            "training_image_filename": fname.name,
                            "training_mask_filename": mfname.name,
                            "annotation_source": "polygon_v2",
                        }
                    )
                    meta_fname = _prefix / (CONFIG.training_meta_prefix + suffix + ".json")
                    with open(meta_fname, "w") as f:
                        json.dump(metadata_to_save, f)
                    saved_samples.append(
                        {
                            "slice_index": int(slice_index),
                            "image_filename": fname.name,
                            "mask_filename": mfname.name,
                            "metadata_filename": meta_fname.name,
                        }
                    )
            else:
                fname = _prefix / (CONFIG.training_img_prefix + tic + ".png")
                mfname = _prefix / (CONFIG.training_seg_prefix + tic + ".png")
                if decoded_image is not None:
                    training_image = extract_annotation_image_slice(
                        decoded_image,
                        sample_metadata,
                    )
                elif sample_metadata is not None:
                    training_image = annotation_display_image(job.id, sample_metadata)
                else:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "image or sample_metadata is required",
                            }
                        ),
                        400,
                    )
                training_image.save(fname)
                _decode_data_url_image(str(mask_b64)).resize(
                    CONFIG.annotation_shape_xyz[:-1]
                ).save(mfname)
                response_mask_path = mfname
                meta_fname = None
                if sample_metadata is not None:
                    meta_fname = _prefix / (CONFIG.training_meta_prefix + tic + ".json")
                    metadata_to_save = dict(sample_metadata)
                    metadata_to_save.update(
                        {
                            "timestamp": tic,
                            "job_id": job.id,
                            "training_image_filename": fname.name,
                            "training_mask_filename": mfname.name,
                            "annotation_source": "legacy_raster",
                        }
                    )
                    with open(meta_fname, "w") as f:
                        json.dump(metadata_to_save, f)
                saved_samples.append(
                    {
                        "slice_index": int(default_slice_index),
                        "image_filename": fname.name,
                        "mask_filename": mfname.name,
                        "metadata_filename": meta_fname.name if meta_fname else None,
                    }
                )

            # If the job is CONVERTING, don't change the status (we need the
            # daemon to continue to run the conversion process.)
            # But we can allow the user to submit more annotations:
            if job.status == JobStatus.CONVERTING:
                pass
            else:
                # Update the job to the JobStatus.ANNOTATED state:
                job_manager.update_job(job_id, update=dict(status=JobStatus.ANNOTATED))

            # Return the mask as a base64 encoded string:
            # Return zeros the same size as the image:
            f = io.BytesIO()
            Image.open(response_mask_path).save(f, format="PNG")
            f.seek(0)
            return jsonify(
                {
                    "prediction": base64.b64encode(f.read()).decode("utf-8"),
                    "saved_samples": saved_samples,
                }
            )

        @self.app.route("/api/annotate/<job_id>/data/predict", methods=["POST"])
        def predict_annotation(job_id):
            job = job_manager.get_job(job_id)
            if job is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            # Get the annotation data from the request:
            data = request.get_json() or {}
            sample_metadata = _updated_sample_metadata(data)
            image_b64 = _payload_field(data, "image")
            if sample_metadata is None and image_b64 is None:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "image or sample_metadata not provided",
                        }
                    ),
                    400,
                )
            if sample_metadata is not None:
                img_np = load_annotation_source_slice(job.id, sample_metadata)
            else:
                display_image = extract_annotation_image_slice(
                    _decode_data_url_image(str(image_b64))
                )
                img_np = np.asarray(display_image)[:, :, 0].T
            # Load the latest model if it exists:
            modelpath = get_latest_segmentation_model(job)
            if modelpath is None:
                return jsonify({"prediction": None})

            # Predict the mask:
            model = RandomForest3DSegmenter()
            model.load(str(modelpath))
            mask = model._segment_slice(img_np)
            mask = mask.T
            annos = np.array(
                Image.fromarray(mask).resize(CONFIG.annotation_shape_xyz[:-1])
            )

            # Return the mask as a base64 encoded string:
            f = io.BytesIO()
            annotation_mask_to_rgba(annos).save(f, format="PNG")
            f.seek(0)
            return jsonify({"prediction": base64.b64encode(f.read()).decode("utf-8")})

        @self.app.route("/job/annotate/<job_id>", methods=["GET"])
        def annotation_page(job_id):
            if job_id is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            job = job_manager.get_job(job_id)
            return render_template("annotation_page.html", job=job)

        @self.app.route("/job/annotate-v2/<job_id>", methods=["GET"])
        def annotation_page_v2(job_id):
            if job_id is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            job = job_manager.get_job(job_id)
            return render_template("annotation_page_v2.html", job=job)

        @self.app.route("/job/<job_id>/annotations", methods=["GET"])
        def annotation_gallery(job_id):
            if job_id is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            job = job_manager.get_job(job_id)

            img_seg_pairs = []
            for img_file, seg_file, _meta_file in get_annotation_pairs(job):
                img_seg_pairs.append((img_file.name, seg_file.name))

            return render_template(
                "annotation_gallery.html",
                job=job,
                num_annotations=len(img_seg_pairs),
                img_seg_pairs=img_seg_pairs,
            )

        @self.app.route("/job/<job_id>/models", methods=["GET"])
        def models_page(job_id):
            if job_id is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            job = job_manager.get_job(job_id)
            metric_key = request.args.get("metric", "train_foreground_dice")
            if metric_key not in MODEL_PAGE_METRIC_OPTIONS:
                metric_key = "train_foreground_dice"
            metric_label = MODEL_PAGE_METRIC_OPTIONS[metric_key]
            models = get_model_runs(job, train_metric_key=metric_key)
            chart_svg = build_model_metric_chart_svg(
                models,
                metric_key=metric_key,
                metric_label=metric_label,
            )
            metric_count = sum(
                1 for model in models if model.get("train_metric_value") is not None
            )
            return render_template(
                "models_page.html",
                job=job,
                models=models,
                chart_svg=chart_svg,
                chart_metric_key=metric_key,
                chart_metric_label=metric_label,
                chart_metric_count=metric_count,
                metric_options=MODEL_PAGE_METRIC_OPTIONS,
            )

        @self.app.route("/api/job/<job_id>/models/<model_id>/migrate", methods=["POST"])
        def migrate_model_metadata(job_id, model_id):
            if job_id is None or model_id is None:
                return (
                    jsonify({"status": "error", "message": "job_id and model_id are required"}),
                    400,
                )

            job = job_manager.get_job(job_id)
            if job is None:
                return (
                    jsonify({"status": "error", "message": "job does not exist"}),
                    400,
                )

            try:
                metadata_path = migrate_model_metadata_sidecar(job, model_id)
            except FileNotFoundError:
                return (
                    jsonify({"status": "error", "message": "model does not exist"}),
                    400,
                )
            except Exception as exc:
                log.exception(
                    "Failed to migrate metadata for job %s model %s",
                    job_id,
                    model_id,
                )
                return (
                    jsonify({"status": "error", "message": str(exc)}),
                    500,
                )

            return jsonify(
                {
                    "status": "success",
                    "model_id": model_id,
                    "metadata_file": metadata_path.name,
                }
            )

        # Satisfy requests for /job/<job_id>/annotations/img1720204733.png
        @self.app.route("/job/<job_id>/annotations/<filename>", methods=["GET"])
        def annotation_image(job_id, filename):
            # Rewrite to /volume/training/<job_id>/<filename>
            return send_from_directory(
                os.path.join(CONFIG.training_directory, job_id), filename
            )



        ###############################
        #
        #  Training & Inference
        #
        ###############################

        # /api/job/{{ job.id }}/retrain
        @self.app.route("/api/job/<job_id>/retrain", methods=["POST"])
        def retrain_model(job_id):
            if job_id is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            job = job_manager.get_job(job_id)
            if job is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            # Update the job to the JobStatus.TRAINING state:
            job_manager.update_job(job_id, update=dict(status=JobStatus.TRAINING))

            # Start the training process:
            train_job(job)

            job_manager.update_job(job_id, update=dict(status=JobStatus.TRAINED))

            return jsonify({"status": "success"})

        @self.app.route("/api/job/<job_id>/start", methods=["POST"])
        def trigger_training(job_id):
            if job_id is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            job = job_manager.get_job(job_id)
            if job is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )
            # Update the job to the JobStatus.TRAINING state:
            job_manager.update_job(
                job_id, update=dict(status=JobStatus.TRAINING_QUEUED)
            )
            return jsonify({"status": "success"})

        ###############################
        #
        #  Meshing
        #
        ###############################

        @self.app.route("/api/job/<job_id>/mesh", methods=["POST"])
        def trigger_meshing(job_id):
            if job_id is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            job = job_manager.get_job(job_id)
            if job is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )
            # Update the job to the JobStatus.TRAINING state:
            job_manager.update_job(job_id, update=dict(status=JobStatus.MESHING_QUEUED))
            return jsonify({"status": "success"})

        ###############################
        #
        #  Visualization
        #
        ###############################
        @self.app.route("/api/job/<job_id>/zarr/<path>", methods=["GET"])
        def render_zarr(job_id, path):
            # Just serve files from the zarr directory:
            return send_from_directory(
                os.path.join(CONFIG.chunked_directory, job_id), path
            )

        @self.app.route(
            "/api/job/<job_id>/segmentation/<seg_id>/zarr/<path>", methods=["GET"]
        )
        def render_zarr_seg(job_id, path, seg_id):
            # Just serve files from the zarr directory:
            return send_from_directory(
                os.path.join(CONFIG.segmented_directory, job_id, seg_id), path
            )

        @self.app.route(
            "/api/job/<job_id>/segmentation/<seg_id>/obj/<path>", methods=["GET"]
        )
        def serve_obj_mesh(job_id, path, seg_id):
            # Just serve files from the zarr directory:
            return send_from_directory(
                os.path.join(CONFIG.meshed_directory, job_id, seg_id), path
            )

        ###############################
        #
        #  Downloading
        #
        ###############################
        @self.app.route("/job/<job_id>/download", methods=["GET"])
        def download_page(job_id):
            if job_id is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            job = job_manager.get_job(job_id)
            latest_mesh_seg_id = get_latest_mesh_id(job)
            return render_template(
                "download_page.html",
                job=job,
                latest_seg_id=get_latest_segmentation_id(job),
                latest_mesh_seg_id=latest_mesh_seg_id,
                segmentation_done=job.status
                in [
                    JobStatus.SEGMENTED,
                    JobStatus.MESHING_QUEUED,
                    JobStatus.MESHING,
                    JobStatus.MESHED,
                    JobStatus.MESH_ERROR,
                ],
            )

        @self.app.route("/api/job/<job_id>/models/<model_id>/download", methods=["GET"])
        def download_model(job_id, model_id):
            job = job_manager.get_job(job_id)
            if job is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            model_path = pathlib.Path(CONFIG.model_directory) / job_id / f"{model_id}.model"
            if not model_path.exists():
                return (
                    jsonify({"status": "error", "message": "model does not exist"}),
                    400,
                )
            return send_file(model_path, as_attachment=True)

        @self.app.route(
            "/api/job/<job_id>/models/<model_id>/download/json", methods=["GET"]
        )
        def download_model_metadata(job_id, model_id):
            job = job_manager.get_job(job_id)
            if job is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            metadata_path = pathlib.Path(CONFIG.model_directory) / job_id / f"{model_id}.json"
            if not metadata_path.exists():
                return (
                    jsonify(
                        {"status": "error", "message": "model metadata does not exist"}
                    ),
                    400,
                )
            return send_file(metadata_path, as_attachment=True)

        @self.app.route(
            "/api/job/<job_id>/segmentation/<seg_id>/download/zarr", methods=["GET"]
        )
        def download_zarr(job_id, seg_id):
            job = job_manager.get_job(job_id)
            if job is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            # Create a zip file in CONFIG.download_cache:
            zip_fname = (
                pathlib.Path(CONFIG.download_cache) / f"{job_id}_{seg_id}.zarr.zip"
            )
            zip_fname.parent.mkdir(parents=True, exist_ok=True)
            if not zip_fname.exists():
                # Make sure the directory exists:
                with zipfile.ZipFile(zip_fname, "w") as zf:
                    for root, dirs, files in os.walk(
                        os.path.join(CONFIG.segmented_directory, job_id, seg_id)
                    ):
                        for file in files:
                            zf.write(
                                os.path.join(root, file),
                                arcname=os.path.join(os.path.basename(root), file),
                            )

            return send_file(zip_fname, as_attachment=True)

        @self.app.route(
            "/api/job/<job_id>/segmentation/<seg_id>/download/png", methods=["GET"]
        )
        def download_png(job_id, seg_id):
            job = job_manager.get_job(job_id)
            if job is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            # Create a zip file in CONFIG.download_cache:
            zip_fname = (
                pathlib.Path(CONFIG.download_cache) / f"{job_id}_{seg_id}.png.zip"
            )
            zip_fname.parent.mkdir(parents=True, exist_ok=True)
            if not zip_fname.exists():
                # Export the segmentation to png, using volume_provider:
                volume_provider = ZarrVolumeProvider(
                    pathlib.Path(CONFIG.segmented_directory) / job_id / seg_id
                )
                png_path = (
                    pathlib.Path(CONFIG.download_cache) / f"{job_id}_{seg_id}" / "png"
                )
                png_path.mkdir(parents=True, exist_ok=True)
                # Export to cache directory:
                export_to_img_stack(volume_provider, png_path, parallel_jobs=-1)

                # Zip the directory:
                with zipfile.ZipFile(zip_fname, "w") as zf:
                    for root, dirs, files in os.walk(png_path):
                        for file in files:
                            zf.write(
                                os.path.join(root, file),
                                arcname=os.path.join(os.path.basename(root), file),
                            )

            return send_file(zip_fname, as_attachment=True)

        @self.app.route("/api/job/<job_id>/meshes/<seg_id>/download", methods=["GET"])
        def download_meshes(job_id, seg_id):
            job = job_manager.get_job(job_id)
            if job is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            # Create a zip file in CONFIG.download_cache:
            zip_fname = (
                pathlib.Path(CONFIG.download_cache) / f"{job_id}_{seg_id}.meshes.zip"
            )
            zip_fname.parent.mkdir(parents=True, exist_ok=True)
            mesh_path = get_mesh_directory(job, seg_id)
            if not mesh_path.exists():
                return (
                    jsonify({"status": "error", "message": "meshes do not exist"}),
                    400,
                )

            stl_files = get_mesh_files(job, seg_id, "*.combined.stl")
            if len(stl_files) == 0:
                if zip_fname.exists():
                    zip_fname.unlink()
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "No STL meshes were generated for this segmentation.",
                        }
                    ),
                    400,
                )

            if zip_fname.exists():
                zip_fname.unlink()
            with zipfile.ZipFile(zip_fname, "w") as zf:
                for stl_file in stl_files:
                    zf.write(stl_file, arcname=stl_file.name)

            return send_file(zip_fname, as_attachment=True)

    def run(self, **kwargs):
        self.app.run(host="0.0.0.0", **kwargs)


app = Flask(__name__)
CORS(app)

# Create the ML4Paleo server:
server = ML4PaleoWebApplication(app)

if __name__ == "__main__":
    # Run the server:
    server.run(debug=True, port=5000)
