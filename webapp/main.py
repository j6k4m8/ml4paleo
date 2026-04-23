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
    count_annotation_samples,
    get_annotation_pairs,
    get_latest_segmentation_id,
    get_latest_mesh_id,
    get_mesh_directory,
    get_mesh_files,
    get_latest_segmentation_model,
    load_annotation_source_slice,
    create_neuroglancer_link,
    get_png_filmstrip,
    normalize_annotation_volume,
)
from segmentrunner import train_job

log = logging.getLogger(__name__)


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

            latest_segmentation_id = get_latest_segmentation_id(job)
            latest_mesh_seg_id = get_latest_mesh_id(job)
            return render_template(
                "job_page.html",
                job=job,
                voxel_count=voxel_count,
                voxel_count_localized=voxel_count_localized,
                num_annotations=num_annotations,
                neuroglancer_link=create_neuroglancer_link(job),
                latest_segmentation_id=latest_segmentation_id,
                latest_mesh_seg_id=latest_mesh_seg_id,
                # Status breakdown:
                has_been_annotated=(
                    job.status
                    not in [
                        JobStatus.PENDING,
                        JobStatus.UPLOADING,
                        JobStatus.UPLOADED,
                        JobStatus.CONVERTING,
                        JobStatus.CONVERTED,
                        JobStatus.CONVERT_ERROR,
                    ]
                ),
                annotation_ready=(
                    job.status
                    not in [
                        JobStatus.CONVERTING,
                        JobStatus.UPLOADING,
                        JobStatus.UPLOADED,
                        JobStatus.CONVERT_ERROR,
                    ]
                ),
                segmentation_ready=(
                    job.status
                    in [
                        JobStatus.SEGMENTED,
                        JobStatus.MESHING_QUEUED,
                        JobStatus.MESHING_QUEUED,
                        JobStatus.MESH_ERROR,
                        JobStatus.MESHED,
                    ]
                ),
                meshes_ready=latest_mesh_seg_id is not None,
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
            data = request.get_json()
            if not data or "image" not in data or "mask" not in data:
                return (
                    jsonify(
                        {"status": "error", "message": "{mask, image} not provided"}
                    ),
                    400,
                )
            image_b64 = data.get("image", None)
            mask_b64 = data.get("mask", None)
            sample_metadata = data.get("sample_metadata", None)
            if image_b64 is None or mask_b64 is None:
                return (
                    jsonify(
                        {"status": "error", "message": "{mask, image} not provided"}
                    ),
                    400,
                )
            # Decode the image and mask:
            tic = str(int(time.time()))
            _prefix = pathlib.Path(CONFIG.training_directory) / job.id
            fname = _prefix / (CONFIG.training_img_prefix + tic + ".png")
            # Make sure the directory exists:
            fname.parent.mkdir(parents=True, exist_ok=True)
            img = Image.open(io.BytesIO(base64.b64decode(image_b64.split(",")[1])))
            # Crop the image to the annotation shape:
            img_height = img.height
            img_slices = CONFIG.annotation_shape_xyz[2]
            middle_slice = img_slices // 2

            img = img.crop(
                (
                    # Left Up Right Lower
                    0,
                    int((img_height / img_slices) * middle_slice),
                    int(img.width),
                    int((img_height / img_slices) * (middle_slice + 1)),
                )
            )
            img.save(fname)
            mfname = _prefix / (CONFIG.training_seg_prefix + tic + ".png")
            Image.open(io.BytesIO(base64.b64decode(mask_b64.split(",")[1]))).resize(
                CONFIG.annotation_shape_xyz[:-1]
            ).save(mfname)
            if isinstance(sample_metadata, dict):
                meta_fname = _prefix / (CONFIG.training_meta_prefix + tic + ".json")
                metadata_to_save = dict(sample_metadata)
                metadata_to_save.update(
                    {
                        "timestamp": tic,
                        "job_id": job.id,
                        "training_image_filename": fname.name,
                        "training_mask_filename": mfname.name,
                    }
                )
                with open(meta_fname, "w") as f:
                    json.dump(metadata_to_save, f)

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
            Image.open(mfname).save(f, format="PNG")
            f.seek(0)
            return jsonify(
                {
                    "prediction": base64.b64encode(f.read()).decode("utf-8"),
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
            data = request.get_json()
            if not data or "image" not in data:
                return (
                    jsonify({"status": "error", "message": "image not provided"}),
                    400,
                )
            image_b64 = data.get("image", None)
            sample_metadata = data.get("sample_metadata", None)
            if isinstance(sample_metadata, dict):
                img_np = load_annotation_source_slice(job.id, sample_metadata)
            else:
                img_np = np.array(
                    Image.open(io.BytesIO(base64.b64decode(image_b64.split(",")[1])))
                )
                slice_count = CONFIG.annotation_shape_xyz[2]
                middle_slice = slice_count // 2
                start_y = int((img_np.shape[0] / slice_count) * middle_slice)
                img_np = img_np[
                    start_y : start_y + CONFIG.annotation_shape_xyz[1],
                    :,
                    0,
                ].T
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
            # Convert to RGBA with annos as R:
            mask = np.zeros((annos.shape[0], annos.shape[1], 4), dtype=np.uint8)
            mask[:, :, 0] = annos
            mask[:, :, 3] = annos

            # Return the mask as a base64 encoded string:
            f = io.BytesIO()
            Image.fromarray(mask).save(f, format="PNG")
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
