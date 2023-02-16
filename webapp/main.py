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
from ml4paleo.volume_providers.io import get_random_tile, export_to_img_stack

from config import CONFIG
from apputils import get_latest_segmentation_id, get_latest_segmentation_model

log = logging.getLogger(__name__)


def _create_neuroglancer_link(job: UploadJob):
    """
    Create a neuroglancer link for the images.

    TODO: If there's segmentation available, include it as a layer.
    """
    # Check for segmentation:

    protocol = request.url.split(":")[0]
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
        # segmentation_path = sorted(zarr_path.glob("*.zarr"))[-1]
        # seg_id = segmentation_path.name
        seg_id = get_latest_segmentation_id(job)

        # Create the neuroglancer layer:
        seg_layer = {
            "type": "segmentation",
            "source": f"zarr://http://{request.host}/api/job/{job.id}/segmentation/{seg_id}/zarr/",
            "tab": "source",
            "name": f"segmentation {seg_id}",
        }
        jsondata["layers"].append(seg_layer)
    jsondata = json.dumps(jsondata)

    return f"https://neuroglancer.bossdb.io/#!{jsondata}"


def _transform_img_slice_for_annotation(img_slice):
    """
    Transform an image slice to be suitable for annotation.
    """
    # Get vmin and vmax, excluding anything within 0..2% of the dynamic range.
    # If this is u8, ignore; if this is u16, then the dynamic range is 0..65535
    # so chop off everything between 0 and (0.02 * 65535) ~= 1300
    THRESH = 1300

    img_out = img_slice.copy()
    if img_out.dtype == np.uint8:
        pass
    elif img_out.dtype == np.uint16:
        img_out[img_out < THRESH] = 0

    # Get vmin and vmax as 3*stdev from the mean
    vmin = img_out.mean() - 8 * img_out.std()
    vmax = img_out.mean() + 8 * img_out.std()

    # Normalize to 0..255
    img_out = (img_out - vmin) / (vmax - vmin) * 255
    img_out = img_out.astype(np.uint8)
    return img_out


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

        @self.app.route("/api/job/new", methods=["POST"])
        def new_job():
            job = UploadJob(
                status=JobStatus.UPLOADING,
                name=(request.get_json() or {}).get("name", "Untitled Job")
                # These fields will be automatically populated:
                # id=None,
                # created_at=None,
            )
            job_id = job_manager.new_job(job)
            return jsonify({"job_id": job_id})

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

            job = job_manager.get_job(job_id)
            return jsonify({"job_id": job_id, "status": str(job.status)})

        @self.app.route("/api/job/status/upload-complete", methods=["POST"])
        def upload_complete():
            job_id = (request.get_json() or {}).get("job_id", None)
            if job_id is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            job = job_manager.get_job(job_id)
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
            num_annotations = (
                len(
                    [
                        f
                        for f in (
                            pathlib.Path(CONFIG.training_directory) / job.id
                        ).glob("*")
                        if f.is_file()
                    ]
                )
                // 2
            )
            return render_template(
                "job_page.html",
                job=job,
                num_annotations=num_annotations,
                neuroglancer_link=_create_neuroglancer_link(job),
                latest_segmentation_id=get_latest_segmentation_id(job),
                # Status breakdown:
                annotation_ready=(
                    job.status
                    not in [
                        JobStatus.CONVERTING,
                        JobStatus.UPLOADING,
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
                meshes_ready=job.status in [JobStatus.MESHED],
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
            imgslice = get_random_tile(
                zarrvol,
                CONFIG.annotation_shape,
            )
            # imgslice = _transform_img_slice_for_annotation(imgslice)
            # Save the imgslice to bytesio:
            img = Image.fromarray(imgslice)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            # Return the slice as a file:
            return send_file(img_bytes, mimetype="image/png")

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
            Image.open(io.BytesIO(base64.b64decode(image_b64.split(",")[1]))).save(
                fname
            )

            mfname = _prefix / (CONFIG.training_seg_prefix + tic + ".png")
            Image.open(io.BytesIO(base64.b64decode(mask_b64.split(",")[1]))).resize(
                CONFIG.annotation_shape
            ).save(mfname)

            # Update the job to the JobStatus.ANNOTATED state:
            job_manager.update_job(job_id, update=dict(status=JobStatus.ANNOTATED))

            # Return the mask as a base64 encoded string:
            # Return zeros the same size as the image:
            f = io.BytesIO()
            Image.open(mfname).save(f, format="PNG")
            f.seek(0)
            return jsonify({"prediction": base64.b64encode(f.read()).decode("utf-8")})

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
            img_np = np.array(
                Image.open(io.BytesIO(base64.b64decode(image_b64.split(",")[1])))
            )

            # Load the latest model if it exists:
            modelpath = get_latest_segmentation_model(job)
            if modelpath is None:
                jsonify({"prediction": None})

            # Predict the mask:
            model = RandomForest3DSegmenter()
            model.load(str(modelpath))
            print("Loaded model: %s", modelpath)
            print(img_np.shape)
            mask = model._segment_slice(img_np[:, :, 0])
            annos = np.array(Image.fromarray(mask).resize(CONFIG.annotation_shape))
            # Convert to RGBA with annos as R:
            mask = np.zeros((annos.shape[0], annos.shape[1], 4), dtype=np.uint8)
            mask[:, :, 0] = annos
            mask[:, :, 3] = annos

            # Return the mask as a base64 encoded string:
            # Return zeros the same size as the image:
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

        ###############################
        #
        #  Training & Inference
        #
        ###############################

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
            return render_template(
                "download_page.html",
                job=job,
                latest_seg_id=get_latest_segmentation_id(job),
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
            if not zip_fname.exists():
                # Meshes are stored in CONFIG.meshes_directory / job_id / seg_id:
                mesh_path = pathlib.Path(CONFIG.meshed_directory) / job_id / seg_id
                # Check if the directory exists:
                if not mesh_path.exists():
                    return (
                        jsonify({"status": "error", "message": "meshes do not exist"}),
                        400,
                    )

                # Compress all .combined.stl files in the directory:
                with zipfile.ZipFile(zip_fname, "w") as zf:
                    for root, dirs, files in os.walk(mesh_path):
                        for file in files:
                            if file.endswith(".combined.stl"):
                                zf.write(
                                    os.path.join(root, file),
                                    arcname=os.path.join(os.path.basename(root), file),
                                )

            return send_file(zip_fname, as_attachment=True)

    def run(self, **kwargs):
        self.app.run(host="0.0.0.0", **kwargs)


if __name__ == "__main__":
    app = Flask(__name__)
    CORS(app)

    # Create the ML4Paleo server:
    server = ML4PaleoWebApplication(app)

    # Run the server:
    server.run(debug=True, port=5000)
