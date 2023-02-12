import logging
import os
from flask import Flask, jsonify, render_template, request, make_response
from werkzeug.utils import secure_filename

from job import JSONFileUploadJobManager, JobStatus, UploadJob

from config import CONFIG

log = logging.getLogger(__name__)
app = Flask(__name__)


class ML4PaleoWebApplication:
    def __init__(self, app: Flask):
        job_manager = JSONFileUploadJobManager("jobs.json")
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
                    ("Not sure why," " but we couldn't write the file to disk", 500)
                )
            total_chunks = int(request.form["dztotalchunkcount"])
            if current_chunk + 1 == total_chunks:
                # This was the last chunk, the file should be complete and the size we expect
                if os.path.getsize(save_path) != int(request.form["dztotalfilesize"]):
                    log.error(
                        f"File {file.filename} was completed, "
                        f"but has a size mismatch."
                        f"Was {os.path.getsize(save_path)} but we"
                        f" expected {request.form['dztotalfilesize']} "
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
        def get_job_status():
            job_id = (request.get_json() or {}).get("job_id", None)
            if job_id is None:
                return (
                    jsonify({"status": "error", "message": "job_id is required"}),
                    400,
                )

            job = job_manager.get_job(job_id)
            return jsonify({"job_id": job_id, "status": job.status})

    def run(self, **kwargs):
        self.app.run(host="0.0.0.0", **kwargs)


if __name__ == "__main__":
    app = Flask(__name__)

    # Create the ML4Paleo server:
    server = ML4PaleoWebApplication(app)

    # Run the server:
    server.run(debug=True, port=5000)
