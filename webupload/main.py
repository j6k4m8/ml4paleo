import abc
import logging
import os
import time
from typing import Dict
import json
import uuid
from enum import Enum
from flask import Flask, jsonify, render_template, request, make_response
from werkzeug.utils import secure_filename

log = logging.getLogger(__name__)

app = Flask(__name__)


class CONFIG:
    upload_directory = "uploads"


class JobStatus(Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    DONE = "done"
    ERROR = "error"


class UploadJob:
    def __init__(self):
        self.status = JobStatus.PENDING

    def to_dict(self):
        return {
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UploadJob":
        job = cls()
        job.status = JobStatus(d["status"])
        return job


UploadJobID = str


class UploadJobManager(abc.ABC):
    @abc.abstractmethod
    def new_job_id(self) -> UploadJobID:
        ...

    @abc.abstractmethod
    def new_job(self, job: UploadJob) -> UploadJobID:
        ...

    @abc.abstractmethod
    def get_job(self, job_id: UploadJobID) -> UploadJob:
        ...

    @abc.abstractmethod
    def has_job(self, job_id: UploadJobID) -> bool:
        ...


class JSONFileUploadJobManager(UploadJobManager):
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                json.dump({}, f)

    def _load_jobs(self) -> Dict[UploadJobID, UploadJob]:
        """
        Get all jobs from the file.
        """
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Move the corrupted file out of the way:
            os.rename(self.file_path, f"{self.file_path}.{time.time()}")
            return {}

    def _save_jobs(self, jobs: Dict[UploadJobID, UploadJob]):
        """
        Save all jobs to the file.
        """
        with open(self.file_path, "w") as f:
            jobs_jsonable = {k: v.to_dict() for k, v in jobs.items()}
            json.dump(jobs_jsonable, f)

    def new_job(self, job: UploadJob) -> UploadJobID:
        """
        Create a new job and return its ID.
        """
        jobs = self._load_jobs()
        job_id = self.new_job_id()
        jobs[job_id] = job
        self._save_jobs(jobs)
        return job_id

    def get_job(self, job_id: UploadJobID) -> UploadJob:
        """
        Get a job by its ID.
        """
        jobs = self._load_jobs()
        return jobs[job_id]

    def has_job(self, job_id: UploadJobID) -> bool:
        """
        Check if a job exists.
        """
        jobs = self._load_jobs()
        return job_id in jobs.keys()

    def new_job_id(self) -> UploadJobID:
        """
        Generate a new job ID.
        """
        return str(uuid.uuid4())


job_manager = JSONFileUploadJobManager("jobs.json")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/job/new", methods=["POST"])
def new_job():
    job = UploadJob()
    job_id = job_manager.new_job(job)
    return jsonify({"job_id": job_id})


# https://codecalamity.com/uploading-large-files-by-chunking-featuring-python-flask-and-dropzone-js/
@app.route("/api/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    # Get the job ID from the headers:
    job_id = request.headers.get("X-Job-ID", None)
    if job_id is None:
        return make_response(("No job ID", 400))
    # Get the job from the job manager:
    if not job_manager.has_job(job_id):
        return make_response(("Invalid job", 400))

    save_path = os.path.join(
        str(CONFIG.upload_directory), job_id, secure_filename(file.filename)
    )
    print(request.form.to_dict())
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


@app.route("/api/job/status", methods=["POST"])
def get_job_status():
    job_id = (request.get_json() or {}).get("job_id", None)
    if job_id is None:
        return jsonify({"status": "error", "message": "job_id is required"}), 400

    job = job_manager.get_job(job_id)
    return jsonify({"job_id": job_id, "status": job.status})


if __name__ == "__main__":
    app.run(debug=True)
