import datetime
import abc
import logging
import os
import time
from typing import Dict, List, Optional
import json
import uuid
from enum import Enum
from marshmallow import Schema, fields
from jque import jque

from enum import Enum
import uuid

log = logging.getLogger(__name__)

# Types:
UploadJobID = str


class JobStatus(Enum):
    # Unused.
    PENDING = "pending"

    # When a user starts a job, it is created in the Uploading state.
    UPLOADING = "uploading"
    # Once the web client registers the 100% mark on the upload, we call the
    # server to mark the job as "uploaded".
    UPLOADED = "uploaded"

    # The server will periodically check for UPLOADED jobs and start the
    # conversion process by switching the status to CONVERTING.
    CONVERTING = "converting"
    # Once the conversion is complete (right now all handled by the same job),
    # the status will be set to CONVERTED.
    CONVERTED = "converted"
    # Unused.
    CONVERT_ERROR = "convert_error"

    # If a job has been annotated AT ALL and there are ANY training examples,
    # the job will be put in the ANNOTATED state. This doesn't mean there is
    # a complete annotation, just that there is at least one training example.
    # This means that the job can be trained, and we can show that option to
    # the user, in case they're low-patience. (But it also means that it's
    # pretty likely that the model will be retrained a few times with an
    # increasing number of training examples...)
    ANNOTATED = "annotated"

    # Once a user manually kicks off the training process, the job will be
    # queued for training. This is the state it will be in until the training
    # process picks it up and converts it to TRAINING.
    TRAINING_QUEUED = "training_queued"

    # Unused:
    SEGMENTING = "segmenting"
    SEGMENTED = "segmented"
    SEGMENT_ERROR = "segment_error"
    DONE = "done"
    ERROR = "error"

    @staticmethod
    def from_string(label: str) -> "JobStatus":
        for status in JobStatus:
            if "." in label:
                label = label.split(".")[-1]
            if status.value.lower() == label.lower():
                return status
        raise ValueError(f"Invalid job status: {label}")


class UploadJobSchema(Schema):
    status = fields.Str()
    name = fields.Str()
    id = fields.Str()
    created_at = fields.Str()


def _new_job_id() -> UploadJobID:
    """
    Return a six digit random string.
    """
    return uuid.uuid4().hex[:6].upper()


class UploadJob:
    def __init__(
        self,
        id: Optional[UploadJobID] = None,
        name: Optional[str] = None,
        status: Optional[JobStatus] = None,
        created_at: Optional[str] = None,
    ):
        created_at = created_at or datetime.datetime.now().isoformat()
        self.status = status or JobStatus.PENDING
        self.name = name or "Untitled Job created at " + created_at
        self.id = id or _new_job_id()
        self.created_at = created_at

    def set_status(self, status: JobStatus):
        self.status = status

    def start_upload(self):
        self.set_status(JobStatus.UPLOADING)

    def complete_upload(self):
        self.set_status(JobStatus.UPLOADED)

    def start_convert(self):
        self.set_status(JobStatus.CONVERTING)

    def complete_convert(self):
        self.set_status(JobStatus.CONVERTED)

    def complete(self):
        self.set_status(JobStatus.DONE)

    def to_dict(self):
        # Serialize the object to a dict, including datetime objects:
        return UploadJobSchema().dump(self)  # type: ignore

    @classmethod
    def from_dict(cls, d: dict) -> "UploadJob":
        res = UploadJob(
            id=d["id"],
            name=d["name"],
            status=JobStatus.from_string(d["status"]),
            created_at=d["created_at"],
        )
        return res


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
                results = {
                    k: (UploadJob.from_dict(v) if isinstance(v, dict) else v)
                    for k, v in json.load(f).items()
                }
                return results
        except json.JSONDecodeError as e:
            # Move the corrupted file out of the way:
            log.exception(
                f"Corrupted jobs file: {self.file_path}, got error {e}. Moving to {self.file_path}.{time.time()}"
            )
            os.rename(self.file_path, f"{self.file_path}.{time.time()}")
            return {}

    def _save_jobs(self, jobs: Dict[UploadJobID, UploadJob]):
        """
        Save all jobs to the file.
        """
        with open(self.file_path, "w") as f:
            jobs_jsonable = {
                k: (v if isinstance(v, dict) else v.to_dict()) for k, v in jobs.items()
            }
            json.dump(jobs_jsonable, f, indent=4)

    def new_job(self, job: UploadJob) -> UploadJobID:
        """
        Create a new job and return its ID.
        """
        jobs = self._load_jobs()
        jobs[job.id] = job
        self._save_jobs(jobs)
        return job.id

    def get_job(self, job_id: UploadJobID) -> UploadJob:
        """
        Get a job by its ID.
        """
        jobs = self._load_jobs()
        try:
            d = jobs[job_id]
            return d
        except KeyError:
            raise IndexError(job_id)

    def update_job(
        self,
        job_id: UploadJobID,
        job: Optional[UploadJob] = None,
        update: Optional[dict] = None,
    ) -> UploadJobID:
        """
        Update a job by its ID.
        """
        jobs = self._load_jobs()
        if job is None:
            job = jobs[job_id]
        if update is not None:
            for k, v in update.items():
                setattr(job, k, v)
        jobs[job_id] = job
        self._save_jobs(jobs)
        return job.id

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
        return _new_job_id()

    def get_jobs_by_status(self, status: JobStatus) -> List[UploadJob]:
        """
        Get all jobs with a given status.
        """
        jobs = {k: v.to_dict() for k, v in self._load_jobs().items()}
        qry = {"status": f"{status}"}
        return [UploadJob.from_dict(u) for u in jque(list(jobs.values())).query(qry)]
