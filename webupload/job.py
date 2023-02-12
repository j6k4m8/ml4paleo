import datetime
import abc
import logging
import os
import time
from typing import Dict, Optional
import json
import uuid
from enum import Enum
from marshmallow import Schema, fields

from enum import Enum
import uuid

log = logging.getLogger(__name__)

# Types:
UploadJobID = str


class JobStatus(Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    CONVERTING = "converting"
    CONVERTED = "converted"
    CONVERT_ERROR = "convert_error"
    SEGMENTING = "segmenting"
    SEGMENTED = "segmented"
    SEGMENT_ERROR = "segment_error"
    DONE = "done"
    ERROR = "error"


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

    def start_segment(self):
        self.set_status(JobStatus.SEGMENTING)

    def complete_segment(self):
        self.set_status(JobStatus.SEGMENTED)

    def complete(self):
        self.set_status(JobStatus.DONE)

    def to_dict(self):
        # Serialize the object to a dict, including datetime objects:
        return UploadJobSchema().dump(self)  # type: ignore

    @classmethod
    def from_dict(cls, d: dict) -> "UploadJob":
        return UploadJobSchema().load(d)  # type: ignore


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
                results = {k: UploadJob.from_dict(v) for k, v in json.load(f).items()}
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
            json.dump(jobs_jsonable, f)

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
        return _new_job_id()
