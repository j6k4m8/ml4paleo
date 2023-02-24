"""
This file contains the UploadJob and JobManager classes, which are responsible
for tracking a single job and collections of jobs, respectively.

This also contains the JobStatus enum, which is used to track the status of a
job over the course of its lifecycle.
This enum is a bit of a frustration, because it becomes very inconvenient to
tell if a job has, say, segmentation ready... since the corresponding job
statuses that indicate this are ANY post-segmentation status. In the future, it
would be nice to have a status system that uses a bitfield or something to
indicate which parts of the job are ready, and which parts are not.

"""

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
    # The job is currently being trained.
    TRAINING = "training"
    # The job has been trained.
    TRAINED = "trained"

    SEGMENTING = "segmenting"
    SEGMENTED = "segmented"
    SEGMENT_ERROR = "segment_error"

    # Any job that is "SEGMENTED" can be manually put into the MESHING_QUEUED
    # state. This is the state it will be in until the meshing process picks
    # it up and converts it to MESHING.
    MESHING_QUEUED = "meshing_queued"
    MESHING = "meshing"
    MESHED = "meshed"
    MESH_ERROR = "mesh_error"

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
    last_updated_at = fields.Str()
    current_job_progress = fields.Float()


def _new_job_id() -> UploadJobID:
    """
    Return a six digit random string.
    """
    return uuid.uuid4().hex[:6].upper()


class UploadJob:
    """
    An UploadJob is a single job that a user has uploaded to the server.

    The job is created in the UPLOADING state, and then transitions through the
    other states as the job is processed.

    """

    def __init__(
        self,
        id: Optional[UploadJobID] = None,
        name: Optional[str] = None,
        status: Optional[JobStatus] = None,
        created_at: Optional[str] = None,
        last_updated_at: Optional[str] = None,
        current_job_progress: Optional[float] = None,
    ):
        """
        Create a new job with the fieldwise constructor.

        All arguments are optional, and will be filled in with default values.

        Arguments:
            id (UploadJobID): The ID of the job. If not provided, a random ID
                will be generated.
            name (str): The name of the job. If not provided, a default name
                will be generated.
            status (JobStatus): The status of the job. If not provided, the
                status will be set to PENDING.
            created_at (str): The time the job was created. If not provided,
                the current time will be used.
            last_updated_at (str): The time the job was last updated. If not
                provided, the current time will be used (equal to created_at)
            current_job_progress (float): The progress of the current operation
                on the job. If not provided, the progress will be set to 0.

        """
        created_at = created_at or datetime.datetime.now().isoformat()
        last_updated_at = last_updated_at or datetime.datetime.now().isoformat()
        self.status = status or JobStatus.PENDING
        self.name = name or "Untitled Job created at " + created_at
        self.id = id or _new_job_id()
        self.created_at = created_at
        self.last_updated_at = last_updated_at
        self.current_job_progress = current_job_progress or 0.0

    def set_status(self, status: JobStatus):
        """
        Set the status of the job, and update the last_updated_at field.

        Arguments:
            status (JobStatus): The new status of the job.

        """
        self.status = status
        self.last_updated_at = datetime.datetime.now().isoformat()

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
            last_updated_at=d.get("last_updated_at", d["created_at"]),
            current_job_progress=d.get("current_job_progress", 0.0),
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
    """
    This class manages the upload jobs by storing them in a JSON file.

    This is mostly production-ready, but is not thread safe, so updating the
    file COULD be a problem if multiple processes are accessing it at the same
    time. Practically this is rarely a problem, but it's something to be aware
    of when scaling up.

    To be consistent, this reads from the file every time a job is requested,
    and writes to the file every time a job is updated. This is not the most
    efficient way to do it, but it's the simplest, and it's not a problem for
    the small number of jobs we expect to have.

    """

    def __init__(self, file_path: str):
        """
        Create a new JSONFileUploadJobManager.

        Arguments:
            file_path (str): The path to the JSON file to use. If it does not
                exist, it will be created.

        """
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                json.dump({}, f)

    def _load_jobs(self) -> Dict[UploadJobID, UploadJob]:
        """
        Get all jobs from the file.
        """
        retries = 3
        for i in range(retries):
            try:
                with open(self.file_path, "r") as f:
                    results = {
                        k: (UploadJob.from_dict(v) if isinstance(v, dict) else v)
                        for k, v in json.load(f).items()
                    }
                    return results
            except json.JSONDecodeError as e:
                time.sleep(0.5)
            except Exception as e:
                log.exception(f"Error loading jobs file: {self.file_path}")
                raise e

        # If we get here, we failed to load the file.
        # Move the corrupted file out of the way:
        log.exception(
            f"Corrupted jobs file: {self.file_path}, moving to {self.file_path}.{time.time()}"
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

        Arguments:
            job (UploadJob): The job to create.

        Returns:
            UploadJobID: The ID of the new job.

        """
        jobs = self._load_jobs()
        jobs[job.id] = job
        self._save_jobs(jobs)
        return job.id

    def get_job(self, job_id: UploadJobID) -> UploadJob:
        """
        Get a job by its ID.

        Arguments:
            job_id (UploadJobID): The ID of the job to get.

        Returns:
            UploadJob: The job.

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

        In all uses, you must pass the `job_id` argument with the job's unique
        identifier in the database. (It is not supported behavior to create a
        new job by passing a new job ID, but it works in this implementation.)
        There are two ways to use this function. The first is to pass in a Job
        object with updated fields. The second is to pass in a dictionary of
        ONLY the fields you want to update (not the whole job) under the
        `update` argument. If you pass both, the `update` operation will be run
        on the passed `job` object, which may or may not be what you want.

        Arguments:
            job_id (UploadJobID): The ID of the job to update.
            job (UploadJob, optional): The job to update. Defaults to None.
            update (dict, optional): A dictionary of fields to update. Defaults
                to None.

        Returns:
            UploadJobID: The ID of the updated job.

        """
        jobs = self._load_jobs()
        if job is None:
            job = jobs[job_id]
        if update is not None:
            for k, v in update.items():
                setattr(job, k, v)
        # Update the last_updated_at field:
        job.last_updated_at = datetime.datetime.now().isoformat()
        jobs[job_id] = job
        self._save_jobs(jobs)
        return job.id

    def has_job(self, job_id: UploadJobID) -> bool:
        """
        Check if a job exists.

        Arguments:
            job_id (UploadJobID): The ID of the job to check.

        Returns:
            bool: True if the job exists, False otherwise.

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

        Arguments:
            status (JobStatus): The status to filter by.

        Returns:
            List[UploadJob]: A list of jobs with the given status.

        """
        jobs = {k: v.to_dict() for k, v in self._load_jobs().items()}
        qry = {"status": f"{status}"}
        return [UploadJob.from_dict(u) for u in jque(list(jobs.values())).query(qry)]
