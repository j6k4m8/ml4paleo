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

import contextlib
import datetime
import abc
import logging
import os
import tempfile
from typing import Dict, Iterator, List, Optional
import json
import uuid
from enum import Enum
from marshmallow import Schema, fields
from jque import jque

try:
    import fcntl
except ImportError:  # Windows has no flock; writes are still atomic there.
    fcntl = None

log = logging.getLogger(__name__)

# Types:
UploadJobID = str
DEFAULT_SOURCE_TYPE = "image_stack"
ALLOWED_SOURCE_TYPES = ("image_stack", "dicom")


def normalize_source_type(source_type: Optional[str]) -> str:
    """
    Normalize user-provided upload source types into stable internal values.
    """
    if source_type is None:
        return DEFAULT_SOURCE_TYPE

    normalized = source_type.strip().lower()
    aliases = {
        "image": "image_stack",
        "images": "image_stack",
        "image-stack": "image_stack",
        "image_stack": "image_stack",
        "dicom": "dicom",
        "dicom-series": "dicom",
        "dicom_series": "dicom",
        "dcm": "dicom",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ALLOWED_SOURCE_TYPES:
        raise ValueError(
            f"Invalid source type: {source_type}. Must be one of {ALLOWED_SOURCE_TYPES}."
        )
    return normalized


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
    source_type = fields.Str()
    created_at = fields.Str()
    last_updated_at = fields.Str()
    current_job_progress = fields.Float()
    shape = fields.List(fields.Int(), allow_none=True)


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
        source_type: Optional[str] = None,
        created_at: Optional[str] = None,
        last_updated_at: Optional[str] = None,
        current_job_progress: Optional[float] = None,
        shape: Optional[List[int]] = None,
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
            shape (List[int]): The shape of the data in the job. If not
                provided, the shape will be set to None.

        """
        created_at = created_at or datetime.datetime.now().isoformat()
        last_updated_at = last_updated_at or datetime.datetime.now().isoformat()
        self.status = status or JobStatus.PENDING
        self.shape = shape
        self.source_type = normalize_source_type(source_type)
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
            source_type=d.get("source_type", DEFAULT_SOURCE_TYPE),
            created_at=d["created_at"],
            last_updated_at=d.get("last_updated_at", d["created_at"]),
            current_job_progress=d.get("current_job_progress", 0.0),
            shape=d.get("shape", None),
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

    The web app and all three job runners read and write the same file from
    separate processes, so two rules keep it consistent:

    1. Every write goes to a temporary file that then atomically replaces the
       jobs file, so readers always see a complete file.
    2. Every read-modify-write holds an exclusive `flock` on a sidecar lock
       file, so concurrent updates cannot overwrite each other.

    Reads do not take the lock. This reads from the file every time a job is
    requested, and writes the whole file every time a job is updated. This is
    not the most efficient way to do it, but it's the simplest, and it's not a
    problem for the small number of jobs we expect to have.

    """

    def __init__(self, file_path: str):
        """
        Create a new JSONFileUploadJobManager.

        Arguments:
            file_path (str): The path to the JSON file to use. If it does not
                exist, it will be created.

        """
        self.file_path = file_path
        self._lock_path = f"{file_path}.lock"
        with self._locked():
            if not os.path.exists(self.file_path):
                self._save_jobs({})

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """
        Hold the exclusive lock that guards every read-modify-write.

        `flock` locks belong to each open file, so this also serializes threads
        within one process (for example, parallel progress callbacks).
        """
        if fcntl is None:
            yield
            return
        with open(self._lock_path, "a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load_jobs(self) -> Dict[UploadJobID, UploadJob]:
        """
        Get all jobs from the file.

        Writes are atomic, so a file that fails to parse is really corrupted.
        In that case this raises rather than silently starting over with no
        jobs, which would let the next save wipe every job.
        """
        try:
            with open(self.file_path, "r") as f:
                raw_jobs = json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            log.exception(f"Corrupted jobs file: {self.file_path}")
            raise
        return {
            k: (UploadJob.from_dict(v) if isinstance(v, dict) else v)
            for k, v in raw_jobs.items()
        }

    def _save_jobs(self, jobs: Dict[UploadJobID, UploadJob]):
        """
        Atomically replace the jobs file with `jobs`.
        """
        jobs_jsonable = {
            k: (v if isinstance(v, dict) else v.to_dict()) for k, v in jobs.items()
        }
        directory = os.path.dirname(os.path.abspath(self.file_path))
        fd, tmp_path = tempfile.mkstemp(
            dir=directory, prefix=".jobs-", suffix=".json.tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(jobs_jsonable, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            # mkstemp creates the file as 0600; keep the existing permissions.
            try:
                os.chmod(tmp_path, os.stat(self.file_path).st_mode & 0o777)
            except FileNotFoundError:
                os.chmod(tmp_path, 0o644)
            os.replace(tmp_path, self.file_path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp_path)
            raise

    def new_job(self, job: UploadJob) -> UploadJobID:
        """
        Create a new job and return its ID.

        If the job's ID is already taken, the job gets a fresh ID instead of
        overwriting the existing job, so always use the returned ID.

        Arguments:
            job (UploadJob): The job to create.

        Returns:
            UploadJobID: The ID of the new job.

        """
        with self._locked():
            jobs = self._load_jobs()
            while job.id in jobs:
                job.id = _new_job_id()
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
        There are two ways to use this function. The preferred way is to pass
        a dictionary of ONLY the fields you want to update (not the whole job)
        under the `update` argument; those fields are applied to the latest
        stored copy of the job under the lock. The second is to pass in a Job
        object, which replaces the stored job entirely, including any fields
        another process changed since that object was read. If you pass both,
        the `update` operation will be run on the passed `job` object, which
        may or may not be what you want.

        Arguments:
            job_id (UploadJobID): The ID of the job to update.
            job (UploadJob, optional): The job to update. Defaults to None.
            update (dict, optional): A dictionary of fields to update. Defaults
                to None.

        Returns:
            UploadJobID: The ID of the updated job.

        """
        with self._locked():
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
