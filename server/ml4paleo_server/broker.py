"""
The credential broker: turns a job's grants (`Job.grants`, paths under
project storage) into `StorageGrant`s for the worker that claimed it.

By default every grant points at the storage proxy
(`/api/worker/v1/jobs/<job>/storage/<n>`), authenticated by the job's lease
token: the worker reaches only that job's paths, with the access the job was
given, and only while it holds the lease. With `storage.worker_access =
"direct"`, workers on this machine get the server's own credentials instead
(faster, but they reach every project). Scoped temporary credentials for
cloud buckets (AWS STS, GCS downscoping) come later.
"""

from ml4paleo.storage import StorageGrant

from .db import Job, Worker
from .settings import Settings
from .storage import project_storage

PROXY_PREFIX = "/api/worker/v1/jobs"


def proxy_url(base_url: str, job: Job, index: int) -> str:
    return f"{base_url.rstrip('/')}{PROXY_PREFIX}/{job.id}/storage/{index}"


def grants_for(
    settings: Settings, worker: Worker, job: Job, lease_token: str, base_url: str
) -> list[StorageGrant]:
    """
    `base_url` is the server's address as the worker reached it.
    """
    direct = settings.storage.worker_access == "direct" and worker.pool == "local"
    root = project_storage(settings)
    grants = []
    for index, spec in enumerate(job.grants):
        if direct:
            grant = root.child(spec["path"]).model_copy(
                update={"access": spec["access"]}
            )
        else:
            grant = StorageGrant(
                url=proxy_url(base_url, job, index),
                access=spec["access"],  # type: ignore[arg-type]
                credentials={"token": lease_token},  # type: ignore[arg-type]
            )
        grants.append(grant)
    return grants
