"""
Uploads: files sent from browsers straight to object storage.

A browser asks the API to start an upload (the file's declared size is
reserved against the project owner's storage quota), then asks for presigned
URLs for the parts it still needs and PUTs each part to storage itself. The
URLs sign the part's exact length, so storage refuses a part of the wrong
size. To resume after a broken connection, the browser asks which parts are
already stored and sends the rest. Finally the API completes the upload,
after checking that every part is there with the expected size.

Uploads need S3-compatible storage (S3, SeaweedFS, R2, or GCS through its S3
API). On a single machine, Caddy lets browsers reach SeaweedFS for signed part
uploads only, on the app's own origin (`storage.public_endpoint`).

Garbage collection aborts uploads that aren't finished within a week, and
deletes finished ones after a month unless a job is still using them; both
give the quota back.
"""

import datetime
import logging
import math
import uuid
from functools import cache
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from sqlalchemy import and_, exists, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.concurrency import run_in_threadpool

from . import quotas
from .db import Job, Project, Upload
from .settings import Settings
from .storage import project_storage

log = logging.getLogger(__name__)

MiB = 1024**2
DEFAULT_PART_SIZE = 64 * MiB
MAX_PARTS = 10_000
# S3's limit for one object.
MAX_UPLOAD_BYTES = 5 * 1024**4
PART_URL_LIFETIME = datetime.timedelta(hours=1)
FINISH_WITHIN = datetime.timedelta(days=7)
KEEP_FINISHED = datetime.timedelta(days=30)
DATA_KEY = "data"
COLLECT_BATCH = 50


class UploadsUnsupported(Exception):
    """
    Project storage isn't S3-compatible, so browsers can't upload to it.
    """


def now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def upload_path(upload: Upload) -> str:
    return f"projects/{upload.project_id}/uploads/{upload.id}"


def part_size_for(size: int) -> int:
    return max(DEFAULT_PART_SIZE, math.ceil(size / MAX_PARTS / MiB) * MiB)


def part_count(upload: Upload) -> int:
    return math.ceil(upload.size / upload.part_size)


def expected_part_size(upload: Upload, number: int) -> int:
    if number < part_count(upload):
        return upload.part_size
    return upload.size - upload.part_size * (part_count(upload) - 1)


class MultipartStorage:
    """
    S3 multipart calls with the server's credentials. Every method blocks;
    call them through `run_in_threadpool`.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        endpoint: str | None,
        public_endpoint: str | None,
        region: str | None,
        access_key_id: str | None,
        secret_access_key: str | None,
    ):
        self.bucket = bucket
        self.prefix = prefix
        options: dict[str, Any] = {
            "aws_access_key_id": access_key_id,
            "aws_secret_access_key": secret_access_key,
            "region_name": region or "us-east-1",
            "config": Config(
                signature_version="s3v4",
                # S3-compatible services usually only support path-style URLs.
                s3={"addressing_style": "path" if endpoint else "auto"},
            ),
        }
        self._client = boto3.client("s3", endpoint_url=endpoint, **options)
        # Presigned URLs name the endpoint as browsers reach it.
        self._public = boto3.client(
            "s3", endpoint_url=public_endpoint or endpoint, **options
        )

    def key(self, upload: Upload) -> str:
        return "/".join(p for p in (self.prefix, upload_path(upload), DATA_KEY) if p)

    def start(self, upload: Upload) -> str:
        response = self._client.create_multipart_upload(
            Bucket=self.bucket,
            Key=self.key(upload),
            ContentType="application/octet-stream",
        )
        return response["UploadId"]

    def part_url(self, upload: Upload, number: int) -> str:
        return self._public.generate_presigned_url(
            "upload_part",
            Params={
                "Bucket": self.bucket,
                "Key": self.key(upload),
                "UploadId": upload.multipart_id,
                "PartNumber": number,
                # Signed, so storage refuses a part of any other length.
                "ContentLength": expected_part_size(upload, number),
            },
            ExpiresIn=int(PART_URL_LIFETIME.total_seconds()),
        )

    def stored_parts(self, upload: Upload) -> dict[int, tuple[int, str]]:
        """
        The parts storage has: {part number: (size, ETag)}.
        """
        parts: dict[int, tuple[int, str]] = {}
        marker = 0
        while True:
            response = self._client.list_parts(
                Bucket=self.bucket,
                Key=self.key(upload),
                UploadId=upload.multipart_id,
                PartNumberMarker=marker,
            )
            for part in response.get("Parts", []):
                parts[part["PartNumber"]] = (part["Size"], part["ETag"])
            if not response.get("IsTruncated"):
                return parts
            marker = response["NextPartNumberMarker"]

    def finish(self, upload: Upload, parts: dict[int, tuple[int, str]]) -> None:
        self._client.complete_multipart_upload(
            Bucket=self.bucket,
            Key=self.key(upload),
            UploadId=upload.multipart_id,
            MultipartUpload={
                "Parts": [
                    {"PartNumber": number, "ETag": parts[number][1]}
                    for number in sorted(parts)
                ]
            },
        )

    def stored_size(self, upload: Upload) -> int | None:
        try:
            response = self._client.head_object(
                Bucket=self.bucket, Key=self.key(upload)
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                return None
            raise
        return response["ContentLength"]

    def abort(self, upload: Upload) -> None:
        try:
            self._client.abort_multipart_upload(
                Bucket=self.bucket, Key=self.key(upload), UploadId=upload.multipart_id
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") != "NoSuchUpload":
                raise

    def delete(self, upload: Upload) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=self.key(upload))


def multipart_storage(settings: Settings) -> MultipartStorage:
    root = project_storage(settings)
    if root.scheme != "s3" or root.bucket is None:
        raise UploadsUnsupported
    storage = settings.storage
    return _cached_storage(
        root.bucket,
        root.path,
        storage.endpoint,
        storage.public_endpoint,
        storage.region,
        root.secret("access_key_id"),
        root.secret("secret_access_key"),
    )


@cache
def _cached_storage(*args) -> MultipartStorage:
    # Making boto3 clients is slow, and they are safe to share.
    return MultipartStorage(*args)


async def abort_or_delete(
    db: AsyncSession, settings: Settings, upload: Upload, state: str
) -> None:
    """
    Give an upload up (if unfinished) or delete it (if finished), and release
    its quota. The caller holds a lock on the row; this commits.
    """
    owner_id = await db.scalar(
        select(Project.owner_id).where(Project.id == upload.project_id)
    )
    storage = multipart_storage(settings)
    if upload.state == "uploading":
        await run_in_threadpool(storage.abort, upload)
        upload.state = "aborted"
        if owner_id is not None:
            await quotas.release_storage(db, owner_id, upload.size)
        await db.commit()
        return
    if upload.state == "complete":
        # Marked first, so no job can be given it while its file goes.
        upload.state = "deleting"
        if owner_id is not None:
            await quotas.release_storage(db, owner_id, upload.size)
        await db.commit()
    if upload.state == "deleting":
        await run_in_threadpool(storage.delete, upload)
        await db.execute(
            update(Upload)
            .where(Upload.id == upload.id, Upload.state == "deleting")
            .values(state=state)
        )
        await db.commit()


def in_use():
    """
    SQL: a waiting or running job has a grant for the upload's files.
    """
    path = func.concat("projects/", Upload.project_id, "/uploads/", Upload.id)
    return exists().where(
        Job.status.in_(("blocked", "queued", "leased")),
        Job.grants.contains(
            func.jsonb_build_array(func.jsonb_build_object("path", path))
        ),
    )


def _collectable():
    deleted_project = exists().where(
        Project.id == Upload.project_id, Project.deleted_at.is_not(None)
    )
    return and_(
        or_(
            Upload.state == "deleting",
            and_(
                Upload.state.in_(("uploading", "complete")),
                ~in_use(),
                or_(Upload.expires_at < now(), deleted_project),
            ),
        ),
    )


async def collect_garbage(
    sessionmaker: async_sessionmaker[AsyncSession], settings: Settings
) -> int:
    """
    Abort unfinished uploads past their deadline and delete finished ones
    past theirs (or in deleted projects). Returns how many were removed.
    """
    try:
        multipart_storage(settings)
    except UploadsUnsupported:
        return 0
    async with sessionmaker() as db:
        candidates = (
            await db.scalars(
                select(Upload.id)
                .where(_collectable())
                .order_by(Upload.expires_at)
                .limit(COLLECT_BATCH)
            )
        ).all()
    removed = 0
    for upload_id in candidates:
        try:
            async with sessionmaker() as db:
                upload = await db.scalar(
                    select(Upload)
                    .where(Upload.id == upload_id)
                    .with_for_update(key_share=True)
                    .execution_options(populate_existing=True)
                )
                if upload is None:
                    continue
                # Check again now that the row is locked.
                still = await db.scalar(
                    select(Upload.id).where(Upload.id == upload_id, _collectable())
                )
                if still is None:
                    continue
                await abort_or_delete(db, settings, upload, "deleted")
                removed += 1
        except Exception:
            log.exception("Could not remove upload %s; will try again", upload_id)
    return removed


def new_upload(
    project_id: uuid.UUID, created_by: uuid.UUID, filename: str, size: int
) -> Upload:
    return Upload(
        project_id=project_id,
        created_by=created_by,
        filename=filename,
        size=size,
        part_size=part_size_for(size),
        state="uploading",
        expires_at=now() + FINISH_WITHIN,
    )
