"""
The conversion runner is responsible for reading uploaded datasets and
converting them to chunked zarr arrays. Uploads are read from the upload
directory (`CONFIG.upload_directory`) and the zarr arrays are written to the
chunked directory (`CONFIG.chunked_directory`). The conversion runner polls
every few seconds to get new jobs to convert.

If a job is in the `JobStatus.UPLOADED` state, it will be queued for automatic
conversion. Depending on the job's `source_type`, the conversion runner will
read either an image stack or a DICOM series from the upload directory and use
`export_zarr_array` to write the result into the chunked directory. The
conversion runner will update the job status to `JobStatus.CONVERTING` when it
starts converting a job, and to `JobStatus.CONVERTED` when it finishes
converting a job. If the conversion job fails, the job status will be set to
`JobStatus.CONVERT_ERROR`.

"""

import logging
import pathlib
import shutil
import tempfile
import time
import zipfile
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from config import CONFIG
from job import DEFAULT_SOURCE_TYPE, JobStatus, JSONFileUploadJobManager, UploadJob

from ml4paleo.volume_providers import ImageStackVolumeProvider
from ml4paleo.volume_providers.io import export_zarr_array

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_job_manager() -> JSONFileUploadJobManager:
    """
    Return the job manager that will be used to read and write jobs.

    This is a convenience function so that we can easily change the job manager
    implementation without having to change the rest of the code.

    Arguments:
        None

    Returns:
        JobManager: The job manager that will be used to read and write jobs.

    """
    return JSONFileUploadJobManager("volume/jobs.json")


def get_next_uploaded_dataset_to_convert() -> Optional[UploadJob]:
    """
    Return the next dataset that has been uploaded but not yet converted.

    Arguments:
        None

    Returns:
        UploadJob: The next dataset that needs to be converted.
        None: If there are no datasets that need to be converted.

    """
    job_manager = get_job_manager()
    next_job = job_manager.get_jobs_by_status(JobStatus.UPLOADED)
    if len(next_job) == 0:
        return None
    return next_job[0]


def _list_uploaded_files(upload_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path for path in upload_dir.iterdir() if path.is_file())


def _should_ignore_source_file(path: pathlib.Path, root: Optional[pathlib.Path] = None) -> bool:
    if root is not None:
        relative_parts = path.resolve().relative_to(root.resolve()).parts
    else:
        relative_parts = path.parts
    if "__MACOSX" in relative_parts:
        return True
    if any(part.startswith(".") for part in relative_parts):
        return True
    if path.name.startswith("._"):
        return True
    return False


def _extract_zip_archive(archive_path: pathlib.Path, destination: pathlib.Path) -> list[pathlib.Path]:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    extracted_files: list[pathlib.Path] = []

    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            member_path = pathlib.Path(member.filename)
            if member_path.is_absolute():
                raise ValueError(f"Archive {archive_path.name} contains absolute paths.")

            target_path = (destination_root / member_path).resolve()
            if not target_path.is_relative_to(destination_root):
                raise ValueError(
                    f"Archive {archive_path.name} contains unsafe path traversal entries."
                )

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_files.append(target_path)

    extracted_source_files = [
        path for path in extracted_files if not _should_ignore_source_file(path, destination)
    ]
    if len(extracted_source_files) == 0:
        raise ValueError(f"Archive {archive_path.name} did not contain any usable files.")
    return sorted(extracted_source_files)


@contextmanager
def _prepare_upload_source_files(job_id: str) -> Iterator[list[pathlib.Path]]:
    upload_dir = pathlib.Path(CONFIG.upload_directory) / job_id
    uploaded_files = _list_uploaded_files(upload_dir)
    if len(uploaded_files) == 0:
        raise ValueError(f"No uploaded files found for job {job_id}.")

    with tempfile.TemporaryDirectory(prefix=f"ml4paleo_upload_{job_id}_") as tmpdir:
        staging_root = pathlib.Path(tmpdir)
        prepared_source_files: list[pathlib.Path] = []

        for upload_path in uploaded_files:
            if _should_ignore_source_file(upload_path, upload_dir):
                continue

            if zipfile.is_zipfile(upload_path):
                archive_target = staging_root / upload_path.stem
                prepared_source_files.extend(
                    _extract_zip_archive(upload_path, archive_target)
                )
            else:
                prepared_source_files.append(upload_path)

        prepared_source_files = sorted(prepared_source_files)
        if len(prepared_source_files) == 0:
            raise ValueError(f"No usable source files found for job {job_id}.")

        yield prepared_source_files


def _count_uploaded_dicom_files(upload_paths: list[pathlib.Path]) -> int:
    try:
        import pydicom
    except ImportError:
        return 0

    dicom_file_count = 0
    for path in upload_paths:
        try:
            pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            continue
        dicom_file_count += 1
    return dicom_file_count


def _get_volume_provider(job: UploadJob, source_files: list[pathlib.Path]):
    source_type = getattr(job, "source_type", DEFAULT_SOURCE_TYPE)
    dicom_file_count = _count_uploaded_dicom_files(source_files)
    if 0 < dicom_file_count < len(source_files):
        raise ValueError(
            f"Upload for job {job.id} contains a mix of DICOM and non-DICOM files. Upload one source type per job."
        )

    if source_type == "dicom" or dicom_file_count == len(source_files):
        try:
            from ml4paleo.volume_providers.dicomvp import DicomVolumeProvider
        except ImportError as exc:
            raise RuntimeError(
                "DICOM support is not installed. Install the dicom dependency group."
            ) from exc

        if source_type != "dicom":
            log.info(
                "Detected DICOM upload for job %s; overriding source_type=%s.",
                job.id,
                source_type,
            )
        return DicomVolumeProvider(source_files), "dicom"

    ordered_source_files = sorted(source_files, key=lambda path: (path.name, str(path)))
    return ImageStackVolumeProvider(ordered_source_files, cache_size=0), "image_stack"


def convert_next():
    """
    Convert the next dataset that has been uploaded but not yet converted.

    This is the function that actually performs the conversion.

    Arguments:
        None

    Returns:
        None

    """
    job_manager = get_job_manager()
    logging.info("Getting next dataset to convert...")
    next_job = get_next_uploaded_dataset_to_convert()
    if next_job is None:
        logging.info("No datasets to convert")
        return
    logging.info("Converting dataset %s", next_job.id)
    next_job.start_convert()
    job_manager.update_job(next_job.id, next_job)
    try:
        with _prepare_upload_source_files(next_job.id) as source_files:
            volume_provider, resolved_source_type = _get_volume_provider(
                next_job, source_files
            )
            if getattr(next_job, "source_type", DEFAULT_SOURCE_TYPE) != resolved_source_type:
                next_job.source_type = resolved_source_type
                job_manager.update_job(
                    next_job.id, update={"source_type": resolved_source_type}
                )

            logging.info(
                "Got %s dataset for job %s with shape %s from %d source files.",
                resolved_source_type,
                next_job.id,
                volume_provider.shape,
                len(source_files),
            )

            def _progress_callback(completed: int, item: Any, total: int) -> None:
                logging.info(f"Converted {completed} / {total} for job {next_job.id}.")
                job_mgr = get_job_manager()
                job_mgr.update_job(
                    next_job.id, update={"current_job_progress": completed / total}
                )

            export_zarr_array(
                volume_provider,
                pathlib.Path(CONFIG.chunked_directory) / next_job.id,
                chunk_size=CONFIG.chunk_size,
                progress_callback=_progress_callback,
                parallel_jobs=CONFIG.conversion_job_parallelism,
                # This needs to be set to the chunk size to avoid multiple jobs trying
                # to write to the same chunk in memory at once, which COULD result in
                # missing data.
                slice_count=CONFIG.chunk_size[2],
            )
    except Exception:
        log.exception("Conversion failed for job %s.", next_job.id)
        job_manager.update_job(next_job.id, update={"status": JobStatus.CONVERT_ERROR})
        return

    logging.info("Finished converting dataset %s", next_job.id)
    next_job.complete_convert()
    # Take only the last three dimensions of the shape, since the first
    # dimension is the channel dimension, which we don't want to include
    next_job.shape = volume_provider.shape[-3:]
    job_manager.update_job(next_job.id, next_job)
    logging.info("Updating job %s", next_job.id)


if __name__ == "__main__":
    while True:
        convert_next()
        time.sleep(CONFIG.job_poll_sec)
