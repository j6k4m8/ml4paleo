"""
The web app and the three job runners share `volume/jobs.json` from separate
processes, so concurrent updates must not overwrite each other and readers must
never see a partly written file.

Run from the repository root with `python -m unittest discover -s tests`.
"""

import json
import multiprocessing
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

WEBAPP_DIR = pathlib.Path(__file__).resolve().parents[1] / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

import job as job_module  # noqa: E402
from job import JobStatus, JSONFileUploadJobManager, UploadJob  # noqa: E402

UPDATES_PER_WRITER = 40


def _rename_repeatedly(file_path: str, job_id: str) -> None:
    manager = JSONFileUploadJobManager(file_path)
    for i in range(UPDATES_PER_WRITER):
        manager.update_job(job_id, update={"name": f"{job_id}-{i}"})


def _read_until_stopped(file_path: str, stop_event) -> None:
    manager = JSONFileUploadJobManager(file_path)
    while not stop_event.is_set():
        manager.get_jobs_by_status(JobStatus.UPLOADING)


class JobManagerTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self._tmpdir.name, "jobs.json")
        self.manager = JSONFileUploadJobManager(self.file_path)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_concurrent_updates_from_many_processes_are_not_lost(self):
        job_ids = [self.manager.new_job(UploadJob(name="start")) for _ in range(6)]

        ctx = multiprocessing.get_context("spawn")
        stop_event = ctx.Event()
        reader = ctx.Process(
            target=_read_until_stopped, args=(self.file_path, stop_event)
        )
        writers = [
            ctx.Process(target=_rename_repeatedly, args=(self.file_path, job_id))
            for job_id in job_ids
        ]
        reader.start()
        for writer in writers:
            writer.start()
        for writer in writers:
            writer.join(timeout=120)
        stop_event.set()
        reader.join(timeout=30)

        self.assertEqual([writer.exitcode for writer in writers], [0] * len(writers))
        self.assertEqual(reader.exitcode, 0, "a reader saw a partly written file")
        for job_id in job_ids:
            self.assertEqual(
                self.manager.get_job(job_id).name,
                f"{job_id}-{UPDATES_PER_WRITER - 1}",
            )

    def test_writes_leave_no_temporary_files(self):
        job_id = self.manager.new_job(UploadJob(name="start"))
        self.manager.update_job(job_id, update={"name": "renamed"})
        self.assertEqual(
            sorted(os.listdir(self._tmpdir.name)), ["jobs.json", "jobs.json.lock"]
        )

    def test_new_job_never_overwrites_an_existing_id(self):
        self.manager.new_job(UploadJob(id="AAAAAA", name="first"))
        with mock.patch.object(job_module, "_new_job_id", return_value="BBBBBB"):
            second_id = self.manager.new_job(UploadJob(id="AAAAAA", name="second"))

        self.assertEqual(second_id, "BBBBBB")
        self.assertEqual(self.manager.get_job("AAAAAA").name, "first")
        self.assertEqual(self.manager.get_job("BBBBBB").name, "second")

    def test_corrupted_file_raises_and_stays_in_place(self):
        pathlib.Path(self.file_path).write_text("{not json")
        with self.assertRaises(json.JSONDecodeError):
            self.manager.get_job("AAAAAA")
        self.assertEqual(pathlib.Path(self.file_path).read_text(), "{not json")


class ConversionRunnerTests(unittest.TestCase):
    def test_rename_during_conversion_is_kept(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        original_cwd = os.getcwd()
        os.chdir(tmpdir.name)
        self.addCleanup(os.chdir, original_cwd)
        pathlib.Path("volume").mkdir()

        import conversionrunner

        manager = conversionrunner.get_job_manager()
        job_id = manager.new_job(UploadJob(name="before", status=JobStatus.UPLOADED))
        upload_dir = pathlib.Path("volume", "uploads", job_id)
        upload_dir.mkdir(parents=True)
        for z in range(3):
            image = np.full((8, 10), z, dtype=np.uint8)
            Image.fromarray(image).save(upload_dir / f"{z:03d}.png")

        real_export = conversionrunner.export_zarr_array

        def rename_then_export(*args, **kwargs):
            manager.update_job(job_id, update={"name": "renamed mid-conversion"})
            return real_export(*args, **kwargs)

        with mock.patch.object(conversionrunner, "export_zarr_array", rename_then_export):
            conversionrunner.convert_next()

        converted = manager.get_job(job_id)
        self.assertEqual(converted.status, JobStatus.CONVERTED)
        self.assertEqual(converted.name, "renamed mid-conversion")
        self.assertEqual(list(converted.shape), [10, 8, 3])


if __name__ == "__main__":
    unittest.main()
