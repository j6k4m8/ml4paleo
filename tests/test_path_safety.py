"""
Route values such as `job_id` and `seg_id` are joined into filesystem paths, so
values like ".." must never reach the view functions.

Run from the repository root with `python -m unittest discover -s tests`.
"""

import os
import pathlib
import sys
import tempfile
import unittest

WEBAPP_DIR = pathlib.Path(__file__).resolve().parents[1] / "webapp"


class PathSegmentRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_cwd = os.getcwd()
        cls._tmpdir = tempfile.TemporaryDirectory()
        # The web app keeps its data under a `volume/` directory relative to
        # the working directory, which is also the app root in production.
        os.chdir(cls._tmpdir.name)
        for subdir in ["chunks", "meshed", "segmented", "training"]:
            pathlib.Path("volume", subdir).mkdir(parents=True)
        pathlib.Path("sentinel.txt").write_text("SENTINEL")
        sys.path.insert(0, str(WEBAPP_DIR))
        import main

        main.app.root_path = cls._tmpdir.name
        cls.client = main.app.test_client()
        response = cls.client.post("/api/job/new", json={"name": "test job"})
        cls.job_id = response.get_json()["job_id"]

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._original_cwd)
        cls._tmpdir.cleanup()

    def test_valid_ids_reach_the_view(self):
        response = self.client.get(f"/api/job/{self.job_id}/status")
        self.assertEqual(response.status_code, 200)

        # The view answers 400 because no meshes exist yet; a rejected ID is 404.
        response = self.client.get(
            f"/api/job/{self.job_id}/meshes/1700000000.zarr/download"
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["message"], "meshes do not exist")

    def test_dot_dot_segments_are_rejected(self):
        urls = [
            # Without validation these return the full job list.
            "/job/../annotations/jobs.json",
            "/job/%2e%2e/annotations/jobs.json",
            "/api/job/%2e%2e/zarr/jobs.json",
            # Without validation these return any file in the app root.
            "/api/job/%2e%2e/segmentation/%2e%2e/zarr/sentinel.txt",
            "/api/job/%2e%2e/segmentation/%2e%2e/obj/sentinel.txt",
            # Without validation this zips every job's segmentations.
            f"/api/job/{self.job_id}/segmentation/%2e%2e/download/zarr",
            f"/api/job/{self.job_id}/meshes/%2e%2e/download",
            f"/api/job/{self.job_id}/models/%2e%2e/download",
            "/api/job/%2e%2e/status",
        ]
        for url in urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 404)
                self.assertNotIn(b"SENTINEL", response.data)
                self.assertNotIn(self.job_id.encode(), response.data)

    def test_other_unsafe_segments_are_rejected(self):
        for job_id in [".", ".hidden", "ABC%0A", "-rf", "a" * 200]:
            with self.subTest(job_id=job_id):
                response = self.client.get(f"/api/job/{job_id}/status")
                self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
