"""
Round trips through upload, PNG export, and meshing must keep the data intact
and in the same orientation as the source.

Run from the repository root with `python -m unittest discover -s tests`.
"""

import io
import json
import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import zarr
from PIL import Image
from stl import mesh as stl_mesh

WEBAPP_DIR = pathlib.Path(__file__).resolve().parents[1] / "webapp"

from ml4paleo.meshing import MESH_INFO_FILENAME, ChunkedMesher  # noqa: E402
from ml4paleo.volume_providers import (  # noqa: E402
    ImageStackVolumeProvider,
    ZarrVolumeProvider,
)
from ml4paleo.volume_providers.io import (  # noqa: E402
    export_to_img_stack,
    export_zarr_array,
)


class TempWorkdirTestCase(unittest.TestCase):
    """
    Run each test in an empty working directory with a `volume/` folder, the
    way the web app and runners expect.
    """

    def setUp(self):
        self._original_cwd = os.getcwd()
        self._tmpdir = tempfile.TemporaryDirectory()
        os.chdir(self._tmpdir.name)
        pathlib.Path("volume").mkdir()
        pathlib.Path("volume", "jobs.json").write_text("{}")

    def tearDown(self):
        os.chdir(self._original_cwd)
        self._tmpdir.cleanup()


class ChunkedUploadTests(TempWorkdirTestCase):
    CHUNK_SIZE = 4

    def setUp(self):
        super().setUp()
        sys.path.insert(0, str(WEBAPP_DIR))
        import main

        main.app.root_path = self._tmpdir.name
        self.client = main.app.test_client()
        response = self.client.post("/api/job/new", json={"name": "upload test"})
        self.job_id = response.get_json()["job_id"]

    def _send_chunk(self, payload: bytes, index: int, upload_uuid: str, name: str):
        chunk = payload[index * self.CHUNK_SIZE : (index + 1) * self.CHUNK_SIZE]
        total_chunks = -(-len(payload) // self.CHUNK_SIZE)
        return self.client.post(
            "/api/upload",
            headers={"X-Job-ID": self.job_id},
            data={
                "file": (io.BytesIO(chunk), name),
                "dzuuid": upload_uuid,
                "dzchunkindex": str(index),
                "dztotalchunkcount": str(total_chunks),
                "dzchunkbyteoffset": str(index * self.CHUNK_SIZE),
                "dztotalfilesize": str(len(payload)),
            },
            content_type="multipart/form-data",
        )

    def test_retried_chunks_do_not_corrupt_the_file(self):
        payload = b"AAAABBBBCCCCDD"
        upload_uuid = "0f8fad5b-d9cb-469f-a165-70867728950e"
        saved_path = pathlib.Path("volume", "uploads", self.job_id, "slice.png")

        # Chunk 0 and chunk 1 are each sent twice, as Dropzone does on retry.
        for index in [0, 0, 1, 1, 2]:
            response = self._send_chunk(payload, index, upload_uuid, "slice.png")
            self.assertEqual(response.status_code, 200, response.data)
            self.assertFalse(saved_path.exists(), "partial file is visible")

        response = self._send_chunk(payload, 3, upload_uuid, "slice.png")
        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(saved_path.read_bytes(), payload)

        # A retry of the final chunk after completion is accepted and harmless.
        response = self._send_chunk(payload, 3, upload_uuid, "slice.png")
        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(saved_path.read_bytes(), payload)
        self.assertEqual(os.listdir(saved_path.parent), ["slice.png"])

    def test_a_different_file_with_the_same_name_is_rejected(self):
        payload = b"AAAA"
        response = self._send_chunk(payload, 0, "aaaa-1111", "slice.png")
        self.assertEqual(response.status_code, 200, response.data)

        response = self._send_chunk(b"ZZZZ", 0, "bbbb-2222", "slice.png")
        self.assertEqual(response.status_code, 400)
        saved_path = pathlib.Path("volume", "uploads", self.job_id, "slice.png")
        self.assertEqual(saved_path.read_bytes(), payload)

    def test_invalid_upload_ids_are_rejected(self):
        response = self._send_chunk(b"AAAA", 0, "../../etc", "slice.png")
        self.assertEqual(response.status_code, 400)


class PngExportOrientationTests(TempWorkdirTestCase):
    def test_exported_slices_match_the_uploaded_images(self):
        # Non-square slices, so a transpose changes the shape as well as the
        # content.
        rng = np.random.default_rng(0)
        slices = [rng.integers(0, 255, size=(3, 5), dtype=np.uint8) for _ in range(2)]
        paths = []
        for z, image in enumerate(slices):
            path = pathlib.Path(f"upload_{z}.png")
            Image.fromarray(image).save(path)
            paths.append(path)

        export_zarr_array(
            ImageStackVolumeProvider(paths, cache_size=0), "volume.zarr"
        )
        export_to_img_stack(ZarrVolumeProvider("volume.zarr"), "exported", progress=False)

        for z, image in enumerate(slices):
            exported = np.array(Image.open(pathlib.Path("exported", f"{z:04d}.png")))
            np.testing.assert_array_equal(exported, image)


class DicomVoxelSizeTests(TempWorkdirTestCase):
    def _write_series(self, directory: pathlib.Path) -> list:
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

        directory.mkdir()
        series_uid = generate_uid()
        paths = []
        for z in range(3):
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = CTImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = Dataset()
            ds.file_meta = meta
            ds.SOPClassUID = CTImageStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.SeriesInstanceUID = series_uid
            ds.InstanceNumber = z + 1
            ds.Rows = 4
            ds.Columns = 6
            # (row spacing, column spacing): 0.5 mm along Y, 0.25 mm along X.
            ds.PixelSpacing = [0.5, 0.25]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = [0, 0, 10 + 2.0 * z]
            ds.SliceThickness = 1.0  # Spacing should come from the positions.
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.PixelData = np.full((4, 6), z, dtype=np.uint16).tobytes()
            path = directory / f"slice_{z}.dcm"
            ds.save_as(path, enforce_file_format=True)
            paths.append(path)
        return paths

    def test_voxel_size_is_read_and_recorded_in_the_zarr(self):
        from ml4paleo.volume_providers.dicomvp import DicomVolumeProvider

        provider = DicomVolumeProvider(self._write_series(pathlib.Path("series")))
        self.assertEqual(provider.shape, (6, 4, 3))
        self.assertEqual(provider.voxel_size_xyz_mm, (0.25, 0.5, 2.0))

        export_zarr_array(provider, "volume.zarr")
        self.assertEqual(
            ZarrVolumeProvider("volume.zarr").voxel_size_xyz_mm, (0.25, 0.5, 2.0)
        )


def _signed_volume(vectors: np.ndarray) -> float:
    a, b, c = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


class MeshOrientationTests(TempWorkdirTestCase):
    VOXEL_SIZE = (0.5, 2.0, 3.0)

    def _mesh_box(self, chunk_size) -> tuple:
        labels = np.zeros((20, 10, 8), dtype=np.uint64)
        labels[2:14, 3:7, 1:5] = 255  # 12 x 4 x 4 voxels, longest along X
        seg = zarr.open("seg.zarr", mode="w", shape=labels.shape, dtype="uint64")
        seg[:] = labels
        seg.attrs["voxel_size_xyz_mm"] = list(self.VOXEL_SIZE)

        ChunkedMesher(
            ZarrVolumeProvider("seg.zarr"), pathlib.Path("meshes"), chunk_size
        ).mesh_all(progress=False)
        combined = stl_mesh.Mesh.from_file("meshes/255.combined.stl")
        mesh_info = json.loads(pathlib.Path("meshes", MESH_INFO_FILENAME).read_text())
        return combined.vectors.reshape(-1, 3, 3), mesh_info

    def test_vertices_are_xyz_in_mm_across_chunks(self):
        vectors, mesh_info = self._mesh_box(chunk_size=(8, 8, 8))
        points = vectors.reshape(-1, 3)
        scale = np.array(self.VOXEL_SIZE)
        # Box faces sit half a voxel outside the labeled voxels. Mesh
        # simplification can move corners slightly, so allow one voxel.
        np.testing.assert_array_less(
            np.abs(points.min(axis=0) - np.array([1.5, 2.5, 0.5]) * scale), scale
        )
        np.testing.assert_array_less(
            np.abs(points.max(axis=0) - np.array([13.5, 6.5, 4.5]) * scale), scale
        )
        self.assertEqual(
            mesh_info,
            {"axis_order": "xyz", "units": "mm", "voxel_size_xyz": list(self.VOXEL_SIZE)},
        )

    def test_normals_point_outward(self):
        vectors, _ = self._mesh_box(chunk_size=(32, 32, 32))
        self.assertGreater(_signed_volume(vectors), 0)


class NeuroglancerMeshTransformTests(TempWorkdirTestCase):
    def setUp(self):
        super().setUp()
        sys.path.insert(0, str(WEBAPP_DIR))
        import apputils

        self.apputils = apputils

    def test_new_meshes_are_scaled_back_to_voxels(self):
        matrix = self.apputils.mesh_to_voxel_transform(
            {"axis_order": "xyz", "voxel_size_xyz": [0.5, 2.0, 4.0]}
        )
        self.assertEqual(matrix, [[2.0, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.25, 0]])

    def test_legacy_meshes_are_flagged_and_keep_the_axis_swap(self):
        legacy_dir = pathlib.Path("volume", "meshed", "ABC123", "1700000000.zarr")
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "255.combined.stl").write_bytes(b"")

        self.assertEqual(
            self.apputils.mesh_to_voxel_transform(self.apputils.load_mesh_info(legacy_dir)),
            [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
        )
        freshness = self.apputils.get_job_artifact_freshness("ABC123")
        self.assertTrue(freshness["mesh_axis_order_legacy"])
        self.assertTrue(freshness["mesh_stale"])

        (legacy_dir / MESH_INFO_FILENAME).write_text(
            json.dumps({"axis_order": "xyz", "voxel_size_xyz": [1, 1, 1]})
        )
        freshness = self.apputils.get_job_artifact_freshness("ABC123")
        self.assertFalse(freshness["mesh_axis_order_legacy"])


if __name__ == "__main__":
    unittest.main()
