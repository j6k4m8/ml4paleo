"""
Shared fixtures for the test suite.
"""

import pathlib
import random

import numpy as np
import pytest

S3_TEST_BUCKET = "ml4paleo-test"


@pytest.fixture(scope="session")
def s3_endpoint():
    """
    Run moto's S3 server in-process, so the S3 path is tested without Docker.
    """
    import boto3
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    ).create_bucket(Bucket=S3_TEST_BUCKET)
    yield endpoint
    server.stop()


@pytest.fixture
def make_dicom_series():
    """
    Return a function that writes a synthetic single-frame DICOM series.

    Slice `i` has pixel values `pixels(i)` (rows, columns), position
    `origin + i * spacing * normal`, and the given orientation. Files are
    named in shuffled order so tests can check that sorting uses geometry.
    """
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    def write(
        directory: pathlib.Path,
        pixels,
        *,
        count: int,
        orientation=(1, 0, 0, 0, 1, 0),
        pixel_spacing=(0.5, 0.25),
        slice_spacing: float = 2.0,
        origin=(0.0, 0.0, 10.0),
    ) -> list[pathlib.Path]:
        directory.mkdir(parents=True)
        normal = np.cross(orientation[:3], orientation[3:6])
        series_uid = generate_uid()
        names = [f"file_{n:03d}.dcm" for n in range(count)]
        random.Random(0).shuffle(names)
        paths = []
        for i in range(count):
            data = np.asarray(pixels(i), dtype=np.uint16)
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = CTImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = Dataset()
            ds.file_meta = meta
            ds.SOPClassUID = CTImageStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.SeriesInstanceUID = series_uid
            ds.Rows, ds.Columns = data.shape
            ds.PixelSpacing = list(pixel_spacing)
            ds.ImageOrientationPatient = list(orientation)
            ds.ImagePositionPatient = [
                float(o + i * slice_spacing * n)
                for o, n in zip(origin, normal, strict=True)
            ]
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.PixelData = data.tobytes()
            path = directory / names[i]
            ds.save_as(path, enforce_file_format=True)
            paths.append(path)
        return paths

    return write
