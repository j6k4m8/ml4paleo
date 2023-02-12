from ml4paleo.volume_providers import ImageStackVolumeProvider
from config import CONFIG

import logging

log = logging.getLogger(__name__)


class ConversionRunner:
    def __init__(self, job_id: UploadJobID, job: UploadJob):
        self.job_id = job_id
        self.job = job

    def run(self):
        log.info("Starting conversion for job %s", self.job_id)
        self.job.status = JobStatus.CONVERTING
        self._convert()
        self.job.status = JobStatus.CONVERTED
        log.info("Conversion done for job %s", self.job_id)

    def _convert(self):
        log.info("Converting job %s", self.job_id)
