"""
What a job handler gets: the job's payload, and ways to report progress and
notice cancellation.
"""

import threading
import uuid
from typing import Any

from ml4paleo.protocol import JobLease


class Cancelled(Exception):
    """
    Raised by `JobContext.check` when the job should stop.
    """


class PermanentError(Exception):
    """
    Raise from a handler when the job can't succeed however often it is tried.
    """


class JobContext:
    """
    Handlers call `progress()` as they go, and `check()` (or look at
    `stopping`) often enough to stop within a few seconds when the job is
    cancelled, its lease is lost, or the worker is shutting down.
    """

    def __init__(self, lease: JobLease):
        self.lease = lease
        self._progress: float | None = None
        self._message: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.stop_reason: str | None = None

    @property
    def job_id(self) -> uuid.UUID:
        return self.lease.job_id

    @property
    def payload(self) -> dict[str, Any]:
        return self.lease.payload

    def progress(self, fraction: float, message: str | None = None) -> None:
        with self._lock:
            self._progress = min(1.0, max(0.0, fraction))
            if message is not None:
                self._message = message[:200]

    def take_progress(self) -> tuple[float | None, str | None]:
        with self._lock:
            return self._progress, self._message

    def stop(self, reason: str) -> None:
        """
        Ask the handler to stop. The first reason given wins.
        """
        with self._lock:
            if self.stop_reason is None:
                self.stop_reason = reason
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def check(self) -> None:
        if self._stop.is_set():
            raise Cancelled(self.stop_reason)

    def sleep(self, seconds: float) -> None:
        """
        Sleep, but wake up and raise `Cancelled` as soon as the job should stop.
        """
        if self._stop.wait(seconds):
            raise Cancelled(self.stop_reason)
