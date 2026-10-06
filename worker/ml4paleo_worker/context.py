"""
What a job handler gets: the job's payload, and ways to report progress and
notice cancellation.
"""

import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml4paleo.protocol import JobLease
from ml4paleo.storage import StorageGrant


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

    def __init__(
        self,
        lease: JobLease,
        memory_budget_bytes: int = 4 * 1024**3,
        threads: int = 1,
        *,
        v1_volume: Path | None = None,
        label_ops: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self.lease = lease
        # How much memory this job may use: the worker's memory shared among
        # its slots. Handlers size what they hold at once from it.
        self.memory_budget_bytes = memory_budget_bytes
        # How many CPU threads this job may use: the worker's CPUs shared
        # among its slots.
        self.threads = threads
        # The v1 app's volume folder, on workers started with --v1-volume.
        self.v1_volume = v1_volume
        self._label_ops = label_ops
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

    @property
    def grants(self) -> list[StorageGrant]:
        """
        The storage this job may use, in the order the job was given it.
        """
        return self.lease.grants

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

    def apply_label_op(self, op: dict[str, Any]) -> dict[str, Any]:
        """
        Apply a label edit (`ml4paleo.protocol.LabelOpIn`, without the lease
        token) to the job's project, through the server's label writer.
        """
        if self._label_ops is None:
            raise RuntimeError("This worker can't send label edits")
        self.check()
        return self._label_ops(op)

    def sleep(self, seconds: float) -> None:
        """
        Sleep, but wake up and raise `Cancelled` as soon as the job should stop.
        """
        if self._stop.wait(seconds):
            raise Cancelled(self.stop_reason)
