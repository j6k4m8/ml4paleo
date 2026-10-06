"""
The worker loop.

Each of the worker's slots claims a job (waiting up to 25 seconds for one),
runs its handler, and reports the outcome. While a handler runs, a heartbeat
thread renews the lease and passes on progress; if the server says the job
was cancelled, or the lease is lost, the handler is asked to stop.

Outcomes:

- The handler returns: report the job complete. If the server answers that
  the lease is gone, the output is thrown away (another worker has the job).
- The handler raises `PermanentError`: report a failure that won't be retried.
- The handler raises anything else: report a failure that may be retried.
- The worker is shutting down: give the job back (`release`), so another
  worker can take it at once without counting it as a failed attempt.
"""

import logging
import threading
import traceback
import uuid
from collections.abc import Mapping

import httpx2

from ml4paleo.protocol import (
    MAX_CLAIM_WAIT_SECONDS,
    MAX_ERROR_CHARS,
    CompleteIn,
    JobLease,
    WorkerCaps,
)

from .client import LeaseLost, ServerClient, Unauthorized, with_retries
from .context import Cancelled, JobContext, PermanentError
from .handlers import HANDLERS, Handler

log = logging.getLogger(__name__)

CANCELLED = "cancelled"
LEASE_LOST = "lease_lost"
SHUTDOWN = "shutdown"
MAX_CLAIM_BACKOFF_SECONDS = 60


class Worker:
    def __init__(
        self,
        client: ServerClient,
        caps: WorkerCaps,
        *,
        handlers: Mapping[str, Handler] = HANDLERS,
        claim_wait_seconds: float = MAX_CLAIM_WAIT_SECONDS,
        heartbeat_seconds: float | None = None,
    ):
        self.client = client
        self.caps = caps
        self.handlers = handlers
        self.claim_wait_seconds = claim_wait_seconds
        # The server suggests a heartbeat interval; tests use a shorter one.
        self._heartbeat_override = heartbeat_seconds
        self.heartbeat_seconds = heartbeat_seconds or 30.0
        self.lease_seconds = 120.0
        self.unauthorized = False
        self.jobs_done = 0
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._running: dict[uuid.UUID, JobContext] = {}
        self._max_jobs: int | None = None

    def run(self, max_jobs: int | None = None) -> None:
        """
        Work until `stop()` is called (or, for tests, until `max_jobs` jobs
        have been handled).
        """
        self._max_jobs = max_jobs
        try:
            hello = with_retries(
                lambda: self.client.hello(self.caps), give_up_after=300
            )
        except Unauthorized:
            log.error("The server refused this worker's token.")
            self.unauthorized = True
            return
        self.heartbeat_seconds = self._heartbeat_override or hello.heartbeat_seconds
        self.lease_seconds = hello.lease_seconds
        log.info(
            "Worker %r ready: %d slot(s), kinds %s, labels %s",
            hello.name,
            self.caps.slots,
            ", ".join(self.caps.kinds),
            ", ".join(self.caps.labels) or "none",
        )
        slots = [
            threading.Thread(target=self._slot, name=f"slot-{i}")
            for i in range(self.caps.slots)
        ]
        for slot in slots:
            slot.start()
        for slot in slots:
            slot.join()

    def stop(self) -> None:
        """
        Stop claiming, and hand running jobs back to the server.
        """
        self._stopping.set()
        with self._lock:
            running = list(self._running.values())
        for ctx in running:
            ctx.stop(SHUTDOWN)

    def _done(self) -> bool:
        with self._lock:
            return self._stopping.is_set() or (
                self._max_jobs is not None and self.jobs_done >= self._max_jobs
            )

    def _slot(self) -> None:
        failures = 0
        while not self._done():
            try:
                lease = self.client.claim(self.caps, self.claim_wait_seconds)
            except Unauthorized:
                log.error("The server refused this worker's token; stopping.")
                self.unauthorized = True
                self._stopping.set()
                return
            except (httpx2.TransportError, httpx2.HTTPStatusError) as exc:
                failures += 1
                delay = min(2**failures, MAX_CLAIM_BACKOFF_SECONDS)
                log.warning("Claim failed (%s); retrying in %d s", exc, delay)
                self._stopping.wait(delay)
                continue
            failures = 0
            if lease is not None:
                self.run_job(lease)
                with self._lock:
                    self.jobs_done += 1

    def run_job(self, lease: JobLease) -> None:
        ctx = JobContext(lease)
        with self._lock:
            self._running[lease.job_id] = ctx
        if self._stopping.is_set():
            ctx.stop(SHUTDOWN)
        log.info(
            "Running %s job %s (attempt %d)", lease.kind, lease.job_id, lease.attempt
        )
        finished = threading.Event()
        beats = threading.Thread(
            target=self._heartbeats,
            args=(ctx, finished),
            name=f"heartbeat-{lease.job_id}",
        )
        beats.start()
        try:
            self._run_handler(ctx)
        finally:
            finished.set()
            beats.join()
            with self._lock:
                self._running.pop(lease.job_id, None)

    def _run_handler(self, ctx: JobContext) -> None:
        lease = ctx.lease
        handler = self.handlers.get(lease.kind)
        try:
            if handler is None:
                raise PermanentError(f"This worker has no handler for {lease.kind!r}.")
            ctx.check()
            result = handler(ctx)
            try:
                # Check the result here, so a bad one fails the job at once
                # instead of leaving it to time out.
                CompleteIn(
                    lease_token=lease.lease_token, result=result
                ).model_dump_json()
            except Exception as exc:  # noqa: BLE001 - any invalid result
                raise PermanentError(f"The job's result can't be sent: {exc}") from exc
        except Cancelled:
            if ctx.stop_reason == SHUTDOWN:
                self._report(
                    ctx,
                    "release",
                    lambda: self.client.release(lease.job_id, lease.lease_token),
                )
            elif ctx.stop_reason == CANCELLED:
                self._report(
                    ctx,
                    "cancellation",
                    lambda: self.client.fail(
                        lease.job_id, lease.lease_token, "Cancelled.", retryable=False
                    ),
                )
            return
        except Exception as exc:  # noqa: BLE001 - any handler error fails the job
            permanent = isinstance(exc, PermanentError)
            log.warning("Job %s failed: %s", lease.job_id, exc)
            error = traceback.format_exc()[-MAX_ERROR_CHARS:]
            self._report(
                ctx,
                "failure",
                lambda: self.client.fail(
                    lease.job_id, lease.lease_token, error, retryable=not permanent
                ),
            )
            return
        rejected = self._report(
            ctx,
            "completion",
            lambda: self.client.complete(lease.job_id, lease.lease_token, result),
        )
        if rejected is not None:
            # The server refused the result itself; retrying won't help.
            self._report(
                ctx,
                "failure",
                lambda: self.client.fail(
                    lease.job_id,
                    lease.lease_token,
                    f"The server rejected the job's result: {rejected}",
                    retryable=False,
                ),
            )

    def _report(self, ctx: JobContext, what: str, call) -> str | None:
        """
        Send a report, retrying network and server errors. Returns the
        server's answer if it refused the request (HTTP 4xx), else None.
        """
        if ctx.stop_reason == LEASE_LOST:
            log.warning("Lost the lease on job %s; discarding its output", ctx.job_id)
            return None
        try:
            with_retries(call, give_up_after=self.lease_seconds)
        except LeaseLost:
            log.warning("Job %s is no longer ours; discarded its %s", ctx.job_id, what)
        except httpx2.HTTPStatusError as exc:
            log.error("The server refused the %s of job %s: %s", what, ctx.job_id, exc)
            if exc.response.status_code < 500:
                return f"{exc.response.status_code} {exc.response.text[:500]}"
        except Exception as exc:  # noqa: BLE001 - the server will reassign it
            log.error("Could not report the %s of job %s: %s", what, ctx.job_id, exc)
        else:
            log.info("Reported the %s of job %s", what, ctx.job_id)
        return None

    def _heartbeats(self, ctx: JobContext, finished: threading.Event) -> None:
        lease = ctx.lease
        while not finished.wait(self.heartbeat_seconds):
            progress, message = ctx.take_progress()
            try:
                beat = self.client.heartbeat(
                    lease.job_id, lease.lease_token, progress, message
                )
            except (LeaseLost, Unauthorized):
                ctx.stop(LEASE_LOST)
                return
            except (httpx2.TransportError, httpx2.HTTPStatusError) as exc:
                # The lease may still be good; try again next time.
                log.warning("Heartbeat for job %s failed: %s", lease.job_id, exc)
                continue
            if beat.cancel:
                ctx.stop(CANCELLED)
