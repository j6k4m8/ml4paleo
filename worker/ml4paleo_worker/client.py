"""
The worker's side of the worker protocol (`ml4paleo.protocol`).
"""

import time
import uuid
from collections.abc import Callable
from typing import Any

import httpx2

from ml4paleo.protocol import (
    MAX_CLAIM_WAIT_SECONDS,
    ClaimIn,
    ClaimOut,
    CompleteIn,
    FailIn,
    HeartbeatIn,
    HeartbeatOut,
    HelloIn,
    HelloOut,
    JobLease,
    ReleaseIn,
    WorkerCaps,
)

API_PREFIX = "/api/worker/v1"


class LeaseLost(Exception):
    """
    The server says this worker no longer holds the job (or the job was
    cancelled): throw its output away.
    """


class Unauthorized(Exception):
    """
    The server refused the worker token.
    """


class ServerClient:
    """
    Talks to the API server. Pass `http` to reuse an existing client (tests
    pass the app's test client); otherwise one is made for `base_url`.
    """

    def __init__(
        self,
        token: str,
        base_url: str | None = None,
        *,
        http: httpx2.Client | None = None,
    ):
        if http is None:
            if base_url is None:
                raise ValueError("Give base_url or http")
            # Long enough for a claim to wait for work.
            http = httpx2.Client(base_url=base_url, timeout=MAX_CLAIM_WAIT_SECONDS + 15)
        self._http = http
        self._headers = {"Authorization": f"Bearer {token}"}

    def close(self) -> None:
        self._http.close()

    def _post(self, path: str, body: Any) -> httpx2.Response:
        response = self._http.post(
            API_PREFIX + path,
            content=body.model_dump_json(),
            headers={**self._headers, "Content-Type": "application/json"},
        )
        if response.status_code == 401:
            raise Unauthorized(response.text)
        if response.status_code == 409:
            raise LeaseLost(response.text)
        response.raise_for_status()
        return response

    def hello(self, caps: WorkerCaps) -> HelloOut:
        return HelloOut.model_validate_json(
            self._post("/hello", HelloIn(caps=caps)).content
        )

    def claim(self, caps: WorkerCaps, wait_seconds: float) -> JobLease | None:
        response = self._post("/claim", ClaimIn(caps=caps, wait_seconds=wait_seconds))
        return ClaimOut.model_validate_json(response.content).job

    def heartbeat(
        self,
        job_id: uuid.UUID,
        lease_token: str,
        progress: float | None = None,
        message: str | None = None,
    ) -> HeartbeatOut:
        response = self._post(
            f"/jobs/{job_id}/heartbeat",
            HeartbeatIn(lease_token=lease_token, progress=progress, message=message),
        )
        return HeartbeatOut.model_validate_json(response.content)

    def complete(
        self, job_id: uuid.UUID, lease_token: str, result: dict[str, Any]
    ) -> None:
        self._post(
            f"/jobs/{job_id}/complete",
            CompleteIn(lease_token=lease_token, result=result),
        )

    def fail(
        self, job_id: uuid.UUID, lease_token: str, error: str, retryable: bool
    ) -> None:
        self._post(
            f"/jobs/{job_id}/fail",
            FailIn(lease_token=lease_token, error=error, retryable=retryable),
        )

    def release(self, job_id: uuid.UUID, lease_token: str) -> None:
        self._post(f"/jobs/{job_id}/release", ReleaseIn(lease_token=lease_token))


def with_retries[T](call: Callable[[], T], *, give_up_after: float) -> T:
    """
    Run `call`, retrying network and server errors with backoff for up to
    `give_up_after` seconds. `LeaseLost` and `Unauthorized` are not retried:
    retrying can't change them.
    """
    deadline = time.monotonic() + give_up_after
    delay = 1.0
    while True:
        try:
            return call()
        except (httpx2.TransportError, httpx2.HTTPStatusError) as exc:
            if (
                isinstance(exc, httpx2.HTTPStatusError)
                and exc.response.status_code < 500
            ):
                raise
            if time.monotonic() + delay > deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)
