"""
`noop`: a diagnostic job that waits, reports progress, and succeeds (or fails,
if asked), to check that workers pick up and finish work. With
`check_storage`, it also writes, reads back, and deletes an object in its
first grant, to check the worker's storage access.

Payload: `{"seconds": 1, "fail": false, "check_storage": false}`.
"""

import secrets
import time
from typing import Any

from ml4paleo.storage import delete_object, get_bytes, put_bytes

from ..context import JobContext

STEP_SECONDS = 0.1


def run(ctx: JobContext) -> dict[str, Any]:
    seconds = float(ctx.payload.get("seconds", 1))
    started = time.monotonic()
    while (elapsed := time.monotonic() - started) < seconds:
        ctx.progress(elapsed / seconds, f"{elapsed:.0f} of {seconds:.0f} s")
        ctx.sleep(min(STEP_SECONDS, seconds - elapsed))
    if ctx.payload.get("fail"):
        raise RuntimeError("The noop job was asked to fail.")
    result: dict[str, Any] = {"seconds": round(time.monotonic() - started, 3)}
    if ctx.payload.get("check_storage"):
        grant = ctx.grants[0]
        data = secrets.token_bytes(64)
        put_bytes(grant, "check", data)
        if get_bytes(grant, "check") != data:
            raise RuntimeError("Storage returned different bytes than were written.")
        delete_object(grant, "check")
        result["storage"] = "ok"
    return result
