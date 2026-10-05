"""
`noop`: a diagnostic job that waits, reports progress, and succeeds (or fails,
if asked), to check that workers pick up and finish work.

Payload: `{"seconds": 1, "fail": false}`.
"""

import time
from typing import Any

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
    return {"seconds": round(time.monotonic() - started, 3)}
