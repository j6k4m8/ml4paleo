"""
Job handlers, by job kind. A handler takes a `JobContext` and returns a small
JSON-able dict (the job's result); it raises to fail the job.

Raise `ml4paleo_worker.context.PermanentError` for failures that a retry
can't fix, such as a file that isn't an image.
"""

from collections.abc import Callable
from typing import Any

from ..context import JobContext
from . import noop

Handler = Callable[[JobContext], dict[str, Any]]

HANDLERS: dict[str, Handler] = {
    "noop": noop.run,
}
