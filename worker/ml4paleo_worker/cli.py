"""
Command-line entry point for `ml4paleo-worker`.

    ml4paleo-worker --server https://ml4paleo.example.org --token-file token

Every option can also come from an environment variable (`M4PW_SERVER`,
`M4PW_TOKEN_FILE`, `M4PW_SLOTS`, `M4PW_LABELS`).
"""

import argparse
import logging
import os
import pathlib
import signal
import sys
from urllib.parse import urlsplit

from ml4paleo.protocol import WORKER_TOKEN_PREFIX
from ml4paleo_worker import __version__

LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ml4paleo-worker", description="Run ml4paleo jobs for a server."
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("M4PW_SERVER"),
        help="The ml4paleo server's URL.",
    )
    parser.add_argument(
        "--token-file",
        type=pathlib.Path,
        default=os.environ.get("M4PW_TOKEN_FILE"),
        help="A file holding this worker's token (from the admin page).",
    )
    parser.add_argument(
        "--slots",
        type=int,
        default=int(os.environ.get("M4PW_SLOTS", "1")),
        help="How many jobs to run at once (default: 1).",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[s for s in os.environ.get("M4PW_LABELS", "").split(",") if s],
        help="A label jobs can ask for (repeatable). 'gpu' is added when a GPU "
        "is found.",
    )
    parser.add_argument(
        "--allow-http",
        action="store_true",
        default=os.environ.get("M4PW_ALLOW_HTTP") == "true",
        help="Allow a plain-HTTP server URL on a private network.",
    )
    args = parser.parse_args(argv)
    if not args.server or not args.token_file:
        parser.error("--server and --token-file are required")
    parts = urlsplit(args.server)
    if parts.scheme != "https" and not (
        parts.scheme == "http" and (args.allow_http or parts.hostname in LOCAL_HOSTS)
    ):
        parser.error(
            "the server URL must be https:// (the worker token would travel in "
            "the clear); pass --allow-http for a private network"
        )
    token = args.token_file.read_text().strip()
    if not token.startswith(WORKER_TOKEN_PREFIX):
        parser.error(f"{args.token_file} does not hold a worker token")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    from ml4paleo_worker.caps import detect
    from ml4paleo_worker.client import ServerClient
    from ml4paleo_worker.main import Worker

    client = ServerClient(token, args.server.rstrip("/"))
    worker = Worker(client, detect(args.label, args.slots))

    def stop(signum, frame):
        logging.getLogger(__name__).info("Stopping: giving running jobs back")
        worker.stop()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        worker.run()
    finally:
        client.close()
    return 1 if worker.unauthorized else 0


if __name__ == "__main__":
    sys.exit(main())
