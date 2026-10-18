"""
Command-line entry point for `ml4paleo-worker`.

    ml4paleo-worker --server https://ml4paleo.example.org --token-file token

Every option can also come from an environment variable (`M4PW_SERVER`,
`M4PW_TOKEN_FILE`, `M4PW_SLOTS`, `M4PW_LABELS`, `M4PW_V1_VOLUME`).
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
        default=int(os.environ.get("M4PW_SLOTS", "2")),
        help="How many jobs to run at once (default: 2).",
    )
    parser.add_argument(
        "--live-slots",
        type=int,
        default=int(os.environ.get("M4PW_LIVE_SLOTS", "1")),
        help="Slots reserved while Live is active (default: 1; always leaves an ordinary slot).",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=pathlib.Path,
        default=os.environ.get("M4PW_FEATURE_CACHE_DIR", "/tmp/ml4paleo-features"),
    )
    parser.add_argument(
        "--feature-cache-mb",
        type=int,
        default=int(os.environ.get("M4PW_FEATURE_CACHE_MB", "128")),
    )
    parser.add_argument(
        "--feature-cache-disk-mb",
        type=int,
        default=int(os.environ.get("M4PW_FEATURE_CACHE_DISK_MB", "2048")),
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[s for s in os.environ.get("M4PW_LABELS", "").split(",") if s],
        help="A label jobs can ask for (repeatable). 'gpu' is added when a GPU "
        "is found.",
    )
    parser.add_argument(
        "--v1-volume",
        type=pathlib.Path,
        default=os.environ.get("M4PW_V1_VOLUME") or None,
        help="The ml4paleo v1 app's volume folder (with jobs.json), to import "
        "v1 jobs from. Adds the 'v1-volume' label, and the worker then runs "
        "only the import's jobs. Mount it read-only.",
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
    if (
        args.slots < 1
        or min(args.live_slots, args.feature_cache_mb, args.feature_cache_disk_mb) < 0
    ):
        parser.error("Slots must be positive and cache/reservation limits nonnegative")
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
    from ml4paleo.segmentation.features import FeatureCache
    from ml4paleo_worker.caps import detect
    from ml4paleo_worker.client import ServerClient
    from ml4paleo_worker.handlers import HANDLERS, V1_HANDLERS
    from ml4paleo_worker.main import Worker

    if "v1-volume" in args.label and not args.v1_volume:
        parser.error("the v1-volume label needs --v1-volume")
    labels = [*args.label, *(["v1-volume"] if args.v1_volume else [])]
    if args.v1_volume and not (args.v1_volume / "jobs.json").is_file():
        parser.error(f"{args.v1_volume} has no jobs.json")
    handlers = V1_HANDLERS if args.v1_volume else HANDLERS
    client = ServerClient(token, args.server.rstrip("/"))
    caps = detect(labels, args.slots, kinds=handlers)
    cache = None
    try:
        cache = FeatureCache(
            args.feature_cache_dir,
            min(
                args.feature_cache_mb * 1024**2,
                int((caps.memory_gb or 4) * 1024**3 * 0.05),
            ),
            args.feature_cache_disk_mb * 1024**2,
        )
    except OSError as error:
        logging.getLogger(__name__).warning("Feature cache unavailable: %s", error)
    worker = Worker(
        client,
        caps,
        handlers=handlers,
        v1_volume=args.v1_volume,
        feature_cache=cache,
        live_slots=0 if args.v1_volume else args.live_slots,
    )

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
