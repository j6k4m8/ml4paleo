"""
Command-line entry point for `ml4paleo-worker`.
"""

import argparse

from ml4paleo_worker import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ml4paleo-worker")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
