"""
Check which compose services a container can reach, for the compose smoke
test. Run it in a container on the network under test:

    python check_network.py --reach api:8000 --no-reach postgres:5432
"""

import argparse
import socket
import sys


def reachable(address: str) -> bool:
    host, _, port = address.rpartition(":")
    try:
        socket.create_connection((host, int(port)), timeout=3).close()
    except OSError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reach", action="append", default=[])
    parser.add_argument("--no-reach", action="append", default=[])
    args = parser.parse_args()
    problems = [f"cannot reach {a}" for a in args.reach if not reachable(a)]
    problems += [f"can reach {a}" for a in args.no_reach if reachable(a)]
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
