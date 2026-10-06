"""
Check which compose services a container can reach, for the compose smoke
test. Run it in a container on the network under test:

    python check_network.py --reach api:8000 --no-reach postgres:5432 \
        --status http://caddy:8080/api/health=404
"""

import argparse
import socket
import sys
import urllib.error
import urllib.request


def reachable(address: str) -> bool:
    host, _, port = address.rpartition(":")
    try:
        socket.create_connection((host, int(port)), timeout=3).close()
    except OSError:
        return False
    return True


def status(url: str) -> int:
    request = urllib.request.Request(url, method="POST", data=b"{}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reach", action="append", default=[])
    parser.add_argument("--no-reach", action="append", default=[])
    # URL=CODE: a POST to URL must get HTTP status CODE.
    parser.add_argument("--status", action="append", default=[])
    args = parser.parse_args()
    problems = [f"cannot reach {a}" for a in args.reach if not reachable(a)]
    problems += [f"can reach {a}" for a in args.no_reach if reachable(a)]
    for check in args.status:
        url, _, expected = check.rpartition("=")
        if (got := status(url)) != int(expected):
            problems.append(f"POST {url} got {got}, expected {expected}")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
