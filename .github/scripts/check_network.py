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


def status(url: str, size: int = 0) -> int:
    """
    POST to `url` and return the HTTP status, or -1 if the server closed the
    connection before answering (as a proxy may for a body that is too big).
    """
    request = urllib.request.Request(url, method="POST", data=b"{}" + b" " * size)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except (urllib.error.URLError, ConnectionError):
        return -1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reach", action="append", default=[])
    parser.add_argument("--no-reach", action="append", default=[])
    # URL=CODE, or URL@BYTES=CODE: a POST to URL (with a body of about BYTES)
    # must get HTTP status CODE. CODE may list alternatives: 413|-1.
    parser.add_argument("--status", action="append", default=[])
    args = parser.parse_args()
    problems = [f"cannot reach {a}" for a in args.reach if not reachable(a)]
    problems += [f"can reach {a}" for a in args.no_reach if reachable(a)]
    for check in args.status:
        target, _, expected = check.rpartition("=")
        url, _, size = target.partition("@")
        if (got := status(url, int(size or 0))) not in {
            int(code) for code in expected.split("|")
        }:
            problems.append(f"POST {target} got {got}, expected {expected}")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
