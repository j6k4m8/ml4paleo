"""
Password hashing and the password policy.

Hashes use argon2id (via pwdlib). Hashing is deliberately slow, so the async
helpers run it in a worker thread.
"""

import functools
import gzip
import pathlib

from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from starlette.concurrency import run_in_threadpool

_hasher = PasswordHash((Argon2Hasher(),))

# A fixed hash checked when a login names an unknown user, so a failed login
# takes the same time whether or not the account exists.
_DUMMY_HASH = _hasher.hash("not a real password; only used for timing")

MAX_PASSWORD_LENGTH = 256

# The 10,000 most common passwords, from SecLists (MIT license):
# https://github.com/danielmiessler/SecLists/blob/master/Passwords/Common-Credentials/10k-most-common.txt
_COMMON_PASSWORDS_PATH = pathlib.Path(__file__).with_name("common_passwords.txt.gz")


@functools.cache
def _common_passwords() -> frozenset[str]:
    with gzip.open(_COMMON_PASSWORDS_PATH, "rt", encoding="utf-8") as f:
        return frozenset(line.strip() for line in f if line.strip())


def password_problems(
    password: str,
    *,
    min_length: int,
    username: str | None = None,
    email: str | None = None,
) -> list[str]:
    """
    Return the reasons a password is not allowed (empty if it is fine).
    """
    problems = []
    if len(password) < min_length:
        problems.append(f"Use at least {min_length} characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        problems.append(f"Use at most {MAX_PASSWORD_LENGTH} characters.")
    lowered = password.lower()
    if lowered in _common_passwords() or len(set(password)) < 4:
        problems.append("This password is too common or too easy to guess.")
    personal = [username or ""]
    if email:
        personal.append(email.split("@")[0])
    if any(len(part) >= 3 and part.lower() in lowered for part in personal):
        problems.append("Don't include your username or email in your password.")
    return problems


async def hash_password(password: str) -> str:
    return await run_in_threadpool(_hasher.hash, password)


async def verify_password(password: str, password_hash: str | None) -> bool:
    """
    Check a password. Pass `None` for a missing account: the check still
    takes the usual time and then fails.
    """
    if password_hash is None:
        await run_in_threadpool(_hasher.verify, password, _DUMMY_HASH)
        return False
    return await run_in_threadpool(_hasher.verify, password, password_hash)
