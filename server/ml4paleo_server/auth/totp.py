"""
Time-based one-time passwords (TOTP) for two-factor sign-in.

Secrets are sealed with the server's secret key (`ml4paleo_server.sealing`),
so a database dump alone can't generate codes. After the server
secret key changes, stored secrets can't be read: an admin must reset each
user's two-factor setup (`ml4paleo-server reset-two-factor USERNAME`).
"""

import hmac
import time

import pyotp

from .. import sealing

ISSUER = "ml4paleo"
STEP_SECONDS = 30


class UnreadableSecret(Exception):
    """
    A stored TOTP secret was encrypted with a different server secret key.
    """


def new_secret() -> str:
    return pyotp.random_base32()


def provisioning_uri(secret: str, username: str) -> str:
    return pyotp.TOTP(secret).provisioning_uri(name=username, issuer_name=ISSUER)


def encrypt(secret_key: str, secret: str) -> str:
    return sealing.seal(secret_key, "totp", secret)


def decrypt(secret_key: str, encrypted: str) -> str:
    try:
        return sealing.unseal(secret_key, "totp", encrypted)
    except sealing.CannotUnseal as exc:
        raise UnreadableSecret from exc


def verify(secret: str, code: str, *, after_step: int | None = None) -> int | None:
    """
    Check a code, allowing for one step of clock drift either way, and return
    the time step it belongs to (or None if it is wrong).

    Pass the step of the last accepted code as `after_step` so a code can't be
    used twice.
    """
    code = code.strip().replace(" ", "")
    if not (code.isdigit() and len(code) == 6):
        return None
    generator = pyotp.TOTP(secret)
    now = time.time()
    for offset in (-1, 0, 1):
        at = now + offset * STEP_SECONDS
        step = int(at // STEP_SECONDS)
        if after_step is not None and step <= after_step:
            continue
        if hmac.compare_digest(generator.at(int(at)), code):
            return step
    return None
