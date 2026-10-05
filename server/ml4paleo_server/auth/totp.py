"""
Time-based one-time passwords (TOTP) for two-factor sign-in.

Secrets are stored encrypted (AES-GCM) with a key derived from the server's
secret key, so a database dump alone can't generate codes. After the server
secret key changes, stored secrets can't be read: an admin must reset each
user's two-factor setup (`ml4paleo-server reset-two-factor USERNAME`).
"""

import base64
import hmac
import os
import time

import pyotp
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

ISSUER = "ml4paleo"
STEP_SECONDS = 30


class UnreadableSecret(Exception):
    """
    A stored TOTP secret was encrypted with a different server secret key.
    """


def _key(secret_key: str) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None, info=b"ml4paleo totp"
    ).derive(secret_key.encode())


def new_secret() -> str:
    return pyotp.random_base32()


def provisioning_uri(secret: str, username: str) -> str:
    return pyotp.TOTP(secret).provisioning_uri(name=username, issuer_name=ISSUER)


def encrypt(secret_key: str, secret: str) -> str:
    nonce = os.urandom(12)
    sealed = AESGCM(_key(secret_key)).encrypt(nonce, secret.encode(), None)
    return base64.urlsafe_b64encode(nonce + sealed).decode()


def decrypt(secret_key: str, encrypted: str) -> str:
    raw = base64.urlsafe_b64decode(encrypted)
    try:
        return AESGCM(_key(secret_key)).decrypt(raw[:12], raw[12:], None).decode()
    except InvalidTag as exc:
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
