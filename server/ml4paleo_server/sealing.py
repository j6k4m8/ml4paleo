"""
Values sealed with the server secret key.

Some rows hold values that would be dangerous in a database dump or backup:
TOTP secrets, and queued emails that carry single-use sign-in links. These are
stored encrypted (AES-GCM) with a key derived from the server secret key, one
key per purpose, so the database alone does not expose them. The secret key
lives outside the database (a Docker secret or `M4P_SECRET_KEY_FILE`).

After the secret key changes, older sealed values can't be opened
(`CannotUnseal`).
"""

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

NONCE_BYTES = 12


class CannotUnseal(Exception):
    """
    A value was sealed with a different server secret key, for a different
    purpose or context, or was altered.
    """


def _key(secret_key: str, purpose: str) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=f"ml4paleo {purpose}".encode(),
    ).derive(secret_key.encode())


def seal(secret_key: str, purpose: str, value: str, context: str = "") -> str:
    """
    Encrypt `value`. `context` (for example the recipient of an email) is
    authenticated but not stored: the value only unseals with the same context.
    """
    nonce = os.urandom(NONCE_BYTES)
    sealed = AESGCM(_key(secret_key, purpose)).encrypt(
        nonce, value.encode(), context.encode() or None
    )
    return base64.urlsafe_b64encode(nonce + sealed).decode()


def unseal(secret_key: str, purpose: str, sealed: str, context: str = "") -> str:
    raw = base64.urlsafe_b64decode(sealed)
    try:
        return (
            AESGCM(_key(secret_key, purpose))
            .decrypt(raw[:NONCE_BYTES], raw[NONCE_BYTES:], context.encode() or None)
            .decode()
        )
    except InvalidTag as exc:
        raise CannotUnseal from exc
