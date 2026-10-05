"""
Time-based one-time passwords (TOTP) for two-factor sign-in.

Secrets are stored encrypted (AES-GCM) with a key derived from the server's
secret key, so a database dump alone can't generate codes. Rotating the
server secret key therefore resets everyone's two-factor setup.
"""

import base64
import os

import pyotp
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

ISSUER = "ml4paleo"


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
    return AESGCM(_key(secret_key)).decrypt(raw[:12], raw[12:], None).decode()


def verify(secret: str, code: str) -> bool:
    """
    Accept the current code and the ones just before and after it, to allow
    for clock drift.
    """
    code = code.strip().replace(" ", "")
    return code.isdigit() and pyotp.TOTP(secret).verify(code, valid_window=1)
