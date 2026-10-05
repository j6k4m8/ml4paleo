"""
Random tokens, their stored hashes, and the CSRF token tied to a session.

Session tokens, email tokens, and invite tokens are random strings given to
the user. The database stores only their SHA-256, so a database leak does not
leak usable tokens.
"""

import hashlib
import hmac
import secrets


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def csrf_token(secret_key: str, session_token: str) -> str:
    """
    The CSRF token for a session. Browsers send it in the `X-CSRF-Token`
    header; a cross-site page can't read it, so it can't forge requests.
    """
    return hmac.new(
        secret_key.encode(), b"csrf:" + session_token.encode(), hashlib.sha256
    ).hexdigest()


def tokens_match(expected: str, given: str | None) -> bool:
    # compare_digest refuses non-ASCII strings, and real tokens are hex.
    return (
        given is not None and given.isascii() and hmac.compare_digest(expected, given)
    )
