"""
CogniVara - Auth primitives
Password hashing/verification and opaque session tokens. Stdlib only.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets

_PBKDF2_ITERATIONS = 200_000


def hash_password(password: str) -> str:
    """PBKDF2-SHA256 hash for persisted credentials."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, _PBKDF2_ITERATIONS)
    return f'pbkdf2_sha256${salt.hex()}${digest.hex()}'


def verify_password(password: str, stored_hash: str | None) -> bool:
    """Recompute the PBKDF2 hash with the stored salt and compare in constant time."""
    if not stored_hash:
        return False
    try:
        scheme, salt_hex, digest_hex = stored_hash.split('$')
    except ValueError:
        return False
    if scheme != 'pbkdf2_sha256':
        return False

    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(digest_hex)
    actual = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, _PBKDF2_ITERATIONS)
    return hmac.compare_digest(actual, expected)


def generate_token() -> str:
    return secrets.token_urlsafe(32)
