"""Single-user app lock helpers.

A minimal password gate (not multi-user auth): the app stores the SHA-256 hex of
the password in ``OSL_APP_PASSWORD_HASH`` and compares candidates in constant
time. When the hash is unset, the app is open.
"""

from __future__ import annotations

import hashlib
import hmac


def hash_password(password: str) -> str:
    """Return the SHA-256 hex digest of a password (for OSL_APP_PASSWORD_HASH)."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(candidate: str, password_hash: str) -> bool:
    """Constant-time check of a candidate password against a stored hash."""
    return hmac.compare_digest(hash_password(candidate), password_hash.strip().lower())
