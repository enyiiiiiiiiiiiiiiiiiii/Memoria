from __future__ import annotations

import hashlib
import hmac
import os
from typing import Optional

from . import db


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt_hex, digest_hex = stored.split("$", 1)
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(digest_hex)
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return hmac.compare_digest(candidate, expected)


def authenticate(username: str, password: str) -> Optional[dict]:
    user = db.fetch_one("SELECT * FROM users WHERE username = ?", (username,))
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return dict(user)


def register_user(username: str, password: str, full_name: str, role: str, email: str = "") -> int:
    password_hash = hash_password(password)
    return db.execute(
        """
        INSERT INTO users (username, password_hash, full_name, role, email)
        VALUES (?, ?, ?, ?, ?)
        """,
        (username, password_hash, full_name, role, email),
    )
