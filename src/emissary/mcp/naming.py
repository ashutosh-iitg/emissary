"""Stable model-facing names for remote tools."""

import hashlib
import re

ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
HASH_LENGTH = 8


def default_alias(namespace: str, original_name: str) -> str:
    """`namespace__name` when that is already portable across providers.

    Otherwise a sanitized, truncated prefix plus a hash of the exact pair. The
    hash is what keeps two different originals that sanitize alike from
    colliding, and it is stable across runs and discovery order.
    """
    joined = f"{namespace}__{original_name}"
    if ALIAS_PATTERN.match(joined):
        return joined
    digest = hashlib.sha256(f"{namespace}\x00{original_name}".encode()).hexdigest()[:HASH_LENGTH]
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", joined)
    return f"{safe[: 64 - HASH_LENGTH - 1]}_{digest}"


def validate_alias(alias: str) -> str:
    if not ALIAS_PATTERN.match(alias):
        raise ValueError(f"alias {alias!r} must be 1-64 characters of letters, digits, '_' or '-'")
    return alias


__all__ = ["default_alias", "validate_alias"]
