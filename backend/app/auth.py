from __future__ import annotations

import logging
import time
from typing import Any

import httpx
import jwt
from jwt.algorithms import RSAAlgorithm

logger = logging.getLogger(__name__)

_jwks_cache: dict[str, tuple[Any, float]] = {}
_JWKS_TTL = 300.0  # seconds


def _fetch_jwks(jwks_url: str) -> Any:
    """Fetch and cache JWKS from Clerk."""
    now = time.monotonic()
    cached = _jwks_cache.get(jwks_url)
    if cached and (now - cached[1]) < _JWKS_TTL:
        return cached[0]
    resp = httpx.get(jwks_url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    _jwks_cache[jwks_url] = (data, now)
    return data


def verify_clerk_token(token: str) -> dict[str, Any]:
    """
    Verify a Clerk JWT and return its decoded payload.
    Raises jwt.PyJWTError on failure.
    """
    # Decode header without verification to get kid
    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")

    # Decode without verification to read the issuer
    unverified_payload = jwt.decode(token, options={"verify_signature": False})
    issuer = unverified_payload.get("iss", "")

    if not issuer:
        raise jwt.InvalidTokenError("Missing issuer in token")

    jwks_url = issuer.rstrip("/") + "/.well-known/jwks.json"
    jwks = _fetch_jwks(jwks_url)

    # Find the matching key
    public_key = None
    for key_data in jwks.get("keys", []):
        if key_data.get("kid") == kid:
            public_key = RSAAlgorithm.from_jwk(key_data)
            break

    if public_key is None:
        # Try first key as fallback
        keys = jwks.get("keys", [])
        if keys:
            public_key = RSAAlgorithm.from_jwk(keys[0])

    if public_key is None:
        raise jwt.InvalidTokenError("No matching public key found in JWKS")

    payload = jwt.decode(
        token,
        public_key,
        algorithms=["RS256"],
        options={"verify_aud": False},
    )
    return payload
