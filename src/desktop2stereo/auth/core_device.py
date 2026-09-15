"""Device-bound key wrapping for protected core grants."""

from __future__ import annotations

import base64
from dataclasses import dataclass

from .storage import TokenStore


_STORE_FIELD = "protected_core_device_private_key"
_MEMORY_PRIVATE_KEY: bytes | None = None
_KEY_WRAP_INFO = b"d2s-parallax-core-v1-key-wrap"


class CoreDeviceKeyError(RuntimeError):
    """Raised when a device key cannot be generated or decoded."""


@dataclass(frozen=True)
class CoreDeviceKey:
    private_key: bytes
    public_key: bytes

    @property
    def public_key_b64(self) -> str:
        return base64.urlsafe_b64encode(self.public_key).rstrip(b"=").decode("ascii")


def _new_key() -> CoreDeviceKey:
    try:
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
        from cryptography.hazmat.primitives import serialization

        private = X25519PrivateKey.generate()
        return CoreDeviceKey(
            private.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()),
            private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw),
        )
    except Exception as exc:
        raise CoreDeviceKeyError("device core key is unavailable") from exc


def get_core_device_key() -> CoreDeviceKey:
    global _MEMORY_PRIVATE_KEY
    store = TokenStore()
    saved = store.load() or {}
    encoded = saved.get(_STORE_FIELD)
    private_bytes = None
    if isinstance(encoded, str):
        try:
            private_bytes = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        except (ValueError, TypeError):
            private_bytes = None
    if private_bytes is None:
        private_bytes = _MEMORY_PRIVATE_KEY
    if private_bytes is None or len(private_bytes) != 32:
        generated = _new_key()
        private_bytes = generated.private_key
        _MEMORY_PRIVATE_KEY = private_bytes
        saved[_STORE_FIELD] = base64.urlsafe_b64encode(private_bytes).rstrip(b"=").decode("ascii")
        # Online sessions may keep the key in process memory when no OS store exists.
        store.save(saved)
    try:
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
        from cryptography.hazmat.primitives import serialization

        private = X25519PrivateKey.from_private_bytes(private_bytes)
        return CoreDeviceKey(
            private_bytes,
            private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw),
        )
    except Exception as exc:
        raise CoreDeviceKeyError("device core key is invalid") from exc


def unwrap_core_key(claims: dict[str, object]) -> bytes:
    required = ("wrapped_core_key", "key_wrap_ephemeral_public_key", "key_wrap_nonce")
    if any(not isinstance(claims.get(name), str) for name in required):
        raise CoreDeviceKeyError("wrapped core key is missing")
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        key = get_core_device_key()
        private = X25519PrivateKey.from_private_bytes(key.private_key)
        ephemeral = X25519PublicKey.from_public_bytes(_decode(claims["key_wrap_ephemeral_public_key"]))
        shared = private.exchange(ephemeral)
        wrapping_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_KEY_WRAP_INFO).derive(shared)
        aad = _wrap_aad(claims)
        unwrapped = AESGCM(wrapping_key).decrypt(_decode(claims["key_wrap_nonce"]), _decode(claims["wrapped_core_key"]), aad)
        if len(unwrapped) != 32:
            raise CoreDeviceKeyError("wrapped core key is invalid")
        return unwrapped
    except CoreDeviceKeyError:
        raise
    except Exception as exc:
        raise CoreDeviceKeyError("wrapped core key cannot be opened") from exc


def _decode(value: object) -> bytes:
    if not isinstance(value, str) or not value or len(value) % 4 == 1:
        raise CoreDeviceKeyError("wrapped core key encoding is invalid")
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, TypeError) as exc:
        raise CoreDeviceKeyError("wrapped core key encoding is invalid") from exc


def _wrap_aad(claims: dict[str, object]) -> bytes:
    return "|".join(str(claims.get(name, "")) for name in ("grant_id", "license_id", "device_hash", "core_id", "core_version", "resource_sha256")).encode("ascii")
