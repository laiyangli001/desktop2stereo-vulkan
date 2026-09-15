"""Verification boundary for the protected parallax core resource."""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .public_keys import PUBLIC_KEYS


CORE_ID = "parallax-core"
CORE_VERSION = 1
RESOURCE_FORMAT = "d2s-parallax-core-v1"


class ProtectedCoreError(RuntimeError):
    """Raised when the protected core cannot be authenticated or decoded."""


@dataclass(frozen=True)
class CoreGrant:
    key_id: str
    grant_id: str
    license_id: str
    device_hash: str
    core_id: str
    core_version: int
    resource_sha256: str
    core_key: bytes
    issued_at: int
    not_before: int
    expires_at: int


def _decode_b64(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise ProtectedCoreError("protected core encoding is invalid")
    if any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for character in value):
        raise ProtectedCoreError("protected core encoding is invalid")
    if len(value) % 4 == 1:
        raise ProtectedCoreError("protected core encoding is invalid")
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, TypeError) as exc:
        raise ProtectedCoreError("protected core encoding is invalid") from exc
    if base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii") != value:
        raise ProtectedCoreError("protected core encoding is invalid")
    return decoded


def _load_json(value: bytes) -> dict[str, Any]:
    try:
        parsed = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtectedCoreError("protected core authorization is invalid") from exc
    if not isinstance(parsed, dict):
        raise ProtectedCoreError("protected core authorization is invalid")
    return parsed


def _verify_signature(header_part: str, payload_part: str, signature: bytes, key_id: str) -> None:
    key_data = PUBLIC_KEYS.get(key_id)
    if not key_data or len(signature) != 64:
        raise ProtectedCoreError("protected core authorization signature is invalid")
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

        public_key = serialization.load_pem_public_key(key_data)
        der_signature = encode_dss_signature(
            int.from_bytes(signature[:32], "big"), int.from_bytes(signature[32:], "big")
        )
        public_key.verify(
            der_signature,
            f"{header_part}.{payload_part}".encode("ascii"),
            ec.ECDSA(hashes.SHA256()),
        )
    except ProtectedCoreError:
        raise
    except Exception as exc:
        raise ProtectedCoreError("protected core authorization signature is invalid") from exc


def verify_core_grant(jws: str, *, now: int, expected_device_hash: str | None = None) -> CoreGrant:
    if isinstance(now, bool) or not isinstance(now, int) or now <= 0:
        raise ProtectedCoreError("protected core authorization time is invalid")
    if not isinstance(jws, str) or jws.count(".") != 2:
        raise ProtectedCoreError("protected core authorization format is invalid")
    header_part, payload_part, signature_part = jws.split(".")
    header = _load_json(_decode_b64(header_part))
    claims = _load_json(_decode_b64(payload_part))
    signature = _decode_b64(signature_part)
    key_id = header.get("kid")
    if header.get("alg") != "ES256" or header.get("typ") != "JWT" or not isinstance(key_id, str) or not key_id.strip():
        raise ProtectedCoreError("protected core authorization header is invalid")
    _verify_signature(header_part, payload_part, signature, key_id)
    required = {
        "version", "key_id", "grant_id", "license_id", "product", "device_hash",
        "core_id", "core_version", "resource_sha256", "wrapped_core_key",
        "key_wrap_ephemeral_public_key", "key_wrap_nonce", "issued_at",
        "not_before", "expires_at",
    }
    if "grant_id" not in claims and "entitlement_id" in claims:
        claims["grant_id"] = claims["entitlement_id"]
    if not required.issubset(claims) or claims.get("version") != 1 or claims.get("key_id") != key_id:
        raise ProtectedCoreError("protected core authorization fields are invalid")
    string_fields = (
        "key_id", "grant_id", "license_id", "product", "device_hash", "core_id", "resource_sha256",
        "wrapped_core_key", "key_wrap_ephemeral_public_key", "key_wrap_nonce",
    )
    if any(not isinstance(claims.get(field), str) or not claims[field].strip() for field in string_fields):
        raise ProtectedCoreError("protected core authorization fields are invalid")
    if claims["product"] != "desktop2stereo" or claims["core_id"] != CORE_ID or claims["core_version"] != CORE_VERSION:
        raise ProtectedCoreError("protected core version is unsupported")
    device_hash = claims["device_hash"]
    resource_hash = claims["resource_sha256"]
    if len(device_hash) != 64 or any(character not in "0123456789abcdef" for character in device_hash):
        raise ProtectedCoreError("protected core device binding is invalid")
    if expected_device_hash is not None and device_hash != expected_device_hash:
        raise ProtectedCoreError("protected core device binding does not match")
    if len(resource_hash) != 64 or any(character not in "0123456789abcdef" for character in resource_hash):
        raise ProtectedCoreError("protected core resource hash is invalid")
    try:
        from .core_device import unwrap_core_key

        core_key = unwrap_core_key(claims)
    except Exception as exc:
        raise ProtectedCoreError("protected core key is invalid") from exc
    times = (claims["issued_at"], claims["not_before"], claims["expires_at"])
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in times):
        raise ProtectedCoreError("protected core authorization time is invalid")
    if claims["issued_at"] > claims["expires_at"] or claims["not_before"] > claims["expires_at"]:
        raise ProtectedCoreError("protected core authorization time is invalid")
    if claims["not_before"] > now or claims["expires_at"] <= now:
        raise ProtectedCoreError("protected core authorization has expired")
    return CoreGrant(
        key_id=key_id,
        grant_id=claims["grant_id"],
        license_id=claims["license_id"],
        device_hash=device_hash,
        core_id=claims["core_id"],
        core_version=claims["core_version"],
        resource_sha256=resource_hash,
        core_key=core_key,
        issued_at=claims["issued_at"],
        not_before=claims["not_before"],
        expires_at=claims["expires_at"],
    )


class _NativeResourceInfo(ctypes.Structure):
    _fields_ = [
        ("core_version", ctypes.c_int),
        ("core_id", ctypes.c_char * 64),
        ("plaintext_sha256", ctypes.c_char * 65),
    ]


def _native_library_candidates(resource_path: Path, native_path: str | Path | None) -> list[Path]:
    if native_path is not None and str(native_path).strip():
        return [Path(native_path)]
    if sys.platform == "win32":
        names = ("d2s_protected_core.dll", "libd2s_protected_core.dll")
        platform_name = "windows"
    elif sys.platform == "darwin":
        names = ("libd2s_protected_core.dylib", "d2s_protected_core.dylib")
        platform_name = "macos"
    else:
        names = ("libd2s_protected_core.so", "d2s_protected_core.so")
        platform_name = "linux"
    roots = (resource_path.parent, resource_path.parent / "native" / platform_name)
    return [root / name for root in roots for name in names]


def _decrypt_with_native(resource_path: Path, grant: CoreGrant, native_path: str | Path | None) -> bytes:
    library_path = next((candidate for candidate in _native_library_candidates(resource_path, native_path) if candidate.is_file()), None)
    if library_path is None:
        raise ProtectedCoreError("protected core native module is unavailable")
    try:
        library = ctypes.WinDLL(str(library_path)) if sys.platform == "win32" else ctypes.CDLL(str(library_path))
        decrypt = library.d2s_core_decrypt_resource
        decrypt.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.POINTER(_NativeResourceInfo),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        decrypt.restype = ctypes.c_int
        release = library.d2s_core_free
        release.argtypes = [ctypes.c_void_p]
        release.restype = None
        key = (ctypes.c_ubyte * len(grant.core_key)).from_buffer_copy(grant.core_key)
        info = _NativeResourceInfo()
        plaintext_ptr = ctypes.c_void_p()
        plaintext_size = ctypes.c_size_t()
        resource_hash = hashlib.sha256(resource_path.read_bytes()).hexdigest().encode("ascii")
        result = decrypt(
            os.fsencode(resource_path),
            resource_hash,
            key,
            len(grant.core_key),
            ctypes.byref(info),
            ctypes.byref(plaintext_ptr),
            ctypes.byref(plaintext_size),
        )
        if result != 0 or not plaintext_ptr.value:
            raise ProtectedCoreError("protected core native decryption failed")
        try:
            plaintext = ctypes.string_at(plaintext_ptr, plaintext_size.value)
        finally:
            release(plaintext_ptr)
        if info.core_version != grant.core_version or info.core_id.rstrip(b"\0").decode("ascii") != grant.core_id:
            raise ProtectedCoreError("protected core native metadata does not match")
        if info.plaintext_sha256.rstrip(b"\0").decode("ascii") != hashlib.sha256(plaintext).hexdigest():
            raise ProtectedCoreError("protected core native plaintext hash does not match")
        return plaintext
    except ProtectedCoreError:
        raise
    except (OSError, ValueError, TypeError, UnicodeError, AttributeError) as exc:
        raise ProtectedCoreError("protected core native module is invalid") from exc


def load_core_module(
    resource_path: str | Path,
    grant_jws: str,
    *,
    now: int,
    expected_device_hash: str | None = None,
    native_path: str | Path | None = None,
    require_native: bool = False,
):
    """Decrypt and compile the protected formula in memory for runtime use."""

    import types

    grant = verify_core_grant(grant_jws, now=now, expected_device_hash=expected_device_hash)
    resource = Path(resource_path)
    if native_path is not None or require_native or os.environ.get("D2S_PARALLAX_CORE_NATIVE", "").strip():
        source = _decrypt_with_native(resource, grant, native_path or os.environ.get("D2S_PARALLAX_CORE_NATIVE"))
    else:
        source = decrypt_core_resource(resource, grant)
    try:
        code = compile(source, "<desktop2stereo-protected-parallax>", "exec")
    except (SyntaxError, TypeError) as exc:
        raise ProtectedCoreError("protected core code is invalid") from exc
    module = types.ModuleType("desktop2stereo_protected_parallax")
    module.__file__ = "<desktop2stereo-protected-parallax>"
    exec(code, module.__dict__)
    for name in ("resolve_parallax_budget", "parallax_debug_info"):
        if not callable(getattr(module, name, None)):
            raise ProtectedCoreError("protected core interface is incomplete")
    return grant, module


def decrypt_core_resource(resource_path: str | Path, grant: CoreGrant) -> bytes:
    """Verify and decrypt a resource without writing plaintext to disk."""

    path = Path(resource_path)
    try:
        resource = path.read_bytes()
        envelope = _load_json(resource)
    except OSError as exc:
        raise ProtectedCoreError("protected core resource is unavailable") from exc
    if hashlib.sha256(resource).hexdigest() != grant.resource_sha256:
        raise ProtectedCoreError("protected core resource hash does not match")
    required = {"format", "core_id", "core_version", "compression", "cipher", "plaintext_sha256", "nonce", "ciphertext"}
    if not required.issubset(envelope) or envelope["format"] != RESOURCE_FORMAT:
        raise ProtectedCoreError("protected core resource format is invalid")
    if envelope["core_id"] != grant.core_id or envelope["core_version"] != grant.core_version:
        raise ProtectedCoreError("protected core resource version does not match")
    if envelope["compression"] != "zlib" or envelope["cipher"] != "AES-256-GCM":
        raise ProtectedCoreError("protected core resource format is unsupported")
    metadata = {
        key: envelope[key]
        for key in ("format", "core_id", "core_version", "compression", "cipher", "plaintext_sha256")
    }
    aad = json.dumps(metadata, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        compressed = AESGCM(grant.core_key).decrypt(
            _decode_b64(envelope["nonce"]), _decode_b64(envelope["ciphertext"]), aad
        )
        plaintext = zlib.decompress(compressed)
    except ProtectedCoreError:
        raise
    except Exception as exc:
        raise ProtectedCoreError("protected core resource decryption failed") from exc
    if hashlib.sha256(plaintext).hexdigest() != envelope.get("plaintext_sha256"):
        raise ProtectedCoreError("protected core plaintext hash does not match")
    return plaintext
