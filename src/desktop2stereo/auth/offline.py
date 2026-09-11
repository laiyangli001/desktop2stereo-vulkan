"""Local storage and verification for server-signed offline entitlements."""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .public_keys import PUBLIC_KEYS


class OfflineEntitlementError(RuntimeError):
    pass


class OfflineEntitlementStorageError(OfflineEntitlementError):
    pass


def _decode_bytes(value: str) -> bytes:
    if not isinstance(value, str):
        raise binascii.Error("invalid unpadded Base64URL")
    if value == "":
        return b""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value) or len(value) % 4 == 1:
        raise binascii.Error("invalid unpadded Base64URL")
    padded = value + "=" * (-len(value) % 4)
    decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
    canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
    if canonical != value:
        raise binascii.Error("non-canonical Base64URL")
    return decoded


def _decode(value: str) -> str:
    return _decode_bytes(value).decode("utf-8")


class OfflineEntitlementStore:
    def __init__(self, app_dir: str | os.PathLike | None = None):
        root = Path(app_dir or Path.home() / ".desktop2stereo")
        self.path = root / "offline-entitlement.jws"

    def load(self) -> str | None:
        try:
            return self.path.read_text(encoding="ascii").strip() or None
        except OSError:
            return None

    def save(self, jws: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(jws, encoding="ascii")
        if os.name != "nt":
            temporary.chmod(0o600)
        os.replace(temporary, self.path)

    def clear(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def verify_entitlement(jws: str, *, now: int | None = None, expected_device_hash: str | None = None) -> dict[str, Any]:
    try:
        encoded_header, encoded_payload, encoded_signature = jws.split(".")
        header = json.loads(_decode(encoded_header))
        claims = json.loads(_decode(encoded_payload))
        signature = _decode_bytes(encoded_signature)
    except (ValueError, TypeError, UnicodeError, binascii.Error, json.JSONDecodeError) as exc:
        raise OfflineEntitlementError("离线授权凭证格式无效") from exc
    if (
        not isinstance(header, dict)
        or not isinstance(header.get("kid"), str)
        or not header.get("kid", "").strip()
        or header.get("alg") != "ES256"
        or header.get("typ") != "JWT"
    ):
        raise OfflineEntitlementError("离线授权签名算法无效")
    if not isinstance(claims, dict):
        raise OfflineEntitlementError("离线授权字段无效")
    key_data = PUBLIC_KEYS.get(header["kid"])
    if not key_data:
        raise OfflineEntitlementError("离线授权公钥不可用")
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

        if len(signature) != 64:
            raise OfflineEntitlementError("离线授权签名长度无效")
        der_signature = encode_dss_signature(int.from_bytes(signature[:32], "big"), int.from_bytes(signature[32:], "big"))
        public_key = serialization.load_pem_public_key(key_data)
        public_key.verify(der_signature, f"{encoded_header}.{encoded_payload}".encode("ascii"), ec.ECDSA(hashes.SHA256()))
    except OfflineEntitlementError:
        raise
    except Exception as exc:
        raise OfflineEntitlementError("离线授权签名验证失败") from exc
    if now is None:
        current = int(time.time())
    elif isinstance(now, bool) or not isinstance(now, int) or now <= 0:
        raise OfflineEntitlementError("离线授权时间参数无效")
    else:
        current = now
    required = {"version", "key_id", "entitlement_id", "license_id", "product", "device_hash", "mode", "features", "issued_at", "not_before", "expires_at", "trial", "offline_period_days"}
    string_fields = ("key_id", "entitlement_id", "license_id", "product", "device_hash", "mode")
    if (
        not required.issubset(claims)
        or isinstance(claims.get("version"), bool)
        or claims.get("version") != 1
        or any(not isinstance(claims.get(field), str) or not claims[field].strip() for field in string_fields)
        or claims.get("product") != "desktop2stereo"
        or claims.get("mode") not in {"offline", "permanent"}
        or claims.get("key_id") != header["kid"]
        or not isinstance(claims.get("features"), list)
        or any(not isinstance(feature, str) or not feature.strip() for feature in claims["features"])
        or not isinstance(claims.get("trial"), bool)
        or len(claims.get("device_hash", "")) != 64
        or any(character not in "0123456789abcdef" for character in claims.get("device_hash", ""))
    ):
        raise OfflineEntitlementError("离线授权字段无效")
    if expected_device_hash is not None and claims.get("device_hash") != expected_device_hash:
        raise OfflineEntitlementError("离线授权设备不匹配")
    not_before = claims["not_before"]
    expires_at = claims["expires_at"]
    issued_at = claims["issued_at"]
    offline_period_days = claims["offline_period_days"]
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in (issued_at, not_before, expires_at, offline_period_days))
        or issued_at <= 0
        or not_before <= 0
        or expires_at <= 0
        or offline_period_days < 0
        or not_before > expires_at
        or issued_at > expires_at
    ):
        raise OfflineEntitlementError("离线授权时间字段无效")
    if claims["mode"] == "offline" and offline_period_days not in {7, 14, 30}:
        raise OfflineEntitlementError("离线授权时长字段无效")
    if claims["mode"] == "permanent" and offline_period_days != 0:
        raise OfflineEntitlementError("永久授权时长字段无效")
    if not_before > current or expires_at <= current:
        raise OfflineEntitlementError("离线授权已过期或尚未生效")
    return claims


def persist_verified_entitlement(
    jws: str,
    *,
    expected_device_hash: str,
    store: OfflineEntitlementStore | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    """Verify a server entitlement before replacing the local cached copy."""

    claims = verify_entitlement(jws, now=now, expected_device_hash=expected_device_hash)
    try:
        (store or OfflineEntitlementStore()).save(jws)
    except OSError as exc:
        raise OfflineEntitlementStorageError("无法保存离线授权凭证") from exc
    return claims
