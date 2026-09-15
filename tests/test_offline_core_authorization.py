import base64
import json
import time

import pytest

from desktop2stereo.auth.offline import OfflineEntitlementError, verify_entitlement


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _signed_entitlement(monkeypatch, *, include_core: bool) -> str:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

    now = int(time.time())
    key = ec.generate_private_key(ec.SECP256R1())
    public = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    monkeypatch.setitem(__import__("desktop2stereo.auth.offline", fromlist=["PUBLIC_KEYS"]).PUBLIC_KEYS, "offline-core-test", public)
    claims = {
        "version": 1, "key_id": "offline-core-test", "entitlement_id": "ent-1", "license_id": "lic-1",
        "product": "desktop2stereo", "device_hash": "a" * 64, "mode": "offline", "features": ["runtime"],
        "issued_at": now, "not_before": now - 1, "expires_at": now + 3600, "trial": False, "offline_period_days": 7,
    }
    if include_core:
        claims.update({
            "core_id": "parallax-core", "core_version": 1, "resource_sha256": "b" * 64,
            "wrapped_core_key": "wrapped", "key_wrap_ephemeral_public_key": "ephemeral", "key_wrap_nonce": "nonce",
        })
    header = _encode(b'{"alg":"ES256","kid":"offline-core-test","typ":"JWT"}')
    payload = _encode(json.dumps(claims, separators=(",", ":")).encode("ascii"))
    der = key.sign(f"{header}.{payload}".encode("ascii"), ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der)
    return f"{header}.{payload}.{_encode(r.to_bytes(32, 'big') + s.to_bytes(32, 'big'))}"


def test_offline_verification_requires_protected_core_for_runtime(monkeypatch):
    pytest.importorskip("cryptography")
    entitlement = _signed_entitlement(monkeypatch, include_core=False)
    with pytest.raises(OfflineEntitlementError, match="核心字段无效"):
        verify_entitlement(entitlement, expected_device_hash="a" * 64, require_core=True)


def test_offline_verification_accepts_device_wrapped_core_fields(monkeypatch):
    pytest.importorskip("cryptography")
    entitlement = _signed_entitlement(monkeypatch, include_core=True)
    claims = verify_entitlement(entitlement, expected_device_hash="a" * 64, require_core=True)
    assert claims["core_id"] == "parallax-core"
