import base64
import hashlib
import json
import zlib
from pathlib import Path

import pytest

from desktop2stereo.auth.core import ProtectedCoreError, decrypt_core_resource, verify_core_grant


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _make_grant(tmp_path: Path, monkeypatch, *, device_hash: str = "a" * 64):
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, x25519
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from desktop2stereo.auth.core_device import CoreDeviceKey

    key = ec.generate_private_key(ec.SECP256R1())
    public = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    monkeypatch.setitem(__import__("desktop2stereo.auth.core", fromlist=["PUBLIC_KEYS"]).PUBLIC_KEYS, "test-core", public)
    core_key = bytes(range(32))
    device_private = x25519.X25519PrivateKey.generate()
    device_material = CoreDeviceKey(
        device_private.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()),
        device_private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw),
    )
    monkeypatch.setattr("desktop2stereo.auth.core_device.get_core_device_key", lambda: device_material)
    now = 2_000_600_000
    plaintext = b"protected parallax formula\n"
    metadata = {
        "format": "d2s-parallax-core-v1",
        "core_id": "parallax-core",
        "core_version": 1,
        "compression": "zlib",
        "cipher": "AES-256-GCM",
        "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
    }
    aad = json.dumps(metadata, separators=(",", ":")).encode("ascii")
    nonce = b"0123456789ab"
    encrypted = AESGCM(core_key).encrypt(nonce, zlib.compress(plaintext), aad)
    resource = tmp_path / "parallax-core.enc"
    resource.write_text(json.dumps({**metadata, "nonce": _b64(nonce), "ciphertext": _b64(encrypted)}), encoding="ascii")
    grant_id = "grant-1"
    wrap_ephemeral = x25519.X25519PrivateKey.generate()
    wrapping_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"d2s-parallax-core-v1-key-wrap").derive(
        wrap_ephemeral.exchange(device_private.public_key())
    )
    wrap_nonce = b"wrapnonce12"
    wrap_aad = "|".join((grant_id, "license-1", device_hash, "parallax-core", "1", hashlib.sha256(resource.read_bytes()).hexdigest())).encode("ascii")
    wrapped_core_key = AESGCM(wrapping_key).encrypt(wrap_nonce, core_key, wrap_aad)
    claims = {
        "version": 1, "key_id": "test-core", "grant_id": grant_id, "license_id": "license-1",
        "product": "desktop2stereo", "device_hash": device_hash, "core_id": "parallax-core",
        "core_version": 1, "resource_sha256": hashlib.sha256(resource.read_bytes()).hexdigest(),
        "wrapped_core_key": _b64(wrapped_core_key),
        "key_wrap_ephemeral_public_key": _b64(wrap_ephemeral.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)),
        "key_wrap_nonce": _b64(wrap_nonce),
        "issued_at": now - 10, "not_before": now - 60, "expires_at": now + 3600,
    }
    header = _b64(b'{"alg":"ES256","kid":"test-core","typ":"JWT"}')
    payload = _b64(json.dumps(claims, separators=(",", ":")).encode("ascii"))
    der = key.sign(f"{header}.{payload}".encode("ascii"), ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der)
    signature = _b64(r.to_bytes(32, "big") + s.to_bytes(32, "big"))
    return now, f"{header}.{payload}.{signature}", resource, plaintext


def test_verify_and_decrypt_protected_core(tmp_path: Path, monkeypatch) -> None:
    now, grant_jws, resource, plaintext = _make_grant(tmp_path, monkeypatch)
    grant = verify_core_grant(grant_jws, now=now, expected_device_hash="a" * 64)
    assert decrypt_core_resource(resource, grant) == plaintext


def test_protected_core_rejects_device_mismatch(tmp_path: Path, monkeypatch) -> None:
    now, grant_jws, _, _ = _make_grant(tmp_path, monkeypatch)
    with pytest.raises(ProtectedCoreError, match="device binding"):
        verify_core_grant(grant_jws, now=now, expected_device_hash="b" * 64)


def test_protected_core_rejects_resource_tampering(tmp_path: Path, monkeypatch) -> None:
    now, grant_jws, resource, _ = _make_grant(tmp_path, monkeypatch)
    grant = verify_core_grant(grant_jws, now=now, expected_device_hash="a" * 64)
    resource.write_text(resource.read_text(encoding="ascii").replace("parallax-core", "other-core"), encoding="ascii")
    with pytest.raises(ProtectedCoreError, match="resource hash"):
        decrypt_core_resource(resource, grant)
