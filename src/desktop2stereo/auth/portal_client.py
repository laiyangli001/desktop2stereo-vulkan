"""PHP portal v2 transport and signed 24-hour credential verification.

The launcher integration is intentionally separate from the legacy v1 client.
Passwords and bearer tokens are never written by this transport.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from http.client import HTTPException
import json
import os
import re
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class PortalError(RuntimeError):
    def __init__(self, code: str, *, status: int | None = None, clear_credentials: bool = False):
        super().__init__(code)
        self.code = code
        self.status_code = status
        self.clear_credentials = clear_credentials

    @property
    def retryable(self) -> bool:
        return self.code == "network_error" or self.status_code == 429 or (self.status_code or 0) >= 500


@dataclass(frozen=True)
class PortalSession:
    access_token: str
    user_id: int
    license: dict[str, Any]
    device_hash: str
    session_version: int
    offline_token: str
    core_grant: str
    server_time: int
    offline_expires_at: int
    heartbeat_seconds: int = 900


def _decode(value: object, length: int | None = None) -> bytes:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("invalid base64url")
    decoded = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    if base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii") != value:
        raise ValueError("noncanonical base64url")
    if length is not None and len(decoded) != length:
        raise ValueError("invalid length")
    return decoded


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _integer(value: object, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("invalid integer")
    return value


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class PortalClient:
    def __init__(
        self,
        device_hash: str,
        device_public_key: str,
        trusted_keys: Mapping[str, bytes],
        *,
        base_url: str | None = None,
        timeout: float = 8.0,
    ):
        if not re.fullmatch(r"[a-f0-9]{64}", device_hash):
            raise ValueError("invalid device hash")
        _decode(device_public_key, 32)
        base = (base_url or os.environ.get("D2S_PORTAL_API_URL") or "https://d2s.site/api/v2").rstrip("/")
        parsed = urlsplit(base)
        local = parsed.hostname in {"127.0.0.1", "::1"}
        if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path != "/api/v2":
            raise ValueError("invalid portal URL")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
            raise ValueError("HTTPS portal required")
        if not parsed.hostname or not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 60:
            raise ValueError("invalid portal configuration")
        self.base_url = base
        self.timeout = timeout
        self.device_hash = device_hash
        self.device_public_key = device_public_key
        self.trusted_keys = dict(trusted_keys)
        # Ignore proxy environment variables for local fixture servers.
        handlers = [_NoRedirect()]
        if local:
            handlers.append(ProxyHandler({}))
        self._opener = build_opener(*handlers)

    def login(self, identifier: str, password: str) -> PortalSession:
        payload = self._request("/client/login", {
            "identifier": identifier, "password": password,
            "device_hash": self.device_hash, "device_public_key": self.device_public_key,
        })
        return self.accept(payload)

    def heartbeat(self, session: PortalSession) -> PortalSession:
        if session.device_hash != self.device_hash:
            raise PortalError("device_mismatch", clear_credentials=True)
        payload = self._request("/client/heartbeat", {"device_hash": self.device_hash}, session.access_token)
        updated = self.accept(payload, token=session.access_token, user_id=session.user_id)
        if updated.license["id"] != session.license["id"] or updated.session_version != session.session_version:
            raise PortalError("session_binding_invalid", clear_credentials=True)
        return updated

    def logout(self, session: PortalSession) -> None:
        result = self._request("/client/logout", {"device_hash": self.device_hash}, session.access_token)
        if result.get("logged_out") is not True:
            raise PortalError("invalid_response", clear_credentials=True)

    def _request(self, path: str, payload: dict, token: str | None = None) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if token is not None:
            if not re.fullmatch(r"[a-f0-9]{64}", token):
                raise PortalError("client_session_invalid", clear_credentials=True)
            headers["Authorization"] = "Bearer " + token
        request = Request(self.base_url + path, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            response = self._opener.open(request, timeout=self.timeout)
        except HTTPError as error:
            try:
                body = json.loads(error.read(262145), object_pairs_hook=_object)
            except (ValueError, UnicodeError, OSError, HTTPException):
                body = {}
            finally:
                error.close()
            code = body.get("error") if isinstance(body, dict) else None
            if not isinstance(code, str) or not re.fullmatch(r"[a-z_]{1,80}", code):
                code = "http_error"
            # A denial can never be turned into an offline fallback by a response flag.
            raise PortalError(code, status=error.code, clear_credentials=error.code in {401, 403, 426}) from None
        except (URLError, TimeoutError, OSError):
            raise PortalError("network_error") from None
        try:
            with response:
                raw = response.read(262145)
            if len(raw) > 262144:
                raise ValueError("response too large")
            body = json.loads(raw, object_pairs_hook=_object)
            if not isinstance(body, dict) or body.get("success") is not True or not isinstance(body.get("data"), dict):
                raise ValueError("invalid response")
            return body["data"]
        except (OSError, HTTPException):
            raise PortalError("network_error") from None
        except (ValueError, UnicodeError):
            raise PortalError("invalid_response", clear_credentials=True) from None

    def _claims(self, token: object, now: int) -> dict:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

        if not isinstance(token, str) or len(token) > 32768:
            raise ValueError("invalid credential")
        head, payload, signature = token.split(".")
        header = json.loads(_decode(head), object_pairs_hook=_object)
        claims = json.loads(_decode(payload), object_pairs_hook=_object)
        if not isinstance(header, dict) or not isinstance(claims, dict):
            raise ValueError("invalid claims")
        kid = header.get("kid")
        if header.get("alg") != "ES256" or header.get("typ") != "JWT" or kid not in self.trusted_keys:
            raise ValueError("untrusted signing key")
        public = serialization.load_pem_public_key(self.trusted_keys[kid])
        if not isinstance(public, ec.EllipticCurvePublicKey) or not isinstance(public.curve, ec.SECP256R1):
            raise ValueError("invalid public key")
        raw = _decode(signature, 64)
        public.verify(
            encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")),
            (head + "." + payload).encode("ascii"), ec.ECDSA(hashes.SHA256()),
        )
        issued, start, end = (_integer(claims.get(field)) for field in ("issued_at", "not_before", "expires_at"))
        if (claims.get("key_id") != kid or _integer(claims.get("version")) != 1 or
                claims.get("product") != "desktop2stereo" or claims.get("device_hash") != self.device_hash or
                issued > now + 60 or start > now or end <= now or end <= issued or start > issued or end > issued + 86400):
            raise ValueError("credential binding or expiry invalid")
        if claims.get("core_id") != "parallax-core" or _integer(claims.get("core_version")) != 1:
            raise ValueError("invalid core")
        if not isinstance(claims.get("resource_sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", claims["resource_sha256"]):
            raise ValueError("invalid resource digest")
        _integer(claims.get("session_version"))
        _decode(claims.get("key_wrap_ephemeral_public_key"), 32)
        _decode(claims.get("key_wrap_nonce"), 12)
        _decode(claims.get("wrapped_core_key"), 48)
        return claims

    def accept(self, payload: dict, *, token: str | None = None, user_id: int | None = None, now: int | None = None, cached: bool = False) -> PortalSession:
        """Verify before handing credentials to the launcher/cache; keys come only from the release trust set."""
        try:
            current = int(time.time()) if now is None else _integer(now)
            offline = self._claims(payload.get("offline_token"), current)
            core = self._claims(payload.get("core_grant"), current)
            license = payload.get("license")
            if not isinstance(license, dict) or not isinstance(license.get("id"), str) or not re.fullmatch(r"[a-f0-9]{32}", license["id"]):
                raise ValueError("invalid license")
            for field in ("license_id", "device_hash", "key_id", "issued_at", "not_before", "expires_at", "session_version", "resource_sha256"):
                if offline.get(field) != core.get(field):
                    raise ValueError("credential pair mismatch")
            if offline["license_id"] != license["id"] or offline.get("mode") != "online" or _integer(offline.get("offline_period_days")) != 1:
                raise ValueError("invalid online grace")
            if not isinstance(offline.get("entitlement_id"), str) or not re.fullmatch(r"[a-f0-9]{32}", offline["entitlement_id"]):
                raise ValueError("invalid entitlement")
            if not isinstance(core.get("grant_id"), str) or not re.fullmatch(r"[a-f0-9]{32}", core["grant_id"]):
                raise ValueError("invalid core grant")
            if offline.get("features") != ["runtime"] or type(offline.get("trial")) is not bool:
                raise ValueError("invalid features")
            if license.get("status") != "active" or license.get("product") != "desktop2stereo" or license.get("kind") not in {"trial", "formal"}:
                raise ValueError("license unavailable")
            if offline["trial"] != (license["kind"] == "trial"):
                raise ValueError("trial mismatch")
            expiry = license.get("expires_at")
            if expiry is not None and (_integer(expiry) <= current or offline["expires_at"] > expiry):
                raise ValueError("license expired")
            if license["kind"] == "trial" and expiry is None:
                raise ValueError("trial expiry missing")
            if _integer(payload.get("session_version")) != offline["session_version"]:
                raise ValueError("session version mismatch")
            if _integer(payload.get("heartbeat_seconds")) != 900 or _integer(payload.get("heartbeat_jitter_percent")) != 10:
                raise ValueError("invalid heartbeat policy")
            if _integer(payload.get("offline_expires_at")) != offline["expires_at"]:
                raise ValueError("expiry mismatch")
            access = "" if cached else payload.get("access_token", token)
            uid = payload.get("user_id", user_id)
            if not cached and (not isinstance(access, str) or not re.fullmatch(r"[a-f0-9]{64}", access)):
                raise ValueError("invalid bearer token")
            if (token is not None and access != token) or (user_id is not None and uid != user_id):
                raise ValueError("response identity mismatch")
            return PortalSession(access, _integer(uid), dict(license), self.device_hash, offline["session_version"],
                payload["offline_token"], payload["core_grant"], offline["issued_at"], offline["expires_at"])
        except Exception:
            raise PortalError("credential_verification_failed", clear_credentials=True) from None
