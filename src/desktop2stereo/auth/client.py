"""HTTP client for the versioned Desktop2Stereo authentication API."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import os
import platform
import re
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlsplit

import httpx

from .device import DeviceIdentityError, current_fingerprint_version, device_identity


API_ROOT = os.environ.get("D2S_API_BASE_URL", "https://100393.com/api/v1").rstrip("/")
D2S_API_VERSION = 1
_DEVICE_CODE_MAX_INTERVAL_SECONDS = 60 * 60
_DEVICE_CODE_SLOW_DOWN_SECONDS = 5
_STATUS_ERROR_CODES = {
    401: "unauthorized",
    403: "forbidden",
    409: "conflict",
    429: "rate_limited",
}
_SERVER_AUTH_ERROR_CODES = {
    "AUTH_UNAUTHORIZED": "unauthorized",
    "AUTH_TOKEN_EXPIRED": "token_expired",
    "AUTH_SESSION_REVOKED": "session_revoked",
    "AUTH_SESSION_LIMIT": "session_limit",
    "AUTH_SESSION_ISSUANCE_LIMIT": "session_issuance_limit",
    "AUTH_SESSION_MISMATCH": "session_mismatch",
    "AUTH_SESSION_REQUIRED": "session_required",
    "AUTH_SESSION_ID_REQUIRED": "session_id_required",
    "AUTH_SESSION_NOT_FOUND": "session_not_found",
    "AUTH_REFRESH_RACE": "refresh_race",
    "AUTH_INTERNAL_ERROR": "server_unavailable",
    "internal_error": "server_unavailable",
}


class AuthError(RuntimeError):
    def __init__(
        self,
        message: str,
        code: str = "auth_error",
        request_id: str | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.request_id = request_id
        self.status_code = status_code


@dataclass
class AuthSession:
    access_token: str
    refresh_token: str | None
    user: dict[str, Any]
    licenses: list[dict[str, Any]]
    selected_license_id: str | None = None
    server_time: int | None = None
    core_grant: str | None = None


@dataclass
class DeviceAuthorization:
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int
    verification_uri_complete: str | None = None


@dataclass
class CaptchaChallenge:
    captcha_id: str
    master_image: str
    thumb_image: str
    width: int
    height: int
    required_clicks: int


class AuthClient:
    def __init__(self, base_url: str | None = None, timeout: float = 8.0):
        # Resolve the override when the client is created so launch-time
        # configuration is honored even if this module was imported earlier.
        self.base_url = (base_url or os.environ.get("D2S_API_BASE_URL") or API_ROOT).rstrip("/")
        self.timeout = timeout
        self._trust_env = not self._is_loopback_url(self.base_url)

    @staticmethod
    def _is_loopback_url(url: str) -> bool:
        hostname = urlsplit(url).hostname
        if not hostname:
            return False
        if hostname.casefold() == "localhost":
            return True
        try:
            return ip_address(hostname).is_loopback
        except ValueError:
            return False

    def _http_options(self) -> dict[str, Any]:
        """Bypass inherited proxies only for local development endpoints."""

        return {"trust_env": False} if not self._trust_env else {}

    def _native_api_url(self, path: str) -> str:
        """Build a URL for the server's non-versioned new-api endpoints."""

        api_root = self.base_url.rsplit("/api/v1", 1)[0] + "/api"
        return f"{api_root}/{path.lstrip('/')}"

    def login(
        self,
        email: str,
        password: str,
        turnstile_token: str | None = None,
        captcha_id: str | None = None,
        captcha_clicks: list[dict[str, int]] | None = None,
    ) -> AuthSession:
        if not email.strip() or not password:
            raise AuthError("邮箱和密码不能为空", "invalid_input")
        payload = {"username": email.strip(), **self._password_fields(password)}
        if turnstile_token and turnstile_token.strip():
            payload["turnstile_token"] = turnstile_token.strip()
        if captcha_id and captcha_id.strip():
            payload["captcha_id"] = captcha_id.strip()
        if captcha_clicks:
            payload["captcha_clicks"] = captcha_clicks
        try:
            response = httpx.post(
                f"{self.base_url}/auth/login",
                json=payload,
                timeout=self.timeout,
                **self._http_options(),
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        return self._session_from_response(response)

    def _password_fields(self, password: str) -> dict[str, str]:
        """Match the server's optional RSA-OAEP password login contract."""

        try:
            response = httpx.get(
                self._native_api_url("user/login/encryption-key"),
                timeout=self.timeout,
                **self._http_options(),
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接登录加密配置服务：{exc}", "network_error") from exc
        data = self._native_data(response)
        if data.get("enabled") is not True:
            return {"password": password}

        key_id = data.get("kid")
        public_key = data.get("public_key")
        if not isinstance(key_id, str) or not key_id.strip() or not isinstance(public_key, str) or not public_key.strip():
            raise AuthError("登录加密公钥无效", "encryption_unavailable")
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding

            key = serialization.load_pem_public_key(public_key.encode("ascii"))
            ciphertext = key.encrypt(
                password.encode("utf-8"),
                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
            )
        except (ImportError, ValueError, TypeError, UnicodeEncodeError, AttributeError) as exc:
            raise AuthError("客户端不支持登录密码加密", "encryption_unavailable") from exc
        return {
            "password_encrypted": base64.b64encode(ciphertext).decode("ascii"),
            "encryption_key_id": key_id,
        }

    def get_captcha(self) -> CaptchaChallenge:
        try:
            response = httpx.get(
                self._native_api_url("captcha"),
                timeout=self.timeout,
                **self._http_options(),
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接验证码服务：{exc}", "network_error") from exc
        if response.status_code >= 400:
            self._raise_response(response)
        try:
            envelope = response.json()
        except ValueError as exc:
            raise AuthError("验证码服务返回了无效响应", "invalid_response") from exc
        data = envelope.get("data") if isinstance(envelope, dict) and envelope.get("success") is True else None
        if not isinstance(data, dict):
            self._raise_response(response)
        try:
            captcha_id = self._required_response_string(data, "id")
            master_image = self._required_response_string(data, "master_image")
            thumb_image = self._required_response_string(data, "thumb_image")
            dimensions = (data["width"], data["height"], data["required_clicks"])
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
                raise ValueError("invalid captcha dimensions")
            return CaptchaChallenge(
                captcha_id=captcha_id,
                master_image=master_image,
                thumb_image=thumb_image,
                width=dimensions[0],
                height=dimensions[1],
                required_clicks=dimensions[2],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AuthError("验证码服务返回了无效挑战", "invalid_response") from exc

    def authorize_device(
        self,
        client_name: str = "Desktop2Stereo",
        device_hash: str | None = None,
        fingerprint_version: int | None = None,
        platform_name: str | None = None,
    ) -> DeviceAuthorization:
        payload = self._device_payload(device_hash, fingerprint_version)
        payload["client_name"] = client_name
        payload["platform"] = platform_name or platform.system().casefold()
        try:
            response = httpx.post(f"{self.base_url}/device/authorize", json=payload, timeout=self.timeout, **self._http_options())
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        try:
            data = self._d2s_data(response)
            expires_in = data["expires_in"]
            interval = data.get("interval", 5)
            if (
                isinstance(expires_in, bool)
                or not isinstance(expires_in, int)
                or isinstance(interval, bool)
                or not isinstance(interval, int)
                or expires_in <= 0
                or interval <= 0
                or interval > _DEVICE_CODE_MAX_INTERVAL_SECONDS
            ):
                raise ValueError("invalid device authorization timing")
            device_code = self._required_response_string(data, "device_code")
            user_code = self._required_response_string(data, "user_code")
            verification_uri = self._required_response_string(data, "verification_uri")
            verification_uri_complete = data.get("verification_uri_complete")
            if verification_uri_complete is not None:
                if not isinstance(verification_uri_complete, str) or not verification_uri_complete.strip():
                    raise ValueError("invalid verification_uri_complete")
                verification_uri_complete = verification_uri_complete.strip()
            return DeviceAuthorization(
                device_code,
                user_code,
                verification_uri,
                expires_in,
                interval,
                verification_uri_complete,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AuthError("授权服务器返回了无效设备码", "invalid_response") from exc

    def device_token(self, device_code: str) -> AuthSession:
        try:
            response = httpx.post(f"{self.base_url}/device/token", json={"device_code": device_code}, timeout=self.timeout, **self._http_options())
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        return self._session_from_d2s_response(response)

    def refresh(self, refresh_token: str) -> AuthSession:
        try:
            response = httpx.post(
                f"{self.base_url}/auth/refresh",
                headers=self._session_headers(),
                json={"refresh_token": refresh_token},
                timeout=self.timeout,
                **self._http_options(),
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        return self._session_from_response(response)

    def change_password(self, access_token: str, current_password: str, new_password: str) -> dict[str, Any]:
        if not current_password or len(new_password) < 8 or len(new_password) > 20:
            raise AuthError("密码长度或当前密码无效", "invalid_input")
        try:
            response = httpx.put(
                self._native_api_url("user/self"),
                headers={"Authorization": f"Bearer {access_token}"},
                json={"original_password": current_password, "password": new_password},
                timeout=self.timeout,
                **self._http_options(),
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        return self._native_data(response)

    def cancel_device(self, device_code: str) -> None:
        try:
            httpx.post(f"{self.base_url}/device/cancel", json={"device_code": device_code}, timeout=self.timeout, **self._http_options())
        except httpx.HTTPError:
            pass

    def status(self, access_token: str) -> dict[str, Any]:
        try:
            response = httpx.get(
                f"{self.base_url}/license/status",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=self.timeout,
                **self._http_options(),
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        data = self._d2s_data(response)
        self._validate_license_list(data)
        self._validate_server_time(data)
        return data

    def license_list(self, access_token: str) -> list[dict[str, Any]]:
        try:
            response = httpx.get(f"{self.base_url}/license/list", headers={"Authorization": f"Bearer {access_token}"}, timeout=self.timeout, **self._http_options())
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        data = self._d2s_data(response)
        self._validate_server_time(data)
        return self._validate_license_list(data)

    def public_keys(self) -> list[dict[str, Any]]:
        """Fetch the server key manifest without changing release trust roots."""

        try:
            response = httpx.get(f"{self.base_url}/license/keys", timeout=self.timeout, **self._http_options())
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        data = self._d2s_data(response)
        keys = data.get("keys")
        if not isinstance(keys, list):
            raise AuthError("授权服务器未返回有效公钥清单", "invalid_response")
        if not keys or any(not self._is_valid_public_jwk(item) for item in keys):
            raise AuthError("授权服务器返回了无效公钥清单", "invalid_response")
        key_ids = [item["kid"] for item in keys]
        if len(set(key_ids)) != len(key_ids):
            raise AuthError("授权服务器返回了重复公钥 ID", "invalid_response")
        return keys

    def activate_license(self, access_token: str, license_id: str, device_hash: str, fingerprint_version: int | None = None) -> dict[str, Any]:
        self._validate_license_id(license_id)
        payload = self._device_payload(device_hash, fingerprint_version)
        payload["license_id"] = license_id
        try:
            response = httpx.post(f"{self.base_url}/license/activate", headers={"Authorization": f"Bearer {access_token}"}, json=payload, timeout=self.timeout, **self._http_options())
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        return self._d2s_data(response)

    def switch_license(self, access_token: str, license_id: str, device_hash: str, fingerprint_version: int | None = None) -> dict[str, Any]:
        """Select/bind an account license without coupling the launcher to GUI code."""
        self._validate_license_id(license_id)
        payload = self._device_payload(device_hash, fingerprint_version)
        payload["license_id"] = license_id
        try:
            response = httpx.post(
                f"{self.base_url}/license/switch",
                headers={"Authorization": f"Bearer {access_token}"},
                json=payload,
                timeout=self.timeout,
                **self._http_options(),
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        return self._d2s_data(response)

    def change_license_mode(
        self,
        access_token: str,
        license_id: str,
        device_hash: str,
        mode: str,
        confirmation: str | None = None,
        offline_period_days: int | None = None,
    ) -> dict[str, Any]:
        """Request a server-side license mode transition."""
        if not isinstance(mode, str) or mode.casefold() not in {"online", "offline", "permanent"}:
            raise AuthError("授权模式无效", "license_mode_invalid")
        self._validate_license_identity(license_id, device_hash)
        mode = mode.casefold()
        if mode == "permanent" and confirmation != "PERMANENT":
            raise AuthError("永久绑定需要输入 PERMANENT 确认", "permanent_confirmation_required")
        if mode == "offline" and not self._is_offline_period(offline_period_days):
            raise AuthError("离线授权时长必须为 7、14 或 30 天", "invalid_input")
        payload: dict[str, Any] = {"license_id": license_id, "device_hash": device_hash, "mode": mode}
        if confirmation is not None:
            payload["confirmation"] = confirmation
        if offline_period_days is not None:
            payload["offline_period_days"] = offline_period_days
        try:
            response = httpx.post(
                f"{self.base_url}/license/change-mode",
                headers={"Authorization": f"Bearer {access_token}"},
                json=payload,
                timeout=self.timeout,
                **self._http_options(),
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        return self._d2s_data(response)

    def revoke_free(self, access_token: str, license_id: str, device_hash: str) -> dict[str, Any]:
        self._validate_license_identity(license_id, device_hash)
        try:
            response = httpx.post(f"{self.base_url}/license/revoke/free", headers={"Authorization": f"Bearer {access_token}"}, json={"license_id": license_id, "device_hash": device_hash}, timeout=self.timeout, **self._http_options())
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        return self._d2s_data(response)

    def renew_offline(self, access_token: str, license_id: str, device_hash: str, offline_period_days: int) -> dict[str, Any]:
        """Renew one bound license; the server calculates the new expiry."""
        self._validate_license_identity(license_id, device_hash)
        if not self._is_offline_period(offline_period_days):
            raise AuthError("离线授权时长必须为 7、14 或 30 天", "invalid_input")
        try:
            from .core_device import get_core_device_key

            device_public_key = get_core_device_key().public_key_b64
        except Exception as exc:
            raise AuthError("设备核心密钥不可用", "secure_storage_unavailable") from exc
        try:
            response = httpx.post(
                f"{self.base_url}/license/renew",
                headers={"Authorization": f"Bearer {access_token}"},
                json={"license_id": license_id, "device_hash": device_hash, "device_public_key": device_public_key, "offline_period_days": offline_period_days},
                timeout=self.timeout,
                **self._http_options(),
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        return self._d2s_data(response)

    def create_offline_extension_order(self, access_token: str, license_id: str, device_hash: str, channel: str) -> dict[str, Any]:
        """Create a server-priced +30 day offline extension order."""
        self._validate_license_identity(license_id, device_hash)
        try:
            response = httpx.post(
                f"{self.base_url}/license/offline/extend",
                headers={"Authorization": f"Bearer {access_token}"},
                json={"license_id": license_id, "device_hash": device_hash, "channel": channel},
                timeout=self.timeout,
                **self._http_options(),
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        return self._d2s_data(response)

    def confirm_permanent(self, access_token: str, license_id: str, device_hash: str) -> dict[str, Any]:
        """Permanently bind one already-selected license after explicit confirmation."""
        self._validate_license_identity(license_id, device_hash)
        try:
            response = httpx.post(
                f"{self.base_url}/license/permanent/confirm",
                headers={"Authorization": f"Bearer {access_token}"},
                json={"license_id": license_id, "device_hash": device_hash, "confirmation": "PERMANENT"},
                timeout=self.timeout,
                **self._http_options(),
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        return self._d2s_data(response)

    def revoke_paid(self, access_token: str, license_id: str, device_hash: str, channel: str = "") -> dict[str, Any]:
        """Create a server-priced paid-revoke order; payment settlement is separate."""
        self._validate_license_identity(license_id, device_hash)
        try:
            response = httpx.post(
                f"{self.base_url}/license/revoke/paid",
                headers={"Authorization": f"Bearer {access_token}"},
                json={"license_id": license_id, "device_hash": device_hash, "channel": channel},
                timeout=self.timeout,
                **self._http_options(),
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        return self._d2s_data(response)

    def request_manual_unbind(self, access_token: str, license_id: str, reason: str, purchase_proof_ref: str | None = None) -> dict[str, Any]:
        self._validate_license_id(license_id)
        if not isinstance(reason, str) or not reason.strip():
            raise AuthError("人工解绑原因不能为空", "invalid_input")
        payload: dict[str, Any] = {"license_id": license_id, "reason": reason}
        if purchase_proof_ref:
            payload["proof_ref"] = purchase_proof_ref
        try:
            response = httpx.post(f"{self.base_url}/license/manual-unbind", headers={"Authorization": f"Bearer {access_token}"}, json=payload, timeout=self.timeout, **self._http_options())
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        return self._d2s_data(response)

    def online_heartbeat(self, access_token: str, license_id: str, device_hash: str, lease_token: str | None = None) -> dict[str, Any]:
        self._validate_license_identity(license_id, device_hash)
        if lease_token is not None and (not isinstance(lease_token, str) or not lease_token.strip()):
            raise AuthError("运行租约令牌无效", "invalid_input")
        try:
            response = httpx.post(f"{self.base_url}/license/online/heartbeat", headers={"Authorization": f"Bearer {access_token}"}, json={"license_id": license_id, "device_hash": device_hash, "lease_token": lease_token}, timeout=self.timeout, **self._http_options())
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        return self._d2s_data(response)

    def core_grant(
        self,
        access_token: str,
        license_id: str,
        device_hash: str,
        core_id: str = "parallax-core",
        core_version: int = 1,
    ) -> dict[str, Any]:
        """Request a server-signed grant for the protected stereo core."""

        self._validate_license_identity(license_id, device_hash)
        if not isinstance(core_id, str) or not core_id.strip():
            raise AuthError("受保护核心标识无效", "invalid_input")
        if isinstance(core_version, bool) or not isinstance(core_version, int) or core_version <= 0:
            raise AuthError("受保护核心版本无效", "invalid_input")
        try:
            from .core_device import get_core_device_key

            device_public_key = get_core_device_key().public_key_b64
        except Exception as exc:
            raise AuthError("设备核心密钥不可用", "secure_storage_unavailable") from exc
        try:
            response = httpx.post(
                f"{self.base_url}/license/core/grant",
                headers={"Authorization": f"Bearer {access_token}"},
                json={
                    "license_id": license_id,
                    "device_hash": device_hash,
                    "core_id": core_id.strip(),
                    "core_version": core_version,
                    "device_public_key": device_public_key,
                },
                timeout=self.timeout,
                **self._http_options(),
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        data = self._d2s_data(response)
        grant = data.get("grant")
        if not isinstance(grant, str) or grant.count(".") != 2:
            raise AuthError("授权服务器未返回有效核心授权包", "invalid_response")
        claims = data.get("claims")
        if not isinstance(claims, dict):
            raise AuthError("授权服务器未返回有效核心授权字段", "invalid_response")
        return {"grant": grant, "claims": claims}

    def online_logout(self, access_token: str, license_id: str, lease_token: str) -> None:
        self._validate_license_id(license_id)
        if not isinstance(lease_token, str) or not lease_token.strip():
            raise AuthError("运行租约令牌无效", "invalid_input")
        try:
            response = httpx.post(
                f"{self.base_url}/license/online/logout",
                headers={"Authorization": f"Bearer {access_token}"},
                json={"license_id": license_id, "lease_token": lease_token},
                timeout=self.timeout,
                **self._http_options(),
            )
        except httpx.HTTPError:
            return
        if response.status_code >= 400:
            self._raise_response(response)

    def issue_offline_entitlement(self, access_token: str, license_id: str, device_hash: str, offline_period_days: int) -> str:
        self._validate_license_identity(license_id, device_hash)
        if not isinstance(offline_period_days, int) or isinstance(offline_period_days, bool) or offline_period_days not in {0, 7, 14, 30}:
            raise AuthError("离线授权时长无效", "invalid_input")
        try:
            from .core_device import get_core_device_key

            device_public_key = get_core_device_key().public_key_b64
        except Exception as exc:
            raise AuthError("设备核心密钥不可用", "secure_storage_unavailable") from exc
        try:
            response = httpx.post(f"{self.base_url}/license/offline/issue", headers={"Authorization": f"Bearer {access_token}"}, json={"license_id": license_id, "device_hash": device_hash, "device_public_key": device_public_key, "offline_period_days": offline_period_days}, timeout=self.timeout, **self._http_options())
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        try:
            data = self._d2s_data(response)
            entitlement = data.get("entitlement")
        except AuthError:
            raise
        if not isinstance(entitlement, str) or not entitlement.strip():
            raise AuthError("服务器未返回离线授权凭证", "invalid_response")
        return entitlement.strip()

    def logout(self, access_token: str | None = None, refresh_token: str | None = None) -> None:
        headers = self._session_headers()
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        payload = {"refresh_token": refresh_token} if refresh_token else {}
        try:
            response = httpx.post(
                f"{self.base_url}/auth/logout",
                headers=headers,
                json=payload,
                timeout=self.timeout,
                **self._http_options(),
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        if response.status_code >= 400 and response.status_code != 401:
            self._raise_response(response)

    def _session_from_response(self, response: httpx.Response) -> AuthSession:
        if response.status_code >= 400:
            self._raise_response(response)
        data = self._native_data(response)
        if data.get("require_2fa") is True:
            raise AuthError("账号已启用双重验证，请使用浏览器授权登录", "two_factor_required")
        session = self._session_from_payload(data)
        if not session.refresh_token:
            try:
                session.refresh_token = response.cookies.get("new_api_refresh")
            except (KeyError, httpx.CookieConflict, RuntimeError) as exc:
                if isinstance(exc, (KeyError, httpx.CookieConflict)):
                    raise AuthError("授权服务器返回了多个刷新令牌", "invalid_response") from exc
        return session

    def _session_from_d2s_response(self, response: httpx.Response) -> AuthSession:
        data = self._d2s_data(response)
        return self._session_from_payload(data, validate_license_ids=True)

    def _session_headers(self) -> dict[str, str]:
        parsed = urlsplit(self.base_url)
        if not parsed.scheme or not parsed.netloc:
            return {}
        return {"Origin": f"{parsed.scheme}://{parsed.netloc}"}

    @staticmethod
    def _session_from_payload(data: dict[str, Any], *, validate_license_ids: bool = False) -> AuthSession:
        access_token = data.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            raise AuthError("授权服务器未返回登录令牌", "invalid_response")
        refresh_token = data.get("refresh_token")
        if refresh_token is not None and not isinstance(refresh_token, str):
            raise AuthError("授权服务器返回了无效刷新令牌", "invalid_response")
        refresh_token = refresh_token or None
        server_time = data.get("server_time")
        if server_time is not None and (isinstance(server_time, bool) or not isinstance(server_time, int) or server_time <= 0):
            raise AuthError("授权服务器时间响应无效", "invalid_response")
        licenses = data.get("licenses", [])
        if not isinstance(licenses, list):
            raise AuthError("授权服务器返回了无效授权列表", "invalid_response")
        if validate_license_ids:
            AuthClient._validate_license_list({"licenses": licenses})
        return AuthSession(
            access_token=access_token,
            refresh_token=refresh_token,
            user=data.get("user") if isinstance(data.get("user"), dict) else {},
            licenses=licenses,
            server_time=server_time,
        )

    @classmethod
    def _native_data(cls, response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            cls._raise_response(response)
        try:
            envelope = response.json()
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        if not isinstance(envelope, dict) or envelope.get("success") is not True:
            raise AuthError("授权服务器返回了无效响应", "invalid_response")
        data = envelope.get("data")
        if not isinstance(data, dict):
            raise AuthError("授权服务器返回了无效响应", "invalid_response")
        return data

    @classmethod
    def _d2s_data(cls, response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            cls._raise_response(response)
        try:
            envelope = response.json()
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        if not isinstance(envelope, dict):
            raise AuthError("授权服务器返回了无效响应", "invalid_response")
        try:
            request_id = cls._optional_response_string(envelope, "request_id")
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        if envelope.get("success") is not True:
            cls._raise_response(response)
        if envelope.get("version") != D2S_API_VERSION:
            raise AuthError(
                "授权服务器协议版本不受支持",
                "unsupported_version",
                request_id,
            )
        data = envelope.get("data")
        if not isinstance(data, dict):
            raise AuthError(
                "授权服务器返回了无效响应",
                "invalid_response",
                request_id,
            )
        return data

    @staticmethod
    def _device_payload(device_hash: str | None, fingerprint_version: int | None) -> dict[str, Any]:
        if device_hash is None:
            try:
                identity = device_identity()
            except DeviceIdentityError as exc:
                raise AuthError(str(exc), "device_identity_unavailable") from exc
            device_hash = identity.device_hash
            if fingerprint_version is None:
                fingerprint_version = identity.fingerprint_version
        if fingerprint_version is None:
            fingerprint_version = current_fingerprint_version()
        if (
            not isinstance(device_hash, str)
            or len(device_hash) != 64
            or any(character not in "0123456789abcdef" for character in device_hash)
            or isinstance(fingerprint_version, bool)
            or not isinstance(fingerprint_version, int)
            or fingerprint_version <= 0
        ):
            raise AuthError("设备指纹或指纹版本无效", "device_identity_unavailable")
        return {"device_hash": device_hash, "fingerprint_version": fingerprint_version}

    @classmethod
    def _validate_license_identity(cls, license_id: str, device_hash: str) -> None:
        cls._validate_license_id(license_id)
        if not isinstance(device_hash, str):
            raise AuthError("设备指纹无效", "device_identity_unavailable")
        cls._device_payload(device_hash, current_fingerprint_version())

    @staticmethod
    def _validate_license_id(license_id: str) -> None:
        if not isinstance(license_id, str) or not license_id.strip():
            raise AuthError("授权 ID 无效", "invalid_input")

    @staticmethod
    def _is_offline_period(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value in {7, 14, 30}

    @staticmethod
    def _required_response_string(data: dict[str, Any], field: str) -> str:
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing or invalid response field: {field}")
        return value.strip()

    @staticmethod
    def _optional_response_string(data: dict[str, Any], field: str) -> str | None:
        value = data.get(field)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid response field: {field}")
        return value.strip()

    @staticmethod
    def _validate_license_list(data: dict[str, Any]) -> list[dict[str, Any]]:
        licenses = data.get("licenses") if isinstance(data, dict) else None
        if not isinstance(licenses, list):
            raise AuthError("授权服务器未返回有效授权列表", "invalid_response")
        if any(
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not item["id"].strip()
            for item in licenses
        ):
            raise AuthError("授权服务器返回了无效授权项", "invalid_response")
        return licenses

    @staticmethod
    def _is_valid_public_jwk(value: object) -> bool:
        if not isinstance(value, dict):
            return False
        if any(value.get(field) != expected for field, expected in (("kty", "EC"), ("crv", "P-256"), ("alg", "ES256"))):
            return False
        if value.get("use") not in (None, "sig"):
            return False
        key_id = value.get("kid")
        x_coordinate = value.get("x")
        y_coordinate = value.get("y")
        return (
            isinstance(key_id, str)
            and bool(key_id.strip())
            and key_id == key_id.strip()
            and AuthClient._is_valid_jwk_coordinate(x_coordinate)
            and AuthClient._is_valid_jwk_coordinate(y_coordinate)
            and AuthClient._is_valid_p256_point(x_coordinate, y_coordinate)
        )

    @staticmethod
    def _is_valid_jwk_coordinate(value: object) -> bool:
        """Require a canonical 32-byte, unpadded Base64URL coordinate."""

        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{43}", value):
            return False
        try:
            decoded = base64.b64decode(value + "=", altchars=b"-_", validate=True)
        except (ValueError, base64.binascii.Error):
            return False
        return len(decoded) == 32 and base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii") == value

    @staticmethod
    def _is_valid_p256_point(x_coordinate: str, y_coordinate: str) -> bool:
        if not AuthClient._is_valid_jwk_coordinate(x_coordinate) or not AuthClient._is_valid_jwk_coordinate(y_coordinate):
            return False
        try:
            from cryptography.hazmat.primitives.asymmetric import ec

            decode = lambda value: int.from_bytes(base64.urlsafe_b64decode(value + "="), "big")
            ec.EllipticCurvePublicNumbers(decode(x_coordinate), decode(y_coordinate), ec.SECP256R1()).public_key()
        except Exception:
            return False
        return True

    @staticmethod
    def _validate_server_time(data: dict[str, Any]) -> None:
        value = data.get("server_time") if isinstance(data, dict) else None
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise AuthError("授权服务器时间响应无效", "invalid_response")

    @staticmethod
    def _raise_response(response: httpx.Response) -> None:
        try:
            data = response.json()
        except ValueError:
            data = {}
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict):
            error_code = error.get("code")
            error_message = error.get("message")
        else:
            error_code = error or (data.get("code") if isinstance(data, dict) else None)
            error_message = None
        message = data.get("message") if isinstance(data, dict) else None
        request_id = data.get("request_id") if isinstance(data, dict) else None
        if not isinstance(error_code, str) or not error_code.strip():
            error_code = None
        else:
            error_code = error_code.strip()
        if not isinstance(error_message, str) or not error_message.strip():
            error_message = None
        else:
            error_message = error_message.strip()
        if not isinstance(message, str) or not message.strip():
            message = None
        else:
            message = message.strip()
        if not isinstance(request_id, str) or not request_id.strip():
            request_id = None
        else:
            request_id = request_id.strip()
        normalized_code = _SERVER_AUTH_ERROR_CODES.get(error_code, error_code)
        if not error_code:
            normalized_code = _STATUS_ERROR_CODES.get(response.status_code)
            if normalized_code is None and response.status_code >= 500:
                normalized_code = "server_unavailable"
        raise AuthError(
            error_message or message or "授权验证失败",
            normalized_code or "auth_error",
            request_id,
            response.status_code,
        )
