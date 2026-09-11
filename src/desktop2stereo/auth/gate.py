"""Launcher authentication gate kept separate from GUI1 and GUI2."""

from __future__ import annotations

import uuid

from .client import AuthClient, AuthError, AuthSession
from .clock import ClockSuspectError, TrustedClock
from .device import DeviceIdentity, DeviceIdentityError, device_identity
from .instance import InstanceLock
from .offline import (
    OfflineEntitlementError,
    OfflineEntitlementStorageError,
    OfflineEntitlementStore,
    persist_verified_entitlement,
    verify_entitlement,
)
from .storage import TokenStore


def require_authentication() -> AuthSession:
    """Return a valid session or raise without importing a runtime GUI."""
    lock = InstanceLock()
    try:
        lock.acquire()
    except Exception as exc:
        raise AuthError(str(exc), "already_running") from exc
    client = AuthClient()
    store = TokenStore()
    trusted_clock = TrustedClock()
    saved = store.load()
    offline = _load_offline_session(saved, trusted_clock)
    has_saved_credentials = _has_saved_credentials(saved)
    if offline and not has_saved_credentials:
        return offline
    if has_saved_credentials:
        try:
            return _restore_saved_session(client, store, saved, trusted_clock)
        except AuthError as error:
            if error.code == "network_error" and offline:
                return offline
            store.clear()
    from .gui import LoginLauncher

    session = LoginLauncher(client=client, store=store).run()
    if session is None:
        raise AuthError("未完成登录，已阻止启动运行界面", "login_required")
    _observe_session_time(session, trusted_clock)
    if session.selected_license_id:
        _activate_license(client, session.access_token, session.selected_license_id)
    return session


def _load_offline_session(saved: dict | None, trusted_clock: TrustedClock | None = None) -> AuthSession | None:
    try:
        clock = trusted_clock or TrustedClock()
        if not clock.has_checkpoint():
            return None
        claims = verify_entitlement(
            OfflineEntitlementStore().load() or "",
            now=clock.now(),
            expected_device_hash=device_identity().device_hash,
        )
    except (OfflineEntitlementError, ClockSuspectError, DeviceIdentityError):
        return None
    license_id = _license_record_id(claims)
    if license_id is None:
        return None
    return AuthSession("", None, saved.get("user", {}) if isinstance(saved, dict) else {}, [claims], license_id)


def validate_saved_authentication() -> AuthSession:
    """Validate saved credentials for a Runtime child without opening Flet."""
    client = AuthClient()
    store = TokenStore()
    saved = store.load()
    trusted_clock = TrustedClock()
    offline = _load_offline_session(saved, trusted_clock)
    if not _has_saved_credentials(saved):
        if offline:
            return offline
        raise AuthError("未找到已保存的登录状态，请先登录", "login_required")
    try:
        return _restore_saved_session(client, store, saved, trusted_clock)
    except AuthError as error:
        if error.code == "network_error" and offline:
            return offline
        store.clear()
        raise


def _has_saved_credentials(saved: dict | None) -> bool:
    """Return whether persisted data contains a syntactically usable token."""

    return isinstance(saved, dict) and _is_refresh_token(saved.get("refresh_token"))


def _is_access_token(value: object) -> bool:
    """Recognize the current server JWT shape without verifying its signature."""

    if not isinstance(value, str):
        return False
    parts = value.strip().split(".")
    return len(parts) == 3 and all(parts)


def _is_refresh_token(value: object) -> bool:
    """Recognize the current server UUID.secret refresh-token shape."""

    if not isinstance(value, str):
        return False
    sid, separator, secret = value.strip().partition(".")
    if not separator or not sid or not secret or "." in secret:
        return False
    try:
        uuid.UUID(sid)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def _persisted_access_token(value: object) -> str:
    return value.strip() if _is_access_token(value) else ""


def _persisted_refresh_token(value: object) -> str:
    return value.strip() if _is_refresh_token(value) else ""


def _license_record_id(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    license_id = value.get("id")
    if not isinstance(license_id, str) or not license_id.strip():
        return None
    return license_id.strip()


def _license_mode(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    mode = value.get("mode")
    return mode.casefold() if isinstance(mode, str) else ""


def _offline_period(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {7, 14, 30}:
        return None
    return value


def _restore_saved_session(client: AuthClient, store: TokenStore, saved: dict, clock: TrustedClock) -> AuthSession:
    """Restore a session without persisting the short-lived access token."""

    access_token = _persisted_access_token(saved.get("access_token"))
    refresh_token = _persisted_refresh_token(saved.get("refresh_token"))
    session: AuthSession | None = None
    if access_token and refresh_token:
        try:
            status = client.status(access_token)
        except AuthError:
            pass
        else:
            _observe_server_time(status, clock)
            session = AuthSession(
                access_token,
                refresh_token or None,
                saved.get("user") if isinstance(saved.get("user"), dict) else {},
                [],
            )
    if session is None:
        if not refresh_token:
            raise AuthError("未找到可用的刷新令牌，请先登录", "login_required")
        session = client.refresh(refresh_token)
        _observe_session_time(session, clock)
        if not session.refresh_token:
            raise AuthError("授权服务器未返回刷新令牌", "invalid_response")
        status = client.status(session.access_token)
        _observe_server_time(status, clock)

    if status.get("valid") is not True:
        raise AuthError("账号没有可用授权", "license_unavailable")
    licenses = status.get("licenses") if isinstance(status.get("licenses"), list) else []
    session.licenses = licenses
    if not session.user and isinstance(saved.get("user"), dict):
        session.user = saved["user"]
    selected_value = saved.get("selected_license_id")
    selected = selected_value.strip() if isinstance(selected_value, str) and selected_value.strip() else None
    if selected is None and len(licenses) == 1:
        selected = _license_record_id(licenses[0])
    if not selected or not any(_license_record_id(item) == selected for item in licenses):
        raise AuthError("当前需要选择有效授权", "license_selection_required")
    _activate_license(client, session.access_token, selected)
    session.selected_license_id = selected
    if not session.refresh_token:
        raise AuthError("授权服务器未返回刷新令牌", "invalid_response")
    if not store.save(_stored_session_payload(session)):
        raise AuthError("无法更新平台安全凭据", "secure_storage_unavailable")
    return session


def _stored_session_payload(session: AuthSession) -> dict:
    """Return the persisted subset; access tokens intentionally stay in memory."""

    return {
        "refresh_token": session.refresh_token,
        "user": session.user,
        "licenses": session.licenses,
        "selected_license_id": session.selected_license_id,
    }


def issue_and_store_offline_entitlement(
    session: AuthSession,
    offline_period_days: int | None = None,
    *,
    client: AuthClient | None = None,
    store: OfflineEntitlementStore | None = None,
) -> dict:
    """Issue, verify, and persist an offline entitlement for the selected license."""

    license_id, identity, requested_days = _resolve_offline_request(session, offline_period_days)
    entitlement = (client or AuthClient()).issue_offline_entitlement(
        session.access_token,
        license_id,
        identity.device_hash,
        requested_days,
    )
    return _persist_offline_entitlement(entitlement, identity, store)


def renew_and_store_offline_entitlement(
    session: AuthSession,
    offline_period_days: int | None = None,
    *,
    client: AuthClient | None = None,
    store: OfflineEntitlementStore | None = None,
) -> dict:
    """Renew, verify, and persist an offline entitlement for the selected license."""

    license_id, identity, requested_days = _resolve_offline_request(session, offline_period_days)
    response = (client or AuthClient()).renew_offline(
        session.access_token,
        license_id,
        identity.device_hash,
        requested_days,
    )
    entitlement = response.get("entitlement") if isinstance(response, dict) else None
    if not isinstance(entitlement, str) or not entitlement.strip():
        raise AuthError("服务器未返回离线授权凭证", "invalid_response")
    return _persist_offline_entitlement(entitlement.strip(), identity, store)


def _persist_offline_entitlement(entitlement: str, identity: DeviceIdentity, store: OfflineEntitlementStore | None) -> dict:
    try:
        clock = TrustedClock()
        claims = persist_verified_entitlement(
            entitlement,
            expected_device_hash=identity.device_hash,
            store=store,
            now=clock.now(),
        )
        issued_at = claims.get("issued_at")
        if issued_at is not None:
            clock.observe(issued_at)
        return claims
    except ClockSuspectError as exc:
        raise AuthError(str(exc), "clock_suspect") from exc
    except (TypeError, ValueError, OverflowError) as exc:
        raise AuthError("离线授权服务器时间字段无效", "invalid_response") from exc
    except OfflineEntitlementStorageError as exc:
        raise AuthError(str(exc), "offline_storage_unavailable") from exc
    except OfflineEntitlementError as exc:
        raise AuthError(str(exc), "offline_entitlement_invalid") from exc


def _resolve_offline_request(session: AuthSession, offline_period_days: int | None) -> tuple[str, DeviceIdentity, int]:
    if not session.access_token:
        raise AuthError("离线授权签发需要在线登录", "login_required")
    license_id = session.selected_license_id
    if not license_id:
        raise AuthError("当前需要选择有效授权", "license_selection_required")
    selected = next((item for item in session.licenses if _license_record_id(item) == license_id), None)
    if selected is None:
        raise AuthError("当前需要选择有效授权", "license_selection_required")
    mode = _license_mode(selected)
    if mode == "permanent":
        requested_days = 0
    else:
        if mode != "offline":
            raise AuthError("当前授权模式不支持离线授权", "license_mode_invalid")
        requested_days = offline_period_days
        if requested_days is None:
            requested_days = selected.get("offline_period_days", 7)
        requested_days = _offline_period(requested_days)
        if requested_days is None:
            raise AuthError("离线授权时长必须为 7、14 或 30 天", "invalid_input")
    try:
        identity = device_identity()
    except DeviceIdentityError as exc:
        raise AuthError(str(exc), "device_identity_unavailable") from exc
    return license_id, identity, requested_days


def _activate_license(client: AuthClient, access_token: str, license_id: str) -> None:
    try:
        identity = device_identity()
    except DeviceIdentityError as exc:
        raise AuthError(str(exc), "device_identity_unavailable") from exc
    client.activate_license(access_token, license_id, identity.device_hash, identity.fingerprint_version)


def _observe_server_time(payload: dict, clock: TrustedClock) -> None:
    value = payload.get("server_time")
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AuthError("授权服务器时间响应无效", "invalid_response")
        try:
            clock.observe(value)
        except (TypeError, ValueError) as exc:
            raise AuthError("授权服务器时间响应无效", "invalid_response") from exc
        except ClockSuspectError as exc:
            raise AuthError(str(exc), "clock_suspect") from exc


def _observe_session_time(session: AuthSession, clock: TrustedClock) -> None:
    if session.server_time is not None:
        try:
            clock.observe(session.server_time)
        except ClockSuspectError as exc:
            raise AuthError(str(exc), "clock_suspect") from exc
