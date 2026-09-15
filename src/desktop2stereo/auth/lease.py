"""Online authorization lease held by the protected Runtime child."""

from __future__ import annotations

import logging
import random
import threading
import time

from .client import AuthClient, AuthError, AuthSession
from .device import DeviceIdentityError, device_identity


DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 15 * 60
HEARTBEAT_JITTER_RATIO = 0.10
# Keep malformed or overly aggressive server values from creating avoidable
# verification traffic. The normal deployment value is still 15 minutes.
MIN_HEARTBEAT_INTERVAL_SECONDS = 5 * 60
MAX_HEARTBEAT_INTERVAL_SECONDS = 60 * 60
HEARTBEAT_RETRY_BASE_DELAY_SECONDS = 1
HEARTBEAT_RETRY_MAX_DELAY_SECONDS = 30
LOGGER = logging.getLogger(__name__)


def _heartbeat_delay(interval_seconds: int) -> float:
    """Return a bounded-jitter delay to avoid synchronized heartbeats."""

    lower = 1.0 - HEARTBEAT_JITTER_RATIO
    upper = 1.0 + HEARTBEAT_JITTER_RATIO
    return max(1.0, interval_seconds * random.uniform(lower, upper))


def _response_heartbeat_interval(payload: dict, fallback: int) -> int:
    if not isinstance(payload, dict):
        raise AuthError("服务器返回了无效心跳参数", "invalid_response")
    if "heartbeat_interval" not in payload:
        return fallback
    value = payload["heartbeat_interval"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuthError("服务器返回了无效心跳参数", "invalid_response")
    return max(MIN_HEARTBEAT_INTERVAL_SECONDS, min(MAX_HEARTBEAT_INTERVAL_SECONDS, value))


def _response_lease_token(payload: dict) -> str:
    value = payload.get("lease_token") if isinstance(payload, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise AuthError("服务器未返回有效运行租约令牌", "invalid_response")
    return value.strip()


def _heartbeat_retry_delay(attempt: int) -> float:
    """Return an exponential retry delay capped to protect the lease window."""

    exponent = max(0, min(5, int(attempt)))
    return min(HEARTBEAT_RETRY_MAX_DELAY_SECONDS, HEARTBEAT_RETRY_BASE_DELAY_SECONDS * (2**exponent))


def _heartbeat_error_retryable(error: AuthError) -> bool:
    return error.code == "network_error" or error.status_code == 429 or (error.status_code or 0) >= 500


class RuntimeLease:
    def __init__(self, session: AuthSession, client: AuthClient | None = None):
        if not session.selected_license_id:
            raise AuthError("当前需要选择有效授权", "license_selection_required")
        self.session = session
        self.client = client or AuthClient()
        self.license_id = session.selected_license_id
        try:
            self.device = device_identity().device_hash
        except DeviceIdentityError as exc:
            raise AuthError(str(exc), "device_identity_unavailable") from exc
        self.lease_token: str | None = None
        self.lease_expires_at: int | None = None
        self.heartbeat_interval = DEFAULT_HEARTBEAT_INTERVAL_SECONDS
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._wakeup = threading.Event()
        self._recheck_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._server_time: int | None = None
        self._server_time_monotonic: float | None = None
        self.core_grant: dict | None = None

    def start(self) -> None:
        result = self.client.online_heartbeat(self.session.access_token, self.license_id, self.device)
        self._observe_server_time(result)
        self.heartbeat_interval = _response_heartbeat_interval(result, self.heartbeat_interval)
        self.lease_token = _response_lease_token(result)
        self.lease_expires_at = self._response_lease_expiry(result)
        self._refresh_core_grant()
        self._thread = threading.Thread(target=self._run, name="D2SOnlineLease", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            delay = self._next_heartbeat_delay()
            if delay <= 0:
                self.lost.set()
                return
            action = self._wait_for_next_action(delay)
            if action == "stop":
                return
            try:
                result = self.client.online_heartbeat(self.session.access_token, self.license_id, self.device, self.lease_token)
                self._apply_heartbeat_result(result)
                self._refresh_core_grant()
            except AuthError as error:
                if not self._retry_until_expiry(error):
                    self.lost.set()
                    return

    def request_recheck(self) -> None:
        """Wake the lease worker for an immediate heartbeat after an environment change."""

        if self._stop.is_set() or self.lost.is_set():
            return
        self._recheck_requested.set()
        self._wakeup.set()

    def _wait_for_next_action(self, delay: float) -> str:
        """Wait for the next heartbeat, stop request, or explicit environment recheck."""

        self._wakeup.clear()
        if self._stop.is_set():
            return "stop"
        if self._recheck_requested.is_set():
            self._recheck_requested.clear()
            return "recheck"
        self._wakeup.wait(max(0.0, delay))
        if self._stop.is_set():
            return "stop"
        if self._recheck_requested.is_set():
            self._recheck_requested.clear()
            return "recheck"
        return "timeout"

    def _next_heartbeat_delay(self) -> float:
        """Never schedule the next heartbeat after the current lease expires."""

        delay = _heartbeat_delay(self.heartbeat_interval)
        if self.lease_expires_at is None:
            return delay
        remaining = self.lease_expires_at - self._trusted_now()
        if remaining <= 1:
            return 0.0
        return min(delay, float(remaining - 1))

    def close(self) -> None:
        self._stop.set()
        self._wakeup.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self.lease_token:
            try:
                self.client.online_logout(self.session.access_token, self.license_id, self.lease_token)
            except AuthError as error:
                if error.code not in {"lease_invalid", "license_not_found"}:
                    LOGGER.warning(
                        "online lease release failed: code=%s request_id=%s",
                        error.code,
                        error.request_id or "-",
                    )

    def _apply_heartbeat_result(self, result: dict) -> None:
        self._observe_server_time(result)
        self.heartbeat_interval = _response_heartbeat_interval(result, self.heartbeat_interval)
        self.lease_token = _response_lease_token(result)
        self.lease_expires_at = self._response_lease_expiry(result)

    def _refresh_core_grant(self) -> None:
        """Refresh the protected-core grant when supported by the auth client."""

        request_grant = getattr(self.client, "core_grant", None)
        if not callable(request_grant):
            self.core_grant = None
            return
        self.core_grant = request_grant(
            self.session.access_token,
            self.license_id,
            self.device,
        )

    def _observe_server_time(self, payload: dict) -> None:
        if not isinstance(payload, dict) or "server_time" not in payload:
            return
        value = payload["server_time"]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AuthError("服务器返回了无效服务端时间", "invalid_response")
        if self._server_time is not None and value < self._server_time:
            return
        self._server_time = value
        self._server_time_monotonic = time.monotonic()

    def _trusted_now(self) -> int:
        if self._server_time is not None and self._server_time_monotonic is not None:
            elapsed = max(0.0, time.monotonic() - self._server_time_monotonic)
            return self._server_time + int(elapsed)
        return int(time.time())

    def _response_lease_expiry(self, payload: dict) -> int:
        value = payload.get("expires_at") if isinstance(payload, dict) else None
        if isinstance(value, bool) or not isinstance(value, int):
            raise AuthError("服务器返回了无效租约有效期", "invalid_response")
        if value <= self._trusted_now():
            raise AuthError("服务器返回的租约已经过期", "lease_invalid")
        return value

    def _retry_until_expiry(self, error: AuthError) -> bool:
        if not _heartbeat_error_retryable(error):
            return False
        attempt = 0
        while not self._stop.is_set():
            remaining = (self.lease_expires_at or 0) - self._trusted_now()
            if remaining <= 0:
                return False
            action = self._wait_for_next_action(
                min(_heartbeat_retry_delay(attempt), max(1, remaining - 1))
            )
            if action == "stop":
                return False
            try:
                result = self.client.online_heartbeat(
                    self.session.access_token,
                    self.license_id,
                    self.device,
                    self.lease_token,
                )
                self._apply_heartbeat_result(result)
                self._refresh_core_grant()
                return True
            except AuthError as retry_error:
                if not _heartbeat_error_retryable(retry_error):
                    return False
                attempt += 1
        return False
