"""Bounded v2 heartbeat retries with an independent expiry watchdog."""
from __future__ import annotations

import random
import threading
import time

from .portal_access import PortalAccess
from .portal_client import PortalError, PortalSession


class PortalLease:
    def __init__(self, access: PortalAccess, *, on_lost=None, on_update=None,
                 monotonic=time.monotonic, jitter=random.uniform):
        self.access = access
        self.on_lost = on_lost
        self.on_update = on_update
        self.monotonic = monotonic
        self.jitter = jitter
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.RLock()
        self._terminal_lock = threading.RLock()
        self._threads = []
        self._started = False
        self._expires_at = 0
        self._expires_monotonic = 0.0
        self.session: PortalSession | None = None
        self.device = access.client.device_hash
        self.license_id: str | None = None
        self.core_grant: dict | None = None
        self.last_error: PortalError | None = None

    def start(self, *, background: bool = True) -> None:
        with self._lock:
            if self._started or self._stop.is_set():
                raise PortalError("lease_already_started")
            session = self.access.session
            if session is None:
                raise PortalError("login_required")
            cached = self.access.cache.load()
            if cached is None or cached.license["id"] != session.license["id"] or cached.session_version != session.session_version:
                raise PortalError("session_cache_mismatch", clear_credentials=True)
            self._apply(session)
            self._started = True
        if background:
            for target, name in [(self._watch_loop, "D2SPortalExpiry"), (self._heartbeat_loop, "D2SPortalHeartbeat")]:
                thread = threading.Thread(target=target, name=name, daemon=True)
                self._threads.append(thread)
                thread.start()

    def _apply(self, session: PortalSession) -> None:
        remaining = session.offline_expires_at - self.access.cache.now()
        if remaining <= 0:
            raise PortalError("credential_expired", clear_credentials=True)
        with self._terminal_lock:
            if self._stop.is_set() or self.lost.is_set():
                return
            if self._started and self._remaining() <= 0:
                raise PortalError("credential_expired", clear_credentials=True)
            self.session = session
            self.access.session = session
            self.license_id = session.license["id"]
            self._expires_at = session.offline_expires_at
            self._expires_monotonic = self.monotonic() + remaining
            self.core_grant = {"grant": session.core_grant}

    def _remaining(self) -> float:
        return min(
            self._expires_at - self.access.cache.now(),
            self._expires_monotonic - self.monotonic(),
        )

    def _fail(self, error: PortalError) -> None:
        # Publish revocation before acquiring any lock held by a slow persistence operation.
        with self._terminal_lock:
            already_lost = self.lost.is_set()
            if not already_lost:
                self.last_error = error
            self.lost.set()
            self._stop.set()
            self._wake.set()
            self.core_grant = None
            self.session = None
        if not already_lost and self.on_lost is not None:
            try:
                self.on_lost(error)
            except Exception:
                # Consumers still observe the already-set lost event.
                pass
        with self._lock:
            self.access.session = None
            self.access.offline_available = False
            try:
                self.access.cache.clear()
            except PortalError as storage_error:
                self.last_error = storage_error

    def _watch_once(self) -> bool:
        if self._stop.is_set():
            return False
        try:
            if self._remaining() <= 0:
                raise PortalError("credential_expired", clear_credentials=True)
            return True
        except PortalError as error:
            self._fail(error)
            return False
        except Exception:
            self._fail(PortalError("lease_clock_unavailable", clear_credentials=True))
            return False

    def verify_now(self) -> bool:
        """One bounded attempt; late responses cannot revive a stopped/expired lease."""
        if not self._started or not self._watch_once():
            return False
        with self._lock:
            previous = self.session
        if previous is None:
            return False
        try:
            # Network I/O never holds the state lock or blocks the expiry watchdog.
            updated = self.access.client.heartbeat(previous) if previous.access_token else self.access.cache.load()
            with self._lock:
                if self._stop.is_set() or self.lost.is_set() or not self._watch_once():
                    return False
                if (updated is None or updated.license["id"] != previous.license["id"] or
                        updated.session_version != previous.session_version or updated.user_id != previous.user_id or
                        updated.device_hash != self.device or
                        (previous.access_token and updated.access_token != previous.access_token)):
                    raise PortalError("session_binding_invalid", clear_credentials=True)
                if previous.access_token:
                    self.access.cache.save(updated)
                if self._stop.is_set() or self.lost.is_set():
                    if self.lost.is_set():
                        self.access.cache.clear()
                    return False
                self._apply(updated)
                if self._stop.is_set() or self.lost.is_set():
                    self.core_grant = None
                    return False
                self.last_error = None
                if not self._stop.is_set() and not self.lost.is_set() and self.on_update is not None:
                    self.on_update(updated)
                # The watchdog may have fired while a callback ran.
                if self._stop.is_set() or self.lost.is_set():
                    self.core_grant = None
                    if self.lost.is_set():
                        self.access.cache.clear()
                    return False
                return True
        except PortalError as error:
            self.last_error = error
            if error.clear_credentials or not error.retryable:
                self._fail(error)
            return False
        except Exception:
            self._fail(PortalError("lease_verification_failed", clear_credentials=True))
            return False

    def _normal_delay(self) -> float:
        return max(810.0, min(990.0, float(self.jitter(810.0, 990.0))))

    def _retry_delay(self, attempt: int) -> float:
        delay = min(300.0, 5.0 * (2 ** min(6, max(0, attempt))))
        return max(1.0, min(330.0, float(self.jitter(delay * 0.9, delay * 1.1))))

    def is_valid(self) -> bool:
        return self._started and not self._stop.is_set() and self._watch_once()

    def request_recheck(self) -> None:
        if self._started and not self._stop.is_set():
            self._wake.set()

    def _heartbeat_loop(self) -> None:
        delay = self._normal_delay()
        attempt = 0
        while not self._stop.is_set():
            try:
                remaining = self._remaining()
            except PortalError as error:
                self._fail(error)
                return
            if remaining <= 0:
                self._fail(PortalError("credential_expired", clear_credentials=True))
                return
            self._wake.wait(min(delay, remaining))
            self._wake.clear()
            if self._stop.is_set():
                return
            if self.verify_now():
                attempt = 0
                delay = self._normal_delay()
            else:
                delay = self._retry_delay(attempt)
                attempt += 1

    def _watch_loop(self) -> None:
        while not self._stop.is_set():
            if not self._watch_once():
                return
            self._stop.wait(0.25)

    def close(self, *, logout: bool = False) -> None:
        with self._terminal_lock:
            self._stop.set()
            self._wake.set()
            self.core_grant = None
            self.session = None
        with self._lock:
            if logout:
                try:
                    self.access.logout()
                except PortalError as error:
                    self.last_error = error
            else:
                # A normal process exit preserves signed offline grace for an offline restart.
                self.access.session = None
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=2)
