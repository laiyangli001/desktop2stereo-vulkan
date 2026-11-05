"""OS-protected v2 offline cache and denial handling, independent of the legacy gate."""
from __future__ import annotations

from dataclasses import replace
import time

from .portal_client import PortalClient, PortalError, PortalSession


_CACHE_FIELD = "portal_offline_v2"


class PortalCache:
    def __init__(self, client: PortalClient, store=None, *, wall_clock=time.time, monotonic=time.monotonic):
        if store is None:
            from .storage import TokenStore
            store = TokenStore()
        self.client = client
        self.store = store
        self.wall_clock = wall_clock
        self.monotonic = monotonic
        self._anchor = None
        self._wall_anchor = None

    @staticmethod
    def _payload(session: PortalSession) -> dict:
        # Opaque bearer tokens and passwords never enter the persisted payload.
        return {
            "user_id": session.user_id, "license": dict(session.license),
            "session_version": session.session_version, "offline_token": session.offline_token,
            "core_grant": session.core_grant, "offline_expires_at": session.offline_expires_at,
            "heartbeat_seconds": 900, "heartbeat_jitter_percent": 10,
        }

    def save(self, session: PortalSession) -> None:
        payload = self._payload(session)
        saved = self.store.load() or {}
        previous = saved.get(_CACHE_FIELD)
        wall = int(self.wall_clock())
        now = session.server_time
        if isinstance(previous, dict) and previous.get("base_url") == self.client.base_url:
            checkpoint, old_wall = previous.get("checkpoint_server"), previous.get("checkpoint_wall")
            if type(checkpoint) is int and type(old_wall) is int:
                now = max(now, checkpoint + max(0, wall - old_wall))
        if self._anchor is not None:
            server, monotonic = self._anchor
            elapsed = self.monotonic() - monotonic
            if elapsed < 0:
                raise PortalError("clock_rollback", clear_credentials=True)
            now = max(now, server + int(elapsed))
        self.client.accept(payload, now=now, cached=True)
        self._anchor = (now, self.monotonic())
        self._wall_anchor = wall
        saved[_CACHE_FIELD] = {
            "version": 2, "base_url": self.client.base_url, "device_hash": self.client.device_hash,
            "payload": payload, "checkpoint_server": now, "checkpoint_wall": wall,
        }
        if not self.store.save(saved):
            self._anchor = None
            self._wall_anchor = None
            raise PortalError("secure_storage_unavailable")

    def load(self) -> PortalSession | None:
        saved = self.store.load() or {}
        cached = saved.get(_CACHE_FIELD)
        if not isinstance(cached, dict):
            return None
        if cached.get("base_url") != self.client.base_url or cached.get("device_hash") != self.client.device_hash:
            return None
        try:
            if cached.get("version") != 2 or not isinstance(cached.get("payload"), dict):
                raise ValueError("cache format")
            anchor, wall = cached.get("checkpoint_server"), cached.get("checkpoint_wall")
            if type(anchor) is not int or anchor <= 0 or type(wall) is not int or wall <= 0:
                raise ValueError("clock checkpoint")
            current_wall = int(self.wall_clock())
            if current_wall < wall:
                raise PortalError("clock_rollback", clear_credentials=True)
            now = anchor + current_wall - wall
            if self._anchor is not None:
                server, monotonic = self._anchor
                elapsed = self.monotonic() - monotonic
                if elapsed < 0:
                    raise PortalError("clock_rollback", clear_credentials=True)
                now = max(now, server + int(elapsed))
            verified = self.client.accept(cached["payload"], now=now, cached=True)
            self._anchor = (now, self.monotonic())
            self._wall_anchor = current_wall
            cached["checkpoint_server"] = now
            cached["checkpoint_wall"] = current_wall
            if not self.store.save(saved):
                raise PortalError("secure_storage_unavailable")
            return verified
        except (ValueError, TypeError, KeyError):
            raise PortalError("cached_credentials_invalid", clear_credentials=True) from None

    def now(self) -> int:
        """Estimate trusted time without re-reading OS storage on every watchdog tick."""
        if self._anchor is None or self._wall_anchor is None:
            if self.load() is None:
                raise PortalError("offline_unavailable", clear_credentials=True)
        server, monotonic = self._anchor
        wall = int(self.wall_clock())
        elapsed = self.monotonic() - monotonic
        if wall < self._wall_anchor or elapsed < 0:
            raise PortalError("clock_rollback", clear_credentials=True)
        return max(server + int(elapsed), server + wall - self._wall_anchor)

    def clear(self) -> None:
        self._anchor = None
        self._wall_anchor = None
        saved = self.store.load()
        if not isinstance(saved, dict) or _CACHE_FIELD not in saved:
            return
        saved.pop(_CACHE_FIELD)
        if not self.store.save(saved):
            raise PortalError("secure_storage_unavailable", clear_credentials=True)


class PortalAccess:
    """Stateful boundary: explicit denials erase v2 cache before any fallback."""
    def __init__(self, client: PortalClient, cache: PortalCache):
        self.client = client
        self.cache = cache
        self.session: PortalSession | None = None
        self.offline_available = False

    def _deny(self, error: PortalError) -> None:
        if error.clear_credentials:
            self.session = None
            self.offline_available = False
            self.cache.clear()

    def login(self, identifier: str, password: str) -> PortalSession:
        self.session = None
        self.offline_available = False
        try:
            session = self.client.login(identifier, password)
            self.cache.save(session)
            self.session = session
            return session
        except PortalError as error:
            self._deny(error)
            # A user must explicitly choose offline after a failed connection.
            self.offline_available = not error.clear_credentials and (
                error.code == "network_error" or (error.status_code or 0) >= 500
            )
            raise

    def offline(self) -> PortalSession:
        if not self.offline_available:
            raise PortalError("offline_not_permitted", clear_credentials=True)
        try:
            session = self.cache.load()
            if session is None:
                raise PortalError("offline_unavailable", clear_credentials=True)
            self.session = session
            return session
        except PortalError as error:
            self._deny(error)
            raise

    def heartbeat(self) -> PortalSession:
        previous = self.session
        if previous is None:
            raise PortalError("login_required", clear_credentials=True)
        try:
            if not previous.access_token:
                session = self.cache.load()
                if session is None:
                    raise PortalError("offline_unavailable", clear_credentials=True)
                self.session = session
                return session
            session = self.client.heartbeat(previous)
            self.cache.save(session)
            self.session = session
            return session
        except PortalError as error:
            self._deny(error)
            if error.retryable and not error.clear_credentials:
                try:
                    cached = self.cache.load()
                    if cached is not None and cached.license["id"] == previous.license["id"] and cached.session_version == previous.session_version:
                        # Keep an online bearer only in memory so a later heartbeat can recover.
                        self.session = replace(cached, access_token=previous.access_token)
                        return self.session
                except PortalError as cache_error:
                    self._deny(cache_error)
                    raise
            raise

    def logout(self) -> None:
        previous = self.session
        self.session = None
        self.offline_available = False
        self.cache.clear()
        if previous is not None and previous.access_token:
            self.client.logout(previous)
