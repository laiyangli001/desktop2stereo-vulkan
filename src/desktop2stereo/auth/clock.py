"""Persisted server-time checkpoint used to detect local clock rollback."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


ROLLBACK_TOLERANCE_SECONDS = 300


class ClockSuspectError(RuntimeError):
    """The local clock moved too far behind the last trusted server time."""


class TrustedClock:
    def __init__(self, app_dir: str | os.PathLike | None = None):
        root = Path(app_dir or Path.home() / ".desktop2stereo")
        self.path = root / "trusted-clock.json"
        self._last_now: int | None = None

    def has_checkpoint(self) -> bool:
        """Return whether a persisted checkpoint exists for offline use."""

        state = self._read_state()
        # A local-only timestamp is editable wall-clock state, not a trusted
        # server anchor and must not authorize offline startup.
        return bool(state and state.get("max_server_time"))

    def observe(self, server_time: int, *, local_time: int | None = None) -> None:
        if isinstance(server_time, bool) or not isinstance(server_time, int) or server_time <= 0:
            raise ValueError("server_time must be a positive integer")
        current = _resolve_local_time(local_time)
        state = self._read_state()
        trusted = state.get("max_server_time") if state else None
        max_local = state.get("max_local_time") if state else None
        if max_local is not None and current + ROLLBACK_TOLERANCE_SECONDS < max_local:
            raise ClockSuspectError("本机系统时间异常，请校准时间后重新联网验证授权")
        if trusted is not None and current + ROLLBACK_TOLERANCE_SECONDS < trusted:
            raise ClockSuspectError("本机系统时间异常，请校准时间后重新联网验证授权")
        if server_time > (trusted or 0):
            self._write(server_time, current)
        elif max_local is None or current > max_local:
            self._write(trusted or 0, current)

    def check(self, *, local_time: int | None = None) -> None:
        state = self._read_state()
        if not state:
            return
        current = _resolve_local_time(local_time)
        trusted = max(state.values())
        if current + ROLLBACK_TOLERANCE_SECONDS < trusted:
            raise ClockSuspectError("本机系统时间异常，请校准时间后重新联网验证授权")

    def now(self, *, local_time: int | None = None) -> int:
        """Return a non-decreasing wall-clock value anchored by server time."""

        current = _resolve_local_time(local_time)
        self.check(local_time=current)
        state = self._read_state() or {}
        trusted = max(state.values(), default=current)
        value = max(current, trusted, self._last_now or current)
        self._last_now = value
        if value > state.get("max_local_time", 0):
            self._write(state.get("max_server_time", 0), value)
        return value

    def _read_state(self) -> dict[str, int] | None:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return None
            state = {}
            for key in ("max_server_time", "max_local_time"):
                if key not in value or value[key] is None:
                    continue
                timestamp = value[key]
                if isinstance(timestamp, bool) or not isinstance(timestamp, int):
                    return None
                if key == "max_server_time" and timestamp < 0:
                    return None
                if key == "max_local_time" and timestamp <= 0:
                    return None
                state[key] = timestamp
            return state or None
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _read(self) -> int | None:
        state = self._read_state()
        return state.get("max_server_time") if state else None

    def _write(self, server_time: int, local_time: int | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        state = {"max_server_time": int(server_time)}
        if local_time is not None and local_time > 0:
            state["max_local_time"] = int(local_time)
        temporary.write_text(json.dumps(state), encoding="utf-8")
        if os.name != "nt":
            temporary.chmod(0o600)
        os.replace(temporary, self.path)


def _resolve_local_time(local_time: int | None) -> int:
    if local_time is None:
        return int(time.time())
    if isinstance(local_time, bool) or not isinstance(local_time, int) or local_time <= 0:
        raise ValueError("local_time must be a positive integer")
    return local_time
