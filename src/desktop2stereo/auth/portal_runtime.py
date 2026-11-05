"""Bind PHP authorization to the actual protected-core gate and runtime stop event."""
from __future__ import annotations

import os
from pathlib import Path

from .portal_access import PortalAccess
from .portal_client import PortalError
from .portal_lease import PortalLease


class PortalRuntime:
    def __init__(self, access: PortalAccess, *, gate=None, runner=None, resource_path=None, native_path=None):
        self.access = access
        self.gate = gate
        self.runner = runner
        self.resource_path = resource_path or os.environ.get("D2S_PARALLAX_CORE_RESOURCE") or str(
            Path(__file__).resolve().parents[1] / "protected" / "parallax-core.enc"
        )
        self.native_path = native_path or os.environ.get("D2S_PARALLAX_CORE_NATIVE") or None
        self.lease = PortalLease(access, on_lost=self._revoke, on_update=self._update)

    def _revoke(self, error=None) -> None:
        if self.gate is not None:
            self.gate.clear_protected_parallax_core()

    def _update(self, session) -> None:
        if not self.lease.is_valid():
            raise PortalError("runtime_authorization_unavailable", clear_credentials=True)
        try:
            if not Path(self.resource_path).is_file():
                raise ValueError("resource missing")
            self.gate.configure_protected_parallax_core(
                self.resource_path, session.core_grant, now=self.access.cache.now(),
                expected_device_hash=session.device_hash, native_path=self.native_path, require_native=True,
            )
        except Exception:
            raise PortalError("protected_core_unavailable", clear_credentials=True) from None
        if not self.lease.is_valid():
            self._revoke()
            raise PortalError("runtime_authorization_unavailable", clear_credentials=True)

    def start(self) -> None:
        if self.gate is None:
            from ..stereo_runtime import parallax
            self.gate = parallax
        self.gate.require_protected_parallax_core()
        self.gate.set_protected_runtime_guard(self.lease.is_valid)
        try:
            self.lease.start()
            self._update(self.access.session)
        except Exception:
            self.lease._fail(PortalError("runtime_start_rejected", clear_credentials=True))
            raise

    def run(self, *, max_seconds=None) -> int:
        try:
            self.start()
            runner = self.runner
            if runner is None:
                from ..app_runtime.runtime_entry import run_processing_runtime
                runner = run_processing_runtime
            return runner(max_seconds=max_seconds, lease_lost=self.lease.lost, lease_recheck=self.lease.request_recheck)
        finally:
            self.close()

    def close(self) -> None:
        self.lease.close()
        self._revoke()
