"""Explicit v2 launch routing and in-memory parent session ownership."""
from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
import sys

from .portal_access import PortalAccess, PortalCache
from .portal_client import PortalClient, PortalError
from .portal_handoff import accept_handoff, encode_handoff, receive_handoff
from .portal_runtime import PortalRuntime


_PARENT_ACCESS: PortalAccess | None = None


def enabled() -> bool:
    value = os.environ.get("D2S_AUTH_PROTOCOL", "1")
    if value not in {"1", "2"}:
        raise PortalError("auth_protocol_invalid")
    return value == "2"


def _source_imports() -> None:
    """Keep direct main.py and package launch paths on one auth module namespace."""
    application = Path(__file__).resolve().parents[1]
    for path in (application.parent, application):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    parent = importlib.import_module("desktop2stereo")
    current = __package__
    other = "auth" if current == "desktop2stereo.auth" else "desktop2stereo.auth"
    for name, module in list(sys.modules.items()):
        if name == current or name.startswith(current + "."):
            alias = other + name[len(current):]
            previous = sys.modules.get(alias)
            if previous is not None and previous is not module:
                raise PortalError("runtime_namespace_conflict")
            sys.modules[alias] = module
    setattr(parent, "auth", sys.modules["desktop2stereo.auth"])


def make_access() -> PortalAccess:
    from .core_device import get_core_device_key
    from .device import device_identity
    from .public_keys import PUBLIC_KEYS
    identity = device_identity()
    core_key = get_core_device_key()
    client = PortalClient(identity.device_hash, core_key.public_key_b64, PUBLIC_KEYS)
    return PortalAccess(client, PortalCache(client))


async def prepare_runtime_handoff() -> bytes:
    access = _PARENT_ACCESS
    if access is None or access.session is None:
        raise PortalError("login_required")
    # Refresh after time spent in the GUI; credentials stay in process memory.
    session = await asyncio.to_thread(access.heartbeat)
    return encode_handoff(session)


def _launch_gui(args) -> None:
    selection = importlib.import_module("app_runtime.gui_selection")
    selected = selection.MODERN_GUI if args.gui2 else selection.LEGACY_GUI if args.gui else selection.read_startup_gui()
    module = importlib.import_module("gui2.gui" if selected == selection.MODERN_GUI else "gui.gui")
    module.main()


def run(args, *, access_factory=None, launcher_factory=None, runtime_factory=None,
        gui_runner=None, stdin=None, instance_factory=None) -> int:
    global _PARENT_ACCESS
    _source_imports()
    if args.runtime:
        stream = stdin if stdin is not None else getattr(sys.stdin, "buffer", None)
        if stream is None:
            raise PortalError("handoff_pipe_missing")
        message = receive_handoff(stream)
        access = (access_factory or make_access)()
        access = accept_handoff(access.client, access.cache, message)
        del message
        return (runtime_factory or PortalRuntime)(access).run(max_seconds=args.runtime_seconds)
    if instance_factory is None:
        from .instance import InstanceLock
        instance_factory = InstanceLock
    lock = instance_factory()
    try:
        lock.acquire()
    except Exception:
        raise PortalError("instance_lock_unavailable") from None
    access = None
    try:
        access = (access_factory or make_access)()
        if launcher_factory is None:
            from .portal_gui import PortalLoginLauncher
            launcher_factory = PortalLoginLauncher
        # Normal startup always asks for account/password; saved grace is an explicit fallback.
        session = launcher_factory(access).run()
        if session is None:
            raise PortalError("login_required")
        _PARENT_ACCESS = access
        (gui_runner or _launch_gui)(args)
        return 0
    finally:
        _PARENT_ACCESS = None
        if access is not None:
            access.session = None
        lock.release()
