from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import threading
import traceback
from collections.abc import Sequence
from pathlib import Path

from .probe import build_capability_report
from .gui_selection import LEGACY_GUI, MODERN_GUI, read_startup_gui


def _install_crash_logging() -> None:
    """Dump any fatal path (exception, thread exception, native signal) to disk."""
    crash_path = os.path.join(
        os.environ.get("D2S_CRASH_LOG_DIR", os.getcwd()), "runtime-crash.log"
    )

    def _dump(payload: str) -> None:
        try:
            with open(crash_path, "a", encoding="utf-8") as fh:
                fh.write(payload)
                fh.write("\n")
        except Exception:
            pass
        sys.stderr.write(payload + "\n")
        sys.stderr.flush()

    def excepthook(exc_type, exc_value, exc_tb):
        _dump("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))

    def thread_hook(args):
        _dump(
            "Thread exception in "
            + str(getattr(args, "thread", None))
            + ":\n"
            + "".join(
                traceback.format_exception(
                    args.exc_type, args.exc_value, args.exc_traceback
                )
            )
        )

    sys.excepthook = excepthook
    threading.excepthook = thread_hook
    try:
        import signal

        faulthandler.register(
            getattr(signal, "SIGBREAK", signal.SIGTERM),
            file=open(crash_path, "a", encoding="utf-8"),
        )
    except Exception:
        pass
    faulthandler.enable(file=open(crash_path, "a", encoding="utf-8"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Desktop2Stereo Python Vulkan runtime")
    parser.add_argument("--probe", action="store_true", help="Print a JSON capability report and exit")
    parser.add_argument("--version", action="store_true", help="Print the project version and exit")
    parser.add_argument("--gui", action="store_true", help="Launch the Desktop2Stereo Flet GUI")
    parser.add_argument("--gui2", action="store_true", help="Launch the isolated Desktop2Stereo GUI2")
    parser.add_argument("--runtime", action="store_true", help="Run the migrated processing runtime")
    parser.add_argument(
        "--runtime-seconds",
        type=float,
        default=None,
        help="Stop the processing runtime after the specified duration (smoke testing)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Triton warns "Failed to find MSVC" on machines without Visual Studio
    # Build Tools; the app handles this by pointing CC at Triton's bundled
    # TinyCC, so the warning is noise. Suppress it app-wide (scoped to the
    # triton.windows_utils module only).
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="Failed to find MSVC.*",
        category=UserWarning,
        module=r"triton\.windows_utils",
    )
    if args.version:
        print("desktop2steoro-vulkan 0.1.0")
        return 0
    if args.probe:
        print(json.dumps(build_capability_report(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.runtime:
        try:
            from desktop2stereo.auth.gate import validate_saved_authentication
            from desktop2stereo.auth.lease import RuntimeLease
            from desktop2stereo.auth.client import AuthError
        except ModuleNotFoundError:
            from auth.gate import validate_saved_authentication
            from auth.lease import RuntimeLease
            from auth.client import AuthError

        try:
            session = validate_saved_authentication()
        except AuthError as exc:
            print(f"[AUTH] {exc} ({exc.code})", file=sys.stderr, flush=True)
            return 1
        lease = RuntimeLease(session) if _uses_online_lease(session) else None
        _install_crash_logging()
        try:
            if lease is not None:
                lease.start()
            _configure_protected_parallax_core(session, lease)
            from .runtime_entry import run_processing_runtime
            return run_processing_runtime(
                max_seconds=args.runtime_seconds,
                lease_lost=lease.lost if lease else None,
                lease_recheck=lease.request_recheck if lease else None,
            )
        except AuthError as exc:
            print(f"[AUTH] {exc} ({exc.code})", file=sys.stderr, flush=True)
            return 1
        except BaseException:
            payload = "FATAL: " + traceback.format_exc()
            sys.stderr.write(payload)
            sys.stderr.flush()
            try:
                with open(
                    os.path.join(
                        os.environ.get("D2S_CRASH_LOG_DIR", os.getcwd()),
                        "runtime-crash.log",
                    ),
                    "a",
                    encoding="utf-8",
                ) as fh:
                    fh.write(payload + "\n")
            except Exception:
                pass
            return 1
        finally:
            if lease is not None:
                lease.close()
    selected_gui = (
        MODERN_GUI if args.gui2
        else LEGACY_GUI if args.gui
        else read_startup_gui()
    )
    # Authenticate before importing GUI1/GUI2. The login launcher is an
    # independent Flet application and does not load Torch, CUDA, Vulkan, or
    # either runtime GUI until the server accepts the session.
    try:
        from desktop2stereo.auth.gate import require_authentication
        from desktop2stereo.auth.client import AuthError
    except ModuleNotFoundError:
        # Keep direct `python src/desktop2stereo/main.py` compatible with the
        # existing source launch path, where sibling packages are top-level.
        from auth.gate import require_authentication
        from auth.client import AuthError

    try:
        require_authentication()
    except AuthError as exc:
        print(f"[AUTH] {exc} ({exc.code})", file=sys.stderr, flush=True)
        return 1
    if selected_gui == MODERN_GUI:
        from gui2.gui import main as gui2_main

        gui2_main()
        return 0
    from gui.gui import main as gui_main

    gui_main()
    return 0


def _uses_online_lease(session) -> bool:
    """Only online licenses need a server-backed runtime lease."""

    if not session.access_token:
        return False
    selected_id = session.selected_license_id
    if not isinstance(selected_id, str) or not selected_id.strip():
        return True
    for item in session.licenses:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        if item["id"] == selected_id:
            mode = item.get("mode", "online")
            return not isinstance(mode, str) or mode.casefold() == "online"
    # Keep legacy sessions fail-safe until their next status refresh.
    return True


def _configure_protected_parallax_core(session, lease) -> None:
    """Authorize the parallax implementation before importing the runtime."""

    resource_path = os.environ.get("D2S_PARALLAX_CORE_RESOURCE", "").strip()
    if not resource_path:
        resource_path = str(Path(__file__).resolve().parents[1] / "protected" / "parallax-core.enc")
    if not Path(resource_path).is_file():
        raise AuthError("3D 核心资源不可用", "core_resource_unavailable")
    grant_jws = None
    expected_device_hash = None
    if lease is not None:
        payload = lease.core_grant
        if isinstance(payload, dict):
            grant_jws = payload.get("grant")
        expected_device_hash = lease.device
    elif isinstance(getattr(session, "core_grant", None), str):
        grant_jws = session.core_grant
        try:
            from desktop2stereo.auth.device import device_identity
        except ModuleNotFoundError:
            from auth.device import device_identity
        expected_device_hash = device_identity().device_hash
    if not isinstance(grant_jws, str) or grant_jws.count(".") != 2:
        raise AuthError("3D 核心授权不可用", "core_grant_unavailable")
    try:
        from desktop2stereo.auth.clock import TrustedClock
        from desktop2stereo.auth.core import ProtectedCoreError
        from desktop2stereo.stereo_runtime.parallax import configure_protected_parallax_core
    except ModuleNotFoundError:
        from auth.clock import TrustedClock
        from auth.core import ProtectedCoreError
        from stereo_runtime.parallax import configure_protected_parallax_core
    try:
        configure_protected_parallax_core(
            resource_path,
            grant_jws,
            now=TrustedClock().now(),
            expected_device_hash=expected_device_hash,
            native_path=os.environ.get("D2S_PARALLAX_CORE_NATIVE", "").strip() or None,
            require_native=True,
        )
    except ProtectedCoreError as exc:
        raise AuthError("3D 核心未开启", "core_unavailable") from exc
