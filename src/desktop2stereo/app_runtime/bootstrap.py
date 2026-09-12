from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import threading
import traceback
from collections.abc import Sequence

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
        lease = RuntimeLease(session) if session.access_token else None
        if lease is not None:
            lease.start()
        _install_crash_logging()
        from .runtime_entry import run_processing_runtime
        try:
            return run_processing_runtime(max_seconds=args.runtime_seconds, lease_lost=lease.lost if lease else None)
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
