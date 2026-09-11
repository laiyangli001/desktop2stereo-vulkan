from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .probe import build_capability_report
from .gui_selection import LEGACY_GUI, MODERN_GUI, read_startup_gui


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
        try:
            if lease is not None:
                lease.start()
            from .runtime_entry import run_processing_runtime
            return run_processing_runtime(
                max_seconds=args.runtime_seconds,
                lease_lost=lease.lost if lease else None,
                lease_recheck=lease.request_recheck if lease else None,
            )
        except AuthError as exc:
            print(f"[AUTH] {exc} ({exc.code})", file=sys.stderr, flush=True)
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
