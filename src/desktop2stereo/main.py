from __future__ import annotations

import sys


if sys.platform == "win32":
    # DPI awareness is first-wins per process.  The runtime_entry import chain
    # (via capture backends / window_control) sets plain per-monitor v1 at
    # import time, which makes GLFW size a borderless fullscreen window down to
    # the *system* scale (e.g. a 1920x1200 window next to a 150% 4K primary
    # becomes 1280x800), leaving the SBS image in the top-left corner.  Request
    # per-monitor v2 *before any module import* so it wins the race.
    from windows_dpi import set_per_monitor_dpi_v2

    set_per_monitor_dpi_v2()


from app_runtime.bootstrap import main


if __name__ == "__main__":
    raise SystemExit(main())
