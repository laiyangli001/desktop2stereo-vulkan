"""Windows process DPI-awareness helpers (first-wins, prefer per-monitor v2).

The local Vulkan/GLFW viewer creates borderless fullscreen windows.  On
Windows, GLFW derives the created window's framebuffer size from the process
DPI context.  With a system-aware *or* per-monitor *v1* context, a window
created for a monitor whose scale differs from the primary/system scale is
incorrectly scaled down -- for example a 1920x1200 window on a 100%-scale
secondary monitor next to a 150%-scale 4K primary is reported as 1280x800, so
the half-SBS image sits in the screen's top-left instead of covering it.
Per-monitor *v2* reports each monitor's true scale and fixes the framing.

DPI awareness is first-wins per process, so call
:func:`set_per_monitor_dpi_v2` as early as possible (ideally the first thing in
``main``) and before any GLFW/Vulkan window is created.
"""

from __future__ import annotations

import ctypes
import sys


# DPI_AWARENESS_CONTEXT values (Windows 10 1607+).  These are sentinel handle
# values: -4 = PER_MONITOR_AWARE_V2, -2 = SYSTEM_AWARE, -3 = UNAWARE.
_PER_MONITOR_AWARE_V2 = -4


def _is_dpi_aware() -> bool:
    """Return True when the process is per-monitor (v1/v2) DPI aware."""
    try:
        kernel32 = ctypes.windll.kernel32
        shcore = ctypes.windll.shcore
        value = ctypes.c_int(0)
        get = shcore.GetProcessDpiAwareness
        get.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        get.restype = ctypes.c_long
        handle = ctypes.c_void_p(kernel32.GetCurrentProcess())
        if int(get(handle, ctypes.byref(value))) < 0:
            return False
        # 0 = unaware, 1 = system aware, 2 = per-monitor aware.
        return int(value.value) >= 2
    except Exception:
        return True


def set_per_monitor_dpi_v2() -> bool:
    """Request per-monitor v2 DPI awareness; safe to call repeatedly.

    DPI awareness is first-wins per process; a later call that is not the
    winner silently fails, so this always returns True when the process is
    (or becomes) per-monitor DPI aware, and False only when the platform is not
    Windows or awareness could not be raised.
    """
    if sys.platform != "win32":
        return False
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.SetProcessDpiAwarenessContext.restype = ctypes.c_bool
        if user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(_PER_MONITOR_AWARE_V2)
        ):
            return True
        # A prior call already set the process.  Accept per-monitor awareness;
        # anything less would rescale the viewer window.
        return _is_dpi_aware()
    except Exception:
        pass
    try:
        # Fallback: per-monitor v1 (Win 8.1+).
        return ctypes.windll.shcore.SetProcessDpiAwareness(2) >= 0
    except Exception:
        try:
            # Last resort: system aware.
            ctypes.windll.user32.SetProcessDPIAware()
            return True
        except Exception:
            return False
