from __future__ import annotations

import sys

from windows_dpi import set_per_monitor_dpi_v2


def test_set_per_monitor_dpi_v2_returns_bool() -> None:
    # On Windows this mutates the process DPI awareness (first-wins); the only
    # contract we can assert portably is that it returns a bool and never
    # raises on the platforms the app targets.
    result = set_per_monitor_dpi_v2()
    assert isinstance(result, bool)


def test_set_per_monitor_dpi_v2_is_noop_off_windows() -> None:
    if sys.platform != "win32":
        assert set_per_monitor_dpi_v2() is False
