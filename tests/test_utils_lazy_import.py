import os
import subprocess
import sys
from pathlib import Path

from path_config import APP_ROOT


ROOT = Path(__file__).resolve().parents[1]
SRC = APP_ROOT


def _run_python(code, cwd):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_light_utils_import_does_not_load_settings(tmp_path):
    result = _run_python(
        "from utils import OS_NAME, read_yaml; print(bool(OS_NAME)); print(callable(read_yaml))",
        tmp_path,
    )

    assert result.stdout.splitlines() == ["True", "True"]


def test_gui_import_does_not_validate_stale_display_profile(tmp_path):
    result = _run_python(
        "import gui.gui; "
        "import sys; "
        "print('xr_viewer.core_openxr_vulkan' not in sys.modules); "
        "print('app_runtime.runtime_entry' not in sys.modules)",
        tmp_path,
    )

    assert result.stdout.splitlines() == ["True", "True"]

