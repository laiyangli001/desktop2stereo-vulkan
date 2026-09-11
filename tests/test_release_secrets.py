from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SECRET_SCANNER = ROOT / "scripts" / "scan-release-secrets.mjs"


def _run_scanner(package: Path) -> subprocess.CompletedProcess[str]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for release secret scan tests")
    return subprocess.run(
        [node, str(SECRET_SCANNER), str(package)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _make_package(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "src/Desktop2Stereo.exe").write_bytes(b"launcher")
    (root / "src/runtime.bin").write_bytes(b"runtime")
    (root / "src/auth.py").write_text(
        'headers = {"Authorization": f"Bearer {access_token}"}\n',
        encoding="utf-8",
    )
    (root / "runtime-manifest.json").write_text(
        json.dumps({"version": 1, "platform": "windows", "python_version": "3.12.10"}),
        encoding="utf-8",
    )


def test_release_secret_scan_allows_runtime_code_without_embedded_credentials(tmp_path: Path):
    package = tmp_path / "Desktop2Stereo"
    _make_package(package)

    result = _run_scanner(package)

    assert result.returncode == 0, result.stderr
    assert "secret scan passed" in result.stdout


@pytest.mark.parametrize(
    ("relative", "content", "expected"),
    [
        (".env", "D2S_API_KEY=not-a-release-value\n", "credential-like file"),
        ("src/config.py", 'API_KEY = "ghp_123456789012345678901234567890123456"\n', "GitHub token"),
        ("src/config.py", 'SERVICE_URL = "https://user:password@example.test/api"\n', "basic-auth URL"),
        ("src/config.py", 'ACCESS_TOKEN = "123456789012345678901234"\n', "literal credential value"),
    ],
)
def test_release_secret_scan_rejects_embedded_secret_material(
    tmp_path: Path, relative: str, content: str, expected: str
):
    package = tmp_path / "Desktop2Stereo"
    _make_package(package)
    target = package / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    result = _run_scanner(package)

    assert result.returncode != 0
    assert expected in result.stderr
