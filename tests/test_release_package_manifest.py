from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_WRITER = ROOT / "scripts" / "write-release-manifest.mjs"
PACKAGE_VERIFIER = ROOT / "scripts" / "verify-release-package.mjs"
BUILD_INFO_WRITER = ROOT / "scripts" / "write-build-info.mjs"


def _run_node(script: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for release package tests")
    return subprocess.run([node, str(script), *args], cwd=cwd, text=True, capture_output=True, check=False)


def _make_windows_package(root: Path) -> None:
    for relative in (
        "src/Desktop2Stereo.exe", "src/python3/python.exe", "src/desktop2stereo/main.py",
        "src/desktop2stereo/icon/icon-256x256.ico", "src/desktop2stereo/icon/icon-256x256.png",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("ascii"))
    (root / "runtime-manifest.json").write_text(
        json.dumps({
            "version": 1,
            "platform": "windows",
            "python_version": "3.12.10",
            "source_url": "https://example.invalid/python.zip",
            "archive_sha256": "a" * 64,
        }),
        encoding="utf-8",
    )
    (root / "build-info.json").write_text(
        json.dumps({"version": 1, "product": "desktop2stereo", "app_version": "3.0beta", "git_sha": "a" * 40}),
        encoding="utf-8",
    )


def _make_posix_package(root: Path, platform: str) -> None:
    binary = "src/Desktop2Stereo" if platform == "linux" else "src/Desktop2Stereo-macos"
    for relative in (
        binary, "src/python3/bin/python", "src/desktop2stereo/main.py",
        "src/desktop2stereo/icon/icon-256x256.ico", "src/desktop2stereo/icon/icon-256x256.png",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("ascii"))
    (root / "runtime-manifest.json").write_text(
        json.dumps({
            "version": 1,
            "platform": platform,
            "python_version": "3.12.14",
            "source_url": "https://example.invalid/python.tar.gz",
            "archive_sha256": "b" * 64,
        }),
        encoding="utf-8",
    )
    (root / "build-info.json").write_text(
        json.dumps({"version": 1, "product": "desktop2stereo", "app_version": "3.0beta", "git_sha": "b" * 40}),
        encoding="utf-8",
    )


def test_build_info_writer_records_release_version_and_git_sha(tmp_path: Path):
    package = tmp_path / "Desktop2Stereo"
    package.mkdir()
    result = _run_node(BUILD_INFO_WRITER, str(package), cwd=ROOT)
    assert result.returncode == 0, result.stderr
    metadata = json.loads((package / "build-info.json").read_text(encoding="utf-8"))
    assert metadata["version"] == 1
    assert metadata["product"] == "desktop2stereo"
    assert metadata["app_version"] == "3.0beta"
    assert len(metadata["git_sha"]) >= 7


def test_release_package_manifest_is_verified_and_detects_tampering(tmp_path: Path):
    package = tmp_path / "Desktop2Stereo"
    _make_windows_package(package)
    written = _run_node(MANIFEST_WRITER, str(package), cwd=ROOT)
    assert written.returncode == 0, written.stderr

    verified = _run_node(PACKAGE_VERIFIER, str(package), "windows", cwd=ROOT)
    assert verified.returncode == 0, verified.stderr
    assert "verification passed" in verified.stdout

    (package / "src/desktop2stereo/main.py").write_bytes(b"tampered")
    rejected = _run_node(PACKAGE_VERIFIER, str(package), "windows", cwd=ROOT)
    assert rejected.returncode != 0
    assert "hash or size mismatch" in rejected.stderr


def test_release_package_rejects_private_key_material(tmp_path: Path):
    package = tmp_path / "Desktop2Stereo"
    _make_windows_package(package)
    private_file = package / "src/key.txt"
    private_file.write_text("-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----\n", encoding="ascii")
    written = _run_node(MANIFEST_WRITER, str(package), cwd=ROOT)
    assert written.returncode == 0, written.stderr

    rejected = _run_node(PACKAGE_VERIFIER, str(package), "windows", cwd=ROOT)
    assert rejected.returncode != 0
    assert "private key material" in rejected.stderr


def test_release_package_allows_public_pem_certificates(tmp_path: Path):
    package = tmp_path / "Desktop2Stereo"
    _make_windows_package(package)
    (package / "src/cacert.pem").write_text(
        "-----BEGIN CERTIFICATE-----\npublic-certificate\n-----END CERTIFICATE-----\n",
        encoding="ascii",
    )
    written = _run_node(MANIFEST_WRITER, str(package), cwd=ROOT)
    assert written.returncode == 0, written.stderr

    verified = _run_node(PACKAGE_VERIFIER, str(package), "windows", cwd=ROOT)
    assert verified.returncode == 0, verified.stderr


@pytest.mark.parametrize("platform", ["linux", "macos"])
def test_release_package_verifies_posix_launcher_layout(tmp_path: Path, platform: str):
    if platform == "linux" and sys.platform in {"win32", "darwin"}:
        pytest.skip("Linux release layout uses case-distinct paths on case-sensitive filesystems")
    package = tmp_path / "Desktop2Stereo"
    _make_posix_package(package, platform)
    written = _run_node(MANIFEST_WRITER, str(package), cwd=ROOT)
    assert written.returncode == 0, written.stderr

    verified = _run_node(PACKAGE_VERIFIER, str(package), platform, cwd=ROOT)
    assert verified.returncode == 0, verified.stderr
