from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage-python-runtime.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage_python_runtime", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _python_version() -> str:
    return ".".join(str(part) for part in sys.version_info[:3])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_windows_runtime_zip_is_verified_and_staged_without_symlinks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    archive = tmp_path / "python.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("tools/python.exe", b"fake windows runtime")

    package = tmp_path / "package"
    destination = package / "src/python3"
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: module.subprocess.CompletedProcess(
            args, 0, stdout=f"Python {_python_version()}\\n", stderr=""
        ),
    )
    module.stage_runtime("windows", archive, destination, package, "https://example.invalid/python.zip", _sha256(archive), _python_version())

    assert (destination / "python.exe").is_file()
    assert not any(path.is_symlink() for path in destination.rglob("*"))
    assert (package / "runtime-manifest.json").is_file()


def test_linux_runtime_tar_links_are_materialized_as_regular_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    archive = tmp_path / "python.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        info = tarfile.TarInfo("python/bin/python3.12")
        info.mode = 0o755
        data = b"fake linux runtime"
        info.size = len(data)
        output.addfile(info, __import__("io").BytesIO(data))
        link = tarfile.TarInfo("python/bin/python")
        link.type = tarfile.SYMTYPE
        link.linkname = "python3.12"
        output.addfile(link)

    package = tmp_path / "package"
    destination = package / "src/python3"
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: module.subprocess.CompletedProcess(
            args, 0, stdout=f"Python {_python_version()}\\n", stderr=""
        ),
    )
    module.stage_runtime("linux", archive, destination, package, "https://example.invalid/python.tar.gz", _sha256(archive), _python_version())

    assert (destination / "bin/python").is_file()
    assert not (destination / "bin/python").is_symlink()
    assert os.access(destination / "bin/python", os.X_OK)


def test_runtime_archive_hash_mismatch_is_rejected(tmp_path: Path):
    module = _load_module()
    archive = tmp_path / "python.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("tools/python.exe", b"not a runtime")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.stage_runtime(
            "windows",
            archive,
            tmp_path / "package/src/python3",
            tmp_path / "package",
            "https://example.invalid/python.zip",
            "0" * 64,
            _python_version(),
        )
