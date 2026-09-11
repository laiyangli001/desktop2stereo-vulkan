from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class ArchiveEntry:
    kind: str
    mode: int
    data: bytes | None = None
    target: str | None = None


def _normal_name(name: str) -> str:
    value = name.replace("\\", "/").lstrip("/")
    normalized = posixpath.normpath(value)
    if normalized in ("", "."):
        return ""
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"archive path escapes root: {name}")
    return normalized


def _link_target(name: str, target: str) -> str:
    if not target or target.startswith(("/", "\\")):
        raise ValueError(f"absolute archive link is not allowed: {name} -> {target}")
    return _normal_name(posixpath.join(posixpath.dirname(name), target))


def _archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_zip(path: Path) -> dict[str, ArchiveEntry]:
    entries: dict[str, ArchiveEntry] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = _normal_name(info.filename)
            if not name:
                continue
            mode = (info.external_attr >> 16) & 0o777777
            if info.is_dir() or info.filename.endswith("/"):
                entries[name.rstrip("/")] = ArchiveEntry("dir", mode or 0o755)
            elif (mode & 0o170000) == 0o120000:
                target = archive.read(info).decode("utf-8")
                entries[name] = ArchiveEntry("link", mode or 0o644, target=target)
            else:
                entries[name] = ArchiveEntry("file", mode or 0o644, data=archive.read(info))
    return entries


def _read_tar(path: Path) -> dict[str, ArchiveEntry]:
    entries: dict[str, ArchiveEntry] = {}
    with tarfile.open(path, "r:*") as archive:
        for info in archive.getmembers():
            name = _normal_name(info.name)
            if not name:
                continue
            if info.isdir():
                entries[name] = ArchiveEntry("dir", info.mode or 0o755)
            elif info.isfile():
                stream = archive.extractfile(info)
                if stream is None:
                    raise ValueError(f"cannot read archive file: {info.name}")
                entries[name] = ArchiveEntry("file", info.mode or 0o644, data=stream.read())
            elif info.issym():
                entries[name] = ArchiveEntry("link", info.mode or 0o644, target=info.linkname)
            elif info.islnk():
                entries[name] = ArchiveEntry("link", info.mode or 0o644, target=info.linkname)
            else:
                raise ValueError(f"unsupported archive entry: {info.name}")
    return entries


def _materialize(entries: dict[str, ArchiveEntry], source_root: str, destination: Path) -> None:
    prefix = source_root.rstrip("/") + "/"
    selected = {name: entry for name, entry in entries.items() if name == source_root or name.startswith(prefix)}
    if not selected:
        raise ValueError(f"archive root is missing: {source_root}")

    resolving: set[str] = set()

    def resolve_entry(name: str) -> tuple[str, ArchiveEntry]:
        entry = entries.get(name)
        if entry is None:
            raise ValueError(f"archive link target is missing: {name}")
        if entry.kind != "link":
            return name, entry
        if name in resolving:
            raise ValueError(f"archive link cycle detected at: {name}")
        resolving.add(name)
        target = _link_target(name, entry.target or "")
        result = resolve_entry(target)
        resolving.remove(name)
        return result

    for name in sorted(selected):
        if name == source_root:
            continue
        relative = name[len(prefix) :]
        target_path = destination / Path(*PurePosixPath(relative).parts)
        resolved_name, entry = resolve_entry(name)
        if entry.kind == "dir":
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        if entry.kind != "file" or entry.data is None:
            raise ValueError(f"archive link target is not a regular file: {name} -> {resolved_name}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(entry.data)
        os.chmod(target_path, entry.mode & 0o777 or 0o644)


def _find_root(entries: dict[str, ArchiveEntry], expected: str) -> str:
    if expected in entries:
        return expected
    prefix = expected.rstrip("/") + "/"
    if any(name.startswith(prefix) for name in entries):
        return expected
    raise ValueError(f"archive root is missing: {expected}")


def _validate_runtime(destination: Path, platform: str, expected_version: str) -> None:
    executable = destination / (Path("python.exe") if platform == "windows" else Path("bin/python"))
    if not executable.is_file() or executable.is_symlink():
        raise ValueError(f"runtime executable is missing: {executable}")
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"runtime contains a symbolic link: {path}")
    result = subprocess.run([str(executable), "--version"], capture_output=True, text=True, check=False)
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0 or not output.startswith(f"Python {expected_version}"):
        raise ValueError(f"runtime version check failed: {output or result.returncode}")


def stage_runtime(
    platform: str,
    archive: Path,
    destination: Path,
    package_root: Path,
    source_url: str,
    expected_sha256: str,
    expected_version: str,
) -> None:
    if platform not in {"windows", "linux", "macos"}:
        raise ValueError(f"unsupported platform: {platform}")
    if not archive.is_file():
        raise ValueError(f"runtime archive is missing: {archive}")
    actual_sha256 = _archive_sha256(archive)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(f"runtime archive SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    if destination.exists():
        raise ValueError(f"runtime destination already exists: {destination}")

    entries = _read_zip(archive) if archive.suffix.lower() == ".zip" else _read_tar(archive)
    source_root = "tools" if platform == "windows" else "python"
    source_root = _find_root(entries, source_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{destination.name}-", dir=destination.parent))
    try:
        _materialize(entries, source_root, staging)
        _validate_runtime(staging, platform, expected_version)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
        manifest = {
            "version": 1,
            "platform": platform,
            "python_version": expected_version,
            "source_url": source_url,
            "archive_sha256": expected_sha256.lower(),
        }
        (package_root / "runtime-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage a verified, symlink-free Python runtime into a release package.")
    parser.add_argument("platform", choices=("windows", "linux", "macos"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    stage_runtime(
        args.platform,
        args.archive,
        args.destination,
        args.package_root,
        args.source_url,
        args.sha256,
        args.expected_version,
    )
    print(f"Staged Python {args.expected_version} runtime for {args.platform}: {args.destination}")


if __name__ == "__main__":
    main()
