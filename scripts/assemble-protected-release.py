"""Assemble and verify a platform release package with protected core assets."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_stager():
    path = ROOT / "scripts" / "stage-protected-core.py"
    spec = importlib.util.spec_from_file_location("stage_protected_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("protected core staging script is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_node(script: str, *args: str) -> None:
    command = ["node", str(ROOT / "scripts" / script), *args]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--private-core", type=Path, required=True)
    parser.add_argument("--native-core", type=Path, required=True)
    parser.add_argument("--platform", choices=("windows", "linux", "macos"), required=True)
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    if not package_root.is_dir():
        raise SystemExit(f"release package does not exist: {package_root}")
    stager = _load_stager()
    target_root = package_root / "src" / "desktop2stereo"
    stager.stage(args.private_core.resolve(), target_root)
    stager.stage_native(args.native_core.resolve(), target_root, args.platform)
    _run_node("write-build-info.mjs", str(package_root))
    _run_node("write-release-manifest.mjs", str(package_root))
    _run_node("verify-release-package.mjs", str(package_root), args.platform, "--require-protected-core")
    print(f"protected release package verified: {package_root} ({args.platform})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
