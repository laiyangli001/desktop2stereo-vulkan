"""Stage an encrypted private-core resource into a release tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def stage(source_root: Path, target_root: Path) -> Path:
    source = source_root / "private" / "parallax-core" / "encrypted" / "parallax-core.enc"
    manifest = source_root / "private" / "parallax-core" / "manifest" / "core.json"
    if not source.is_file() or not manifest.is_file():
        raise FileNotFoundError("private core encrypted resource or manifest is missing")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    expected = metadata.get("resource_sha256")
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if not isinstance(expected, str) or expected.casefold() != actual:
        raise ValueError("private core resource hash does not match its manifest")
    target = target_root / "protected" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-core", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, default=Path("src/desktop2stereo"))
    args = parser.parse_args()
    target = stage(args.private_core, args.target_root)
    print(f"staged protected core resource: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
