from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SBOM_WRITER = ROOT / "scripts" / "write-release-sbom.mjs"


def _run_node(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for release SBOM tests")
    return subprocess.run([node, str(SBOM_WRITER), *args], cwd=cwd, text=True, capture_output=True, check=False)


def test_release_sbom_resolves_recursive_pinned_requirements(tmp_path: Path):
    child = tmp_path / "child.txt"
    child.write_text("urllib3==2.2.2\n", encoding="utf-8")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("requests==2.32.3\n-r child.txt\nREQUESTS==2.32.3\n", encoding="utf-8")
    output = tmp_path / "sbom.cdx.json"

    result = _run_node(str(output), str(requirements), cwd=ROOT)
    assert result.returncode == 0, result.stderr
    bom = json.loads(output.read_text(encoding="utf-8"))
    assert bom["bomFormat"] == "CycloneDX"
    assert bom["specVersion"] == "1.5"
    assert [item["purl"] for item in bom["components"]] == [
        "pkg:pypi/requests@2.32.3",
        "pkg:pypi/urllib3@2.2.2",
    ]


def test_release_sbom_rejects_unpinned_requirement(tmp_path: Path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("pygltflib>=1.16.5\n", encoding="utf-8")
    result = _run_node(str(tmp_path / "sbom.cdx.json"), str(requirements), cwd=ROOT)
    assert result.returncode != 0
    assert "is not pinned with == or a sha256-pinned direct URL" in result.stderr


def test_release_sbom_records_sha256_pinned_direct_url(tmp_path: Path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "migraphx @ https://example.com/migraphx-0.1.2+multiarch-cp312-cp312-win_amd64.whl#sha256="
        + "a" * 64
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "sbom.cdx.json"

    result = _run_node(str(output), str(requirements), cwd=ROOT)
    assert result.returncode == 0, result.stderr
    component = json.loads(output.read_text(encoding="utf-8"))["components"][0]
    assert component["name"] == "migraphx"
    assert component["version"] == "0.1.2+multiarch"
    assert component["properties"] == [{"name": "integrity", "value": "sha256:" + "a" * 64}]


def test_release_sbom_rejects_unhashed_direct_url(tmp_path: Path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("migraphx @ https://example.com/migraphx.whl\n", encoding="utf-8")
    result = _run_node(str(tmp_path / "sbom.cdx.json"), str(requirements), cwd=ROOT)
    assert result.returncode != 0
    assert "direct URL must include #sha256" in result.stderr
