from __future__ import annotations

import os
import site
import sys
import sysconfig
import warnings
from pathlib import Path


def configure_platform_environment(os_name: str) -> None:
    if os_name == "Darwin":
        _configure_macos_environment()
    elif os_name == "Windows":
        configure_rocm_environment(os_name)


def _configure_macos_environment() -> None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    warnings.filterwarnings(
        "ignore",
        message=".*aten::upsample_bicubic2d.out.*MPS backend.*",
        category=UserWarning,
    )


def configure_rocm_environment(os_name: str, rocm_path: str | os.PathLike[str] | None = None) -> str | None:
    if os_name != "Windows":
        return None

    root = _resolve_rocm_sdk_path(rocm_path)
    if root is None:
        return None

    os.environ["MIOPEN_ENABLE_LOGGING"] = "0"
    os.environ["HIP_PLATFORM"] = "amd"
    os.environ["HIP_PATH"] = str(root)
    os.environ["HIP_CLANG_PATH"] = str(root / "llvm" / "bin")
    os.environ["HIP_INCLUDE_PATH"] = str(root / "include")
    os.environ["HIP_LIB_PATH"] = str(root / "lib")
    os.environ["HIP_DEVICE_LIB_PATH"] = str(root / "lib" / "llvm" / "amdgcn" / "bitcode")
    _prepend_env_paths("PATH", [root / "bin", root / "llvm" / "bin"])
    _prepend_env_paths("CPATH", [root / "include"])
    _prepend_env_paths("LIBRARY_PATH", [root / "lib", root / "lib64"])
    _prepend_env_paths("PKG_CONFIG_PATH", [root / "lib" / "pkgconfig"])
    # The HIP runtime libs (rocblas.dll / libhipblaslt.dll) ship in the
    # sibling _rocm_sdk_libraries package; torch must resolve them from PATH
    # or large GEMMs fall back to slow kernels (40x slowdown observed).
    _libraries = _resolve_rocm_libraries_path(root)
    if _libraries is not None:
        _prepend_env_paths(
            "PATH", [_libraries / "bin", _libraries / "lib"]
        )
        _prepend_env_paths("LIBRARY_PATH", [_libraries / "lib"])
    # Portable MSVC/SDK-free triton JIT: the bundled clang compiles
    # hip_utils against project-local stub headers (msvcstub/) instead of the
    # Windows SDK + MSVC vcruntime (not installed on this machine). INCLUDE
    # and TRITON_NO_WINSDK are process-local; nothing is written to the
    # user/system environment.
    _msvcstub = Path(__file__).resolve().parent.parent / "msvcstub"
    if _msvcstub.is_dir():
        os.environ["INCLUDE"] = str(_msvcstub)
        _msvcstub_lib = _msvcstub / "lib"
        if _msvcstub_lib.is_dir():
            os.environ["LIB"] = str(_msvcstub_lib)
            os.environ["LIBPATH"] = str(_msvcstub_lib)
            _prepend_env_paths("LIBPATH", [_msvcstub_lib])
        os.environ.setdefault("TRITON_NO_WINSDK", "1")
    return str(root)


def _resolve_rocm_libraries_path(root: Path) -> Path | None:
    for candidate in (root.parent / "_rocm_sdk_libraries",):
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved.is_dir() and (resolved / "bin").is_dir():
            return resolved
    return None


def _resolve_rocm_sdk_path(rocm_path: str | os.PathLike[str] | None = None) -> Path | None:
    candidates: list[Path] = []
    if rocm_path:
        candidates.append(Path(rocm_path))
    if os.environ.get("HIP_PATH"):
        candidates.append(Path(os.environ["HIP_PATH"]))
    candidates.extend(_site_package_rocm_candidates())

    for candidate in candidates:
        try:
            root = candidate.resolve()
        except OSError:
            root = candidate
        if _looks_like_rocm_sdk(root):
            return root
    return None


def _site_package_rocm_candidates() -> list[Path]:
    paths: list[Path] = []
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        paths.append(Path(purelib))
    paths.append(Path(sys.prefix) / "Lib" / "site-packages")
    try:
        paths.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass

    candidates = []
    seen = set()
    for base in paths:
        candidate = base / "_rocm_sdk_devel"
        key = os.path.normcase(str(candidate))
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)
    return candidates


def _looks_like_rocm_sdk(path: Path) -> bool:
    return path.is_dir() and (path / "bin").is_dir() and (path / "include").is_dir()


def _prepend_env_paths(name: str, paths: list[Path]) -> None:
    existing = os.environ.get(name, "")
    existing_parts = [part for part in existing.split(os.pathsep) if part]
    existing_keys = {os.path.normcase(os.path.abspath(part)) for part in existing_parts}
    additions = []
    for path in paths:
        value = str(path)
        key = os.path.normcase(os.path.abspath(value))
        if key not in existing_keys:
            additions.append(value)
            existing_keys.add(key)
    os.environ[name] = os.pathsep.join(additions + existing_parts)
