"""Privacy-preserving, multi-source device identity for license binding."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import platform as platform_module
from pathlib import Path
import re
import subprocess
import unicodedata


FINGERPRINT_VERSION = 2
FINGERPRINT_PROFILE = "optimal"
_MINIMUM_SOURCE_COUNT = 2
_SOURCE_ORDER = (
    "smbios_uuid",
    "machine_guid",
    "io_platform_uuid",
    "machine_id",
    "product_uuid",
    "disk_uuid",
    "cpu_model",
)
_INVALID_VALUES = {
    "",
    "none",
    "null",
    "unknown",
    "unavailable",
    "not available",
    "not applicable",
    "default string",
    "to be filled by o.e.m.",
}


class DeviceIdentityError(RuntimeError):
    """Raised when a trustworthy multi-source identity cannot be collected."""


@dataclass(frozen=True)
class DeviceIdentity:
    """Hashed device identity and non-sensitive diagnostic metadata."""

    device_hash: str
    fingerprint_version: int
    platform: str
    stability: str
    sources: tuple[str, ...]


def device_identity() -> DeviceIdentity:
    """Collect the current platform signals and build the licensing identity."""

    system = platform_module.system()
    return build_device_identity(collect_device_features(system), system=system)


def device_hash() -> str:
    """Return the 64-character hash sent to the authorization server."""

    return device_identity().device_hash


def current_fingerprint_version() -> int:
    """Return the protocol version paired with :func:`device_hash`."""

    return FINGERPRINT_VERSION


def collect_device_features(system: str | None = None) -> dict[str, str]:
    """Collect stable and moderately stable non-network platform signals.

    Low-stability network values such as MAC addresses and IP addresses are
    intentionally excluded from the licensing fingerprint.
    """

    system = system or platform_module.system()
    raw: dict[str, str] = {}
    if system == "Windows":
        _put(raw, "smbios_uuid", _windows_smbios_uuid())
        _put(raw, "machine_guid", _windows_machine_guid())
        _put(raw, "cpu_model", platform_module.processor() or platform_module.uname().processor)
    elif system == "Linux":
        _put(raw, "product_uuid", _read_first_line("/sys/class/dmi/id/product_uuid"))
        _put(raw, "machine_id", _first_available_text(("/etc/machine-id", "/var/lib/dbus/machine-id")))
        _put(raw, "disk_uuid", _command_first_line(["findmnt", "-no", "UUID", "/"]))
        _put(raw, "cpu_model", _linux_cpu_model())
    elif system == "Darwin":
        _put(raw, "io_platform_uuid", _macos_platform_uuid())
        _put(raw, "disk_uuid", _macos_volume_uuid())
        _put(raw, "cpu_model", _command_first_line(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform_module.processor())
    else:
        raise DeviceIdentityError(f"不支持的平台，无法生成稳定设备指纹：{system or 'unknown'}")
    return raw


def build_device_identity(features: dict[str, str], *, system: str | None = None) -> DeviceIdentity:
    """Normalize, filter, order, and hash a feature set deterministically."""

    system = system or platform_module.system()
    normalized = {
        name: value
        for name, value in ((name, _normalize_feature(value)) for name, value in features.items())
        if name in _SOURCE_ORDER and _is_valid_feature(value)
    }
    ordered = [(name, normalized[name]) for name in _SOURCE_ORDER if name in normalized]
    if len(ordered) < _MINIMUM_SOURCE_COUNT:
        raise DeviceIdentityError("可用稳定设备信号不足，已阻止生成不可靠的设备指纹")

    material = [
        "desktop2stereo",
        f"fingerprint_version={FINGERPRINT_VERSION}",
        f"profile={FINGERPRINT_PROFILE}",
        f"platform={_normalize_feature(system)}",
        *(f"{name}={value}" for name, value in ordered),
    ]
    digest = hashlib.sha256("\n".join(material).encode("utf-8")).hexdigest()
    return DeviceIdentity(
        device_hash=digest,
        fingerprint_version=FINGERPRINT_VERSION,
        platform=_normalize_feature(system),
        stability=FINGERPRINT_PROFILE,
        sources=tuple(name for name, _ in ordered),
    )


def _put(target: dict[str, str], name: str, value: str | None) -> None:
    normalized = _normalize_feature(value)
    if _is_valid_feature(normalized):
        target[name] = normalized


def _normalize_feature(value: object) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return " ".join(normalized.split())


def _is_valid_feature(value: str) -> bool:
    if value in _INVALID_VALUES:
        return False
    compact = re.sub(r"[^0-9a-f]", "", value)
    if len(compact) >= 8:
        # Firmware commonly exposes repeated sentinel values instead of an ID.
        if len(set(compact)) == 1:
            return False
        # Some BIOSes report a zero-filled value with a trailing status nibble.
        if compact.endswith("a") and set(compact[:-1]) == {"0"}:
            return False
    return bool(value)


def _read_first_line(path: str) -> str | None:
    try:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                return line.strip()
    except OSError:
        return None
    return None


def _first_available_text(paths: tuple[str, ...]) -> str | None:
    for path in paths:
        value = _read_first_line(path)
        if value:
            return value
    return None


def _command_first_line(command: list[str]) -> str | None:
    output = _run_command(command)
    if not output:
        return None
    return next((line.strip() for line in output.splitlines() if line.strip()), None)


def _run_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _windows_machine_guid() -> str | None:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
            value, _ = winreg.QueryValueEx(key, "MachineGuid")
            return str(value)
    except (OSError, ImportError):
        return None


def _windows_smbios_uuid() -> str | None:
    output = _run_command(["wmic", "csproduct", "get", "uuid"])
    if not output:
        output = _run_command(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance -ClassName Win32_ComputerSystemProduct | Select-Object -ExpandProperty UUID",
            ]
        )
    if not output:
        return None
    for line in output.splitlines():
        value = line.strip()
        if value.casefold() != "uuid" and _is_valid_feature(_normalize_feature(value)):
            return value
    return None


def _linux_cpu_model() -> str | None:
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return platform_module.processor() or platform_module.uname().processor
    for line in lines:
        key, separator, value = line.partition(":")
        if separator and key.strip().casefold() in {"model name", "hardware", "processor"} and value.strip():
            return value.strip()
    return platform_module.processor() or platform_module.uname().processor


def _macos_platform_uuid() -> str | None:
    output = _run_command(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"])
    if not output:
        return None
    for line in output.splitlines():
        if "IOPlatformUUID" in line:
            return line.split("=", 1)[-1].strip().strip('"')
    return None


def _macos_volume_uuid() -> str | None:
    output = _run_command(["diskutil", "info", "/"])
    if not output:
        return None
    for line in output.splitlines():
        if "UUID" in line.upper() and ":" in line:
            value = line.split(":", 1)[1].strip()
            if _is_valid_feature(_normalize_feature(value)):
                return value
    return None
