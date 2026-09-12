"""Redacted launcher diagnostics for support and authorization troubleshooting."""

from __future__ import annotations

import json
import os
import platform
import re
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_DEFAULT_API_BASE_URL = "https://100393.com/api/v1"
_DEFAULT_APP_VERSION = "3.0beta"
_SAFE_GIT_SHA = re.compile(r"^[0-9a-fA-F]{7,64}$")
_SAFE_APP_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")


def authorization_diagnostics(*, error_code: str | None = None, request_id: str | None = None) -> dict[str, Any]:
    """Return support metadata without credentials, raw hardware IDs, or full URLs."""

    api_host, server_environment = _api_target()
    result: dict[str, Any] = {
        "product": "desktop2stereo",
        "app_version": _app_version(),
        "git_sha": _git_sha(),
        "client_platform": platform.system().casefold() or "unknown",
        "server_host": api_host,
        "server_environment": server_environment,
    }
    if isinstance(error_code, str) and error_code.strip():
        result["error_code"] = error_code.strip()
    if isinstance(request_id, str) and request_id.strip():
        result["request_id"] = request_id.strip()
    return result


def format_authorization_diagnostics(*, error_code: str | None = None, request_id: str | None = None) -> str:
    """Serialize redacted diagnostics for clipboard/export support workflows."""

    return json.dumps(
        authorization_diagnostics(error_code=error_code, request_id=request_id),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )


def _git_sha() -> str:
    value = os.environ.get("D2S_GIT_SHA") or os.environ.get("GITHUB_SHA") or _build_info().get("git_sha", "")
    normalized = value.strip() if isinstance(value, str) else ""
    return normalized.lower() if _SAFE_GIT_SHA.fullmatch(normalized) else "unknown"


def _app_version() -> str:
    candidates = [os.environ.get("D2S_APP_VERSION", ""), _build_info().get("app_version", "")]
    source_path = Path(__file__).resolve().parents[2] / "utils" / "app_info.py"
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError:
        source = ""
    match = re.search(r"^VERSION\s*=\s*[\"']([^\"']+)[\"']", source, re.MULTILINE)
    if match:
        candidates.append(match.group(1))
    for value in candidates:
        if isinstance(value, str) and _SAFE_APP_VERSION.fullmatch(value.strip()):
            return value.strip()
    return _DEFAULT_APP_VERSION


def _build_info() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[3] / "build-info.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _api_target() -> tuple[str, str]:
    value = os.environ.get("D2S_API_BASE_URL", _DEFAULT_API_BASE_URL).strip().rstrip("/")
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
    except ValueError:
        return "invalid", "invalid"
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not host:
        return "invalid", "invalid"
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        return host.casefold(), "invalid"
    return host.casefold(), _classify_server_environment(host)


def _classify_server_environment(host: str) -> str:
    if host.casefold() == "localhost":
        return "local"
    try:
        if ip_address(host).is_loopback:
            return "local"
    except ValueError:
        pass
    if host.casefold() == "100393.com":
        return "production"
    return "custom"
