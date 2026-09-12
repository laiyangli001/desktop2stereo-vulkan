"""Platform token storage for the authentication launcher."""

from __future__ import annotations

import base64
import ctypes
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


SERVICE_NAME = "Desktop2Stereo"
ACCOUNT_NAME = "refresh-token"


class TokenStore:
    """Store refresh tokens in the platform credential store when available."""

    def __init__(self, app_dir: str | os.PathLike | None = None):
        self.app_dir = Path(app_dir or Path.home() / ".desktop2stereo")
        self.fallback_path = self.app_dir / "auth-session.json"

    def load(self) -> dict | None:
        raw = self._read_secure()
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def save(self, session: dict) -> bool:
        if not isinstance(session, dict):
            return False
        # Access tokens are process-memory credentials and must never be
        # persisted, even when a caller accidentally includes one.
        persisted = {key: value for key, value in session.items() if key != "access_token"}
        try:
            raw = json.dumps(persisted, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return False
        return self._write_secure(raw)

    def clear(self) -> None:
        system = platform.system()
        if system == "Windows":
            self._windows_delete()
        elif system == "Darwin":
            self._run_security(["delete-generic-password", "-s", SERVICE_NAME, "-a", ACCOUNT_NAME])
        elif shutil.which("secret-tool"):
            self._run_security(["clear", "service", SERVICE_NAME, "account", ACCOUNT_NAME])
        try:
            self.fallback_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _read_secure(self) -> str | None:
        system = platform.system()
        if system == "Windows":
            value = self._windows_read()
        elif system == "Darwin":
            value = self._run_security(["find-generic-password", "-s", SERVICE_NAME, "-a", ACCOUNT_NAME, "-w"])
        elif shutil.which("secret-tool"):
            value = self._run_security(["lookup", "service", SERVICE_NAME, "account", ACCOUNT_NAME])
        else:
            value = None
        if value:
            return value
        return None

    def _write_secure(self, raw: str) -> bool:
        system = platform.system()
        if system == "Windows" and self._windows_write(raw):
            return True
        if system == "Darwin":
            self._run_security(["delete-generic-password", "-s", SERVICE_NAME, "-a", ACCOUNT_NAME])
            if self._run_security(["add-generic-password", "-s", SERVICE_NAME, "-a", ACCOUNT_NAME, "-w", raw]) is not None:
                return True
        if shutil.which("secret-tool") and self._run_security(["store", "--label", SERVICE_NAME, "service", SERVICE_NAME, "account", ACCOUNT_NAME], raw) is not None:
            return True
        # Never persist access or refresh tokens in an unencrypted file. A
        # login remains usable for the current process, but the next launch
        # will require authentication when no OS credential store exists.
        return False

    @staticmethod
    def _run_security(args: list[str], stdin: str | None = None) -> str | None:
        try:
            result = subprocess.run(
                (["security"] if platform.system() == "Darwin" else ["secret-tool"]) + args,
                input=stdin,
                text=True,
                capture_output=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    def _windows_read(self) -> str | None:
        # DPAPI is intentionally kept behind a tiny ctypes adapter so the
        # launcher has no third-party dependency.
        try:
            protected = base64.b64decode(self.fallback_path.read_text(encoding="ascii"))
            return self._dpapi_unprotect(protected)
        except (OSError, ValueError, TypeError, ctypes.ArgumentError):
            return None

    def _windows_write(self, raw: str) -> bool:
        try:
            protected = self._dpapi_protect(raw.encode("utf-8"))
            self.app_dir.mkdir(parents=True, exist_ok=True)
            self.fallback_path.write_text(base64.b64encode(protected).decode("ascii"), encoding="ascii")
            return True
        except (OSError, ValueError, TypeError, ctypes.ArgumentError):
            return False

    @staticmethod
    def _dpapi_protect(value: bytes) -> bytes:
        class Blob(ctypes.Structure):
            _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

        source = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
        input_blob = Blob(len(value), source)
        output_blob = Blob()
        if not ctypes.windll.crypt32.CryptProtectData(ctypes.byref(input_blob), "Desktop2Stereo", None, None, None, 0, ctypes.byref(output_blob)):
            raise OSError("CryptProtectData failed")
        try:
            return ctypes.string_at(output_blob.pbData, output_blob.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(output_blob.pbData)

    @staticmethod
    def _dpapi_unprotect(value: bytes) -> str:
        class Blob(ctypes.Structure):
            _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

        source = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
        input_blob = Blob(len(value), source)
        output_blob = Blob()
        if not ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(input_blob), None, None, None, None, 0, ctypes.byref(output_blob)):
            raise OSError("CryptUnprotectData failed")
        try:
            return ctypes.string_at(output_blob.pbData, output_blob.cbData).decode("utf-8")
        finally:
            ctypes.windll.kernel32.LocalFree(output_blob.pbData)

    def _windows_delete(self) -> None:
        try:
            self.fallback_path.unlink(missing_ok=True)
        except OSError:
            pass
