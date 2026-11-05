"""Single-message anonymous-pipe handoff; no tokens in argv, environment or files."""
from __future__ import annotations

import asyncio
import json
import secrets
import struct
import threading
import time

from .portal_access import PortalAccess, PortalCache
from .portal_client import PortalClient, PortalError, PortalSession, _object


_MAX_FRAME = 65536


def encode_handoff(session: PortalSession, *, now: int | None = None) -> bytes:
    payload = PortalCache._payload(session)
    payload["access_token"] = session.access_token
    message = {
        "version": 2, "created_at": int(time.time()) if now is None else now,
        "nonce": secrets.token_hex(16),
        "mode": "online" if session.access_token else "offline", "payload": payload,
    }
    try:
        raw = json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if not 0 < len(raw) <= _MAX_FRAME:
            raise ValueError("oversized handoff")
        return struct.pack(">I", len(raw)) + raw
    except (ValueError, TypeError):
        raise PortalError("handoff_invalid") from None


def _read_exact(stream, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = stream.read(size - len(output))
        if not isinstance(chunk, bytes) or not chunk:
            raise PortalError("handoff_truncated")
        output.extend(chunk)
    return bytes(output)


def receive_handoff(stream, *, timeout: float = 10, now: int | None = None) -> dict:
    """Read once and close the channel, even for malformed or rejected messages."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 30:
        raise PortalError("handoff_invalid")
    done = threading.Event()
    result = {}

    def read():
        try:
            length = struct.unpack(">I", _read_exact(stream, 4))[0]
            if not 0 < length <= _MAX_FRAME:
                raise PortalError("handoff_too_large")
            raw = _read_exact(stream, length)
            value = json.loads(raw, object_pairs_hook=_object)
            current = int(time.time()) if now is None else now
            if (not isinstance(value, dict) or set(value) != {"version", "created_at", "nonce", "mode", "payload"} or
                    type(value["version"]) is not int or value["version"] != 2 or
                    type(value["created_at"]) is not int or type(current) is not int or
                    not current - 30 <= value["created_at"] <= current + 5 or
                    not isinstance(value["nonce"], str) or len(value["nonce"]) != 32 or
                    any(c not in "0123456789abcdef" for c in value["nonce"]) or
                    value["mode"] not in {"online", "offline"} or not isinstance(value["payload"], dict)):
                raise PortalError("handoff_invalid")
            result["value"] = value
        except PortalError as error:
            result["error"] = error
        except Exception:
            result["error"] = PortalError("handoff_invalid")
        finally:
            try:
                stream.close()
            except Exception:
                result.setdefault("error", PortalError("handoff_channel_close_failed"))
            finally:
                done.set()

    worker = threading.Thread(target=read, name="D2SPortalHandoff", daemon=True)
    worker.start()
    if not done.wait(timeout):
        # The daemon owns the read end. Runtime exits on timeout rather than blocking on close().
        raise PortalError("handoff_timeout")
    if "error" in result:
        raise result["error"]
    return result["value"]


def accept_handoff(client: PortalClient, cache: PortalCache, message: dict) -> PortalAccess:
    """Trust configuration is resolved by the child, never supplied by the message."""
    access = PortalAccess(client, cache)
    payload = message["payload"]
    try:
        if message["mode"] == "offline":
            if payload.get("access_token") != "":
                raise PortalError("handoff_invalid", clear_credentials=True)
            cached = cache.load()
            if cached is None:
                raise PortalError("offline_unavailable", clear_credentials=True)
            incoming = client.accept(payload, now=cache.now(), cached=True)
            if (incoming.offline_token != cached.offline_token or incoming.core_grant != cached.core_grant or
                    incoming.session_version != cached.session_version or incoming.user_id != cached.user_id):
                raise PortalError("handoff_cache_mismatch", clear_credentials=True)
            access.session = cached
        else:
            incoming = client.accept(payload)
            if not incoming.access_token:
                raise PortalError("handoff_invalid", clear_credentials=True)
            access.session = incoming
            # A server denial clears cache; a temporary failure may use only already valid grace.
            access.heartbeat()
        return access
    except PortalError as error:
        access._deny(error)
        raise
    except Exception:
        access._deny(PortalError("handoff_invalid", clear_credentials=True))
        raise PortalError("handoff_invalid", clear_credentials=True) from None


async def send_handoff(process, frame: bytes) -> None:
    """Use the child's private stdin; the frame is never included in diagnostics."""
    if process.stdin is None:
        raise PortalError("handoff_pipe_missing")
    try:
        process.stdin.write(frame)
        await asyncio.wait_for(process.stdin.drain(), timeout=10)
    except Exception:
        if process.returncode is None:
            process.terminate()
        raise PortalError("handoff_failed") from None
    finally:
        process.stdin.close()
