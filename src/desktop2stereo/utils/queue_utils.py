from __future__ import annotations

import queue
from typing import Any, Callable


def _release_item(item: Any) -> None:
    """Release an optional borrowed capture resource before dropping an item."""
    if isinstance(item, tuple) and len(item) == 2:
        item = item[0]
    resource = getattr(item, "native_resource", None)
    release = getattr(resource, "release", None)
    if callable(release):
        try:
            release()
        except Exception:
            pass
    direct = getattr(item, "viewer_frame_direct", None)
    release_direct = getattr(direct, "_release_direct", None)
    if callable(release_direct):
        try:
            release_direct()
        except Exception:
            pass


def put_latest(q: queue.Queue, item: Any) -> None:
    """Keep only the newest item without blocking producer threads."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                _release_item(q.get_nowait())
            except queue.Empty:
                return


def clear_nonblocking(q: queue.Queue) -> None:
    while True:
        try:
            _release_item(q.get_nowait())
        except queue.Empty:
            return


def drain_latest(
    q: queue.Queue,
    first_item: Any,
    *,
    on_drop: Callable[[], None] | None = None,
) -> Any:
    """Drop stale queued items and return the newest available frame."""
    latest = first_item
    while True:
        try:
            candidate = q.get_nowait()
            _release_item(latest)
            latest = candidate
            if on_drop is not None:
                on_drop()
        except queue.Empty:
            return latest
