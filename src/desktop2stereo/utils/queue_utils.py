from __future__ import annotations

import queue
from typing import Any, Callable


def _release_item(item: Any) -> None:
    """Release an optional borrowed capture resource before dropping an item."""
    if isinstance(item, tuple) and len(item) == 2:
        item = item[0]
    released_ids = set()
    for resource in (
        getattr(item, "native_resource", None),
        getattr(item, "sck_zero_copy", None),
    ):
        resource_id = id(resource)
        if resource is None or resource_id in released_ids:
            continue
        released_ids.add(resource_id)
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
    native = getattr(item, "viewer_native", None)
    release_native = getattr(native, "release", None)
    if callable(release_native):
        try:
            release_native()
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
