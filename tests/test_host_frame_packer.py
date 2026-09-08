"""Host frame packer gate tests (ported macOS pipeline)."""

import sys
from types import SimpleNamespace

from viewer.direct_sink import DirectSinkRegistry
from viewer.host_frame_packer import maybe_install_local_viewer_packer


def test_packer_gated_off_off_darwin(monkeypatch) -> None:
    if sys.platform == "darwin":
        monkeypatch.delenv("D2S_VIEWER_HOST_FRAME", raising=False)
        monkeypatch.setenv("D2S_VIEWER_HOST_FRAME", "0")
    kwargs: dict = {"runtime_q": object()}
    installed = maybe_install_local_viewer_packer(
        pipeline_q=object(),
        local_viewer_kwargs=kwargs,
        os_name="Windows" if sys.platform != "win32" else "Linux",
    )
    assert installed is False


class _DirectSource:
    size = (4, 2)
    direct_view = bytearray(size[0] * size[1] * 4)

    def __init__(self):
        self.claimed = False
        self.completed = False

    def claim_direct(self):
        if self.claimed or self.completed:
            return False
        self.claimed = True
        return True

    def _release_direct(self):
        self.claimed = False

    def _mark_direct_submitted(self):
        self.claimed = False
        self.completed = True

    def _complete_direct(self):
        self.completed = False


def test_direct_sink_claim_is_single_use_until_fence_completion(monkeypatch) -> None:
    monkeypatch.setattr("viewer.direct_sink.sys.platform", "darwin")
    source = _DirectSource()
    registry = DirectSinkRegistry()
    registry.register(source)

    claimed, view = registry.acquire(4, 2)
    assert claimed is source
    assert view is source.direct_view
    assert registry.acquire(4, 2) == (None, None)

    source._mark_direct_submitted()
    assert registry.acquire(4, 2) == (None, None)
    source._complete_direct()
    claimed, view = registry.acquire(4, 2)
    assert claimed is source
    assert view is source.direct_view


def test_direct_sink_release_makes_claimed_slot_available(monkeypatch) -> None:
    monkeypatch.setattr("viewer.direct_sink.sys.platform", "darwin")
    source = _DirectSource()
    registry = DirectSinkRegistry()
    registry.register(source)

    claimed, _ = registry.acquire(4, 2)
    assert claimed is source
    registry.release(source)
    claimed, _ = registry.acquire(4, 2)
    assert claimed is source


def test_dropping_a_queue_item_releases_its_direct_source() -> None:
    from utils.queue_utils import _release_item

    source = _DirectSource()
    assert source.claim_direct()

    _release_item(SimpleNamespace(viewer_frame_direct=source))

    assert source.claimed is False
