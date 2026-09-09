from __future__ import annotations

import ctypes
import queue
from types import SimpleNamespace
from pathlib import Path

import pytest

from capture.types import CapturedFrame
from stereo_runtime.providers.apple.native.coreml_io import (
    NativeCoreMLBusy,
    NativeCoreMLFrame,
    NativeCoreMLIOBridge,
    _NativeResult,
    _NativeWarpConfig,
)
from utils.queue_utils import put_latest


_NATIVE_SOURCE = Path(__file__).resolve().parents[1] / (
    "src/desktop2stereo/stereo_runtime/providers/apple/native/macos_coreml_io.mm"
)


def _native_frame(bridge) -> NativeCoreMLFrame:
    return NativeCoreMLFrame(
        bridge=bridge,
        slot=1,
        frame_id=42,
        source_width=1920,
        source_height=1080,
        depth_width=336,
        depth_height=196,
        finite_depth=True,
        nonfinite_count=0,
        preprocess_ms=0.4,
        model_ms=8.0,
        postprocess_ms=1.0,
        input_shared=True,
        output_backing_used=True,
        output_zero_copy=False,
    )


def test_native_result_abi_has_stable_layout() -> None:
    assert ctypes.sizeof(_NativeResult) == 80
    assert _NativeResult.source_width.offset > _NativeResult.slot.offset
    assert ctypes.sizeof(_NativeWarpConfig) == 76


def test_native_scalar_kernels_use_one_dimensional_threadgroups() -> None:
    source = _NATIVE_SOURCE.read_text()
    normalize_start = source.index("static void d2s_encode_normalize")
    normalize_end = source.index("static float d2s_half_to_float", normalize_start)
    normalize = source[normalize_start:normalize_end]
    pack_start = source.index("int32_t d2s_coreml_io_pack")
    pack = source[pack_start:]

    assert "dispatchThreads:grid" in normalize
    assert "dispatchThreads:grid" in pack
    assert "threadsPerThreadgroup:MTLSizeMake(w, 1, 1)" in normalize
    assert "threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)" in pack
    assert "dispatchThreadgroups:groups" not in normalize
    assert "dispatchThreadgroups:groups" not in pack
    assert "threadsPerThreadgroup:MTLSizeMake(w, h, 1)" not in normalize
    assert "threadsPerThreadgroup:MTLSizeMake(w, h, 1)" not in pack


def test_native_metal_bridge_preserves_metal_logical_rgb_channel_order() -> None:
    source = _NATIVE_SOURCE.read_text()
    preprocess_start = source.index("kernel void d2s_preprocess")
    preprocess_end = source.index("kernel void d2s_normalize_half", preprocess_start)
    preprocess = source[preprocess_start:preprocess_end]
    pack_start = source.index("kernel void d2s_warp_pack")
    pack_end = source.index(")D2S\";", pack_start)
    pack = source[pack_start:pack_end]

    assert "output[offset] = (pixel.r - p.mean0) / p.std0;" in preprocess
    assert "output[plane + offset] = (pixel.g - p.mean1) / p.std1;" in preprocess
    assert "output[2u * plane + offset] = (pixel.b - p.mean2) / p.std2;" in preprocess
    assert "output[offset + 0u] = uchar(clamp(pixel.r * 255.0f" in pack
    assert "pixel.g * 255.0f" in pack
    assert "output[offset + 2u] = uchar(clamp(pixel.b * 255.0f" in pack


def test_native_warp_uses_depth_dimensions_for_depth_sampling() -> None:
    source = _NATIVE_SOURCE.read_text()
    warp_start = source.index("struct WarpParams")
    warp_end = source.index("kernel void d2s_warp_pack", warp_start)
    warp = source[warp_start:warp_end]
    pack_start = source.index("int32_t d2s_coreml_io_pack")
    pack = source[pack_start:]

    assert "uint depth_width;" in warp
    assert "uint depth_height;" in warp
    assert "uint32_t depth_width;" in source
    assert "uint32_t depth_height;" in source
    assert "(uint32_t)ctx->depth_width" in pack
    assert "(uint32_t)ctx->depth_height" in pack
    assert "depth_sample(depth, p.depth_width, p.depth_height" in source
    assert "depth, p.source_width, p.source_height, p" not in source
    assert "layered_warp_at" in warp
    assert "occlusion_at" in warp


def test_native_frame_pack_uses_destination_without_host_conversion() -> None:
    class RecordingBridge:
        def __init__(self):
            self.pack_calls = []
            self.release_calls = []

        def pack(self, frame, destination, destination_size, width, height, output_format):
            self.pack_calls.append(
                (frame.slot, destination, destination_size, width, height, output_format)
            )

        def release(self, slot):
            self.release_calls.append(slot)

    bridge = RecordingBridge()
    frame = _native_frame(bridge)
    destination = bytearray(2 * 1 * 4)

    frame.pack(destination, (2, 1), "half_sbs")
    frame.release()
    frame.release()

    assert len(bridge.pack_calls) == 1
    assert bridge.pack_calls[0][2] == len(destination)
    assert bridge.release_calls == [1]


def test_native_frame_accepts_shared_layered_warp_configuration() -> None:
    bridge = object()
    frame = _native_frame(bridge)

    frame.configure_warp(
        depth_strength=0.25,
        max_disparity_px=48.0,
        layers=2,
        edge_dilation=1,
        hole_fill_mode=2,
        occlusion_enabled=1,
    )

    assert frame.warp_config == {
        "depth_strength": 0.25,
        "max_disparity_px": 48.0,
        "layers": 2,
        "edge_dilation": 1,
        "hole_fill_mode": 2,
        "occlusion_enabled": 1,
    }


def test_runtime_native_warp_configuration_uses_vulkan_controls() -> None:
    from stereo_runtime.runtime import _configure_native_coreml_warp

    calls = []

    class Native:
        def configure_warp(self, **values):
            calls.append(values)

    config = SimpleNamespace(
        depth_strength=0.25,
        max_disparity_px=None,
        output_format="half_sbs",
        convergence=0.0,
        parallax_preset="standard",
        edge_threshold=0.04,
        hole_fill="none",
        hole_fill_mode="none",
        hole_fill_radius=3,
        hole_fill_strength=1.0,
        mask_feather_radius=3,
        symmetric=True,
        layers=2,
        foreground_shift_scale=1.0,
        midground_shift_scale=1.0,
        background_shift_scale=1.0,
        edge_dilation=1,
        screen_edge_mask_suppression=0,
        occlusion=True,
        depth_pop=0.0,
        depth_antialias_strength=0.0,
    )
    debug = _configure_native_coreml_warp(Native(), config, width=1920, height=1080)

    assert debug["native_coreml_stereo_postprocess"] == "vulkan_layered_equivalent"
    assert calls[0]["layers"] == 2
    assert calls[0]["hole_fill_mode"] == 2
    assert calls[0]["occlusion_enabled"] == 1
    assert calls[0]["max_disparity_px"] == 96.0
    assert debug["native_coreml_parallax_gain"] == 2.0


def test_native_full_sbs_keeps_public_parallax_budget() -> None:
    from stereo_runtime.runtime import _configure_native_coreml_warp

    calls = []

    class Native:
        def configure_warp(self, **values):
            calls.append(values)

    config = SimpleNamespace(
        output_format="full_sbs",
        depth_strength=0.25,
        max_disparity_px=None,
        convergence=0.0,
        parallax_preset="standard",
        hole_fill="none",
        hole_fill_mode="none",
        hole_fill_radius=0,
        hole_fill_strength=0.0,
        mask_feather_radius=1,
        symmetric=True,
        layers=2,
        foreground_shift_scale=1.0,
        midground_shift_scale=1.0,
        background_shift_scale=1.0,
        edge_dilation=1,
        screen_edge_mask_suppression=0,
        occlusion=True,
        depth_pop=0.0,
        depth_antialias_strength=0.0,
    )
    debug = _configure_native_coreml_warp(Native(), config, width=1920, height=1080)

    assert calls[0]["max_disparity_px"] == 48.0
    assert debug["native_coreml_parallax_gain"] == 1.0


def test_layered_native_backend_is_not_reported_as_coreml_fallback() -> None:
    from pathlib import Path

    pipeline = Path(__file__).parents[1] / "src/desktop2stereo/stereo_runtime/pipeline.py"
    source = pipeline.read_text()

    assert 'startswith(\n                "native_coreml_metal"' in source


def test_native_busy_code_is_explicit_not_a_cpu_fallback(monkeypatch) -> None:
    class Library:
        def d2s_coreml_io_predict(self, *_args):
            return -2

        def d2s_coreml_io_last_error(self, _handle):
            return b"native CoreML resource ring is full"

    bridge = object.__new__(NativeCoreMLIOBridge)
    bridge._library = Library()
    bridge._handle = ctypes.c_void_p(1)
    bridge.last_error = ""
    monkeypatch.setattr("objc.pyobjc_id", lambda _value: 1)

    with pytest.raises(NativeCoreMLBusy):
        bridge.predict(object(), 7)


def test_latest_queue_drop_releases_stale_native_depth() -> None:
    class Owned:
        def __init__(self):
            self.released = 0

        def release(self):
            self.released += 1

    first = Owned()
    second = Owned()
    items = queue.Queue(maxsize=1)

    put_latest(items, SimpleNamespace(viewer_native=first))
    put_latest(items, SimpleNamespace(viewer_native=second))

    assert first.released == 1
    assert second.released == 0


def test_latest_queue_drop_releases_stale_sck_lease() -> None:
    class Owned:
        def __init__(self):
            self.released = 0

        def release(self):
            self.released += 1

    first = Owned()
    second = Owned()
    items = queue.Queue(maxsize=1)

    put_latest(items, CapturedFrame(None, (8, 4), 1.0, sck_zero_copy=first))
    put_latest(items, CapturedFrame(None, (8, 4), 2.0, sck_zero_copy=second))

    assert first.released == 1
    assert second.released == 0
