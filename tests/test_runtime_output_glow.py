from __future__ import annotations

from types import SimpleNamespace

from app_runtime.runtime_output import CudaVulkanOutputAdapter


def _adapter_with_backend(backend):
    adapter = object.__new__(CudaVulkanOutputAdapter)
    adapter.presenter = SimpleNamespace(
        _filament_glow_mode="glow",
        _filament_glow_environment_enabled=True,
        _filament_glow_sample_hz=30.0,
        _filament_glow_smoothing_seconds=0.10,
        _controller_screen_light_sample_hz=12.0,
        filament_bridge=SimpleNamespace(),
        vulkan=object(),
    )
    adapter._glow_gpu_backend = backend
    adapter._glow_gpu_last_submit = 0.0
    adapter._glow_gpu_status = None
    adapter._glow_gpu_submission_disabled = False
    return adapter


def test_cuda_adapter_keeps_vulkan_screen_light_sampling_in_glb_environment() -> None:
    class Backend:
        def __init__(self) -> None:
            self.submits = 0

        def poll(self) -> None:
            return None

        def submit(
            self, source, *, mode, source_crop_uv, detect_crop,
            screen_light_only=False,
        ) -> bool:
            self.submits += 1
            assert source == "cuda-source"
            assert mode == "screen_light"
            assert screen_light_only is True
            assert source_crop_uv == (0.0, 0.0, 1.0, 1.0)
            assert detect_crop is False
            return True

        def acquire(self, frame_id: int):
            assert frame_id == 1
            return {
                "glow_vulkan_image": SimpleNamespace(width=320, height=180),
                "screen_light_linear_rgb": (0.1, 0.2, 0.3),
                "screen_light_sample_path": "vulkan_compute_reduction",
                "_vulkan_glow_release": lambda _frame_id: None,
            }

    backend = Backend()
    adapter = _adapter_with_backend(backend)
    adapter.presenter._filament_glow_environment_enabled = False

    metadata = adapter._update_glow_gpu_source("cuda-source", frame_id=1)

    assert backend.submits == 1
    assert "glow_vulkan_image" not in metadata
    assert metadata["screen_light_linear_rgb"] == (0.1, 0.2, 0.3)
    assert callable(metadata["_vulkan_glow_release"])


def test_cuda_adapter_uses_surround_sampling_for_room_edge_reflection() -> None:
    class Backend:
        def poll(self) -> None:
            return None

        def submit(
            self, source, *, mode, temporal_smoothing_seconds, source_crop_uv,
            detect_crop,
            screen_light_only=False,
        ) -> bool:
            assert source == "cuda-source"
            assert mode == "surround"
            assert temporal_smoothing_seconds == 0.18
            assert screen_light_only is False
            assert source_crop_uv == (0.0, 0.0, 1.0, 1.0)
            assert detect_crop is False
            return True

        def acquire(self, frame_id: int):
            return {
                "glow_vulkan_image": SimpleNamespace(width=320, height=180),
                "screen_edge_light_linear_rgb": tuple(
                    (0.1, 0.2, 0.3) for _ in range(24)
                ),
                "_vulkan_glow_release": lambda _frame_id: None,
            }

    adapter = _adapter_with_backend(Backend())
    adapter.presenter._filament_glow_environment_enabled = False
    adapter.presenter._environment_screen_light_enabled = True
    adapter.presenter._environment_screen_light_sample_hz = 12.0
    adapter.presenter._environment_screen_light_smoothing_seconds = 0.18
    adapter.presenter.config = SimpleNamespace(filament_glb_path="room.glb")

    metadata = adapter._update_glow_gpu_source("cuda-source", frame_id=2)

    assert "glow_vulkan_image" not in metadata
    assert len(metadata["screen_edge_light_linear_rgb"]) == 24


def test_cuda_adapter_reuses_completed_glow_while_new_dispatch_runs() -> None:
    resource = SimpleNamespace(width=320, height=180)

    class Backend:
        def __init__(self) -> None:
            self.submits = 0
            self.polls = 0

        def poll(self) -> None:
            self.polls += 1

        def submit(
            self, source, *, mode, temporal_smoothing_seconds, source_crop_uv,
            detect_crop,
        ) -> bool:
            self.submits += 1
            assert source == "cuda-source"
            assert mode == "glow"
            assert temporal_smoothing_seconds == 0.10
            assert source_crop_uv == (0.0, 0.0, 1.0, 1.0)
            assert detect_crop is False
            return True

        def acquire(self, frame_id: int):
            return {
                "glow_vulkan_image": resource,
                "glow_vulkan_serial": 7,
                "_vulkan_glow_release": lambda _frame_id: None,
            }

    backend = Backend()
    adapter = _adapter_with_backend(backend)

    metadata = adapter._update_glow_gpu_source("cuda-source", frame_id=12)

    assert backend.polls == 1
    assert backend.submits == 1
    assert metadata["glow_vulkan_image"] is resource
    assert metadata["glow_source_size"] == (320, 180)


def test_cuda_adapter_sends_crop_to_glow_and_applies_compact_detector_result() -> None:
    class Backend:
        def __init__(self) -> None:
            self.submitted = None

        def poll(self) -> None:
            return None

        def submit(self, source, **kwargs) -> bool:
            assert source == "cuda-source"
            self.submitted = kwargs
            return True

        def acquire(self, _frame_id: int):
            return {
                "crop_detection_uv": (0.0, 0.1, 1.0, 0.8),
                "crop_detection_serial": 9,
            }

    backend = Backend()
    adapter = _adapter_with_backend(backend)
    submitted = []
    detected = []
    adapter.presenter._screen_crop_source_request = lambda: ((0.0, 0.1, 1.0, 0.8), True)
    adapter.presenter._screen_crop_detection_submitted = lambda: submitted.append(True)
    adapter.presenter._apply_screen_crop_detection = lambda crop, serial: detected.append((crop, serial))

    adapter._update_glow_gpu_source("cuda-source", frame_id=77)

    assert backend.submitted["source_crop_uv"] == (0.0, 0.1, 1.0, 0.8)
    assert backend.submitted["detect_crop"] is True
    assert submitted == [True]
    assert detected == [((0.0, 0.1, 1.0, 0.8), 9)]


def test_cuda_adapter_keeps_last_glow_image_after_submit_failure() -> None:
    resource = SimpleNamespace(width=320, height=180)

    class Backend:
        def poll(self) -> None:
            return None

        def submit(
            self, _source, *, mode, temporal_smoothing_seconds, source_crop_uv,
            detect_crop,
        ) -> bool:
            raise RuntimeError("dispatch failed")

        def acquire(self, frame_id: int):
            return {
                "glow_vulkan_image": resource,
                "glow_vulkan_serial": 3,
                "_vulkan_glow_release": lambda _frame_id: None,
            }

    adapter = _adapter_with_backend(Backend())

    metadata = adapter._update_glow_gpu_source("cuda-source", frame_id=9)

    assert metadata["glow_vulkan_image"] is resource
    assert adapter._glow_gpu_submission_disabled is True
