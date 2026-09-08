"""Consume runtime results without introducing a CPU image round trip."""

from __future__ import annotations

import os
import queue
import threading
import time

from viewer.cuda_vulkan_interop import CudaVulkanImageImporter
from viewer.rocm_vulkan_interop import RocmVulkanImageImporter
from viewer.vulkan_context import is_vulkan_device_lost_error
from viewer.vulkan_resources import (
    VulkanBinarySemaphore,
    VulkanExportableBuffer,
    VulkanExportableImage,
    VulkanExportableSemaphore,
    VulkanHostImage,
    VulkanImageResource,
)

from .gpu_producer import GpuProducerAdapter, register_gpu_producer_adapter
from .output_contract import VulkanStereoOutputFrame


def _rocm_glow_vulkan_compute_enabled(context: Any) -> bool:
    """Whether ROCm glow should use the Vulkan compute backend."""
    value = os.environ.get("D2S_ROCM_GLOW_VULKAN_COMPUTE", "1")
    if value.strip().lower() in {"0", "false", "off", "no", "disabled"}:
        return False
    return int(getattr(context, "compute_queue_index", 0) or 0) != 0


class CudaVulkanOutputAdapter(GpuProducerAdapter):
    """Convert CUDA RGBA tensors into persistent Vulkan image slots."""

    backend_name = "cuda"

    def _create_importer(self):
        return CudaVulkanImageImporter()

    @staticmethod
    def _source_image_contract(resource: VulkanImageResource) -> dict[str, object]:
        """Compatibility alias for callers using the former CUDA adapter API."""
        return GpuProducerAdapter.source_image_contract(resource)

    @staticmethod
    def _external_semaphore_requested() -> bool:
        # NVIDIA keeps the verified 41d asynchronous GPU handoff by default.
        # The synchronized-copy path remains an explicit diagnostic override;
        # cudaStreamSynchronize stalls the presenter and can make the XR
        # screen appear frozen on slower GPUs. No image data crosses CPU.
        value = os.environ.get("D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE", "1")
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def __init__(self, presenter):
        self.presenter = presenter
        self.importer = None
        self.ring_size = max(2, int(os.environ.get("D2S_VULKAN_OUTPUT_RING_SIZE", "3")))
        self.left_slots = []
        self.right_slots = []
        self.left_ready_semaphores = []
        self.right_ready_semaphores = []
        self.left_release_semaphores = []
        self.right_release_semaphores = []
        self.left_visible_semaphores = []
        self.right_visible_semaphores = []
        self.left_ready_values = []
        self.right_ready_values = []
        self.left_release_values = []
        self.right_release_values = []
        self.external_semaphore_enabled = False
        self._external_semaphore_error: str | None = None
        self._external_semaphore_request_reason: str | None = None
        self._external_semaphore_request_enabled = False
        self._logged_external_sync_mode = False
        self.left_slot = None
        self.right_slot = None
        self._extent = None
        self._lease_condition = threading.Condition()
        self._active_leases: dict[int, int] = {}
        self._closed = False
        self._screen_light_rgb = (0.18, 0.18, 0.18)
        self._screen_light_sample_path = "vulkan_compute_reduction"
        self._screen_light_pending = None
        self._glow_gpu_backend = None
        self._glow_gpu_last_submit = 0.0
        self._glow_gpu_status: str | None = None
        self._glow_gpu_submission_disabled = False
        self._release_signaled: set[tuple[int, int]] = set()
        # Only CUDA's semaphore-disabled GPU-copy path uses this. Vulkan's
        # release barrier is asynchronous, so CUDA must wait before rewriting
        # a recycled external image slot.
        self._cuda_release_timelines: list[int] = []
        self._source_frames: dict[int, tuple[object, object, int]] = {}
        self._released_source_frames: set[int] = set()
        self._prepared_source_eyes: set[tuple[int, int]] = set()


        self._screen_light_last_submit = 0.0

    def _update_screen_light_sample(self, left, right) -> None:
        """Asynchronously reduce display sRGB eyes to one linear screen color."""
        pending = self._screen_light_pending
        if pending is not None:
            host, event = pending
            if event.query():
                values = host.tolist()
                self._screen_light_rgb = tuple(
                    max(0.0, min(8.0, float(value))) for value in values[:3]
                )
                self._screen_light_pending = None

        now = time.monotonic()
        if (
            self._screen_light_pending is not None
            or now - self._screen_light_last_submit < (
                1.0 / max(1.0, float(getattr(
                    self.presenter, "_controller_screen_light_sample_hz", 12.0
                )))
            )
        ):
            return
        try:
            import torch

            if (
                not isinstance(left, torch.Tensor)
                or not isinstance(right, torch.Tensor)
                or not left.is_cuda
                or not right.is_cuda
                or left.ndim != 3
                or right.ndim != 3
                or left.shape[-1] < 3
                or right.shape[-1] < 3
            ):
                return
            row_step = max(1, int(left.shape[0]) // 32)
            column_step = max(1, int(left.shape[1]) // 32)
            sampled = torch.cat(
                (
                    left[::row_step, ::column_step, :3].reshape(-1, 3),
                    right[::row_step, ::column_step, :3].reshape(-1, 3),
                ),
                dim=0,
            ).to(dtype=torch.float32)
            if left.dtype == torch.uint8:
                sampled = sampled / 255.0
            sampled = sampled.clamp(0.0, 1.0)
            linear = torch.where(
                sampled <= 0.04045,
                sampled / 12.92,
                ((sampled + 0.055) / 1.055).pow(2.4),
            )
            rgb = linear.mean(dim=0)
            host = torch.empty(3, dtype=torch.float32, pin_memory=True)
            host.copy_(rgb, non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(left.device))
            self._screen_light_pending = (host, event)
            self._screen_light_last_submit = now

        except Exception:
            # Screen illumination is supplemental and must never break output.
            self._screen_light_pending = None

    def _glow_environment_enabled(self) -> bool:
        return bool(
            getattr(
                self.presenter,
                "_filament_glow_environment_enabled",
                True,
            )
        )

    def _set_glow_gpu_status(self, status: str) -> None:
        normalized = str(status or "unknown")
        if normalized == self._glow_gpu_status:
            return
        self._glow_gpu_status = normalized
        print(f"[VulkanOutput] Glow/screen-light source: {normalized}", flush=True)

    def _update_glow_gpu_source(self, source, *, frame_id: int) -> dict[str, object]:
        """Publish completed Glow images and Vulkan screen-light reductions.

        This method never waits for completion. If the newest dispatch is
        still running, acquire() returns the last completed slot instead.
        """
        mode = str(
            getattr(self.presenter, "_filament_glow_mode", "off") or "off"
        ).strip().lower()
        glow_active = (
            self._glow_environment_enabled()
            and mode not in {"off", "none", "false", "0"}
        )
        edge_light_active = bool(
            getattr(self.presenter, "_environment_screen_light_enabled", False)
            and getattr(self.presenter.config, "filament_glb_path", None)
        )
        if self.backend_name != "cuda":
            # AMD ROCm must use the Vulkan compute glow path. HIP writes only
            # exported staging buffers; Vulkan compute produces the sampled
            # glow image. ROCm must not route glow through a CPU readback when
            # the dedicated Vulkan compute queue is unavailable.
            vulkan_compute_glow = _rocm_glow_vulkan_compute_enabled(
                self.presenter.vulkan
            )
            if not vulkan_compute_glow:
                unavailable_callback = getattr(
                    self.presenter, "_screen_crop_detection_unavailable", None
                )
                if callable(unavailable_callback):
                    unavailable_callback()
                self._set_glow_gpu_status(
                    f"vulkan_compute_unavailable backend={self.backend_name}"
                )
                return {}
        # The source image is produced by the Vulkan Glow worker.  Do not use
        # the removed Filament screen/Glow ABI as a capability gate: the
        # Projection Composer owns the eventual sampling pass.
        glow_image_available = self.presenter.vulkan is not None
        gpu_glow_active = glow_active and glow_image_available
        try:
            if self._glow_gpu_backend is None:
                if self.backend_name != "cuda":
                    prewarmed = getattr(
                        self.presenter, "_prewarmed_glow_backend", None
                    )
                    if prewarmed is not None:
                        self._glow_gpu_backend = prewarmed
                        self.presenter._prewarmed_glow_backend = None
                    else:
                        from stereo_runtime.vulkan_glow_source import (
                            VulkanGlowSourceComputeBackend,
                        )

                        self._glow_gpu_backend = VulkanGlowSourceComputeBackend(
                            self.presenter.vulkan
                        )
                        self._set_glow_gpu_status(
                            "vulkan_compute_external_image backend=rocm"
                        )
                else:
                    from stereo_runtime.vulkan_glow_source import (
                        VulkanGlowSourceComputeBackend,
                    )

                    self._glow_gpu_backend = VulkanGlowSourceComputeBackend(
                        self.presenter.vulkan
                    )
            if gpu_glow_active:
                if self.backend_name != "cuda":
                    self._set_glow_gpu_status(
                        "vulkan_compute_external_image async_queue=True"
                    )
                else:
                    self._set_glow_gpu_status(
                        "vulkan_compute_external_image async_queue=True"
                    )
            else:
                reason = (
                    "vulkan_projection_composer_source"
                    if glow_active and not glow_image_available
                    else "glow_inactive"
                )
                self._set_glow_gpu_status(
                    "vulkan_compute_reduction screen_light_only=True "
                    f"glow={reason}"
                )
            backend = self._glow_gpu_backend
            backend.poll()
            crop_request = getattr(self.presenter, "_screen_crop_source_request", None)
            if callable(crop_request):
                source_crop_uv, detect_crop = crop_request()
            else:
                source_crop_uv, detect_crop = ((0.0, 0.0, 1.0, 1.0), False)
            now = time.monotonic()
            sample_hz = max(1.0, float(getattr(
                self.presenter,
                "_filament_glow_sample_hz" if gpu_glow_active else (
                    "_environment_screen_light_sample_hz" if edge_light_active
                    else "_controller_screen_light_sample_hz"
                ),
                30.0 if gpu_glow_active else 12.0,
            )))
            submit_interval = 1.0 / sample_hz
            if (
                not self._glow_gpu_submission_disabled
                and now - self._glow_gpu_last_submit >= submit_interval
            ):
                submit_kwargs = {
                    "mode": mode if gpu_glow_active else (
                        "surround" if edge_light_active else "screen_light"
                    ),
                    "source_crop_uv": source_crop_uv,
                    "detect_crop": bool(detect_crop),
                }
                if gpu_glow_active or edge_light_active:
                    submit_kwargs["temporal_smoothing_seconds"] = max(
                        0.0,
                        float(getattr(
                            self.presenter,
                            "_environment_screen_light_smoothing_seconds"
                            if edge_light_active and not gpu_glow_active
                            else "_filament_glow_smoothing_seconds",
                            0.10,
                        )),
                    )
                if not gpu_glow_active and not edge_light_active:
                    submit_kwargs["screen_light_only"] = True
                submitted = backend.submit(source, **submit_kwargs)
                if submitted:
                    self._glow_gpu_last_submit = now
                    if detect_crop:
                        submitted_callback = getattr(
                            self.presenter, "_screen_crop_detection_submitted", None
                        )
                        if callable(submitted_callback):
                            submitted_callback()
            metadata = backend.acquire(frame_id)
            detection_callback = getattr(
                self.presenter, "_apply_screen_crop_detection", None
            )
            if callable(detection_callback) and metadata.get("crop_detection_uv") is not None:
                detection_callback(
                    metadata.get("crop_detection_uv"),
                    metadata.get("crop_detection_serial"),
                )
            if not gpu_glow_active:
                metadata = {
                    key: value
                    for key, value in metadata.items()
                    if key in {
                        "screen_light_linear_rgb",
                        "screen_light_sample_path",
                        "screen_edge_light_linear_rgb",
                        "crop_detection_uv",
                        "crop_detection_serial",
                        "_vulkan_glow_release",
                    }
                }
            resource = metadata.get("glow_vulkan_image")
            if resource is not None:
                metadata["glow_source_size"] = (
                    int(getattr(resource, "width", 0)),
                    int(getattr(resource, "height", 0)),
                )
                metadata["glow_composer_source_ready"] = True
                metadata["glow_composer_source_owner"] = "vulkan_projection_composer"
            if os.environ.get("D2S_GLOW_DIAGNOSTIC"):
                now_diag = time.monotonic()
                if now_diag - getattr(self, "_glow_diag_last_log", 0.0) >= 2.0:
                    self._glow_diag_last_log = now_diag
                    print(
                        "[VulkanOutput] Glow diag: "
                        f"status={self._glow_gpu_status} "
                        f"screen_light={metadata.get('screen_light_linear_rgb')} "
                        f"glow_size={metadata.get('glow_source_size')} "
                        f"submit_ms={getattr(backend, '_last_submit_ms', None)}",
                        flush=True,
                    )
            return metadata
        except Exception as exc:
            import traceback as _tb

            unavailable_callback = getattr(
                self.presenter, "_screen_crop_detection_unavailable", None
            )
            if callable(unavailable_callback):
                unavailable_callback()

            self._set_glow_gpu_status(
                f"reuse_last_completed reason={type(exc).__name__}: {exc}"
            )
            if type(exc).__name__ == "VkErrorUnknown" and not getattr(
                self, "_glow_vk_error_traceback_logged", False
            ):
                self._glow_vk_error_traceback_logged = True
                print(
                    "[VulkanOutput] Glow VkErrorUnknown traceback:\n"
                    + _tb.format_exc().rstrip(),
                    flush=True,
                )
            backend = self._glow_gpu_backend
            self._glow_gpu_submission_disabled = True
            if backend is not None:
                try:
                    metadata = backend.acquire(frame_id)
                    resource = metadata.get("glow_vulkan_image")
                    if resource is not None:
                        metadata["glow_source_size"] = (
                            int(getattr(resource, "width", 0)),
                            int(getattr(resource, "height", 0)),
                        )
                        metadata["glow_composer_source_ready"] = True
                        metadata["glow_composer_source_owner"] = "vulkan_projection_composer"
                        return metadata
                except Exception:
                    pass
            return {}

    @staticmethod
    def _tensor_extent(tensor):
        shape = tuple(int(value) for value in getattr(tensor, "shape", ()))
        if len(shape) != 3 or shape[-1] != 4:
            raise ValueError("runtime eye must be an HxWx4 tensor")
        return shape[1], shape[0]

    def _wait_for_runtime_cuda_event(self, runtime_result, tensor) -> None:
        """Enqueue producer completion on CUDA's consumer stream.

        Runtime inference and output conversion use different Python threads.
        Host polling a torch event can keep the latest-frame queue blocked even
        though the GPU can make progress.  Wait on the CUDA stream instead;
        this remains entirely GPU-side and is intentionally CUDA-only so the
        validated ROCm staging path is unchanged.
        """
        if self.backend_name != "cuda":
            return
        event = getattr(runtime_result, "cuda_ready_event", None)
        if event is None:
            return
        wait_event = getattr(event, "wait", None)
        if not callable(wait_event):
            raise RuntimeError("CUDA runtime result has no GPU-waitable ready event")
        device = getattr(tensor, "device", None)
        try:
            import torch

            if getattr(device, "type", None) == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.current_stream(device=device).wait_event(event)
                return
        except Exception:
            # ``Event.wait()`` is still a CUDA stream enqueue.  Keep this
            # narrow fallback for torch/runtime versions without the explicit
            # Stream.wait_event route; never synchronize the host or copy pixels
            # through CPU memory.
            pass
        wait_event()

    def _claim_slot(self, slot_index: int, frame_id: int) -> None:
        while True:
            with self._lease_condition:
                if self._closed:
                    raise RuntimeError("Vulkan output adapter is closed")
                if slot_index not in self._active_leases:
                    self._active_leases[slot_index] = int(frame_id)
                    return

            # Do not call back into the presenter while holding the adapter
            # condition: releasing a displayed frame re-enters this adapter.
            presenter = getattr(self, "presenter", None)
            release_displayed = getattr(
                presenter, "release_displayed_output_for_reuse", None
            )
            if callable(release_displayed) and release_displayed(slot_index):
                continue
            with self._lease_condition:
                if slot_index not in self._active_leases:
                    continue
                self._lease_condition.wait()

    def release_frame(self, frame_id: int) -> None:
        """Release a producer slot after the XR consumer no longer samples it."""
        with self._lease_condition:
            for slot_index, lease_frame_id in tuple(self._active_leases.items()):
                if lease_frame_id == int(frame_id):
                    del self._active_leases[slot_index]
                    self._lease_condition.notify_all()
                    if not any(
                        prepared_frame == int(frame_id)
                        for prepared_frame, _eye in getattr(
                            self, "_prepared_source_eyes", ()
                        )
                    ):
                        getattr(self, "_source_frames", {}).pop(int(frame_id), None)
                    return

    def _discard_source_frame_after_device_loss(self, frame_id: int) -> None:
        """Release only Python-side leases after the Vulkan device is dead."""
        frame_key = int(frame_id)
        self._prepared_source_eyes = {
            item for item in self._prepared_source_eyes if item[0] != frame_key
        }
        self.release_frame(frame_key)
        self._released_source_frames.add(frame_key)
        self._source_frames.pop(frame_key, None)

    def prepare_source_for_sampling(self, frame_id: int, eye_index: int):
        """Wait for the producer and publish a post-barrier semaphore to Filament."""
        frame_key = int(frame_id)
        eye = int(eye_index)
        entry = self._source_frames.get(frame_key)
        if entry is None:
            raise RuntimeError(f"unknown Vulkan source frame {frame_id}")
        left, right, slot_index = entry
        if len(self.left_visible_semaphores) <= slot_index:
            # External-semaphore path was never initialized (e.g. on ROCm/HIP
            # where timeline external semaphores are unsupported). convert()
            # already GPU-copied and synchronized (hipStreamSynchronize) the
            # eyes; just transition the imported image to shader-readable so
            # Filament can sample it (no producer-ready semaphore to wait on).
            resource = left if eye == 0 else right
            if (frame_key, eye) not in self._prepared_source_eyes:
                self.presenter.vulkan.prepare_external_image_for_sampling(
                    resource.resource
                )
                self._prepared_source_eyes.add((frame_key, eye))
            return None
        visible = (
            self.left_visible_semaphores[slot_index]
            if eye == 0
            else self.right_visible_semaphores[slot_index]
        )
        if (frame_key, eye) in self._prepared_source_eyes:
            # A reused output frame is already producer-ready and in sampling
            # layout. Filament must not wait on another binary semaphore; the
            # previous acquire consumed it and the source content is unchanged.
            return None
        resource = left if eye == 0 else right
        ready = (
            self.left_ready_semaphores[slot_index]
            if eye == 0
            else self.right_ready_semaphores[slot_index]
        )
        self.presenter.vulkan.prepare_external_image_for_sampling(
            resource.resource,
            wait_semaphore=ready.semaphore,
            wait_semaphore_value=(
                self.left_ready_values[slot_index]
                if eye == 0
                else self.right_ready_values[slot_index]
            ),
            signal_semaphore=visible.semaphore,
        )
        self._prepared_source_eyes.add((frame_key, eye))
        return visible.semaphore

    def release_consumer_frame(
        self,
        frame_id: int,
        consumer_semaphores=None,
        *,
        wait_for_timeline: int | None = None,
    ) -> None:
        """Signal producer-release semaphores after Filament has finished sampling."""
        frame_key = int(frame_id)
        if frame_key in self._released_source_frames:
            return
        entry = self._source_frames.get(frame_key)
        if entry is None:
            self.release_frame(frame_key)
            self._released_source_frames.add(frame_key)
            return
        left, right, slot_index = entry
        waits = tuple(consumer_semaphores or ())
        context = self.presenter.vulkan
        if bool(getattr(context, "device_lost", False)):
            self._discard_source_frame_after_device_loss(frame_key)
            return
        try:
            for eye_index, resource in (
                (0, left.resource),
                (1, right.resource),
            ):
                if (frame_key, eye_index) not in self._prepared_source_eyes:
                    continue
                release_semaphores = (
                    self.left_release_semaphores
                    if eye_index == 0
                    else self.right_release_semaphores
                )
                if len(release_semaphores) <= slot_index:
                    # No external release semaphores (e.g. ROCm/HIP timeline
                    # unsupported). The import was GPU-copied + synchronized by
                    # convert(); return the image to producer-writable GENERAL
                    # so the next frame's sampling transition sees GENERAL,
                    # otherwise the composer raises "must be in GENERAL".
                    release_timeline = context.release_external_image_from_sampling(
                        resource,
                        wait_for_timeline=wait_for_timeline,
                        wait_semaphore=(
                            waits[eye_index] if eye_index < len(waits) else None
                        ),
                    )
                    if self.backend_name == "cuda":
                        if (
                            release_timeline is not None
                            and slot_index < len(self._cuda_release_timelines)
                        ):
                            self._cuda_release_timelines[slot_index] = max(
                                int(self._cuda_release_timelines[slot_index]),
                                int(release_timeline),
                            )
                    elif self.backend_name == "rocm":
                        release_timelines = getattr(self, "_rocm_release_timelines", ())
                        if (
                            release_timeline is not None
                            and slot_index < len(release_timelines)
                        ):
                            release_timelines[slot_index] = max(
                                int(release_timelines[slot_index]),
                                int(release_timeline),
                            )
                    self._prepared_source_eyes.discard((frame_key, eye_index))
                    continue
                release_values = (
                    self.left_release_values
                    if eye_index == 0
                    else self.right_release_values
                )
                release_semaphore = release_semaphores[slot_index]
                release_value = release_values[slot_index] + 1
                context.release_external_image_from_sampling(
                    resource,
                    wait_for_timeline=wait_for_timeline,
                    wait_semaphore=(
                        waits[eye_index] if eye_index < len(waits) else None
                    ),
                    signal_semaphore=release_semaphore.semaphore,
                    signal_semaphore_value=release_value,
                )
                release_values[slot_index] = release_value
                self._release_signaled.add((eye_index, slot_index))
                self._prepared_source_eyes.discard((frame_key, eye_index))
        except Exception:
            if bool(getattr(context, "device_lost", False)):
                self._discard_source_frame_after_device_loss(frame_key)
            raise
        self.release_frame(frame_key)
        self._released_source_frames.add(frame_key)
        self._source_frames.pop(frame_key, None)

    def _ensure_slots(self, width: int, height: int) -> None:
        if not bool(getattr(self.presenter, "initialized", False)):
            raise RuntimeError("OpenXR Vulkan presenter is not initialized")
        context = self.presenter.vulkan
        if context is None:
            raise RuntimeError("OpenXR Vulkan context is unavailable")
        if self._extent == (width, height):
            return
        self.close()
        with self._lease_condition:
            self._closed = False
        self.importer = self._create_importer()
        self._external_semaphore_error = None
        self._external_semaphore_request_reason = None
        # Runtime eye tensors contain display-referred sRGB bytes. Keep the
        # Vulkan image format sRGB so Quad Layer copies remain byte-preserving.
        output_format = context.vk.VK_FORMAT_R8G8B8A8_SRGB
        self.left_slots = [
            VulkanExportableImage(
                context, width, height, label=f"runtime-left-eye-{index}", format=output_format
            )
            for index in range(self.ring_size)
        ]
        self.right_slots = [
            VulkanExportableImage(
                context, width, height, label=f"runtime-right-eye-{index}", format=output_format
            )
            for index in range(self.ring_size)
        ]
        env_external_semaphore_requested = self._external_semaphore_requested()
        presenter_ready_semaphore_available = bool(
            getattr(self.presenter, "source_ready_semaphore_available", False)
        )
        external_semaphore_requested = bool(
            env_external_semaphore_requested and presenter_ready_semaphore_available
        )
        self._external_semaphore_request_enabled = external_semaphore_requested
        if not external_semaphore_requested:
            if not env_external_semaphore_requested:
                self._external_semaphore_request_reason = (
                    "disabled_by_D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE"
                )
            elif not presenter_ready_semaphore_available:
                self._external_semaphore_request_reason = (
                    "projection_composer_source_semaphore_unavailable"
                )
        try:
            if not external_semaphore_requested:
                self._extent = (width, height)
                if self.backend_name == "cuda":
                    self._cuda_release_timelines = [0] * self.ring_size
                return
            self.left_ready_semaphores = [
                VulkanExportableSemaphore(
                    context,
                    label=f"runtime-left-ready-{index}",
                    timeline=True,
                )
                for index in range(self.ring_size)
            ]
            self.right_ready_semaphores = [
                VulkanExportableSemaphore(
                    context,
                    label=f"runtime-right-ready-{index}",
                    timeline=True,
                )
                for index in range(self.ring_size)
            ]
            self.left_release_semaphores = [
                VulkanExportableSemaphore(
                    context,
                    label=f"runtime-left-release-{index}",
                    timeline=True,
                )
                for index in range(self.ring_size)
            ]
            self.right_release_semaphores = [
                VulkanExportableSemaphore(
                    context,
                    label=f"runtime-right-release-{index}",
                    timeline=True,
                )
                for index in range(self.ring_size)
            ]
            self.left_visible_semaphores = [
                VulkanBinarySemaphore(context, label=f"runtime-left-visible-{index}")
                for index in range(self.ring_size)
            ]
            self.right_visible_semaphores = [
                VulkanBinarySemaphore(context, label=f"runtime-right-visible-{index}")
                for index in range(self.ring_size)
            ]
            for semaphore in (
                *self.left_ready_semaphores,
                *self.right_ready_semaphores,
                *self.left_release_semaphores,
                *self.right_release_semaphores,
            ):
                self.importer.register_semaphore(semaphore)
            self.external_semaphore_enabled = bool(
                self.importer.capabilities.external_semaphore
            )
            if not self.external_semaphore_enabled:
                self._external_semaphore_error = "producer_external_semaphore_api_unavailable"
            self.left_ready_values = [0] * self.ring_size
            self.right_ready_values = [0] * self.ring_size
            self.left_release_values = [0] * self.ring_size
            self.right_release_values = [0] * self.ring_size
        except Exception as exc:
            self._external_semaphore_error = (
                f"{type(exc).__name__}: {exc}"
            )
            print(
                "[VulkanOutput] External semaphore setup failed; "
                f"using synchronized GPU-copy source: {self._external_semaphore_error}",
                flush=True,
            )
            for semaphore in (
                *self.left_ready_semaphores,
                *self.right_ready_semaphores,
                *self.left_release_semaphores,
                *self.right_release_semaphores,
                *self.left_visible_semaphores,
                *self.right_visible_semaphores,
            ):
                semaphore.close()
            self.left_ready_semaphores = []
            self.right_ready_semaphores = []
            self.left_release_semaphores = []
            self.right_release_semaphores = []
            self.left_visible_semaphores = []
            self.right_visible_semaphores = []
            self.left_ready_values = []
            self.right_ready_values = []
            self.left_release_values = []
            self.right_release_values = []
            self.external_semaphore_enabled = False
        self._extent = (width, height)
        if self.backend_name == "cuda":
            self._cuda_release_timelines = [0] * self.ring_size

    def convert(self, runtime_result, *, frame_id: int, timestamp: float):
        left = getattr(runtime_result, "left_eye", None)
        right = getattr(runtime_result, "right_eye", None)
        width, height = self._tensor_extent(left)
        if self._tensor_extent(right) != (width, height):
            raise ValueError("left/right runtime eye dimensions differ")
        self._wait_for_runtime_cuda_event(runtime_result, left)
        self._ensure_slots(width, height)
        slot_index = int(frame_id) % self.ring_size
        self._claim_slot(slot_index, frame_id)
        self.left_slot = self.left_slots[slot_index]
        self.right_slot = self.right_slots[slot_index]
        glow_metadata: dict[str, object] = {}
        try:
            if self._screen_light_sample_path != "vulkan_compute_reduction":
                self._update_screen_light_sample(left, right)
            glow_source = getattr(runtime_result, "source_rgb", None)
            glow_source = glow_source if glow_source is not None else left
            glow_metadata = self._update_glow_gpu_source(
                glow_source, frame_id=frame_id
            )
            sampled_light = glow_metadata.get("screen_light_linear_rgb")
            if isinstance(sampled_light, (list, tuple)) and len(sampled_light) >= 3:
                self._screen_light_rgb = tuple(float(value) for value in sampled_light[:3])
                self._screen_light_sample_path = str(
                    glow_metadata.get(
                        "screen_light_sample_path", "vulkan_compute_reduction"
                    )
                )
            left_ready = None
            right_ready = None
            use_external_semaphore = bool(
                self.external_semaphore_enabled
                and getattr(self.presenter, "source_ready_semaphore_available", False)
            )
            external_semaphore_requested = bool(
                self._external_semaphore_request_enabled
            )
            if not self._logged_external_sync_mode:
                external_sync_message = (
                    "[VulkanOutput] CUDA external timeline semaphore sync: "
                    f"requested={external_semaphore_requested} "
                    f"available={self.external_semaphore_enabled} "
                    f"active={use_external_semaphore} "
                    f"blocked_reason={self._external_semaphore_request_reason or 'none'}"
                )
                if self._external_semaphore_error:
                    external_sync_message += f" error={self._external_semaphore_error}"
                print(external_sync_message, flush=True)
                self._logged_external_sync_mode = True
            if use_external_semaphore:
                for eye_index, release_semaphore, release_values in (
                    (
                        0,
                        self.left_release_semaphores[slot_index],
                        self.left_release_values,
                    ),
                    (
                        1,
                        self.right_release_semaphores[slot_index],
                        self.right_release_values,
                    ),
                ):
                    if (eye_index, slot_index) not in self._release_signaled:
                        continue
                    self.importer.wait_semaphore(
                        release_semaphore,
                        value=release_values[slot_index],
                    )
                    self._release_signaled.discard((eye_index, slot_index))
                self.importer.copy_tensor(left, self.left_slot)
                self.importer.copy_tensor(right, self.right_slot)
                left_ready = self.left_ready_semaphores[slot_index]
                right_ready = self.right_ready_semaphores[slot_index]
                self.left_ready_values[slot_index] += 1
                self.right_ready_values[slot_index] += 1
                stream = None
                self.importer.signal_semaphore(
                    left_ready,
                    value=self.left_ready_values[slot_index],
                    stream=stream,
                )
                self.importer.signal_semaphore(
                    right_ready,
                    value=self.right_ready_values[slot_index],
                    stream=stream,
                )
            if not use_external_semaphore:
                if self.backend_name == "cuda" and slot_index < len(
                    self._cuda_release_timelines
                ):
                    release_timeline = int(self._cuda_release_timelines[slot_index])
                    if release_timeline > 0:
                        self.presenter.vulkan.wait_for_timeline(release_timeline)
                        self._cuda_release_timelines[slot_index] = 0
                self.importer.copy_tensor(left, self.left_slot)
                self.importer.copy_tensor(right, self.right_slot)
                self.importer.synchronize()
        except Exception:
            glow_release = glow_metadata.get("_vulkan_glow_release")
            if callable(glow_release):
                glow_release(frame_id)
            self.release_frame(frame_id)
            raise
        left_contract = self.source_image_contract(self.left_slot.resource)
        right_contract = self.source_image_contract(self.right_slot.resource)
        self._source_frames[int(frame_id)] = (
            self.left_slot,
            self.right_slot,
            slot_index,
        )
        self._released_source_frames.discard(int(frame_id))
        return VulkanStereoOutputFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            left_eye=self.left_slot.resource,
            right_eye=self.right_slot.resource,
            ready_timeline=None,
            metadata={
                **dict(getattr(runtime_result, "debug_info", None) or {}),
                "vulkan_output_ring_slot": slot_index,
                "vulkan_output_ring_size": self.ring_size,
                "vulkan_output_sync": (
                    self.external_semaphore_sync_mode
                    if use_external_semaphore
                    else self.output_sync_mode
                ),
                "vulkan_ready_semaphore_left": (
                    left_ready.semaphore if left_ready is not None else None
                ),
                "vulkan_ready_semaphore_right": (
                    right_ready.semaphore if right_ready is not None else None
                ),
                "vulkan_external_semaphore_available": bool(
                    use_external_semaphore
                ),
                "vulkan_external_semaphore_type": (
                    "timeline" if use_external_semaphore else None
                ),
                "vulkan_external_semaphore_requested": bool(
                    external_semaphore_requested
                ),
                "vulkan_external_semaphore_request_reason": (
                    self._external_semaphore_request_reason
                ),
                "vulkan_external_semaphore_error": self._external_semaphore_error,
                # CUDA writes directly into presenter-owned Vulkan images. The
                # external semaphore only changes visibility synchronization;
                # it does not introduce a CPU readback or a different image
                # ownership path.
                "vulkan_readback": "none",
                "vulkan_output_path": "presenter_owned_storage_image",
                "vulkan_output_image_direct": True,
                "vulkan_gpu_to_cpu": False,
                "vulkan_zero_cpu_readback": True,
                # The runtime CUDA tensor is imported into Vulkan input
                # buffers by a device-side copy.  Do not overclaim strict
                # zero-copy until a native CUDA image producer is available.
                "vulkan_zero_copy": False,
                "vulkan_source_layout_left": left_contract["layout"],
                "vulkan_source_layout_right": right_contract["layout"],
                "vulkan_source_queue_family_left": left_contract["queue_family"],
                "vulkan_source_queue_family_right": right_contract["queue_family"],
                "_vulkan_source_prepare_for_sampling": self.prepare_source_for_sampling,
                "_vulkan_source_consumer_release": self.release_consumer_frame,
                "_vulkan_output_release": self.release_frame,
                "screen_light_linear_rgb": self._screen_light_rgb,
                "screen_light_sample_path": self._screen_light_sample_path,
                **glow_metadata,
            },
            color_space="srgb",
            image_origin="top_left",
        )

    def close(self) -> None:
        with self._lease_condition:
            self._closed = True
            self._active_leases.clear()
            self._lease_condition.notify_all()
        glow_backend = self._glow_gpu_backend
        self._glow_gpu_backend = None
        if glow_backend is not None:
            glow_backend.close()
        if self.importer is not None:
            self.importer.close()
        self.importer = None
        for slot in (*self.left_slots, *self.right_slots):
            if slot is not None:
                slot.close()
        self.left_slots = []
        self.right_slots = []
        for semaphore in (
            *self.left_ready_semaphores,
            *self.right_ready_semaphores,
            *self.left_release_semaphores,
            *self.right_release_semaphores,
            *self.left_visible_semaphores,
            *self.right_visible_semaphores,
        ):
            if semaphore is not None:
                semaphore.close()
        self.left_ready_semaphores = []
        self.right_ready_semaphores = []
        self.left_release_semaphores = []
        self.right_release_semaphores = []
        self.left_visible_semaphores = []
        self.right_visible_semaphores = []
        self.left_ready_values = []
        self.right_ready_values = []
        self.left_release_values = []
        self.right_release_values = []
        self.external_semaphore_enabled = False
        self._external_semaphore_request_enabled = False
        self.left_slot = None
        self.right_slot = None
        self._extent = None
        self._screen_light_pending = None
        self._release_signaled.clear()
        self._cuda_release_timelines = []
        self._source_frames.clear()
        self._released_source_frames.clear()
        self._prepared_source_eyes.clear()


class RocmVulkanOutputAdapter(CudaVulkanOutputAdapter):
    """Convert HIP tensors using the AMD ROCm Vulkan interop importer."""

    backend_name = "rocm"

    def __init__(self, presenter):
        super().__init__(presenter)
        self._rocm_ready_pending: set[tuple[int, int]] = set()
        self._rocm_release_timelines: list[int] = []
        self._rocm_ready_timelines: dict[int, int] = {}
        self._eye_buffers: list[tuple[VulkanExportableBuffer, VulkanExportableBuffer]] = []
        self._buffer_frames: dict[int, tuple[VulkanExportableBuffer, VulkanExportableBuffer]] = {}
        self._host_eye_slots: list[tuple[VulkanHostImage, VulkanHostImage]] = []
        self._host_frame_slots: dict[int, int] = {}
        self._host_staging_enabled = self._rocm_eye_host_staging_enabled()
        if not self._host_staging_enabled:
            self._host_eye_slots = []
        # Old diagnostic shells can retain the removed Windows user override.
        # Keep AMD's intentional draw-disable switch explicit and vendor-local.
        disable_glow = os.environ.get("D2S_ROCM_DISABLE_GLOW_DRAW", "0").strip().lower()
        if disable_glow in {"1", "true", "yes", "on"}:
            os.environ["D2S_OPENXR_DISABLE_GLOW_DRAW"] = "1"
            print("[VulkanOutput] ROCm glow draw disabled by D2S_ROCM_DISABLE_GLOW_DRAW", flush=True)
        elif os.environ.pop("D2S_OPENXR_DISABLE_GLOW_DRAW", None) is not None:
            print("[VulkanOutput] Cleared inherited legacy glow-disable diagnostic for ROCm", flush=True)

    @staticmethod
    def _external_semaphore_requested() -> bool:
        # ROCm external binary-semaphore cleanup can leave a dropped frame
        # waiting forever on some Windows/AMD driver combinations. The
        # buffer-staging + Vulkan copy path is the stable default; retain an
        # explicit opt-in for driver/interoperability experiments.
        value = os.environ.get("D2S_ENABLE_ROCM_EXTERNAL_SEMAPHORE", "0")
        normalized = value.strip().lower()
        if normalized in {"", "auto", "default"}:
            return True
        return normalized in {"1", "true", "yes", "on"}

    @staticmethod
    def _rocm_kmt_memory_enabled() -> bool:
        value = os.environ.get("D2S_ROCM_KMT_EXTERNAL_MEMORY", "auto")
        return value.strip().lower() not in {"", "0", "false", "off", "no", "disabled"}

    @staticmethod
    def _rocm_eye_host_staging_enabled() -> bool:
        value = os.environ.get("D2S_ROCM_EYE_HOST_STAGING", "0")
        return value.strip().lower() not in {"0", "false", "off", "no", "disabled"}

    @staticmethod
    def _rocm_glow_vulkan_compute_enabled(context: Any) -> bool:
        value = os.environ.get("D2S_ROCM_GLOW_VULKAN_COMPUTE", "1")
        if value.strip().lower() in {"0", "false", "off", "no", "disabled"}:
            return False
        return int(getattr(context, "compute_queue_index", 0) or 0) != 0

    @staticmethod
    def _tensor_host_rgba(tensor, *, width: int, height: int):
        try:
            import torch

            value = tensor.detach().to(device="cpu")
            if value.ndim == 4:
                value = value[0]
            if (
                value.ndim == 3
                and int(value.shape[-1]) == 4
                and str(value.dtype) == "torch.uint8"
                and tuple(value.shape[:2]) == (height, width)
            ):
                return value.numpy()
        except Exception:
            pass
        # Fall back to the correctness path proven by VulkanHostOutputAdapter.
        return VulkanHostOutputAdapter._tensor_to_rgba(
            tensor, width=width, height=height
        )

    def _create_importer(self):
        return RocmVulkanImageImporter()

    def _ensure_slots(self, width: int, height: int) -> None:
        """Create AMD binary handoffs without invoking CUDA timeline setup."""
        if not bool(getattr(self.presenter, "initialized", False)):
            raise RuntimeError("OpenXR Vulkan presenter is not initialized")
        context = self.presenter.vulkan
        if context is None:
            raise RuntimeError("OpenXR Vulkan context is unavailable")
        if self._extent == (width, height):
            return
        self.close()
        with self._lease_condition:
            self._closed = False
        if self._host_staging_enabled:
            try:
                self.importer = None
                self.external_semaphore_enabled = False
                self._external_semaphore_request_enabled = False
                self._external_semaphore_error = None
                self._external_semaphore_request_reason = (
                    "host_staging_eye_upload"
                )
                for eye in ("left", "right"):
                    slots = getattr(self, f"{eye}_slots")
                    for index in range(self.ring_size):
                        image = VulkanExportableImage(
                            context, width, height,
                            label=f"runtime-{eye}-eye-{index}",
                            format=context.vk.VK_FORMAT_R8G8B8A8_SRGB,
                        )
                        slots.append(image)
                        context.prepare_external_image_for_producer(
                            image.resource, wait=False
                        )
                self._host_eye_slots = []
                for index in range(self.ring_size):
                    left_host = VulkanHostImage(
                        context, width, height,
                        format=context.vk.VK_FORMAT_R8G8B8A8_SRGB,
                        label=f"runtime-left-eye-host-{index}",
                    )
                    right_host = VulkanHostImage(
                        context, width, height,
                        format=context.vk.VK_FORMAT_R8G8B8A8_SRGB,
                        label=f"runtime-right-eye-host-{index}",
                    )
                    self._host_eye_slots.append((left_host, right_host))
                for eye in ("left", "right"):
                    for index in range(self.ring_size):
                        getattr(self, f"{eye}_visible_semaphores").append(
                            VulkanBinarySemaphore(
                                context, label=f"runtime-{eye}-visible-{index}"
                            )
                        )
            except Exception:
                self.close()
                raise
            self._extent = (width, height)
            self._logged_external_sync_mode = True
            print(
                "[VulkanOutput] ROCm eye upload: vulkan_host_staging "
                "(no HIP/Vulkan shared memory)",
                flush=True,
            )
            return
        self.importer = self._create_importer()
        self._external_semaphore_error = None
        self._external_semaphore_request_reason = None
        requested = self._external_semaphore_requested()
        available = bool(
            self.importer.capabilities.external_semaphore
            and getattr(self.presenter, "source_ready_semaphore_available", False)
        )
        self._external_semaphore_request_enabled = requested
        self.external_semaphore_enabled = requested and available
        if not requested:
            self._external_semaphore_request_reason = "disabled_by_D2S_ENABLE_ROCM_EXTERNAL_SEMAPHORE"
        elif not available:
            self._external_semaphore_request_reason = "rocm_external_semaphore_unavailable"
        try:
            memory_handle_type = None
            if self._rocm_kmt_memory_enabled():
                memory_handle_type = getattr(
                    context.vk,
                    "VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT",
                    None,
                )
            buffer_usage = (
                context.vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | context.vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            )
            self._eye_buffers = []
            for eye in ("left", "right"):
                slots = getattr(self, f"{eye}_slots")
                for index in range(self.ring_size):
                    image = VulkanExportableImage(
                        context, width, height,
                        label=f"runtime-{eye}-eye-{index}",
                        format=context.vk.VK_FORMAT_R8G8B8A8_SRGB,
                        memory_handle_type=memory_handle_type,
                    )
                    slots.append(image)
                    self.importer.register_slot(image)
                for index in range(self.ring_size):
                    left_buffer = VulkanExportableBuffer(
                        context,
                        width * height * 4,
                        label=f"runtime-left-eye-buffer-{index}",
                        usage=buffer_usage,
                        memory_handle_type=memory_handle_type,
                    )
                    right_buffer = VulkanExportableBuffer(
                        context,
                        width * height * 4,
                        label=f"runtime-right-eye-buffer-{index}",
                        usage=buffer_usage,
                        memory_handle_type=memory_handle_type,
                    )
                    self.importer.register_buffer(left_buffer)
                    self.importer.register_buffer(right_buffer)
                    self._eye_buffers.append((left_buffer, right_buffer))
                if not self.external_semaphore_enabled:
                    continue
                for kind in ("ready", "release"):
                    semaphores = getattr(self, f"{eye}_{kind}_semaphores")
                    for index in range(self.ring_size):
                        semaphore = VulkanExportableSemaphore(
                            context, label=f"runtime-{eye}-{kind}-{index}", timeline=False,
                        )
                        semaphores.append(semaphore)
                        self.importer.register_semaphore(semaphore)
                    setattr(self, f"{eye}_{kind}_values", [0] * self.ring_size)
                for index in range(self.ring_size):
                    getattr(self, f"{eye}_visible_semaphores").append(
                        VulkanBinarySemaphore(context, label=f"runtime-{eye}-visible-{index}")
                    )
        except Exception:
            self.close()
            raise
        self._extent = (width, height)
        self._rocm_release_timelines = [0 for _ in range(self.ring_size)]
        memory_mode = (
            "kmt_win32"
            if memory_handle_type is not None
            else "opaque_win32"
        )
        print(
            "[VulkanOutput] ROCm external binary semaphore sync: "
            f"requested={requested} available={available} active={self.external_semaphore_enabled} "
            f"blocked_reason={self._external_semaphore_request_reason or 'none'} "
            f"memory={memory_mode}",
            flush=True,
        )
        self._logged_external_sync_mode = True

    def _claim_slot(self, slot_index: int, frame_id: int) -> None:
        if not self.external_semaphore_enabled and slot_index < len(self._rocm_release_timelines):
            release_timeline = int(self._rocm_release_timelines[slot_index])
            if release_timeline > 0:
                # Wait only for this ring slot. Device-wide idle here stalls
                # Filament and can starve the XR frame loop.
                self.presenter.vulkan.wait_for_timeline(release_timeline)
                if self._rocm_release_timelines[slot_index] == release_timeline:
                    self._rocm_release_timelines[slot_index] = 0
        super()._claim_slot(slot_index, frame_id)
        if not self.external_semaphore_enabled and slot_index < len(self._rocm_release_timelines):
            release_timeline = int(self._rocm_release_timelines[slot_index])
            if release_timeline > 0:
                try:
                    self.presenter.vulkan.wait_for_timeline(release_timeline)
                except Exception:
                    # Do not strand the lease when a bounded wait expires;
                    # the timeline remains recorded for the next attempt.
                    super().release_frame(frame_id)
                    raise
                if self._rocm_release_timelines[slot_index] == release_timeline:
                    self._rocm_release_timelines[slot_index] = 0

    def _convert_host_staging(
        self, runtime_result, *, frame_id: int, timestamp: float
    ) -> VulkanStereoOutputFrame:
        left = getattr(runtime_result, "left_eye", None)
        right = getattr(runtime_result, "right_eye", None)
        width, height = self._tensor_extent(left)
        if self._tensor_extent(right) != (width, height):
            raise ValueError("left/right runtime eye dimensions differ")
        self._ensure_slots(width, height)
        slot_index = int(frame_id) % self.ring_size
        self._claim_slot(slot_index, frame_id)
        self.left_slot = self.left_slots[slot_index]
        self.right_slot = self.right_slots[slot_index]
        glow_metadata: dict[str, object] = {}
        try:
            if self._screen_light_sample_path != "vulkan_compute_reduction":
                self._update_screen_light_sample(left, right)
            glow_source = getattr(runtime_result, "source_rgb", None)
            glow_source = glow_source if glow_source is not None else left
            glow_metadata = self._update_glow_gpu_source(
                glow_source, frame_id=frame_id
            )
            sampled_light = glow_metadata.get("screen_light_linear_rgb")
            if isinstance(sampled_light, (list, tuple)) and len(sampled_light) >= 3:
                self._screen_light_rgb = tuple(
                    float(value) for value in sampled_light[:3]
                )
                self._screen_light_sample_path = str(
                    glow_metadata.get(
                        "screen_light_sample_path", "vulkan_compute_reduction"
                    )
                )
            if len(self._host_eye_slots) <= slot_index:
                raise RuntimeError("ROCm host eye staging slots are unavailable")
            left_host, right_host = self._host_eye_slots[slot_index]
            left_host.upload(
                self._tensor_host_rgba(left, width=width, height=height)
            )
            right_host.upload(
                self._tensor_host_rgba(right, width=width, height=height)
            )
        except Exception:
            glow_release = glow_metadata.get("_vulkan_glow_release")
            if callable(glow_release):
                glow_release(frame_id)
            self.release_frame(frame_id)
            raise
        self._host_frame_slots[int(frame_id)] = slot_index
        self._source_frames[int(frame_id)] = (
            self.left_slot,
            self.right_slot,
            slot_index,
        )
        self._released_source_frames.discard(int(frame_id))
        left_contract = self.source_image_contract(self.left_slot.resource)
        right_contract = self.source_image_contract(self.right_slot.resource)
        return VulkanStereoOutputFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            left_eye=self.left_slot.resource,
            right_eye=self.right_slot.resource,
            ready_timeline=None,
            metadata={
                **dict(getattr(runtime_result, "debug_info", None) or {}),
                "vulkan_output_ring_slot": slot_index,
                "vulkan_output_ring_size": self.ring_size,
                "vulkan_output_sync": self.output_sync_mode,
                "vulkan_external_semaphore_available": False,
                "vulkan_external_semaphore_requested": False,
                "vulkan_readback": "none",
                "vulkan_output_path": "rocm_host_eye_staging",
                "vulkan_output_image_direct": False,
                "vulkan_gpu_to_cpu": True,
                "vulkan_zero_cpu_readback": False,
                "vulkan_zero_copy": False,
                "vulkan_source_layout_left": left_contract["layout"],
                "vulkan_source_layout_right": right_contract["layout"],
                "vulkan_source_queue_family_left": left_contract["queue_family"],
                "vulkan_source_queue_family_right": right_contract["queue_family"],
                "_vulkan_source_prepare_for_sampling": self.prepare_source_for_sampling,
                "_vulkan_source_consumer_release": self.release_consumer_frame,
                "_vulkan_output_release": self.release_frame,
                "screen_light_linear_rgb": self._screen_light_rgb,
                "screen_light_sample_path": self._screen_light_sample_path,
                **glow_metadata,
            },
            color_space="srgb",
            image_origin="top_left",
        )

    def convert(self, runtime_result, *, frame_id: int, timestamp: float):
        if self._host_staging_enabled:
            return self._convert_host_staging(
                runtime_result, frame_id=frame_id, timestamp=timestamp
            )
        return self._convert_buffer_staging(
            runtime_result, frame_id=frame_id, timestamp=timestamp
        )

    def _convert_buffer_staging(self, runtime_result, *, frame_id: int, timestamp: float):
        left = getattr(runtime_result, "left_eye", None)
        right = getattr(runtime_result, "right_eye", None)
        width, height = self._tensor_extent(left)
        if self._tensor_extent(right) != (width, height):
            raise ValueError("left/right runtime eye dimensions differ")
        self._ensure_slots(width, height)
        slot_index = int(frame_id) % self.ring_size
        self._claim_slot(slot_index, frame_id)
        self.left_slot = self.left_slots[slot_index]
        self.right_slot = self.right_slots[slot_index]
        glow_metadata: dict[str, object] = {}
        use_external_semaphore = bool(
            self.external_semaphore_enabled
            and getattr(self.presenter, "source_ready_semaphore_available", False)
        )
        external_semaphore_requested = bool(
            self._external_semaphore_request_enabled
        )
        try:
            if self._screen_light_sample_path != "vulkan_compute_reduction":
                self._update_screen_light_sample(left, right)
            glow_source = getattr(runtime_result, "source_rgb", None)
            glow_source = glow_source if glow_source is not None else left
            glow_metadata = self._update_glow_gpu_source(
                glow_source, frame_id=frame_id
            )
            sampled_light = glow_metadata.get("screen_light_linear_rgb")
            if isinstance(sampled_light, (list, tuple)) and len(sampled_light) >= 3:
                self._screen_light_rgb = tuple(
                    float(value) for value in sampled_light[:3]
                )
                self._screen_light_sample_path = str(
                    glow_metadata.get(
                        "screen_light_sample_path", "vulkan_compute_reduction"
                    )
                )
            left_ready = None
            right_ready = None
            if not self._eye_buffers:
                raise RuntimeError("ROCm eye staging buffers are unavailable")
            left_buffer, right_buffer = self._eye_buffers[slot_index]
            if use_external_semaphore:
                for eye_index, release_semaphore, release_values in (
                    (
                        0,
                        self.left_release_semaphores[slot_index],
                        self.left_release_values,
                    ),
                    (
                        1,
                        self.right_release_semaphores[slot_index],
                        self.right_release_values,
                    ),
                ):
                    if (eye_index, slot_index) not in self._release_signaled:
                        continue
                    self.importer.wait_semaphore(
                        release_semaphore,
                        value=release_values[slot_index],
                    )
                    self._release_signaled.discard((eye_index, slot_index))
                self.importer.copy_tensor_to_buffer(left, left_buffer)
                self.importer.copy_tensor_to_buffer(right, right_buffer)
                left_ready = self.left_ready_semaphores[slot_index]
                right_ready = self.right_ready_semaphores[slot_index]
                self.left_ready_values[slot_index] += 1
                self.right_ready_values[slot_index] += 1
                self.importer.signal_semaphore(
                    left_ready,
                    value=self.left_ready_values[slot_index],
                )
                self.importer.signal_semaphore(
                    right_ready,
                    value=self.right_ready_values[slot_index],
                )
                # Keep the HIP signal completion explicit before Vulkan queues
                # its wait; host-side hipMemcpy completion alone does not
                # establish the cross-API visibility handoff on AMD Windows.
                self.importer.synchronize()
                self._rocm_ready_pending.update(
                    ((int(frame_id), 0), (int(frame_id), 1))
                )
            else:
                self.importer.copy_tensor_to_buffer(left, left_buffer)
                self.importer.copy_tensor_to_buffer(right, right_buffer)
                if not bool(getattr(self.importer, "uses_synchronous_buffer_copy", False)):
                    self.importer.synchronize()
                copy_timelines = []
                for eye_index, resource, buffer in (
                    (0, self.left_slot.resource, left_buffer),
                    (1, self.right_slot.resource, right_buffer),
                ):
                    copy_timelines.append(
                        self.presenter.vulkan.copy_buffer_to_image(buffer, resource)
                    )
                # Both copies are ordered on the graphics queue. The final
                # timeline is carried with the frame so the presenter can wait
                # on the GPU queue instead of blocking this thread.
                ready_timeline = max(copy_timelines)
                for eye_index, _resource, _buffer in (
                    (0, self.left_slot.resource, left_buffer),
                    (1, self.right_slot.resource, right_buffer),
                ):
                    self._prepared_source_eyes.add((int(frame_id), eye_index))
                self._rocm_ready_timelines[int(frame_id)] = int(ready_timeline)
        except Exception:
            glow_release = glow_metadata.get("_vulkan_glow_release")
            if callable(glow_release):
                glow_release(frame_id)
            self.release_frame(frame_id)
            raise
        self._buffer_frames[int(frame_id)] = (left_buffer, right_buffer)
        self._source_frames[int(frame_id)] = (
            self.left_slot,
            self.right_slot,
            slot_index,
        )
        self._released_source_frames.discard(int(frame_id))
        left_contract = self.source_image_contract(self.left_slot.resource)
        right_contract = self.source_image_contract(self.right_slot.resource)
        return VulkanStereoOutputFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            left_eye=self.left_slot.resource,
            right_eye=self.right_slot.resource,
            ready_timeline=int(self._rocm_ready_timelines[int(frame_id)]),
            metadata={
                **dict(getattr(runtime_result, "debug_info", None) or {}),
                "vulkan_output_ring_slot": slot_index,
                "vulkan_output_ring_size": self.ring_size,
                "vulkan_output_sync": (
                    self.external_semaphore_sync_mode
                    if use_external_semaphore
                    else self.output_sync_mode
                ),
                "vulkan_ready_semaphore_left": (
                    left_ready.semaphore if left_ready is not None else None
                ),
                "vulkan_ready_semaphore_right": (
                    right_ready.semaphore if right_ready is not None else None
                ),
                "vulkan_external_semaphore_available": bool(
                    use_external_semaphore
                ),
                "vulkan_external_semaphore_type": (
                    "binary" if use_external_semaphore else None
                ),
                "vulkan_external_semaphore_requested": bool(
                    external_semaphore_requested
                ),
                "vulkan_external_semaphore_request_reason": (
                    self._external_semaphore_request_reason
                ),
                "vulkan_external_semaphore_error": self._external_semaphore_error,
                "vulkan_readback": "none",
                "vulkan_output_path": "hip_staging_buffer_to_vulkan_image",
                "vulkan_output_image_direct": False,
                "vulkan_gpu_to_cpu": False,
                "vulkan_zero_cpu_readback": False,
                "vulkan_zero_copy": False,
                "vulkan_source_layout_left": left_contract["layout"],
                "vulkan_source_layout_right": right_contract["layout"],
                "vulkan_source_queue_family_left": left_contract["queue_family"],
                "vulkan_source_queue_family_right": right_contract["queue_family"],
                "_vulkan_source_prepare_for_sampling": self.prepare_source_for_sampling,
                "_vulkan_source_consumer_release": self.release_consumer_frame,
                "_vulkan_output_release": self.release_frame,
                "screen_light_linear_rgb": self._screen_light_rgb,
                "screen_light_sample_path": self._screen_light_sample_path,
                **glow_metadata,
            },
            color_space="srgb",
            image_origin="top_left",
        )

    def prepare_source_for_sampling(self, frame_id: int, eye_index: int):
        frame_key = int(frame_id)
        eye = int(eye_index)
        if (frame_key, eye) in self._prepared_source_eyes:
            return None
        if self._host_staging_enabled:
            slot_index = self._host_frame_slots.get(frame_key)
            entry = self._source_frames.get(frame_key)
            if slot_index is None or entry is None:
                raise RuntimeError(f"unknown ROCm host source frame {frame_id}")
            resource = entry[0 if eye == 0 else 1].resource
            host_slots = self._host_eye_slots[slot_index]
            visible = (
                self.left_visible_semaphores[slot_index]
                if eye == 0
                else self.right_visible_semaphores[slot_index]
            )
            self.presenter.vulkan.copy_image(
                host_slots[eye].resource,
                resource,
                wait_semaphore=None,
            )
            self._transition_host_eye_to_sampling(
                resource, signal_semaphore=visible.semaphore
            )
            self._prepared_source_eyes.add((frame_key, eye))
            return visible.semaphore
        buffers = self._buffer_frames.get(frame_key)
        entry = self._source_frames.get(frame_key)
        if buffers is None or entry is None:
            raise RuntimeError(f"unknown ROCm Vulkan source frame {frame_id}")
        slot_index = entry[2]
        resource = entry[0 if eye == 0 else 1].resource
        buffer = buffers[eye]
        if not self.external_semaphore_enabled:
            self._prepared_source_eyes.add((frame_key, eye))
            return None
        ready = (
            self.left_ready_semaphores[slot_index]
            if eye == 0
            else self.right_ready_semaphores[slot_index]
        )
        visible = (
            self.left_visible_semaphores[slot_index]
            if eye == 0
            else self.right_visible_semaphores[slot_index]
        )
        self.presenter.vulkan.copy_buffer_to_image(
            buffer,
            resource,
            wait_semaphore=ready.semaphore,
            wait_semaphore_value=(
                self.left_ready_values[slot_index]
                if eye == 0
                else self.right_ready_values[slot_index]
            ),
            signal_semaphore=visible.semaphore,
        )
        self._prepared_source_eyes.add((frame_key, eye))
        self._rocm_ready_pending.discard((frame_key, eye))
        return visible.semaphore

    def _transition_host_eye_to_sampling(
        self, resource: Any, *, signal_semaphore: Any
    ) -> None:
        """Finish a VulkanHostImage copy into a sampling layout."""
        vk = self.presenter.vulkan.vk
        state = self.presenter.vulkan.image_state(resource.image)

        def record(command_buffer: Any) -> None:
            barrier = vk.VkImageMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                oldLayout=int(state.layout),
                newLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=resource.image,
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            )
            vk.vkCmdPipelineBarrier(
                command_buffer,
                vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                | vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [barrier],
            )

        self.presenter.vulkan.submit_on(
            "graphics", record, signal_semaphore=signal_semaphore
        )
        self.presenter.vulkan.register_image_state(
            resource.image,
            type(state)(
                layout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                access_mask=vk.VK_ACCESS_SHADER_READ_BIT,
                stage_mask=(
                    vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                    | vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                ),
                queue_family_index=self.presenter.vulkan.queue_family_index,
            ),
        )

    def release_consumer_frame(
        self,
        frame_id: int,
        consumer_semaphores=None,
        *,
        wait_for_timeline: int | None = None,
    ) -> None:
        frame_key = int(frame_id)
        effective_wait_for_timeline = wait_for_timeline
        if effective_wait_for_timeline is None:
            effective_wait_for_timeline = getattr(
                self, "_rocm_ready_timelines", {}
            ).get(frame_key)
        if not self._host_staging_enabled:
            return super().release_consumer_frame(
                frame_id,
                consumer_semaphores,
                wait_for_timeline=effective_wait_for_timeline,
            )
        if frame_key in self._released_source_frames:
            return
        entry = self._source_frames.get(frame_key)
        if entry is None:
            self.release_frame(frame_key)
            self._released_source_frames.add(frame_key)
            return
        context = self.presenter.vulkan
        if bool(getattr(context, "device_lost", False)):
            self._discard_source_frame_after_device_loss(frame_key)
            return
        waits = tuple(consumer_semaphores or ())
        try:
            for eye_index, resource in (
                (0, entry[0].resource),
                (1, entry[1].resource),
            ):
                if (frame_key, eye_index) not in self._prepared_source_eyes:
                    continue
                context.release_external_image_from_sampling(
                    resource,
                    wait_for_timeline=effective_wait_for_timeline,
                    wait_semaphore=(
                        waits[eye_index] if eye_index < len(waits) else None
                    ),
                )
                self._prepared_source_eyes.discard((frame_key, eye_index))
        except Exception:
            if bool(getattr(context, "device_lost", False)):
                self._discard_source_frame_after_device_loss(frame_key)
            raise
        self.release_frame(frame_key)
        self._released_source_frames.add(frame_key)
        self._source_frames.pop(frame_key, None)
        getattr(self, "_rocm_ready_timelines", {}).pop(frame_key, None)

    def release_frame(self, frame_id: int) -> None:
        frame_key = int(frame_id)
        entry = self._source_frames.get(frame_key)
        if entry is not None:
            slot_index = entry[2]
            pending = [
                semaphores[slot_index].semaphore
                for eye, semaphores in enumerate(
                    (self.left_ready_semaphores, self.right_ready_semaphores)
                )
                if (frame_key, eye) in self._rocm_ready_pending
            ]
            if pending and not bool(
                getattr(self.presenter.vulkan, "device_lost", False)
            ):
                # A dropped frame may never reach prepare_source_for_sampling;
                # consume its binary ready signals before reusing the slot.
                timeline = self.presenter.vulkan.submit_on(
                    "graphics",
                    lambda _command_buffer: None,
                    wait_semaphore=tuple(pending),
                )
                self.presenter.vulkan.wait_for_timeline(timeline)
        self._rocm_ready_pending.difference_update(((frame_key, 0), (frame_key, 1)))
        self._buffer_frames.pop(frame_key, None)
        self._host_frame_slots.pop(frame_key, None)
        getattr(self, "_rocm_ready_timelines", {}).pop(frame_key, None)
        super().release_frame(frame_key)

    def close(self) -> None:
        self._rocm_ready_pending.clear()
        self._buffer_frames.clear()
        self._host_frame_slots.clear()
        self._rocm_release_timelines = []
        self._rocm_ready_timelines.clear()
        super().close()
        for host_slots in self._host_eye_slots:
            for host in host_slots:
                try:
                    host.close()
                except Exception:
                    pass
        self._host_eye_slots = []
        for buffers in self._eye_buffers:
            for buffer in buffers:
                try:
                    buffer.close()
                except Exception:
                    pass
        self._eye_buffers = []


class VulkanHostOutputAdapter(GpuProducerAdapter):
    """Upload Vulkan-compute results through host-visible Vulkan images.

    This is the cross-vendor correctness fallback. It intentionally reports
    synchronized GPU copy rather than external-semaphore sync, so the
    Presenter uses its regular projection/Quad copy path.
    """

    backend_name = "vulkan_host"

    def __init__(self, presenter):
        self.presenter = presenter
        self.ring_size = max(2, int(os.environ.get("D2S_VULKAN_OUTPUT_RING_SIZE", "3")))
        self.left_slots: list[VulkanHostImage] = []
        self.right_slots: list[VulkanHostImage] = []
        self._extent: tuple[int, int] | None = None

    @staticmethod
    def _tensor_to_rgba(tensor, *, width: int, height: int):
        import numpy as np

        try:
            import torch

            value = tensor.detach().to(device="cpu")
            if value.ndim == 4:
                value = value[0]
            if value.ndim == 3 and int(value.shape[0]) in (3, 4):
                value = value[:3].permute(1, 2, 0)
            elif value.ndim == 3 and int(value.shape[-1]) in (3, 4):
                value = value[..., :3]
            else:
                raise ValueError(f"expected RGB tensor, got {tuple(value.shape)}")
            if tuple(value.shape[:2]) != (height, width):
                raise ValueError(
                    f"Vulkan host output dimensions differ: {tuple(value.shape[:2])} != {(height, width)}"
                )
            value = torch.nan_to_num(value.float(), nan=0.0, posinf=1.0, neginf=0.0)
            if getattr(tensor, "dtype", None) == torch.uint8:
                pixels = value.clamp(0.0, 255.0).to(torch.uint8).numpy()
            else:
                pixels = value.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
        except AttributeError:
            pixels = np.asarray(tensor)
            if pixels.ndim == 3 and pixels.shape[0] in (3, 4):
                pixels = np.transpose(pixels[:3], (1, 2, 0))
            pixels = np.asarray(pixels, dtype=np.float32)
            if pixels.max(initial=0.0) <= 1.0:
                pixels = pixels * 255.0
            pixels = np.clip(np.rint(pixels), 0.0, 255.0).astype(np.uint8)
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = pixels
        rgba[..., 3] = 255
        return np.ascontiguousarray(rgba)

    def _ensure_slots(self, width: int, height: int) -> None:
        extent = (int(width), int(height))
        if self._extent == extent and self.left_slots and self.right_slots:
            return
        context = getattr(self.presenter, "vulkan", None)
        if context is None:
            raise RuntimeError("Presenter Vulkan context is unavailable")
        context.wait_idle()
        for slot in (*self.left_slots, *self.right_slots):
            slot.close()
        format_value = context.vk.VK_FORMAT_R8G8B8A8_UNORM
        self.left_slots = [
            VulkanHostImage(
                context,
                extent[0],
                extent[1],
                format=format_value,
                label=f"runtime-vulkan-host-left-{index}",
            )
            for index in range(self.ring_size)
        ]
        self.right_slots = [
            VulkanHostImage(
                context,
                extent[0],
                extent[1],
                format=format_value,
                label=f"runtime-vulkan-host-right-{index}",
            )
            for index in range(self.ring_size)
        ]
        self._extent = extent

    def convert(self, runtime_result, *, frame_id: int, timestamp: float):
        left = getattr(runtime_result, "left_eye", None)
        right = getattr(runtime_result, "right_eye", None)
        width, height = self._tensor_extent(left)
        if self._tensor_extent(right) != (width, height):
            raise ValueError("left/right runtime eye dimensions differ")
        self._ensure_slots(width, height)
        # The host image ring is reused only after the previous Vulkan copy has
        # completed. This is slower than external semaphores but race-free.
        wait_start = time.perf_counter()
        self.presenter.vulkan.wait_idle()
        wait_idle_ms = (time.perf_counter() - wait_start) * 1000.0
        slot_index = int(frame_id) % self.ring_size
        left_slot = self.left_slots[slot_index]
        right_slot = self.right_slots[slot_index]
        upload_start = time.perf_counter()
        left_slot.upload(self._tensor_to_rgba(left, width=width, height=height))
        right_slot.upload(self._tensor_to_rgba(right, width=width, height=height))
        upload_ms = (time.perf_counter() - upload_start) * 1000.0
        left_contract = self.source_image_contract(left_slot.resource)
        right_contract = self.source_image_contract(right_slot.resource)
        return VulkanStereoOutputFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            left_eye=left_slot.resource,
            right_eye=right_slot.resource,
            metadata={
                **dict(getattr(runtime_result, "debug_info", None) or {}),
                "vulkan_output_ring_slot": slot_index,
                "vulkan_output_ring_size": self.ring_size,
                "vulkan_output_sync": self.output_sync_mode,
                "vulkan_output_wait_idle_ms": wait_idle_ms,
                "vulkan_output_upload_ms": upload_ms,
                "vulkan_output_path": "host_visible_vulkan_image",
                "vulkan_source_layout_left": left_contract["layout"],
                "vulkan_source_layout_right": right_contract["layout"],
                "vulkan_source_queue_family_left": left_contract["queue_family"],
                "vulkan_source_queue_family_right": right_contract["queue_family"],
                "_vulkan_output_release": self.release_frame,
            },
            color_space="srgb",
            image_origin="top_left",
        )

    @staticmethod
    def _tensor_extent(tensor) -> tuple[int, int]:
        shape = tuple(getattr(tensor, "shape", ()))
        if len(shape) == 4:
            return int(shape[-1]), int(shape[-2])
        if len(shape) == 3:
            # Runtime OpenXR eyes are HxWxC, while compatibility callers may
            # still provide CxHxW tensors.
            if int(shape[-1]) in (3, 4):
                return int(shape[1]), int(shape[0])
            if int(shape[0]) in (3, 4):
                return int(shape[2]), int(shape[1])
        raise ValueError(f"Vulkan host output requires BCHW, HWC, or CHW tensor, got {shape}")

    def release_frame(self, frame_id: int) -> None:
        return None

    def close(self) -> None:
        context = getattr(self.presenter, "vulkan", None)
        if context is not None:
            context.wait_idle()
        for slot in (*self.left_slots, *self.right_slots):
            slot.close()
        self.left_slots = []
        self.right_slots = []
        self._extent = None


class VulkanZeroCopyOutputAdapter(CudaVulkanOutputAdapter):
    """Run Vulkan stereo Compute into images consumed directly by Filament."""

    backend_name = "vulkan_zero_copy"

    def __init__(self, presenter):
        super().__init__(presenter)
        self._compute_backend = None
        self._release_timelines: list[int] = []

    @staticmethod
    def _external_semaphore_requested() -> bool:
        return True

    def _ensure_slots(self, width: int, height: int) -> None:
        extent = (int(width), int(height))
        if self._extent == extent and self.left_slots and self.right_slots:
            return
        context = getattr(self.presenter, "vulkan", None)
        if context is None or not bool(getattr(self.presenter, "initialized", False)):
            raise RuntimeError("Presenter Vulkan context is unavailable")
        self.close()
        with self._lease_condition:
            self._closed = False
        from stereo_runtime.vulkan_backend import VulkanStereoImageComputeBackend

        self._compute_backend = VulkanStereoImageComputeBackend(context)
        self.left_slots = [
            VulkanExportableImage(
                context,
                extent[0],
                extent[1],
                label=f"runtime-zero-copy-left-{index}",
                format=context.vk.VK_FORMAT_R8G8B8A8_UNORM,
            )
            for index in range(self.ring_size)
        ]
        self.right_slots = [
            VulkanExportableImage(
                context,
                extent[0],
                extent[1],
                label=f"runtime-zero-copy-right-{index}",
                format=context.vk.VK_FORMAT_R8G8B8A8_UNORM,
            )
            for index in range(self.ring_size)
        ]
        self.left_visible_semaphores = [
            VulkanBinarySemaphore(context, label=f"runtime-zero-copy-left-visible-{index}")
            for index in range(self.ring_size)
        ]
        self.right_visible_semaphores = [
            VulkanBinarySemaphore(context, label=f"runtime-zero-copy-right-visible-{index}")
            for index in range(self.ring_size)
        ]
        self._release_timelines = [0 for _ in range(self.ring_size)]
        for slot in (*self.left_slots, *self.right_slots):
            context.prepare_external_image_for_producer(slot.resource)
        self._extent = extent

    def prepare_source_for_sampling(self, frame_id: int, eye_index: int):
        frame_key = int(frame_id)
        eye = int(eye_index)
        entry = self._source_frames.get(frame_key)
        if entry is None:
            raise RuntimeError(f"unknown Vulkan zero-copy source frame {frame_id}")
        _left, _right, slot_index = entry
        visible = (
            self.left_visible_semaphores[slot_index]
            if eye == 0
            else self.right_visible_semaphores[slot_index]
        )
        if (frame_key, eye) in self._prepared_source_eyes:
            # Vulkan compute signalled visible once when the frame was
            # produced. A reused frame has the same layout and content, so
            # Filament can sample it without another ready semaphore.
            return None
        self._prepared_source_eyes.add((frame_key, eye))
        return visible.semaphore

    def release_consumer_frame(
        self,
        frame_id: int,
        consumer_semaphores=None,
        *,
        wait_for_timeline: int | None = None,
    ) -> None:
        frame_key = int(frame_id)
        if frame_key in self._released_source_frames:
            return
        entry = self._source_frames.get(frame_key)
        if entry is None:
            self.release_frame(frame_key)
            self._released_source_frames.add(frame_key)
            return
        left, right, slot_index = entry
        waits = tuple(consumer_semaphores or ())
        context = self.presenter.vulkan
        if bool(getattr(context, "device_lost", False)):
            self._discard_source_frame_after_device_loss(frame_key)
            return
        try:
            for eye_index, slot in ((0, left), (1, right)):
                if (frame_key, eye_index) not in self._prepared_source_eyes:
                    continue
                timeline = context.release_external_image_from_sampling(
                    slot.resource,
                    wait_for_timeline=wait_for_timeline,
                    wait_semaphore=waits[eye_index] if eye_index < len(waits) else None,
                )
                self._release_timelines[slot_index] = max(
                    int(self._release_timelines[slot_index]), int(timeline)
                )
                self._prepared_source_eyes.discard((frame_key, eye_index))
        except Exception:
            if bool(getattr(context, "device_lost", False)):
                self._discard_source_frame_after_device_loss(frame_key)
            raise
        self.release_frame(frame_key)
        self._released_source_frames.add(frame_key)
        self._source_frames.pop(frame_key, None)

    def release_output_frame(self, frame_id: int, *, wait_for_timeline: int | None = None) -> None:
        """Return a source image after the non-Filament GPU-copy fallback."""
        frame_key = int(frame_id)
        entry = self._source_frames.get(frame_key)
        if entry is None:
            self.release_frame(frame_key)
            return
        left, right, slot_index = entry
        if bool(getattr(self.presenter.vulkan, "device_lost", False)):
            self._discard_source_frame_after_device_loss(frame_key)
            return
        for slot in (left, right):
            state = self.presenter.vulkan.image_state(slot.resource.image)
            if state.layout == self.presenter.vulkan.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                timeline = self.presenter.vulkan.release_external_image_from_sampling(
                    slot.resource,
                    wait_for_timeline=wait_for_timeline,
                )
                self._release_timelines[slot_index] = max(
                    int(self._release_timelines[slot_index]), int(timeline)
                )
        self.release_frame(frame_key)
        self._source_frames.pop(frame_key, None)

    def convert(self, runtime_result, *, frame_id: int, timestamp: float):
        request = getattr(runtime_result, "vulkan_compute_request", None)
        if request is None:
            raise ValueError("Vulkan zero-copy output requires a deferred Compute request")
        rgb = request.rgb
        depth = request.depth
        shape = tuple(int(value) for value in getattr(rgb, "shape", ()))
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            raise ValueError(f"Vulkan zero-copy RGB request requires [1,3,H,W], got {shape}")
        width, height = shape[-1], shape[-2]
        self._ensure_slots(width, height)
        slot_index = int(frame_id) % self.ring_size
        self._claim_slot(slot_index, frame_id)
        left_slot = self.left_slots[slot_index]
        right_slot = self.right_slots[slot_index]
        try:
            if self._compute_backend is None:
                raise RuntimeError("Vulkan zero-copy Compute backend is unavailable")
            compute_timeline, backend_debug = self._compute_backend.submit_to_images(
                rgb,
                depth,
                left_slot.resource,
                right_slot.resource,
                params=request.params,
                ready_timeline=self._release_timelines[slot_index] or None,
            )
            left_visible = self.left_visible_semaphores[slot_index]
            right_visible = self.right_visible_semaphores[slot_index]
            self.presenter.vulkan.prepare_external_image_for_sampling(
                left_slot.resource,
                wait_for_timeline=compute_timeline,
                signal_semaphore=left_visible.semaphore,
            )
            self.presenter.vulkan.prepare_external_image_for_sampling(
                right_slot.resource,
                wait_for_timeline=compute_timeline,
                signal_semaphore=right_visible.semaphore,
            )
        except Exception:
            self.release_frame(frame_id)
            raise
        self._source_frames[int(frame_id)] = (left_slot, right_slot, slot_index)
        self._released_source_frames.discard(int(frame_id))
        left_contract = self.source_image_contract(left_slot.resource)
        right_contract = self.source_image_contract(right_slot.resource)
        return VulkanStereoOutputFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            left_eye=left_slot.resource,
            right_eye=right_slot.resource,
            ready_timeline=None,
            metadata={
                **dict(getattr(runtime_result, "debug_info", None) or {}),
                **backend_debug,
                "vulkan_output_ring_slot": slot_index,
                "vulkan_output_ring_size": self.ring_size,
                "vulkan_output_sync": "vulkan_compute_external_semaphore",
                "vulkan_ready_semaphore_left": left_visible.semaphore,
                "vulkan_ready_semaphore_right": right_visible.semaphore,
                "vulkan_external_semaphore_available": True,
                "vulkan_source_layout_left": left_contract["layout"],
                "vulkan_source_layout_right": right_contract["layout"],
                "vulkan_source_queue_family_left": left_contract["queue_family"],
                "vulkan_source_queue_family_right": right_contract["queue_family"],
                "vulkan_output_path": "presenter_owned_storage_image",
                "vulkan_output_image_direct": True,
                "vulkan_gpu_to_cpu": False,
                "vulkan_zero_cpu_readback": True,
                # The current request carries CUDA tensors and the Vulkan
                # compute bridge imports them into external input buffers.
                # That is GPU-only, but not strict no-copy ownership.
                "vulkan_zero_copy": False,
                "_vulkan_source_prepare_for_sampling": self.prepare_source_for_sampling,
                "_vulkan_source_consumer_release": self.release_consumer_frame,
                "_vulkan_output_release": self.release_output_frame,
            },
            color_space="srgb",
            image_origin="top_left",
        )

    def close(self) -> None:
        context = getattr(self.presenter, "vulkan", None)
        if context is not None:
            context.wait_idle()
        if self._compute_backend is not None:
            self._compute_backend.close()
        self._compute_backend = None
        for slot in (*self.left_slots, *self.right_slots):
            slot.close()
        for semaphore in (*self.left_visible_semaphores, *self.right_visible_semaphores):
            semaphore.close()
        self.left_slots = []
        self.right_slots = []
        self.left_visible_semaphores = []
        self.right_visible_semaphores = []
        self._release_timelines = []
        self._source_frames.clear()
        self._released_source_frames.clear()
        self._prepared_source_eyes.clear()
        self._extent = None
        with self._lease_condition:
            self._closed = True
            self._active_leases.clear()
            self._lease_condition.notify_all()


class VulkanRuntimeOutputConsumer:
    """Bridge the bounded runtime queue to a Vulkan-capable output sink."""

    def __init__(self, *, runtime_q, shutdown_event, source_stat_inc, sink=None, gpu_adapter=None):
        self.runtime_q = runtime_q
        self.shutdown_event = shutdown_event
        self.source_stat_inc = source_stat_inc
        self.sink = sink
        self.gpu_adapter = gpu_adapter
        self._next_frame_id = 0
        self._fatal_error_reported = False

    def _handle_sink_error(self, exc: Exception) -> None:
        """Fail the child closed when the presenter can no longer be trusted."""
        self.source_stat_inc(
            "runtime_output_sink_errors",
            last_error=f"{type(exc).__name__}: {exc}",
        )
        context = getattr(self.sink, "vulkan", None)
        fatal = bool(getattr(context, "device_lost", False)) or is_vulkan_device_lost_error(exc)
        if fatal:
            # A failed release may still own a producer-ring slot. Continuing
            # would allow the producer to overwrite an image the GPU may be
            # waiting on. Stop the child and let the normal teardown path
            # destroy the adapter/context in order.
            if not self._fatal_error_reported:
                self._fatal_error_reported = True
                print(
                    "[VulkanOutput] presenter synchronization failed; "
                    "stopping XR child safely: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            self.shutdown_event.set()

    def _take_latest(self):
        try:
            item = self.runtime_q.get(timeout=0.05)
        except queue.Empty:
            return None
        if bool(getattr(self.runtime_q, "_d2s_ordered", False)):
            return item
        while True:
            try:
                item = self.runtime_q.get_nowait()
                self.source_stat_inc("runtime_output_overwrite")
            except queue.Empty:
                return item

    def _to_output_frame(self, item):
        try:
            runtime_result, capture_timestamp = item
        except (TypeError, ValueError):
            self.source_stat_inc("runtime_output_invalid_item")
            return None

        left_eye = getattr(runtime_result, "left_eye", None)
        right_eye = getattr(runtime_result, "right_eye", None)
        if not isinstance(left_eye, VulkanImageResource) or not isinstance(
            right_eye, VulkanImageResource
        ):
            if self.gpu_adapter is None:
                # Torch/CPU results wait for a vendor interop importer; never copy them here.
                self.source_stat_inc("runtime_output_waiting_for_vulkan_importer")
                return None
            try:
                frame = self.gpu_adapter.convert(
                    runtime_result,
                    frame_id=self._next_frame_id,
                    timestamp=float(capture_timestamp or time.monotonic()),
                )
            except Exception as exc:
                self.source_stat_inc(
                    "runtime_output_import_errors",
                    last_error=f"{type(exc).__name__}: {exc}",
                )
                return None
            self._next_frame_id += 1
            self.source_stat_inc("runtime_output_gpu_copies")
            return frame

        frame = VulkanStereoOutputFrame(
            frame_id=self._next_frame_id,
            timestamp=float(capture_timestamp or time.monotonic()),
            left_eye=left_eye,
            right_eye=right_eye,
            sbs=getattr(runtime_result, "sbs", None),
            ready_timeline=getattr(runtime_result, "ready_timeline", None),
            metadata=dict(getattr(runtime_result, "debug_info", None) or {}),
            color_space=str(
                (getattr(runtime_result, "debug_info", None) or {}).get(
                    "output_color_space", "srgb"
                )
            ),
            image_origin=str(
                (getattr(runtime_result, "debug_info", None) or {}).get(
                    "output_image_origin", "top_left"
                )
            ),
        )
        self._next_frame_id += 1
        return frame

    def _release_frame(self, frame) -> None:
        release = getattr(self.gpu_adapter, "release_frame", None)
        if callable(release):
            release(frame.frame_id)

    def _safe_release_frame(self, frame) -> None:
        try:
            self._release_frame(frame)
        except Exception as exc:
            self._handle_sink_error(exc)

    def run(self) -> None:
        while not self.shutdown_event.is_set():
            if self.sink is not None:
                ready = getattr(self.sink, "output_ready", None)
                if ready is None:
                    ready = getattr(self.sink, "initialized", True)
                if not bool(ready):
                    self.source_stat_inc("runtime_output_waiting_for_openxr")
                    self.shutdown_event.wait(0.01)
                    continue
            item = self._take_latest()
            if item is None:
                continue
            try:
                runtime_result, capture_timestamp = item
            except (TypeError, ValueError):
                self.source_stat_inc("runtime_output_invalid_item")
                continue
            submit_runtime_result = getattr(self.sink, "submit_runtime_result", None)
            if callable(submit_runtime_result):
                try:
                    submit_runtime_result(
                        runtime_result,
                        float(capture_timestamp or time.monotonic()),
                    )
                except Exception as exc:
                    self._handle_sink_error(exc)
                    continue
                self.source_stat_inc("runtime_output_frames")
                continue
            frame = self._to_output_frame(item)
            if frame is None:
                continue
            if self.sink is None:
                self.source_stat_inc("runtime_output_no_sink")
                self._safe_release_frame(frame)
                continue
            try:
                self.sink.submit_output(frame)
            except Exception as exc:
                self._safe_release_frame(frame)
                self.source_stat_inc(
                    "runtime_output_submit_errors",
                    last_error=f"{type(exc).__name__}: {exc}",
                )
            else:
                self.source_stat_inc("runtime_output_frames")

    def close(self) -> None:
        close = getattr(self.gpu_adapter, "close", None)
        if callable(close):
            close()


register_gpu_producer_adapter("cuda", CudaVulkanOutputAdapter)
register_gpu_producer_adapter("nvidia", CudaVulkanOutputAdapter)
register_gpu_producer_adapter("rocm", RocmVulkanOutputAdapter)
register_gpu_producer_adapter("hip", RocmVulkanOutputAdapter)
register_gpu_producer_adapter("vulkan", VulkanHostOutputAdapter)
register_gpu_producer_adapter("vulkan_host", VulkanHostOutputAdapter)
register_gpu_producer_adapter("vulkan_zero_copy", VulkanZeroCopyOutputAdapter)
