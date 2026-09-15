"""GPU-only bridge from Vulkan eye images to Intel D3D11 final-SBS input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stereo_runtime.providers.intel.d3d11_sbs_surface import D3D11SbsSurface


@dataclass(frozen=True, slots=True)
class IntelVulkanSbsFrame:
    """Borrowed final BGRA texture contract consumed by the native encoder."""

    texture: int
    device: int
    width: int
    height: int
    adapter_luid: int
    gpu_to_cpu: bool = False
    # Strict end-to-end zero-copy remains false until a real Intel target
    # validates the complete Vulkan -> D3D11 -> VideoProcessor -> oneVPL path.
    zero_copy: bool = False
    # Vulkan writes the imported BGRA image, then VideoProcessor converts it
    # to the NV12 surface consumed by oneVPL: one GPU conversion is real.
    gpu_copy_count: int = 1
    producer_ready: bool = False
    ready_timeline: int = 0


class IntelVulkanSbsComposer:
    """Compose Vulkan left/right images into a D3D11-owned shared texture.

    The current bridge uses two Vulkan blits because the stereo shader produces
    separate eye images. There is no CPU readback, but the two GPU blits mean
    this is explicitly a GPU-only copy path, not strict zero-copy yet.
    """

    def __init__(self, context: Any, eye_width: int, eye_height: int) -> None:
        self.context = context
        self.eye_width = int(eye_width)
        self.eye_height = int(eye_height)
        if self.eye_width < 1 or self.eye_height < 1:
            raise ValueError("Intel Vulkan SBS eye dimensions must be positive")
        adapter_luid = int(getattr(context.device_info, "adapter_luid", 0))
        if adapter_luid == 0:
            raise RuntimeError("Vulkan Adapter LUID is unavailable for Intel SBS composition")
        self.adapter_luid = adapter_luid
        self.surface = D3D11SbsSurface(
            width=self.eye_width * 2,
            height=self.eye_height,
            adapter_luid=adapter_luid,
        )
        if self.surface.adapter_luid != adapter_luid:
            self.surface.close()
            raise RuntimeError("D3D11 SBS surface Adapter LUID does not match Vulkan")
        self.imported = None
        try:
            self.imported = self.surface.import_bgra_into_vulkan(context)
        except Exception:
            self.surface.close()
            raise

    def compose(
        self,
        left_resource: Any,
        right_resource: Any,
        *,
        ready_timeline: int | None = None,
    ) -> IntelVulkanSbsFrame:
        if self.imported is None or self.imported.resource is None:
            raise RuntimeError("Intel Vulkan SBS imported destination is unavailable")
        if getattr(left_resource, "context", None) is not self.context or getattr(
            right_resource, "context", None
        ) is not self.context:
            raise RuntimeError("Vulkan SBS eye resources belong to a different context")
        timeline = self.context.compose_sbs_images(
            left_resource,
            right_resource,
            self.imported.resource,
            wait_for_timeline=ready_timeline,
        )
        # D3D11 has no Vulkan semaphore object in this bridge yet. Waiting for
        # the submitted Vulkan timeline is still GPU synchronization; it does
        # not map or copy image data through the CPU. The encoder consumer will
        # issue the D3D11 VideoProcessor conversion exactly once.
        self.context.wait_for_timeline(timeline)
        return IntelVulkanSbsFrame(
            texture=self.surface.bgra_texture,
            device=self.surface.device,
            width=self.surface.width,
            height=self.surface.height,
            adapter_luid=self.adapter_luid,
            producer_ready=True,
            ready_timeline=int(timeline),
        )

    def frame_after_vulkan_submit(self, timeline: int) -> IntelVulkanSbsFrame:
        """Return the D3D11 surface after a packed shader write completes."""
        if self.imported is None or self.imported.resource is None:
            raise RuntimeError("Intel Vulkan SBS imported destination is unavailable")
        self.context.wait_for_timeline(int(timeline))
        return IntelVulkanSbsFrame(
            texture=self.surface.bgra_texture,
            device=self.surface.device,
            width=self.surface.width,
            height=self.surface.height,
            adapter_luid=self.adapter_luid,
            producer_ready=True,
            ready_timeline=int(timeline),
            gpu_copy_count=1,
        )

    def close(self) -> None:
        if self.imported is not None:
            self.imported.close()
            self.imported = None
        if self.surface is not None:
            self.surface.close()
            self.surface = None


class IntelVulkanSbsRuntimeBridge:
    """Run a deferred Vulkan stereo request for the Intel network sink.

    Network mode has no OpenXR presenter that can provide image slots. This
    bridge owns a Vulkan context and dispatches the existing layered image pass
    directly into the D3D11-imported SBS image. The bridge is deliberately
    opt-in and does not perform an intermediate eye-image blit.
    """

    def __init__(self, eye_width: int, eye_height: int, *, ring_size: int = 3) -> None:
        from viewer.vulkan_context import VulkanContext, VulkanContextConfig
        from stereo_runtime.vulkan_backend import VulkanStereoImageComputeBackend

        self.eye_width = int(eye_width)
        self.eye_height = int(eye_height)
        self.ring_size = max(2, int(ring_size))
        self.context = VulkanContext.create(
            VulkanContextConfig(frame_context_count=self.ring_size)
        )
        self.backend = VulkanStereoImageComputeBackend(self.context)
        self.composer = IntelVulkanSbsComposer(
            self.context, self.eye_width, self.eye_height
        )
        destination = getattr(self.composer.imported, "resource", None)
        if destination is None:
            raise RuntimeError("Intel Vulkan SBS imported destination is unavailable")
        self.context.prepare_external_image_for_producer(destination)
        self._frame_id = 0

    def submit(self, request: Any) -> IntelVulkanSbsFrame:
        rgb = getattr(request, "rgb", None)
        depth = getattr(request, "depth", None)
        shift = getattr(request, "shift", None)
        params = getattr(request, "params", None)
        if rgb is None or depth is None or shift is None or params is None:
            raise ValueError("Vulkan Intel SBS request is incomplete")
        height = int(getattr(rgb, "shape", (0, 0, 0, 0))[-2])
        width = int(getattr(rgb, "shape", (0, 0, 0, 0))[-1])
        if (width, height) != (self.eye_width, self.eye_height):
            raise RuntimeError(
                "Vulkan Intel SBS request dimensions changed: "
                f"expected={self.eye_width}x{self.eye_height} actual={width}x{height}"
            )
        destination = getattr(self.composer.imported, "resource", None)
        if destination is None:
            raise RuntimeError("Vulkan Intel SBS imported destination is unavailable")
        timeline, _debug = self.backend.submit_to_images(
            rgb,
            depth,
            shift,
            destination,
            destination,
            params=params,
            packed_output=True,
        )
        frame = self.composer.frame_after_vulkan_submit(timeline)
        self._frame_id += 1
        return frame

    def close(self) -> None:
        backend = getattr(self, "backend", None)
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
            self.backend = None
        composer = getattr(self, "composer", None)
        if composer is not None:
            try:
                composer.close()
            except Exception:
                pass
            self.composer = None
        context = getattr(self, "context", None)
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
            self.context = None


__all__ = [
    "IntelVulkanSbsComposer",
    "IntelVulkanSbsFrame",
    "IntelVulkanSbsRuntimeBridge",
]
