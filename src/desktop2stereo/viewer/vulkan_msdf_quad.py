from __future__ import annotations

"""Vulkan MSDF rasterization into an OpenXR Quad-layer image."""

from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np

from viewer.vulkan_compute_pipeline import VulkanComputePipeline
from viewer.vulkan_descriptors import (
    DescriptorBinding,
    DescriptorBudget,
    VulkanDescriptorArena,
    VulkanStorageBuffer,
    VulkanStorageImage,
)
from viewer.vulkan_resources import VulkanHostImage
from viewer.vulkan_context import ImageState


@dataclass(frozen=True, slots=True)
class VulkanMsdfQuadRequest:
    width: int
    height: int
    runs: tuple[dict[str, Any], ...]
    background: tuple[int, int, int, int] = (32, 32, 36, 210)
    radius: float = 12.0

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.height), int(self.width), 4)


class VulkanMsdfQuadRenderer:
    """Keep the MSDF atlas on the GPU and render changed OSDs with compute."""

    _WORKGROUP_SIZE = 8
    # The full Chinese operation guide can contain more than 800 glyphs.
    # Keep one dispatch for the complete panel instead of splitting text
    # across multiple Quad images. 8192 covers the largest bilingual guide
    # panels on the GPU path (399 KB glyph buffer, dispatch-bounded only).
    _MAX_GLYPHS = 8192
    _GLYPH_STRIDE = 48
    _PUSH_CONSTANT_SIZE = 64

    def __init__(self, context: Any, atlas: Any) -> None:
        self.context = context
        self.vk = context.vk
        self.atlas = atlas
        self.atlas_image: VulkanStorageImage | None = None
        self.glyph_buffer: VulkanStorageBuffer | None = None
        self.pipeline: VulkanComputePipeline | None = None
        self.descriptor_arena: VulkanDescriptorArena | None = None
        self.descriptor_set = None
        self.outputs: dict[tuple[int, int], VulkanStorageImage] = {}
        self._last_use_timeline = 0
        self._create()

    @property
    def available(self) -> bool:
        return self.pipeline is not None and self.atlas_image is not None

    def supports_destination_format(self, destination_format: int) -> bool:
        """Return whether the fixed RGBA8 intermediate can copy to the target."""
        return int(destination_format) in {
            int(self.vk.VK_FORMAT_R8G8B8A8_UNORM),
            int(self.vk.VK_FORMAT_R8G8B8A8_SRGB),
        }

    def _create(self) -> None:
        shader_path = (
            Path(__file__).resolve().parents[1] / "shaders" / "d2s_msdf_quad.spv"
        )
        try:
            import vulkan as vk

            self._create_atlas_image(vk)
            self.glyph_buffer = VulkanStorageBuffer(
                self.context, self._MAX_GLYPHS * self._GLYPH_STRIDE
            )
            self.pipeline = VulkanComputePipeline(
                self.context,
                shader_path,
                descriptor_bindings=[
                    DescriptorBinding(0, vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
                    DescriptorBinding(1, vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
                    DescriptorBinding(2, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
                ],
                push_constants_size=self._PUSH_CONSTANT_SIZE,
            )
            self.descriptor_arena = VulkanDescriptorArena(
                self.context,
                DescriptorBudget(
                    max_sets=1,
                    storage_buffers_per_set=1,
                    storage_images_per_set=2,
                ),
            )
            self.descriptor_set = self.descriptor_arena.allocate(
                self.pipeline.descriptor_set_layout
            )
            self.descriptor_arena.update_storage_image(
                self.descriptor_set, 0, self.atlas_image
            )
            self.descriptor_arena.update_storage_buffer(
                self.descriptor_set, 2, self.glyph_buffer
            )
        except Exception:
            self.close()
            raise

    def has_output_for(self, width: int, height: int) -> bool:
        """Whether a render-target image already exists for this canvas size."""
        return (int(width), int(height)) in self.outputs

    def prewarm_outputs(self, sizes) -> None:
        """Allocate the render-target images before the XR frame loop starts.

        The first mid-frame allocation+layout-transition submission stalled on
        VDXR/AMD (vkWaitSemaphores VkTimeout); the identical code completes
        instantly at session start, so callers pass the overlay canvas sizes
        up front and mid-frame renders are then pure cache hits.
        """
        for width, height in sizes:
            try:
                self._ensure_output(int(width), int(height))
            except Exception:
                pass

    def _create_atlas_image(self, vk: Any) -> None:
        pages = [self.atlas.page_rgba(page) for page in range(len(self.atlas.pages))]
        if not pages:
            raise ValueError("MSDF atlas has no pages")
        page_height, page_width, channels = pages[0].shape
        if channels != 4 or any(page.shape != pages[0].shape for page in pages):
            raise ValueError("MSDF atlas pages must have identical RGBA dimensions")
        combined = np.ascontiguousarray(np.concatenate(pages, axis=1), dtype=np.uint8)
        self.atlas_image = VulkanStorageImage(
            self.context,
            width=int(page_width * len(pages)),
            height=int(page_height),
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            usage=vk.VK_IMAGE_USAGE_STORAGE_BIT | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            queue_role="graphics",
        )
        staging = VulkanHostImage(
            self.context,
            int(combined.shape[1]),
            int(combined.shape[0]),
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            label="msdf-atlas-upload",
        )
        try:
            staging.upload(combined)
            copy_timeline = self.context.copy_image(staging.resource, self.atlas_image)
            transition_timeline = self.atlas_image.transition_to_general(
                role="graphics", dst_access_mask=vk.VK_ACCESS_SHADER_READ_BIT
            )
            self.context.wait_for_timeline(max(copy_timeline, transition_timeline))
        finally:
            staging.close()

    def _ensure_output(self, width: int, height: int) -> VulkanStorageImage:
        import vulkan as vk

        key = (int(width), int(height))
        output = self.outputs.get(key)
        if output is not None:
            return output
        output = VulkanStorageImage(
            self.context,
            width=int(width),
            height=int(height),
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            usage=vk.VK_IMAGE_USAGE_STORAGE_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            queue_role="graphics",
        )
        transition_timeline = output.transition_to_general(role="graphics")
        self.context.wait_for_timeline(transition_timeline)
        self.outputs[key] = output
        return output

    def _pack_glyphs(self, request: VulkanMsdfQuadRequest) -> tuple[bytes, int]:
        atlas_width = float(self.atlas.page_width * len(self.atlas.pages))
        records = []
        for run in request.runs:
            instances = self.atlas.layout(
                str(run.get("text", "")),
                origin=(float(run.get("x", 0.0)), float(run.get("y", 0.0))),
                scale=float(run.get("scale", 1.0)),
            )
            color = np.asarray(run.get("color", (255, 255, 255, 255)), dtype=np.float32)
            if color.shape != (4,):
                raise ValueError("MSDF run color must contain four components")
            for instance in instances:
                page_offset = float(instance.page * self.atlas.page_width)
                records.append(
                    struct.pack(
                        "<12f",
                        float(instance.position[0]),
                        float(instance.position[1]),
                        float(instance.size[0]),
                        float(instance.size[1]),
                        (page_offset + float(instance.uv_min[0]) * self.atlas.page_width)
                        / atlas_width,
                        float(instance.uv_min[1]),
                        (page_offset + float(instance.uv_max[0]) * self.atlas.page_width)
                        / atlas_width,
                        float(instance.uv_max[1]),
                        *(float(component) / 255.0 for component in color),
                    )
                )
        if len(records) > self._MAX_GLYPHS:
            raise ValueError("MSDF Quad glyph capacity exceeded")
        return b"".join(records), len(records)

    def render(
        self,
        request: VulkanMsdfQuadRequest,
        *,
        destination_format: int,
    ) -> VulkanStorageImage:
        if not self.available:
            raise RuntimeError("Vulkan MSDF Quad renderer is unavailable")
        if self._last_use_timeline:
            self.context.wait_for_timeline(self._last_use_timeline)
        glyph_payload, glyph_count = self._pack_glyphs(request)
        assert self.glyph_buffer is not None
        assert self.pipeline is not None
        assert self.descriptor_arena is not None
        assert self.descriptor_set is not None
        output = self._ensure_output(request.width, request.height)
        self.glyph_buffer.write_bytes(glyph_payload)
        self.descriptor_arena.update_storage_image(self.descriptor_set, 1, output)
        import vulkan as vk

        srgb_formats = {
            int(vk.VK_FORMAT_R8G8B8A8_SRGB),
        }
        push_constants = struct.pack(
            "<8I8f",
            int(request.width),
            int(request.height),
            int(self.atlas.page_width * len(self.atlas.pages)),
            int(self.atlas.page_height),
            int(glyph_count),
            1 if int(destination_format) in srgb_formats else 0,
            0,
            0,
            *(float(component) / 255.0 for component in request.background),
            float(request.radius),
            float(self.atlas.distance_range),
            0.0,
            0.0,
        )

        def record(command_buffer: Any) -> None:
            self.pipeline.record_dispatch(
                command_buffer,
                group_count_x=math.ceil(request.width / self._WORKGROUP_SIZE),
                group_count_y=math.ceil(request.height / self._WORKGROUP_SIZE),
                descriptor_set=self.descriptor_set,
                push_constants=push_constants,
            )

        timeline = self.context.submit_on("graphics", record)
        self.context.register_image_state(
            output.image,
            ImageState(
                layout=vk.VK_IMAGE_LAYOUT_GENERAL,
                access_mask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                stage_mask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                queue_family_index=self.context.queue_family_index,
            ),
        )
        self._last_use_timeline = timeline
        return output

    def notify_copy_timeline(self, timeline: int) -> None:
        self._last_use_timeline = max(self._last_use_timeline, int(timeline))

    def close(self) -> None:
        if self.context.device is not None:
            try:
                self.context.wait_idle()
            except Exception:
                pass
        for output in self.outputs.values():
            try:
                output.close()
            except Exception:
                pass
        self.outputs.clear()
        if self.descriptor_arena is not None:
            self.descriptor_arena.close()
        if self.pipeline is not None:
            self.pipeline.close()
        if self.glyph_buffer is not None:
            self.glyph_buffer.close()
        if self.atlas_image is not None:
            self.atlas_image.close()
        self.descriptor_arena = None
        self.pipeline = None
        self.glyph_buffer = None
        self.atlas_image = None
        self.descriptor_set = None
        self._last_use_timeline = 0
