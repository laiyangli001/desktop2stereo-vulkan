from __future__ import annotations

from dataclasses import dataclass


# These are policy buckets for filtering, not forced image dimensions. The
# source frame keeps its real aspect ratio and actual width/height.
STANDARD_INPUT_RESOLUTIONS = {
    1: (1920, 1080),
    2: (2560, 1440),
    4: (3840, 2160),
}
INPUT_TO_HEADSET_TIER = {1: 2, 2: 4, 4: 8}
HEADSET_TIERS = (2, 4, 8)


@dataclass(frozen=True, slots=True)
class ScreenSamplingPlan:
    source_width: int
    source_height: int
    input_tier_k: int
    headset_tier_k: int
    recommended_headset_tier_k: int
    effective_tier_k: int
    filter_scale: float
    upscale_scale: float
    mode: str
    quality_width: int | None = None
    quality_height: int | None = None

    @property
    def source_texel_size(self) -> tuple[float, float]:
        return (1.0 / self.source_width, 1.0 / self.source_height)

    @property
    def matrix_label(self) -> str:
        return (
            f"input_{self.input_tier_k}k->headset_{self.recommended_headset_tier_k}k "
            f"selected={self.headset_tier_k}k effective={self.effective_tier_k}k"
        )

    @property
    def quality_size(self) -> tuple[int, int] | None:
        if self.quality_width is None or self.quality_height is None:
            return None
        return self.quality_width, self.quality_height


@dataclass(frozen=True, slots=True)
class OutputSamplingPlan:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    mode: str
    target_kind: str

    @property
    def scale(self) -> float:
        return self.target_width / float(self.source_width)


def build_output_sampling_plan(
    source_width: int,
    source_height: int,
    *,
    headset_tier_k: int,
) -> OutputSamplingPlan:
    """Resolve the shared headset target for every output consumer."""
    width = int(source_width)
    height = int(source_height)
    if width <= 0 or height <= 0:
        raise ValueError("source resolution must be positive")
    plan = build_screen_sampling_plan(width, height, int(headset_tier_k))
    scale = plan.upscale_scale if plan.mode == "upscale_easu" else 1.0 / plan.filter_scale
    out_width = max(2, int(round(width * scale)) // 2 * 2)
    out_height = max(2, int(round(height * scale)) // 2 * 2)
    return OutputSamplingPlan(
        source_width=width,
        source_height=height,
        target_width=out_width,
        target_height=out_height,
        mode=plan.mode,
        target_kind="headset",
    )


def classify_input_resolution(width: int, height: int) -> int:
    """Bucket input for the final screen-quality route.

    Only a complete 4K source may use the native-MIP route. The GUI 3K
    tier is a reduced 4K frame, so it remains in the upscale bucket and uses
    the same EASU/RCAS chain as the 1K/2K inputs.
    """
    width = int(width)
    height = int(height)
    long_edge = max(width, height)
    short_edge = min(width, height)
    if long_edge <= 0 or short_edge <= 0:
        raise ValueError("input resolution must be positive")
    if long_edge <= 1920:
        return 1
    full_4k = (
        (long_edge >= 3840 and short_edge >= 1600)
        or (
            width * height >= 3840 * 2160 * 0.85
            and long_edge >= 3200
            and short_edge >= 1600
        )
    )
    return 4 if full_4k else 2


def build_screen_sampling_plan(
    source_width: int,
    source_height: int,
    headset_tier_k: int,
) -> ScreenSamplingPlan:
    """Build the input/headset matrix without changing the source geometry.

    The selected headset tier is capped by the 2x input-to-headset policy. If
    a lower-tier headset receives a higher-tier source, the filter footprint is
    widened only enough to prefilter that downsample; matched paths stay at the
    source texel footprint and keep the existing sharp path.
    """
    width = int(source_width)
    height = int(source_height)
    if width <= 0 or height <= 0:
        raise ValueError("source resolution must be positive")
    input_tier = classify_input_resolution(width, height)
    headset_tier = int(headset_tier_k)
    if headset_tier not in HEADSET_TIERS:
        raise ValueError(f"unsupported headset resolution tier: {headset_tier_k}")
    recommended = INPUT_TO_HEADSET_TIER[input_tier]
    effective = min(headset_tier, recommended)
    filter_scale = max(1.0, input_tier / float(effective))
    upscale_scale = max(1.0, effective / float(input_tier))
    if upscale_scale > 1.0:
        mode = "upscale_easu"
    elif filter_scale > 1.0:
        mode = "downsample_lanczos_rcas"
    else:
        mode = "native_mip"
    return ScreenSamplingPlan(
        source_width=width,
        source_height=height,
        input_tier_k=input_tier,
        headset_tier_k=headset_tier,
        recommended_headset_tier_k=recommended,
        effective_tier_k=effective,
        filter_scale=filter_scale,
        upscale_scale=upscale_scale,
        mode=mode,
    )


def build_projection_screen_sampling_plan(
    source_width: int,
    source_height: int,
    footprint_width: float,
    footprint_height: float,
    *,
    oversample: float = 1.0,
) -> ScreenSamplingPlan:
    """Choose one GPU quality image from the visible OpenXR screen footprint.

    Unlike the legacy tier matrix, this uses the actual projected screen size
    in the eye swapchain.  The result remains source-aspect-correct and is
    deliberately quantized to even dimensions for Vulkan image allocation.
    """
    width = int(source_width)
    height = int(source_height)
    if width <= 0 or height <= 0:
        raise ValueError("source resolution must be positive")
    footprint_width = float(footprint_width)
    footprint_height = float(footprint_height)
    if footprint_width <= 0.0 or footprint_height <= 0.0:
        raise ValueError("screen footprint must be positive")
    oversample = max(1.0, float(oversample))
    aspect = width / float(height)
    quality_width = max(16, int(round(footprint_width * oversample)) & ~1)
    # Width is authoritative so a curved/oblique projection cannot introduce
    # a second aspect-ratio conversion in the quality pass.
    quality_height = max(16, int(round(quality_width / aspect)) & ~1)
    scale = quality_width / float(width)
    input_tier = classify_input_resolution(width, height)
    if scale > 1.01:
        mode = "upscale_easu"
        upscale_scale = scale
        filter_scale = 1.0
    elif scale < 0.99:
        mode = "downsample_lanczos_rcas"
        upscale_scale = 1.0
        filter_scale = 1.0 / max(scale, 1e-6)
    else:
        mode = "native_mip"
        upscale_scale = 1.0
        filter_scale = 1.0
        quality_width = width
        quality_height = height
    return ScreenSamplingPlan(
        source_width=width,
        source_height=height,
        input_tier_k=input_tier,
        headset_tier_k=0,
        recommended_headset_tier_k=0,
        effective_tier_k=0,
        filter_scale=filter_scale,
        upscale_scale=upscale_scale,
        mode=mode,
        quality_width=quality_width,
        quality_height=quality_height,
    )
