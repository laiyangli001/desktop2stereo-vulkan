from __future__ import annotations

import torch
import triton
import triton.language as tl

from .triton_runtime import triton_runtime_available


@triton.jit
def _warp_composite2_kernel(
    rgb,
    depth,
    base_shift,
    left,
    right,
    total: tl.constexpr,
    width: tl.constexpr,
    height: tl.constexpr,
    pixels: tl.constexpr,
    softness: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width
    channel = offsets // pixels

    depth_value = tl.load(depth + pixel, mask=active, other=0.0)
    w0_raw = tl.exp(-((depth_value - 0.0) * (depth_value - 0.0)) / softness)
    w1_raw = tl.exp(-((depth_value - 1.0) * (depth_value - 1.0)) / softness)
    wsum = w0_raw + w1_raw
    w0 = w0_raw / wsum
    w1 = w1_raw / wsum

    shift = tl.load(base_shift + pixel, mask=active, other=0.0)
    left_value = _sample_two_layers(rgb, channel, y, x, shift, 0.875, 1.0, width, pixels, active, w0, w1)
    right_value = _sample_two_layers(rgb, channel, y, x, shift, -0.875, -1.0, width, pixels, active, w0, w1)
    tl.store(left + offsets, left_value, mask=active)
    tl.store(right + offsets, right_value, mask=active)


@triton.jit
def _warp_composite2_rgba_u8_kernel(
    rgb,
    depth,
    base_shift,
    left,
    right,
    pixels: tl.constexpr,
    width: tl.constexpr,
    softness: tl.constexpr,
    block: tl.constexpr,
):
    pixel = tl.program_id(0) * block + tl.arange(0, block)
    active = pixel < pixels
    y = pixel // width
    x = pixel - y * width

    depth_value = tl.load(depth + pixel, mask=active, other=0.0)
    w0_raw = tl.exp(-(depth_value * depth_value) / softness)
    depth_from_one = depth_value - 1.0
    w1_raw = tl.exp(-(depth_from_one * depth_from_one) / softness)
    weight_sum = w0_raw + w1_raw
    w0 = w0_raw / weight_sum
    w1 = w1_raw / weight_sum
    shift = tl.load(base_shift + pixel, mask=active, other=0.0)

    left_r = _sample_two_layers(rgb, 0, y, x, shift, 0.875, 1.0, width, pixels, active, w0, w1)
    left_g = _sample_two_layers(rgb, 1, y, x, shift, 0.875, 1.0, width, pixels, active, w0, w1)
    left_b = _sample_two_layers(rgb, 2, y, x, shift, 0.875, 1.0, width, pixels, active, w0, w1)
    right_r = _sample_two_layers(rgb, 0, y, x, shift, -0.875, -1.0, width, pixels, active, w0, w1)
    right_g = _sample_two_layers(rgb, 1, y, x, shift, -0.875, -1.0, width, pixels, active, w0, w1)
    right_b = _sample_two_layers(rgb, 2, y, x, shift, -0.875, -1.0, width, pixels, active, w0, w1)

    output_offset = pixel * 4
    tl.store(left + output_offset, _rgba_u8(left_r), mask=active)
    tl.store(left + output_offset + 1, _rgba_u8(left_g), mask=active)
    tl.store(left + output_offset + 2, _rgba_u8(left_b), mask=active)
    tl.store(left + output_offset + 3, 255, mask=active)
    tl.store(right + output_offset, _rgba_u8(right_r), mask=active)
    tl.store(right + output_offset + 1, _rgba_u8(right_g), mask=active)
    tl.store(right + output_offset + 2, _rgba_u8(right_b), mask=active)
    tl.store(right + output_offset + 3, 255, mask=active)


@triton.jit
def _warp_composite2_full_sbs_kernel(
    rgb,
    depth,
    base_shift,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    out_width: tl.constexpr,
    pixels: tl.constexpr,
    softness: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % (pixels * 2)
    y = pixel // out_width
    x = pixel - y * out_width
    channel = offsets // (pixels * 2)

    use_left = x < width
    source_x = tl.where(use_left, x, x - width)
    source_pixel = y * width + source_x
    depth_value = tl.load(depth + source_pixel, mask=active, other=0.0)
    w0_raw = tl.exp(-((depth_value - 0.0) * (depth_value - 0.0)) / softness)
    w1_raw = tl.exp(-((depth_value - 1.0) * (depth_value - 1.0)) / softness)
    wsum = w0_raw + w1_raw
    w0 = w0_raw / wsum
    w1 = w1_raw / wsum
    shift = tl.load(base_shift + source_pixel, mask=active, other=0.0)
    left_value = _sample_two_layers(rgb, channel, y, source_x, shift, 0.875, 1.0, width, pixels, active, w0, w1)
    right_value = _sample_two_layers(rgb, channel, y, source_x, shift, -0.875, -1.0, width, pixels, active, w0, w1)
    tl.store(out + offsets, tl.where(use_left, left_value, right_value), mask=active)


@triton.jit
def _sample_composite_at(
    rgb,
    depth,
    base_shift,
    channel,
    y,
    x,
    scale0,
    scale1,
    width: tl.constexpr,
    pixels: tl.constexpr,
    softness: tl.constexpr,
    active,
):
    source_pixel = y * width + x
    depth_value = tl.load(depth + source_pixel, mask=active, other=0.0)
    w0_raw = tl.exp(-(depth_value * depth_value) / softness)
    depth_from_one = depth_value - 1.0
    w1_raw = tl.exp(-(depth_from_one * depth_from_one) / softness)
    weight_sum = w0_raw + w1_raw
    shift = tl.load(base_shift + source_pixel, mask=active, other=0.0)
    return _sample_two_layers(
        rgb,
        channel,
        y,
        x,
        shift,
        scale0,
        scale1,
        width,
        pixels,
        active,
        w0_raw / weight_sum,
        w1_raw / weight_sum,
    )


@triton.jit
def _warp_composite2_half_sbs_kernel(
    rgb,
    depth,
    base_shift,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    half_width: tl.constexpr,
    pixels: tl.constexpr,
    softness: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width
    channel = offsets // pixels

    use_left = x < half_width
    source_x = tl.where(use_left, x, x - half_width)
    x0 = source_x * 2
    xm1 = tl.maximum(x0 - 1, 0)
    x1 = x0 + 1
    x2 = tl.minimum(x0 + 2, width - 1)

    left_m1 = _sample_composite_at(rgb, depth, base_shift, channel, y, xm1, 0.875, 1.0, width, pixels, softness, active)
    left0 = _sample_composite_at(rgb, depth, base_shift, channel, y, x0, 0.875, 1.0, width, pixels, softness, active)
    left1 = _sample_composite_at(rgb, depth, base_shift, channel, y, x1, 0.875, 1.0, width, pixels, softness, active)
    left2 = _sample_composite_at(rgb, depth, base_shift, channel, y, x2, 0.875, 1.0, width, pixels, softness, active)
    right_m1 = _sample_composite_at(rgb, depth, base_shift, channel, y, xm1, -0.875, -1.0, width, pixels, softness, active)
    right0 = _sample_composite_at(rgb, depth, base_shift, channel, y, x0, -0.875, -1.0, width, pixels, softness, active)
    right1 = _sample_composite_at(rgb, depth, base_shift, channel, y, x1, -0.875, -1.0, width, pixels, softness, active)
    right2 = _sample_composite_at(rgb, depth, base_shift, channel, y, x2, -0.875, -1.0, width, pixels, softness, active)
    left_value = (-left_m1 + 9.0 * left0 + 9.0 * left1 - left2) * 0.0625
    right_value = (-right_m1 + 9.0 * right0 + 9.0 * right1 - right2) * 0.0625
    value = tl.where(use_left, left_value, right_value)
    tl.store(out + offsets, value, mask=active)


@triton.jit
def _rgba_u8(value):
    return (tl.minimum(tl.maximum(value, 0.0), 1.0) * 255.0).to(tl.uint8)


@triton.jit
def _sample_two_layers(rgb, channel, y, x, shift, scale0, scale1, width: tl.constexpr, pixels: tl.constexpr, active, w0, w1):
    x0 = x + shift * scale0
    x1 = x + shift * scale1
    v0 = _sample_border_linear(rgb, channel, y, x0, width, pixels, active)
    v1 = _sample_border_linear(rgb, channel, y, x1, width, pixels, active)
    return v0 * w0 + v1 * w1


@triton.jit
def _sample_border_linear(rgb, channel, y, sample_x, width: tl.constexpr, pixels: tl.constexpr, active):
    # Reflection mirrors the source at both boundaries and turns a displaced
    # edge into a visible second edge.  Border sampling extends the nearest
    # real pixel without introducing that mirrored geometry.
    x_clamped = tl.minimum(tl.maximum(sample_x, 0.0), width - 1.0)
    x0_float = tl.floor(x_clamped)
    x0 = x0_float.to(tl.int64)
    x1 = tl.minimum(x0 + 1, width - 1)
    frac = x_clamped - x0_float
    base = channel * pixels + y * width
    v0 = tl.load(rgb + base + x0, mask=active, other=0.0)
    v1 = tl.load(rgb + base + x1, mask=active, other=0.0)
    return v0 + (v1 - v0) * frac


def can_use_triton_warp_composite2(rgb: torch.Tensor, depth: torch.Tensor, base_shift: torch.Tensor, *, layers: int, symmetric: bool) -> bool:
    return (
        layers == 2
        and symmetric
        and triton_runtime_available(rgb.device)
        and rgb.dtype == torch.float32
        and depth.dtype == torch.float32
        and base_shift.dtype == torch.float32
        and rgb.ndim == 4
        and depth.ndim == 4
        and base_shift.ndim == 4
        and rgb.shape[0] == 1
        and rgb.shape[1] == 3
        and depth.shape[0] == 1
        and depth.shape[1] == 1
        and base_shift.shape == depth.shape
        and rgb.shape[-2:] == depth.shape[-2:]
    )


def warp_composite2(rgb: torch.Tensor, depth: torch.Tensor, base_shift: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = rgb.contiguous()
    depth = depth.contiguous()
    base_shift = base_shift.contiguous()
    left = torch.empty_like(rgb)
    right = torch.empty_like(rgb)
    _, _, height, width = rgb.shape
    pixels = height * width
    total = rgb.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _warp_composite2_kernel[grid](rgb, depth, base_shift, left, right, total, width, height, pixels, 0.08, block)
    return left, right


def warp_composite2_half_sbs(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    base_shift: torch.Tensor,
) -> torch.Tensor:
    rgb = rgb.contiguous()
    depth = depth.contiguous()
    base_shift = base_shift.contiguous()
    _, channels, height, width = rgb.shape
    if width % 2:
        raise ValueError("half-SBS direct output requires an even width")
    out = torch.empty_like(rgb)
    pixels = height * width
    total = out.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _warp_composite2_half_sbs_kernel[
        grid
    ](rgb, depth, base_shift, out, total, width, width // 2, pixels, 0.08, block)
    return out


def warp_composite2_full_sbs(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    base_shift: torch.Tensor,
) -> torch.Tensor:
    rgb = rgb.contiguous()
    depth = depth.contiguous()
    base_shift = base_shift.contiguous()
    _, channels, height, width = rgb.shape
    out = torch.empty((1, channels, height, width * 2), device=rgb.device, dtype=rgb.dtype)
    pixels = height * width
    total = out.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _warp_composite2_full_sbs_kernel[
        grid
    ](rgb, depth, base_shift, out, total, width, width * 2, pixels, 0.08, block)
    return out


def warp_composite2_rgba_u8(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    base_shift: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = rgb.contiguous()
    depth = depth.contiguous()
    base_shift = base_shift.contiguous()
    _, _, height, width = rgb.shape
    left = torch.empty((height, width, 4), device=rgb.device, dtype=torch.uint8)
    right = torch.empty_like(left)
    pixels = height * width
    block = 256
    grid = (triton.cdiv(pixels, block),)
    _warp_composite2_rgba_u8_kernel[
        grid
    ](rgb, depth, base_shift, left, right, pixels, width, 0.08, block)
    return left, right
