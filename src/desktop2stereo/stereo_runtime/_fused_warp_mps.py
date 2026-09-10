"""Fused warp+SBS-pack Metal kernel (torch.mps.compile_shader) for the
Vulkan local viewer path.

Replaces the torch node chain (permute/cat/contiguous + separate depth
quantize) with ONE Metal kernel that applies the exact Metal-warp math
(gaussian taps, asymmetric shaping, edge falloff) and writes final
Half-SBS RGBA8 bytes. The packer thread then only moves bytes into the
IOSurface stage / host frame — no per-frame SBS synthesis on the MPS
stream, minimal queue coupling.

Kernel validated pixel-exact against a numpy reference
(max abs diff 0 on random inputs); see tools/ notes in docs/33.

Platform: darwin + MPS only. Kill switch: D2S_VK_FUSED_WARP=0.
"""

from __future__ import annotations

import functools
import os
import sys

import numpy as np

WARP_MSL = r"""
#include <metal_stdlib>
using namespace metal;

static inline float sampf(device float* img, uint W, uint H, float u, float v) {
    float x = clamp(u * (float)W - 0.5f, 0.0f, (float)W - 1.001f);
    float y = clamp(v * (float)H - 0.5f, 0.0f, (float)H - 1.001f);
    int x0 = (int)floor(x), y0 = (int)floor(y);
    int x1 = min(x0 + 1, (int)W - 1), y1 = min(y0 + 1, (int)H - 1);
    float fx = x - (float)x0, fy = y - (float)y0;
    float a = img[y0 * W + x0], b_ = img[y0 * W + x1];
    float c = img[y1 * W + x0], d = img[y1 * W + x1];
    return mix(mix(a, b_, fx), mix(c, d, fx), fy);
}

kernel void warp_pack(
    device uchar* out  [[buffer(0)]],
    device float* col  [[buffer(1)]],
    device float* dep  [[buffer(2)]],
    constant float& eyeOffset     [[buffer(3)]],
    constant float& depthStrength [[buffer(4)]],
    constant float& convergence   [[buffer(5)]],
    constant uint&  srcW          [[buffer(6)]],
    constant uint&  srcH          [[buffer(7)]],
    constant uint&  outW          [[buffer(8)]],
    constant uint&  outH          [[buffer(9)]],
    constant float& smoothTexels  [[buffer(10)]],
    uint idx [[thread_position_in_grid]])
{
    // HALF-SBS contract: output frame is outW x outH (runtime input
    // resolution per the NVIDIA-path contract), each eye squeezed into
    // outW/2; full_sbs passes outW = 2*srcW for unsqueezed eyes.
    uint pw = outW / 2u;           // per-eye width
    uint total = outW * outH * 4u;
    if (idx >= total) return;
    uint py = idx / (outW * 4u);
    uint rem = idx % (outW * 4u);
    uint px = rem / 4u;
    uint comp = rem % 4u;
    if (comp == 3u) { out[idx] = 255u; return; }

    bool left = px < pw;
    uint lx = left ? px : (px - pw);       // coordinate within the eye
    float eye = left ? -eyeOffset : eyeOffset;
    float u = ((float)lx + 0.5f) / (float)pw;   // normalized across the eye
    float v = ((float)py + 0.5f) / (float)outH;

    // 3-tap gaussian depth smoothing. Aperture scales with the eye/output
    // ratio so the EFFECTIVE smoothing matches whatever presentation res --
    // native-res warping otherwise amplifies depth estimation noise into
    // visible edge shimmer that the upscaled-540p era never showed.
    float du = smoothTexels / (float)srcW;
    float d0 = sampf(dep, srcW, srcH, u, v);
    float dm = sampf(dep, srcW, srcH, u - du, v);
    float dp_ = sampf(dep, srcW, srcH, u + du, v);
    float d = clamp(d0 * 0.7f + dm * 0.15f + dp_ * 0.15f, 0.0f, 1.0f);
    float d_shaped = d * (1.0f + 0.35f * (1.0f - d));
    float shift = (d_shaped - convergence) * depthStrength * eye;
    float e0 = smoothstep(0.0f, 0.05f, u);
    float e1 = smoothstep(0.0f, 0.05f, 1.0f - u);
    shift *= e0 * e1;
    float fx = clamp(u + shift, 0.0f, 1.0f);

    float cval = sampf(col + (uint)comp * (uint)(srcW * srcH),
                       srcW, srcH, fx, v)
                 * 255.0f + 0.5f;  // planar CHW channel base
    out[idx] = (uchar)clamp(cval, 0.0f, 255.0f);
}

static inline float sample_common(
    device const float* image,
    uint batch,
    uint channel,
    uint W,
    uint H,
    uint C,
    float x,
    float y
) {
    // Match grid_sample(..., align_corners=True, padding_mode=border).
    x = clamp(x, 0.0f, (float)W - 1.0f);
    y = clamp(y, 0.0f, (float)H - 1.0f);
    uint x0 = (uint)floor(x), y0 = (uint)floor(y);
    uint x1 = min(x0 + 1u, W - 1u), y1 = min(y0 + 1u, H - 1u);
    float fx = x - (float)x0, fy = y - (float)y0;
    uint plane = H * W;
    uint base = (batch * C + channel) * plane;
    float a = image[base + y0 * W + x0];
    float b = image[base + y0 * W + x1];
    float c = image[base + y1 * W + x0];
    float d = image[base + y1 * W + x1];
    return mix(mix(a, b, fx), mix(c, d, fx), fy);
}

static inline float blend_common(
    device const float* col,
    device const float* dep,
    device const float* shift,
    uint batch,
    uint channel,
    uint W,
    uint H,
    uint C,
    uint x,
    uint y,
    float eye_sign
) {
    uint plane = H * W;
    uint depth_idx = batch * plane + y * W + x;
    float d = clamp(dep[depth_idx], 0.0f, 1.0f);
    float w0 = exp(-(d * d) / 0.08f);
    float w1 = exp(-((d - 1.0f) * (d - 1.0f)) / 0.08f);
    float weight_sum = max(w0 + w1, 1.0e-6f);
    float base = shift[depth_idx];
    float shift0 = base * 0.875f;
    float shift1 = base;
    float sample0 = sample_common(
        col, batch, channel, W, H, C,
        (float)x + shift0 * eye_sign, (float)y
    );
    float sample1 = sample_common(
        col, batch, channel, W, H, C,
        (float)x + shift1 * eye_sign, (float)y
    );
    return (w0 * sample0 + w1 * sample1) / weight_sum;
}

static inline float downsample_common(
    device const float* col,
    device const float* dep,
    device const float* shift,
    uint batch,
    uint channel,
    uint W,
    uint H,
    uint C,
    uint x,
    uint y,
    bool horizontal,
    float eye_sign
) {
    uint center = 2u * (horizontal ? x : y);
    float values[4];
    values[0] = -1.0f;
    values[1] = 9.0f;
    values[2] = 9.0f;
    values[3] = -1.0f;
    float total = 0.0f;
    for (uint tap = 0u; tap < 4u; ++tap) {
        int source = (int)center + (int)tap - 1;
        uint limit = (horizontal ? W : H) - 1u;
        uint coordinate = (uint)clamp(source, 0, (int)limit);
        uint sx = horizontal ? coordinate : x;
        uint sy = horizontal ? y : coordinate;
        total += values[tap] * blend_common(
            col, dep, shift, batch, channel, W, H, C, sx, sy, eye_sign
        );
    }
    return total * (1.0f / 16.0f);
}

kernel void warp_composite2_u8(
    device uchar* out        [[buffer(0)]],
    device const float* col  [[buffer(1)]],
    device const float* dep  [[buffer(2)]],
    device const float* shift [[buffer(3)]],
    constant uint& B         [[buffer(4)]],
    constant uint& C         [[buffer(5)]],
    constant uint& W         [[buffer(6)]],
    constant uint& H         [[buffer(7)]],
    constant uint& outW      [[buffer(8)]],
    constant uint& outH      [[buffer(9)]],
    constant uint& format    [[buffer(10)]],
    uint idx [[thread_position_in_grid]])
{
    uint total = B * C * outH * outW;
    if (idx >= total) return;
    uint out_plane = outH * outW;
    uint out_pixels = C * out_plane;
    uint batch = idx / out_pixels;
    uint rem = idx % out_pixels;
    uint channel = rem / out_plane;
    uint pixel = rem % out_plane;
    uint ox = pixel % outW;
    uint oy = pixel / outW;
    bool horizontal = (format == 0u || format == 1u);
    bool half_res = (format == 0u || format == 2u);
    uint eyeW = horizontal ? (half_res ? W / 2u : W) : W;
    uint eyeH = horizontal ? H : (half_res ? H / 2u : H);
    bool right_eye = horizontal ? ox >= eyeW : oy >= eyeH;
    uint x = horizontal ? (right_eye ? ox - eyeW : ox) : ox;
    uint y = horizontal ? oy : (right_eye ? oy - eyeH : oy);
    float eye_sign = right_eye ? -1.0f : 1.0f;
    float value;
    if (half_res) {
        value = downsample_common(
            col, dep, shift, batch, channel, W, H, C, x, y,
            horizontal, eye_sign
        );
    } else {
        value = blend_common(
            col, dep, shift, batch, channel, W, H, C, x, y, eye_sign
        );
    }
    out[idx] = (uchar)clamp(value * 255.0f + 0.5f, 0.0f, 255.0f);
}

kernel void warp_composite2(
    device float* left       [[buffer(0)]],
    device float* right      [[buffer(1)]],
    device const float* col  [[buffer(2)]],
    device const float* dep  [[buffer(3)]],
    device const float* shift [[buffer(4)]],
    constant uint& B         [[buffer(5)]],
    constant uint& C         [[buffer(6)]],
    constant uint& W         [[buffer(7)]],
    constant uint& H         [[buffer(8)]],
    uint idx [[thread_position_in_grid]])
{
    uint total = B * C * H * W;
    if (idx >= total) return;
    uint plane = H * W;
    uint pixels = C * plane;
    uint batch = idx / pixels;
    uint rem = idx % pixels;
    uint channel = rem / plane;
    uint pixel = rem % plane;
    uint y = pixel / W;
    uint x = pixel % W;
    uint depth_idx = batch * plane + pixel;
    float d = clamp(dep[depth_idx], 0.0f, 1.0f);
    float w0 = exp(-(d * d) / 0.08f);
    float w1 = exp(-((d - 1.0f) * (d - 1.0f)) / 0.08f);
    float weight_sum = max(w0 + w1, 1.0e-6f);
    w0 /= weight_sum;
    w1 /= weight_sum;
    float base = shift[depth_idx];
    // layers.py/synthesis.py use factors 0.875 and 1.0 for two layers.
    float shift0 = base * 0.875f;
    float shift1 = base;
    float left0 = sample_common(col, batch, channel, W, H, C, (float)x + shift0, (float)y);
    float left1 = sample_common(col, batch, channel, W, H, C, (float)x + shift1, (float)y);
    float right0 = sample_common(col, batch, channel, W, H, C, (float)x - shift0, (float)y);
    float right1 = sample_common(col, batch, channel, W, H, C, (float)x - shift1, (float)y);
    left[idx] = w0 * left0 + w1 * left1;
    right[idx] = w0 * right0 + w1 * right1;
}

static inline float read_eye(
    device const float* eye,
    uint batch,
    uint channel,
    uint W,
    uint H,
    uint C,
    uint x,
    uint y
) {
    return eye[(batch * C + channel) * H * W + y * W + x];
}

kernel void pack_eyes_u8(
    device uchar* out       [[buffer(0)]],
    device const float* left [[buffer(1)]],
    device const float* right [[buffer(2)]],
    constant uint& B        [[buffer(3)]],
    constant uint& C        [[buffer(4)]],
    constant uint& W        [[buffer(5)]],
    constant uint& H        [[buffer(6)]],
    constant uint& outW     [[buffer(7)]],
    constant uint& outH     [[buffer(8)]],
    constant uint& format   [[buffer(9)]],
    uint idx [[thread_position_in_grid]])
{
    uint total = B * C * outH * outW;
    if (idx >= total) return;
    uint out_plane = outH * outW;
    uint out_pixels = C * out_plane;
    uint batch = idx / out_pixels;
    uint rem = idx % out_pixels;
    uint channel = rem / out_plane;
    uint pixel = rem % out_plane;
    uint ox = pixel % outW;
    uint oy = pixel / outW;
    bool horizontal = (format == 0u || format == 1u);
    bool half_res = (format == 0u || format == 2u);
    uint eyeW = horizontal ? (half_res ? W / 2u : W) : W;
    uint eyeH = horizontal ? H : (half_res ? H / 2u : H);
    bool right_eye = horizontal ? ox >= eyeW : oy >= eyeH;
    uint x = horizontal ? (right_eye ? ox - eyeW : ox) : ox;
    uint y = horizontal ? oy : (right_eye ? oy - eyeH : oy);
    device const float* eye = right_eye ? right : left;
    float value;
    if (half_res) {
        uint center = 2u * (horizontal ? x : y);
        float taps[4] = {-1.0f, 9.0f, 9.0f, -1.0f};
        value = 0.0f;
        for (uint tap = 0u; tap < 4u; ++tap) {
            int source = (int)center + (int)tap - 1;
            uint limit = (horizontal ? W : H) - 1u;
            uint coordinate = (uint)clamp(source, 0, (int)limit);
            uint sx = horizontal ? coordinate : x;
            uint sy = horizontal ? y : coordinate;
            value += taps[tap] * read_eye(
                eye, batch, channel, W, H, C, sx, sy
            );
        }
        value *= 1.0f / 16.0f;
    } else {
        value = read_eye(eye, batch, channel, W, H, C, x, y);
    }
    out[idx] = (uchar)clamp(value * 255.0f + 0.5f, 0.0f, 255.0f);
}
"""


@functools.lru_cache(maxsize=1)
def _lib():
    import torch

    return torch.mps.compile_shader(WARP_MSL)


def warp_params_from_env() -> tuple[float, float, float]:
    """Mirror macos_metal_viewer's calibration knobs exactly."""
    ipd_uv = float(os.environ.get("D2S_METAL_WARP_IPD", "0.064") or 0.064)
    depth_strength = 0.1 * float(
        os.environ.get("D2S_METAL_WARP_DEPTH_STRENGTH", "4.0") or 4.0
    )
    convergence = float(os.environ.get("D2S_METAL_WARP_CONVERGENCE", "0.0") or 0.0)
    return ipd_uv / 2.0, depth_strength, convergence


def fused_enabled() -> bool:
    """Darwin + Vulkan viewer + not explicitly disabled."""
    return (
        sys.platform == "darwin"
        and os.environ.get("D2S_MAC_VIEWER") == "vulkan"
        and os.environ.get("D2S_VK_FUSED_WARP", "1")
        not in {"0", "false", "off"}
    )


@functools.lru_cache(maxsize=8)
def _smooth_texels_cached(key: tuple[int, int, int]) -> float:
    import os as _os

    raw = _os.environ.get("D2S_WARP_DEPTH_SMOOTH_TEXELS", "")
    try:
        return max(0.0, float(raw))
    except Exception:
        pass
    eye_w, src_w, base = key
    # Scale-invariant default: 1.5 texels at the reference where eye width
    # equals source width; grows proportionally when the packed frame is
    # larger than the source (native-res presentation).
    return 1.5 * (float(eye_w) / float(src_w)) if src_w else 1.5


def warp_smooth_texels(src_w: int, out_w: int) -> float:
    """Depth-smoothing aperture in source texels for the warp kernel."""
    eye_w = max(1, int(out_w) // 2)  # half-SBS per-eye width
    return _smooth_texels_cached((eye_w, int(src_w), int(out_w)))


def pack_target(src_w: int, src_h: int, output_format: str = "half_sbs") -> tuple[int, int]:
    """Frame dims for a given runtime input size and output format.

    Mirrors the NVIDIA local-mode contract: SBS geometry follows the
    RUNTIME input resolution and the selected format -- never the viewer
    window. 1080p in -> half_sbs/half_tab 1920x1080, full_sbs 3840x1080,
    and full_tab 1920x2160. Mono and diagnostic modes preserve source size.
    """
    sw, sh = int(src_w), int(src_h)
    output_format = str(output_format)
    if output_format == "full_sbs":
        return sw * 2, sh
    if output_format == "full_tab":
        return sw, sh * 2
    if output_format in {"half_sbs", "half_tab"}:
        # The packed half modes preserve the source frame dimensions while
        # requiring an even split for the local viewer's eye regions.
        return (sw - sw % 2, sh) if output_format == "half_sbs" else (sw, sh)
    return sw, sh


def fused_sbs_pack(rgb_f32_chw, depth_f32, host_out=None, out_size=None,
                   output_format: str = "half_sbs"):
    """Run the fused kernel; return (host_view|None, w, h) like
    _pack_sbs_host_frame, or None on any failure (caller falls back).

    Default frame dims follow the NVIDIA local-mode contract: derived from
    the runtime INPUT resolution and ``output_format`` (half_sbs WxH,
    full_sbs 2WxH). ``out_size`` remains as an explicit override."""
    try:
        import torch

        if rgb_f32_chw.dim() == 4 and int(rgb_f32_chw.shape[0]) == 1:
            rgb_f32_chw = rgb_f32_chw.squeeze(0)  # BCHW -> CHW
        if rgb_f32_chw.dim() != 3:
            if os.environ.get("D2S_FUSED_DEBUG"):
                print(f"[fused] skip: dim={rgb_f32_chw.dim()}", flush=True)
            return None
        channels, h, w = (
            int(rgb_f32_chw.shape[0]),
            int(rgb_f32_chw.shape[-2]),
            int(rgb_f32_chw.shape[-1]),
        )
        if channels != 3 or h <= 0 or w <= 0:
            if os.environ.get("D2S_FUSED_DEBUG"):
                print(f"[fused] skip: ch={channels} h={h} w={w}", flush=True)
            return None
        dep = depth_f32
        if dep.dim() == 3:
            dep = dep.squeeze(0)
        ow, oh = (
            (int(out_size[0]), int(out_size[1]))
            if out_size is not None
            else pack_target(w, h, output_format)
        )
        if ow % 2 != 0 or ow < 4 or oh < 4:
            return None  # half-SBS needs an even frame width
        out_t = torch.empty(ow * oh * 4, dtype=torch.uint8, device="mps")
        eo, ds, cv = warp_params_from_env()
        stex = warp_smooth_texels(w, ow)
        _lib().warp_pack(
            out_t, rgb_f32_chw.contiguous(), dep.contiguous(),
            float(eo), float(ds), float(cv),
            int(w), int(h), int(ow), int(oh), float(stex),
        )
        # Half-SBS: reported dims are the FRAME dims (ow x oh), matching the
        # synthesized half_sbs contract the viewer was built around.
        if host_out is not None:
            dst = torch.frombuffer(host_out, dtype=torch.uint8)
            dst.copy_(out_t)
            view = np.frombuffer(host_out, dtype=np.uint8).reshape(oh, ow, 4)
            return view, ow, oh
        host = out_t.cpu().numpy().reshape(oh, ow, 4)
        return host, ow, oh
    except Exception as exc:
        if os.environ.get("D2S_FUSED_DEBUG"):
            print(f"[fused] pack failed: {exc!r}", flush=True)
        return None


def mps_warp_composite2(rgb_f32, depth_f32, base_shift):
    """Run the canonical two-layer warp without MPS grid_sample launches.

    The kernel mirrors the common synthesis path for the streaming profile:
    two depth layers, symmetric eyes, bilinear border sampling, and the
    already-resolved pixel shift. It intentionally returns ``None`` for any
    unsupported shape so the caller can use the existing torch path.
    """
    try:
        import torch

        if sys.platform != "darwin" or rgb_f32.device.type != "mps":
            return None
        if rgb_f32.ndim == 3:
            rgb_f32 = rgb_f32.unsqueeze(0)
        if rgb_f32.ndim != 4 or rgb_f32.dtype != torch.float32:
            return None
        if depth_f32.ndim == 3:
            depth_f32 = depth_f32.unsqueeze(1)
        if base_shift.ndim == 3:
            base_shift = base_shift.unsqueeze(1)
        if depth_f32.ndim != 4 or base_shift.ndim != 4:
            return None
        batch, channels, height, width = map(int, rgb_f32.shape)
        if channels != 3 or tuple(depth_f32.shape) != (batch, 1, height, width):
            return None
        if tuple(base_shift.shape) != (batch, 1, height, width):
            return None
        left = torch.empty_like(rgb_f32)
        right = torch.empty_like(rgb_f32)
        _lib().warp_composite2(
            left,
            right,
            rgb_f32.contiguous(),
            depth_f32.contiguous(),
            base_shift.contiguous(),
            int(batch),
            int(channels),
            int(width),
            int(height),
        )
        return left, right
    except Exception as exc:
        if os.environ.get("D2S_FUSED_DEBUG"):
            print(f"[mps-warp] canonical kernel failed: {exc!r}", flush=True)
        return None


def mps_warp_composite2_u8(rgb_f32, depth_f32, base_shift, output_format: str):
    """Pack the canonical two-layer output directly into an MPS uint8 tensor."""
    try:
        import torch

        if sys.platform != "darwin" or rgb_f32.device.type != "mps":
            return None
        if rgb_f32.ndim == 3:
            rgb_f32 = rgb_f32.unsqueeze(0)
        if rgb_f32.ndim != 4 or rgb_f32.dtype != torch.float32:
            return None
        if depth_f32.ndim == 3:
            depth_f32 = depth_f32.unsqueeze(1)
        if base_shift.ndim == 3:
            base_shift = base_shift.unsqueeze(1)
        batch, channels, height, width = map(int, rgb_f32.shape)
        if channels != 3 or height % 2 or width % 2:
            return None
        if tuple(depth_f32.shape) != (batch, 1, height, width):
            return None
        if tuple(base_shift.shape) != (batch, 1, height, width):
            return None
        formats = {"half_sbs": 0, "full_sbs": 1, "half_tab": 2, "full_tab": 3}
        format_id = formats.get(str(output_format))
        if format_id is None:
            return None
        out_width = width * 2 if format_id == 1 else width
        out_height = height * 2 if format_id == 3 else height
        warped = mps_warp_composite2(rgb_f32, depth_f32, base_shift)
        if warped is None:
            return None
        left, right = warped
        out = torch.empty(
            (batch, channels, out_height, out_width),
            dtype=torch.uint8,
            device="mps",
        )
        _lib().pack_eyes_u8(
            out,
            left,
            right,
            int(batch),
            int(channels),
            int(width),
            int(height),
            int(out_width),
            int(out_height),
            int(format_id),
        )
        return out
    except Exception as exc:
        if os.environ.get("D2S_FUSED_DEBUG"):
            print(f"[mps-warp] packed canonical kernel failed: {exc!r}", flush=True)
        return None
