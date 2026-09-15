"""Target-only Intel Vulkan -> D3D11 -> oneVPL -> MediaMTX smoke test.

Run this on a Windows Intel machine after the native DLL bundle is installed.
The test is intentionally strict: it never labels the path zero-copy. With
--mux-url it also feeds the encoded H.264 packets to an existing MediaMTX
RTSP publisher and verifies that the mux process remains alive.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--mux-url", default="")
    parser.add_argument("--ffmpeg", default="")
    return parser.parse_args()


def run_smoke(
    *,
    width: int,
    height: int,
    fps: int,
    frames: int,
    mux_url: str = "",
    ffmpeg_path: str = "",
) -> dict[str, object]:
    if os.name != "nt":
        raise RuntimeError("Intel Vulkan/oneVPL smoke test requires Windows")

    import torch

    from desktop2stereo.stereo_runtime.runtime import VulkanComputeRequest
    from desktop2stereo.stereo_runtime.intel_vulkan_sbs import IntelVulkanSbsRuntimeBridge
    from desktop2stereo.stereo_runtime.vulkan_stereo_pass import VulkanStereoFusedParams
    from desktop2stereo.stereo_runtime.providers.intel.onevpl_d3d11_encoder import (
        OneVPLD3D11SurfaceEncoder,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/Vulkan torch device is unavailable")
    device = torch.device("cuda")
    bridge = IntelVulkanSbsRuntimeBridge(width, height)
    mux: subprocess.Popen[bytes] | None = None
    packet_count = 0
    packet_bytes = 0
    try:
        if mux_url:
            if not ffmpeg_path:
                raise RuntimeError("--ffmpeg is required when --mux-url is used")
            mux = subprocess.Popen(
                [
                    ffmpeg_path, "-hide_banner", "-loglevel", "warning",
                    "-f", "h264", "-r", str(fps), "-i", "pipe:0",
                    "-an", "-c:v", "copy", "-f", "rtsp",
                    "-rtsp_transport", "tcp", mux_url,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

        request = VulkanComputeRequest(
            rgb=torch.rand((1, 3, height, width), device=device, dtype=torch.float32),
            depth=torch.rand((1, 1, height, width), device=device, dtype=torch.float32),
            shift=torch.rand((1, 1, height, width), device=device, dtype=torch.float32),
            params=VulkanStereoFusedParams(),
        )
        for _ in range(max(1, frames)):
            frame = bridge.submit(request)
            if not frame.producer_ready or frame.ready_timeline < 1:
                raise RuntimeError("Vulkan producer completion contract was not satisfied")
            if frame.zero_copy or frame.gpu_copy_count != 1 or frame.gpu_to_cpu:
                raise RuntimeError(
                    "Intel path reported an unsafe zero-copy/copy contract: "
                    f"gpu_to_cpu={frame.gpu_to_cpu} zero_copy={frame.zero_copy} "
                    f"gpu_copy_count={frame.gpu_copy_count}"
                )
            encoder = OneVPLD3D11SurfaceEncoder(
                width=frame.width,
                height=frame.height,
                fps=fps,
                bitrate=10_000_000,
                d3d11_device=frame.device,
            )
            try:
                if int(encoder.adapter_luid) != int(frame.adapter_luid):
                    raise RuntimeError(
                        "Vulkan/D3D11/oneVPL Adapter LUID mismatch: "
                        f"vulkan={frame.adapter_luid} onevpl={encoder.adapter_luid}"
                    )
                encoder.submit_nv12(frame.texture, packet_count)
                while True:
                    packet = encoder.read_packet()
                    if not packet:
                        break
                    packet_count += 1
                    packet_bytes += len(packet)
                    if mux is not None and mux.stdin is not None:
                        mux.stdin.write(packet)
                        mux.stdin.flush()
                if mux is not None and mux.poll() is not None:
                    raise RuntimeError(f"MediaMTX mux process exited: {mux.returncode}")
            finally:
                encoder.close()
        return {
            "ok": True,
            "frames": max(1, frames),
            "packets": packet_count,
            "packet_bytes": packet_bytes,
            "media_mtx_mux_checked": bool(mux_url),
            "adapter_luid": int(frame.adapter_luid),
            "gpu_to_cpu": False,
            "zero_copy": False,
            "gpu_copy_count": 1,
            "note": "strict zero-copy remains gated until target display validation",
        }
    finally:
        if mux is not None:
            if mux.stdin is not None:
                mux.stdin.close()
            try:
                mux.wait(timeout=3)
            except subprocess.TimeoutExpired:
                mux.kill()
        bridge.close()


def main() -> int:
    args = _parse_args()
    try:
        result = run_smoke(
            width=args.width,
            height=args.height,
            fps=args.fps,
            frames=args.frames,
            mux_url=args.mux_url,
            ffmpeg_path=args.ffmpeg,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
