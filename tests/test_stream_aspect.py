from __future__ import annotations

import numpy as np

from streaming.aspect import (
    apply_aspect_on_cpu,
    presentation_blit_regions as stream_regions,
    transport_canvas_size,
)
from viewer.vulkan_local_viewer import presentation_blit_regions as viewer_regions


def _aspect(size: tuple[int, int]) -> float:
    return size[0] / size[1]


def _contain_cases():
    return [
        # 16:10 input (Half-SBS) into a 16:9 transport canvas.
        ((1920, 1200), (2132, 1200), "Half-SBS", (1920, 1200)),
        # Pure 16:9 target with a 16:10 Half-SBS source.
        ((1920, 1200), (1920, 1080), "Half-SBS", (1920, 1200)),
        # 16:9 input / 16:10 target.
        ((3840, 2160), (1920, 1200), "Half-SBS", (3840, 2160)),
        # Full-SBS keeps the full eye aspect.
        ((7680, 2160), (3840, 1080), "Full-SBS", (3840, 2160)),
        # Cover / stretch are unaffected by input_size.
        ((3840, 2160), (1920, 1200), "cover", "Half-SBS", (3840, 2160)),
        # No input_size falls back to the packed eye size.
        ((1920, 1200), (2132, 1200), "contain", "Half-SBS", None),
    ]


def test_presentation_blit_regions_match_local_viewer() -> None:
    """The stream must apply exactly the same per-eye placement as the viewer."""
    for case in _contain_cases():
        source, target = case[0], case[1]
        fit_mode = case[2]
        display_mode = case[3]
        input_size = case[4] if len(case) > 4 else None
        assert stream_regions(
            source, target, fit_mode, display_mode, input_size
        ) == viewer_regions(
            source, target, fit_mode, display_mode, input_size
        ), case


def test_contain_transport_canvas_is_16_9_for_16_10_input() -> None:
    canvas = transport_canvas_size(
        (1920, 1200), "contain", input_size=(1920, 1200), display_mode="Half-SBS"
    )
    # 2132x1200 = 16:9; must be strictly wider than the 16:10 source.
    assert abs(_aspect(canvas) - 16 / 9) < 0.01
    assert _aspect(canvas) > 16 / 10


def test_contain_16_10_output_keeps_per_eye_ratio() -> None:
    """A 16:10 input in a 16:9 canvas is pillarboxed without distorting each eye."""
    h, w = 1200, 1920
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, : w // 2, 0] = 255  # left eye -> red
    frame[:, w // 2 :, 1] = 255  # right eye -> green

    out = apply_aspect_on_cpu(
        frame.copy(),
        source_size=(w, h),
        target_size=(2132, 1200),
        fit_mode="contain",
        display_mode="Half-SBS",
        input_size=(w, h),
    )
    assert out.shape == (1200, 2132, 3)
    assert abs(_aspect((2132, 1200)) - 16 / 9) < 0.01

    # Each eye occupies a 960-wide content region (deduplicated by the fact the
    # half canvas is 1066 wide), centered, with the eye's own aspect preserved:
    # no eye pixel may be stretched beyond its source column range.
    center_row = out[600]
    red = np.where((center_row[:, 0] > 200) & (center_row[:, 1] < 50))[0]
    green = np.where((center_row[:, 1] > 200) & (center_row[:, 0] < 50))[0]
    assert len(red) == 960  # left eye content width
    assert len(green) == 960  # right eye content width
    assert red.min() == 53 and red.max() == 1012  # pillarboxed, not stretched
    assert green.min() == 1119 and green.max() == 2078
