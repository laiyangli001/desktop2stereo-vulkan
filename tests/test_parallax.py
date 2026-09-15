from __future__ import annotations

import pytest
import torch

from stereo_runtime.baseline_shift import ShiftParams, compute_shift_px
from stereo_runtime.parallax import parallax_debug_info, resolve_parallax_budget


@pytest.mark.parametrize("preset", ["standard", None])
def test_budget_table_is_not_available_without_protected_core(preset):
    with pytest.raises(RuntimeError, match="protected parallax core is required"):
        resolve_parallax_budget(1920, 1080, preset, convergence=0.0)


def test_debug_info_requires_protected_core(monkeypatch):
    import stereo_runtime.parallax as parallax

    monkeypatch.setattr(parallax, "_PROTECTED_PARALLAX_CORE", None)
    monkeypatch.setattr(parallax, "_PROTECTED_CORE_REQUIRED", False)
    parallax.require_protected_parallax_core()
    with pytest.raises(RuntimeError, match="protected parallax core is required"):
        parallax_debug_info(None)


def test_explicit_test_budget_keeps_generic_shift_regression_available():
    budget = resolve_parallax_budget(1920, 1080, "standard", max_disparity_px=40.0)
    assert budget.max_disparity_px == 40.0


def test_compute_shift_px_uses_half_of_total_max_disparity_for_each_eye():
    depth = torch.ones(1, 1, 1, 1)
    shift = compute_shift_px(
        depth,
        1920,
        ShiftParams(depth_strength=1.0, convergence=0.0, max_disparity_px=96.0),
    )

    assert shift.item() == pytest.approx(-48.0)


def test_compute_shift_px_scales_actual_displacement_by_depth_strength():
    depth = torch.ones(1, 1, 1, 1)

    normal = compute_shift_px(depth, 1920, ShiftParams(depth_strength=1.0, convergence=0.0, max_disparity_px=40.0))
    strong = compute_shift_px(depth, 1920, ShiftParams(depth_strength=2.5, convergence=0.0, max_disparity_px=40.0))
    flat = compute_shift_px(depth, 1920, ShiftParams(depth_strength=0.0, convergence=0.0, max_disparity_px=40.0))

    assert normal.item() == pytest.approx(-20.0)
    assert strong.item() == pytest.approx(-50.0)
    assert flat.item() == pytest.approx(0.0)


def test_compute_shift_px_applies_layered_parallax_scales():
    depth = torch.tensor([[[[0.0, 0.5, 1.0]]]])
    shift = compute_shift_px(
        depth,
        1920,
        ShiftParams(
            depth_strength=1.0,
            convergence=0.5,
            max_disparity_px=40.0,
            foreground_shift_scale=2.0,
            midground_shift_scale=1.0,
            background_shift_scale=0.5,
        ),
    )

    assert shift[0, 0, 0, 0].item() == pytest.approx(5.0)
    assert shift[0, 0, 0, 1].item() == pytest.approx(0.0)
    assert shift[0, 0, 0, 2].item() == pytest.approx(-20.0)


def test_layered_shift_fast_path_matches_weight_map_reference(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STEREO_RUNTIME_DISABLE_TRITON", "1")
    depth = torch.tensor([[[[-0.2, 0.0, 0.25, 0.5, 0.75, 1.0, 1.2]]]])
    params = ShiftParams(
        depth_strength=0.25,
        convergence=0.15,
        max_disparity_px=96.0,
        foreground_shift_scale=1.15,
        midground_shift_scale=1.05,
        background_shift_scale=0.85,
    )

    normalized = depth.clamp(0.0, 1.0)
    background_weight = ((0.5 - normalized) * 2.0).clamp(0.0, 1.0)
    foreground_weight = ((normalized - 0.5) * 2.0).clamp(0.0, 1.0)
    midground_weight = (1.0 - background_weight - foreground_weight).clamp(0.0, 1.0)
    scale = (
        background_weight * params.background_shift_scale
        + midground_weight * params.midground_shift_scale
        + foreground_weight * params.foreground_shift_scale
    )
    expected = -(normalized - params.convergence) * scale * params.depth_strength * params.max_disparity_px * 0.5

    actual = compute_shift_px(depth, 3840, params)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_layered_shift_matches_torch_reference():
    pytest.importorskip("triton")
    from stereo_runtime.baseline_shift_triton import can_use_triton_layered_shift, compute_layered_shift

    depth = torch.rand((1, 1, 64, 96), device="cuda", dtype=torch.float32)
    if not can_use_triton_layered_shift(depth, 0.15):
        pytest.skip("Triton runtime is unavailable")

    params = ShiftParams(
        depth_strength=0.25,
        convergence=0.15,
        max_disparity_px=96.0,
        foreground_shift_scale=1.15,
        midground_shift_scale=1.05,
        background_shift_scale=0.85,
    )
    normalized = depth.clamp(0.0, 1.0)
    background_scale = 0.85 + (2.0 * normalized) * (1.05 - 0.85)
    foreground_scale = 1.05 + (2.0 * normalized - 1.0) * (1.15 - 1.05)
    scale = torch.where(normalized < 0.5, background_scale, foreground_scale)
    expected = -(normalized - 0.15) * scale * 0.25 * 96.0 * 0.5

    actual = compute_layered_shift(
        depth,
        convergence=params.convergence,
        output_scale=-params.depth_strength * params.max_disparity_px * 0.5,
        foreground_scale=params.foreground_shift_scale,
        midground_scale=params.midground_shift_scale,
        background_scale=params.background_shift_scale,
    )

    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_shift_params_do_not_expose_legacy_ipd_formula_fields():
    fields = ShiftParams.__dataclass_fields__

    assert "ipd" not in fields
    assert "ipd_mm" not in fields
    assert "stereo_scale" not in fields
    assert "max_shift_ratio" not in fields
