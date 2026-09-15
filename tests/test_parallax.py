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


def test_compute_shift_px_does_not_use_public_layered_formula():
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

    expected = torch.tensor([[[[10.0, 0.0, -20.0]]]])
    assert torch.equal(shift, expected)


def test_shift_params_do_not_expose_legacy_ipd_formula_fields():
    fields = ShiftParams.__dataclass_fields__

    assert "ipd" not in fields
    assert "ipd_mm" not in fields
    assert "stereo_scale" not in fields
    assert "max_shift_ratio" not in fields
