from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DepthResponseFn = Callable[[Any], Any]

_PROTECTED_PARALLAX_CORE = None
_PROTECTED_PARALLAX_GRANT = None

PARALLAX_RESOLVER_VERSION = 1
DEPTH_RESPONSE_NAME = "linear_clamp_convergence_v1"

PARALLAX_BUDGET_TABLE: dict[str, dict[int, float]] = {
    "comfort": {720: 24.0, 1080: 32.0, 1440: 48.0, 2160: 64.0},
    "standard": {720: 36.0, 1080: 48.0, 1440: 64.0, 2160: 96.0},
    "strong": {720: 48.0, 1080: 64.0, 1440: 88.0, 2160: 128.0},
    "extreme": {720: 64.0, 1080: 80.0, 1440: 112.0, 2160: 160.0},
}

_STRENGTH_ALIASES = {
    "comfortable": "comfort",
    "soft": "comfort",
    "low": "comfort",
    "normal": "standard",
    "balanced": "standard",
    "default": "standard",
    "std": "standard",
    "high": "strong",
    "enhanced": "strong",
    "very_strong": "extreme",
    "max": "extreme",
    "maximum": "extreme",
}


@dataclass(frozen=True)
class ParallaxBudget:
    max_disparity_px: float
    depth_response: DepthResponseFn
    preset: str
    depth_response_name: str = DEPTH_RESPONSE_NAME
    resolver_version: int = PARALLAX_RESOLVER_VERSION


def resolve_parallax_budget(
    render_width: int,
    render_height: int,
    preset: str,
    convergence: Any = 0.0,
    *,
    max_disparity_px: float | None = None,
) -> ParallaxBudget:
    if _PROTECTED_PARALLAX_CORE is not None:
        return _PROTECTED_PARALLAX_CORE.resolve_parallax_budget(
            render_width,
            render_height,
            preset,
            convergence,
            max_disparity_px=max_disparity_px,
        )
    normalized_preset = _normalize_strength_preset(preset)
    width = max(1, int(render_width))
    height = max(1, int(render_height))

    if max_disparity_px is not None:
        resolved_max_disparity = max(0.0, float(max_disparity_px))
    else:
        resolved_max_disparity = _resolve_table_budget(width, height, normalized_preset)

    def depth_response(depth):
        conv = convergence
        if hasattr(conv, "to") and hasattr(depth, "device"):
            conv = conv.to(device=depth.device, dtype=depth.dtype)
        return depth.clamp(0, 1) - conv

    return ParallaxBudget(
        max_disparity_px=float(resolved_max_disparity),
        depth_response=depth_response,
        preset=normalized_preset,
        depth_response_name=DEPTH_RESPONSE_NAME,
    )


def parallax_debug_info(budget: ParallaxBudget) -> dict[str, float | int | str]:
    if _PROTECTED_PARALLAX_CORE is not None:
        return _PROTECTED_PARALLAX_CORE.parallax_debug_info(budget)
    return {
        "resolved_max_disparity_px": float(budget.max_disparity_px),
        "parallax_budget_preset": str(budget.preset),
        "depth_response": str(budget.depth_response_name),
        "parallax_resolver_version": int(budget.resolver_version),
    }


def configure_protected_parallax_core(
    resource_path: str,
    grant_jws: str,
    *,
    now: int,
    expected_device_hash: str | None = None,
    native_path: str | None = None,
    require_native: bool = False,
) -> dict[str, Any]:
    """Load the authorized parallax implementation before runtime imports it."""

    global _PROTECTED_PARALLAX_CORE, _PROTECTED_PARALLAX_GRANT
    from desktop2stereo.auth.core import load_core_module

    grant, module = load_core_module(
        resource_path,
        grant_jws,
        now=now,
        expected_device_hash=expected_device_hash,
        native_path=native_path,
        require_native=require_native,
    )
    _PROTECTED_PARALLAX_GRANT = grant
    _PROTECTED_PARALLAX_CORE = module
    return {
        "core_id": grant.core_id,
        "core_version": grant.core_version,
        "license_id": grant.license_id,
        "grant_id": grant.grant_id,
        "expires_at": grant.expires_at,
    }


def _normalize_strength_preset(preset: str | None) -> str:
    key = str(preset or "standard").strip().lower().replace("-", "_").replace(" ", "_")
    key = _STRENGTH_ALIASES.get(key, key)
    if key in PARALLAX_BUDGET_TABLE:
        return key
    raise ValueError(f"unknown parallax strength preset: {preset!r}")


def _resolve_table_budget(width: int, height: int, preset: str) -> float:
    short_side = float(min(width, height))
    table = PARALLAX_BUDGET_TABLE[preset]
    levels = sorted(table)
    if short_side <= levels[0]:
        base = table[levels[0]] * short_side / float(levels[0])
    elif short_side >= levels[-1]:
        base = table[levels[-1]] * short_side / float(levels[-1])
    else:
        lower = levels[0]
        upper = levels[-1]
        for idx in range(len(levels) - 1):
            if levels[idx] <= short_side <= levels[idx + 1]:
                lower = levels[idx]
                upper = levels[idx + 1]
                break
        t = (short_side - lower) / float(upper - lower)
        base = table[lower] + (table[upper] - table[lower]) * t
    return float(base * _aspect_protection_factor(width, height))


def _aspect_protection_factor(width: int, height: int) -> float:
    short_side = max(1.0, float(min(width, height)))
    aspect = max(float(width), float(height)) / short_side
    if aspect <= 2.0:
        return 1.0
    return max(0.70, min(1.0, 2.0 / aspect))
