from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DepthResponseFn = Callable[[Any], Any]

_PROTECTED_PARALLAX_CORE = None
_PROTECTED_PARALLAX_GRANT = None
_PROTECTED_CORE_REQUIRED = False

PARALLAX_RESOLVER_VERSION = 1
DEPTH_RESPONSE_NAME = "linear_clamp_convergence_v1"

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
    if _PROTECTED_PARALLAX_CORE is None and (_PROTECTED_CORE_REQUIRED or max_disparity_px is None):
        raise RuntimeError("protected parallax core is required before runtime use")
    if _PROTECTED_PARALLAX_CORE is None:
        def depth_response(depth):
            conv = convergence
            if hasattr(conv, "to") and hasattr(depth, "device"):
                conv = conv.to(device=depth.device, dtype=depth.dtype)
            return depth.clamp(0, 1) - conv

        return ParallaxBudget(
            max_disparity_px=max(0.0, float(max_disparity_px)),
            depth_response=depth_response,
            preset=str(preset or "standard"),
        )
    return _PROTECTED_PARALLAX_CORE.resolve_parallax_budget(
        render_width,
        render_height,
        preset,
        convergence,
        max_disparity_px=max_disparity_px,
    )


def parallax_debug_info(budget: ParallaxBudget) -> dict[str, float | int | str]:
    if _PROTECTED_PARALLAX_CORE is None and _PROTECTED_CORE_REQUIRED:
        raise RuntimeError("protected parallax core is required before runtime use")
    if _PROTECTED_PARALLAX_CORE is None:
        return {
            "resolved_max_disparity_px": float(budget.max_disparity_px),
            "parallax_budget_preset": str(budget.preset),
            "depth_response": str(budget.depth_response_name),
            "parallax_resolver_version": int(budget.resolver_version),
        }
    return _PROTECTED_PARALLAX_CORE.parallax_debug_info(budget)


def require_protected_parallax_core() -> None:
    """Make runtime callers fail closed until the protected core is loaded."""

    global _PROTECTED_CORE_REQUIRED
    _PROTECTED_CORE_REQUIRED = True


def compute_protected_shift(depth: Any, width: int, params: Any):
    """Delegate depth-to-disparity computation to the loaded protected core."""

    if _PROTECTED_PARALLAX_CORE is not None:
        compute = getattr(_PROTECTED_PARALLAX_CORE, "compute_shift_px", None)
        if not callable(compute):
            raise RuntimeError("protected parallax core does not expose shift computation")
        return compute(depth, width, params)
    if _PROTECTED_CORE_REQUIRED:
        raise RuntimeError("protected parallax core is required before runtime use")
    return None


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
