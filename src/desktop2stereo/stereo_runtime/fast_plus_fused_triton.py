"""Compatibility guard for the retired public Fast+ disparity kernel.

The production path must receive the protected core's per-pixel shift map.
This module remains importable for older integrations, but never computes
disparity from depth or exposes a public formula fallback.
"""

from __future__ import annotations

from typing import NoReturn


def can_use_fast_plus_fused_half_sbs_uint8(*_args: object, **_kwargs: object) -> bool:
    return False


def make_fast_plus_fused_half_sbs_uint8(*_args: object, **_kwargs: object) -> NoReturn:
    raise RuntimeError("protected parallax core is required for Fast+ fused output")
