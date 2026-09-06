from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .adapter import OutputFormat, StereoQuality, _normalize_runtime_mode


class SnapshotChangeClass(Enum):
    NO_CHANGE = "no_change"
    HOT_RELOAD = "hot_reload"
    PIPELINE_REBUILD = "pipeline_rebuild"
    SESSION_RESTART = "session_restart"


_HOT_RELOAD_FIELDS = frozenset(
    {
        "runtime_quality_mode",
        "language",
        "presentation_flags",
        "debug_flags",
        "debug_output",
        "stereo_quality",
        "stereo_preset",
        "output_format",
        "depth_strength",
        "convergence",
        "max_disparity_px",
        "parallax_preset",
        "parallax_budget_preset",
        "temporal",
        "temporal_enabled",
        "temporal_strength",
        "auto_reset_temporal",
        "scene_reset_threshold",
        "depth_pop",
        "foreground_shift_scale",
        "midground_shift_scale",
        "background_shift_scale",
        "dynamic_convergence_enabled",
        "dynamic_convergence_strength",
        "dynamic_convergence_target",
        "dynamic_convergence_alpha",
        "depth_antialias_strength",
        "edge_dilation",
        "edge_threshold",
        "mask_feather_radius",
        "hole_fill_mode",
        "hole_fill_radius",
        "hole_fill_strength",
        "screen_edge_mask_suppression",
        "cross_eyed",
        "anaglyph_method",
        "color_brightness",
        "color_contrast",
        "color_saturation",
        "color_gamma",
        "color_temperature",
        "color_tint",
        "vulkan_projection_min_lod",
        "vulkan_projection_max_lod",
        "vulkan_projection_mip_lod_bias",
        "vulkan_projection_rcas_sharpness",
        "output_headset_tier_k",
        "fused",
    }
)

_PIPELINE_REBUILD_FIELDS = frozenset(
    {
        "stereo_synthesis_mode",
        "render_size_policy",
        "stereo_render_scale",
        "output_transport",
        "depth_backend",
        "model_id",
        "export_height",
        "export_width",
        "profile_sync",
        "use_cuda_graph",
    }
)

_SESSION_RESTART_FIELDS = frozenset({"application_runtime_target", "capture_source", "capture_target", "device"})

_CONFIG_UPDATE_FIELDS = frozenset(
    {
        "stereo_quality",
        "stereo_preset",
        "output_format",
        "depth_strength",
        "convergence",
        "max_disparity_px",
        "parallax_preset",
        "temporal",
        "temporal_strength",
        "auto_reset_temporal",
        "scene_reset_threshold",
        "depth_pop",
        "foreground_shift_scale",
        "midground_shift_scale",
        "background_shift_scale",
        "dynamic_convergence_enabled",
        "dynamic_convergence_strength",
        "dynamic_convergence_target",
        "dynamic_convergence_alpha",
        "depth_antialias_strength",
        "edge_dilation",
        "edge_threshold",
        "mask_feather_radius",
        "hole_fill_mode",
        "hole_fill_radius",
        "hole_fill_strength",
        "screen_edge_mask_suppression",
        "cross_eyed",
        "anaglyph_method",
        "color_brightness",
        "color_contrast",
        "color_saturation",
        "color_gamma",
        "color_temperature",
        "color_tint",
        "output_headset_tier_k",
        "fused",
        "debug_output",
        "depth_backend",
        "model_id",
        "export_height",
        "export_width",
        "profile_sync",
        "use_cuda_graph",
        "device",
    }
)


@dataclass(frozen=True)
class RuntimeSettingsSnapshot:
    version: int
    timestamp: float
    source: str | None = None
    language: str | None = None
    application_runtime_target: str | None = None
    runtime_quality_mode: str | None = None
    stereo_synthesis_mode: str | None = None
    render_size_policy: str | None = None
    stereo_render_scale: str | None = None
    output_transport: str | None = None
    capture_source: str | None = None
    capture_target: str | None = None
    presentation_flags: dict[str, Any] | None = None
    debug_flags: dict[str, Any] | None = None
    debug_output: bool | None = None
    stereo_quality: StereoQuality | None = None
    stereo_preset: str | None = None
    output_format: OutputFormat | None = None
    depth_strength: float | None = None
    convergence: float | None = None
    max_disparity_px: float | None = None
    parallax_preset: str | None = None
    parallax_budget_preset: str | None = None
    temporal: bool | None = None
    temporal_enabled: bool | None = None
    temporal_strength: float | None = None
    auto_reset_temporal: bool | None = None
    scene_reset_threshold: float | None = None
    depth_pop: float | None = None
    foreground_shift_scale: float | None = None
    midground_shift_scale: float | None = None
    background_shift_scale: float | None = None
    dynamic_convergence_enabled: bool | None = None
    dynamic_convergence_strength: float | None = None
    dynamic_convergence_target: float | None = None
    dynamic_convergence_alpha: float | None = None
    depth_antialias_strength: float | None = None
    edge_dilation: int | None = None
    edge_threshold: float | None = None
    mask_feather_radius: int | None = None
    hole_fill_mode: str | None = None
    hole_fill_radius: int | None = None
    hole_fill_strength: float | None = None
    screen_edge_mask_suppression: int | None = None
    cross_eyed: bool | None = None
    anaglyph_method: str | None = None
    color_brightness: float | None = None
    color_contrast: float | None = None
    color_saturation: float | None = None
    color_gamma: float | None = None
    color_temperature: float | None = None
    color_tint: float | None = None
    vulkan_projection_min_lod: float | None = None
    vulkan_projection_max_lod: float | None = None
    vulkan_projection_mip_lod_bias: float | None = None
    vulkan_projection_rcas_sharpness: float | None = None
    output_headset_tier_k: int | None = None
    fused: bool | None = None
    depth_backend: str | None = None
    model_id: str | None = None
    export_height: int | None = None
    export_width: int | None = None
    profile_sync: bool | None = None
    use_cuda_graph: bool | None = None
    device: str | None = None

    def classify(self) -> SnapshotChangeClass:
        if self._has_any(_SESSION_RESTART_FIELDS):
            return SnapshotChangeClass.SESSION_RESTART
        if self._has_any(_PIPELINE_REBUILD_FIELDS):
            return SnapshotChangeClass.PIPELINE_REBUILD
        if self._has_any(_HOT_RELOAD_FIELDS):
            return SnapshotChangeClass.HOT_RELOAD
        return SnapshotChangeClass.NO_CHANGE

    def to_config_updates(self) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        for field_name in _CONFIG_UPDATE_FIELDS:
            value = getattr(self, field_name)
            if value is not None:
                updates[field_name] = value
        if self.parallax_budget_preset is not None and self.parallax_preset is None:
            updates["parallax_preset"] = self.parallax_budget_preset
        if self.temporal_enabled is not None and self.temporal is None:
            updates["temporal"] = self.temporal_enabled
        if self.hole_fill_mode is not None:
            mode = str(self.hole_fill_mode).strip().lower()
            updates["hole_fill"] = "none" if mode == "none" else "edge_aware"
        if self.runtime_quality_mode is not None:
            updates["mode"] = _normalize_runtime_mode(self.runtime_quality_mode)
        if self.vulkan_projection_min_lod is not None:
            updates["output_min_lod"] = self.vulkan_projection_min_lod
        if self.vulkan_projection_max_lod is not None:
            updates["output_max_lod"] = self.vulkan_projection_max_lod
        if self.vulkan_projection_mip_lod_bias is not None:
            updates["output_mip_lod_bias"] = self.vulkan_projection_mip_lod_bias
        if self.vulkan_projection_rcas_sharpness is not None:
            updates["output_rcas_sharpness"] = self.vulkan_projection_rcas_sharpness
        return updates

    def _has_any(self, field_names: frozenset[str]) -> bool:
        return any(getattr(self, field_name) is not None for field_name in field_names)


class RuntimeSettingsRestartRequired(RuntimeError):
    def __init__(self, snapshot: RuntimeSettingsSnapshot):
        super().__init__(f"Runtime settings snapshot {snapshot.version} requires session restart")
        self.snapshot = snapshot


class RuntimeSettingsPipelineRebuildRequired(RuntimeError):
    def __init__(self, snapshot: RuntimeSettingsSnapshot, changed_fields: tuple[str, ...]):
        fields = ", ".join(changed_fields)
        super().__init__(f"Runtime settings snapshot {snapshot.version} requires pipeline rebuild: {fields}")
        self.snapshot = snapshot
        self.changed_fields = changed_fields
