from __future__ import annotations

import pytest
import torch

from stereo_runtime import StereoRuntime, StereoRuntimeConfig
from stereo_runtime.depth_provider import DepthProfileResult
from stereo_runtime.settings_snapshot import (
    RuntimeSettingsPipelineRebuildRequired,
    RuntimeSettingsRestartRequired,
    RuntimeSettingsSnapshot,
    SnapshotChangeClass,
)


class FakeDepthProvider:
    def __init__(self):
        self.closed = False
        self.loaded = False

    def load(self):
        self.loaded = True

    def predict_profile(self, rgb_frame):
        height = int(rgb_frame.shape[-2])
        width = int(rgb_frame.shape[-1])
        depth = torch.zeros((1, 1, height, width), dtype=torch.float32, device=rgb_frame.device)
        return DepthProfileResult(depth=depth, preprocess_ms=0.0, model_ms=0.0, postprocess_ms=0.0)

    def close(self):
        self.closed = True


def test_settings_snapshot_classifies_change_levels():
    assert RuntimeSettingsSnapshot(version=1, timestamp=1.0).classify() is SnapshotChangeClass.NO_CHANGE
    assert RuntimeSettingsSnapshot(version=2, timestamp=1.0, temporal_strength=0.5).classify() is SnapshotChangeClass.HOT_RELOAD
    assert RuntimeSettingsSnapshot(version=2, timestamp=1.0, language="CN").classify() is SnapshotChangeClass.HOT_RELOAD
    assert RuntimeSettingsSnapshot(version=3, timestamp=1.0, depth_backend="pytorch_cuda").classify() is SnapshotChangeClass.PIPELINE_REBUILD
    assert RuntimeSettingsSnapshot(version=4, timestamp=1.0, profile_sync=True).classify() is SnapshotChangeClass.PIPELINE_REBUILD
    assert RuntimeSettingsSnapshot(version=4, timestamp=1.0, device="cuda:1").classify() is SnapshotChangeClass.SESSION_RESTART


def test_settings_snapshot_classifies_spec_layer_fields():
    assert (
        RuntimeSettingsSnapshot(version=5, timestamp=1.0, runtime_quality_mode="movie").classify()
        is SnapshotChangeClass.HOT_RELOAD
    )
    assert (
        RuntimeSettingsSnapshot(version=6, timestamp=1.0, render_size_policy="scaled").classify()
        is SnapshotChangeClass.PIPELINE_REBUILD
    )
    assert (
        RuntimeSettingsSnapshot(version=7, timestamp=1.0, stereo_render_scale="1K / 50%").classify()
        is SnapshotChangeClass.PIPELINE_REBUILD
    )
    assert (
        RuntimeSettingsSnapshot(version=8, timestamp=1.0, stereo_synthesis_mode="full_synthesis_eyes").classify()
        is SnapshotChangeClass.PIPELINE_REBUILD
    )
    assert (
        RuntimeSettingsSnapshot(version=9, timestamp=1.0, application_runtime_target="openxr").classify()
        is SnapshotChangeClass.SESSION_RESTART
    )


def test_settings_snapshot_maps_runtime_quality_mode_to_runtime_config_mode():
    snapshot = RuntimeSettingsSnapshot(
        version=10,
        timestamp=1.0,
        runtime_quality_mode="game_low_latency",
        render_size_policy="scaled",
        stereo_render_scale="1K / 50%",
        stereo_synthesis_mode="full_synthesis_eyes",
        output_transport="openxr_swapchain",
        presentation_flags={"cross_eyed": True},
        debug_flags={"depth": True},
        depth_strength=1.25,
    )

    updates = snapshot.to_config_updates()

    assert updates == {"depth_strength": 1.25, "mode": "game"}


def test_settings_snapshot_does_not_expose_legacy_ipd_fields():
    fields = RuntimeSettingsSnapshot.__dataclass_fields__

    assert "ipd" not in fields
    assert "ipd_mm" not in fields
    assert "stereo_scale" not in fields
    assert "max_shift_ratio" not in fields


def test_settings_snapshot_maps_parallax_budget_fields():
    snapshot = RuntimeSettingsSnapshot(
        version=11,
        timestamp=1.0,
        max_disparity_px=96.0,
        parallax_preset="standard",
    )

    updates = snapshot.to_config_updates()

    assert snapshot.classify() is SnapshotChangeClass.HOT_RELOAD
    assert updates["max_disparity_px"] == 96.0
    assert updates["parallax_preset"] == "standard"


def test_settings_snapshot_maps_spec_alias_fields_to_runtime_config():
    snapshot = RuntimeSettingsSnapshot(
        version=16,
        timestamp=1.0,
        parallax_budget_preset="strong",
        temporal_enabled=False,
    )

    updates = snapshot.to_config_updates()

    assert snapshot.classify() is SnapshotChangeClass.HOT_RELOAD
    assert updates == {"parallax_preset": "strong", "temporal": False}


def test_runtime_applies_spec_alias_fields_and_tracks_them_in_result():
    runtime = StereoRuntime(
        StereoRuntimeConfig(model_id="Distill-Any-Depth-Base", cache_dir="models"),
        depth_provider=FakeDepthProvider(),
        collect_memory_stats=False,
    )
    runtime.temporal_state.left = torch.ones((1, 3, 2, 2), dtype=torch.float32)

    change_class = runtime.apply_settings_snapshot(
        RuntimeSettingsSnapshot(
            version=17,
            timestamp=1.0,
            parallax_budget_preset="strong",
            temporal_enabled=False,
        )
    )

    assert change_class is SnapshotChangeClass.HOT_RELOAD
    assert runtime.config.parallax_preset == "strong"
    assert runtime.config.temporal is False
    assert runtime.temporal_state.left is None

    result = runtime.process_openxr_frame(torch.zeros((1, 3, 2, 2), dtype=torch.float32))

    assert result.hot_reload_changed_fields == ("parallax_budget_preset", "temporal_enabled")
    assert result.debug_info["hot_reload_changed_fields"] == ["parallax_budget_preset", "temporal_enabled"]
    assert result.debug_info["parallax_budget_preset"] == "strong"
    assert result.debug_info["temporal_reset_reason"] == "settings_changed"


def test_settings_snapshot_maps_temporal_reset_fields():
    snapshot = RuntimeSettingsSnapshot(
        version=12,
        timestamp=1.0,
        auto_reset_temporal=True,
        scene_reset_threshold=0.33,
    )

    updates = snapshot.to_config_updates()

    assert snapshot.classify() is SnapshotChangeClass.HOT_RELOAD
    assert updates["auto_reset_temporal"] is True
    assert updates["scene_reset_threshold"] == 0.33
    assert ("reset_" + "cooldown" + "_frames") not in updates


def test_runtime_hot_switches_hole_fill_master_gate_with_mode() -> None:
    runtime = StereoRuntime(
        StereoRuntimeConfig(model_id="Distill-Any-Depth-Base", cache_dir="models", stereo_preset="cinema"),
        depth_provider=FakeDepthProvider(),
        collect_memory_stats=False,
    )

    runtime.apply_settings_snapshot(
        RuntimeSettingsSnapshot(
            version=20,
            timestamp=1.0,
            hole_fill_mode="none",
            hole_fill_radius=0,
            hole_fill_strength=0.0,
        )
    )
    assert runtime.config.hole_fill == "none"
    assert runtime.stereo_config.hole_fill == "none"

    runtime.apply_settings_snapshot(
        RuntimeSettingsSnapshot(
            version=21,
            timestamp=2.0,
            hole_fill_mode="quality",
            hole_fill_radius=3,
            hole_fill_strength=1.0,
        )
    )
    assert runtime.config.hole_fill == "edge_aware"
    assert runtime.stereo_config.hole_fill == "edge_aware"
    assert runtime.stereo_config.hole_fill_mode == "quality"
    assert runtime.stereo_config.hole_fill_radius == 3
    assert runtime.stereo_config.hole_fill_strength == pytest.approx(1.0)


def test_settings_snapshot_maps_profile_sync_as_provider_rebuild_config():
    snapshot = RuntimeSettingsSnapshot(version=15, timestamp=1.0, profile_sync=True)

    updates = snapshot.to_config_updates()

    assert snapshot.classify() is SnapshotChangeClass.PIPELINE_REBUILD
    assert updates == {"profile_sync": True}


def test_settings_snapshot_maps_cuda_graph_as_provider_rebuild_config():
    snapshot = RuntimeSettingsSnapshot(version=16, timestamp=1.0, use_cuda_graph=True)

    updates = snapshot.to_config_updates()

    assert snapshot.classify() is SnapshotChangeClass.PIPELINE_REBUILD
    assert updates == {"use_cuda_graph": True}


def test_runtime_applies_hot_settings_snapshot_without_rebuild():
    runtime = StereoRuntime(
        StereoRuntimeConfig(model_id="Distill-Any-Depth-Base", cache_dir="models", stereo_preset="cinema"),
        depth_provider=FakeDepthProvider(),
    )
    snapshot = RuntimeSettingsSnapshot(
        version=12,
        timestamp=1.0,
        depth_strength=1.5,
        temporal_strength=0.4,
        parallax_preset="strong",
    )

    change_class = runtime.apply_settings_snapshot(snapshot, active_preset="cinema")

    assert change_class is SnapshotChangeClass.HOT_RELOAD
    assert runtime.config.depth_strength == 1.5
    assert runtime.config.temporal_strength == 0.4
    assert runtime.config.parallax_preset == "strong"
    assert not hasattr(runtime.config, "ipd")
    assert runtime.active_settings_version == 12


def test_runtime_applies_runtime_quality_mode_snapshot():
    runtime = StereoRuntime(
        StereoRuntimeConfig(model_id="Distill-Any-Depth-Base", cache_dir="models", mode="movie"),
        depth_provider=FakeDepthProvider(),
        collect_memory_stats=False,
    )

    change_class = runtime.apply_settings_snapshot(
        RuntimeSettingsSnapshot(version=16, timestamp=1.0, runtime_quality_mode="Still Image / HQ")
    )

    assert change_class is SnapshotChangeClass.HOT_RELOAD
    assert runtime.config.mode == "image"
    assert runtime.stereo_config.backend == "quality_4k"
    assert runtime.active_settings_snapshot.runtime_quality_mode == "Still Image / HQ"


def test_runtime_settings_snapshot_temporal_change_resets_temporal_history():
    runtime = StereoRuntime(
        StereoRuntimeConfig(model_id="Distill-Any-Depth-Base", cache_dir="models"),
        depth_provider=FakeDepthProvider(),
        collect_memory_stats=False,
    )
    runtime.temporal_state.left = torch.ones((1, 3, 2, 2), dtype=torch.float32)
    runtime.temporal_state.right = torch.ones((1, 3, 2, 2), dtype=torch.float32)
    runtime._openxr_depth_temporal = torch.ones((1, 1, 2, 2), dtype=torch.float32)

    runtime.apply_settings_snapshot(RuntimeSettingsSnapshot(version=12, timestamp=1.0, temporal=False))

    assert runtime.temporal_state.left is None
    assert runtime.temporal_state.right is None
    assert runtime._openxr_depth_temporal is None

    result = runtime.process_openxr_frame(torch.zeros((1, 3, 2, 2), dtype=torch.float32))

    assert result.debug_info["temporal_reset_reason"] == "settings_changed"
    second = runtime.process_openxr_frame(torch.zeros((1, 3, 2, 2), dtype=torch.float32))
    assert "temporal_reset_reason" not in second.debug_info


def test_runtime_rejects_pipeline_rebuild_snapshot_before_merging_active_settings():
    runtime = StereoRuntime(
        StereoRuntimeConfig(model_id="Distill-Any-Depth-Base", cache_dir="models"),
        depth_provider=FakeDepthProvider(),
        collect_memory_stats=False,
    )
    snapshot = RuntimeSettingsSnapshot(version=12, timestamp=1.0, render_size_policy="scaled")

    with pytest.raises(RuntimeSettingsPipelineRebuildRequired) as exc_info:
        runtime.apply_settings_snapshot(snapshot)

    assert exc_info.value.changed_fields == ("render_size_policy",)
    assert runtime.active_settings_version == 0
    assert runtime.active_settings_snapshot.render_size_policy is None


def test_runtime_handles_profile_sync_with_depth_provider_rebuild(monkeypatch):
    created = []

    def fake_create_depth_provider(depth_config):
        provider = FakeDepthProvider()
        provider.depth_config = depth_config
        created.append(provider)
        return provider

    monkeypatch.setattr("stereo_runtime.runtime.create_depth_provider", fake_create_depth_provider)
    initial_provider = FakeDepthProvider()
    runtime = StereoRuntime(
        StereoRuntimeConfig(
            model_id="Distill-Any-Depth-Base",
            cache_dir="models",
            profile_sync=False,
        ),
        depth_provider=initial_provider,
        collect_memory_stats=False,
    )

    change_class = runtime.apply_settings_snapshot(
        RuntimeSettingsSnapshot(version=15, timestamp=1.0, profile_sync=True)
    )

    assert change_class is SnapshotChangeClass.PIPELINE_REBUILD
    assert runtime.config.profile_sync is True
    assert initial_provider.closed is True
    assert runtime.depth_provider is created[-1]
    assert runtime.depth_config.profile_sync is True
    assert runtime.active_settings_snapshot.profile_sync is True


def test_runtime_handles_cuda_graph_with_depth_provider_rebuild(monkeypatch):
    created = []

    def fake_create_depth_provider(depth_config):
        provider = FakeDepthProvider()
        provider.depth_config = depth_config
        created.append(provider)
        return provider

    monkeypatch.setattr("stereo_runtime.runtime.create_depth_provider", fake_create_depth_provider)
    initial_provider = FakeDepthProvider()
    runtime = StereoRuntime(
        StereoRuntimeConfig(
            model_id="Distill-Any-Depth-Base",
            cache_dir="models",
            use_cuda_graph=False,
        ),
        depth_provider=initial_provider,
        collect_memory_stats=False,
    )

    change_class = runtime.apply_settings_snapshot(
        RuntimeSettingsSnapshot(version=16, timestamp=1.0, use_cuda_graph=True)
    )

    assert change_class is SnapshotChangeClass.PIPELINE_REBUILD
    assert runtime.config.use_cuda_graph is True
    assert initial_provider.closed is True
    assert runtime.depth_provider is created[-1]
    assert runtime.depth_config.use_cuda_graph is True
    assert runtime.active_settings_snapshot.use_cuda_graph is True


def test_runtime_result_debug_info_tracks_active_settings_snapshot_fields():
    runtime = StereoRuntime(
        StereoRuntimeConfig(model_id="Distill-Any-Depth-Base", cache_dir="models"),
        depth_provider=FakeDepthProvider(),
        collect_memory_stats=False,
    )
    provider = runtime.depth_provider
    runtime.apply_settings_snapshot(
        RuntimeSettingsSnapshot(
            version=12,
            timestamp=1.0,
            runtime_quality_mode="movie",
            presentation_flags={"eye_order": "left_first"},
            debug_flags={"timing": True},
        )
    )
    assert runtime.depth_provider is provider
    runtime.apply_settings_snapshot(
        RuntimeSettingsSnapshot(
            version=13,
            timestamp=2.0,
            depth_strength=1.2,
            output_format="half_sbs",
            max_disparity_px=42.0,
            parallax_preset="strong",
            convergence=0.15,
            hole_fill_mode="balanced",
        )
    )

    result = runtime.process_openxr_frame(torch.zeros((1, 3, 2, 2), dtype=torch.float32))

    assert result.debug_info["active_settings_version"] == 13
    assert result.debug_info["hot_reload_class"] == SnapshotChangeClass.HOT_RELOAD.value
    assert result.debug_info["hot_reload_changed_fields"] == [
        "convergence",
        "depth_strength",
        "hole_fill_mode",
        "max_disparity_px",
        "output_format",
        "parallax_preset",
    ]
    assert result.debug_info["runtime_quality_mode"] == "movie"
    assert result.debug_info["presentation_flags"] == {"eye_order": "left_first"}
    assert result.debug_info["debug_flags"] == {"timing": True}
    assert result.debug_info["output_format"] == "half_sbs"
    assert result.debug_info["max_disparity_px"] == 42.0
    assert result.debug_info["parallax_preset"] == "strong"
    assert result.debug_info["convergence"] == 0.15
    assert result.debug_info["hole_fill_mode"] == "balanced"


def test_runtime_result_debug_info_tracks_active_settings_version():
    runtime = StereoRuntime(
        StereoRuntimeConfig(model_id="Distill-Any-Depth-Base", cache_dir="models"),
        depth_provider=FakeDepthProvider(),
        collect_memory_stats=False,
    )
    runtime.apply_settings_snapshot(RuntimeSettingsSnapshot(version=13, timestamp=1.0, depth_strength=1.2))

    result = runtime.process_openxr_frame(torch.zeros((1, 3, 2, 2), dtype=torch.float32))

    assert result.debug_info["active_settings_version"] == 13
    assert result.debug_info["hot_reload_class"] == SnapshotChangeClass.HOT_RELOAD.value
    assert result.debug_info["hot_reload_changed_fields"] == ["depth_strength"]


def test_runtime_rejects_session_restart_snapshot():
    runtime = StereoRuntime(
        StereoRuntimeConfig(model_id="Distill-Any-Depth-Base", cache_dir="models"),
        depth_provider=FakeDepthProvider(),
    )
    snapshot = RuntimeSettingsSnapshot(version=14, timestamp=1.0, device="cuda:1")

    with pytest.raises(RuntimeSettingsRestartRequired):
        runtime.apply_settings_snapshot(snapshot)
