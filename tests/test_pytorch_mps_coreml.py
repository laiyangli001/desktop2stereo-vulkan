"""CoreML provider plumbing tests (ported macOS behavior)."""

from types import SimpleNamespace

from stereo_runtime.depth_provider import (
    DepthProviderConfig,
    DepthProviderInfo,
    GenericAutoDepthProvider,
    create_depth_provider,
)
from stereo_runtime.providers.apple.pytorch_mps import (
    CoreMLEngine,
    GenericAutoDepthMpsProvider,
    create_pytorch_mps_provider,
    is_coreml_available,
)


def test_is_coreml_available_returns_bool() -> None:
    assert isinstance(is_coreml_available(), bool)


def test_generic_provider_accepts_coreml_flags() -> None:
    provider = GenericAutoDepthMpsProvider(
        model_id="xingyang1/Distill-Any-Depth-Small-hf",
        device="mps",
        cache_dir="/tmp/d2s-models",
        local_files_only=True,
        use_coreml=True,
        recompile_coreml=False,
    )
    assert provider.use_coreml is True
    assert provider.recompile_coreml is False
    assert CoreMLEngine.__module__ == "stereo_runtime.providers.apple.pytorch_mps"


def test_create_pytorch_mps_provider_forwards_coreml_flags() -> None:
    provider = create_pytorch_mps_provider(
        model_id="xingyang1/Distill-Any-Depth-Small-hf",
        device="mps",
        cache_dir="/tmp/d2s-models",
        local_files_only=True,
        depth_upsample="bilinear",
        depth_upsample_edge_strength=0.35,
        use_coreml=True,
        recompile_coreml=True,
    )
    assert isinstance(provider, GenericAutoDepthMpsProvider)
    assert provider.use_coreml is True
    assert provider.recompile_coreml is True


def test_coreml_load_does_not_preload_transformers(monkeypatch) -> None:
    """A cached CoreML model must not load the PyTorch model at startup."""
    import stereo_runtime.providers.apple.pytorch_mps as mps_module

    monkeypatch.setattr(mps_module.sys, "platform", "darwin")

    def fail_transformers_load(_provider):
        raise AssertionError("CoreML startup must not load Transformers")

    monkeypatch.setattr(
        GenericAutoDepthProvider, "load", fail_transformers_load
    )
    provider = create_pytorch_mps_provider(
        model_id="xingyang1/Distill-Any-Depth-Small-hf",
        device="cpu",
        cache_dir="/tmp/d2s-models",
        local_files_only=True,
        use_coreml=True,
    )

    assert isinstance(provider, GenericAutoDepthMpsProvider)
    assert provider.load() is None


def test_depth_provider_config_carries_coreml_fields() -> None:
    cfg = DepthProviderConfig(
        model_id="xingyang1/Distill-Any-Depth-Small-hf",
        device="mps",
        cache_dir="/tmp/d2s-models",
        local_files_only=True,
        use_coreml=True,
        recompile_coreml=False,
    )
    assert cfg.use_coreml is True
    assert cfg.recompile_coreml is False


def test_create_depth_provider_mps_backend_returns_mps_provider() -> None:
    cfg = DepthProviderConfig(
        model_id="xingyang1/Distill-Any-Depth-Small-hf",
        device="mps",
        cache_dir="/tmp/d2s-models",
        local_files_only=True,
        use_coreml=False,
    )
    provider = create_depth_provider(cfg)
    assert isinstance(provider, GenericAutoDepthMpsProvider)
    assert provider.use_coreml is False


def test_coreml_provider_info_reports_active_backend_without_fallback():
    provider = object.__new__(GenericAutoDepthMpsProvider)
    provider.info = DepthProviderInfo(
        provider="test",
        model_name="test",
        model_id="test",
        depth_resolution=336,
        cache_dir="/tmp",
    )

    provider._set_coreml_info()

    assert provider.info.depth_backend == "coreml"
    assert provider.info.runtime == "coreml"
    assert provider.info.execution_provider == "Apple Core ML"
    assert provider.info.fallback_reason is None


def test_coreml_engine_sanitizes_nonfinite_output():
    import numpy as np
    import pytest
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("requires MPS")

    engine = object.__new__(CoreMLEngine)
    engine.device = torch.device("mps")
    engine.model = SimpleNamespace(
        predict=lambda _inputs: {
            "depth": np.array([[[np.nan, np.inf], [-np.inf, 0.5]]], dtype=np.float32)
        }
    )

    output = engine(torch.zeros(1, 3, 2, 2))

    assert output.nonfinite_count == 3
    assert output.finite_depth is True
    assert bool(torch.isfinite(output.predicted_depth).all())
