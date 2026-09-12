import sys
from pathlib import Path

from path_config import APP_ROOT

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from stereo_runtime.model_artifacts import artifact_paths_for_model
from stereo_runtime.onnx_export import (
    _da3_preset,
    _is_da3_model,
    _load_da3_checkpoint,
    _quiet_onnx_export_warnings,
    choose_export_dtype,
    export_depth_model_onnx,
    probe_model_dtype,
)


def test_choose_export_dtype_auto_cuda_defaults_fp16():
    dtype, name, reason = choose_export_dtype("lc700x/Distill-Any-Depth-Base-hf", torch.device("cuda"), "auto")

    assert dtype == torch.float16
    assert name == "fp16"
    assert "CUDA" in reason


def test_choose_export_dtype_auto_cpu_uses_fp32():
    dtype, name, reason = choose_export_dtype("lc700x/Distill-Any-Depth-Base-hf", torch.device("cpu"), "auto")

    assert dtype == torch.float32
    assert name == "fp32"
    assert "non-CUDA" in reason


def test_choose_export_dtype_force_fp32_keyword():
    dtype, name, reason = choose_export_dtype("lc700x/InfiniDepth-Large", torch.device("cuda"), "auto")

    assert dtype == torch.float32
    assert name == "fp32"
    assert "requires fp32" in reason


def test_da3_models_use_bundled_preset_mapping():
    assert _is_da3_model("depth-anything/DA3-BASE") is True
    assert _da3_preset("depth-anything/DA3-BASE") == "da3-base"
    assert _da3_preset("depth-anything/DA3NESTED-GIANT-LARGE-1.1") == "da3nested-giant-large"
    assert _da3_preset("depth-anything/DA3METRIC-LARGE") == "da3metric-large"
    assert _da3_preset("depth-anything/DA3MONO-LARGE") == "da3mono-large"


def test_da3_safetensors_loader_handles_shared_layernorm_aliases(tmp_path):
    class TiedLayerNormModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            shared = torch.nn.LayerNorm(4)
            self.left = shared
            self.right = shared

    from safetensors.torch import save_model

    source = TiedLayerNormModel()
    with torch.no_grad():
        source.left.weight.fill_(2.0)
        source.left.bias.fill_(3.0)
    path = tmp_path / "tied.safetensors"
    save_model(source, str(path))

    target = TiedLayerNormModel()
    _load_da3_checkpoint(target, path, torch)

    assert torch.equal(target.left.weight, source.left.weight)
    assert torch.equal(target.right.bias, source.right.bias)


def test_default_output_path_uses_actual_dtype_name():
    path = artifact_paths_for_model("xingyang1/Distill-Any-Depth-Large-hf", cache_dir=ROOT / "models").onnx_path_for_dtype("fp16")

    assert path.name == "model_fp16_294x518.onnx"
    assert "models--xingyang1--Distill-Any-Depth-Large-hf" in str(path)


def test_probe_model_dtype_rejects_all_zero_output():
    class Output:
        predicted_depth = torch.zeros(1, 4, 4)

    class Model:
        def eval(self):
            return self

        def __call__(self, pixel_values):
            return Output()

    ok, reason = probe_model_dtype(Model(), device=torch.device("cpu"), dtype=torch.float32, height=4, width=4)

    assert ok is False
    assert "all zero" in reason


def test_probe_model_dtype_accepts_finite_dynamic_output():
    class Output:
        predicted_depth = torch.arange(16, dtype=torch.float32).view(1, 4, 4)

    class Model:
        def eval(self):
            return self

        def __call__(self, pixel_values):
            return Output()

    ok, reason = probe_model_dtype(Model(), device=torch.device("cpu"), dtype=torch.float32, height=4, width=4)

    assert ok is True
    assert "ok:" in reason


def test_quiet_onnx_export_warnings_only_suppresses_known_export_noise():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _quiet_onnx_export_warnings():
            warnings.warn(
                "Converting a tensor to a Python boolean might cause the trace to be incorrect",
                torch.jit.TracerWarning,
            )
            warnings.warn(
                "ONNX export mode is set to TrainingMode.EVAL, but operator 'instance_norm' is set to train=True. Exporting with train=True.",
                UserWarning,
            )
            warnings.warn("real warning", UserWarning)

    assert [str(item.message) for item in caught] == ["real warning"]


def test_explicit_fp16_export_falls_back_to_fp32_when_probe_fails(monkeypatch, tmp_path):
    calls = []

    def fake_load_model_for_dtype(*_args, dtype, **_kwargs):
        calls.append(dtype)
        return object()

    def fake_probe_model_dtype(_model, *, dtype, **_kwargs):
        if dtype == torch.float16:
            return False, "RuntimeError: Input type (float) and bias type (struct c10::Half) should be the same"
        return True, "ok"

    def fake_export(_model, _dummy_input, output_path, **_kwargs):
        Path(output_path).write_bytes(b"onnx")

    monkeypatch.setattr("stereo_runtime.onnx_export.load_model_for_dtype", fake_load_model_for_dtype)
    monkeypatch.setattr("stereo_runtime.onnx_export.probe_model_dtype", fake_probe_model_dtype)
    monkeypatch.setattr(torch.onnx, "export", fake_export)
    result = export_depth_model_onnx(
        model_id="test/model",
        output_path=tmp_path / "model_fp16_294x518.onnx",
        cache_dir=tmp_path,
        device="cpu",
        dtype="fp16",
    )

    assert calls == [torch.float16, torch.float32]
    assert result.dtype_name == "fp32"
    assert result.output_path.name == "model_fp32_294x518.onnx"
    assert "fp16 probe failed" in result.dtype_reason
