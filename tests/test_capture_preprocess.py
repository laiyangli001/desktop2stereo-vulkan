import numpy as np
import pytest

from capture import capture_frame_to_rgb, prepare_rgb_for_depth_runtime


def test_capture_frame_to_rgb_converts_bgr_numpy():
    frame = np.array([[[10, 20, 30], [40, 50, 60]], [[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    rgb = capture_frame_to_rgb(frame, target_height=2)

    assert rgb.shape == (2, 2, 3)
    assert rgb[0, 0].tolist() == [30, 20, 10]


def test_capture_frame_to_rgb_converts_bgra_numpy():
    frame = np.array(
        [
            [[10, 20, 30, 255], [40, 50, 60, 128]],
            [[10, 20, 30, 255], [40, 50, 60, 128]],
        ],
        dtype=np.uint8,
    )

    rgb = capture_frame_to_rgb(frame, target_height=2)

    assert rgb.shape == (2, 2, 3)
    assert rgb[0, 0].tolist() == [30, 20, 10]


def test_capture_frame_to_rgb_keeps_even_resize_dimensions():
    frame = np.zeros((3, 5, 3), dtype=np.uint8)

    rgb = capture_frame_to_rgb(frame, target_height=5)

    assert rgb.shape[0] % 2 == 0
    assert rgb.shape[1] % 2 == 0


def test_capture_frame_to_rgb_tensor_path_records_cpu_device_metadata():
    torch = pytest.importorskip("torch")
    frame = np.array([[[10, 20, 30], [40, 50, 60]], [[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    tensor = capture_frame_to_rgb(frame, target_height=2, device="cpu", output="tensor")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 2, 2)
    assert tensor.device.type == "cpu"
    assert tensor[0, :, 0, 0].tolist() == pytest.approx([30 / 255, 20 / 255, 10 / 255])
    assert tensor._d2s_preprocess_backend == "torch_bgr_norm"
    assert tensor._d2s_preprocess_input_kind == "numpy"
    assert tensor._d2s_preprocess_device_origin == "cpu"
    assert tensor._d2s_preprocess_device_output == "cpu"
    assert tensor._d2s_preprocess_device_transfer == "cpu->cpu"


def test_capture_frame_to_rgb_tensor_path_records_capture_metadata_overrides():
    torch = pytest.importorskip("torch")
    frame = np.array([[[10, 20, 30], [40, 50, 60]], [[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    tensor = capture_frame_to_rgb(
        frame,
        target_height=2,
        device="cpu",
        output="tensor",
        frame_raw_device="cuda",
        capture_copy_mode="clone",
        capture_zero_copy=False,
    )

    assert isinstance(tensor, torch.Tensor)
    assert tensor._d2s_preprocess_device_origin == "cuda"
    assert tensor._d2s_preprocess_device_output == "cpu"
    assert tensor._d2s_preprocess_device_transfer == "cuda->cpu"
    assert tensor._d2s_capture_copy_mode == "clone"
    assert tensor._d2s_capture_zero_copy is False


@pytest.mark.skipif(
    not __import__("torch").backends.mps.is_available(),
    reason="requires MPS",
)
def test_mps_tensor_keeps_raw_cpu_capture_for_coreml_staging():
    torch = pytest.importorskip("torch")
    frame = np.zeros((4, 6, 4), dtype=np.uint8)

    tensor = capture_frame_to_rgb(frame, target_height=4, device="mps", output="tensor")

    raw = tensor._d2s_coreml_raw_bgr
    assert isinstance(raw, torch.Tensor)
    assert raw.device.type == "cpu"
    assert tuple(raw.shape) == (4, 6, 4)


def test_capture_frame_to_rgb_tensor_path_accepts_torch_hwc():
    torch = pytest.importorskip("torch")
    frame = torch.tensor([[[10, 20, 30], [40, 50, 60]], [[10, 20, 30], [40, 50, 60]]], dtype=torch.uint8)

    tensor = capture_frame_to_rgb(frame, (2, 2), device="cpu", output="tensor")

    assert tensor.shape == (1, 3, 2, 2)
    assert tensor._d2s_preprocess_input_kind == "torch.Tensor"
    assert tensor._d2s_preprocess_device_transfer == "cpu->cpu"


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps", "xpu", "hip"])
def test_same_torch_device_treats_default_device_as_device_zero(device_type):
    torch = pytest.importorskip("torch")
    from capture.preprocess import _same_torch_device

    assert _same_torch_device(torch.device(f"{device_type}:0"), torch.device(device_type), torch)
    assert not _same_torch_device(torch.device(f"{device_type}:1"), torch.device(device_type), torch)


def test_capture_frame_to_rgb_requires_one_target_resolution_argument():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="Provide exactly one"):
        capture_frame_to_rgb(frame)
    with pytest.raises(TypeError, match="Provide exactly one"):
        capture_frame_to_rgb(frame, 2, target_height=2)


def test_prepare_rgb_for_depth_runtime_accepts_numpy_hwc():
    torch = pytest.importorskip("torch")
    frame = np.full((2, 3, 3), 255, dtype=np.uint8)

    tensor = prepare_rgb_for_depth_runtime(frame, device="cpu")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 2, 3)
    assert float(tensor.max()) == 1.0


def test_prepare_rgb_for_depth_runtime_accepts_chw_and_bchw():
    torch = pytest.importorskip("torch")
    chw = torch.ones((3, 2, 4), dtype=torch.float32)
    bchw = torch.ones((1, 3, 2, 4), dtype=torch.float32)

    assert prepare_rgb_for_depth_runtime(chw, device="cpu").shape == (1, 3, 2, 4)
    assert prepare_rgb_for_depth_runtime(bchw, device="cpu").shape == (1, 3, 2, 4)


def test_prepare_rgb_for_depth_runtime_does_not_rescale_normalized_input():
    torch = pytest.importorskip("torch")
    frame = torch.full((3, 2, 2), 0.5, dtype=torch.float32)

    tensor = prepare_rgb_for_depth_runtime(frame, device="cpu")

    assert float(tensor.max()) == pytest.approx(0.5)
