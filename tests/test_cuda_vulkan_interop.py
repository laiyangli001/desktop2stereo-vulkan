import ctypes
import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from app_runtime.gpu_producer import (
    GpuProducerAdapter,
    GpuProducerUnavailableError,
    _ADAPTER_FACTORIES,
    create_gpu_producer_adapter,
)
from app_runtime.runtime_output import (
    CudaVulkanOutputAdapter,
    RocmVulkanOutputAdapter,
)
from viewer.cuda_vulkan_interop import (
    CudaVulkanImageImporter,
    _ExternalMemoryBufferDesc,
    _ExternalSemaphoreWaitParams,
    _ExternalSemaphoreSignalParams,
    _SemaphoreSignalParams,
)
from viewer.vulkan_context import VulkanContext, VulkanContextConfig
from viewer.vulkan_resources import VulkanExportableImage, VulkanExportableSemaphore
from viewer.vulkan_resources import VulkanImageResource


def test_cuda_external_semaphore_signal_params_match_runtime_abi() -> None:
    # CUDA's v2 signal parameter layout is 72 bytes of nested params,
    # followed by flags and sixteen reserved uint32 values.
    assert _SemaphoreSignalParams.fence_value.offset == 0
    assert _SemaphoreSignalParams.nv_sci_sync.offset == 8
    assert _SemaphoreSignalParams.keyed_mutex_key.offset == 16
    assert _SemaphoreSignalParams.reserved.offset == 24
    assert ctypes.sizeof(_SemaphoreSignalParams) == 72
    assert _ExternalSemaphoreSignalParams.params.offset == 0
    assert _ExternalSemaphoreSignalParams.flags.offset == 72
    assert ctypes.sizeof(_ExternalSemaphoreSignalParams) == 144
    assert _ExternalSemaphoreWaitParams.params.offset == 0
    assert _ExternalSemaphoreWaitParams.flags.offset == 72
    assert ctypes.sizeof(_ExternalSemaphoreWaitParams) == 144


def test_cuda_external_buffer_desc_matches_runtime_abi() -> None:
    assert _ExternalMemoryBufferDesc.offset.offset == 0
    assert _ExternalMemoryBufferDesc.size.offset == ctypes.sizeof(ctypes.c_size_t)
    assert _ExternalMemoryBufferDesc.flags.offset == ctypes.sizeof(ctypes.c_size_t) * 2
    # CUDA aligns the trailing uint32 field to the host size_t alignment.
    assert ctypes.sizeof(_ExternalMemoryBufferDesc) == 24


def test_cuda_external_semaphore_is_default_and_can_be_disabled(monkeypatch) -> None:
    monkeypatch.delenv("D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE", raising=False)
    assert CudaVulkanOutputAdapter._external_semaphore_requested()
    monkeypatch.setenv("D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE", "0")
    assert not CudaVulkanOutputAdapter._external_semaphore_requested()


def test_cuda_source_prepare_is_idempotent_for_reused_frame() -> None:
    adapter = CudaVulkanOutputAdapter.__new__(CudaVulkanOutputAdapter)
    adapter._prepared_source_eyes = set()
    adapter._source_frames = {
        7: (
            SimpleNamespace(resource=object()),
            SimpleNamespace(resource=object()),
            0,
        )
    }
    adapter.left_ready_semaphores = [SimpleNamespace(semaphore="left-ready")]
    adapter.right_ready_semaphores = [SimpleNamespace(semaphore="right-ready")]
    adapter.left_ready_values = [3]
    adapter.right_ready_values = [4]
    adapter.left_visible_semaphores = [SimpleNamespace(semaphore="left-visible")]
    adapter.right_visible_semaphores = [SimpleNamespace(semaphore="right-visible")]
    calls: list[object] = []
    re_signals: list[object] = []
    adapter.presenter = SimpleNamespace(
        vulkan=SimpleNamespace(
            prepare_external_image_for_sampling=lambda *_args, **_kwargs: calls.append(
                (_args, _kwargs)
            ),
            submit_on=lambda _role, _record, **_kwargs: re_signals.append(_kwargs),
        )
    )

    assert adapter.prepare_source_for_sampling(7, 0) == "left-visible"
    assert adapter.prepare_source_for_sampling(7, 0) is None
    assert len(calls) == 1
    assert calls[0][1]["wait_semaphore_value"] == 3
    assert re_signals == []


def test_cuda_timeline_semaphore_handle_types_match_runtime_enum() -> None:
    assert CudaVulkanImageImporter._CUDA_TIMELINE_SEMAPHORE_FD == 9
    assert CudaVulkanImageImporter._CUDA_TIMELINE_SEMAPHORE_WIN32 == 10
    timeline = SimpleNamespace(timeline=True)
    binary = SimpleNamespace(timeline=False)
    assert CudaVulkanImageImporter._semaphore_value(timeline, 7) == 7
    assert CudaVulkanImageImporter._semaphore_value(binary, 0) == 0
    with pytest.raises(RuntimeError, match="positive"):
        CudaVulkanImageImporter._semaphore_value(timeline, 0)
    with pytest.raises(RuntimeError, match="must be zero"):
        CudaVulkanImageImporter._semaphore_value(binary, 1)


def test_rocm_external_semaphore_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("D2S_ENABLE_ROCM_EXTERNAL_SEMAPHORE", raising=False)
    assert not RocmVulkanOutputAdapter._external_semaphore_requested()
    monkeypatch.setenv("D2S_ENABLE_ROCM_EXTERNAL_SEMAPHORE", "0")
    assert not RocmVulkanOutputAdapter._external_semaphore_requested()


def test_rocm_source_prepare_waits_for_hip_ready_semaphore() -> None:
    adapter = RocmVulkanOutputAdapter.__new__(RocmVulkanOutputAdapter)
    adapter._prepared_source_eyes = set()
    adapter._rocm_ready_pending = {(7, 0)}
    adapter._source_frames = {
        7: (
            SimpleNamespace(resource=object()),
            SimpleNamespace(resource=object()),
            0,
        )
    }
    adapter._buffer_frames = {
        7: (SimpleNamespace(), SimpleNamespace())
    }
    adapter._host_staging_enabled = False
    adapter.external_semaphore_enabled = True
    adapter.left_ready_semaphores = [SimpleNamespace(semaphore="left-ready")]
    adapter.right_ready_semaphores = [SimpleNamespace(semaphore="right-ready")]
    adapter.left_ready_values = [3]
    adapter.right_ready_values = [4]
    adapter.left_visible_semaphores = [SimpleNamespace(semaphore="left-visible")]
    adapter.right_visible_semaphores = [SimpleNamespace(semaphore="right-visible")]
    calls = []
    adapter.presenter = SimpleNamespace(
        vulkan=SimpleNamespace(
            copy_buffer_to_image=lambda *args, **kwargs: calls.append((args, kwargs))
        )
    )

    assert adapter.prepare_source_for_sampling(7, 0) == "left-visible"
    assert calls[0][1] == {
        "wait_semaphore": "left-ready",
        "wait_semaphore_value": 3,
        "signal_semaphore": "left-visible",
    }
    assert (7, 0) not in adapter._rocm_ready_pending


def test_cuda_output_adapter_implements_backend_neutral_gpu_contract() -> None:
    adapter = CudaVulkanOutputAdapter.__new__(CudaVulkanOutputAdapter)
    assert isinstance(adapter, GpuProducerAdapter)
    assert adapter.backend_name == "cuda"
    assert adapter.output_sync_mode == "gpu_synchronized"
    assert adapter.external_semaphore_sync_mode == "gpu_external_semaphore"


def test_gpu_producer_factory_selects_cuda_without_importing_vendor_api() -> None:
    adapter = create_gpu_producer_adapter(SimpleNamespace(), backend="cuda")
    assert isinstance(adapter, CudaVulkanOutputAdapter)


def test_gpu_producer_factory_rejects_unregistered_backend() -> None:
    assert _ADAPTER_FACTORIES["rocm"] is RocmVulkanOutputAdapter


def test_gpu_producer_factory_reports_unknown_backend() -> None:
    with pytest.raises(GpuProducerUnavailableError, match="unknown"):
        create_gpu_producer_adapter(SimpleNamespace(), backend="unknown")


def test_vulkan_output_slot_waits_for_consumer_release() -> None:
    adapter = CudaVulkanOutputAdapter.__new__(CudaVulkanOutputAdapter)
    adapter._lease_condition = threading.Condition()
    adapter._active_leases = {}
    adapter._closed = False
    adapter._claim_slot(0, 10)
    claimed = threading.Event()

    def claim_reused_slot() -> None:
        adapter._claim_slot(0, 11)
        claimed.set()

    worker = threading.Thread(target=claim_reused_slot)
    worker.start()
    assert not claimed.wait(0.05)
    adapter.release_frame(10)
    worker.join(timeout=1.0)
    assert claimed.is_set()
    adapter.release_frame(11)


def test_output_contract_publishes_actual_source_layout_and_queue_family() -> None:
    vk = SimpleNamespace(
        VK_IMAGE_LAYOUT_GENERAL=1,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL=2,
    )
    state = SimpleNamespace(layout=1, queue_family_index=4)
    context = SimpleNamespace(
        vk=vk,
        image_state=lambda _image: state,
    )
    resource = VulkanImageResource(
        context=context,
        image=object(),
        view=None,
        width=2,
        height=2,
        format=37,
        layout=1,
        access_mask=0,
        stage_mask=0,
        queue_family_index=4,
    )

    contract = GpuProducerAdapter.source_image_contract(resource)
    assert contract == {"layout": "general", "queue_family": 4}

    state.layout = 2
    assert CudaVulkanOutputAdapter._source_image_contract(resource)["layout"] == (
        "shader_read_only_optimal"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable")
def test_cuda_tensor_reaches_vulkan_output_slot_without_cpu_roundtrip(monkeypatch) -> None:
    # This test exercises the production CUDA timeline-semaphore path.
    monkeypatch.setenv("D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE", "1")
    required_extensions = tuple(
        dict.fromkeys(
            VulkanExportableImage.required_device_extensions()
            + VulkanExportableSemaphore.required_device_extensions()
        )
    )
    context = VulkanContext.create(
        VulkanContextConfig(
            required_device_extensions=required_extensions
        )
    )
    destination = None
    adapter = None
    try:
        presenter = SimpleNamespace(
            initialized=True,
            vulkan=context,
            source_ready_semaphore_available=True,
        )
        adapter = CudaVulkanOutputAdapter(presenter)
        result = SimpleNamespace(
            left_eye=torch.full((32, 48, 4), 7, dtype=torch.uint8, device="cuda"),
            right_eye=torch.full((32, 48, 4), 11, dtype=torch.uint8, device="cuda"),
            debug_info={"output_format": "openxr_full_synthesis_eyes"},
        )
        for frame_id in range(300):
            frame = adapter.convert(result, frame_id=frame_id, timestamp=1.0)
            visible = (
                adapter.prepare_source_for_sampling(frame_id, 0),
                adapter.prepare_source_for_sampling(frame_id, 1),
            )
            context.submit_on(
                "graphics",
                lambda _command_buffer: None,
                wait_semaphore=visible,
            )
            adapter.release_consumer_frame(frame_id)

        assert frame.left_eye.width == 48
        assert frame.left_eye.height == 32
        assert frame.left_eye.format == context.vk.VK_FORMAT_R8G8B8A8_SRGB
        assert len(adapter.left_slots) == 3
        assert len(adapter.right_slots) == 3
        assert frame.metadata["vulkan_output_ring_slot"] == 2
        assert frame.metadata["vulkan_output_ring_size"] == 3
        assert frame.metadata["vulkan_output_sync"] == "gpu_external_semaphore"
        assert frame.metadata["vulkan_external_semaphore_type"] == "timeline"
        assert adapter.left_ready_values == [100, 100, 100]
        assert adapter.right_ready_values == [100, 100, 100]
        assert adapter.left_release_values == [100, 100, 100]
        assert adapter.right_release_values == [100, 100, 100]
        assert context.image_state(frame.left_eye.image).layout == context.vk.VK_IMAGE_LAYOUT_GENERAL

        destination = VulkanExportableImage(
            context,
            48,
            32,
            label="cuda-output-destination",
            format=context.vk.VK_FORMAT_R8G8B8A8_SRGB,
        )
        timeline = context.copy_image(frame.left_eye, destination.resource)
        context.wait_idle()
        assert timeline > 0
    finally:
        if adapter is not None:
            adapter.close()
        if destination is not None:
            destination.close()
        context.close()
