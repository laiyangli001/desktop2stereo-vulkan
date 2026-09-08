from __future__ import annotations

from pathlib import Path

import pytest

from path_config import APP_ROOT, PROJECT_ROOT

from app_runtime.probe import build_capability_report
from stereo_runtime import VulkanImageCopyPass as PublicVulkanImageCopyPass
from stereo_runtime.vulkan_graph import (
    VulkanComputeGraph,
    VulkanComputePass,
    VulkanGraphState,
    VulkanPassDeclaration,
    VulkanStereoSubmission,
)
from viewer.vulkan_compute_pipeline import (
    VulkanComputePipelineError,
    read_spirv_words,
)
from viewer.vulkan_descriptors import DescriptorBinding, DescriptorBudget
from stereo_runtime.vulkan_image_pass import VulkanImageCopyPass




def test_capability_report_identifies_new_project():
    report = build_capability_report()
    assert report["project"] == "desktop2steoro-vulkan"
    assert report["migration"]["python_vulkan_runtime"] == "phase1_implemented"
    assert report["migration"]["openxr_vulkan_session"] == "phase1_validated"
    assert report["gpu_producers"]["selection"] in {"auto", "override"}
    assert "selected_backend" in report["gpu_producers"]


def test_capability_report_marks_static_openxr_boundary_and_runtime_size():
    report = build_capability_report()

    assert isinstance(report["openxr"]["extensions"], list)
    assert report["runtime_configuration"]["fallback_status"]["session_created"] is False
    assert report["runtime_configuration"]["fallback_status"]["gpu_to_cpu"] == "not_probed"
    assert report["runtime_configuration"]["openxr"]["target_size"] == "session_required"


def test_current_style_source_layout_is_present():
    expected = [
        APP_ROOT / "main.py",
        APP_ROOT / "app_runtime",
        APP_ROOT / "capture",
        APP_ROOT / "gui",
        APP_ROOT / "stereo_runtime",
        APP_ROOT / "viewer",
        APP_ROOT / "xr_viewer",
        PROJECT_ROOT / "native/filament/bridge/CMakeLists.txt",
        PROJECT_ROOT / ".github/workflows/filament-bridge.yml",
    ]
    for path in expected:
        assert path.exists(), path


def test_forbidden_legacy_runtime_directories_are_not_migrated():
    forbidden = [
        APP_ROOT / "xr_viewer/panda_runtime",
        APP_ROOT / "capture/dxgi/native",
    ]
    for path in forbidden:
        assert not path.exists(), path


def test_vulkan_submission_contract_is_python_native():
    submission = VulkanStereoSubmission(
        frame_id=7,
        rgb_handle=object(),
        depth_handle=object(),
        config_version=3,
    )
    assert submission.frame_id == 7
    assert VulkanGraphState.CREATED.value == "created"


def test_vulkan_compute_graph_submits_latest_frame_to_compute_queue():
    calls = []

    class FakeContext:
        def submit_on(self, role, record):
            command_buffer = object()
            record(command_buffer)
            calls.append(role)
            return 12

    def record_pass(command_buffer, submission):
        calls.append((command_buffer, submission.frame_id))

    graph = VulkanComputeGraph(FakeContext(), record_pass)
    graph.enqueue(VulkanStereoSubmission(1, object(), object(), 1))
    graph.enqueue(VulkanStereoSubmission(2, object(), object(), 1))
    assert graph.flush() == 12
    assert calls[0][1] == 2
    assert calls[-1] == "compute"


def test_vulkan_compute_graph_overload_keeps_only_latest_frame():
    submitted = []

    class FakeContext:
        def submit_on(self, role, record):
            record("command-buffer")
            return 1

    def record_pass(_command_buffer, submission):
        submitted.append(submission.frame_id)

    graph = VulkanComputeGraph(FakeContext(), record_pass)
    for frame_id in range(1000):
        graph.enqueue(VulkanStereoSubmission(frame_id, object(), object(), 1))

    assert graph.flush() == 1
    assert submitted == [999]


def test_vulkan_compute_graph_from_pipeline_records_dispatch():
    calls = []

    class FakeContext:
        def submit_on(self, role, record):
            record("command-buffer")
            calls.append(role)
            return 4

    class FakePipeline:
        def record_dispatch(self, command_buffer, **counts):
            calls.append((command_buffer, counts))

    graph = VulkanComputeGraph.from_pipeline(
        FakeContext(), FakePipeline(), group_counts=(2, 3, 1)
    )
    assert graph.submit(VulkanStereoSubmission(1, object(), object(), 1)) == 4
    assert calls == [
        ("command-buffer", {
            "group_count_x": 2,
            "group_count_y": 3,
            "group_count_z": 1,
        }),
        "compute",
    ]


def test_vulkan_compute_graph_forwards_input_ready_timeline():
    calls = []

    class FakeContext:
        def submit_on(self, role, record, **kwargs):
            record("command-buffer")
            calls.append((role, kwargs))
            return 9

    graph = VulkanComputeGraph(FakeContext(), lambda *_: None)
    assert graph.submit(
        VulkanStereoSubmission(3, object(), object(), 1, ready_timeline=7)
    ) == 9
    assert calls == [("compute", {"wait_for_timeline": 7})]


def test_vulkan_compute_graph_from_passes_inserts_barrier_between_passes():
    calls = []

    class FakeVk:
        VK_STRUCTURE_TYPE_MEMORY_BARRIER = 1
        VK_ACCESS_SHADER_WRITE_BIT = 2
        VK_ACCESS_SHADER_READ_BIT = 4
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 8

        @staticmethod
        def VkMemoryBarrier(**kwargs):
            return kwargs

        @staticmethod
        def vkCmdPipelineBarrier(*args):
            calls.append(args)

    class FakeContext:
        vk = FakeVk()

        def submit_on(self, role, record, **kwargs):
            assert role == "compute"
            record("command-buffer")
            return 11

    class FakePipeline:
        def __init__(self, name):
            self.name = name

        def record_dispatch(self, command_buffer, **kwargs):
            calls.append((self.name, command_buffer, kwargs))

    graph = VulkanComputeGraph.from_passes(
        FakeContext(),
        (
            VulkanComputePass(
                VulkanPassDeclaration("normalize", writes=("normalized",)),
                FakePipeline("a"),
            ),
            VulkanComputePass(VulkanPassDeclaration("warp", reads=("normalized",)), FakePipeline("b")),
        ),
    )
    assert graph.submit(VulkanStereoSubmission(1, object(), object(), 1)) == 11
    assert calls[0][0] == "a"
    assert calls[-1][0] == "b"
    assert any(call[0] == "command-buffer" for call in calls[1:-1])


def test_vulkan_compute_graph_close_rejects_new_work():
    graph = VulkanComputeGraph(type("Context", (), {})(), lambda *_: None)
    graph.close()
    with pytest.raises(RuntimeError, match="not ready"):
        graph.enqueue(VulkanStereoSubmission(1, object(), object(), 1))


def test_vulkan_image_copy_pass_binds_images_and_uses_ceiled_workgroups():
    assert PublicVulkanImageCopyPass is VulkanImageCopyPass

    class FakeVk:
        VK_IMAGE_LAYOUT_GENERAL = 7

    class FakeContext:
        vk = FakeVk()
        compute_queue_family_index = 3

        @staticmethod
        def image_state(_image):
            return type("State", (), {"layout": 7, "queue_family_index": 3})()

    class FakeGraph:
        def submit(self, submission):
            self.submission = submission
            return 18

    class FakeArena:
        def __init__(self):
            self.updates = []

        def update_storage_image(self, descriptor_set, binding, image):
            self.updates.append((descriptor_set, binding, image))

    image_pass = object.__new__(VulkanImageCopyPass)
    image_pass.context = FakeContext()
    image_pass.width = 17
    image_pass.height = 9
    image_pass.graph = FakeGraph()
    image_pass.descriptor_arena = FakeArena()
    image_pass.descriptor_set = "set-0"
    source = type(
        "Image", (), {"context": image_pass.context, "width": 17, "height": 9, "image": "source"}
    )()
    output = type(
        "Image", (), {"context": image_pass.context, "width": 17, "height": 9, "image": "output"}
    )()

    assert image_pass.group_counts == (3, 2, 1)
    assert image_pass.submit(source, output, frame_id=4, config_version=2) == 18
    assert [item[1] for item in image_pass.descriptor_arena.updates] == [0, 1]
    assert image_pass.graph.submission.rgb_handle == "source"
    assert image_pass.graph.submission.depth_handle == "output"


def test_vulkan_image_copy_pass_rejects_non_general_images():
    class FakeVk:
        VK_IMAGE_LAYOUT_GENERAL = 7

    class FakeContext:
        vk = FakeVk()
        compute_queue_family_index = 3

        @staticmethod
        def image_state(_image):
            return type("State", (), {"layout": 1, "queue_family_index": 3})()

    image_pass = object.__new__(VulkanImageCopyPass)
    image_pass.context = FakeContext()
    image_pass.width = 1
    image_pass.height = 1
    image_pass.graph = object()
    image_pass.descriptor_arena = object()
    image_pass.descriptor_set = "set-0"
    image = type(
        "Image", (), {"context": image_pass.context, "width": 1, "height": 1, "image": "image-a"}
    )()
    output = type(
        "Image", (), {"context": image_pass.context, "width": 1, "height": 1, "image": "image-b"}
    )()

    with pytest.raises(RuntimeError, match="GENERAL layout"):
        image_pass.submit(image, output, frame_id=1, config_version=1)


def test_vulkan_image_copy_pass_rejects_cross_context_images():
    image_pass = object.__new__(VulkanImageCopyPass)
    image_pass.context = object()
    image_pass.width = 1
    image_pass.height = 1
    image_pass.graph = object()
    image_pass.descriptor_arena = object()
    image_pass.descriptor_set = "set-0"
    image = type(
        "Image", (), {"context": object(), "width": 1, "height": 1, "image": "image-a"}
    )()
    output = type(
        "Image", (), {"context": object(), "width": 1, "height": 1, "image": "image-b"}
    )()

    with pytest.raises(RuntimeError, match="different Vulkan context"):
        image_pass.submit(image, output, frame_id=1, config_version=1)


def test_vulkan_image_copy_pass_rejects_source_output_alias():
    context = object()
    image_pass = object.__new__(VulkanImageCopyPass)
    image_pass.context = context
    image_pass.width = 1
    image_pass.height = 1
    image_pass.graph = object()
    image_pass.descriptor_arena = object()
    image_pass.descriptor_set = "set-0"
    image = type("Image", (), {"context": context, "width": 1, "height": 1, "image": object()})()

    with pytest.raises(RuntimeError, match="must be distinct"):
        image_pass.submit(image, image, frame_id=1, config_version=1)


def test_spirv_loader_validates_magic_and_word_alignment(tmp_path):
    shader = tmp_path / "noop.spv"
    shader.write_bytes((0x07230203).to_bytes(4, "little") + b"\0" * 4)
    assert read_spirv_words(shader) == [0x07230203, 0]

    invalid = tmp_path / "invalid.spv"
    invalid.write_bytes(b"bad")
    with pytest.raises(VulkanComputePipelineError, match="32-bit"):
        read_spirv_words(invalid)


def test_descriptor_budget_is_bounded():
    budget = DescriptorBudget(max_sets=4, storage_buffers_per_set=2)
    assert budget.max_sets == 4
    with pytest.raises(ValueError, match="at least one"):
        DescriptorBudget(max_sets=0)


def test_descriptor_binding_requires_positive_count():
    binding = DescriptorBinding(binding=0, descriptor_type=7)
    assert binding.descriptor_count == 1
    with pytest.raises(ValueError, match="positive"):
        DescriptorBinding(binding=0, descriptor_type=7, descriptor_count=0)
