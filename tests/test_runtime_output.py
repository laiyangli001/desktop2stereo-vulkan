from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

from app_runtime.runtime_output import (
    CudaVulkanOutputAdapter,
    RocmVulkanOutputAdapter,
    VulkanRuntimeOutputConsumer,
)
from viewer.vulkan_context import VulkanTimelineTimeout


def test_cuda_runtime_ready_event_uses_gpu_wait_without_host_sync():
    waits = []
    event = SimpleNamespace(wait=lambda: waits.append("gpu"))
    adapter = object.__new__(CudaVulkanOutputAdapter)
    adapter.backend_name = "cuda"

    adapter._wait_for_runtime_cuda_event(
        SimpleNamespace(cuda_ready_event=event),
        SimpleNamespace(device=None),
    )

    assert waits == ["gpu"]


def test_rocm_runtime_ready_event_is_not_touched_by_cuda_wait():
    waits = []
    event = SimpleNamespace(wait=lambda: waits.append("unexpected"))
    adapter = object.__new__(CudaVulkanOutputAdapter)
    adapter.backend_name = "rocm"

    adapter._wait_for_runtime_cuda_event(
        SimpleNamespace(cuda_ready_event=event),
        SimpleNamespace(device=None),
    )

    assert waits == []


def test_cuda_consumer_release_discards_cpu_lease_after_device_loss():
    calls = []

    class LostContext:
        device_lost = True

        @staticmethod
        def release_external_image_from_sampling(*_args, **_kwargs):
            raise AssertionError("dead Vulkan device must not be accessed")

    adapter = object.__new__(CudaVulkanOutputAdapter)
    adapter.presenter = SimpleNamespace(vulkan=LostContext())
    adapter._released_source_frames = set()
    adapter._source_frames = {7: (SimpleNamespace(), SimpleNamespace(), 0)}
    adapter._prepared_source_eyes = {(7, 0), (7, 1)}
    adapter.release_frame = lambda frame_id: calls.append(frame_id)

    adapter.release_consumer_frame(7, ("left", "right"))

    assert calls == [7]
    assert adapter._prepared_source_eyes == set()
    assert adapter._source_frames == {}
    assert adapter._released_source_frames == {7}


def test_cuda_consumer_release_uses_filament_completion_timeline_once():
    submissions = []

    class Context:
        device_lost = False

        @staticmethod
        def release_external_image_from_sampling(resource, **kwargs):
            submissions.append((resource, kwargs))
            return 40 + len(submissions)

    adapter = object.__new__(CudaVulkanOutputAdapter)
    adapter.presenter = SimpleNamespace(vulkan=Context())
    adapter._released_source_frames = set()
    adapter._prepared_source_eyes = {(7, 0), (7, 1)}
    adapter._release_signaled = set()
    adapter.left_release_values = [2]
    adapter.right_release_values = [4]
    adapter._source_frames = {
        7: (
            SimpleNamespace(resource="left"),
            SimpleNamespace(resource="right"),
            0,
        )
    }
    adapter.left_release_semaphores = [SimpleNamespace(semaphore="left-release")]
    adapter.right_release_semaphores = [SimpleNamespace(semaphore="right-release")]
    adapter.release_frame = lambda _frame_id: None

    adapter.release_consumer_frame(7, wait_for_timeline=39)

    assert submissions == [
        (
            "left",
            {
                "wait_for_timeline": 39,
                "wait_semaphore": None,
                "signal_semaphore": "left-release",
                "signal_semaphore_value": 3,
            },
        ),
        (
            "right",
            {
                "wait_for_timeline": 39,
                "wait_semaphore": None,
                "signal_semaphore": "right-release",
                "signal_semaphore_value": 5,
            },
        ),
    ]
    assert adapter.left_release_values == [3]
    assert adapter.right_release_values == [5]


def test_cuda_consumer_release_without_prepared_eyes_needs_no_semaphore_slots():
    released = []
    adapter = object.__new__(CudaVulkanOutputAdapter)
    adapter.presenter = SimpleNamespace(vulkan=SimpleNamespace(device_lost=False))
    adapter._released_source_frames = set()
    adapter._prepared_source_eyes = set()
    adapter._release_signaled = set()
    adapter.left_release_semaphores = []
    adapter.right_release_semaphores = []
    adapter.left_release_values = []
    adapter.right_release_values = []
    adapter._source_frames = {
        7: (
            SimpleNamespace(resource="left"),
            SimpleNamespace(resource="right"),
            2,
        )
    }
    adapter.release_frame = released.append

    adapter.release_consumer_frame(7)

    assert released == [7]
    assert adapter._source_frames == {}
    assert adapter._released_source_frames == {7}


def test_cuda_synchronized_copy_tracks_vulkan_release_timeline():
    released = []
    submissions = []

    class Context:
        device_lost = False

        @staticmethod
        def release_external_image_from_sampling(resource, **kwargs):
            submissions.append((resource, kwargs))
            return 40 if resource == "left" else 42

    adapter = object.__new__(CudaVulkanOutputAdapter)
    adapter.backend_name = "cuda"
    adapter.presenter = SimpleNamespace(vulkan=Context())
    adapter._released_source_frames = set()
    adapter._prepared_source_eyes = {(7, 0), (7, 1)}
    adapter._release_signaled = set()
    adapter._cuda_release_timelines = [0]
    adapter.left_release_semaphores = []
    adapter.right_release_semaphores = []
    adapter._source_frames = {
        7: (
            SimpleNamespace(resource="left"),
            SimpleNamespace(resource="right"),
            0,
        )
    }
    adapter.release_frame = released.append

    adapter.release_consumer_frame(
        7,
        ("left-wait", "right-wait"),
        wait_for_timeline=39,
    )

    assert adapter._cuda_release_timelines == [42]
    assert released == [7]
    assert submissions == [
        (
            "left",
            {
                "wait_for_timeline": 39,
                "wait_semaphore": "left-wait",
            },
        ),
        (
            "right",
            {
                "wait_for_timeline": 39,
                "wait_semaphore": "right-wait",
            },
        ),
    ]


def test_rocm_synchronized_copy_tracks_vulkan_release_timeline():
    released = []
    submissions = []

    class Context:
        device_lost = False

        @staticmethod
        def release_external_image_from_sampling(resource, **kwargs):
            submissions.append((resource, kwargs))
            return 50 if resource == "left" else 52

    adapter = object.__new__(RocmVulkanOutputAdapter)
    adapter.backend_name = "rocm"
    adapter.presenter = SimpleNamespace(vulkan=Context())
    adapter._host_staging_enabled = False
    adapter._released_source_frames = set()
    adapter._prepared_source_eyes = {(7, 0), (7, 1)}
    adapter._release_signaled = set()
    adapter._rocm_release_timelines = [0]
    adapter.left_release_semaphores = []
    adapter.right_release_semaphores = []
    adapter._source_frames = {
        7: (
            SimpleNamespace(resource="left"),
            SimpleNamespace(resource="right"),
            0,
        )
    }
    adapter.release_frame = released.append

    adapter.release_consumer_frame(7)

    assert adapter._rocm_release_timelines == [52]
    assert released == [7]
    assert len(submissions) == 2


def test_rocm_slot_claim_waits_for_slot_timeline_without_device_idle():
    waits = []
    adapter = object.__new__(RocmVulkanOutputAdapter)
    adapter.external_semaphore_enabled = False
    adapter._rocm_release_timelines = [37]
    adapter.presenter = SimpleNamespace(
        vulkan=SimpleNamespace(
            wait_for_timeline=lambda value: waits.append(value),
            wait_idle=lambda: (_ for _ in ()).throw(AssertionError("device idle is forbidden")),
        )
    )
    adapter._lease_condition = threading.Condition()
    adapter._active_leases = {}
    adapter._closed = False

    adapter._claim_slot(0, 8)

    assert waits == [37]
    assert adapter._rocm_release_timelines == [0]
    assert adapter._active_leases == {0: 8}


def test_consumer_keeps_running_for_recoverable_vulkan_timeout():
    runtime_q = queue.Queue(maxsize=1)
    shutdown = threading.Event()
    stats = []

    class PresenterSink:
        output_ready = True
        vulkan = SimpleNamespace(device_lost=False)

        @staticmethod
        def submit_runtime_result(_result, _timestamp):
            raise VulkanTimelineTimeout("timeline wait expired")

    runtime_q.put((SimpleNamespace(left_eye="cuda-left", right_eye="cuda-right"), 1.0))
    consumer = VulkanRuntimeOutputConsumer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        source_stat_inc=lambda name, amount=1, **values: stats.append((name, amount, values)),
        sink=PresenterSink(),
    )
    worker = threading.Thread(target=consumer.run)
    worker.start()
    deadline = time.monotonic() + 1.0
    while not stats:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert worker.is_alive()
    assert not shutdown.is_set()
    shutdown.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()


def test_screen_light_sample_completion_is_non_blocking_and_clamped():
    adapter = CudaVulkanOutputAdapter(None)
    adapter._screen_light_pending = (
        SimpleNamespace(tolist=lambda: [-1.0, 0.25, 12.0]),
        SimpleNamespace(query=lambda: True),
    )
    adapter._screen_light_last_submit = time.monotonic()

    adapter._update_screen_light_sample(object(), object())

    assert adapter._screen_light_rgb == (0.0, 0.25, 8.0)
    assert adapter._screen_light_pending is None


def test_consumer_rejects_non_vulkan_results_without_cpu_conversion():
    runtime_q = queue.Queue(maxsize=1)
    shutdown = threading.Event()
    stats = []
    consumer = VulkanRuntimeOutputConsumer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        source_stat_inc=lambda name, amount=1, **values: stats.append((name, amount, values)),
    )

    assert consumer._to_output_frame((SimpleNamespace(left_eye="cuda", right_eye="cuda"), 1.0)) is None
    assert any(item[0] == "runtime_output_waiting_for_vulkan_importer" for item in stats)


def test_consumer_overwrites_stale_queue_items():
    runtime_q = queue.Queue(maxsize=2)
    shutdown = threading.Event()
    stats = []
    consumer = VulkanRuntimeOutputConsumer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        source_stat_inc=lambda name, amount=1, **values: stats.append((name, amount, values)),
    )
    runtime_q.put((SimpleNamespace(left_eye="old", right_eye="old"), 1.0))
    runtime_q.put((SimpleNamespace(left_eye="new", right_eye="new"), 2.0))
    assert consumer._take_latest()[1] == 2.0


def test_consumer_preserves_first_frame_until_openxr_is_initialized():
    runtime_q = queue.Queue(maxsize=1)
    runtime_q.put((SimpleNamespace(left_eye="left", right_eye="right"), 1.0))
    shutdown = threading.Event()
    stats = []
    sink = SimpleNamespace(initialized=False)
    consumer = VulkanRuntimeOutputConsumer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        source_stat_inc=lambda name, amount=1, **values: stats.append(name),
        sink=sink,
    )
    worker = threading.Thread(target=consumer.run)

    worker.start()
    deadline = time.monotonic() + 1.0
    while "runtime_output_waiting_for_openxr" not in stats:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    shutdown.set()
    worker.join(timeout=1.0)

    assert runtime_q.qsize() == 1
    assert "runtime_output_waiting_for_openxr" in stats


def test_consumer_dispatches_raw_result_to_presenter_without_local_conversion():
    runtime_q = queue.Queue(maxsize=1)
    shutdown = threading.Event()
    calls = []
    stats = []
    runtime_result = SimpleNamespace(left_eye="cuda-left", right_eye="cuda-right")

    class PresenterSink:
        output_ready = True

        def submit_runtime_result(self, result, timestamp):
            calls.append((result, timestamp))

    runtime_q.put((runtime_result, 3.5))
    consumer = VulkanRuntimeOutputConsumer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        source_stat_inc=lambda name, amount=1, **values: stats.append(name),
        sink=PresenterSink(),
    )
    worker = threading.Thread(target=consumer.run)
    worker.start()
    deadline = time.monotonic() + 1.0
    while not calls:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    shutdown.set()
    worker.join(timeout=1.0)

    assert calls == [(runtime_result, 3.5)]
    assert "runtime_output_frames" in stats


def test_consumer_stops_cleanly_when_presenter_release_times_out():
    runtime_q = queue.Queue(maxsize=1)
    shutdown = threading.Event()
    stats = []

    class PresenterSink:
        output_ready = True
        vulkan = SimpleNamespace(device_lost=True)

        @staticmethod
        def submit_runtime_result(_result, _timestamp):
            raise RuntimeError("VkTimeout: timeline wait expired")

    runtime_q.put((SimpleNamespace(left_eye="cuda-left", right_eye="cuda-right"), 1.0))
    consumer = VulkanRuntimeOutputConsumer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        source_stat_inc=lambda name, amount=1, **values: stats.append((name, amount, values)),
        sink=PresenterSink(),
    )
    worker = threading.Thread(target=consumer.run)
    worker.start()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert shutdown.is_set()
    assert any(name == "runtime_output_sink_errors" for name, _amount, _values in stats)
