import json
import sys
import types
from dataclasses import dataclass

from miles.utils.p0b_system_profile import P0BSystemProfiler, _RayObjectStoreReader, tensor_payload_stats


class FakeStorage:
    def __init__(self, pointer, size):
        self._pointer = pointer
        self._size = size

    def data_ptr(self):
        return self._pointer

    def nbytes(self):
        return self._size


class FakeTensor:
    device = "cuda:0"

    def __init__(self, *, pointer, storage_bytes, logical_bytes):
        self._storage = FakeStorage(pointer, storage_bytes)
        self._logical_bytes = logical_bytes

    def untyped_storage(self):
        return self._storage

    def numel(self):
        return self._logical_bytes // 2

    def element_size(self):
        return 2


class FakeCuda:
    reset_calls = 0

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def current_device():
        return 0

    @staticmethod
    def reset_peak_memory_stats():
        FakeCuda.reset_calls += 1
        return None

    @staticmethod
    def memory_allocated(device):
        return 100

    @staticmethod
    def memory_reserved(device):
        return 200

    @staticmethod
    def max_memory_allocated(device):
        return 300

    @staticmethod
    def max_memory_reserved(device):
        return 400


class FakeTorch:
    cuda = FakeCuda()

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)


class SnapshotProvider:
    def __init__(self, *values):
        self.values = list(values)
        self.calls = 0

    def __call__(self):
        value = self.values[self.calls]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return value


def ray_snapshot(*, used, spilled):
    return {
        "object_store_bytes_used": used,
        "spilled_bytes_total": spilled,
        "ray_version": "2.44.1",
        "source": "unit-test",
    }


def test_pinned_ray_reader_uses_fresh_global_reply_and_caches_state():
    calls = {"state": 0, "reply": 0}
    state = object()

    def get_state_from_address():
        calls["state"] += 1
        return state

    def get_memory_info_reply(actual_state):
        assert actual_state is state
        calls["reply"] += 1
        stats = types.SimpleNamespace(
            object_store_bytes_used=100 + calls["reply"],
            spilled_bytes_total=200 + calls["reply"],
        )
        return types.SimpleNamespace(store_stats=stats)

    fake_ray = types.ModuleType("ray")
    fake_ray.__version__ = "2.44.1"
    fake_private = types.ModuleType("ray._private")
    fake_internal_api = types.ModuleType("ray._private.internal_api")
    fake_internal_api.get_state_from_address = get_state_from_address
    fake_internal_api.get_memory_info_reply = get_memory_info_reply
    fake_private.internal_api = fake_internal_api
    fake_ray._private = fake_private

    names = ("ray", "ray._private", "ray._private.internal_api")
    previous = {name: sys.modules.get(name) for name in names}
    sys.modules.update(
        {
            "ray": fake_ray,
            "ray._private": fake_private,
            "ray._private.internal_api": fake_internal_api,
        }
    )
    try:
        reader = _RayObjectStoreReader()
        assert reader()["object_store_bytes_used"] == 101
        assert reader()["spilled_bytes_total"] == 202
        assert calls == {"state": 1, "reply": 2}
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@dataclass
class FakePredictiveMicrobatch:
    old_inputs_concat: FakeTensor
    old_logits_concat: FakeTensor
    selected_total_tokens: int
    original_total_tokens: int


def test_tensor_payload_stats_deduplicates_shared_storage():
    first = FakeTensor(pointer=7, storage_bytes=64, logical_bytes=32)
    view = FakeTensor(pointer=7, storage_bytes=64, logical_bytes=16)
    stats = tensor_payload_stats({"items": [first, view]}, FakeTorch)
    assert stats == {
        "logical_bytes": 48,
        "unique_storage_bytes": 64,
        "tensor_count": 2,
        "ndarray_count": 0,
    }


def test_default_off_emits_nothing():
    lines = []
    FakeCuda.reset_calls = 0
    snapshots = SnapshotProvider(RuntimeError("must not be called"))
    profiler = P0BSystemProfiler(
        torch_module=FakeTorch,
        rank=0,
        world_size=8,
        primary=True,
        environ={},
        output=lambda *args, **kwargs: lines.append(args[0]),
        ray_snapshot_provider=snapshots,
    )
    profiler.begin_outer(3)
    profiler.emit_outer()
    assert not profiler.enabled
    assert lines == []
    assert FakeCuda.reset_calls == 0
    assert snapshots.calls == 0


def test_exact_zero_and_non_primary_are_disabled():
    lines = []
    FakeCuda.reset_calls = 0
    for environ, primary in (({"MILES_P0B_SYSTEM_PROFILE": "0"}, True), ({"MILES_P0B_SYSTEM_PROFILE": "1"}, False)):
        profiler = P0BSystemProfiler(
            torch_module=FakeTorch,
            rank=1,
            world_size=8,
            primary=primary,
            environ=environ,
            output=lambda *args, **kwargs: lines.append(args[0]),
        )
        profiler.begin_outer(3)
        profiler.emit_outer()
        assert not profiler.enabled
    assert lines == []
    assert FakeCuda.reset_calls == 0


def test_invalid_opt_in_values_fail_closed():
    for value in ("", "true", "01", " 1", "1 ", "2"):
        try:
            P0BSystemProfiler(
                torch_module=FakeTorch,
                rank=0,
                world_size=8,
                primary=True,
                environ={"MILES_P0B_SYSTEM_PROFILE": value},
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid opt-in value did not fail closed: {value!r}")


def test_invalid_explicit_outer_lists_fail_closed():
    for value in ("", " ", "3,,4", "three", "-1", "3,3"):
        try:
            P0BSystemProfiler(
                torch_module=FakeTorch,
                rank=0,
                world_size=8,
                primary=False,
                environ={"MILES_P0B_MEASURED_OUTER_IDS": value},
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid measured outer list did not fail closed: {value!r}")


def test_measured_outer_emits_detail_and_exact_collector_schema():
    lines = []
    output = lambda *args, **kwargs: lines.append(args[0])
    snapshots = SnapshotProvider(
        ray_snapshot(used=1000, spilled=40),
        ray_snapshot(used=1200, spilled=64),
    )
    profiler = P0BSystemProfiler(
        torch_module=FakeTorch,
        rank=0,
        world_size=8,
        primary=True,
        environ={"MILES_P0B_SYSTEM_PROFILE": "1", "MILES_P0B_MEASURED_OUTER_IDS": "3"},
        output=output,
        ray_snapshot_provider=snapshots,
    )
    payload_tensor = FakeTensor(pointer=10, storage_bytes=128, logical_bytes=96)
    cache_tensor = FakeTensor(pointer=11, storage_bytes=256, logical_bytes=128)
    profiler.begin_outer(3)
    profiler.mark("data_preprocess_start")
    profiler.observe_rollout_payload(
        {
            "tokens": [payload_tensor],
            "total_lengths": [17, 19],
            "response_lengths": [5, 7],
        }
    )
    profiler.mark("data_preprocess_end")
    profiler.observe_predictive_cache(
        [FakePredictiveMicrobatch(cache_tensor, cache_tensor, 8, 36)]
    )
    profiler.observe_completed_actor_steps(4)
    profiler.emit_outer()

    assert len(lines) == 2
    assert lines[0].startswith("P0B_DETAIL outer_id=3 ")
    detail = json.loads(lines[0].split(" ", 2)[2])
    assert detail["scope"] == "primary_training_rank_process_local"
    assert detail["token_counts"] == {
        "local_samples": 2,
        "local_total_tokens": 36,
        "local_response_tokens": 12,
    }
    assert detail["retained_predictive_cache"]["unique_storage_bytes"] == 256
    assert detail["completed_actor_steps"] == {"count": 4, "step_ids": [12, 13, 14, 15]}
    assert detail["accelerator_memory"]["peak_allocated_bytes"] == 300
    assert detail["phase_wall_seconds"]["data_preprocess"] >= 0
    assert detail["phase_wall_seconds"]["actor_update"] is None
    assert detail["ray_object_store"]["scope"] == "ray_cluster_global"
    assert detail["ray_object_store"]["spill_status"] == "measured_monotonic_delta"
    assert detail["ray_object_store"]["begin"]["spilled_bytes_total"] == 40
    assert detail["ray_object_store"]["end"]["object_store_bytes_used"] == 1200
    assert "ray_object_store_bytes" not in detail["unavailable"]
    assert "ray_spill_bytes" not in detail["unavailable"]

    assert lines[1].startswith("P0B_SYSTEM outer_id=3 ")
    system = json.loads(lines[1].split(" ", 2)[2])
    assert set(system) == {
        "gpu_allocated_bytes",
        "gpu_reserved_bytes",
        "host_rss_bytes",
        "ray_object_store_bytes",
        "ray_spill_bytes",
        "rollout_payload_bytes",
        "collective_bytes",
        "collective_time_seconds",
    }
    assert system["gpu_allocated_bytes"] == 300
    assert system["gpu_reserved_bytes"] == 400
    assert system["rollout_payload_bytes"] == 128
    assert system["ray_object_store_bytes"] == 1200
    assert system["ray_spill_bytes"] == 24
    assert system["collective_bytes"] is None
    assert snapshots.calls == 2


def test_ray_spill_counter_regression_is_unavailable_not_negative_or_zero():
    lines = []
    snapshots = SnapshotProvider(
        ray_snapshot(used=1000, spilled=40),
        ray_snapshot(used=900, spilled=39),
    )
    profiler = P0BSystemProfiler(
        torch_module=FakeTorch,
        rank=0,
        world_size=8,
        primary=True,
        environ={"MILES_P0B_SYSTEM_PROFILE": "1", "MILES_P0B_MEASURED_OUTER_IDS": "3"},
        output=lambda *args, **kwargs: lines.append(args[0]),
        ray_snapshot_provider=snapshots,
    )
    profiler.begin_outer(3)
    profiler.emit_outer()

    detail = json.loads(lines[0].split(" ", 2)[2])
    system = json.loads(lines[1].split(" ", 2)[2])
    assert system["ray_object_store_bytes"] == 900
    assert system["ray_spill_bytes"] is None
    assert detail["ray_object_store"]["spill_status"] == "counter_regressed"
    assert detail["unavailable"].count("ray_spill_bytes") == 1


def test_ray_snapshot_failure_and_malformed_values_fail_closed():
    for bad_begin in (
        RuntimeError("RPC failed"),
        ray_snapshot(used=True, spilled=0),
        ray_snapshot(used=1, spilled=-1),
    ):
        lines = []
        snapshots = SnapshotProvider(bad_begin, RuntimeError("RPC failed"))
        profiler = P0BSystemProfiler(
            torch_module=FakeTorch,
            rank=0,
            world_size=8,
            primary=True,
            environ={"MILES_P0B_SYSTEM_PROFILE": "1", "MILES_P0B_MEASURED_OUTER_IDS": "3"},
            output=lambda *args, **kwargs: lines.append(args[0]),
            ray_snapshot_provider=snapshots,
        )
        profiler.begin_outer(3)
        profiler.emit_outer()

        detail = json.loads(lines[0].split(" ", 2)[2])
        system = json.loads(lines[1].split(" ", 2)[2])
        assert system["ray_object_store_bytes"] is None
        assert system["ray_spill_bytes"] is None
        assert detail["ray_object_store"]["begin"]["available"] is False
        assert detail["ray_object_store"]["end"]["available"] is False
        assert "ray_object_store_bytes" in detail["unavailable"]
        assert "ray_spill_bytes" in detail["unavailable"]


def test_warmup_outer_does_not_emit_marker():
    lines = []
    snapshots = SnapshotProvider(RuntimeError("must not be called"))
    profiler = P0BSystemProfiler(
        torch_module=FakeTorch,
        rank=0,
        world_size=8,
        primary=True,
        environ={"MILES_P0B_SYSTEM_PROFILE": "1"},
        output=lambda *args, **kwargs: lines.append(args[0]),
        ray_snapshot_provider=snapshots,
    )
    profiler.begin_outer(2)
    profiler.emit_outer()
    assert lines == []
    assert snapshots.calls == 0


def test_default_window_emits_exactly_ten_outers_and_resets_each_outer():
    lines = []
    FakeCuda.reset_calls = 0
    profiler = P0BSystemProfiler(
        torch_module=FakeTorch,
        rank=0,
        world_size=8,
        primary=True,
        environ={"MILES_P0B_SYSTEM_PROFILE": "1"},
        output=lambda *args, **kwargs: lines.append(args[0]),
        ray_snapshot_provider=lambda: ray_snapshot(used=1000, spilled=40),
    )
    tensor = FakeTensor(pointer=99, storage_bytes=128, logical_bytes=64)
    for outer_id in range(13):
        profiler.begin_outer(outer_id)
        profiler.observe_rollout_payload(
            {"tokens": [tensor], "total_lengths": [8], "response_lengths": [4]}
        )
        profiler.emit_outer()
    system_lines = [line for line in lines if line.startswith("P0B_SYSTEM")]
    assert [int(line.split(" ", 2)[1].split("=", 1)[1]) for line in system_lines] == list(range(3, 13))
    assert FakeCuda.reset_calls == 13
