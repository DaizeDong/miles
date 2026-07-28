import copy
import hashlib
import json
import os
import socket
import types

import pytest

import miles.utils.p0b_collective_trace as trace_module
from miles.utils.p0b_collective_trace import (
    CollectiveTraceError,
    FlightRecorderCollectiveProfiler,
)


class DumpSequence:
    def __init__(self, *values):
        self.values = list(values)
        self.calls = 0

    def __call__(self):
        index = min(self.calls, len(self.values) - 1)
        self.calls += 1
        value = self.values[index]
        if isinstance(value, Exception):
            raise value
        return json.dumps(value)


class FakeWork:
    def __init__(self, *, completed=True, duration_ms=0.75):
        self.completed = completed
        self.duration_ms = duration_ms
        self.wait_calls = 0

    def is_completed(self):
        return self.completed

    def _get_duration(self):
        return self.duration_ms

    def wait(self):
        self.wait_calls += 1
        self.completed = True


class FakeGroup:
    group_name = "0"

    def __init__(self):
        self.sequence = 0

    def _get_sequence_number_for_group(self):
        return self.sequence


class FakeDistributed:
    def __init__(self):
        self.group = FakeGroup()
        self.async_work = FakeWork()
        self.distributed_c10d = self

    def _get_default_group(self):
        return self.group

    @staticmethod
    def _async_requested(args, kwargs, index):
        return kwargs.get("async_op", args[index] if len(args) > index else False)

    def all_reduce(self, *args, **kwargs):
        self.group.sequence += 1
        if self._async_requested(args, kwargs, 3):
            return self.async_work
        return None

    def barrier(self, *args, **kwargs):
        self.group.sequence += 1
        if self._async_requested(args, kwargs, 1):
            return self.async_work
        return None

    def monitored_barrier(self, *args, **kwargs):
        return None


class FakeTorch:
    def __init__(self, dump):
        self.distributed = FakeDistributed()
        c10d = types.SimpleNamespace(_dump_nccl_trace_json=dump)
        self._C = types.SimpleNamespace(_distributed_c10d=c10d)


def flight_entry(
    record_id,
    *,
    sequence=None,
    op="all_reduce",
    shape=None,
    dtype="Float",
    duration_ms=0.25,
    retired=True,
    state="completed",
    is_p2p=False,
):
    if sequence is None:
        sequence = record_id + 1
    if shape is None:
        shape = [4]
    return {
        "record_id": record_id,
        "collective_seq_id": sequence,
        "op_id": sequence,
        "pg_id": 0,
        "process_group": ["0", "default_pg"],
        "profiling_name": f"nccl:{op}",
        "input_dtypes": [dtype],
        "input_sizes": [shape],
        "is_p2p": is_p2p,
        "state": state,
        "retired": retired,
        "duration_ms": duration_ms,
        "time_created_ns": 1000 + record_id * 10,
        "time_discovered_started_ns": 1001 + record_id * 10,
        "time_discovered_completed_ns": 1002 + record_id * 10,
    }


def trace(entries, *, version="2.9", nccl_version="2.26.6"):
    return {
        "version": version,
        "nccl_version": nccl_version,
        "entries": entries,
        "pg_config": {
            "0": {"name": "0", "desc": "default_pg", "ranks": "[0, 1]"}
        },
        "pg_status": {
            "0": {
                "last_enqueued_collective": str(len(entries)),
                "last_completed_collective": str(len(entries)),
            }
        },
    }


def boundary(entries):
    value = trace(copy.deepcopy(entries))
    for entry in value["entries"]:
        entry["retired"] = False
        entry["duration_ms"] = None
    return value


def environment(tmp_path):
    hostname = socket.gethostname().split(".", 1)[0]
    return {
        "TORCH_FR_BUFFER_SIZE": "65536",
        "TORCH_NCCL_ENABLE_TIMING": "1",
        "MILES_P0B_TRACE_ROOT": str(tmp_path / "traces"),
        "SLURM_JOB_ID": "3261",
        "SLURMD_NODENAME": hostname,
    }


def profiler_for(tmp_path, *dumps, world_size=2):
    fake_torch = FakeTorch(DumpSequence(*dumps))
    profiler = FlightRecorderCollectiveProfiler(
        torch_module=fake_torch,
        rank=0,
        world_size=world_size,
        enabled=True,
        environ=environment(tmp_path),
        require_shared_nfs=False,
    )
    return profiler, fake_torch


@pytest.fixture(autouse=True)
def fast_retirement_poll(monkeypatch):
    monkeypatch.setattr(trace_module, "POLL_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(trace_module, "POLL_INTERVAL_SECONDS", 0.0)


def test_3261_style_sync_then_async_same_op_binds_by_sequence_without_wait(tmp_path):
    retired = [
        flight_entry(0, sequence=1, shape=[4], duration_ms=0.125),
        flight_entry(1, sequence=2, shape=[2, 3], duration_ms=0.375),
    ]
    profiler, fake_torch = profiler_for(
        tmp_path,
        trace([]),
        boundary(retired),
        trace(retired),
    )
    # Deliberately different from both trace durations.  Work timing is
    # independent corroboration, not an operation-name pairing heuristic.
    fake_torch.distributed.async_work.duration_ms = 9.5

    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce("sync-tensor")
    work = fake_torch.distributed.all_reduce("async-tensor", async_op=True)
    collective_bytes, collective_seconds, evidence = profiler.end_outer(3)

    assert collective_bytes == 16 + 24
    assert collective_seconds == pytest.approx(0.0005)
    assert evidence["available"] is True
    assert evidence["world_size_multiplier_applied"] is False
    assert evidence["entry_count"] == 2
    assert evidence["public_call_count"] == 2
    assert evidence["async_work_count"] == 1
    assert work.wait_calls == 0

    trace_path = evidence["trace_path"]
    raw = open(trace_path, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == evidence["trace_sha256"]
    payload = json.loads(raw)
    assert payload["runtime_identity"] == {
        "hostname": socket.gethostname().split(".", 1)[0],
        "pid": os.getpid(),
        "rank": 0,
        "slurm_job_id": "3261",
        "slurm_node_name": socket.gethostname().split(".", 1)[0],
        "world_size": 2,
    }
    calls = payload["public_call_evidence"]
    assert [(item["call_id"], item["async_op"], item["flight_record_id"]) for item in calls] == [
        (0, False, 0),
        (1, True, 1),
    ]
    assert calls[1]["work_device_duration_ms"] == 9.5
    assert calls[1]["work_duration_compared_to_flight_duration"] is False
    assert payload["record_id_set_frozen_at_outer_close"] is True
    assert payload["new_records_during_retirement_poll"] == 0
    assert all(
        entry["retired_trace_host_discovery"]["used_for_device_duration"] is False
        for entry in payload["entries"]
    )


def test_barrier_internal_tensor_is_zero_payload_and_not_world_multiplied(tmp_path):
    retired = [
        flight_entry(0, sequence=1, shape=[8], duration_ms=0.2),
        flight_entry(
            1,
            sequence=2,
            op="all_reduce_barrier",
            shape=[1],
            duration_ms=0.3,
        ),
    ]
    profiler, fake_torch = profiler_for(
        tmp_path, trace([]), boundary(retired), trace(retired), world_size=8
    )
    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce("tensor")
    fake_torch.distributed.barrier()
    collective_bytes, collective_seconds, evidence = profiler.end_outer(3)
    assert collective_bytes == 8 * 4
    assert collective_seconds == pytest.approx(0.0005)
    assert evidence["group_size_histogram"] == {"2": 2}


@pytest.mark.parametrize(
    "version,nccl_version",
    [(None, "2.26.6"), ("", "2.26.6"), ("2.9", None), ("2.9", "")],
)
def test_begin_rejects_missing_or_empty_trace_versions(
    tmp_path, version, nccl_version
):
    profiler, _ = profiler_for(
        tmp_path, trace([], version=version, nccl_version=nccl_version)
    )
    with pytest.raises(CollectiveTraceError):
        profiler.begin_outer(3)


def run_one_call_failure(tmp_path, boundary_trace, final_trace=None, *, async_op=False):
    dumps = [trace([]), boundary_trace]
    if final_trace is not None:
        dumps.append(final_trace)
    profiler, fake_torch = profiler_for(tmp_path, *dumps)
    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce("tensor", async_op=async_op)
    return profiler.end_outer(3), fake_torch


@pytest.mark.parametrize(
    "entry,error_fragment",
    [
        (flight_entry(0, op="mystery_collective"), "unsupported communication operation"),
        (flight_entry(0, is_p2p=True), "P2P entry"),
        (flight_entry(0, duration_ms=float("nan")), "invalid duration"),
        (flight_entry(0, duration_ms=0.0), "invalid duration"),
    ],
)
def test_unknown_p2p_and_nonpositive_or_nonfinite_duration_fail_closed(
    tmp_path, entry, error_fragment
):
    result, _ = run_one_call_failure(tmp_path, boundary([entry]), trace([entry]))
    collective_bytes, collective_seconds, evidence = result
    assert collective_bytes is None and collective_seconds is None
    assert evidence["available"] is False
    assert error_fragment in evidence["error_message"]


def test_ring_gap_at_outer_close_fails_closed(tmp_path):
    entries = [flight_entry(0, sequence=1), flight_entry(2, sequence=2)]
    profiler, fake_torch = profiler_for(tmp_path, trace([]), boundary(entries))
    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce("a")
    fake_torch.distributed.all_reduce("b")
    values = profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "not contiguous" in values[2]["error_message"]


def test_missing_duration_times_out_fail_closed(tmp_path):
    entry = flight_entry(0, duration_ms=None, retired=False)
    result, _ = run_one_call_failure(tmp_path, boundary([entry]), trace([entry]))
    assert result[0] is None and result[1] is None
    assert "did not retire" in result[2]["error_message"]


def test_outstanding_async_work_fails_at_boundary_without_wait(tmp_path):
    entry = flight_entry(0, sequence=1)
    profiler, fake_torch = profiler_for(
        tmp_path, trace([]), boundary([entry]), trace([entry])
    )
    fake_torch.distributed.async_work.completed = False
    profiler.begin_outer(3)
    work = fake_torch.distributed.all_reduce("tensor", async_op=True)
    values = profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "remained outstanding at outer close" in values[2]["error_message"]
    assert work.wait_calls == 0


@pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), float("inf")])
def test_async_work_duration_must_be_positive_finite(tmp_path, duration):
    entry = flight_entry(0, sequence=1)
    profiler, fake_torch = profiler_for(
        tmp_path, trace([]), boundary([entry]), trace([entry])
    )
    fake_torch.distributed.async_work.duration_ms = duration
    profiler.begin_outer(3)
    work = fake_torch.distributed.all_reduce("tensor", async_op=True)
    values = profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "duration is invalid" in values[2]["error_message"]
    assert work.wait_calls == 0


def test_new_record_after_outer_close_is_rejected(tmp_path):
    first = flight_entry(0, sequence=1)
    second = flight_entry(1, sequence=2)
    profiler, fake_torch = profiler_for(
        tmp_path,
        trace([]),
        boundary([first]),
        trace([first, second]),
    )
    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce("tensor")
    values = profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "outer-close set changed" in values[2]["error_message"]


def test_version_drift_within_outer_fails_closed(tmp_path):
    entry = flight_entry(0)
    profiler, fake_torch = profiler_for(
        tmp_path,
        trace([]),
        boundary([entry]) | {"version": "3.0"},
    )
    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce("tensor")
    values = profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "version drifted" in values[2]["error_message"]


def test_unsupported_collective_like_public_api_marks_outer_unavailable(tmp_path):
    profiler, fake_torch = profiler_for(tmp_path, trace([]))
    profiler.begin_outer(3)
    fake_torch.distributed.monitored_barrier()
    values = profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "unsupported object/P2P collective API" in values[2]["error_message"]


def test_runtime_job_node_binding_is_strict(tmp_path):
    env = environment(tmp_path)
    env["SLURM_JOB_ID"] = "03261"
    fake_torch = FakeTorch(DumpSequence(trace([])))
    with pytest.raises(CollectiveTraceError, match="canonical positive integer"):
        FlightRecorderCollectiveProfiler(
            torch_module=fake_torch,
            rank=0,
            world_size=2,
            enabled=True,
            environ=env,
            require_shared_nfs=False,
        )


def test_adjacent_outers_reset_record_and_call_state(tmp_path):
    first = flight_entry(0, sequence=1, shape=[2], duration_ms=0.1)
    second = flight_entry(1, sequence=2, shape=[5], duration_ms=0.2)
    profiler, fake_torch = profiler_for(
        tmp_path,
        trace([]),
        boundary([first]),
        trace([first]),
        trace([first]),
        boundary([first, second]),
        trace([first, second]),
    )
    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce("first")
    first_values = profiler.end_outer(3)
    profiler.begin_outer(4)
    fake_torch.distributed.all_reduce("second")
    second_values = profiler.end_outer(4)
    assert first_values[0] == 8
    assert second_values[0] == 20
    assert first_values[2]["begin_record_id"] == -1
    assert second_values[2]["begin_record_id"] == 0


def test_trace_is_create_once(tmp_path):
    entry = flight_entry(0)
    profiler, fake_torch = profiler_for(
        tmp_path, trace([]), boundary([entry]), trace([entry])
    )
    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce("first")
    assert profiler.end_outer(3)[2]["available"] is True

    second_profiler, second_torch = profiler_for(
        tmp_path, trace([]), boundary([entry]), trace([entry])
    )
    second_profiler.begin_outer(3)
    second_torch.distributed.all_reduce("second")
    values = second_profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "refusing to overwrite" in values[2]["error_message"]
