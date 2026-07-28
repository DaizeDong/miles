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
        if callable(value):
            value = value()
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
    def __init__(self, name="0", backend="nccl"):
        self.group_name = name
        self.backend = backend
        self.sequence = 0

    def _get_sequence_number_for_group(self):
        return self.sequence

    def size(self):
        return 2


class FakeReloadableGroup:
    """Production-shaped wrapper whose own sequence never advances."""

    def __init__(self, inner):
        self.group = inner
        self.wrapper_sequence = 900

    def _p0b_current_process_group(self):
        return self.group

    def _get_sequence_number_for_group(self):
        return self.wrapper_sequence


class FakeTensor:
    dtype = "torch.float32"
    device = "cuda:0"

    def __init__(self, shape, *, element_size=4, contiguous=True):
        self.shape = tuple(shape)
        self._element_size = element_size
        self._contiguous = contiguous

    def numel(self):
        value = 1
        for dimension in self.shape:
            value *= dimension
        return value

    def element_size(self):
        return self._element_size

    def is_contiguous(self):
        return self._contiguous


class FakeDistributed:
    def __init__(self):
        self.group = FakeGroup()
        self.gloo_group = FakeGroup(name="gloo-control", backend="gloo")
        self.async_work = FakeWork()
        self.distributed_c10d = self

    def _get_default_group(self):
        return self.group

    @staticmethod
    def _async_requested(args, kwargs, index):
        return kwargs.get("async_op", args[index] if len(args) > index else False)

    def _call_group(self, args, kwargs, index):
        group = kwargs.get("group", args[index] if len(args) > index else None)
        if group is None:
            return self.group
        resolver = getattr(group, "_p0b_current_process_group", None)
        return resolver() if callable(resolver) else group

    def all_reduce(self, *args, **kwargs):
        self._call_group(args, kwargs, 2).sequence += 1
        if self._async_requested(args, kwargs, 3):
            return self.async_work
        return None

    def barrier(self, *args, **kwargs):
        self._call_group(args, kwargs, 0).sequence += 1
        if self._async_requested(args, kwargs, 1):
            return self.async_work
        return None

    def monitored_barrier(self, *args, **kwargs):
        return None

    def gather(self, *args, **kwargs):
        self._call_group(args, kwargs, 3).sequence += 1
        return None

    @staticmethod
    def get_backend(group):
        return group.backend

    @staticmethod
    def get_process_group_ranks(group):
        return list(range(group.size()))

    def gather_object(self, *args, **kwargs):
        group = self._call_group(args, kwargs, 3)
        # Mirror the high-level object's private tensor lowering.  The P0-B
        # Gloo-control wrapper must guard this wrapped tensor API so it is not
        # misclassified as an accelerator tensor collective.
        self.gather(FakeTensor([2]), dst=0, group=group)
        return None


class FakeTorch:
    def __init__(self, dump):
        self.distributed = FakeDistributed()
        c10d = types.SimpleNamespace(_dump_nccl_trace_json=dump)
        self._C = types.SimpleNamespace(_distributed_c10d=c10d)

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)


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


def trace(
    entries,
    *,
    version="2.9",
    nccl_version="2.26.6",
    ranks="[0, 1]",
):
    return {
        "version": version,
        "nccl_version": nccl_version,
        "entries": entries,
        "pg_config": {
            "0": {"name": "0", "desc": "default_pg", "ranks": ranks}
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
    fake_torch.distributed.async_work.duration_ms = 0.375

    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce(FakeTensor([4]))
    work = fake_torch.distributed.all_reduce(FakeTensor([2, 3]), async_op=True)
    frozen = profiler.freeze_outer_boundary(3)
    assert frozen["available"] is True
    assert frozen["entry_count"] == 2
    collective_bytes, collective_seconds, evidence = profiler.end_outer(3)

    assert collective_bytes == 16 + 24, evidence
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
    assert calls[1]["work_device_duration_ms"] == 0.375
    assert calls[1]["work_duration_compared_to_flight_duration"] is True
    assert calls[1]["runtime_flight_bytes_match"] is True
    assert payload["record_id_set_frozen_at_outer_close"] is True
    assert payload["new_records_during_retirement_poll"] == 0
    assert all(
        entry["retired_trace_host_discovery"]["used_for_device_duration"] is False
        for entry in payload["entries"]
    )


def test_reloadable_process_group_binds_the_live_inner_sequence(tmp_path):
    retired = [flight_entry(0, sequence=1, shape=[4], duration_ms=0.25)]
    profiler, fake_torch = profiler_for(
        tmp_path,
        trace([]),
        boundary(retired),
        trace(retired),
    )
    inner = fake_torch.distributed.group
    wrapper = FakeReloadableGroup(inner)

    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce(FakeTensor([4]), group=wrapper)
    collective_bytes, _, evidence = profiler.end_outer(3)

    assert collective_bytes == 16
    assert evidence["available"] is True
    assert wrapper.wrapper_sequence == 900
    payload = json.loads(open(evidence["trace_path"], encoding="utf-8").read())
    assert payload["public_call_evidence"][0]["group_resolution"] == (
        "reloadable_current_inner_process_group"
    )


def test_missing_retired_pg_config_binds_flight_name_to_live_runtime_ranks(tmp_path):
    entry = flight_entry(0, sequence=1, shape=[4], duration_ms=0.25)
    entry["pg_id"] = 47
    entry["process_group"] = ["47", "reloaded_inner_pg"]
    boundary_trace = boundary([entry])
    boundary_trace["pg_config"] = {}
    boundary_trace["pg_status"] = {}
    retired_trace = trace([entry])
    retired_trace["pg_config"] = {}
    retired_trace["pg_status"] = {}
    profiler, fake_torch = profiler_for(
        tmp_path, trace([]), boundary_trace, retired_trace
    )
    fake_torch.distributed.group = FakeGroup(name="47")

    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce(FakeTensor([4]))
    frozen = profiler.freeze_outer_boundary(3)
    assert frozen["available"] is True
    collective_bytes, collective_seconds, evidence = profiler.end_outer(3)

    assert collective_bytes == 16
    assert collective_seconds == pytest.approx(0.00025)
    assert evidence["available"] is True
    payload = json.loads(open(evidence["trace_path"], encoding="utf-8").read())
    call = payload["public_call_evidence"][0]
    parsed = payload["entries"][0]
    assert call["group_name"] == "47"
    assert call["group_ranks"] == [0, 1]
    assert call["flight_group_identity_source"] == (
        "flight_entry_process_group_and_live_runtime_ranks"
    )
    assert call["flight_pg_config_available"] is False
    assert parsed["group_identity_source"] == (
        "flight_entry_process_group_and_live_runtime_ranks"
    )
    assert parsed["group_ranks"] == [0, 1]
    assert parsed["flight_pg_config_available"] is False


@pytest.mark.parametrize(
    "process_group",
    [None, [], ["47"], ["", "desc"], ["47", ""], [47, "desc"]],
)
def test_missing_pg_config_requires_exact_flight_process_group_schema(
    tmp_path, process_group
):
    entry = flight_entry(0, sequence=1, shape=[4])
    entry["pg_id"] = 47
    entry["process_group"] = process_group
    boundary_trace = boundary([entry])
    boundary_trace["pg_config"] = {}
    profiler, fake_torch = profiler_for(tmp_path, trace([]), boundary_trace)
    fake_torch.distributed.group = FakeGroup(name="47")

    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce(FakeTensor([4]))
    frozen = profiler.freeze_outer_boundary(3)

    assert frozen["available"] is False
    assert "invalid process_group for pg_id=47" in frozen["error_message"]


def test_missing_pg_config_rejects_name_bound_to_a_different_pg_id(tmp_path):
    entry = flight_entry(0, sequence=1, shape=[4])
    entry["pg_id"] = 47
    entry["process_group"] = ["47", "retired"]
    boundary_trace = boundary([entry])
    boundary_trace["pg_config"] = {
        "12": {"name": "47", "desc": "other", "ranks": "[0, 1]"}
    }
    profiler, fake_torch = profiler_for(tmp_path, trace([]), boundary_trace)
    fake_torch.distributed.group = FakeGroup(name="47")

    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce(FakeTensor([4]))
    frozen = profiler.freeze_outer_boundary(3)

    assert frozen["available"] is False
    assert "belongs to config IDs ['12'], not missing pg_id=47" in frozen[
        "error_message"
    ]


def test_torch_2_9_coalescing_transaction_aggregates_runtime_inputs_and_real_work(
    tmp_path,
):
    retired_entry = flight_entry(
        0,
        sequence=1,
        op="reduce_scatter_tensor_coalesced",
        shape=[4],
        duration_ms=0.375,
    )
    retired_entry["input_dtypes"] = ["Float", "Float"]
    retired_entry["input_sizes"] = [[4], [2, 3]]
    profiler, fake_torch = profiler_for(
        tmp_path,
        trace([]),
        boundary([retired_entry]),
        trace([retired_entry]),
    )
    fake_torch.distributed.async_work.duration_ms = 0.375
    inner = fake_torch.distributed.group
    wrapper = FakeReloadableGroup(inner)

    def reduce_scatter_tensor():
        raise AssertionError("identity-only fake operation must not execute")

    operations = [
        types.SimpleNamespace(op=reduce_scatter_tensor, tensor=FakeTensor([4])),
        types.SimpleNamespace(op=reduce_scatter_tensor, tensor=FakeTensor([2, 3])),
    ]
    fake_torch.distributed._world = types.SimpleNamespace(
        pg_coalesce_state={wrapper: operations}
    )

    profiler.begin_outer(3)
    transaction = profiler.prepare_coalescing_manager(
        (wrapper,), {"async_ops": True}
    )
    profiler.capture_coalescing_manager(transaction)
    inner.sequence += 1
    manager = types.SimpleNamespace(works=[fake_torch.distributed.async_work])
    profiler.complete_coalescing_manager(transaction, manager)
    collective_bytes, _, evidence = profiler.end_outer(3)

    assert collective_bytes == 40
    assert wrapper.wrapper_sequence == 900
    assert fake_torch.distributed.async_work.wait_calls == 0
    payload = json.loads(open(evidence["trace_path"], encoding="utf-8").read())
    call = payload["public_call_evidence"][0]
    assert call["public_api"] == "_coalescing_manager[reduce_scatter_tensor]"
    assert call["coalesced_subcall_count"] == 2
    assert call["observation_source"] == "torch_2_9_coalescing_transaction"
    assert call["flight_operation"] == "reduce_scatter_tensor_coalesced"
    assert call["runtime_rank_local_logical_input_bytes"] == 40
    assert call["runtime_flight_bytes_match"] is True


def test_every_supported_public_api_has_runtime_input_extractor():
    tensor2 = FakeTensor([2])
    tensor3 = FakeTensor([3])
    output = FakeTensor([8])
    cases = {
        "all_reduce": ((tensor2,), 8),
        "all_reduce_coalesced": (([tensor2, tensor3],), 20),
        "all_gather": (([output], tensor2), 8),
        "all_gather_coalesced": (([[output], [output]], [tensor2, tensor3]), 20),
        "all_gather_into_tensor": ((output, tensor2), 8),
        "_all_gather_base": ((output, tensor2), 8),
        "all_gather_into_tensor_coalesced": (([output], [tensor2, tensor3]), 20),
        "reduce_scatter": ((output, [tensor2, tensor3]), 20),
        "reduce_scatter_tensor": ((output, tensor2), 8),
        "_reduce_scatter_base": ((output, tensor2), 8),
        "reduce_scatter_tensor_coalesced": (([output], [tensor2, tensor3]), 20),
        "all_to_all": (([output], [tensor2, tensor3]), 20),
        "all_to_all_single": ((output, tensor2), 8),
        "broadcast": ((tensor2,), 8),
        "reduce": ((tensor2,), 8),
        "scatter": ((output, [tensor2, tensor3]), 20),
        "gather": ((tensor2,), 8),
        "barrier": ((), 0),
    }
    assert set(cases) == set(trace_module._PUBLIC_CALL_SPECS)
    assert trace_module._PUBLIC_CALL_SPECS[
        "all_reduce_coalesced"
    ].expected_low_level_ops == frozenset({"allreduce_coalesced"})
    assert trace_module._PUBLIC_CALL_SPECS[
        "all_to_all_single"
    ].expected_low_level_ops == frozenset({"all_to_all"})
    fake_torch = FakeTorch(DumpSequence(trace([])))
    for name, (args, expected_bytes) in cases.items():
        actual_bytes, tensors = trace_module._runtime_input_payload(
            name,
            trace_module._PUBLIC_CALL_SPECS[name],
            args,
            {},
            fake_torch,
        )
        assert actual_bytes == expected_bytes, name
        assert sum(item["logical_bytes"] for item in tensors) == expected_bytes


def test_noncontiguous_runtime_tensor_uses_logical_numel_times_element_size():
    tensor = FakeTensor([3, 5], element_size=2, contiguous=False)
    fake_torch = FakeTorch(DumpSequence(trace([])))
    logical_bytes, tensors = trace_module._runtime_input_payload(
        "all_reduce",
        trace_module._PUBLIC_CALL_SPECS["all_reduce"],
        (tensor,),
        {},
        fake_torch,
    )
    assert logical_bytes == 30
    assert tensors == [
        {
            "dtype": "torch.float32",
            "device": "cuda:0",
            "shape": [3, 5],
            "numel": 15,
            "element_size": 2,
            "logical_bytes": 30,
            "is_contiguous": False,
        }
    ]


def test_scatter_non_source_none_input_is_exact_zero():
    fake_torch = FakeTorch(DumpSequence(trace([])))
    logical_bytes, tensors = trace_module._runtime_input_payload(
        "scatter",
        trace_module._PUBLIC_CALL_SPECS["scatter"],
        (FakeTensor([4]), None),
        {},
        fake_torch,
    )
    assert logical_bytes == 0
    assert tensors == []


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
    fake_torch.distributed.all_reduce(FakeTensor([8]))
    fake_torch.distributed.barrier()
    collective_bytes, collective_seconds, evidence = profiler.end_outer(3)
    assert collective_bytes == 8 * 4
    assert collective_seconds == pytest.approx(0.0005)
    assert evidence["group_size_histogram"] == {"2": 2}


@pytest.mark.parametrize(
    "version,nccl_version",
    [
        (None, "2.26.6"),
        ("", "2.26.6"),
        ("3.0", "2.26.6"),
        ("2.9", None),
        ("2.9", ""),
        ("2.9", "9.9.9"),
    ],
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
    fake_torch.distributed.all_reduce(FakeTensor([4]), async_op=async_op)
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
    fake_torch.distributed.all_reduce(FakeTensor([4]))
    fake_torch.distributed.all_reduce(FakeTensor([4]))
    values = profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "not contiguous" in values[2]["error_message"]


def test_missing_duration_times_out_fail_closed(tmp_path):
    entry = flight_entry(0, duration_ms=None, retired=False)
    result, _ = run_one_call_failure(tmp_path, boundary([entry]), trace([entry]))
    assert result[0] is None and result[1] is None
    assert "did not retire" in result[2]["error_message"]


def test_runtime_and_flight_byte_mismatch_fails_at_boundary(tmp_path):
    entry = flight_entry(0, shape=[4])
    profiler, fake_torch = profiler_for(
        tmp_path, trace([]), boundary([entry]), trace([entry])
    )
    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce(FakeTensor([5]))
    frozen = profiler.freeze_outer_boundary(3)
    assert frozen["available"] is False
    assert "runtime/flight byte mismatch" in frozen["error_message"]
    values = profiler.end_outer(3)
    assert values[0] is None and values[1] is None


def test_retired_trace_immutable_tensor_schema_drift_fails(tmp_path):
    entry = flight_entry(0, shape=[4])
    drifted = flight_entry(0, shape=[5])
    profiler, fake_torch = profiler_for(
        tmp_path, trace([]), boundary([entry]), trace([drifted])
    )
    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce(FakeTensor([4]))
    assert profiler.freeze_outer_boundary(3)["available"] is True
    values = profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "immutable fields drifted" in values[2]["error_message"]


def test_retired_trace_discovery_timestamp_refresh_is_host_evidence_only(tmp_path):
    entry = flight_entry(0, shape=[4])
    drifted = flight_entry(0, shape=[4])
    drifted["time_discovered_completed_ns"] += 1
    profiler, fake_torch = profiler_for(
        tmp_path, trace([]), boundary([entry]), trace([drifted])
    )
    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce(FakeTensor([4]))
    assert profiler.freeze_outer_boundary(3)["available"] is True
    values = profiler.end_outer(3)
    assert values[0] == 16
    assert values[2]["available"] is True
    payload = json.loads(open(values[2]["trace_path"], encoding="utf-8").read())
    assert payload["entries"][0]["retired_trace_host_discovery"][
        "used_for_device_duration"
    ] is False


def test_retired_trace_invalid_discovery_timestamps_fail_closed(tmp_path):
    entry = flight_entry(0, shape=[4])
    malformed = flight_entry(0, shape=[4])
    malformed["time_discovered_completed_ns"] = (
        malformed["time_discovered_started_ns"] - 1
    )
    profiler, fake_torch = profiler_for(
        tmp_path, trace([]), boundary([entry]), trace([malformed])
    )
    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce(FakeTensor([4]))
    assert profiler.freeze_outer_boundary(3)["available"] is True
    values = profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "invalid retired discovery timestamps" in values[2]["error_message"]


def test_existing_boundary_duration_cannot_be_rewritten_at_retirement(tmp_path):
    entry = flight_entry(0, shape=[4], duration_ms=0.25, retired=False)
    drifted = flight_entry(0, shape=[4], duration_ms=0.5, retired=True)
    profiler, fake_torch = profiler_for(
        tmp_path, trace([]), trace([entry]), trace([drifted])
    )
    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce(FakeTensor([4]))
    assert profiler.freeze_outer_boundary(3)["available"] is True
    values = profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "duration drifted" in values[2]["error_message"]


def test_async_work_and_exact_bound_flight_duration_must_match(tmp_path):
    entry = flight_entry(0, duration_ms=0.25)
    profiler, fake_torch = profiler_for(
        tmp_path, trace([]), boundary([entry]), trace([entry])
    )
    fake_torch.distributed.async_work.duration_ms = 0.5
    profiler.begin_outer(3)
    work = fake_torch.distributed.all_reduce(FakeTensor([4]), async_op=True)
    assert profiler.freeze_outer_boundary(3)["available"] is True
    values = profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "Work/flight device-duration mismatch" in values[2]["error_message"]
    assert work.wait_calls == 0


def test_outstanding_async_work_fails_at_boundary_without_wait(tmp_path):
    entry = flight_entry(0, sequence=1)
    profiler, fake_torch = profiler_for(
        tmp_path, trace([]), boundary([entry]), trace([entry])
    )
    fake_torch.distributed.async_work.completed = False
    profiler.begin_outer(3)
    work = fake_torch.distributed.all_reduce(FakeTensor([4]), async_op=True)
    frozen = profiler.freeze_outer_boundary(3)
    assert frozen["available"] is False
    # Completion during later profiler overhead must never retroactively make
    # the call eligible.
    work.completed = True
    values = profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "remained outstanding at outer close" in values[2]["error_message"]
    assert work.wait_calls == 0


def test_dump_side_effect_cannot_make_outstanding_work_boundary_eligible(tmp_path):
    entry = flight_entry(0, sequence=1)
    holder = {}

    def completing_boundary_dump():
        holder["work"].completed = True
        return boundary([entry])

    dump = DumpSequence(trace([]), completing_boundary_dump, trace([entry]))
    fake_torch = FakeTorch(dump)
    profiler = FlightRecorderCollectiveProfiler(
        torch_module=fake_torch,
        rank=0,
        world_size=2,
        enabled=True,
        environ=environment(tmp_path),
        require_shared_nfs=False,
    )
    fake_torch.distributed.async_work.completed = False
    holder["work"] = fake_torch.distributed.async_work
    profiler.begin_outer(3)
    work = fake_torch.distributed.all_reduce(FakeTensor([4]), async_op=True)
    frozen = profiler.freeze_outer_boundary(3)
    assert dump.calls == 2
    assert work.completed is True
    assert frozen["available"] is False
    assert "remained outstanding at outer close" in frozen["error_message"]
    assert work.wait_calls == 0


@pytest.mark.parametrize(
    "ranks,error_fragment",
    [
        ("[0, 0]", "invalid process-group ranks"),
        ("[-1, 0]", "invalid process-group ranks"),
        ("[0, 2]", "exceed world_size"),
        ("[1]", "local rank 0 is absent"),
        ("[0]", "group-size mismatch"),
    ],
)
def test_invalid_process_group_rank_schema_fails_closed(
    tmp_path, ranks, error_fragment
):
    entry = flight_entry(0)
    profiler, fake_torch = profiler_for(
        tmp_path,
        trace([]),
        boundary([entry]) | {"pg_config": trace([], ranks=ranks)["pg_config"]},
        trace([entry], ranks=ranks),
    )
    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce(FakeTensor([4]))
    frozen = profiler.freeze_outer_boundary(3)
    assert frozen["available"] is False
    assert error_fragment in frozen["error_message"]


@pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), float("inf")])
def test_async_work_duration_must_be_positive_finite(tmp_path, duration):
    entry = flight_entry(0, sequence=1)
    profiler, fake_torch = profiler_for(
        tmp_path, trace([]), boundary([entry]), trace([entry])
    )
    fake_torch.distributed.async_work.duration_ms = duration
    profiler.begin_outer(3)
    work = fake_torch.distributed.all_reduce(FakeTensor([4]), async_op=True)
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
    fake_torch.distributed.all_reduce(FakeTensor([4]))
    assert profiler.freeze_outer_boundary(3)["available"] is True
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
    fake_torch.distributed.all_reduce(FakeTensor([4]))
    values = profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "!= pinned" in values[2]["error_message"]


def test_unsupported_collective_like_public_api_marks_outer_unavailable(tmp_path):
    profiler, fake_torch = profiler_for(tmp_path, trace([]))
    profiler.begin_outer(3)
    fake_torch.distributed.monitored_barrier()
    values = profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "unsupported object/P2P collective API" in values[2]["error_message"]


def test_gloo_object_control_collective_is_explicitly_excluded(tmp_path):
    entry = flight_entry(0)
    profiler, fake_torch = profiler_for(
        tmp_path, trace([]), boundary([entry]), trace([entry])
    )
    profiler.begin_outer(3)
    fake_torch.distributed.gather_object(
        {"metric": 1.0}, [None, None], dst=0,
        group=fake_torch.distributed.gloo_group,
    )
    fake_torch.distributed.all_reduce(FakeTensor([4]))
    assert profiler.freeze_outer_boundary(3)["available"] is True
    collective_bytes, collective_time, evidence = profiler.end_outer(3)
    assert collective_bytes == 16
    assert collective_time == pytest.approx(0.00025)
    exclusions = evidence["excluded_control_collectives"]
    assert exclusions == [
        {
            "public_api": "gather_object",
            "backend": "gloo",
            "group_name": "gloo-control",
            "group_size": 2,
            "group_resolution": "direct_process_group",
            "metric_inclusion": "excluded",
            "reason": (
                "verified CPU/Gloo object control collective outside the "
                "accelerator tensor-collective byte/time metric"
            ),
        }
    ]
    trace_payload = json.loads(open(evidence["trace_path"], encoding="utf-8").read())
    assert trace_payload["excluded_control_collectives"] == exclusions
    profiler.begin_outer(4)
    assert profiler._excluded_control_collectives == []
    reset_values = profiler.end_outer(4)
    assert reset_values[0] is None and reset_values[1] is None


def test_object_collective_on_non_gloo_backend_fails_closed(tmp_path):
    profiler, fake_torch = profiler_for(tmp_path, trace([]))
    profiler.begin_outer(3)
    fake_torch.distributed.gather_object(
        {"metric": 1.0}, [None, None], dst=0,
        group=fake_torch.distributed.group,
    )
    values = profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "backend must be exactly gloo" in values[2]["error_message"]


def test_direct_process_group_flight_entry_without_public_binding_fails(tmp_path):
    entry = flight_entry(0)
    profiler, _ = profiler_for(
        tmp_path, trace([]), boundary([entry]), trace([entry])
    )
    profiler.begin_outer(3)
    frozen = profiler.freeze_outer_boundary(3)
    assert frozen["available"] is False
    assert "public-call/flight-entry coverage mismatch" in frozen["error_message"]


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


def test_trace_root_rejects_parent_symlink_and_dotdot(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    fake_torch = FakeTorch(DumpSequence(trace([])))
    for root, fragment in (
        (linked / "child", "symlink component"),
        (tmp_path / "real" / ".." / "escape", "cannot contain '..'"),
    ):
        env = environment(tmp_path)
        env["MILES_P0B_TRACE_ROOT"] = str(root)
        with pytest.raises(CollectiveTraceError, match=fragment):
            FlightRecorderCollectiveProfiler(
                torch_module=fake_torch,
                rank=0,
                world_size=2,
                enabled=True,
                environ=env,
                require_shared_nfs=False,
            )


def test_precreated_trace_root_inode_swap_before_publication_fails_closed(tmp_path):
    entry = flight_entry(0, sequence=1)
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    fake_torch = FakeTorch(
        DumpSequence(trace([]), boundary([entry]), trace([entry]))
    )
    profiler = FlightRecorderCollectiveProfiler(
        torch_module=fake_torch,
        rank=0,
        world_size=2,
        enabled=True,
        environ=environment(tmp_path),
        require_shared_nfs=False,
    )
    profiler.begin_outer(3)
    fake_torch.distributed.all_reduce(FakeTensor([4]))
    assert profiler.freeze_outer_boundary(3)["available"] is True
    trace_root.rename(tmp_path / "original-traces")
    trace_root.mkdir()
    values = profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "inode identity drifted" in values[2]["error_message"]


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
    fake_torch.distributed.all_reduce(FakeTensor([2]))
    first_values = profiler.end_outer(3)
    profiler.begin_outer(4)
    fake_torch.distributed.all_reduce(FakeTensor([5]))
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
    fake_torch.distributed.all_reduce(FakeTensor([4]))
    first_evidence = profiler.end_outer(3)[2]
    assert first_evidence["available"] is True
    assert os.stat(first_evidence["trace_path"]).st_mode & 0o222 == 0

    second_profiler, second_torch = profiler_for(
        tmp_path, trace([]), boundary([entry]), trace([entry])
    )
    second_profiler.begin_outer(3)
    second_torch.distributed.all_reduce(FakeTensor([4]))
    values = second_profiler.end_outer(3)
    assert values[0] is None and values[1] is None
    assert "refusing to overwrite" in values[2]["error_message"]
