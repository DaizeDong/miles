"""Direct ProcessGroupNCCL flight-recorder evidence for P0-B.

The public byte metric is the sum of logical input-tensor bytes recorded by
the live c10d flight recorder on the primary training rank.  It is not wire
traffic and is never multiplied by world size.  Device time comes only from
the recorder's ``duration_ms`` field, which ProcessGroupNCCL derives from its
accelerator start/end events when timing is enabled.  Host discovery or
enqueue timestamps are retained as raw evidence but never used as duration.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import math
import os
import socket
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


TRACE_BUFFER_ENV = "TORCH_FR_BUFFER_SIZE"
TIMING_ENV = "TORCH_NCCL_ENABLE_TIMING"
TRACE_ROOT_ENV = "MILES_P0B_TRACE_ROOT"
MIN_TRACE_BUFFER_SIZE = 65536
TRACE_SCHEMA_VERSION = 1
POLL_TIMEOUT_SECONDS = 30.0
POLL_INTERVAL_SECONDS = 0.01

_DTYPE_BYTES = {
    "Bool": 1,
    "Byte": 1,
    "Char": 1,
    "Short": 2,
    "Int": 4,
    "Long": 8,
    "Half": 2,
    "Float": 4,
    "Double": 8,
    "BFloat16": 2,
    "ComplexHalf": 4,
    "ComplexFloat": 8,
    "ComplexDouble": 16,
}

# Exact low-level ProcessGroupNCCL names.  New names are unavailable until a
# reviewed parser version explicitly assigns byte semantics to them.
_SUPPORTED_OPS = {
    "all_reduce",
    "all_reduce_coalesced",
    "_all_gather_base",
    "all_gather",
    "all_gather_coalesced",
    "all_gather_into_tensor_coalesced",
    "_reduce_scatter_base",
    "reduce_scatter",
    "reduce_scatter_tensor_coalesced",
    "all_to_all",
    "all_to_all_single",
    "alltoall",
    "alltoall_base",
    "broadcast",
    "reduce",
    "gather",
    "scatter",
    "all_reduce_barrier",
    "barrier",
}
_BARRIER_OPS = {"all_reduce_barrier", "barrier"}
_PUBLIC_CALL_SPECS = {
    # name: (group positional index, async positional index, exact low-level
    # profiling-name suffixes accepted for the sequence-bound call)
    "all_reduce": (2, 3, frozenset({"all_reduce"})),
    "all_reduce_coalesced": (2, 3, frozenset({"all_reduce_coalesced"})),
    "all_gather": (2, 3, frozenset({"all_gather"})),
    "all_gather_coalesced": (2, 3, frozenset({"all_gather_coalesced"})),
    "all_gather_into_tensor": (2, 3, frozenset({"_all_gather_base"})),
    "_all_gather_base": (2, 3, frozenset({"_all_gather_base"})),
    "all_gather_into_tensor_coalesced": (
        2,
        3,
        frozenset({"all_gather_into_tensor_coalesced"}),
    ),
    "reduce_scatter": (3, 4, frozenset({"reduce_scatter"})),
    "reduce_scatter_tensor": (3, 4, frozenset({"_reduce_scatter_base"})),
    "_reduce_scatter_base": (3, 4, frozenset({"_reduce_scatter_base"})),
    "reduce_scatter_tensor_coalesced": (
        3,
        4,
        frozenset({"reduce_scatter_tensor_coalesced"}),
    ),
    "all_to_all": (2, 3, frozenset({"all_to_all", "alltoall"})),
    "all_to_all_single": (
        4,
        5,
        frozenset({"all_to_all_single", "alltoall_base"}),
    ),
    "broadcast": (2, 3, frozenset({"broadcast"})),
    "reduce": (3, 4, frozenset({"reduce"})),
    "scatter": (3, 4, frozenset({"scatter"})),
    "gather": (3, 4, frozenset({"gather"})),
    "barrier": (0, 1, frozenset({"all_reduce_barrier", "barrier"})),
}
# Object collectives and point-to-point APIs do not have the tensor-payload
# semantics claimed by ``collective_bytes``.  Observe them at the public API
# boundary so an implementation that happens to lower one through an
# allowlisted ProcessGroup operation cannot silently turn it into a tensor
# collective.
_UNSUPPORTED_HIGH_LEVEL_NAMES = {
    "all_gather_object",
    "batch_isend_irecv",
    "broadcast_object_list",
    "gather_object",
    "irecv",
    "isend",
    "monitored_barrier",
    "recv",
    "recv_object_list",
    "scatter_object_list",
    "send",
    "send_object_list",
}

_ACTIVE_TRACKERS: dict[int, "FlightRecorderCollectiveProfiler"] = {}
_TRACKER_GUARD = threading.local()


class CollectiveTraceError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounded(value: Any, limit: int = 512) -> str:
    rendered = str(value)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _shape_numel(shape: Any) -> int:
    if not isinstance(shape, list):
        raise CollectiveTraceError(f"flight trace tensor shape is not a list: {shape!r}")
    total = 1
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise CollectiveTraceError(f"invalid flight trace tensor dimension: {dimension!r}")
        total *= dimension
    return total


def _entry_input_bytes(entry: Mapping[str, Any], op: str) -> tuple[int, list[dict[str, Any]]]:
    dtypes = entry.get("input_dtypes")
    sizes = entry.get("input_sizes")
    if not isinstance(dtypes, list) or not isinstance(sizes, list) or len(dtypes) != len(sizes):
        raise CollectiveTraceError(
            f"flight trace {op} input dtype/size schema mismatch: {dtypes!r} / {sizes!r}"
        )
    tensors: list[dict[str, Any]] = []
    total = 0
    for dtype, shape in zip(dtypes, sizes, strict=True):
        if dtype not in _DTYPE_BYTES:
            raise CollectiveTraceError(f"unsupported flight trace dtype {dtype!r} for {op}")
        numel = _shape_numel(shape)
        logical_bytes = numel * _DTYPE_BYTES[dtype]
        tensors.append(
            {
                "dtype": dtype,
                "shape": shape,
                "numel": numel,
                "element_size": _DTYPE_BYTES[dtype],
                "logical_bytes": logical_bytes,
            }
        )
        total += logical_bytes
    # ProcessGroupNCCL implements barrier with an internal one-element
    # all-reduce.  That tensor is not a user API payload, so rank-local API
    # bytes are exactly zero while its device duration remains collective time.
    if op in _BARRIER_OPS:
        return 0, tensors
    if not tensors:
        raise CollectiveTraceError(f"supported flight trace op {op} has no input tensors")
    return total, tensors


def _parse_group_identity(
    trace: Mapping[str, Any], entry: Mapping[str, Any]
) -> tuple[str, int]:
    pg_id = entry.get("pg_id")
    config = trace.get("pg_config")
    if not isinstance(pg_id, int) or not isinstance(config, Mapping):
        raise CollectiveTraceError("flight trace lacks pg_id/pg_config")
    group = config.get(str(pg_id))
    if not isinstance(group, Mapping):
        raise CollectiveTraceError(f"flight trace lacks config for pg_id={pg_id}")
    name = group.get("name")
    if not isinstance(name, str) or not name:
        raise CollectiveTraceError(
            f"flight trace lacks stable process-group name for pg_id={pg_id}: {name!r}"
        )
    ranks_raw = group.get("ranks")
    try:
        ranks = ast.literal_eval(ranks_raw) if isinstance(ranks_raw, str) else ranks_raw
    except (SyntaxError, ValueError) as exc:
        raise CollectiveTraceError(f"invalid process-group ranks {ranks_raw!r}") from exc
    if (
        not isinstance(ranks, list)
        or not ranks
        or any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks)
    ):
        raise CollectiveTraceError(f"invalid process-group ranks {ranks!r}")
    return name, len(ranks)


def _canonical_op(entry: Mapping[str, Any]) -> str:
    name = entry.get("profiling_name")
    if not isinstance(name, str) or not name.startswith("nccl:"):
        raise CollectiveTraceError(f"unknown communication profiling_name={name!r}")
    op = name.split(":", 1)[1]
    if op not in _SUPPORTED_OPS:
        raise CollectiveTraceError(f"unsupported communication operation {name!r}")
    if entry.get("is_p2p") is not False:
        raise CollectiveTraceError(f"P2P entry is outside collective byte semantics: {name!r}")
    return op


def _positional_or_keyword(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    keyword: str,
    index: int,
    default: Any,
) -> Any:
    if keyword in kwargs:
        return kwargs[keyword]
    return args[index] if len(args) > index else default


def _async_flag(
    name: str,
    async_index: int,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> bool:
    if "async_op" in kwargs:
        value = kwargs["async_op"]
    else:
        value = args[async_index] if len(args) > async_index else False
    if not isinstance(value, bool):
        raise CollectiveTraceError(f"{name} async_op must be bool, got {value!r}")
    return value


def _install_collective_observers(dist_module: Any) -> None:
    """Install process-idempotent, non-mutating collective observers.

    The wrappers never alter ``async_op`` and never call ``Work.wait``.  They
    retain returned asynchronous Work objects only so natural completion can
    be checked after the workload's existing synchronization point.
    """

    for name, spec in _PUBLIC_CALL_SPECS.items():
        original = getattr(dist_module, name, None)
        if original is None or getattr(original, "_p0b_async_tracker_wrapper", False):
            continue

        @functools.wraps(original)
        def wrapped(
            *args: Any,
            __name: str = name,
            __spec: tuple[int, int, frozenset[str]] = spec,
            __original: Any = original,
            **kwargs: Any,
        ) -> Any:
            if getattr(_TRACKER_GUARD, "active", False):
                return __original(*args, **kwargs)
            tracker = _ACTIVE_TRACKERS.get(os.getpid())
            context: dict[str, Any] | None = None
            if tracker is not None and tracker.active_outer is not None:
                try:
                    context = tracker.prepare_public_call(
                        __name, __spec, args, kwargs
                    )
                except Exception as exc:
                    tracker.mark_unavailable(
                        f"public collective pre-call binding failed for {__name}: {exc}"
                    )
            _TRACKER_GUARD.active = True
            try:
                result = __original(*args, **kwargs)
            finally:
                _TRACKER_GUARD.active = False
            if tracker is not None and context is not None:
                try:
                    tracker.complete_public_call(context, result)
                except Exception as exc:
                    tracker.mark_unavailable(
                        f"public collective post-call binding failed for {__name}: {exc}"
                    )
            return result

        wrapped._p0b_async_tracker_wrapper = True  # type: ignore[attr-defined]
        setattr(dist_module, name, wrapped)

    for name in _UNSUPPORTED_HIGH_LEVEL_NAMES:
        original = getattr(dist_module, name, None)
        if original is None or getattr(original, "_p0b_unsupported_tracker_wrapper", False):
            continue

        @functools.wraps(original)
        def wrapped_unsupported(
            *args: Any,
            __name: str = name,
            __original: Any = original,
            **kwargs: Any,
        ) -> Any:
            if getattr(_TRACKER_GUARD, "active", False):
                return __original(*args, **kwargs)
            _TRACKER_GUARD.active = True
            try:
                result = __original(*args, **kwargs)
            finally:
                _TRACKER_GUARD.active = False
            tracker = _ACTIVE_TRACKERS.get(os.getpid())
            if tracker is not None and tracker.active_outer is not None:
                tracker.mark_unavailable(
                    f"unsupported object/P2P collective API observed: {__name}"
                )
            return result

        wrapped_unsupported._p0b_unsupported_tracker_wrapper = True  # type: ignore[attr-defined]
        setattr(dist_module, name, wrapped_unsupported)


class FlightRecorderCollectiveProfiler:
    """Collect one primary-rank outer from ProcessGroupNCCL flight evidence."""

    def __init__(
        self,
        *,
        torch_module: Any,
        rank: int,
        world_size: int,
        enabled: bool,
        environ: Mapping[str, str],
        require_shared_nfs: bool = True,
    ) -> None:
        self._torch = torch_module
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._enabled = bool(enabled)
        self._environ = environ
        self._require_shared_nfs = require_shared_nfs
        self.active_outer: int | None = None
        self._begin_record_id: int | None = None
        self._begin_version: str | None = None
        self._begin_nccl_version: str | None = None
        self._errors: list[str] = []
        self._observed_calls: list[dict[str, Any]] = []
        self._next_call_id = 0
        self._trace_root: Path | None = None
        self._runtime_identity: dict[str, Any] | None = None
        if not self._enabled:
            return
        self._validate_environment()
        self._runtime_identity = self._capture_runtime_identity()
        _install_collective_observers(torch_module.distributed)
        _ACTIVE_TRACKERS[os.getpid()] = self

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _validate_environment(self) -> None:
        raw_size = self._environ.get(TRACE_BUFFER_ENV)
        try:
            size = int(raw_size) if raw_size is not None else -1
        except ValueError as exc:
            raise CollectiveTraceError(f"{TRACE_BUFFER_ENV} must be an integer") from exc
        if str(size) != raw_size or size < MIN_TRACE_BUFFER_SIZE:
            raise CollectiveTraceError(
                f"{TRACE_BUFFER_ENV} must be canonical integer >= {MIN_TRACE_BUFFER_SIZE}, got {raw_size!r}"
            )
        if self._environ.get(TIMING_ENV) != "1":
            raise CollectiveTraceError(f"{TIMING_ENV} must be exactly '1'")
        root_raw = self._environ.get(TRACE_ROOT_ENV)
        if not isinstance(root_raw, str) or not root_raw:
            raise CollectiveTraceError(f"{TRACE_ROOT_ENV} is required")
        root = Path(root_raw)
        if not root.is_absolute():
            raise CollectiveTraceError(f"{TRACE_ROOT_ENV} must be absolute")
        if self._require_shared_nfs and not str(root).startswith("/shared_nfs/"):
            raise CollectiveTraceError(f"{TRACE_ROOT_ENV} must reside on /shared_nfs")
        if os.path.lexists(root) and (not root.is_dir() or root.is_symlink()):
            raise CollectiveTraceError(f"trace root is not a non-symlink directory: {root}")
        self._trace_root = root

    def _capture_runtime_identity(self) -> dict[str, Any]:
        job_id = self._environ.get("SLURM_JOB_ID")
        if (
            not isinstance(job_id, str)
            or not job_id.isdigit()
            or str(int(job_id)) != job_id
            or int(job_id) <= 0
        ):
            raise CollectiveTraceError(
                f"SLURM_JOB_ID must be a canonical positive integer, got {job_id!r}"
            )
        slurm_node = self._environ.get("SLURMD_NODENAME")
        if not isinstance(slurm_node, str) or not slurm_node.strip():
            raise CollectiveTraceError(
                f"SLURMD_NODENAME must be a non-empty string, got {slurm_node!r}"
            )
        hostname = socket.gethostname().split(".", 1)[0]
        if not hostname:
            raise CollectiveTraceError("hostname is empty")
        if hostname != slurm_node:
            raise CollectiveTraceError(
                f"Slurm node/hostname mismatch: SLURMD_NODENAME={slurm_node!r} hostname={hostname!r}"
            )
        pid = os.getpid()
        if pid <= 0:
            raise CollectiveTraceError(f"invalid process id {pid!r}")
        if self._rank < 0 or self._world_size <= 0 or self._rank >= self._world_size:
            raise CollectiveTraceError(
                f"invalid distributed identity rank={self._rank} world_size={self._world_size}"
            )
        return {
            "slurm_job_id": job_id,
            "slurm_node_name": slurm_node,
            "hostname": hostname,
            "pid": pid,
            "rank": self._rank,
            "world_size": self._world_size,
        }

    @staticmethod
    def _trace_versions(trace: Mapping[str, Any]) -> tuple[str, str]:
        version = trace.get("version")
        nccl_version = trace.get("nccl_version")
        if not isinstance(version, str) or not version.strip():
            raise CollectiveTraceError(
                f"flight recorder version must be a non-empty string, got {version!r}"
            )
        if not isinstance(nccl_version, str) or not nccl_version.strip():
            raise CollectiveTraceError(
                "flight recorder NCCL/RCCL version must be a non-empty string, "
                f"got {nccl_version!r}"
            )
        return version, nccl_version

    def _dump(self) -> dict[str, Any]:
        c10d = self._torch._C._distributed_c10d
        dump = getattr(c10d, "_dump_nccl_trace_json", None)
        if not callable(dump):
            raise CollectiveTraceError("torch c10d _dump_nccl_trace_json is unavailable")
        raw = dump()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not isinstance(raw, str):
            raise CollectiveTraceError(f"flight recorder returned {type(raw).__name__}, expected JSON")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CollectiveTraceError(f"flight recorder returned invalid JSON: {exc}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("entries"), list):
            raise CollectiveTraceError("flight recorder JSON lacks entries")
        return value

    @staticmethod
    def _record_ids(trace: Mapping[str, Any]) -> list[int]:
        record_ids: list[int] = []
        for entry in trace["entries"]:
            if not isinstance(entry, Mapping):
                raise CollectiveTraceError("flight recorder entry is not an object")
            record_id = entry.get("record_id")
            if isinstance(record_id, bool) or not isinstance(record_id, int) or record_id < 0:
                raise CollectiveTraceError(f"invalid flight recorder record_id={record_id!r}")
            record_ids.append(record_id)
        if len(record_ids) != len(set(record_ids)):
            raise CollectiveTraceError("flight recorder contains duplicate record IDs")
        record_ids.sort()
        if record_ids and record_ids != list(range(record_ids[0], record_ids[-1] + 1)):
            raise CollectiveTraceError(
                f"flight recorder record IDs are not contiguous: first={record_ids[:3]} "
                f"last={record_ids[-3:]}"
            )
        return record_ids

    def begin_outer(self, outer_id: int) -> None:
        if not self._enabled:
            return
        if self.active_outer is not None:
            raise CollectiveTraceError("collective profiler outer already active")
        trace = self._dump()
        record_ids = self._record_ids(trace)
        version, nccl_version = self._trace_versions(trace)
        current_identity = self._capture_runtime_identity()
        if current_identity != self._runtime_identity:
            raise CollectiveTraceError(
                "runtime job/node/process identity drifted before outer begin"
            )
        self.active_outer = int(outer_id)
        self._begin_record_id = max(record_ids, default=-1)
        self._begin_version = version
        self._begin_nccl_version = nccl_version
        self._errors = []
        self._observed_calls = []
        self._next_call_id = 0

    def _default_group(self) -> Any:
        dist_module = self._torch.distributed
        for container in (
            getattr(dist_module, "distributed_c10d", None),
            dist_module,
        ):
            function = getattr(container, "_get_default_group", None)
            if callable(function):
                return function()
        raise CollectiveTraceError("torch.distributed default process group is unavailable")

    def _group_name(self, group: Any) -> str:
        value = getattr(group, "group_name", None)
        if callable(value):
            value = value()
        if isinstance(value, str) and value:
            return value
        dist_module = self._torch.distributed
        for container in (
            getattr(dist_module, "distributed_c10d", None),
            dist_module,
        ):
            function = getattr(container, "_get_process_group_name", None)
            if callable(function):
                value = function(group)
                if isinstance(value, str) and value:
                    return value
        raise CollectiveTraceError("process group has no stable group_name")

    @staticmethod
    def _group_sequence(group: Any) -> int:
        function = getattr(group, "_get_sequence_number_for_group", None)
        if not callable(function):
            raise CollectiveTraceError(
                "process group lacks _get_sequence_number_for_group"
            )
        value = function()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CollectiveTraceError(
                f"invalid process-group collective sequence number {value!r}"
            )
        return value

    def prepare_public_call(
        self,
        name: str,
        spec: tuple[int, int, frozenset[str]],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.active_outer is None:
            raise CollectiveTraceError("public collective observed outside an active outer")
        group_index, async_index, expected_ops = spec
        group = _positional_or_keyword(args, kwargs, "group", group_index, None)
        if group is None:
            group = self._default_group()
        call_id = self._next_call_id
        self._next_call_id += 1
        return {
            "call_id": call_id,
            "public_api": name,
            "async_op": _async_flag(name, async_index, args, kwargs),
            "expected_low_level_ops": expected_ops,
            "group": group,
            "group_name": self._group_name(group),
            "sequence_before": self._group_sequence(group),
        }

    def complete_public_call(self, context: dict[str, Any], result: Any) -> None:
        sequence_after = self._group_sequence(context["group"])
        sequence_before = context["sequence_before"]
        if sequence_after != sequence_before + 1:
            raise CollectiveTraceError(
                f"{context['public_api']} changed process-group sequence "
                f"{sequence_before}->{sequence_after}, expected exactly one"
            )
        context = dict(context)
        context.pop("group")
        context["sequence_after"] = sequence_after
        context["work"] = result if context["async_op"] else None
        if context["async_op"] and (
            result is None or not callable(getattr(result, "is_completed", None))
        ):
            raise CollectiveTraceError(
                f"async {context['public_api']} did not return a trackable Work"
            )
        self._observed_calls.append(context)

    def mark_unavailable(self, message: str) -> None:
        self._errors.append(_bounded(message))

    def _boundary_delta(self, trace: Mapping[str, Any]) -> list[dict[str, Any]]:
        assert self._begin_record_id is not None
        self._trace_versions(trace)
        self._record_ids(trace)
        entries = [
            dict(entry)
            for entry in trace["entries"]
            if entry["record_id"] > self._begin_record_id
        ]
        entries.sort(key=lambda entry: entry["record_id"])
        if not entries:
            raise CollectiveTraceError("outer-close boundary has no collective entries")
        record_ids = [entry["record_id"] for entry in entries]
        expected_ids = list(range(self._begin_record_id + 1, record_ids[-1] + 1))
        if record_ids != expected_ids:
            raise CollectiveTraceError(
                f"flight recorder ring discontinuity at outer close: first={record_ids[:3]} "
                f"last={record_ids[-3:]} expected_start={self._begin_record_id + 1}"
            )
        for entry in entries:
            if entry.get("state") != "completed":
                raise CollectiveTraceError(
                    f"flight record {entry['record_id']} was not completed at outer close: "
                    f"state={entry.get('state')!r}"
                )
            started = entry.get("time_discovered_started_ns")
            completed = entry.get("time_discovered_completed_ns")
            if (
                isinstance(started, bool)
                or not isinstance(started, int)
                or started <= 0
                or isinstance(completed, bool)
                or not isinstance(completed, int)
                or completed < started
            ):
                raise CollectiveTraceError(
                    f"flight record {entry['record_id']} lacks valid outer-close discovery timestamps"
                )
        return entries

    def _poll_retired_delta(
        self,
        frozen_record_ids: list[int],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        """Wait only for retirement metadata on the frozen outer-close set."""

        assert self._begin_record_id is not None
        poll_started = time.monotonic()
        deadline = poll_started + POLL_TIMEOUT_SECONDS
        poll_count = 0
        last_delta: list[dict[str, Any]] = []
        while True:
            trace = self._dump()
            poll_count += 1
            self._trace_versions(trace)
            self._record_ids(trace)
            entries = [
                dict(entry)
                for entry in trace["entries"]
                if entry["record_id"] > self._begin_record_id
            ]
            entries.sort(key=lambda entry: entry["record_id"])
            last_delta = entries
            current_record_ids = [entry["record_id"] for entry in entries]
            if current_record_ids != frozen_record_ids:
                raise CollectiveTraceError(
                    "flight recorder outer-close set changed during retirement poll: "
                    f"frozen={frozen_record_ids[:3]}...{frozen_record_ids[-3:]} "
                    f"current={current_record_ids[:3]}...{current_record_ids[-3:]}"
                )
            all_retired = bool(entries) and all(
                entry.get("state") == "completed"
                and entry.get("retired") is True
                and entry.get("duration_ms") is not None
                for entry in entries
            )
            if all_retired:
                return trace, entries, {
                    "poll_count": poll_count,
                    "poll_elapsed_seconds": time.monotonic() - poll_started,
                    "poll_timeout_seconds": POLL_TIMEOUT_SECONDS,
                    "record_id_set_frozen_at_outer_close": True,
                }
            if time.monotonic() >= deadline:
                states = [
                    {
                        "record_id": entry.get("record_id"),
                        "state": entry.get("state"),
                        "retired": entry.get("retired"),
                        "duration_ms": entry.get("duration_ms"),
                    }
                    for entry in last_delta[-8:]
                ]
                raise CollectiveTraceError(
                    f"flight recorder entries did not retire within {POLL_TIMEOUT_SECONDS}s: {states}"
                )
            time.sleep(POLL_INTERVAL_SECONDS)

    def _validate_call_bindings(
        self,
        trace: Mapping[str, Any],
        entries: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
        """Bind public calls to flight entries by exact PG sequence identity.

        Operation-order guessing is forbidden: a sync and async all-reduce can
        have the same name.  The binding key is instead the stable process
        group name plus its exact post-enqueue collective sequence number.
        """

        if len(self._observed_calls) != len(entries):
            raise CollectiveTraceError(
                "public-call/flight-entry coverage mismatch at outer close: "
                f"calls={len(self._observed_calls)} entries={len(entries)}"
            )
        flight_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for entry in entries:
            group_name, _ = _parse_group_identity(trace, entry)
            sequence = entry.get("collective_seq_id")
            if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
                raise CollectiveTraceError(
                    f"flight record {entry['record_id']} has invalid collective_seq_id={sequence!r}"
                )
            key = (group_name, sequence)
            if key in flight_by_key:
                raise CollectiveTraceError(
                    f"ambiguous duplicate flight process-group sequence key {key!r}"
                )
            flight_by_key[key] = entry

        call_evidence: list[dict[str, Any]] = []
        by_record_id: dict[int, dict[str, Any]] = {}
        for call in self._observed_calls:
            key = (call["group_name"], call["sequence_after"])
            entry = flight_by_key.get(key)
            if entry is None:
                raise CollectiveTraceError(
                    f"public call_id={call['call_id']} has no exact flight sequence binding: {key!r}"
                )
            operation = _canonical_op(entry)
            if operation not in call["expected_low_level_ops"]:
                raise CollectiveTraceError(
                    f"public call_id={call['call_id']} API={call['public_api']} bound to "
                    f"unexpected operation {operation!r}"
                )
            evidence = {
                "call_id": call["call_id"],
                "public_api": call["public_api"],
                "async_op": call["async_op"],
                "group_name": call["group_name"],
                "sequence_before": call["sequence_before"],
                "collective_seq_id": call["sequence_after"],
                "flight_record_id": entry["record_id"],
                "flight_operation": operation,
                "binding": "exact_process_group_name_and_collective_sequence",
            }
            if call["async_op"]:
                work = call["work"]
                if not bool(work.is_completed()):
                    raise CollectiveTraceError(
                        f"async call_id={call['call_id']} {call['public_api']} Work "
                        "remained outstanding at outer close"
                    )
                get_duration = getattr(work, "_get_duration", None)
                if not callable(get_duration):
                    raise CollectiveTraceError(
                        f"async call_id={call['call_id']} {call['public_api']} Work "
                        "lacks direct device duration"
                    )
                work_duration_ms = float(get_duration())
                if not math.isfinite(work_duration_ms) or work_duration_ms <= 0:
                    raise CollectiveTraceError(
                        f"async call_id={call['call_id']} {call['public_api']} Work "
                        f"duration is invalid: {work_duration_ms!r}"
                    )
                evidence.update(
                    {
                        "work_completed_at_outer_close_without_profiler_wait": True,
                        "work_device_duration_ms": work_duration_ms,
                        "work_duration_source": (
                            "ProcessGroupNCCL Work._get_duration accelerator events"
                        ),
                        "work_duration_compared_to_flight_duration": False,
                    }
                )
            call_evidence.append(evidence)
            by_record_id[entry["record_id"]] = evidence
        if len(by_record_id) != len(entries):
            raise CollectiveTraceError(
                "not every flight entry has a unique exact public-call binding"
            )
        call_evidence.sort(key=lambda item: item["call_id"])
        return call_evidence, by_record_id

    def _write_trace(self, outer_id: int, payload: dict[str, Any]) -> tuple[Path, int, str]:
        assert self._trace_root is not None
        if not os.path.lexists(self._trace_root):
            self._trace_root.mkdir(parents=True, mode=0o755)
        if not self._trace_root.is_dir() or self._trace_root.is_symlink():
            raise CollectiveTraceError(f"trace root is unsafe: {self._trace_root}")
        path = self._trace_root / f"outer-{outer_id:04d}.collectives.json"
        if os.path.lexists(path):
            raise CollectiveTraceError(f"refusing to overwrite collective trace: {path}")
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            digest = _sha256_file(temporary)
            os.chmod(temporary, 0o444)
            try:
                # Atomic create-once publication: unlike os.replace(), link()
                # cannot overwrite a target that appeared after the lstat.
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise CollectiveTraceError(
                    f"refusing to overwrite collective trace: {path}"
                ) from exc
            temporary.unlink()
        finally:
            if temporary.exists():
                temporary.unlink()
        return path, path.stat().st_size, digest

    def end_outer(self, outer_id: int) -> tuple[int | None, float | None, dict[str, Any]]:
        if not self._enabled:
            return None, None, {"available": False, "reason": "disabled"}
        try:
            if self.active_outer != int(outer_id) or self._begin_record_id is None:
                raise CollectiveTraceError(
                    f"collective profiler active outer={self.active_outer}, requested={outer_id}"
                )
            if self._errors:
                raise CollectiveTraceError("; ".join(self._errors))
            current_identity = self._capture_runtime_identity()
            if current_identity != self._runtime_identity:
                raise CollectiveTraceError(
                    "runtime job/node/process identity drifted within outer"
                )
            # Freeze the exact evidence set at the canonical outer-close
            # boundary.  Polling below may wait for recorder retirement
            # metadata only; it cannot make a late Work/event eligible.
            boundary_trace = self._dump()
            boundary_version, boundary_nccl_version = self._trace_versions(
                boundary_trace
            )
            if boundary_version != self._begin_version:
                raise CollectiveTraceError(
                    "flight recorder schema version drifted before outer close"
                )
            if boundary_nccl_version != self._begin_nccl_version:
                raise CollectiveTraceError(
                    "flight recorder NCCL/RCCL version drifted before outer close"
                )
            boundary_entries = self._boundary_delta(boundary_trace)
            record_ids = [entry["record_id"] for entry in boundary_entries]
            call_evidence, call_binding_by_record_id = self._validate_call_bindings(
                boundary_trace, boundary_entries
            )
            async_evidence = [
                item for item in call_evidence if item["async_op"]
            ]
            trace, entries, retirement_poll = self._poll_retired_delta(record_ids)
            version, nccl_version = self._trace_versions(trace)
            if version != self._begin_version:
                raise CollectiveTraceError("flight recorder schema version drifted within outer")
            if nccl_version != self._begin_nccl_version:
                raise CollectiveTraceError("flight recorder NCCL/RCCL version drifted within outer")
            trace_capacity = int(self._environ[TRACE_BUFFER_ENV])
            if len(entries) >= trace_capacity:
                raise CollectiveTraceError(
                    f"flight recorder delta count {len(entries)} reached buffer capacity "
                    f"{trace_capacity}; ring-wrap absence cannot be proven"
                )

            total_bytes = 0
            total_duration_ms = 0.0
            group_histogram: Counter[str] = Counter()
            op_histogram: Counter[str] = Counter()
            parsed_entries: list[dict[str, Any]] = []
            boundary_by_record_id = {
                entry["record_id"]: entry for entry in boundary_entries
            }
            for entry in entries:
                op = _canonical_op(entry)
                duration = entry.get("duration_ms")
                if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                    raise CollectiveTraceError(
                        f"flight trace record {entry['record_id']} lacks numeric device duration"
                    )
                duration_ms = float(duration)
                if not math.isfinite(duration_ms) or duration_ms <= 0:
                    raise CollectiveTraceError(
                        f"flight trace record {entry['record_id']} has invalid duration {duration!r}"
                    )
                logical_bytes, tensors = _entry_input_bytes(entry, op)
                group_name, group_size = _parse_group_identity(trace, entry)
                if group_size > self._world_size:
                    raise CollectiveTraceError(
                        f"process-group size {group_size} exceeds world_size={self._world_size}"
                    )
                total_bytes += logical_bytes
                total_duration_ms += duration_ms
                op_histogram[op] += 1
                group_histogram[str(group_size)] += 1
                parsed_entries.append(
                    {
                        "record_id": entry["record_id"],
                        "collective_seq_id": entry.get("collective_seq_id"),
                        "op_id": entry.get("op_id"),
                        "pg_id": entry.get("pg_id"),
                        "process_group": entry.get("process_group"),
                        "group_name": group_name,
                        "group_size": group_size,
                        "operation": op,
                        "rank_local_logical_input_bytes": logical_bytes,
                        "input_tensors": tensors,
                        "device_duration_ms": duration_ms,
                        "duration_source": "ProcessGroupNCCL flight recorder accelerator events",
                        "state": entry.get("state"),
                        "retired": entry.get("retired"),
                        "public_call_binding": call_binding_by_record_id[
                            entry["record_id"]
                        ],
                        "outer_close_boundary": {
                            "state": boundary_by_record_id[entry["record_id"]].get("state"),
                            "retired": boundary_by_record_id[entry["record_id"]].get("retired"),
                            "duration_ms": boundary_by_record_id[entry["record_id"]].get(
                                "duration_ms"
                            ),
                            "time_created_ns": boundary_by_record_id[entry["record_id"]].get(
                                "time_created_ns"
                            ),
                            "time_discovered_started_ns": boundary_by_record_id[
                                entry["record_id"]
                            ].get("time_discovered_started_ns"),
                            "time_discovered_completed_ns": boundary_by_record_id[
                                entry["record_id"]
                            ].get("time_discovered_completed_ns"),
                        },
                        "retired_trace_host_discovery": {
                            "time_created_ns": entry.get("time_created_ns"),
                            "time_discovered_started_ns": entry.get(
                                "time_discovered_started_ns"
                            ),
                            "time_discovered_completed_ns": entry.get(
                                "time_discovered_completed_ns"
                            ),
                            "used_for_device_duration": False,
                        },
                    }
                )
            if total_bytes <= 0:
                raise CollectiveTraceError("ABBA outer has no positive collective input payload")
            if not math.isfinite(total_duration_ms) or total_duration_ms <= 0:
                raise CollectiveTraceError("ABBA outer has no positive collective device duration")
            parser_path = Path(__file__).resolve()
            parser_sha = _sha256_file(parser_path)
            assert self._runtime_identity is not None
            payload = {
                "schema_version": TRACE_SCHEMA_VERSION,
                "kind": "p0b-primary-rank-collective-flight-trace-v2",
                "outer_id": int(outer_id),
                "rank": self._rank,
                "world_size": self._world_size,
                "runtime_identity": dict(self._runtime_identity),
                "trace_version": trace.get("version"),
                "nccl_version": trace.get("nccl_version"),
                "begin_record_id": self._begin_record_id,
                "end_record_id": record_ids[-1],
                "record_ids_contiguous": True,
                "record_id_set_frozen_at_outer_close": True,
                "new_records_during_retirement_poll": 0,
                "retirement_poll": retirement_poll,
                "environment": {
                    TRACE_BUFFER_ENV: self._environ[TRACE_BUFFER_ENV],
                    TIMING_ENV: self._environ[TIMING_ENV],
                    TRACE_ROOT_ENV: str(self._trace_root),
                },
                "semantics": {
                    "collective_bytes": (
                        "sum of logical input-tensor bytes directly recorded at supported c10d "
                        "collective calls issued by this rank; not multiplied by group size and not wire bytes"
                    ),
                    "collective_time_seconds": (
                        "sum of ProcessGroupNCCL accelerator-event duration_ms for completed retired "
                        "collective entries; host enqueue/discovery timestamps excluded"
                    ),
                },
                "pg_config": trace.get("pg_config"),
                "pg_status": trace.get("pg_status"),
                "operation_histogram": dict(sorted(op_histogram.items())),
                "group_size_histogram": dict(sorted(group_histogram.items())),
                "public_call_evidence": call_evidence,
                "async_work_evidence": async_evidence,
                "total_rank_local_logical_input_bytes": total_bytes,
                "total_device_duration_ms": total_duration_ms,
                "entries": parsed_entries,
            }
            path, size, digest = self._write_trace(int(outer_id), payload)
            evidence = {
                "available": True,
                "scope": "primary_training_rank_rank_local_api_payload_and_device_kernels",
                "rank": self._rank,
                "world_size": self._world_size,
                "runtime_identity": dict(self._runtime_identity),
                "trace_path": str(path),
                "trace_size_bytes": size,
                "trace_sha256": digest,
                "parser_path": str(parser_path),
                "parser_sha256": parser_sha,
                "trace_schema_version": TRACE_SCHEMA_VERSION,
                "begin_record_id": self._begin_record_id,
                "end_record_id": record_ids[-1],
                "entry_count": len(entries),
                "trace_buffer_capacity": trace_capacity,
                "retirement_poll": retirement_poll,
                "operation_histogram": dict(sorted(op_histogram.items())),
                "group_size_histogram": dict(sorted(group_histogram.items())),
                "async_work_count": len(async_evidence),
                "public_call_count": len(call_evidence),
                "public_call_binding": (
                    "exact_process_group_name_and_collective_sequence"
                ),
                "outstanding_async_work_count": 0,
                "rank_local_logical_input_bytes": total_bytes,
                "device_duration_ms": total_duration_ms,
                "duration_source": "ProcessGroupNCCL flight recorder accelerator events",
                "host_discovery_timestamps_used_for_duration": False,
                "world_size_multiplier_applied": False,
            }
            return total_bytes, total_duration_ms / 1000.0, evidence
        except Exception as exc:
            return None, None, {
                "available": False,
                "error_type": type(exc).__name__,
                "error_message": _bounded(exc),
                "rank": self._rank,
                "world_size": self._world_size,
                "runtime_identity": self._runtime_identity,
                "host_discovery_timestamps_used_for_duration": False,
                "world_size_multiplier_applied": False,
            }
        finally:
            self.active_outer = None
            self._begin_record_id = None
            self._begin_version = None
            self._begin_nccl_version = None
            self._observed_calls = []
            self._next_call_id = 0
