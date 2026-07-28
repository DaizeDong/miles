"""Direct ProcessGroupNCCL flight-recorder evidence for P0-B.

The public byte metric is the sum of ``numel * element_size`` from actual
input tensor arguments at allowlisted public collective calls on the primary
training rank.  It is cross-checked against the live c10d flight recorder,
is not wire traffic, and is never multiplied by world size.  Device time
comes only from the recorder's ``duration_ms`` field, which ProcessGroupNCCL
derives from accelerator start/end events when timing is enabled.  Host
discovery or enqueue timestamps are retained but never used as duration.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import functools
import hashlib
import json
import math
import os
import socket
import sys
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
TRACE_SCHEMA_VERSION = 2
PINNED_FLIGHT_TRACE_VERSION = "2.9"
PINNED_RCCL_VERSION = "2.26.6"
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
    "allreduce_coalesced",
    "_all_gather_base",
    "all_gather",
    "all_gather_coalesced",
    "all_gather_into_tensor_coalesced",
    "_reduce_scatter_base",
    "reduce_scatter",
    "reduce_scatter_tensor_coalesced",
    "all_to_all",
    "alltoall",
    "broadcast",
    "reduce",
    "gather",
    "scatter",
    "all_reduce_barrier",
    "barrier",
}
_BARRIER_OPS = {"all_reduce_barrier", "barrier"}
@dataclasses.dataclass(frozen=True)
class _PublicCallSpec:
    group_index: int
    async_index: int
    expected_low_level_ops: frozenset[str]
    input_index: int | None
    input_keyword: str | None
    input_optional: bool = False


_PUBLIC_CALL_SPECS = {
    # name: (group positional index, async positional index, exact low-level
    # profiling-name suffixes accepted for the sequence-bound call), plus the
    # exact public API input-tensor argument.  Output buffers are never counted.
    "all_reduce": _PublicCallSpec(2, 3, frozenset({"all_reduce"}), 0, "tensor"),
    "all_reduce_coalesced": _PublicCallSpec(
        2, 3, frozenset({"allreduce_coalesced"}), 0, "tensors"
    ),
    "all_gather": _PublicCallSpec(2, 3, frozenset({"all_gather"}), 1, "tensor"),
    "all_gather_coalesced": _PublicCallSpec(
        2, 3, frozenset({"all_gather_coalesced"}), 1, "input_tensor_list"
    ),
    "all_gather_into_tensor": _PublicCallSpec(
        2, 3, frozenset({"_all_gather_base"}), 1, "input_tensor"
    ),
    "_all_gather_base": _PublicCallSpec(
        2, 3, frozenset({"_all_gather_base"}), 1, "input_tensor"
    ),
    "all_gather_into_tensor_coalesced": _PublicCallSpec(
        2,
        3,
        frozenset({"all_gather_into_tensor_coalesced"}),
        1,
        "input_tensor_list",
    ),
    "reduce_scatter": _PublicCallSpec(
        3, 4, frozenset({"reduce_scatter"}), 1, "input_list"
    ),
    "reduce_scatter_tensor": _PublicCallSpec(
        3, 4, frozenset({"_reduce_scatter_base"}), 1, "input"
    ),
    "_reduce_scatter_base": _PublicCallSpec(
        3, 4, frozenset({"_reduce_scatter_base"}), 1, "input"
    ),
    "reduce_scatter_tensor_coalesced": _PublicCallSpec(
        3,
        4,
        frozenset({"reduce_scatter_tensor_coalesced"}),
        1,
        "input_tensor_list",
    ),
    "all_to_all": _PublicCallSpec(
        2, 3, frozenset({"all_to_all", "alltoall"}), 1, "input_tensor_list"
    ),
    "all_to_all_single": _PublicCallSpec(
        4,
        5,
        frozenset({"all_to_all"}),
        1,
        "input",
    ),
    "broadcast": _PublicCallSpec(
        2, 3, frozenset({"broadcast"}), 0, "tensor"
    ),
    "reduce": _PublicCallSpec(3, 4, frozenset({"reduce"}), 0, "tensor"),
    "scatter": _PublicCallSpec(
        3, 4, frozenset({"scatter"}), 1, "scatter_list", input_optional=True
    ),
    "gather": _PublicCallSpec(3, 4, frozenset({"gather"}), 0, "tensor"),
    "barrier": _PublicCallSpec(
        0, 1, frozenset({"all_reduce_barrier", "barrier"}), None, None
    ),
}
# CPU/Gloo object collectives are control-plane traffic, not accelerator tensor
# collectives.  They are explicitly recorded and excluded only after binding
# the live process-group backend to Gloo.  Object calls on any other backend,
# and all tensor P2P APIs, remain fail-closed.
_GLOO_CONTROL_OBJECT_GROUP_INDEX = {
    "gather_object": 3,
}
_UNSUPPORTED_HIGH_LEVEL_NAMES = {
    "all_gather_object",
    "batch_isend_irecv",
    "broadcast_object_list",
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

_MEGATRON_DIRECT_ALIASES = {
    "megatron.core.tensor_parallel.mappings": {
        "dist_all_gather_func": "all_gather_into_tensor",
        "dist_reduce_scatter_func": "reduce_scatter_tensor",
    },
    "megatron.core.tensor_parallel.layers": {
        "dist_all_gather_func": "all_gather_into_tensor",
        "dist_reduce_scatter_func": "reduce_scatter_tensor",
    },
    "megatron.core.tensor_parallel.utils": {
        "dist_all_gather_func": "all_gather_into_tensor",
    },
    "megatron.core.timers": {
        "dist_all_gather_func": "all_gather_into_tensor",
    },
    "megatron.core.transformer.multi_token_prediction": {
        "dist_all_gather_func": "all_gather_into_tensor",
    },
}
_MEGATRON_COALESCING_MODULE = "megatron.core.distributed.param_and_grad_buffer"


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
    trace: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    local_rank: int | None = None,
    world_size: int | None = None,
) -> tuple[str, int]:
    pg_id = entry.get("pg_id")
    config = trace.get("pg_config")
    if (
        isinstance(pg_id, bool)
        or not isinstance(pg_id, int)
        or pg_id < 0
        or not isinstance(config, Mapping)
    ):
        raise CollectiveTraceError("flight trace lacks pg_id/pg_config")
    group = config.get(str(pg_id))
    if not isinstance(group, Mapping):
        raise CollectiveTraceError(f"flight trace lacks config for pg_id={pg_id}")
    name = group.get("name")
    if not isinstance(name, str) or not name:
        raise CollectiveTraceError(
            f"flight trace lacks stable process-group name for pg_id={pg_id}: {name!r}"
        )
    matching_names = [
        key
        for key, candidate in config.items()
        if isinstance(candidate, Mapping) and candidate.get("name") == name
    ]
    if matching_names != [str(pg_id)]:
        raise CollectiveTraceError(
            f"flight trace process-group name {name!r} is not uniquely bound to pg_id={pg_id}"
        )
    ranks_raw = group.get("ranks")
    try:
        ranks = ast.literal_eval(ranks_raw) if isinstance(ranks_raw, str) else ranks_raw
    except (SyntaxError, ValueError) as exc:
        raise CollectiveTraceError(f"invalid process-group ranks {ranks_raw!r}") from exc
    if (
        not isinstance(ranks, list)
        or not ranks
        or any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
            for rank in ranks
        )
        or len(ranks) != len(set(ranks))
    ):
        raise CollectiveTraceError(f"invalid process-group ranks {ranks!r}")
    if world_size is not None and any(rank >= world_size for rank in ranks):
        raise CollectiveTraceError(
            f"process-group ranks {ranks!r} exceed world_size={world_size}"
        )
    if local_rank is not None and local_rank not in ranks:
        raise CollectiveTraceError(
            f"local rank {local_rank} is absent from process-group ranks {ranks!r}"
        )
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


_MISSING_ARGUMENT = object()


def _runtime_tensor_evidence(value: Any, torch_module: Any) -> dict[str, Any]:
    try:
        is_tensor = bool(torch_module.is_tensor(value))
    except Exception as exc:
        raise CollectiveTraceError(
            f"runtime tensor type check failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not is_tensor:
        raise CollectiveTraceError(
            f"runtime collective input is not a tensor: {type(value).__name__}"
        )
    try:
        numel = int(value.numel())
        element_size = int(value.element_size())
        shape = [int(dimension) for dimension in value.shape]
        is_contiguous = bool(value.is_contiguous())
    except Exception as exc:
        raise CollectiveTraceError(
            f"cannot inspect runtime collective tensor: {type(exc).__name__}: {exc}"
        ) from exc
    if numel < 0 or element_size <= 0 or any(dimension < 0 for dimension in shape):
        raise CollectiveTraceError(
            f"invalid runtime tensor schema numel={numel} element_size={element_size} shape={shape}"
        )
    logical_bytes = numel * element_size
    return {
        "dtype": str(value.dtype),
        "device": str(value.device),
        "shape": shape,
        "numel": numel,
        "element_size": element_size,
        "logical_bytes": logical_bytes,
        "is_contiguous": is_contiguous,
    }


def _runtime_input_payload(
    name: str,
    spec: _PublicCallSpec,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    torch_module: Any,
) -> tuple[int, list[dict[str, Any]]]:
    """Extract logical input bytes from the actual public API arguments."""

    if spec.input_index is None or spec.input_keyword is None:
        if name != "barrier":
            raise CollectiveTraceError(f"{name} has no reviewed runtime input extractor")
        return 0, []
    value = _positional_or_keyword(
        args,
        kwargs,
        spec.input_keyword,
        spec.input_index,
        _MISSING_ARGUMENT,
    )
    if value is _MISSING_ARGUMENT:
        raise CollectiveTraceError(
            f"{name} lacks required runtime input argument {spec.input_keyword!r}"
        )
    if value is None:
        if spec.input_optional:
            return 0, []
        raise CollectiveTraceError(
            f"{name} runtime input argument {spec.input_keyword!r} is None"
        )

    tensors: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        try:
            is_tensor = bool(torch_module.is_tensor(item))
        except Exception as exc:
            raise CollectiveTraceError(
                f"{name} runtime input type check failed: {type(exc).__name__}: {exc}"
            ) from exc
        if is_tensor:
            tensors.append(_runtime_tensor_evidence(item, torch_module))
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        raise CollectiveTraceError(
            f"{name} runtime input contains unsupported {type(item).__name__}"
        )

    visit(value)
    if not tensors and not spec.input_optional:
        raise CollectiveTraceError(f"{name} runtime input contains no tensors")
    return sum(item["logical_bytes"] for item in tensors), tensors


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
            __spec: _PublicCallSpec = spec,
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

    for name, group_index in _GLOO_CONTROL_OBJECT_GROUP_INDEX.items():
        original = getattr(dist_module, name, None)
        if original is None or getattr(original, "_p0b_gloo_control_tracker_wrapper", False):
            continue

        @functools.wraps(original)
        def wrapped_gloo_control(
            *args: Any,
            __name: str = name,
            __group_index: int = group_index,
            __original: Any = original,
            **kwargs: Any,
        ) -> Any:
            if getattr(_TRACKER_GUARD, "active", False):
                return __original(*args, **kwargs)
            tracker = _ACTIVE_TRACKERS.get(os.getpid())
            context: dict[str, Any] | None = None
            if tracker is not None and tracker.active_outer is not None:
                try:
                    context = tracker.prepare_gloo_control_object_call(
                        __name, __group_index, args, kwargs
                    )
                except Exception as exc:
                    tracker.mark_unavailable(
                        f"object collective backend binding failed for {__name}: {exc}"
                    )
            # Suppress the object's private tensor lowering.  It belongs to the
            # verified CPU/Gloo control call and is outside the accelerator
            # tensor-collective metric; recording it as a public tensor API
            # would double-classify one workload action.
            _TRACKER_GUARD.active = True
            try:
                result = __original(*args, **kwargs)
            finally:
                _TRACKER_GUARD.active = False
            if tracker is not None and context is not None:
                tracker.complete_gloo_control_object_call(context)
            return result

        wrapped_gloo_control._p0b_gloo_control_tracker_wrapper = True  # type: ignore[attr-defined]
        setattr(dist_module, name, wrapped_gloo_control)

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


def _install_megatron_collective_alias_observers(dist_module: Any) -> list[dict[str, str]]:
    """Repair import-time Megatron aliases without changing their workload.

    Several Megatron modules bind public distributed functions during import,
    before MILES and P0-B install their wrappers.  Direct aliases are rebound
    to the already wrapped public API.  The param/grad buffer is different: it
    intentionally keeps its raw tensor functions inside Torch 2.9's fast-path
    coalescing manager, so only that manager alias is wrapped and its private
    ``_CollOp`` transaction is observed without modifying enqueue behavior.
    """

    bindings: list[dict[str, str]] = []
    for module_name, aliases in _MEGATRON_DIRECT_ALIASES.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for alias_name, public_name in aliases.items():
            old = getattr(module, alias_name, None)
            replacement = getattr(dist_module, public_name, None)
            if not callable(old) or not callable(replacement):
                raise CollectiveTraceError(
                    f"Megatron alias {module_name}.{alias_name} is unavailable"
                )
            expected_names = {public_name}
            if public_name == "all_gather_into_tensor":
                expected_names.add("_all_gather_base")
            elif public_name == "reduce_scatter_tensor":
                expected_names.add("_reduce_scatter_base")
            old_name = getattr(old, "__name__", None)
            if old is not replacement and old_name not in expected_names:
                raise CollectiveTraceError(
                    f"Megatron alias {module_name}.{alias_name} has unexpected "
                    f"identity {old_name!r}"
                )
            setattr(module, alias_name, replacement)
            bindings.append(
                {
                    "module": module_name,
                    "alias": alias_name,
                    "public_api": public_name,
                    "binding": "p0b_wrapped_public_api",
                }
            )

    module = sys.modules.get(_MEGATRON_COALESCING_MODULE)
    if module is None:
        return bindings
    raw_manager = getattr(module, "_coalescing_manager", None)
    if not callable(raw_manager):
        raise CollectiveTraceError("Megatron param/grad coalescing manager is unavailable")
    if getattr(raw_manager, "_p0b_coalescing_tracker_wrapper", False):
        setattr(dist_module, "_coalescing_manager", raw_manager)
        bindings.append(
            {
                "module": _MEGATRON_COALESCING_MODULE,
                "alias": "_coalescing_manager",
                "public_api": "_coalescing_manager",
                "binding": "existing_p0b_coalescing_transaction_wrapper",
            }
        )
        return bindings
    if getattr(raw_manager, "__name__", None) != "_coalescing_manager":
        raise CollectiveTraceError(
            "Megatron param/grad coalescing manager is not the pinned raw Torch API"
        )

    @contextlib.contextmanager
    @functools.wraps(raw_manager)
    def wrapped_coalescing_manager(*args: Any, **kwargs: Any):
        tracker = _ACTIVE_TRACKERS.get(os.getpid())
        transaction: dict[str, Any] | None = None
        if tracker is not None and tracker.active_outer is not None:
            try:
                transaction = tracker.prepare_coalescing_manager(args, kwargs)
            except Exception as exc:
                tracker.mark_unavailable(
                    f"coalescing-manager pre-entry observation failed: {exc}"
                )
        manager: Any = None
        body_completed = False
        try:
            with raw_manager(*args, **kwargs) as manager:
                yield manager
                body_completed = True
                if transaction is not None:
                    try:
                        tracker.capture_coalescing_manager(transaction)
                    except Exception as exc:
                        transaction["observation_error"] = _bounded(exc)
                        tracker.mark_unavailable(
                            f"coalescing-manager pre-enqueue observation failed: {exc}"
                        )
        except BaseException:
            raise
        else:
            if transaction is not None and body_completed:
                try:
                    tracker.complete_coalescing_manager(transaction, manager)
                except Exception as exc:
                    tracker.mark_unavailable(
                        f"coalescing-manager post-enqueue observation failed: {exc}"
                    )

    wrapped_coalescing_manager._p0b_coalescing_tracker_wrapper = True  # type: ignore[attr-defined]
    setattr(module, "_coalescing_manager", wrapped_coalescing_manager)
    # Future imports must receive the same transaction-preserving wrapper.
    # This deliberately bypasses MILES' generic group-unwrapping wrapper: the
    # raw manager and its raw tensor aliases must use the same wrapper PG key.
    setattr(dist_module, "_coalescing_manager", wrapped_coalescing_manager)
    bindings.append(
        {
            "module": _MEGATRON_COALESCING_MODULE,
            "alias": "_coalescing_manager",
            "public_api": "_coalescing_manager",
            "binding": "p0b_coalescing_transaction_wrapper",
        }
    )
    return bindings


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
        self._excluded_control_collectives: list[dict[str, Any]] = []
        self._next_call_id = 0
        self._frozen_outer_id: int | None = None
        self._frozen_boundary_trace: dict[str, Any] | None = None
        self._frozen_boundary_entries: list[dict[str, Any]] = []
        self._frozen_record_ids: list[int] = []
        self._frozen_call_evidence: list[dict[str, Any]] = []
        self._frozen_call_bindings: dict[int, dict[str, Any]] = {}
        self._frozen_monotonic_ns: int | None = None
        self._freeze_completed_monotonic_ns: int | None = None
        self._freeze_error: dict[str, str] | None = None
        self._trace_root: Path | None = None
        self._trace_root_identity: tuple[int, int] | None = None
        self._runtime_identity: dict[str, Any] | None = None
        self._megatron_alias_bindings: list[dict[str, str]] = []
        if not self._enabled:
            return
        self._validate_environment()
        self._runtime_identity = self._capture_runtime_identity()
        _install_collective_observers(torch_module.distributed)
        self._megatron_alias_bindings = _install_megatron_collective_alias_observers(
            torch_module.distributed
        )
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
        if ".." in root.parts:
            raise CollectiveTraceError(f"{TRACE_ROOT_ENV} cannot contain '..'")
        if self._require_shared_nfs:
            anchor = Path("/shared_nfs")
            try:
                resolved_root = root.resolve(strict=True)
                resolved_anchor = anchor.resolve(strict=True)
            except OSError as exc:
                raise CollectiveTraceError(
                    f"{TRACE_ROOT_ENV} must be a pre-created /shared_nfs directory: {exc}"
                ) from exc
            if (
                os.path.commonpath((str(resolved_root), str(resolved_anchor)))
                != str(resolved_anchor)
                or resolved_root == resolved_anchor
            ):
                raise CollectiveTraceError(
                    f"{TRACE_ROOT_ENV} realpath must be below /shared_nfs"
                )
        current = Path(root.anchor)
        for part in root.parts[1:]:
            current /= part
            if not os.path.lexists(current):
                break
            if current.is_symlink():
                raise CollectiveTraceError(
                    f"trace root path contains symlink component: {current}"
                )
        if os.path.lexists(root) and (not root.is_dir() or root.is_symlink()):
            raise CollectiveTraceError(f"trace root is not a non-symlink directory: {root}")
        self._trace_root = root
        if os.path.lexists(root):
            root_stat = root.stat(follow_symlinks=False)
            self._trace_root_identity = (root_stat.st_dev, root_stat.st_ino)

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
        if version != PINNED_FLIGHT_TRACE_VERSION:
            raise CollectiveTraceError(
                f"flight recorder version {version!r} != pinned "
                f"{PINNED_FLIGHT_TRACE_VERSION!r}"
            )
        if nccl_version != PINNED_RCCL_VERSION:
            raise CollectiveTraceError(
                f"flight recorder NCCL/RCCL version {nccl_version!r} != pinned "
                f"{PINNED_RCCL_VERSION!r}"
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
        self._excluded_control_collectives = []
        self._next_call_id = 0
        self._frozen_outer_id = None
        self._frozen_boundary_trace = None
        self._frozen_boundary_entries = []
        self._frozen_record_ids = []
        self._frozen_call_evidence = []
        self._frozen_call_bindings = {}
        self._frozen_monotonic_ns = None
        self._freeze_completed_monotonic_ns = None
        self._freeze_error = None

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

    def _group_size(self, group: Any) -> int:
        function = getattr(group, "size", None)
        value: Any = None
        if callable(function):
            value = function()
        else:
            function = getattr(self._torch.distributed, "get_world_size", None)
            if callable(function):
                value = function(group)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CollectiveTraceError(f"invalid runtime process-group size {value!r}")
        return value

    @staticmethod
    def _resolve_runtime_group(group: Any) -> tuple[Any, str]:
        """Resolve a MILES reloadable wrapper to the PG actually enqueued.

        ``monkey_patch_torch_dist`` unwraps ``ReloadableProcessGroup`` before
        invoking the original torch.distributed function.  Sequence, name and
        size evidence must consequently be captured from that same live inner
        group, never from the wrapper's independent ProcessGroup base state.
        """

        resolver = getattr(group, "_p0b_current_process_group", None)
        if resolver is None:
            return group, "direct_process_group"
        if not callable(resolver):
            raise CollectiveTraceError(
                "process-group P0-B resolver is present but not callable"
            )
        resolved = resolver()
        if resolved is None or resolved is group:
            raise CollectiveTraceError(
                "process-group P0-B resolver returned an invalid inner group"
            )
        if getattr(resolved, "_p0b_current_process_group", None) is not None:
            raise CollectiveTraceError(
                "nested reloadable process-group resolution is unsupported"
            )
        return resolved, "reloadable_current_inner_process_group"

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

    def prepare_coalescing_manager(
        self,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Capture a pinned Torch 2.9 coalescing transaction before entry."""

        if self.active_outer is None:
            raise CollectiveTraceError("coalescing manager observed outside an active outer")
        if self._frozen_outer_id is not None:
            raise CollectiveTraceError(
                "coalescing manager observed after outer-close boundary freeze"
            )
        manager_group = _positional_or_keyword(args, kwargs, "group", 0, None)
        if manager_group is None:
            manager_group = self._default_group()
        runtime_group, group_resolution = self._resolve_runtime_group(manager_group)
        device = _positional_or_keyword(args, kwargs, "device", 1, None)
        if device is not None:
            raise CollectiveTraceError(
                "device-based legacy coalescing is outside the pinned P0-B transaction parser"
            )
        async_ops = _positional_or_keyword(args, kwargs, "async_ops", 2, False)
        if not isinstance(async_ops, bool):
            raise CollectiveTraceError(
                f"coalescing manager async_ops must be bool, got {async_ops!r}"
            )
        return {
            "manager_group": manager_group,
            "runtime_group": runtime_group,
            "group_resolution": group_resolution,
            "group_name": self._group_name(runtime_group),
            "group_size": self._group_size(runtime_group),
            "sequence_before": self._group_sequence(runtime_group),
            "async_ops": async_ops,
            "observed_call_count_before": len(self._observed_calls),
            "captured": False,
            "empty": False,
        }

    def capture_coalescing_manager(self, transaction: dict[str, Any]) -> None:
        """Read runtime ``_CollOp`` inputs immediately before manager enqueue."""

        if len(self._observed_calls) != transaction["observed_call_count_before"]:
            raise CollectiveTraceError(
                "coalescing transaction mixed captured public calls with raw aliases"
            )
        if self._group_sequence(transaction["runtime_group"]) != transaction[
            "sequence_before"
        ]:
            raise CollectiveTraceError(
                "coalescing transaction advanced the live process-group sequence before exit"
            )
        dist_module = self._torch.distributed
        c10d = getattr(dist_module, "distributed_c10d", None)
        world = getattr(c10d, "_world", None)
        state = getattr(world, "pg_coalesce_state", None)
        if not isinstance(state, Mapping):
            raise CollectiveTraceError(
                "Torch 2.9 coalescing state mapping is unavailable"
            )
        if transaction["manager_group"] not in state:
            raise CollectiveTraceError(
                "coalescing manager group is absent from Torch 2.9 transaction state"
            )
        operations = list(state[transaction["manager_group"]])
        if not operations:
            transaction["empty"] = True
            transaction["captured"] = True
            return

        operation_names: list[str] = []
        tensor_evidence: list[dict[str, Any]] = []
        for operation in operations:
            function = getattr(operation, "op", None)
            operation_name = getattr(function, "__name__", None)
            if operation_name not in {
                "all_reduce",
                "all_gather_into_tensor",
                "reduce_scatter_tensor",
            }:
                raise CollectiveTraceError(
                    f"unsupported Torch coalescing operation {operation_name!r}"
                )
            operation_names.append(operation_name)
            tensor_evidence.append(
                _runtime_tensor_evidence(getattr(operation, "tensor", None), self._torch)
            )
        if len(set(operation_names)) != 1:
            raise CollectiveTraceError(
                f"mixed Torch coalescing operations are unsupported: {operation_names!r}"
            )
        operation_name = operation_names[0]
        expected_low_level_ops = {
            "all_reduce": frozenset({"allreduce_coalesced"}),
            "all_gather_into_tensor": frozenset(
                {"all_gather_into_tensor_coalesced"}
            ),
            "reduce_scatter_tensor": frozenset(
                {"reduce_scatter_tensor_coalesced"}
            ),
        }[operation_name]
        call_id = self._next_call_id
        self._next_call_id += 1
        transaction.update(
            {
                "captured": True,
                "call_id": call_id,
                "operation_name": operation_name,
                "expected_low_level_ops": expected_low_level_ops,
                "runtime_input_tensors": tensor_evidence,
                "runtime_rank_local_logical_input_bytes": sum(
                    item["logical_bytes"] for item in tensor_evidence
                ),
                "coalesced_subcall_count": len(operations),
            }
        )

    def complete_coalescing_manager(
        self,
        transaction: dict[str, Any],
        manager: Any,
    ) -> None:
        """Bind the one real post-exit coalesced Work/sequence to the transaction."""

        if transaction.get("observation_error") is not None:
            raise CollectiveTraceError(transaction["observation_error"])
        if transaction.get("captured") is not True:
            raise CollectiveTraceError("coalescing transaction was not captured before exit")
        if transaction.get("empty") is True:
            if self._group_sequence(transaction["runtime_group"]) != transaction[
                "sequence_before"
            ]:
                raise CollectiveTraceError(
                    "empty coalescing transaction advanced the process-group sequence"
                )
            return
        works = getattr(manager, "works", None)
        if transaction["async_ops"]:
            if not isinstance(works, list) or len(works) != 1:
                raise CollectiveTraceError(
                    "async coalescing transaction did not expose exactly one real Work"
                )
            result = works[0]
        else:
            if not isinstance(works, list) or works:
                raise CollectiveTraceError(
                    "sync coalescing transaction unexpectedly retained Work objects"
                )
            result = None
        context = {
            "call_id": transaction["call_id"],
            "public_api": (
                f"_coalescing_manager[{transaction['operation_name']}]"
            ),
            "async_op": transaction["async_ops"],
            "expected_low_level_ops": transaction["expected_low_level_ops"],
            "group": transaction["runtime_group"],
            "group_name": transaction["group_name"],
            "group_size": transaction["group_size"],
            "group_resolution": transaction["group_resolution"],
            "sequence_before": transaction["sequence_before"],
            "runtime_rank_local_logical_input_bytes": transaction[
                "runtime_rank_local_logical_input_bytes"
            ],
            "runtime_input_tensors": transaction["runtime_input_tensors"],
            "observation_source": "torch_2_9_coalescing_transaction",
            "coalesced_subcall_count": transaction["coalesced_subcall_count"],
        }
        self.complete_public_call(context, result)

    def prepare_public_call(
        self,
        name: str,
        spec: _PublicCallSpec,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.active_outer is None:
            raise CollectiveTraceError("public collective observed outside an active outer")
        if self._frozen_outer_id is not None:
            raise CollectiveTraceError(
                f"public collective {name} observed after outer-close boundary freeze"
            )
        group = _positional_or_keyword(
            args, kwargs, "group", spec.group_index, None
        )
        if group is None:
            group = self._default_group()
        group, group_resolution = self._resolve_runtime_group(group)
        logical_bytes, tensor_evidence = _runtime_input_payload(
            name, spec, args, kwargs, self._torch
        )
        call_id = self._next_call_id
        self._next_call_id += 1
        return {
            "call_id": call_id,
            "public_api": name,
            "async_op": _async_flag(name, spec.async_index, args, kwargs),
            "expected_low_level_ops": spec.expected_low_level_ops,
            "group": group,
            "group_name": self._group_name(group),
            "group_size": self._group_size(group),
            "group_resolution": group_resolution,
            "sequence_before": self._group_sequence(group),
            "runtime_rank_local_logical_input_bytes": logical_bytes,
            "runtime_input_tensors": tensor_evidence,
        }

    def prepare_gloo_control_object_call(
        self,
        name: str,
        group_index: int,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.active_outer is None:
            raise CollectiveTraceError("Gloo object collective observed outside an active outer")
        if self._frozen_outer_id is not None:
            raise CollectiveTraceError(
                f"Gloo object collective {name} observed after outer-close boundary freeze"
            )
        group = _positional_or_keyword(args, kwargs, "group", group_index, None)
        if group is None:
            group = self._default_group()
        group, group_resolution = self._resolve_runtime_group(group)
        get_backend = getattr(self._torch.distributed, "get_backend", None)
        if not callable(get_backend):
            raise CollectiveTraceError("torch.distributed.get_backend is unavailable")
        backend_value = get_backend(group)
        backend = str(backend_value).strip().lower()
        if backend != "gloo":
            raise CollectiveTraceError(
                f"object collective {name} backend must be exactly gloo, got {backend_value!r}"
            )
        return {
            "public_api": name,
            "backend": backend,
            "group_name": self._group_name(group),
            "group_size": self._group_size(group),
            "group_resolution": group_resolution,
            "metric_inclusion": "excluded",
            "reason": (
                "verified CPU/Gloo object control collective outside the "
                "accelerator tensor-collective byte/time metric"
            ),
        }

    def complete_gloo_control_object_call(self, context: Mapping[str, Any]) -> None:
        self._excluded_control_collectives.append(dict(context))

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

    @staticmethod
    def _validate_retired_trace_immutability(
        boundary_trace: Mapping[str, Any],
        boundary_entries: list[dict[str, Any]],
        retired_trace: Mapping[str, Any],
        retired_entries: list[dict[str, Any]],
    ) -> None:
        if boundary_trace.get("pg_config") != retired_trace.get("pg_config"):
            raise CollectiveTraceError(
                "flight recorder pg_config drifted after outer-close freeze"
            )
        # Discovery timestamps are observations made by each JSON dump and
        # legitimately refresh (confirmed by the pinned 2.9 recorder).  They
        # remain raw host evidence only and are never a duration source.
        allowed_to_change = {
            "retired",
            "duration_ms",
            "time_discovered_started_ns",
            "time_discovered_completed_ns",
        }
        boundary_by_id = {entry["record_id"]: entry for entry in boundary_entries}
        retired_by_id = {entry["record_id"]: entry for entry in retired_entries}
        if set(boundary_by_id) != set(retired_by_id):
            raise CollectiveTraceError(
                "flight record IDs drifted after outer-close freeze"
            )
        for record_id, boundary in boundary_by_id.items():
            retired = retired_by_id[record_id]
            frozen_fields = {
                key: value
                for key, value in boundary.items()
                if key not in allowed_to_change
            }
            retired_frozen_fields = {
                key: value
                for key, value in retired.items()
                if key not in allowed_to_change
            }
            if retired_frozen_fields != frozen_fields:
                raise CollectiveTraceError(
                    f"flight record {record_id} immutable fields drifted after outer-close freeze"
                )
            for label, observed in (("boundary", boundary), ("retired", retired)):
                started = observed.get("time_discovered_started_ns")
                completed = observed.get("time_discovered_completed_ns")
                if (
                    isinstance(started, bool)
                    or not isinstance(started, int)
                    or started <= 0
                    or isinstance(completed, bool)
                    or not isinstance(completed, int)
                    or completed < started
                ):
                    raise CollectiveTraceError(
                        f"flight record {record_id} has invalid {label} discovery timestamps"
                    )
            boundary_retired = boundary.get("retired")
            retired_value = retired.get("retired")
            if (
                not isinstance(boundary_retired, bool)
                or retired_value is not True
                or (boundary_retired is True and retired_value is not True)
            ):
                raise CollectiveTraceError(
                    f"flight record {record_id} has invalid retirement transition "
                    f"{boundary_retired!r}->{retired_value!r}"
                )
            boundary_duration = boundary.get("duration_ms")
            retired_duration = retired.get("duration_ms")
            if boundary_duration is None:
                if (
                    isinstance(retired_duration, bool)
                    or not isinstance(retired_duration, (int, float))
                    or not math.isfinite(float(retired_duration))
                    or float(retired_duration) <= 0
                ):
                    raise CollectiveTraceError(
                        f"flight record {record_id} has invalid duration "
                        f"{retired_duration!r} after retirement"
                    )
            elif (
                isinstance(boundary_duration, bool)
                or not isinstance(boundary_duration, (int, float))
                or not math.isfinite(float(boundary_duration))
                or float(boundary_duration) <= 0
                or retired_duration != boundary_duration
            ):
                raise CollectiveTraceError(
                    f"flight record {record_id} duration drifted after outer-close freeze"
                )

    @staticmethod
    def _cross_check_async_work_durations(
        call_evidence: list[dict[str, Any]],
        retired_entries: list[dict[str, Any]],
    ) -> None:
        entries_by_id = {entry["record_id"]: entry for entry in retired_entries}
        for call in call_evidence:
            if not call["async_op"]:
                continue
            flight_duration_ms = float(
                entries_by_id[call["flight_record_id"]]["duration_ms"]
            )
            work_duration_ms = float(call["work_device_duration_ms"])
            tolerance_ms = max(
                1e-4,
                max(abs(work_duration_ms), abs(flight_duration_ms)) * 1e-3,
            )
            if abs(work_duration_ms - flight_duration_ms) > tolerance_ms:
                raise CollectiveTraceError(
                    f"async call_id={call['call_id']} Work/flight device-duration mismatch: "
                    f"Work={work_duration_ms}ms flight={flight_duration_ms}ms "
                    f"tolerance={tolerance_ms}ms"
                )
            call["flight_device_duration_ms"] = flight_duration_ms
            call["work_flight_duration_tolerance_ms"] = tolerance_ms
            call["work_flight_device_duration_match"] = True
            call["work_duration_compared_to_flight_duration"] = True

    def _validate_call_bindings(
        self,
        trace: Mapping[str, Any],
        entries: list[dict[str, Any]],
        async_work_snapshots: Mapping[int, Mapping[str, Any]],
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
            group_name, _ = _parse_group_identity(
                trace,
                entry,
                local_rank=self._rank,
                world_size=self._world_size,
            )
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
            flight_logical_bytes, flight_tensors = _entry_input_bytes(
                entry, operation
            )
            runtime_logical_bytes = call[
                "runtime_rank_local_logical_input_bytes"
            ]
            if flight_logical_bytes != runtime_logical_bytes:
                raise CollectiveTraceError(
                    f"public call_id={call['call_id']} runtime/flight byte mismatch: "
                    f"runtime={runtime_logical_bytes} flight={flight_logical_bytes}"
                )
            _, flight_group_size = _parse_group_identity(
                trace,
                entry,
                local_rank=self._rank,
                world_size=self._world_size,
            )
            if flight_group_size != call["group_size"]:
                raise CollectiveTraceError(
                    f"public call_id={call['call_id']} runtime/flight group-size mismatch: "
                    f"runtime={call['group_size']} flight={flight_group_size}"
                )
            evidence = {
                "call_id": call["call_id"],
                "public_api": call["public_api"],
                "async_op": call["async_op"],
                "group_name": call["group_name"],
                "group_size": call["group_size"],
                "group_resolution": call["group_resolution"],
                "sequence_before": call["sequence_before"],
                "collective_seq_id": call["sequence_after"],
                "flight_record_id": entry["record_id"],
                "flight_operation": operation,
                "binding": "exact_process_group_name_and_collective_sequence",
                "runtime_rank_local_logical_input_bytes": runtime_logical_bytes,
                "runtime_input_tensors": call["runtime_input_tensors"],
                "flight_rank_local_logical_input_bytes": flight_logical_bytes,
                "flight_input_tensors": flight_tensors,
                "runtime_flight_bytes_match": True,
                "runtime_flight_group_size_match": True,
            }
            if "observation_source" in call:
                evidence["observation_source"] = call["observation_source"]
            if "coalesced_subcall_count" in call:
                evidence["coalesced_subcall_count"] = call[
                    "coalesced_subcall_count"
                ]
            if call["async_op"]:
                work_snapshot = async_work_snapshots.get(call["call_id"])
                if not isinstance(work_snapshot, Mapping):
                    raise CollectiveTraceError(
                        f"async call_id={call['call_id']} lacks outer-close Work snapshot"
                    )
                if work_snapshot.get("completed") is not True:
                    raise CollectiveTraceError(
                        f"async call_id={call['call_id']} {call['public_api']} Work "
                        "remained outstanding at outer close"
                    )
                work_duration_ms = float(work_snapshot["device_duration_ms"])
                evidence.update(
                    {
                        "work_completed_at_outer_close_without_profiler_wait": True,
                        "work_device_duration_ms": work_duration_ms,
                        "work_snapshot_monotonic_ns": work_snapshot[
                            "captured_monotonic_ns"
                        ],
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

    def _snapshot_async_works_at_boundary(self) -> dict[int, dict[str, Any]]:
        """Snapshot Work state before any flight dump or profiler overhead."""

        snapshots: dict[int, dict[str, Any]] = {}
        for call in self._observed_calls:
            if not call["async_op"]:
                continue
            captured_monotonic_ns = time.monotonic_ns()
            work = call["work"]
            completed = bool(work.is_completed())
            snapshot: dict[str, Any] = {
                "completed": completed,
                "captured_monotonic_ns": captured_monotonic_ns,
            }
            if completed:
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
                snapshot["device_duration_ms"] = work_duration_ms
            snapshots[call["call_id"]] = snapshot
        return snapshots

    def freeze_outer_boundary(self, outer_id: int) -> dict[str, Any]:
        """Freeze eligibility at canonical outer close without polling/writes.

        This method must be called immediately after the workload's existing
        synchronization and before Ray/RSS/CUDA queries or any profiler I/O.
        It fixes the exact flight record set and checks Work completion at
        that instant.  Later finalization may only wait for retirement
        metadata for this frozen set; it can never make a late event eligible.
        """

        if not self._enabled:
            return {"available": False, "reason": "disabled"}
        if self._frozen_outer_id is not None or self._freeze_error is not None:
            return {
                "available": False,
                "error_type": "CollectiveTraceError",
                "error_message": "collective outer boundary was already frozen",
                "outer_id": int(outer_id),
            }
        try:
            if self.active_outer != int(outer_id) or self._begin_record_id is None:
                raise CollectiveTraceError(
                    f"collective profiler active outer={self.active_outer}, requested={outer_id}"
                )
            # Work eligibility is the first profiler observation at the
            # canonical close boundary.  Even identity checks and error
            # rendering are deliberately later so their overhead cannot make
            # a still-running async Work appear boundary-complete.
            frozen_monotonic_ns = time.monotonic_ns()
            async_work_snapshots = self._snapshot_async_works_at_boundary()
            if self._errors:
                raise CollectiveTraceError("; ".join(self._errors))
            current_identity = self._capture_runtime_identity()
            if current_identity != self._runtime_identity:
                raise CollectiveTraceError(
                    "runtime job/node/process identity drifted within outer"
                )
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
            call_evidence, call_bindings = self._validate_call_bindings(
                boundary_trace, boundary_entries, async_work_snapshots
            )
            freeze_completed_monotonic_ns = time.monotonic_ns()
            self._frozen_outer_id = int(outer_id)
            self._frozen_boundary_trace = boundary_trace
            self._frozen_boundary_entries = boundary_entries
            self._frozen_record_ids = record_ids
            self._frozen_call_evidence = call_evidence
            self._frozen_call_bindings = call_bindings
            self._frozen_monotonic_ns = frozen_monotonic_ns
            self._freeze_completed_monotonic_ns = freeze_completed_monotonic_ns
            return {
                "available": True,
                "outer_id": int(outer_id),
                "frozen_monotonic_ns": frozen_monotonic_ns,
                "freeze_completed_monotonic_ns": freeze_completed_monotonic_ns,
                "freeze_overhead_seconds": (
                    freeze_completed_monotonic_ns - frozen_monotonic_ns
                )
                / 1e9,
                "begin_record_id": self._begin_record_id,
                "end_record_id": record_ids[-1],
                "entry_count": len(record_ids),
                "public_call_count": len(call_evidence),
                "async_work_count": sum(
                    1 for item in call_evidence if item["async_op"]
                ),
                "record_id_set_frozen": True,
            }
        except Exception as exc:
            self._freeze_error = {
                "error_type": type(exc).__name__,
                "error_message": _bounded(exc),
            }
            return {
                "available": False,
                **self._freeze_error,
                "outer_id": int(outer_id),
            }

    def _write_trace(self, outer_id: int, payload: dict[str, Any]) -> tuple[Path, int, str]:
        assert self._trace_root is not None
        if not os.path.lexists(self._trace_root):
            if self._require_shared_nfs:
                raise CollectiveTraceError(
                    f"pre-created production trace root disappeared: {self._trace_root}"
                )
            self._trace_root.mkdir(parents=True, mode=0o755)
        if not self._trace_root.is_dir() or self._trace_root.is_symlink():
            raise CollectiveTraceError(f"trace root is unsafe: {self._trace_root}")
        current = Path(self._trace_root.anchor)
        for part in self._trace_root.parts[1:]:
            current /= part
            if not os.path.lexists(current) or current.is_symlink():
                raise CollectiveTraceError(
                    f"trace root component disappeared or became a symlink: {current}"
                )
        if self._require_shared_nfs:
            resolved_root = self._trace_root.resolve(strict=True)
            resolved_anchor = Path("/shared_nfs").resolve(strict=True)
            if (
                os.path.commonpath((str(resolved_root), str(resolved_anchor)))
                != str(resolved_anchor)
                or resolved_root == resolved_anchor
            ):
                raise CollectiveTraceError(
                    "trace root realpath drifted outside /shared_nfs before publication"
                )
        root_stat = self._trace_root.stat(follow_symlinks=False)
        current_identity = (root_stat.st_dev, root_stat.st_ino)
        if (
            self._trace_root_identity is not None
            and current_identity != self._trace_root_identity
        ):
            raise CollectiveTraceError(
                "trace root inode identity drifted before publication"
            )
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
            if self._frozen_outer_id is None and self._freeze_error is None:
                self.freeze_outer_boundary(int(outer_id))
            if self._freeze_error is not None:
                raise CollectiveTraceError(
                    "outer-close boundary freeze failed: "
                    f"{self._freeze_error['error_type']}: "
                    f"{self._freeze_error['error_message']}"
                )
            if (
                self.active_outer != int(outer_id)
                or self._frozen_outer_id != int(outer_id)
                or self._begin_record_id is None
                or self._frozen_boundary_trace is None
                or self._frozen_monotonic_ns is None
                or self._freeze_completed_monotonic_ns is None
            ):
                raise CollectiveTraceError(
                    f"collective profiler/frozen outer mismatch: active={self.active_outer} "
                    f"frozen={self._frozen_outer_id} requested={outer_id}"
                )
            if self._errors:
                raise CollectiveTraceError("; ".join(self._errors))
            current_identity = self._capture_runtime_identity()
            if current_identity != self._runtime_identity:
                raise CollectiveTraceError(
                    "runtime job/node/process identity drifted within outer"
                )
            boundary_trace = self._frozen_boundary_trace
            boundary_entries = self._frozen_boundary_entries
            record_ids = self._frozen_record_ids
            call_evidence = self._frozen_call_evidence
            call_binding_by_record_id = self._frozen_call_bindings
            async_evidence = [
                item for item in call_evidence if item["async_op"]
            ]
            trace, entries, retirement_poll = self._poll_retired_delta(record_ids)
            self._validate_retired_trace_immutability(
                boundary_trace, boundary_entries, trace, entries
            )
            self._cross_check_async_work_durations(call_evidence, entries)
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
            total_flight_bytes = 0
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
                group_name, group_size = _parse_group_identity(
                    trace,
                    entry,
                    local_rank=self._rank,
                    world_size=self._world_size,
                )
                if group_size > self._world_size:
                    raise CollectiveTraceError(
                        f"process-group size {group_size} exceeds world_size={self._world_size}"
                    )
                call_binding = call_binding_by_record_id[entry["record_id"]]
                runtime_logical_bytes = call_binding[
                    "runtime_rank_local_logical_input_bytes"
                ]
                total_bytes += runtime_logical_bytes
                total_flight_bytes += logical_bytes
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
                        "rank_local_logical_input_bytes": runtime_logical_bytes,
                        "runtime_input_tensors": call_binding[
                            "runtime_input_tensors"
                        ],
                        "flight_rank_local_logical_input_bytes": logical_bytes,
                        "flight_input_tensors": tensors,
                        "device_duration_ms": duration_ms,
                        "duration_source": "ProcessGroupNCCL flight recorder accelerator events",
                        "state": entry.get("state"),
                        "retired": entry.get("retired"),
                        "public_call_binding": call_binding,
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
            if total_flight_bytes != total_bytes:
                raise CollectiveTraceError(
                    f"outer runtime/flight byte total mismatch: runtime={total_bytes} "
                    f"flight={total_flight_bytes}"
                )
            if not math.isfinite(total_duration_ms) or total_duration_ms <= 0:
                raise CollectiveTraceError("ABBA outer has no positive collective device duration")
            parser_path = Path(__file__).resolve()
            parser_sha = _sha256_file(parser_path)
            assert self._runtime_identity is not None
            payload = {
                "schema_version": TRACE_SCHEMA_VERSION,
                "kind": "p0b-primary-rank-runtime-flight-trace-v3",
                "outer_id": int(outer_id),
                "rank": self._rank,
                "world_size": self._world_size,
                "runtime_identity": dict(self._runtime_identity),
                "trace_version": trace.get("version"),
                "nccl_version": trace.get("nccl_version"),
                "pinned_trace_version": PINNED_FLIGHT_TRACE_VERSION,
                "pinned_nccl_version": PINNED_RCCL_VERSION,
                "begin_record_id": self._begin_record_id,
                "end_record_id": record_ids[-1],
                "record_ids_contiguous": True,
                "record_id_set_frozen_at_outer_close": True,
                "new_records_during_retirement_poll": 0,
                "boundary_freeze": {
                    "started_monotonic_ns": self._frozen_monotonic_ns,
                    "completed_monotonic_ns": self._freeze_completed_monotonic_ns,
                    "overhead_seconds": (
                        self._freeze_completed_monotonic_ns
                        - self._frozen_monotonic_ns
                    )
                    / 1e9,
                    "polling_or_trace_write_performed": False,
                },
                "retirement_poll": retirement_poll,
                "boundary_frozen_before_finalize": True,
                "environment": {
                    TRACE_BUFFER_ENV: self._environ[TRACE_BUFFER_ENV],
                    TIMING_ENV: self._environ[TIMING_ENV],
                    TRACE_ROOT_ENV: str(self._trace_root),
                    "trace_root_realpath": str(
                        self._trace_root.resolve(strict=self._require_shared_nfs)
                    ),
                },
                "semantics": {
                    "collective_bytes": (
                        "sum of numel*element_size from actual input tensor arguments at "
                        "allowlisted public collective calls issued by this rank, cross-checked "
                        "against flight entries; not multiplied by group size and not wire bytes"
                    ),
                    "collective_time_seconds": (
                        "sum of ProcessGroupNCCL accelerator-event duration_ms for completed retired "
                        "collective entries; host enqueue/discovery timestamps excluded"
                    ),
                },
                "pg_config": trace.get("pg_config"),
                "pg_status": trace.get("pg_status"),
                "megatron_collective_alias_bindings": self._megatron_alias_bindings,
                "excluded_control_collectives": list(self._excluded_control_collectives),
                "operation_histogram": dict(sorted(op_histogram.items())),
                "group_size_histogram": dict(sorted(group_histogram.items())),
                "public_call_evidence": call_evidence,
                "async_work_evidence": async_evidence,
                "total_rank_local_logical_input_bytes": total_bytes,
                "total_flight_rank_local_logical_input_bytes": total_flight_bytes,
                "runtime_flight_byte_totals_match": True,
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
                "pinned_trace_version": PINNED_FLIGHT_TRACE_VERSION,
                "pinned_nccl_version": PINNED_RCCL_VERSION,
                "begin_record_id": self._begin_record_id,
                "end_record_id": record_ids[-1],
                "entry_count": len(entries),
                "trace_buffer_capacity": trace_capacity,
                "retirement_poll": retirement_poll,
                "boundary_freeze": {
                    "started_monotonic_ns": self._frozen_monotonic_ns,
                    "completed_monotonic_ns": self._freeze_completed_monotonic_ns,
                    "overhead_seconds": (
                        self._freeze_completed_monotonic_ns
                        - self._frozen_monotonic_ns
                    )
                    / 1e9,
                    "polling_or_trace_write_performed": False,
                },
                "boundary_frozen_before_finalize": True,
                "operation_histogram": dict(sorted(op_histogram.items())),
                "group_size_histogram": dict(sorted(group_histogram.items())),
                "async_work_count": len(async_evidence),
                "public_call_count": len(call_evidence),
                "public_call_binding": (
                    "exact_process_group_name_and_collective_sequence"
                ),
                "megatron_collective_alias_bindings": self._megatron_alias_bindings,
                "excluded_control_collectives": list(self._excluded_control_collectives),
                "outstanding_async_work_count": 0,
                "rank_local_logical_input_bytes": total_bytes,
                "flight_rank_local_logical_input_bytes": total_flight_bytes,
                "runtime_flight_byte_totals_match": True,
                "collective_bytes_source": "actual_public_api_runtime_input_tensors",
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
            self._excluded_control_collectives = []
            self._next_call_id = 0
            self._frozen_outer_id = None
            self._frozen_boundary_trace = None
            self._frozen_boundary_entries = []
            self._frozen_record_ids = []
            self._frozen_call_evidence = []
            self._frozen_call_bindings = {}
            self._frozen_monotonic_ns = None
            self._freeze_completed_monotonic_ns = None
            self._freeze_error = None
