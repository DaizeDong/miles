"""Opt-in, process-local measurements for the P0-B fixed-rollout profile.

This module deliberately reports only quantities observed by trusted runtime
sources. Ray object-store/spill values come from Ray's pinned 2.44.1 global
memory-info reply. Collective bytes and device time come from the retained
ProcessGroupNCCL flight recorder, never configured shapes, wire-byte formulae,
or host enqueue timing. The feature is inert unless
``MILES_P0B_SYSTEM_PROFILE=1``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import numbers
import os
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from miles.utils.p0b_collective_trace import FlightRecorderCollectiveProfiler


_ENABLED_ENV = "MILES_P0B_SYSTEM_PROFILE"
_OUTERS_ENV = "MILES_P0B_MEASURED_OUTER_IDS"
_DEFAULT_MEASURED_OUTERS = frozenset(range(3, 13))
_PINNED_RAY_VERSION = "2.44.1"
_RAY_SOURCE = "ray._private.internal_api.get_memory_info_reply"
_PROFILE_ID = "p0b_system_profile_v2"
_SCHEMA_VERSION = 2


def _source_identity() -> dict[str, str]:
    path = os.path.realpath(__file__)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": path, "sha256": digest.hexdigest()}


class _RayObjectStoreReader:
    """Read Ray's cluster-global object-store counters through the pinned API.

    The state connection is cached, but each call performs a fresh
    ``FormatGlobalMemoryInfo`` RPC.  Imports are lazy so the default-off path
    neither imports Ray nor opens a control-plane connection.
    """

    def __init__(self) -> None:
        self._state: Any = None
        self._state_identity: tuple[str, str, str] | None = None

    def __call__(self) -> Mapping[str, Any]:
        import ray
        from ray._private import internal_api
        from ray._private import worker as ray_worker

        version = str(ray.__version__)
        if version != _PINNED_RAY_VERSION:
            raise RuntimeError(
                f"P0-B Ray counters require Ray {_PINNED_RAY_VERSION}, got {version}"
            )
        gcs_client = getattr(ray_worker.global_worker, "gcs_client", None)
        gcs_address = getattr(gcs_client, "address", None)
        if not isinstance(gcs_address, str) or not gcs_address:
            raise RuntimeError("P0-B Ray snapshot cannot bind the live GCS address")
        cluster_id = _runtime_identifier(getattr(gcs_client, "cluster_id", None))
        if cluster_id is None:
            raise RuntimeError("P0-B Ray snapshot cannot bind the live Ray cluster ID")
        context = ray.get_runtime_context()
        job_id = _runtime_identifier(context.get_job_id())
        if job_id is None:
            raise RuntimeError("P0-B Ray snapshot cannot bind the live Ray job ID")
        live_identity = (gcs_address, cluster_id, job_id)
        if self._state is None:
            self._state = internal_api.get_state_from_address(gcs_address)
            self._state_identity = live_identity
        elif live_identity != self._state_identity:
            raise RuntimeError(
                "P0-B Ray cached state identity drifted: "
                f"cached={self._state_identity!r} live={live_identity!r}"
            )
        reply = internal_api.get_memory_info_reply(self._state)
        stats = reply.store_stats
        return {
            "object_store_bytes_used": stats.object_store_bytes_used,
            "spilled_bytes_total": stats.spilled_bytes_total,
            "ray_version": version,
            "source": _RAY_SOURCE,
            "cluster_identity": {
                "gcs_address": gcs_address,
                "cluster_id": cluster_id,
                "job_id": job_id,
            },
            "captured_monotonic_ns": time.monotonic_ns(),
        }


def _bounded_repr(value: Any, limit: int = 160) -> str:
    rendered = repr(value)
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."


def _runtime_identifier(value: Any) -> str | None:
    """Canonicalize a Ray binary ID without accepting a missing identity."""

    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, bytes):
        return value.hex() or None
    hex_method = getattr(value, "hex", None)
    if callable(hex_method):
        try:
            rendered = hex_method()
        except Exception:
            rendered = None
        if isinstance(rendered, str) and rendered:
            return rendered
    rendered = str(value)
    return rendered if rendered else None


def _validated_integral_counter(value: Any, name: str) -> tuple[int, dict[str, Any]]:
    """Validate a runtime numeric counter while retaining its raw schema."""

    raw = {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": _bounded_repr(value),
    }
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"invalid Ray counter {name}: type={raw['type']} value={raw['repr']} is not numeric"
        )
    # Integral counters must never round-trip through binary64: object-store
    # byte counters can legitimately exceed 2**53.
    if isinstance(value, numbers.Integral):
        integer = int(value)
        if integer < 0:
            raise ValueError(
                f"invalid Ray counter {name}: type={raw['type']} value={raw['repr']} "
                "must be finite, non-negative, and integral"
            )
    else:
        try:
            numeric = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid Ray counter {name}: type={raw['type']} value={raw['repr']} is not finite"
            ) from exc
        if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
            raise ValueError(
                f"invalid Ray counter {name}: type={raw['type']} value={raw['repr']} "
                "must be finite, non-negative, and integral"
            )
        integer = int(numeric)
    raw["validated_integer"] = integer
    return integer, raw


def _validated_ray_snapshot(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate live Ray counters without coercing malformed values to zero."""

    values: dict[str, int] = {}
    raw_fields: dict[str, dict[str, Any]] = {}
    for name in ("object_store_bytes_used", "spilled_bytes_total"):
        value, evidence = _validated_integral_counter(raw.get(name), name)
        values[name] = value
        raw_fields[name] = evidence
    version = raw.get("ray_version")
    source = raw.get("source")
    cluster_identity = raw.get("cluster_identity")
    captured_monotonic_ns = raw.get("captured_monotonic_ns")
    if version != _PINNED_RAY_VERSION:
        raise ValueError(f"invalid Ray snapshot version={version!r}")
    if source != _RAY_SOURCE:
        raise ValueError(f"invalid Ray snapshot source={source!r}")
    if not isinstance(cluster_identity, Mapping) or not all(
        isinstance(cluster_identity.get(key), str) and cluster_identity.get(key)
        for key in ("gcs_address", "cluster_id", "job_id")
    ):
        raise ValueError(f"invalid Ray snapshot cluster_identity={cluster_identity!r}")
    if (
        isinstance(captured_monotonic_ns, bool)
        or not isinstance(captured_monotonic_ns, int)
        or captured_monotonic_ns <= 0
    ):
        raise ValueError(
            f"invalid Ray snapshot captured_monotonic_ns={captured_monotonic_ns!r}"
        )
    return {
        "available": True,
        **values,
        "raw_fields": raw_fields,
        "ray_version": version,
        "source": source,
        "cluster_identity": dict(cluster_identity),
        "captured_monotonic_ns": captured_monotonic_ns,
    }


def _profile_enabled(environ: Mapping[str, str]) -> bool:
    raw = environ.get(_ENABLED_ENV)
    if raw is None or raw == "0":
        return False
    if raw == "1":
        return True
    raise ValueError(f"{_ENABLED_ENV} must be exactly '0' or '1', got {raw!r}")


def _measured_outer_ids(environ: Mapping[str, str]) -> frozenset[int]:
    raw = environ.get(_OUTERS_ENV)
    if raw is None:
        return _DEFAULT_MEASURED_OUTERS
    values = [part.strip() for part in raw.split(",")]
    if not values or any(not part for part in values):
        raise ValueError(f"{_OUTERS_ENV} must be a non-empty comma-separated integer list")
    try:
        parsed = [int(part) for part in values]
    except ValueError as exc:
        raise ValueError(f"{_OUTERS_ENV} contains a non-integer value: {raw!r}") from exc
    if any(outer_id < 0 for outer_id in parsed):
        raise ValueError(f"{_OUTERS_ENV} cannot contain negative outer IDs: {raw!r}")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{_OUTERS_ENV} cannot contain duplicate outer IDs: {raw!r}")
    return frozenset(parsed)


def _proc_memory_bytes(path: str = "/proc/self/status") -> tuple[int | None, int | None]:
    """Return Linux process (current RSS, lifetime peak RSS) in bytes."""

    values: dict[str, int] = {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                key, separator, remainder = line.partition(":")
                if separator and key in {"VmRSS", "VmHWM"}:
                    fields = remainder.split()
                    if len(fields) >= 2 and fields[1] == "kB":
                        values[key] = int(fields[0]) * 1024
    except (OSError, ValueError):
        return None, None
    return values.get("VmRSS"), values.get("VmHWM")


def _storage_identity_and_bytes(value: Any) -> tuple[object, int]:
    """Return a stable-in-process tensor storage identity and allocated bytes."""

    try:
        storage = value.untyped_storage()
        size = int(storage.nbytes())
        try:
            identity: object = (str(value.device), int(storage.data_ptr()), size)
        except Exception:
            identity = id(storage)
        return identity, size
    except Exception:
        logical = int(value.numel()) * int(value.element_size())
        return id(value), logical


def tensor_payload_stats(value: Any, torch_module: Any) -> dict[str, int]:
    """Measure recursive tensor/ndarray bytes without serializing the payload.

    ``logical_bytes`` sums tensor views. ``unique_storage_bytes`` deduplicates
    shared tensor storage and NumPy base arrays, which is the closer local
    retained-memory quantity. Python-container and Ray wire-format overhead are
    intentionally excluded.
    """

    logical_bytes = 0
    unique_storage_bytes = 0
    tensor_count = 0
    ndarray_count = 0
    seen_objects: set[int] = set()
    seen_storages: set[object] = set()

    def visit(item: Any) -> None:
        nonlocal logical_bytes, unique_storage_bytes, tensor_count, ndarray_count
        if item is None or isinstance(item, (str, bytes, bytearray, int, float, bool)):
            return
        try:
            is_tensor = bool(torch_module.is_tensor(item))
        except Exception:
            is_tensor = False
        if is_tensor:
            tensor_count += 1
            logical_bytes += int(item.numel()) * int(item.element_size())
            identity, storage_bytes = _storage_identity_and_bytes(item)
            if identity not in seen_storages:
                seen_storages.add(identity)
                unique_storage_bytes += storage_bytes
            return

        # Avoid importing NumPy in this opt-in utility. Its public array
        # protocol is sufficient for byte accounting.
        if hasattr(item, "nbytes") and hasattr(item, "__array_interface__"):
            ndarray_count += 1
            item_bytes = int(item.nbytes)
            logical_bytes += item_bytes
            owner = item
            while getattr(owner, "base", None) is not None:
                owner = owner.base
            identity = ("numpy", id(owner))
            if identity not in seen_storages:
                seen_storages.add(identity)
                unique_storage_bytes += int(getattr(owner, "nbytes", item_bytes))
            return

        object_id = id(item)
        if object_id in seen_objects:
            return
        seen_objects.add(object_id)
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            for field in dataclasses.fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, Sequence):
            for child in item:
                visit(child)

    visit(value)
    return {
        "logical_bytes": logical_bytes,
        "unique_storage_bytes": unique_storage_bytes,
        "tensor_count": tensor_count,
        "ndarray_count": ndarray_count,
    }


def _sum_ints(values: Any) -> int | None:
    if values is None:
        return None
    try:
        return sum(int(value) for value in values)
    except (TypeError, ValueError):
        return None


def _empty_payload_stats() -> dict[str, int]:
    return {
        "logical_bytes": 0,
        "unique_storage_bytes": 0,
        "tensor_count": 0,
        "ndarray_count": 0,
    }


class P0BSystemProfiler:
    """Collect one measured outer on the primary Megatron training rank."""

    def __init__(
        self,
        *,
        torch_module: Any,
        rank: int,
        world_size: int,
        primary: bool,
        environ: Mapping[str, str] | None = None,
        output=print,
        ray_snapshot_provider: Callable[[], Mapping[str, Any]] | None = None,
        collective_profiler: Any | None = None,
    ) -> None:
        environ = os.environ if environ is None else environ
        self._environ = environ
        self._torch = torch_module
        requested = _profile_enabled(environ)
        self._measured_outers = _measured_outer_ids(environ)
        self._enabled = primary and requested
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._output = output
        self._ray_snapshot_provider = ray_snapshot_provider or _RayObjectStoreReader()
        self._collective_profiler = (
            collective_profiler
            if collective_profiler is not None
            else FlightRecorderCollectiveProfiler(
                torch_module=torch_module,
                rank=self._rank,
                world_size=self._world_size,
                enabled=self._enabled,
                environ=environ,
            )
        )
        self._source_identity = _source_identity() if self._enabled else None
        self._ray_start: dict[str, Any] | None = None
        self._collective_begin_error: dict[str, Any] | None = None
        self._outer_id: int | None = None
        self._detail: dict[str, Any] = {}
        self._payload_stats = _empty_payload_stats()
        self._predictive_stats = _empty_payload_stats()
        self._payload_observed = False
        self._predictive_observed = False
        self._completed_steps_observed = False
        self._observation_errors: list[str] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def begin_outer(self, outer_id: int) -> None:
        if not self._enabled:
            return
        self._outer_id = int(outer_id)
        self._ray_start = None
        self._collective_begin_error = None
        self._payload_stats = _empty_payload_stats()
        self._predictive_stats = _empty_payload_stats()
        self._payload_observed = False
        self._predictive_observed = False
        self._completed_steps_observed = False
        self._observation_errors = []
        self._detail = {
            "schema_version": _SCHEMA_VERSION,
            "profile_id": _PROFILE_ID,
            "scope": "primary_training_rank_process_local",
            "rank": self._rank,
            "world_size": self._world_size,
            "hostname": socket.gethostname().split(".", 1)[0],
            "slurm_job_id": self._environ.get("SLURM_JOB_ID"),
            "process_id": os.getpid(),
            "source": self._source_identity,
            "timestamps": {},
            "unavailable": [],
            "metric_semantics": {
                "ray_object_store_bytes": (
                    "end-of-outer cluster-global Ray object_store_bytes_used gauge"
                ),
                "ray_spill_bytes": (
                    "cluster-global spilled_bytes_total end-minus-begin delta"
                ),
                "collective_bytes": (
                    "primary-rank logical input-tensor bytes at supported c10d collective calls; "
                    "not multiplied by group size and not network/wire bytes"
                ),
                "collective_time_seconds": (
                    "sum of primary-rank ProcessGroupNCCL accelerator-event durations; "
                    "host enqueue/discovery time excluded"
                ),
            },
        }
        rss, peak_rss = _proc_memory_bytes()
        self._detail["process_memory_at_start"] = {
            "rss_bytes": rss,
            "lifetime_peak_rss_bytes": peak_rss,
        }
        if self._outer_id in self._measured_outers:
            try:
                self._collective_profiler.begin_outer(self._outer_id)
            except Exception as exc:
                self._collective_begin_error = {
                    "available": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:512],
                }
            self._ray_start = self._capture_ray_snapshot()
        cuda = getattr(self._torch, "cuda", None)
        if cuda is not None:
            try:
                cuda.reset_peak_memory_stats()
            except Exception:
                pass
        # The evidence setup above is deliberately outside the measured
        # outer interval.  ``emit_outer`` likewise marks the interval end
        # before its end-of-outer RPC and serialization work.
        self.mark("outer_start")

    def mark(self, name: str) -> None:
        if not self._enabled or self._outer_id is None:
            return
        self._detail.setdefault("timestamps", {})[name] = {
            "wall_time_ns": time.time_ns(),
            "monotonic_time_ns": time.perf_counter_ns(),
        }

    def observe_rollout_payload(self, rollout_data: Any) -> None:
        if not self._enabled or self._outer_id is None:
            return
        if self._payload_observed:
            self._observation_errors.append("observe_rollout_payload called more than once")
            return
        self._payload_observed = True
        self._payload_stats = tensor_payload_stats(rollout_data, self._torch)
        total_lengths = rollout_data.get("total_lengths") if isinstance(rollout_data, Mapping) else None
        response_lengths = rollout_data.get("response_lengths") if isinstance(rollout_data, Mapping) else None
        self._detail["token_counts"] = {
            "local_samples": len(total_lengths) if total_lengths is not None else None,
            "local_total_tokens": _sum_ints(total_lengths),
            "local_response_tokens": _sum_ints(response_lengths),
        }
        self._detail["rollout_payload_tensors"] = dict(self._payload_stats)

    def observe_predictive_cache(self, microbatches: Any) -> None:
        if not self._enabled or self._outer_id is None:
            return
        if self._predictive_observed:
            self._observation_errors.append("observe_predictive_cache called more than once")
            return
        self._predictive_observed = True
        self._predictive_stats = tensor_payload_stats(microbatches, self._torch)
        selected = sum(int(getattr(item, "selected_total_tokens", 0)) for item in microbatches)
        original = sum(int(getattr(item, "original_total_tokens", 0)) for item in microbatches)
        self._detail["retained_predictive_cache"] = {
            **self._predictive_stats,
            "microbatch_count": len(microbatches),
            "selected_tokens": selected,
            "original_tokens": original,
        }

    def observe_completed_actor_steps(self, num_steps_per_outer: int) -> None:
        """Record the actor steps after the real training loop has returned."""

        if not self._enabled or self._outer_id is None:
            return
        if self._completed_steps_observed:
            self._observation_errors.append(
                "observe_completed_actor_steps called more than once"
            )
            return
        self._completed_steps_observed = True
        count = int(num_steps_per_outer)
        if count <= 0:
            self._observation_errors.append(
                f"completed actor step count must be positive, got {count}"
            )
        first = self._outer_id * count
        self._detail["completed_actor_steps"] = {
            "count": count,
            "step_ids": list(range(first, first + count)),
        }

    def emit_outer(self) -> None:
        if not self._enabled or self._outer_id is None:
            return
        outer_id = self._outer_id
        self.mark("outer_end_before_perf_log")
        ray_end: dict[str, Any] | None = None
        if outer_id in self._measured_outers:
            # Freeze the exact collective record/call/Work set at the canonical
            # outer boundary before any profiler work can retire a late async
            # operation.  The freeze is dump-only: it must not poll, write, or
            # wait on Work.  Capture the Ray end gauge immediately afterwards,
            # before RSS/CUDA reads or collective retirement/serialization can
            # perturb cluster-global object-store and spill state.
            if self._collective_begin_error is None:
                try:
                    self._detail["collective_boundary_freeze"] = (
                        self._collective_profiler.freeze_outer_boundary(outer_id)
                    )
                except Exception as exc:
                    self._collective_begin_error = {
                        "available": False,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:512],
                        "stage": "freeze_outer_boundary",
                    }
            ray_end = self._capture_ray_snapshot()

        rss, peak_rss = _proc_memory_bytes()
        accelerator = self._accelerator_memory()
        self._detail["process_memory_at_end"] = {
            "rss_bytes": rss,
            "lifetime_peak_rss_bytes": peak_rss,
        }
        self._detail["accelerator_memory"] = accelerator
        timestamps = self._detail["timestamps"]
        start = timestamps.get("outer_start", {}).get("monotonic_time_ns")
        end = timestamps.get("outer_end_before_perf_log", {}).get("monotonic_time_ns")
        self._detail["outer_wall_seconds"] = None if start is None or end is None else (end - start) / 1e9
        self._detail["phase_wall_seconds"] = {
            phase: self._phase_seconds(timestamps, begin, finish)
            for phase, begin, finish in (
                ("data_preprocess", "data_preprocess_start", "data_preprocess_end"),
                ("actor_update", "actor_update_start", "actor_update_end"),
                ("training_pipeline", "training_pipeline_start", "training_pipeline_end"),
            )
        }

        if outer_id in self._measured_outers:
            required_observations = {
                "rollout_payload": self._payload_observed,
                "predictive_cache": self._predictive_observed,
                "completed_actor_steps": self._completed_steps_observed,
            }
            self._detail["required_observations"] = {
                **required_observations,
                "errors": list(self._observation_errors),
            }
            for name, observed in required_observations.items():
                if not observed:
                    self._detail["unavailable"].append(name)
            if self._collective_begin_error is None:
                collective_bytes, collective_time_seconds, collective_evidence = (
                    self._collective_profiler.end_outer(outer_id)
                )
            else:
                collective_bytes = None
                collective_time_seconds = None
                collective_evidence = self._collective_begin_error
            self._detail["collective_trace"] = collective_evidence
            if collective_bytes is None:
                self._detail["unavailable"].append("collective_bytes")
            if collective_time_seconds is None:
                self._detail["unavailable"].append("collective_time_seconds")
            assert ray_end is not None
            ray_object_store_bytes, ray_spill_bytes, ray_evidence = self._ray_outer_metrics(
                self._ray_start, ray_end
            )
            self._detail["ray_object_store"] = ray_evidence
            if ray_object_store_bytes is None:
                self._detail["unavailable"].append("ray_object_store_bytes")
            if ray_spill_bytes is None:
                self._detail["unavailable"].append("ray_spill_bytes")
            system = {
                "gpu_allocated_bytes": accelerator["peak_allocated_bytes"],
                "gpu_reserved_bytes": accelerator["peak_reserved_bytes"],
                "host_rss_bytes": peak_rss,
                "ray_object_store_bytes": ray_object_store_bytes,
                "ray_spill_bytes": ray_spill_bytes,
                "rollout_payload_bytes": (
                    self._payload_stats["unique_storage_bytes"]
                    if self._payload_observed
                    else None
                ),
                "collective_bytes": collective_bytes,
                "collective_time_seconds": collective_time_seconds,
            }
            observation_gate_complete = (
                all(required_observations.values()) and not self._observation_errors
            )
            if not observation_gate_complete:
                # A P0B_SYSTEM row may never remain fully numeric when any
                # mandatory runtime hook is missing, duplicated, or invalid.
                # Keep independent Ray/collective evidence in P0B_DETAIL, but
                # make the rollout field explicitly unavailable so a generic
                # table consumer cannot mistake this for a claim-ready row.
                system["rollout_payload_bytes"] = None
                self._detail["unavailable"].append("required_observation_contract")
            self._detail["profile_status"] = (
                "complete"
                if (
                    observation_gate_complete
                    and not any(system_value is None for system_value in system.values())
                )
                else "incomplete_evidence"
            )
            compact = {key: value for key, value in self._detail.items()}
            self._output(
                f"P0B_DETAIL outer_id={outer_id} "
                f"{json.dumps(compact, sort_keys=True, separators=(',', ':'))}",
                flush=True,
            )
            self._output(
                f"P0B_SYSTEM outer_id={outer_id} "
                f"{json.dumps(system, sort_keys=True, separators=(',', ':'))}",
                flush=True,
            )
        self._outer_id = None
        self._ray_start = None
        self._collective_begin_error = None

    def _capture_ray_snapshot(self) -> dict[str, Any]:
        try:
            raw = self._ray_snapshot_provider()
            if not isinstance(raw, Mapping):
                raise TypeError(f"Ray snapshot must be a mapping, got {type(raw).__name__}")
            return _validated_ray_snapshot(raw)
        except Exception as exc:
            return {
                "available": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:512],
                "captured_monotonic_ns": time.monotonic_ns(),
            }

    @staticmethod
    def _ray_outer_metrics(
        begin: Mapping[str, Any] | None,
        end: Mapping[str, Any],
    ) -> tuple[int | None, int | None, dict[str, Any]]:
        """Return end-of-outer usage and in-outer cumulative spill delta."""

        end_available = bool(end.get("available"))
        begin_available = begin is not None and bool(begin.get("available"))
        identity_matches = bool(
            begin_available
            and end_available
            and begin.get("ray_version") == end.get("ray_version") == _PINNED_RAY_VERSION
            and begin.get("source") == end.get("source")
            and begin.get("cluster_identity") == end.get("cluster_identity")
            and isinstance(begin.get("captured_monotonic_ns"), int)
            and isinstance(end.get("captured_monotonic_ns"), int)
            and int(end["captured_monotonic_ns"]) >= int(begin["captured_monotonic_ns"])
        )
        if not identity_matches:
            end_available = False
            begin_available = False
        object_store_bytes = int(end["object_store_bytes_used"]) if end_available else None
        spill_bytes: int | None = None
        spill_status = "unavailable"
        if begin_available and end_available:
            before = int(begin["spilled_bytes_total"])
            after = int(end["spilled_bytes_total"])
            if after >= before:
                spill_bytes = after - before
                spill_status = "measured_monotonic_delta"
            else:
                spill_status = "counter_regressed"
        return object_store_bytes, spill_bytes, {
            "scope": "ray_cluster_global",
            "object_store_value": "end_of_outer_object_store_bytes_used_gauge",
            "spill_value": "end_minus_begin_spilled_bytes_total",
            "begin": begin,
            "end": end,
            "spill_status": spill_status,
            "snapshot_identity_matches": identity_matches,
        }

    @staticmethod
    def _phase_seconds(timestamps: Mapping[str, Any], begin: str, finish: str) -> float | None:
        started = timestamps.get(begin, {}).get("monotonic_time_ns")
        ended = timestamps.get(finish, {}).get("monotonic_time_ns")
        if started is None or ended is None:
            return None
        return (ended - started) / 1e9

    def _accelerator_memory(self) -> dict[str, Any]:
        cuda = getattr(self._torch, "cuda", None)
        unavailable = {
            "available": False,
            "device": None,
            "current_allocated_bytes": None,
            "current_reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
        if cuda is None:
            return unavailable
        try:
            if not cuda.is_available():
                return unavailable
            device = int(cuda.current_device())
            return {
                "available": True,
                "device": device,
                "current_allocated_bytes": int(cuda.memory_allocated(device)),
                "current_reserved_bytes": int(cuda.memory_reserved(device)),
                "peak_allocated_bytes": int(cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(cuda.max_memory_reserved(device)),
            }
        except Exception:
            return unavailable
