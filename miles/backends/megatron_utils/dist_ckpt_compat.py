from __future__ import annotations

import functools
import logging
import math
import os
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _count_legacy_step_leaves(value: Any, path: tuple[Any, ...] = ()) -> int:
    """Require a container tree whose only leaves are finite scalar ``step`` tensors."""
    if isinstance(value, dict):
        count = 0
        for key, child in value.items():
            child_path = (*path, key)
            if key == "step":
                if (
                    not isinstance(child, torch.Tensor)
                    or child.numel() != 1
                    or not bool(torch.isfinite(child).all().item())
                ):
                    raise RuntimeError(
                        f"legacy optimizer param_state step at {child_path!r} is not a "
                        f"finite scalar tensor: type={type(child).__name__}, "
                        f"shape={getattr(child, 'shape', None)}"
                    )
                count += 1
            elif isinstance(child, (dict, list, tuple)):
                count += _count_legacy_step_leaves(child, child_path)
            else:
                raise RuntimeError(
                    "refusing to discard legacy optimizer param_state because it contains "
                    f"a non-step leaf at {child_path!r}: {type(child).__name__}"
                )
        return count
    if isinstance(value, (list, tuple)):
        return sum(_count_legacy_step_leaves(child, (*path, index)) for index, child in enumerate(value))
    raise RuntimeError(
        "refusing to discard legacy optimizer param_state because it contains "
        f"a non-container leaf at {path!r}: {type(value).__name__}"
    )


def _strip_legacy_common_param_state(common_state: Any) -> int:
    """Strip only pre-fix dp_reshardable param-state trees containing raw steps."""
    if not isinstance(common_state, dict) or "optimizer" not in common_state:
        return 0

    pending: list[tuple[dict, int]] = []

    def visit(value: Any, path: tuple[Any, ...]) -> None:
        if isinstance(value, dict):
            if value.get("param_state_sharding_type") == "dp_reshardable" and "param_state" in value:
                step_count = _count_legacy_step_leaves(value["param_state"], (*path, "param_state"))
                if step_count:
                    pending.append((value, step_count))
            for key, child in value.items():
                if isinstance(child, (dict, list, tuple)):
                    visit(child, (*path, key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                if isinstance(child, (dict, list, tuple)):
                    visit(child, (*path, index))

    visit(common_state["optimizer"], ("optimizer",))

    # Validate every candidate before mutating any loaded state. Removing only
    # the step leaves is insufficient: the rank-local list skeleton itself can
    # differ across DP ranks and still fail dist_checkpointing.dict_utils.merge.
    for optimizer_state, _ in pending:
        del optimizer_state["param_state"]
    return sum(step_count for _, step_count in pending)


def _drop_per_param_steps(value: Any) -> int:
    """Remove per-parameter steps from a dp_reshardable guidance tree."""
    removed = 0
    if isinstance(value, dict):
        if "step" in value:
            del value["step"]
            removed += 1
        for child in value.values():
            removed += _drop_per_param_steps(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            removed += _drop_per_param_steps(child)
    return removed


def install_dp_reshardable_hdo_step_compat() -> None:
    """Install the pinned-MCore CPU-offload checkpoint compatibility fixes once."""
    from megatron.core.dist_checkpointing.strategies.common import TorchCommonLoadStrategy
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    sentinel = "_miles_dp_reshardable_hdo_step_compat"
    if getattr(DistributedOptimizer, sentinel, False):
        return

    original_load_common = TorchCommonLoadStrategy.load_common
    original_sharded_param_state = DistributedOptimizer.sharded_param_state_dp_reshardable
    original_set_param_state = DistributedOptimizer._set_main_param_and_optimizer_states

    @functools.wraps(original_load_common)
    def load_common_without_legacy_step(self, checkpoint_dir):
        common_state = original_load_common(self, checkpoint_dir)
        removed = _strip_legacy_common_param_state(common_state)
        if removed:
            logger.warning(
                "[Miles checkpoint compat] stripped %d leaked dp_reshardable step tensors "
                "from legacy common state %s before merge",
                removed,
                checkpoint_dir,
            )
        return common_state

    @functools.wraps(original_sharded_param_state)
    def sharded_param_state_without_step(self, *args, **kwargs):
        state = original_sharded_param_state(self, *args, **kwargs)
        removed = _drop_per_param_steps(state)
        if removed and (not dist.is_initialized() or dist.get_rank() == 0):
            is_loading = kwargs.get("is_loading", args[1] if len(args) > 1 else False)
            logger.info(
                "[Miles checkpoint compat] excluded %d per-param step entries from "
                "dp_reshardable guidance is_loading=%s",
                removed,
                is_loading,
            )
        return state

    @functools.wraps(original_set_param_state)
    def set_param_state_without_step_override(self, model_param, tensors):
        tensors = {key: value for key, value in tensors.items() if key != "step"}
        if not getattr(self.config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False):
            group_index, group_order = self.model_param_group_index_map[model_param]
            optimizer_param = self.optimizer.param_groups[group_index]["params"][group_order]
            canonical_step = self.optimizer.state.get(optimizer_param, {}).get("step")
            if canonical_step is not None:
                # The original non-precision branch iterates destination keys,
                # including step. Feeding its current, param-group-restored step
                # makes that copy a no-op instead of restoring a local dummy step.
                tensors["step"] = canonical_step
        return original_set_param_state(self, model_param, tensors)

    TorchCommonLoadStrategy._miles_original_load_common = original_load_common
    TorchCommonLoadStrategy.load_common = load_common_without_legacy_step
    DistributedOptimizer._miles_original_sharded_param_state_dp_reshardable = original_sharded_param_state
    DistributedOptimizer.sharded_param_state_dp_reshardable = sharded_param_state_without_step
    DistributedOptimizer._miles_original_set_main_param_and_optimizer_states = original_set_param_state
    DistributedOptimizer._set_main_param_and_optimizer_states = set_param_state_without_step_override
    setattr(DistributedOptimizer, sentinel, True)


def _step_as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"optimizer step is not scalar: shape={tuple(value.shape)}")
        value = value.item()
    step = float(value)
    if not math.isfinite(step):
        raise RuntimeError(f"optimizer step is not finite: {step}")
    return step


def _collect_local_optimizer_resume_record(args, optimizer, iteration: int, predictor_lr: float) -> dict:
    children = getattr(optimizer, "chained_optimizers", [optimizer])
    child_records = []
    predictor_group_count = 0
    predictor_state_count = 0
    predictor_numel = 0
    exp_avg_abs_sum = 0.0
    exp_avg_sq_abs_sum = 0.0

    for child_index, child in enumerate(children):
        inner = child.optimizer
        state_count = len(inner.state)
        hdo_steps = []
        missing_hdo_steps = 0
        for state in inner.state.values():
            if "step" not in state:
                missing_hdo_steps += 1
            else:
                hdo_steps.append(_step_as_float(state["step"]))

        sub_state_count = 0
        sub_steps = []
        missing_sub_steps = 0
        for sub_optimizer in getattr(inner, "sub_optimizers", []):
            for state in sub_optimizer.state.values():
                sub_state_count += 1
                if "step" not in state:
                    missing_sub_steps += 1
                else:
                    sub_steps.append(_step_as_float(state["step"]))

        child_records.append(
            {
                "index": child_index,
                "is_hdo": hasattr(inner, "sub_optimizers"),
                "state_count": state_count,
                "steps": sorted(set(hdo_steps)),
                "missing_steps": missing_hdo_steps,
                "sub_state_count": sub_state_count,
                "sub_steps": sorted(set(sub_steps)),
                "missing_sub_steps": missing_sub_steps,
            }
        )

        for group in inner.param_groups:
            group_max_lr = float(group.get("max_lr", group.get("lr", 0.0)))
            if not math.isclose(group_max_lr, predictor_lr, rel_tol=1e-7, abs_tol=0.0):
                continue
            predictor_group_count += 1
            for param in group["params"]:
                state = inner.state.get(param, {})
                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                if exp_avg is None or exp_avg_sq is None:
                    continue
                predictor_state_count += 1
                predictor_numel += exp_avg.numel()
                exp_avg_abs_sum += float(exp_avg.detach().float().abs().sum().item())
                exp_avg_sq_abs_sum += float(exp_avg_sq.detach().float().abs().sum().item())

    return {
        "rank": dist.get_rank(),
        "iteration": iteration,
        "local_error": None,
        "lrs": [float(group["lr"]) for group in optimizer.param_groups],
        "max_lrs": [float(group.get("max_lr", group["lr"])) for group in optimizer.param_groups],
        "children": child_records,
        "predictor_group_count": predictor_group_count,
        "predictor_state_count": predictor_state_count,
        "predictor_numel": predictor_numel,
        "exp_avg_abs_sum": exp_avg_abs_sum,
        "exp_avg_sq_abs_sum": exp_avg_sq_abs_sum,
    }


def maybe_validate_pr2_optimizer_resume(args, optimizer, iteration: int) -> None:
    """Run the opt-in, distributed PR2 optimizer-resume acceptance probe."""
    raw_expected_step = os.environ.get("PR2_EXPECT_RESUME_HDO_STEP")
    if (
        os.environ.get("PR2_VALIDATE", "0") != "1"
        or raw_expected_step is None
        or not raw_expected_step.strip()
    ):
        return

    expected_step = float(raw_expected_step.strip())
    expected_iteration = int(os.environ.get("PR2_EXPECT_CHECKPOINT_ITERATION", iteration))
    expected_predictor_numel = int(os.environ.get("PR2_EXPECT_PREDICTOR_NUMEL", "0")) or None
    base_lr = float(args.lr)
    predictor_lr = base_lr * float(args.bias_predictor_lr_mult)
    expected_lrs = sorted((base_lr, predictor_lr, base_lr))

    try:
        local_record = _collect_local_optimizer_resume_record(args, optimizer, iteration, predictor_lr)
    except Exception as exc:
        # Every rank must still enter the collective so a malformed local state
        # is reported coherently instead of stranding the remaining ranks.
        local_record = {
            "rank": dist.get_rank(),
            "iteration": iteration,
            "local_error": f"{type(exc).__name__}: {exc}",
        }

    from miles.utils.distributed_utils import get_gloo_group

    gloo_group = get_gloo_group()
    records = [None] * dist.get_world_size(group=gloo_group)
    dist.all_gather_object(records, local_record, group=gloo_group)

    error = None
    if dist.get_rank() == 0:
        errors = []
        for record in records:
            if record["local_error"] is not None:
                errors.append(f"rank {record['rank']} local probe failed: {record['local_error']}")
                continue
            if record["iteration"] != expected_iteration:
                errors.append(f"rank {record['rank']} iteration={record['iteration']}")
            for field in ("lrs", "max_lrs"):
                actual = sorted(record[field])
                if len(actual) != 3 or any(
                    not math.isclose(got, want, rel_tol=1e-7, abs_tol=0.0)
                    for got, want in zip(actual, expected_lrs)
                ):
                    errors.append(f"rank {record['rank']} {field}={record[field]}")
            for child in record["children"]:
                if not child["is_hdo"] or child["state_count"] == 0:
                    errors.append(f"rank {record['rank']} child {child['index']} is not populated HDO")
                    continue
                if child["missing_steps"] or child["steps"] != [expected_step]:
                    errors.append(
                        f"rank {record['rank']} child {child['index']} HDO "
                        f"steps={child['steps']} missing={child['missing_steps']}"
                    )
                if child["missing_sub_steps"] or child["sub_steps"] != [expected_step]:
                    errors.append(
                        f"rank {record['rank']} child {child['index']} suboptimizer "
                        f"steps={child['sub_steps']} missing={child['missing_sub_steps']}"
                    )

        valid_records = [record for record in records if record["local_error"] is None]
        total_predictor_groups = sum(record["predictor_group_count"] for record in valid_records)
        total_predictor_states = sum(record["predictor_state_count"] for record in valid_records)
        total_predictor_numel = sum(record["predictor_numel"] for record in valid_records)
        total_exp_avg = sum(record["exp_avg_abs_sum"] for record in valid_records)
        total_exp_avg_sq = sum(record["exp_avg_sq_abs_sum"] for record in valid_records)
        if expected_predictor_numel is not None and total_predictor_numel != expected_predictor_numel:
            errors.append(f"predictor_numel={total_predictor_numel}, expected={expected_predictor_numel}")
        if (
            total_predictor_groups == 0
            or total_predictor_states == 0
            or total_predictor_numel == 0
            or not math.isfinite(total_exp_avg)
            or not math.isfinite(total_exp_avg_sq)
            or total_exp_avg <= 0.0
            or total_exp_avg_sq <= 0.0
        ):
            errors.append(
                "predictor Adam moments missing/nonfinite/zero: "
                f"groups={total_predictor_groups} states={total_predictor_states} "
                f"numel={total_predictor_numel} exp_avg_abs_sum={total_exp_avg} "
                f"exp_avg_sq_abs_sum={total_exp_avg_sq}"
            )

        first_valid_record = valid_records[0] if valid_records else {"lrs": [], "max_lrs": []}
        logger.info(
            "[PR2_VALIDATE] optimizer_resume iteration=%s expected_hdo_step=%s "
            "lr_groups=%s max_lr_groups=%s predictor_groups=%s predictor_states=%s "
            "predictor_numel=%s expected_predictor_numel=%s exp_avg_abs_sum=%.9g "
            "exp_avg_sq_abs_sum=%.9g full=%s errors=%s",
            iteration,
            expected_step,
            first_valid_record["lrs"],
            first_valid_record["max_lrs"],
            total_predictor_groups,
            total_predictor_states,
            total_predictor_numel,
            expected_predictor_numel,
            total_exp_avg,
            total_exp_avg_sq,
            not errors,
            errors,
        )
        if errors:
            error = "; ".join(errors)

    result = [error]
    dist.broadcast_object_list(result, src=0, group=gloo_group)
    if result[0] is not None:
        raise RuntimeError(f"[PR2_VALIDATE] optimizer resume validation failed: {result[0]}")


def maybe_validate_pr2_model_reload(args, model, iteration: int) -> None:
    """Validate the predictor model tensors after a model-only checkpoint reload.

    This probe is deliberately independent of optimizer state: the rebuttal
    model-only gate uses ``--no-load-optim`` and therefore cannot reuse the
    optimizer-resume acceptance check above.  The frozen Moonlight topology
    keeps a complete copy of every router predictor on each rank, so each rank
    must independently match the audited parameter count and contain finite,
    non-zero loaded weights.
    """

    if os.environ.get("PR2_MODEL_ONLY_RELOAD_GATE", "0") != "1":
        return
    if os.environ.get("PR2_VALIDATE", "0") != "1":
        raise RuntimeError("PR2 model-only reload validation requires PR2_VALIDATE=1")
    if (
        int(getattr(args, "tensor_model_parallel_size", 0)) != 1
        or int(getattr(args, "pipeline_model_parallel_size", 0)) != 1
    ):
        raise RuntimeError("PR2 model-only reload validation requires tensor/pipeline parallel size 1")

    raw_expected_numel = os.environ.get("PR2_EXPECT_PREDICTOR_NUMEL")
    if raw_expected_numel is None:
        raise RuntimeError("PR2 model-only reload validation requires PR2_EXPECT_PREDICTOR_NUMEL")
    try:
        expected_predictor_numel = int(raw_expected_numel)
    except ValueError as exc:
        raise RuntimeError("PR2_EXPECT_PREDICTOR_NUMEL must be a positive integer") from exc
    if expected_predictor_numel <= 0:
        raise RuntimeError("PR2_EXPECT_PREDICTOR_NUMEL must be a positive integer")

    raw_expected_iteration = os.environ.get("PR2_EXPECT_CHECKPOINT_ITERATION")
    if raw_expected_iteration is None:
        raise RuntimeError("PR2 model-only reload validation requires PR2_EXPECT_CHECKPOINT_ITERATION")
    try:
        expected_iteration = int(raw_expected_iteration)
    except ValueError as exc:
        raise RuntimeError("PR2_EXPECT_CHECKPOINT_ITERATION must be a positive integer") from exc
    if expected_iteration <= 0:
        raise RuntimeError("PR2_EXPECT_CHECKPOINT_ITERATION must be a positive integer")

    try:
        from .predictive_router_replay import collect_predictive_param_stats

        stats = collect_predictive_param_stats(model)
        local_record = {
            "rank": dist.get_rank(),
            "iteration": iteration,
            "local_error": None,
            "num_predictor_params": int(stats["num_predictor_params"]),
            "num_nonzero_predictor_params": int(stats["num_nonzero_predictor_params"]),
            "predictor_numel": int(stats["predictor_numel"]),
            "all_weights_finite": bool(stats["all_weights_finite"]),
            "weight_sum": float(stats["weight_sum"]),
            "weight_abs_sum": float(stats["weight_abs_sum"]),
            "weight_l2_norm": float(stats["weight_l2_norm"]),
        }
    except Exception as exc:
        # All ranks must still enter the collective so one malformed shard does
        # not strand the rest of the allocation.
        local_record = {
            "rank": dist.get_rank(),
            "iteration": iteration,
            "local_error": f"{type(exc).__name__}: {exc}",
        }

    from miles.utils.distributed_utils import get_gloo_group

    gloo_group = get_gloo_group()
    records = [None] * dist.get_world_size(group=gloo_group)
    dist.all_gather_object(records, local_record, group=gloo_group)

    error = None
    if dist.get_rank() == 0:
        errors = []
        for record in records:
            if record["local_error"] is not None:
                errors.append(f"rank {record['rank']} local probe failed: {record['local_error']}")
                continue
            if record["iteration"] != expected_iteration:
                errors.append(
                    f"rank {record['rank']} iteration={record['iteration']}, expected={expected_iteration}"
                )
            if record["predictor_numel"] != expected_predictor_numel:
                errors.append(
                    f"rank {record['rank']} predictor_numel={record['predictor_numel']}, "
                    f"expected={expected_predictor_numel}"
                )
            if record["num_predictor_params"] <= 0:
                errors.append(f"rank {record['rank']} has no predictor parameter tensors")
            if record["num_nonzero_predictor_params"] != record["num_predictor_params"]:
                errors.append(
                    f"rank {record['rank']} nonzero predictor tensors="
                    f"{record['num_nonzero_predictor_params']}/{record['num_predictor_params']}"
                )
            checksums = (
                record["weight_sum"],
                record["weight_abs_sum"],
                record["weight_l2_norm"],
            )
            if not record["all_weights_finite"] or not all(math.isfinite(value) for value in checksums):
                errors.append(f"rank {record['rank']} predictor weights/checksums are non-finite")
            if record["weight_abs_sum"] <= 0.0 or record["weight_l2_norm"] <= 0.0:
                errors.append(f"rank {record['rank']} predictor weights are all zero")

        valid_records = [record for record in records if record["local_error"] is None]
        logger.info(
            "[PR2_VALIDATE] model_reload iteration=%s expected_iteration=%s "
            "expected_predictor_numel=%s rank_records=%s full=%s errors=%s",
            iteration,
            expected_iteration,
            expected_predictor_numel,
            valid_records,
            not errors,
            errors,
        )
        if errors:
            error = "; ".join(errors)

    result = [error]
    dist.broadcast_object_list(result, src=0, group=gloo_group)
    if result[0] is not None:
        raise RuntimeError(f"[PR2_VALIDATE] model-only reload validation failed: {result[0]}")
