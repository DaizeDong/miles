#!/usr/bin/env python3

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ROLLOUT_METRICS_RE = re.compile(r"rollout (\d+): (\{.*\})")
PREDICTIVE_STEP_RE = re.compile(r"rollout (\d+): step 0 uses SKIP_PREDICTIVE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the current state of a predictive-routing run from its "
            "checkpoint root and optional train log."
        )
    )
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        required=True,
        help="Checkpoint root for one training run.",
    )
    parser.add_argument(
        "--train-log",
        type=Path,
        default=None,
        help="Optional train host log to parse rollout progress.",
    )
    parser.add_argument(
        "--deep-layer-start",
        type=int,
        default=45,
        help="Layers >= this index are treated as deep layers in summaries.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="aime2024",
        help="Benchmark JSON name to look for under the checkpoint root.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def safe_read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


def find_latest_router_step(router_logits_dir: Path) -> int | None:
    if not router_logits_dir.is_dir():
        return None
    numeric_dirs = [int(p.name) for p in router_logits_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return max(numeric_dirs) if numeric_dirs else None


def find_latest_predictive_metrics_file(step_dir: Path) -> Path | None:
    matches = sorted(step_dir.glob("*_predictive_metrics.json"))
    return matches[-1] if matches else None


def _mean_from_per_layer(per_layer: dict[str, dict[str, float]], metric: str, *, deep_layer_start: int | None = None) -> float | None:
    values = per_layer.get(metric, {})
    if not values:
        return None
    layer_values = []
    for layer_str, value in values.items():
        layer_idx = int(layer_str)
        if deep_layer_start is not None and layer_idx < deep_layer_start:
            continue
        layer_values.append(float(value))
    if not layer_values:
        return None
    return sum(layer_values) / len(layer_values)


def summarize_predictive_metrics(path: Path, deep_layer_start: int) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    per_layer = data.get("per_layer", {})
    aggregates = data.get("aggregates", {})
    raw_debug = data.get("debug", {})
    predictor_stats = raw_debug.get("predictor_param_stats", {})
    compact_debug = {
        key: raw_debug[key]
        for key in (
            "applied_action",
            "capped_by_max_total_tokens",
            "original_total_tokens",
            "selected_total_tokens",
            "total_token_count",
            "predictive_max_total_tokens",
            "predictive_loss_scale",
            "predictive_storage_dtype",
            "predictive_train_mode",
            "rollout_id",
            "router_logits_step_name",
            "sampled_indices",
            "sampled_keep_counts",
        )
        if key in raw_debug
    }
    compact_debug["predictor_param_stats"] = {}
    for stage in ("before_optimizer_step", "after_optimizer_step"):
        stats = predictor_stats.get(stage)
        if not isinstance(stats, dict):
            continue
        compact_debug["predictor_param_stats"][stage] = {
            key: stats[key]
            for key in (
                "weight_abs_sum",
                "weight_l2_norm",
                "max_weight_abs",
                "main_grad_abs_sum",
                "main_grad_l2_norm",
                "max_main_grad_abs",
                "num_predictor_params",
                "num_predictor_params_with_main_grad",
            )
            if key in stats
        }
    result = {
        "path": str(path),
        "step_name": data.get("step"),
        "aggregates": aggregates,
        "debug": compact_debug,
        "all_topk_accuracy": _mean_from_per_layer(per_layer, "predictive_topk_accuracy"),
        "deep_topk_accuracy": _mean_from_per_layer(
            per_layer, "predictive_topk_accuracy", deep_layer_start=deep_layer_start
        ),
        "all_stabilized_ratio": _mean_from_per_layer(per_layer, "predictive_stabilized_bias_to_logits_ratio"),
        "deep_stabilized_ratio": _mean_from_per_layer(
            per_layer, "predictive_stabilized_bias_to_logits_ratio", deep_layer_start=deep_layer_start
        ),
        "all_raw_ratio": _mean_from_per_layer(per_layer, "predictive_raw_bias_to_logits_ratio"),
        "deep_raw_ratio": _mean_from_per_layer(
            per_layer, "predictive_raw_bias_to_logits_ratio", deep_layer_start=deep_layer_start
        ),
        "selected_layers": {},
    }
    selected_layers = [0, 24, 45, 46, 47]
    for layer_idx in selected_layers:
        layer_key = str(layer_idx)
        layer_summary = {}
        for metric in (
            "predictive_topk_accuracy",
            "predictive_stabilized_bias_to_logits_ratio",
            "predictive_raw_bias_to_logits_ratio",
            "predictive_layer_gate_scale",
            "predictive_ratio_clip_scale",
            "predictive_margin_clip_scale_mean",
            "predictive_margin_clip_scale_min",
            "predictive_topk_boundary_margin_mean",
            "predictive_effective_topk_margin_ratio",
            "predictive_topk_margin_ratio_anneal_progress",
        ):
            layer_values = per_layer.get(metric, {})
            if layer_key in layer_values:
                layer_summary[metric] = layer_values[layer_key]
        if layer_summary:
            result["selected_layers"][layer_key] = layer_summary
    return result


def summarize_checkpoint_state(ckpt_root: Path) -> dict[str, Any]:
    iter_dirs = sorted(p.name for p in ckpt_root.glob("iter_*") if p.is_dir())
    latest_tracker_path = ckpt_root / "latest_checkpointed_iteration.txt"
    latest_tracker = None
    if latest_tracker_path.exists():
        latest_tracker = latest_tracker_path.read_text(encoding="utf-8").strip() or None
    hf_root = ckpt_root / "hf"
    hf_dirs = sorted(p.name for p in hf_root.glob("rollout_*") if p.is_dir()) if hf_root.is_dir() else []
    return {
        "iter_dir_count": len(iter_dirs),
        "latest_iter_dir": iter_dirs[-1] if iter_dirs else None,
        "latest_checkpointed_iteration": latest_tracker,
        "hf_export_count": len(hf_dirs),
        "latest_hf_export": hf_dirs[-1] if hf_dirs else None,
    }


def summarize_eval_state(ckpt_root: Path, benchmark: str) -> dict[str, Any]:
    benchmark_files = sorted(
        p
        for p in ckpt_root.rglob(f"{benchmark}.json")
        if "router_logits" not in p.parts and "hf" not in p.parts
    )
    if not benchmark_files:
        return {"benchmark": benchmark, "found": False}

    latest = max(benchmark_files, key=lambda p: p.stat().st_mtime)
    payload = None
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        payload = None

    score = None
    metric_key = None
    if isinstance(payload, dict):
        if "score" in payload:
            score = payload.get("score")
            metric_key = payload.get("metric_key")
        elif benchmark in payload:
            score = payload.get(benchmark)
            metric_key = benchmark
    if isinstance(score, list) and len(score) == 1:
        score = score[0]

    return {
        "benchmark": benchmark,
        "found": True,
        "latest_file": str(latest),
        "score": score,
        "metric_key": metric_key,
    }


def summarize_train_log(train_log: Path | None) -> dict[str, Any]:
    if train_log is None:
        return {"found": False}
    text = safe_read_text(train_log)
    if text is None:
        return {"found": False, "path": str(train_log)}

    lines = [strip_ansi(line) for line in text.splitlines()]
    rollout_metrics = []
    predictive_rollouts = []
    for line in lines:
        rollout_match = ROLLOUT_METRICS_RE.search(line)
        if rollout_match:
            rollout_id = int(rollout_match.group(1))
            metrics_str = rollout_match.group(2)
            try:
                metrics = ast.literal_eval(metrics_str)
            except Exception:
                metrics = None
            rollout_metrics.append((rollout_id, metrics))
            continue
        predictive_match = PREDICTIVE_STEP_RE.search(line)
        if predictive_match:
            predictive_rollouts.append(int(predictive_match.group(1)))

    latest_rollout = None
    latest_rollout_metrics = None
    if rollout_metrics:
        latest_rollout, latest_rollout_metrics = rollout_metrics[-1]

    return {
        "found": True,
        "path": str(train_log),
        "latest_rollout_metrics_id": latest_rollout,
        "latest_rollout_metrics": latest_rollout_metrics,
        "latest_predictive_rollout_id": predictive_rollouts[-1] if predictive_rollouts else None,
        "correct_correct_occurrences": text.count("correct correct correct"),
    }


def main() -> None:
    args = parse_args()
    ckpt_root = args.ckpt_root
    router_logits_dir = ckpt_root / "router_logits"
    latest_router_step = find_latest_router_step(router_logits_dir)

    predictive = {"found": False}
    if latest_router_step is not None:
        step_dir = router_logits_dir / str(latest_router_step)
        metrics_file = find_latest_predictive_metrics_file(step_dir)
        predictive = {
            "found": True,
            "latest_router_step": latest_router_step,
            "step_dir": str(step_dir),
            "files": sorted(p.name for p in step_dir.iterdir()),
        }
        if metrics_file is not None:
            predictive["latest_metrics"] = summarize_predictive_metrics(metrics_file, args.deep_layer_start)

    summary = {
        "ckpt_root": str(ckpt_root),
        "checkpoint_state": summarize_checkpoint_state(ckpt_root),
        "predictive_state": predictive,
        "eval_state": summarize_eval_state(ckpt_root, args.benchmark),
        "train_state": summarize_train_log(args.train_log),
    }

    if args.pretty:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
