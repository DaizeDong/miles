#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze predictive replay state shift using predictive metric tensor sidecars. "
            "Measures how old/current hidden states and routing decisions diverge across depth."
        )
    )
    parser.add_argument(
        "--router-logits-dir",
        type=Path,
        required=True,
        help="Path to the router_logits directory under a checkpoint root.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        required=True,
        help="Comma-separated global rollout steps, e.g. 10,80,150,260",
    )
    parser.add_argument(
        "--mini-index",
        type=int,
        default=1,
        help="Mini-batch index for predictive compute artifacts. Defaults to 1.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=2,
        help="Top-k used for predictive routing accuracy. Defaults to 2.",
    )
    parser.add_argument(
        "--deep-layer-start",
        type=int,
        default=32,
        help="Layers >= this index are treated as the deep band.",
    )
    parser.add_argument(
        "--selected-layers",
        type=str,
        default="0,24,47",
        help="Comma-separated layers to include in the detailed per-step view.",
    )
    parser.add_argument(
        "--max-tokens-per-layer",
        type=int,
        default=4096,
        help="Deterministic token cap per layer for efficient analysis. Use 0 to disable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--selected-layers-only",
        action="store_true",
        help="Only analyze tensors for --selected-layers and use sidecar aggregates for the rest.",
    )
    return parser.parse_args()


def _parse_int_list(values: str) -> list[int]:
    return [int(value.strip()) for value in values.split(",") if value.strip()]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _squeeze_router_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 3 and tensor.shape[1] == 1:
        return tensor.squeeze(1)
    return tensor


def _find_tensor_file(router_logits_dir: Path, step: int, mini_index: int) -> Path:
    step_dir = router_logits_dir / str(step)
    matches = sorted(step_dir.glob(f"training_{step}_mini{mini_index}_tp*_pp*_predictive_metric_tensors.pt"))
    if not matches:
        raise FileNotFoundError(f"No predictive metric tensor file found under {step_dir} for mini_index={mini_index}.")
    return matches[0]


def _find_metrics_file(router_logits_dir: Path, step: int, mini_index: int) -> Path | None:
    step_dir = router_logits_dir / str(step)
    matches = sorted(step_dir.glob(f"training_{step}_mini{mini_index}_tp*_pp*_predictive_metrics.json"))
    return matches[0] if matches else None


def _maybe_subsample_tensor(tensor: torch.Tensor, max_tokens: int | None) -> torch.Tensor:
    if max_tokens is None or max_tokens <= 0 or tensor.shape[0] <= max_tokens:
        return tensor
    indices = torch.linspace(
        0,
        tensor.shape[0] - 1,
        steps=int(max_tokens),
        dtype=torch.float64,
    ).round().to(dtype=torch.long)
    indices = torch.unique(indices, sorted=True)
    return tensor.index_select(0, indices)


def _calculate_topk_accuracy(*, logits1: torch.Tensor, logits2: torch.Tensor, topk: int) -> float:
    _, topk_indices1 = torch.topk(logits1, k=topk, dim=-1)
    _, topk_indices2 = torch.topk(logits2, k=topk, dim=-1)
    matches = (topk_indices1.unsqueeze(-1) == topk_indices2.unsqueeze(-2)).any(dim=-1)
    return float(matches.float().mean().item())


def _hidden_metrics(old_inputs: torch.Tensor, current_inputs: torch.Tensor) -> dict[str, float]:
    old_inputs = old_inputs.float()
    current_inputs = current_inputs.float()
    diff = current_inputs - old_inputs

    old_norm = torch.linalg.vector_norm(old_inputs, dim=-1)
    current_norm = torch.linalg.vector_norm(current_inputs, dim=-1)
    diff_norm = torch.linalg.vector_norm(diff, dim=-1)
    cosine = torch.nn.functional.cosine_similarity(old_inputs, current_inputs, dim=-1, eps=1e-8)

    return {
        "hidden_cosine_mean": float(cosine.mean().item()),
        "hidden_relative_delta_norm": float((diff_norm.mean() / (old_norm.mean() + 1e-10)).item()),
        "hidden_mean_abs_delta_ratio": float(
            (diff.abs().mean() / (old_inputs.abs().mean() + 1e-10)).item()
        ),
        "old_hidden_norm_mean": float(old_norm.mean().item()),
        "current_hidden_norm_mean": float(current_norm.mean().item()),
    }


def _routing_metrics(
    old_logits: torch.Tensor,
    current_logits: torch.Tensor,
    predicted_delta_logits: torch.Tensor,
    topk: int,
) -> dict[str, float]:
    old_logits = old_logits.float()
    current_logits = current_logits.float()
    predicted_delta_logits = predicted_delta_logits.float()
    predicted_logits = old_logits + predicted_delta_logits
    teacher_delta = current_logits - old_logits
    residual = teacher_delta - predicted_delta_logits

    old_to_current_topk = _calculate_topk_accuracy(logits1=old_logits, logits2=current_logits, topk=topk)
    predictive_topk = _calculate_topk_accuracy(logits1=predicted_logits, logits2=current_logits, topk=topk)

    return {
        "old_to_current_topk_accuracy": old_to_current_topk,
        "predictive_topk_accuracy_from_tensors": predictive_topk,
        "predictive_topk_gain_vs_old": float(predictive_topk - old_to_current_topk),
        "teacher_delta_ratio": float((teacher_delta.abs().mean() / (old_logits.abs().mean() + 1e-10)).item()),
        "predictive_delta_ratio": float(
            (predicted_delta_logits.abs().mean() / (old_logits.abs().mean() + 1e-10)).item()
        ),
        "residual_delta_ratio": float((residual.abs().mean() / (old_logits.abs().mean() + 1e-10)).item()),
    }


def summarize_step(
    *,
    router_logits_dir: Path,
    step: int,
    mini_index: int,
    topk: int,
    deep_layer_start: int,
    selected_layers: list[int],
    max_tokens_per_layer: int | None,
    selected_layers_only: bool,
) -> dict[str, object]:
    tensor_file = _find_tensor_file(router_logits_dir, step, mini_index)
    metrics_file = _find_metrics_file(router_logits_dir, step, mini_index)
    payload = torch.load(tensor_file, map_location="cpu", mmap=True)
    sidecar = None
    if metrics_file is not None:
        with open(metrics_file, "r", encoding="utf-8") as f:
            sidecar = json.load(f)

    layer_payload = payload["layers"]
    hidden_cosines = []
    hidden_rel_deltas = []
    old_to_current_topks = []
    predictive_topks = []
    predictive_gains = []
    teacher_delta_ratios = []
    residual_delta_ratios = []

    deep_hidden_cosines = []
    deep_hidden_rel_deltas = []
    deep_old_to_current_topks = []
    deep_predictive_topks = []
    deep_predictive_gains = []
    deep_teacher_delta_ratios = []
    deep_residual_delta_ratios = []

    selected = {}
    layer_items = list(layer_payload.items())
    if selected_layers_only:
        selected_layer_set = {str(layer_idx) for layer_idx in selected_layers}
        layer_items = [(layer_idx_str, tensors) for layer_idx_str, tensors in layer_items if layer_idx_str in selected_layer_set]

    for layer_idx_str, tensors in layer_items:
        layer_idx = int(layer_idx_str)
        old_inputs = _maybe_subsample_tensor(_squeeze_router_dim(tensors["old_inputs"]), max_tokens_per_layer)
        current_inputs = _maybe_subsample_tensor(_squeeze_router_dim(tensors["current_inputs"]), max_tokens_per_layer)
        old_logits = _maybe_subsample_tensor(_squeeze_router_dim(tensors["old_logits"]), max_tokens_per_layer)
        current_logits = _maybe_subsample_tensor(_squeeze_router_dim(tensors["current_logits"]), max_tokens_per_layer)
        predicted_delta_logits = _maybe_subsample_tensor(
            _squeeze_router_dim(tensors["predicted_delta_logits"]),
            max_tokens_per_layer,
        )

        hidden = _hidden_metrics(old_inputs=old_inputs, current_inputs=current_inputs)
        routing = _routing_metrics(
            old_logits=old_logits,
            current_logits=current_logits,
            predicted_delta_logits=predicted_delta_logits,
            topk=topk,
        )

        hidden_cosines.append(hidden["hidden_cosine_mean"])
        hidden_rel_deltas.append(hidden["hidden_relative_delta_norm"])
        old_to_current_topks.append(routing["old_to_current_topk_accuracy"])
        predictive_topks.append(routing["predictive_topk_accuracy_from_tensors"])
        predictive_gains.append(routing["predictive_topk_gain_vs_old"])
        teacher_delta_ratios.append(routing["teacher_delta_ratio"])
        residual_delta_ratios.append(routing["residual_delta_ratio"])

        if layer_idx >= deep_layer_start:
            deep_hidden_cosines.append(hidden["hidden_cosine_mean"])
            deep_hidden_rel_deltas.append(hidden["hidden_relative_delta_norm"])
            deep_old_to_current_topks.append(routing["old_to_current_topk_accuracy"])
            deep_predictive_topks.append(routing["predictive_topk_accuracy_from_tensors"])
            deep_predictive_gains.append(routing["predictive_topk_gain_vs_old"])
            deep_teacher_delta_ratios.append(routing["teacher_delta_ratio"])
            deep_residual_delta_ratios.append(routing["residual_delta_ratio"])

        if layer_idx in selected_layers:
            selected[str(layer_idx)] = {
                **hidden,
                **routing,
            }
            if sidecar is not None:
                per_layer = sidecar.get("per_layer", {})
                topk_map = per_layer.get("predictive_topk_accuracy", {})
                if str(layer_idx) in topk_map:
                    selected[str(layer_idx)]["predictive_topk_accuracy_sidecar"] = float(topk_map[str(layer_idx)])

    summary = {
        "step": int(step),
        "tensor_file": str(tensor_file),
        "metrics_file": None if metrics_file is None else str(metrics_file),
        "num_layers": int(len(layer_payload)),
        "aggregates": {
            "hidden_cosine_mean": _mean(hidden_cosines),
            "hidden_relative_delta_norm_mean": _mean(hidden_rel_deltas),
            "old_to_current_topk_accuracy_mean": _mean(old_to_current_topks),
            "predictive_topk_accuracy_mean": _mean(predictive_topks),
            "predictive_topk_gain_vs_old_mean": _mean(predictive_gains),
            "teacher_delta_ratio_mean": _mean(teacher_delta_ratios),
            "residual_delta_ratio_mean": _mean(residual_delta_ratios),
            "deep_hidden_cosine_mean": _mean(deep_hidden_cosines),
            "deep_hidden_relative_delta_norm_mean": _mean(deep_hidden_rel_deltas),
            "deep_old_to_current_topk_accuracy_mean": _mean(deep_old_to_current_topks),
            "deep_predictive_topk_accuracy_mean": _mean(deep_predictive_topks),
            "deep_predictive_topk_gain_vs_old_mean": _mean(deep_predictive_gains),
            "deep_teacher_delta_ratio_mean": _mean(deep_teacher_delta_ratios),
            "deep_residual_delta_ratio_mean": _mean(deep_residual_delta_ratios),
        },
        "selected_layers": selected,
    }
    if sidecar is not None:
        summary["sidecar_aggregates"] = sidecar.get("aggregates", {})
        if selected_layers_only:
            summary["aggregate_note"] = (
                "Tensor-derived aggregates use only selected layers; full-run aggregate routing metrics "
                "should be read from sidecar_aggregates."
            )
        debug_payload = sidecar.get("debug", {})
        summary["debug"] = {
            "rollout_id": debug_payload.get("rollout_id"),
            "selected_total_tokens": debug_payload.get("selected_total_tokens"),
            "original_total_tokens": debug_payload.get("original_total_tokens"),
            "sampled_indices": debug_payload.get("sampled_indices"),
        }
    return summary


def main() -> None:
    args = parse_args()
    steps = _parse_int_list(args.steps)
    selected_layers = _parse_int_list(args.selected_layers)
    output = {
        "router_logits_dir": str(args.router_logits_dir),
        "topk": int(args.topk),
        "deep_layer_start": int(args.deep_layer_start),
        "selected_layers": selected_layers,
        "max_tokens_per_layer": None if args.max_tokens_per_layer <= 0 else int(args.max_tokens_per_layer),
        "steps": [],
    }
    for step in steps:
        output["steps"].append(
            summarize_step(
                router_logits_dir=args.router_logits_dir,
                step=step,
                mini_index=args.mini_index,
                topk=args.topk,
                deep_layer_start=args.deep_layer_start,
                selected_layers=selected_layers,
                max_tokens_per_layer=None if args.max_tokens_per_layer <= 0 else int(args.max_tokens_per_layer),
                selected_layers_only=bool(args.selected_layers_only),
            )
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, sort_keys=True)
        print(args.output)
        return

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
