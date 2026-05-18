#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize predictive router metric tensors and evaluate offline "
            "counterfactual depth-gate / trust-region stabilizer settings."
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
        help="Comma-separated global rollout steps, e.g. 10,40,90,140",
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
        "--configs",
        type=str,
        default="sqrt_decay:0.5:0.02,sqrt_decay:0.5:0.015,linear_decay:0.5:0.02",
        help=(
            "Comma-separated counterfactual configs in the form "
            "schedule:min_scale:ratio_cap. Use 'none' for ratio_cap to disable clipping."
        ),
    )
    parser.add_argument(
        "--max-tokens-per-layer",
        type=int,
        default=2048,
        help=(
            "Optional deterministic token cap applied independently per layer "
            "before computing top-k accuracies. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Prints to stdout when omitted.",
    )
    return parser.parse_args()


def _parse_int_list(values: str) -> list[int]:
    return [int(value.strip()) for value in values.split(",") if value.strip()]


def _parse_configs(values: str) -> list[dict[str, object]]:
    configs = []
    for item in values.split(","):
        item = item.strip()
        if not item:
            continue
        schedule, min_scale, ratio_cap = [part.strip() for part in item.split(":")]
        ratio_cap_value = None if ratio_cap.lower() == "none" else float(ratio_cap)
        config_name = f"{schedule}_m{min_scale}_r{ratio_cap}"
        config_name = config_name.replace(".", "")
        configs.append(
            {
                "name": config_name,
                "schedule": schedule,
                "min_scale": float(min_scale),
                "ratio_cap": ratio_cap_value,
            }
        )
    return configs


def calculate_topk_accuracy(
    *,
    logits1: torch.Tensor | None = None,
    logits2: torch.Tensor | None = None,
    topk_indices1: torch.Tensor | None = None,
    topk_indices2: torch.Tensor | None = None,
    topk: int,
) -> float:
    if topk_indices1 is None:
        if logits1 is None:
            raise ValueError("logits1 must be provided when topk_indices1 is None.")
        _, topk_indices1 = torch.topk(logits1, k=topk, dim=-1)
    if topk_indices2 is None:
        if logits2 is None:
            raise ValueError("logits2 must be provided when topk_indices2 is None.")
        _, topk_indices2 = torch.topk(logits2, k=topk, dim=-1)
    matches = (topk_indices1.unsqueeze(-1) == topk_indices2.unsqueeze(-2)).any(dim=-1)
    return float(matches.float().mean().item())


def compute_layer_scale(*, layer_idx: int, num_layers: int, schedule: str, min_scale: float) -> float:
    if schedule == "none" or num_layers <= 1:
        return 1.0

    depth_fraction = min(max(float(layer_idx) / float(num_layers - 1), 0.0), 1.0)
    if schedule == "linear_decay":
        decay_fraction = depth_fraction
    elif schedule == "sqrt_decay":
        decay_fraction = depth_fraction**0.5
    elif schedule == "cosine_decay":
        decay_fraction = 0.5 * (1.0 - math.cos(math.pi * depth_fraction))
    else:
        raise ValueError(f"Unsupported schedule: {schedule}")
    return 1.0 - decay_fraction * (1.0 - float(min_scale))


def stabilize_delta(
    *,
    predicted_delta_logits: torch.Tensor,
    old_logits: torch.Tensor,
    layer_idx: int,
    num_layers: int,
    schedule: str,
    min_scale: float,
    ratio_cap: float | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    stabilized = predicted_delta_logits.float()
    gate = compute_layer_scale(
        layer_idx=layer_idx,
        num_layers=num_layers,
        schedule=schedule,
        min_scale=min_scale,
    )
    stabilized = stabilized * gate

    clip = 1.0
    if ratio_cap is not None:
        reference_mean_abs = float(old_logits.float().abs().mean().item())
        delta_mean_abs = float(stabilized.abs().mean().item())
        if reference_mean_abs > 1e-10 and delta_mean_abs > reference_mean_abs * ratio_cap:
            clip = (reference_mean_abs * ratio_cap) / (delta_mean_abs + 1e-10)
            stabilized = stabilized * clip

    return stabilized, {
        "gate": float(gate),
        "clip": float(clip),
        "ratio": float((stabilized.abs().mean() / (old_logits.float().abs().mean() + 1e-10)).item()),
    }


def _squeeze_router_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 3 and tensor.shape[1] == 1:
        return tensor.squeeze(1)
    return tensor


def find_predictive_tensor_file(router_logits_dir: Path, step: int, mini_index: int) -> Path:
    step_dir = router_logits_dir / str(step)
    matches = sorted(step_dir.glob(f"training_{step}_mini{mini_index}_tp*_pp*_predictive_metric_tensors.pt"))
    if not matches:
        raise FileNotFoundError(
            f"No predictive metric tensor file found under {step_dir} for mini_index={mini_index}."
        )
    return matches[0]


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def maybe_subsample_tokens(tensor: torch.Tensor, max_tokens_per_layer: int | None) -> tuple[torch.Tensor, int, int]:
    original_tokens = int(tensor.shape[0])
    if max_tokens_per_layer is None or max_tokens_per_layer <= 0 or original_tokens <= max_tokens_per_layer:
        return tensor, original_tokens, original_tokens

    indices = torch.linspace(
        0,
        original_tokens - 1,
        steps=int(max_tokens_per_layer),
        dtype=torch.float64,
    ).round().to(dtype=torch.long)
    indices = torch.unique(indices, sorted=True)
    return tensor.index_select(0, indices), original_tokens, int(indices.numel())


def summarize_step(
    *,
    payload: dict[str, object],
    step: int,
    topk: int,
    deep_layer_start: int,
    selected_layers: list[int],
    configs: list[dict[str, object]],
    max_tokens_per_layer: int | None,
) -> dict[str, object]:
    layer_payload = payload["layers"]
    num_layers = len(layer_payload)
    raw_accs: list[float] = []
    raw_ratios: list[float] = []
    deep_raw_accs: list[float] = []
    deep_raw_ratios: list[float] = []
    details: dict[str, dict[str, object]] = {}
    config_accs = {config["name"]: [] for config in configs}
    config_ratios = {config["name"]: [] for config in configs}
    config_deep_accs = {config["name"]: [] for config in configs}
    config_deep_ratios = {config["name"]: [] for config in configs}

    for layer_idx_str, tensors in layer_payload.items():
        layer_idx = int(layer_idx_str)
        old_logits = _squeeze_router_dim(tensors["old_logits"]).float()
        current_logits = _squeeze_router_dim(tensors["current_logits"]).float()
        predicted_delta_logits = _squeeze_router_dim(tensors["predicted_delta_logits"]).float()

        old_logits, original_tokens, kept_tokens = maybe_subsample_tokens(old_logits, max_tokens_per_layer)
        current_logits, _, _ = maybe_subsample_tokens(current_logits, max_tokens_per_layer)
        predicted_delta_logits, _, _ = maybe_subsample_tokens(predicted_delta_logits, max_tokens_per_layer)
        _, current_topk_indices = torch.topk(current_logits, k=topk, dim=-1)

        raw_ratio = float((predicted_delta_logits.abs().mean() / (old_logits.abs().mean() + 1e-10)).item())
        raw_acc = calculate_topk_accuracy(
            logits1=old_logits + predicted_delta_logits,
            topk_indices2=current_topk_indices,
            topk=topk,
        )
        raw_accs.append(raw_acc)
        raw_ratios.append(raw_ratio)
        if layer_idx >= deep_layer_start:
            deep_raw_accs.append(raw_acc)
            deep_raw_ratios.append(raw_ratio)

        layer_detail = None
        if layer_idx in selected_layers:
            layer_detail = {
                "raw_ratio": raw_ratio,
                "raw_acc": raw_acc,
                "num_tokens_before_cap": original_tokens,
                "num_tokens_after_cap": kept_tokens,
            }

        for config in configs:
            stabilized, metrics = stabilize_delta(
                predicted_delta_logits=predicted_delta_logits,
                old_logits=old_logits,
                layer_idx=layer_idx,
                num_layers=num_layers,
                schedule=config["schedule"],
                min_scale=float(config["min_scale"]),
                ratio_cap=config["ratio_cap"],
            )
            stabilized_acc = calculate_topk_accuracy(
                logits1=old_logits + stabilized,
                topk_indices2=current_topk_indices,
                topk=topk,
            )
            config_accs[config["name"]].append(stabilized_acc)
            config_ratios[config["name"]].append(metrics["ratio"])
            if layer_idx >= deep_layer_start:
                config_deep_accs[config["name"]].append(stabilized_acc)
                config_deep_ratios[config["name"]].append(metrics["ratio"])

            if layer_detail is not None:
                layer_detail[config["name"]] = {
                    "gate": metrics["gate"],
                    "clip": metrics["clip"],
                    "ratio": metrics["ratio"],
                    "acc": stabilized_acc,
                }

        if layer_detail is not None:
            details[str(layer_idx)] = layer_detail

    summary: dict[str, object] = {
        "step": int(step),
        "raw_all_acc": average(raw_accs),
        "raw_all_ratio": average(raw_ratios),
        "raw_deep_acc": average(deep_raw_accs),
        "raw_deep_ratio": average(deep_raw_ratios),
        "max_tokens_per_layer": None if max_tokens_per_layer is None or max_tokens_per_layer <= 0 else int(max_tokens_per_layer),
        "selected_layers": details,
    }
    for config in configs:
        config_name = config["name"]
        summary[f"{config_name}_all_acc"] = average(config_accs[config_name])
        summary[f"{config_name}_all_ratio"] = average(config_ratios[config_name])
        summary[f"{config_name}_deep_acc"] = average(config_deep_accs[config_name])
        summary[f"{config_name}_deep_ratio"] = average(config_deep_ratios[config_name])
    return summary


def main() -> None:
    args = parse_args()
    steps = _parse_int_list(args.steps)
    selected_layers = _parse_int_list(args.selected_layers)
    configs = _parse_configs(args.configs)

    output = {
        "router_logits_dir": str(args.router_logits_dir),
        "steps": steps,
        "mini_index": int(args.mini_index),
        "topk": int(args.topk),
        "deep_layer_start": int(args.deep_layer_start),
        "max_tokens_per_layer": None if args.max_tokens_per_layer <= 0 else int(args.max_tokens_per_layer),
        "selected_layers": selected_layers,
        "configs": configs,
        "summaries": [],
    }

    for step in steps:
        tensor_file = find_predictive_tensor_file(args.router_logits_dir, step=step, mini_index=args.mini_index)
        payload = torch.load(tensor_file, map_location="cpu")
        step_summary = summarize_step(
            payload=payload,
            step=step,
            topk=args.topk,
            deep_layer_start=args.deep_layer_start,
            selected_layers=selected_layers,
            configs=configs,
            max_tokens_per_layer=args.max_tokens_per_layer,
        )
        step_summary["tensor_file"] = str(tensor_file)
        output["summaries"].append(step_summary)

    rendered = json.dumps(output, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
