#!/usr/bin/env python3
import os
import re
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

import miles.utils.eval_config
from miles.backends.megatron_utils.initialize import init, is_megatron_main_rank
from miles.backends.megatron_utils.model import initialize_model_and_optimizer, save_hf_model
from miles.utils.arguments import parse_args
from miles.utils.distributed_utils import init_gloo_group
from miles.utils.logging_utils import configure_logger


def _resolve_rollout_id(load_path: str) -> int:
    path = Path(load_path).resolve()
    match = re.fullmatch(r"iter_(\d{7})", path.name)
    if match:
        return int(match.group(1))

    tracker = path / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        text = tracker.read_text(encoding="utf-8").strip()
        if text.isdigit():
            return int(text)

    raise ValueError(f"cannot resolve rollout id from load path: {load_path}")


def _hf_export_is_complete(export_path: Path) -> bool:
    if not export_path.is_dir():
        return False

    config_path = export_path / "config.json"
    if not config_path.is_file():
        return False

    single_weight = export_path / "model.safetensors"
    index_file = export_path / "model.safetensors.index.json"
    if single_weight.is_file():
        return True
    if not index_file.is_file():
        return False

    try:
        import json

        index_data = json.loads(index_file.read_text(encoding="utf-8"))
    except Exception:
        return False

    mapped_files = {
        value
        for value in index_data.get("weight_map", {}).values()
        if isinstance(value, str) and value
    }
    if not mapped_files:
        return False
    return all((export_path / value).is_file() for value in mapped_files)


def _format_save_hf_path(save_hf_template: str, rollout_id: int) -> Path:
    try:
        return Path(save_hf_template.format(rollout_id=rollout_id)).resolve()
    except ValueError as exc:
        if "Single '}' encountered in format string" not in str(exc):
            raise
        sanitized = re.sub(r"(\{rollout_id:[^{}]+\})\}+$", r"\1", save_hf_template)
        if sanitized == save_hf_template:
            raise
        print(f"[WARN] Sanitized malformed save_hf template: {save_hf_template} -> {sanitized}")
        return Path(sanitized.format(rollout_id=rollout_id)).resolve()


def main():
    configure_logger()
    args = parse_args()
    requested_load_path = os.environ.get("EXPORT_LOAD_PATH") or args.load
    megatron_load_path = os.environ.get("MEGATRON_LOAD_PATH") or args.load
    if requested_load_path is None:
        raise ValueError("EXPORT_LOAD_PATH is not set and args.load is None; cannot determine rollout id to export.")
    if megatron_load_path is None:
        raise ValueError("MEGATRON_LOAD_PATH is not set and args.load is None; cannot determine Megatron checkpoint root.")

    rollout_id = _resolve_rollout_id(requested_load_path)
    args.load = megatron_load_path

    export_path = _format_save_hf_path(args.save_hf, rollout_id)
    args.save_hf = str(export_path.parent / "rollout_{rollout_id:04d}")

    if _hf_export_is_complete(export_path):
        print(f"[INFO] HF export already exists: {export_path}")
        return
    if export_path.exists():
        print(f"[INFO] HF export is incomplete; retrying export into: {export_path}")

    torch.serialization.add_safe_globals([miles.utils.eval_config.EvalDatasetConfig])

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(f"cuda:{local_rank}")

    dist.init_process_group(
        backend=args.distributed_backend,
        timeout=timedelta(minutes=args.distributed_timeout_minutes),
    )
    init_gloo_group()

    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()

    init(args)
    model, _optimizer, _scheduler, loaded_rollout_id = initialize_model_and_optimizer(args, role="actor")
    if loaded_rollout_id != rollout_id and is_megatron_main_rank():
        print(
            f"[WARN] loaded rollout id {loaded_rollout_id} differs from resolved tracker rollout id {rollout_id}; "
            "keeping the resolved rollout id for export path"
        )

    save_hf_model(args, rollout_id, model)
    dist.barrier()

    export_complete = torch.tensor([1 if _hf_export_is_complete(export_path) else 0], device=f"cuda:{local_rank}")
    dist.all_reduce(export_complete, op=dist.ReduceOp.MIN)
    if int(export_complete.item()) != 1:
        raise RuntimeError(f"HF export is incomplete after save: {export_path}")

    if is_megatron_main_rank():
        print(f"[INFO] HF export completed: {export_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
