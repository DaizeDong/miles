import inspect
import logging
import re
from argparse import Namespace
from collections.abc import Iterator, Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles.backends.megatron_utils.misc_utils import strip_param_name_prefix
from miles.backends.megatron_utils.predictive_router_replay import is_predictive_router_parameter_name
from miles.utils.types import ParamInfo

logger = logging.getLogger(__name__)


def _gather_with_stride(
    param_partitions: list[torch.Tensor], partition_dim: int, partition_stride: int
) -> torch.Tensor:
    """Gather partitions respecting partition_stride (strided/interleaved TP sharding)."""
    if partition_stride == 1:
        return torch.cat(param_partitions, dim=partition_dim)
    # Interleaved (strided) partitioning, e.g. linear_fc1.weight under GLU/SwiGLU
    chunks_per_rank = [p.chunk(partition_stride, dim=partition_dim) for p in param_partitions]
    interleaved = [chunks_per_rank[r][s] for s in range(partition_stride) for r in range(len(param_partitions))]
    return torch.cat(interleaved, dim=partition_dim)


def _check_and_fix_partition(args: Namespace, name: str, partition_stride: int, partition_dim: int) -> tuple[int, int]:
    """Validate partition_stride values for known parameter patterns.

    After Megatron-LM PR #2708, linear_fc1 correctly reports partition_stride=2
    (GLU/SwiGLU interleaved [gate, up]), so assert partition_stride==2 is removed.
    But TEGroupedLinear still does not set partition_stride/partition_dim correctly for grouped moe gemm
    """
    if "linear_fc1.weight" in name and args.swiglu:
        partition_stride = 2
    elif "linear_fc2.weight" in name:
        assert partition_stride == 1, f"Expected partition_stride=1 for {name}, got {partition_stride}"
        if partition_dim == 0:
            partition_dim = 1
    else:
        assert partition_stride == 1, f"Expected partition_stride=1 for {name}, got {partition_stride}"
    return partition_stride, partition_dim


def all_gather_param(args: Namespace, name: str, param: torch.nn.Parameter) -> torch.Tensor:
    """
    All-gather TP-sharded param to full tensor. expert_bias→param, non-TP/duplicated→param.data.
    Uses expert-TP for ".experts.", else regular-TP. Handles strided partitioning via partition_stride.
    """
    if "expert_bias" in name:
        return param

    assert hasattr(param, "tensor_model_parallel"), f"{name} does not have tensor_model_parallel attribute"
    if not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
        return param.data

    if ".experts." in name:
        tp_size = mpu.get_expert_tensor_parallel_world_size()
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group()

    param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
    dist.all_gather(param_partitions, param.data, group=tp_group)
    partition_dim = param.partition_dim
    partition_stride = param.partition_stride

    partition_stride, partition_dim = _check_and_fix_partition(args, name, partition_stride, partition_dim)
    param = _gather_with_stride(param_partitions, partition_dim, partition_stride)
    return param


def all_gather_params_async(
    args: Namespace,
    param_infos_and_params: list[tuple[ParamInfo, torch.Tensor]],
) -> list[torch.Tensor]:
    """
    Parallel TP all-gather for multiple params. Loop 1: for each TP param, allocate buffers +
    dist.all_gather(async_op=True) on expert-TP/regular-TP group (skip expert_bias/non-TP/duplicated).
    Loop 2: wait all NCCL handles (enables overlap). Loop 3: concat partitions + apply GLU rechunk/MoE dim fix.
    """
    # Phase 1: Start all async all_gather operations
    gather_tasks = []
    handles = []

    for info, param in param_infos_and_params:
        # Prepare async all_gather
        if "expert_bias" in info.name:
            gather_tasks.append((info, param, None, None, None, None))
            handles.append(None)
        elif not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
            gather_tasks.append((info, param.data, None, None, None, None))
            handles.append(None)
        else:
            # Start async all_gather
            if ".experts." in info.name:
                tp_size = mpu.get_expert_tensor_parallel_world_size()
                tp_group = mpu.get_expert_tensor_parallel_group()
            else:
                tp_size = mpu.get_tensor_model_parallel_world_size()
                tp_group = mpu.get_tensor_model_parallel_group()

            param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
            handle = dist.all_gather(param_partitions, param.data, group=tp_group, async_op=True)
            gather_tasks.append((info, None, handle, param_partitions, param.partition_dim, param.partition_stride))
            handles.append(handle)

    # Phase 2: Wait for ALL async operations to complete at once
    # This ensures maximum parallelism by not blocking on individual operations
    for handle in handles:
        if handle is not None:
            handle.wait()

    # Phase 3: Process all results after all communications are done
    gathered_params = []
    for info, direct_param, handle, param_partitions, partition_dim, partition_stride in gather_tasks:
        if handle is None:
            # No all_gather needed
            param = direct_param
        else:
            partition_stride, partition_dim = _check_and_fix_partition(
                args, info.name, partition_stride, partition_dim
            )
            param = _gather_with_stride(param_partitions, partition_dim, partition_stride)

        gathered_params.append(param)

    return gathered_params


def named_params_and_buffers(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    convert_to_global_name: bool = True,
    translate_gpu_to_cpu: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    if convert_to_global_name:
        ans = _named_params_and_buffers_global(args, model)
    else:
        ans = _named_params_and_buffers_vanilla(model)

    if translate_gpu_to_cpu:
        ans = ((name, _maybe_get_cpu_backup(tensor)) for name, tensor in ans)

    return ans


def rollout_sync_named_params_and_buffers(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    convert_to_global_name: bool = True,
    translate_gpu_to_cpu: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    return (
        (name, tensor)
        for name, tensor in named_params_and_buffers(
            args,
            model,
            convert_to_global_name=convert_to_global_name,
            translate_gpu_to_cpu=translate_gpu_to_cpu,
        )
        if not is_predictive_router_parameter_name(name)
    )


def _maybe_get_cpu_backup(x: torch.Tensor):
    from torch_memory_saver import torch_memory_saver

    if (cpu_tensor := torch_memory_saver.get_cpu_backup(x)) is not None:
        return cpu_tensor

    return x


def _named_params_and_buffers_vanilla(model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    for vp_stage, model_module in enumerate(model):

        def _compute_fqn(name, vp_stage=vp_stage):
            return f"vp_stages.{vp_stage}.{strip_param_name_prefix(name)}"

        for name, param in model_module.named_parameters():
            yield _compute_fqn(name), param

        for name, buffer in model_module.named_buffers():
            # TODO shall we handle (almost) all buffers like Megatron Bridge
            if "expert_bias" not in name:
                continue
            yield _compute_fqn(name), buffer


def _named_params_and_buffers_global(
    args: Namespace, model: Sequence[torch.nn.Module]
) -> Iterator[tuple[str, torch.Tensor]]:
    """
    Yield (global_name, param/buffer) with consistent names across PP/EP. Adjusts indices for
    virtual PP + EP offsets. Handles decoder.layers, mtp.layers (Multi-Token Prediction), expert_bias.
    """
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()
    if args.num_experts:
        expert_offset = ep_rank * args.num_experts // ep_size

    sig = inspect.signature(get_transformer_layer_offset)
    need_vp_stage = "vp_stage" in sig.parameters

    for vp_stage, model_module in enumerate(model):
        if need_vp_stage:
            layer_offset = get_transformer_layer_offset(model_module.config, vp_stage)
        else:
            layer_offset = get_transformer_layer_offset(model_module.config)
        for name, param in model_module.named_parameters():
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                # MTP (Multi-Token Prediction) layers for speculative decoding
                mtp_layers_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
                match = re.match(mtp_layers_pattern, name)
                if not match:
                    yield name, param
                    continue

                # MTP layer indices start from 0
                layer_idx, rest = match.groups()
                expert_pattern = r"transformer_layer.mlp.experts\.(.+)\.weight(\d+)"
                match = re.match(expert_pattern, rest)
                if not match:
                    yield name, param
                    continue

                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"module.module.mtp.layers.{layer_idx}.transformer_layer.mlp.experts.{rest}.weight{expert_idx}", param
                continue

            layer_idx, rest = match.groups()
            layer_idx = int(layer_idx) + layer_offset

            # Expert weights appear in two Megatron layouts and must yield the
            # SAME canonical global name in both, otherwise the downstream EP
            # broadcast (hf_weight_iterator_direct._get_megatron_full_params)
            # and the HF converters desync across ranks:
            #  - grouped GEMM (TEGroupedLinear): mlp.experts.linear_fcN.weight{idx}
            #  - SequentialMLP (--moe-grouped-gemm off):
            #      mlp.experts.local_experts.{local_idx}.linear_fcN.weight
            # The grouped form already carries a GLOBAL expert index; the
            # SequentialMLP form carries only a per-EP-rank LOCAL index, so
            # every rank would emit identical local names (0..local_experts-1)
            # and the per-expert broadcast would pick inconsistent src ranks.
            # Rewrite the SequentialMLP form to the canonical grouped name with
            # the global index = local_idx + expert_offset.
            grouped_expert_match = re.match(r"mlp.experts\.(.+)\.weight(\d+)", rest)
            seq_expert_match = re.match(r"mlp\.experts\.local_experts\.(\d+)\.(.+)\.weight", rest)
            if grouped_expert_match:
                sub, expert_idx = grouped_expert_match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"module.module.decoder.layers.{layer_idx}.mlp.experts.{sub}.weight{expert_idx}", param
            elif seq_expert_match:
                local_idx, sub = seq_expert_match.groups()
                expert_idx = int(local_idx) + expert_offset
                yield f"module.module.decoder.layers.{layer_idx}.mlp.experts.{sub}.weight{expert_idx}", param
            else:
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", param

        # treat expert bias as normal parameters
        for name, buffer in model_module.named_buffers():
            # TODO shall we handle (almost) all buffers like Megatron Bridge
            if "expert_bias" not in name:
                continue
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                yield name, buffer
            else:
                layer_idx, rest = match.groups()
                layer_idx = int(layer_idx) + layer_offset
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", buffer
