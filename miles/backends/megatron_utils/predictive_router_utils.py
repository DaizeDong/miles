import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

PREDICTIVE_STORAGE_DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


@dataclass(frozen=True)
class PredictiveSampleSelection:
    old_inputs: list[torch.Tensor]
    old_logits: list[torch.Tensor]
    sampled_mask: torch.Tensor
    sampled_indices: list[int]


@dataclass(frozen=True)
class PreparedPredictiveRouterData:
    old_inputs_concat: torch.Tensor | None
    old_logits_concat: torch.Tensor | None
    valid_mask: torch.Tensor
    valid_indices: list[int]
    selected_current_lens: list[int]

    @property
    def has_valid_samples(self) -> bool:
        return self.old_inputs_concat is not None and self.old_logits_concat is not None and bool(self.valid_indices)


def predictive_storage_dtype_to_torch_dtype(storage_dtype: str) -> torch.dtype:
    if storage_dtype not in PREDICTIVE_STORAGE_DTYPE_MAP:
        raise ValueError(
            f"Unsupported predictive storage dtype: {storage_dtype}. "
            f"Expected one of {tuple(PREDICTIVE_STORAGE_DTYPE_MAP)}."
        )
    return PREDICTIVE_STORAGE_DTYPE_MAP[storage_dtype]


def _as_list(values):
    if isinstance(values, np.ndarray):
        return list(values)
    return values


def _as_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def build_predictive_valid_mask(
    *,
    attention_mask: torch.Tensor,
    valid_indices: list[int],
    old_lengths: list[int],
    old_token_positions_list: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, list[int]]:
    seq_lens = attention_mask.sum(dim=1, dtype=torch.int32)
    cumsum_lens = torch.cumsum(seq_lens, dim=0)
    total_valid_tokens = int(cumsum_lens[-1].item()) if cumsum_lens.numel() > 0 else 0
    valid_mask = torch.zeros(total_valid_tokens, dtype=torch.bool, device=attention_mask.device)
    if not valid_indices:
        return valid_mask, []

    start_indices = torch.cat(
        [
            torch.tensor([0], device=attention_mask.device, dtype=torch.long),
            cumsum_lens[:-1],
        ]
    )

    selected_current_lens = []
    for list_idx, batch_idx in enumerate(valid_indices):
        sample_start = int(start_indices[batch_idx].item())
        sample_seq_len = int(seq_lens[batch_idx].item())
        old_len = int(old_lengths[list_idx])

        if old_token_positions_list is None:
            current_len = min(old_len, sample_seq_len)
            if current_len > 0:
                valid_mask[sample_start : sample_start + current_len] = True
            selected_current_lens.append(current_len)
            continue

        token_positions = old_token_positions_list[list_idx].to(device=attention_mask.device, dtype=torch.long)
        current_len = int(token_positions.numel())
        if current_len > 0:
            valid_mask[sample_start + token_positions] = True
        selected_current_lens.append(current_len)

    return valid_mask, selected_current_lens


def select_predictive_samples(
    *,
    old_inputs_list,
    old_logits_list,
    downsample_batch_size: int | None = None,
    max_len_limit: int | None = None,
    storage_dtype: str = "bf16",
    generator: torch.Generator | None = None,
) -> PredictiveSampleSelection:
    old_inputs_list = _as_list(old_inputs_list)
    old_logits_list = _as_list(old_logits_list)
    if len(old_inputs_list) != len(old_logits_list):
        raise ValueError(
            f"old_inputs_list length {len(old_inputs_list)} != old_logits_list length {len(old_logits_list)}"
        )

    normalized_inputs = []
    normalized_logits = []
    valid_indices = []
    for index, (old_input, old_logit) in enumerate(zip(old_inputs_list, old_logits_list, strict=True)):
        old_input = _as_tensor(old_input)
        old_logit = _as_tensor(old_logit)
        normalized_inputs.append(old_input)
        normalized_logits.append(old_logit)
        if old_input is None or old_logit is None:
            continue
        if old_input.shape[0] != old_logit.shape[0]:
            raise ValueError(
                f"Predictive sample {index} has mismatched token lengths: {old_input.shape[0]} vs {old_logit.shape[0]}"
            )
        valid_indices.append(index)

    if downsample_batch_size is None or downsample_batch_size >= len(valid_indices):
        sampled_indices = list(valid_indices)
    else:
        if max_len_limit is None:
            filtered_indices = list(valid_indices)
        else:
            filtered_indices = [index for index in valid_indices if normalized_inputs[index].shape[0] <= max_len_limit]

        if len(filtered_indices) >= downsample_batch_size:
            perm = torch.randperm(len(filtered_indices), generator=generator)[:downsample_batch_size].tolist()
            sampled_indices = sorted(filtered_indices[idx] for idx in perm)
        else:
            sampled_indices = sorted(
                valid_indices,
                key=lambda index: (normalized_inputs[index].shape[0], index),
            )[:downsample_batch_size]
            sampled_indices.sort()

    sampled_mask = torch.zeros(len(normalized_inputs), dtype=torch.bool)
    sampled_mask[sampled_indices] = True
    target_dtype = predictive_storage_dtype_to_torch_dtype(storage_dtype)
    sampled_inputs = [normalized_inputs[index].to(target_dtype) for index in sampled_indices]
    sampled_logits = [normalized_logits[index].to(target_dtype) for index in sampled_indices]
    return PredictiveSampleSelection(
        old_inputs=sampled_inputs,
        old_logits=sampled_logits,
        sampled_mask=sampled_mask,
        sampled_indices=sampled_indices,
    )


def restore_predictive_samples(sampled_values: list[torch.Tensor], sampled_mask: torch.Tensor) -> list[torch.Tensor | None]:
    restored_values: list[torch.Tensor | None] = []
    value_index = 0
    for keep_value in sampled_mask.tolist():
        if keep_value:
            if value_index >= len(sampled_values):
                raise ValueError("sampled_values is shorter than sampled_mask.sum().")
            restored_values.append(sampled_values[value_index])
            value_index += 1
        else:
            restored_values.append(None)

    if value_index != len(sampled_values):
        raise ValueError("sampled_values is longer than sampled_mask.sum().")
    return restored_values


def prepare_predictive_router_data(
    *,
    old_inputs_list,
    old_logits_list,
    attention_mask: torch.Tensor,
    old_token_positions_list=None,
) -> PreparedPredictiveRouterData:
    old_inputs_list = _as_list(old_inputs_list)
    old_logits_list = _as_list(old_logits_list)
    old_token_positions_list = _as_list(old_token_positions_list) if old_token_positions_list is not None else None

    if len(old_inputs_list) != len(old_logits_list):
        raise ValueError(
            f"old_inputs_list length {len(old_inputs_list)} != old_logits_list length {len(old_logits_list)}"
        )
    if len(old_inputs_list) != attention_mask.shape[0]:
        raise ValueError(
            f"Predictive sample count {len(old_inputs_list)} does not match attention_mask batch size {attention_mask.shape[0]}"
        )
    if old_token_positions_list is not None and len(old_token_positions_list) != len(old_inputs_list):
        raise ValueError(
            f"old_token_positions_list length {len(old_token_positions_list)} != predictive sample count {len(old_inputs_list)}"
        )

    seq_lens = attention_mask.sum(dim=1, dtype=torch.int32).tolist()
    valid_indices = []
    valid_old_inputs = []
    valid_old_logits = []
    valid_old_token_positions = []
    use_positions = old_token_positions_list is not None

    for sample_idx, (old_input, old_logit) in enumerate(zip(old_inputs_list, old_logits_list, strict=True)):
        old_input = _as_tensor(old_input)
        old_logit = _as_tensor(old_logit)
        if old_input is None or old_logit is None:
            continue
        if old_input.shape[0] != old_logit.shape[0]:
            logger.warning(
                "Dropping predictive data for sample %s because old_inputs/old_logits token lengths differ: %s vs %s",
                sample_idx,
                old_input.shape[0],
                old_logit.shape[0],
            )
            continue

        if use_positions:
            token_positions = _as_tensor(old_token_positions_list[sample_idx])
            if token_positions is None or token_positions.ndim != 1:
                logger.warning("Dropping predictive data for sample %s because token positions are missing or invalid.", sample_idx)
                continue
            token_positions = token_positions.to(dtype=torch.long)
            if token_positions.shape[0] != old_input.shape[0]:
                logger.warning(
                    "Dropping predictive data for sample %s because token position count %s != token count %s.",
                    sample_idx,
                    token_positions.shape[0],
                    old_input.shape[0],
                )
                continue
            if token_positions.numel() > 0:
                current_seq_len = int(seq_lens[sample_idx])
                if torch.any(token_positions < 0) or torch.any(token_positions >= current_seq_len):
                    logger.warning(
                        "Dropping predictive data for sample %s because token positions exceed current seq len %s.",
                        sample_idx,
                        current_seq_len,
                    )
                    continue
                if torch.unique(token_positions).numel() != token_positions.numel():
                    logger.warning("Dropping predictive data for sample %s because token positions contain duplicates.", sample_idx)
                    continue
            valid_old_token_positions.append(token_positions)
        else:
            current_len = min(int(old_input.shape[0]), int(seq_lens[sample_idx]))
            if current_len <= 0:
                continue
            old_input = old_input[:current_len]
            old_logit = old_logit[:current_len]

        valid_indices.append(sample_idx)
        valid_old_inputs.append(old_input)
        valid_old_logits.append(old_logit)

    old_lengths = [int(tensor.shape[0]) for tensor in valid_old_inputs]
    valid_mask, selected_current_lens = build_predictive_valid_mask(
        attention_mask=attention_mask,
        valid_indices=valid_indices,
        old_lengths=old_lengths,
        old_token_positions_list=valid_old_token_positions if use_positions else None,
    )

    if not valid_old_inputs:
        return PreparedPredictiveRouterData(
            old_inputs_concat=None,
            old_logits_concat=None,
            valid_mask=valid_mask,
            valid_indices=[],
            selected_current_lens=[],
        )

    old_inputs_concat = torch.cat(valid_old_inputs, dim=0)
    old_logits_concat = torch.cat(valid_old_logits, dim=0)
    return PreparedPredictiveRouterData(
        old_inputs_concat=old_inputs_concat,
        old_logits_concat=old_logits_concat,
        valid_mask=valid_mask,
        valid_indices=valid_indices,
        selected_current_lens=selected_current_lens,
    )
