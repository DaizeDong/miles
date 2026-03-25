import logging
from contextlib import contextmanager
from enum import Enum
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RouterPredictiveAction(str, Enum):
    DISABLED = "disabled"
    RECORD = "record"
    SKIP_PREDICTIVE = "skip_predictive"
    COMPUTE_PREDICTIVE_LOSS = "compute_predictive_loss"


def is_predictive_router_parameter_name(name: str) -> bool:
    return "bias_predictor" in name


def calculate_topk_accuracy(
    *,
    topk: int,
    logits1: torch.Tensor | None = None,
    logits2: torch.Tensor | None = None,
    topk_indices1: torch.Tensor | None = None,
    topk_indices2: torch.Tensor | None = None,
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
    return matches.float().mean().item()


def compute_predictive_bias_ratio(predicted_delta_logits: torch.Tensor, reference_logits: torch.Tensor) -> float:
    denom = torch.abs(reference_logits).mean() + 1e-10
    return (torch.abs(predicted_delta_logits).mean() / denom).item()


def compute_predictive_loss(
    *,
    old_logits: torch.Tensor,
    current_logits: torch.Tensor,
    predicted_delta_logits: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    if old_logits.shape != current_logits.shape:
        raise ValueError(f"old_logits shape {old_logits.shape} != current_logits shape {current_logits.shape}")
    if old_logits.shape != predicted_delta_logits.shape:
        raise ValueError(
            f"old_logits shape {old_logits.shape} != predicted_delta_logits shape {predicted_delta_logits.shape}"
        )

    logits_diff = current_logits - old_logits

    if loss_type == "l2":
        return F.mse_loss(predicted_delta_logits, logits_diff.detach(), reduction="mean")

    if loss_type == "kl":
        pred_log_probs = torch.log_softmax(predicted_delta_logits, dim=-1)
        target_probs = torch.softmax(logits_diff, dim=-1)
        return F.kl_div(pred_log_probs, target_probs.detach(), reduction="batchmean")

    if loss_type == "kl-post":
        pred_log_probs = torch.log_softmax(old_logits + predicted_delta_logits, dim=-1)
        target_probs = torch.softmax(current_logits, dim=-1)
        return F.kl_div(pred_log_probs, target_probs.detach(), reduction="batchmean")

    raise ValueError(f"Unsupported predictive loss type: {loss_type}")


def build_synthetic_predictive_loss(
    *,
    bias_predictor: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    synthetic_delta_logits = bias_predictor(input_tensor.detach())
    return (synthetic_delta_logits * 0.0).sum()


def _ensure_bias_predictor_runtime_placement(
    *,
    bias_predictor: nn.Module,
    reference_tensor: torch.Tensor,
) -> None:
    target_device = reference_tensor.device
    target_dtype = reference_tensor.dtype

    for parameter in bias_predictor.parameters():
        if parameter.device == target_device and parameter.dtype == target_dtype:
            continue
        parameter.data = parameter.data.to(device=target_device, dtype=target_dtype)
        if parameter.grad is not None:
            parameter.grad.data = parameter.grad.data.to(device=target_device, dtype=target_dtype)
        main_grad = getattr(parameter, "main_grad", None)
        if main_grad is not None:
            parameter.main_grad = main_grad.to(device=target_device, dtype=target_dtype)

    for buffer_name, buffer in bias_predictor.named_buffers():
        if buffer.device == target_device and buffer.dtype == target_dtype:
            continue
        setattr(bias_predictor, buffer_name, buffer.to(device=target_device, dtype=target_dtype))


def _iter_module_chunks(model_chunks):
    if isinstance(model_chunks, torch.nn.Module):
        model_chunks = [model_chunks]
    for model_chunk in model_chunks:
        yield getattr(model_chunk, "module", model_chunk)


def _build_bias_predictor(router) -> nn.Linear:
    gating_weight = getattr(router.gating, "weight", None)
    if gating_weight is not None:
        in_features = gating_weight.shape[1]
        out_features = gating_weight.shape[0]
        device = gating_weight.device
        dtype = gating_weight.dtype
    else:
        in_features = router.config.hidden_size
        out_features = router.config.num_moe_experts
        device = None
        dtype = None

    bias_predictor = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
    nn.init.zeros_(bias_predictor.weight)
    if device is not None or dtype is not None:
        bias_predictor = bias_predictor.to(device=device, dtype=dtype)

    for param in bias_predictor.parameters():
        setattr(param, "is_bias_predictor", True)
        setattr(param, "allreduce", True)
        setattr(param, "sequence_parallel", False)
        setattr(param, "tensor_model_parallel", False)
        setattr(param, "partition_dim", 0)
        setattr(param, "partition_stride", 1)

    return bias_predictor


def initialize_predictive_router_modules(
    *,
    model_chunks,
    enabled: bool,
    loss_type: str,
    lr_mult: float,
) -> None:
    from megatron.core.transformer.moe.router import TopKRouter

    PredictiveRouterReplayState.reset_registry()

    seen_modules = set()
    for module_chunk in _iter_module_chunks(model_chunks):
        for submodule in module_chunk.modules():
            if not isinstance(submodule, TopKRouter):
                continue
            if id(submodule) in seen_modules:
                continue
            seen_modules.add(id(submodule))

            submodule.config.enable_router_bias_predictor = enabled
            submodule.config.bias_predictor_loss_type = loss_type
            submodule.config.bias_predictor_lr_mult = lr_mult

            if not enabled:
                submodule.predictive_router_replay = None
                if hasattr(submodule, "bias_predictor"):
                    submodule.bias_predictor = None
                continue

            PredictiveRouterReplayState.register_router(submodule)
            submodule.bias_predictor = _build_bias_predictor(submodule)


def apply_predictive_router_replay_patch() -> None:
    from megatron.core.transformer.moe.moe_utils import apply_random_logits
    from megatron.core.transformer.moe.router import TopKRouter

    if hasattr(TopKRouter, "_predictive_router_replay_patched"):
        return

    original_forward = TopKRouter.forward

    def patched_forward(self, input: torch.Tensor):
        predictive_state = getattr(self, "predictive_router_replay", None)
        bias_predictor = getattr(self, "bias_predictor", None)

        if predictive_state is None or bias_predictor is None:
            return original_forward(self, input)

        _ensure_bias_predictor_runtime_placement(bias_predictor=bias_predictor, reference_tensor=input)

        predictive_action = predictive_state.predictive_action
        if predictive_action in {None, RouterPredictiveAction.DISABLED, RouterPredictiveAction.SKIP_PREDICTIVE}:
            return original_forward(self, input)

        self._maintain_float32_expert_bias()
        input = self.apply_input_jitter(input)
        logits = self.gating(input)

        if self.config.moe_router_force_load_balancing:
            logits = apply_random_logits(logits)

        if predictive_action == RouterPredictiveAction.RECORD:
            with torch.no_grad():
                predictive_state.record_predictive_data(input, logits)
                predicted_delta_logits = bias_predictor(input)
                PredictiveRouterReplayState.record_predictive_bias_ratio(
                    predictive_state.layer_idx,
                    compute_predictive_bias_ratio(predicted_delta_logits, logits),
                )
            return self.routing(logits + predicted_delta_logits)

        if predictive_action != RouterPredictiveAction.COMPUTE_PREDICTIVE_LOSS:
            raise ValueError(f"Unsupported predictive router action: {predictive_action}")

        old_inputs, old_logits, valid_mask = predictive_state.get_predictive_data()
        if predictive_state.has_valid_predictive_data():
            old_inputs = old_inputs.to(
                device=input.device,
                dtype=input.dtype,
                non_blocking=old_inputs.device.type == "cpu",
            )
            old_logits = old_logits.to(
                device=logits.device,
                dtype=logits.dtype,
                non_blocking=old_logits.device.type == "cpu",
            )
            current_logits = logits
            if valid_mask is not None:
                valid_mask = valid_mask.to(device=input.device, non_blocking=valid_mask.device.type == "cpu")
                current_logits = logits[valid_mask]
            old_logits = old_logits.detach()
            current_logits = current_logits.detach()
            with torch.enable_grad():
                predicted_delta_logits = bias_predictor(old_inputs.detach())
                predictive_loss = compute_predictive_loss(
                    old_logits=old_logits,
                    current_logits=current_logits,
                    predicted_delta_logits=predicted_delta_logits,
                    loss_type=self.config.bias_predictor_loss_type,
                )
            PredictiveRouterReplayState.record_predictive_loss(predictive_state.layer_idx, predictive_loss.item())
            PredictiveRouterReplayState.record_predictive_topk_accuracy(
                predictive_state.layer_idx,
                calculate_topk_accuracy(
                    topk=self.topk,
                    logits1=old_logits + predicted_delta_logits.detach(),
                    logits2=current_logits,
                ),
            )
        else:
            with torch.enable_grad():
                predictive_loss = build_synthetic_predictive_loss(bias_predictor=bias_predictor, input_tensor=input)

        probs, routing_map = self.routing(logits)
        predictive_loss.backward()
        predictive_state.clear_predictive_data()
        return probs, routing_map

    TopKRouter.forward = patched_forward
    TopKRouter._predictive_router_replay_patched = True


@contextmanager
def predictive_action_scope(action: RouterPredictiveAction):
    PredictiveRouterReplayState.set_global_predictive_action(action)
    try:
        yield
    finally:
        PredictiveRouterReplayState.clear_global_predictive_action()


class PredictiveRouterReplayBuffer:
    microbatches: ClassVar[list[object]] = []
    train_index: ClassVar[int] = 0

    @classmethod
    def clear(cls) -> None:
        cls.microbatches.clear()
        cls.train_index = 0

    @classmethod
    def append(cls, microbatch_data) -> None:
        cls.microbatches.append(microbatch_data)

    @classmethod
    def reset_train_cursor(cls) -> None:
        cls.train_index = 0

    @classmethod
    def pop_next(cls):
        if cls.train_index >= len(cls.microbatches):
            raise IndexError(
                f"Predictive replay buffer underflow: train_index={cls.train_index}, buffered={len(cls.microbatches)}"
            )
        microbatch_data = cls.microbatches[cls.train_index]
        cls.train_index += 1
        return microbatch_data

    @classmethod
    def buffered_microbatch_count(cls) -> int:
        return len(cls.microbatches)

    @classmethod
    def remaining_microbatch_count(cls) -> int:
        return len(cls.microbatches) - cls.train_index


class PredictiveTrainStepState:
    used_valid_predictive_data: ClassVar[bool] = False

    @classmethod
    def reset(cls) -> None:
        cls.used_valid_predictive_data = False

    @classmethod
    def mark_used(cls) -> None:
        cls.used_valid_predictive_data = True


def _is_predictive_param_group(param_group: dict) -> bool:
    return any(getattr(param, "is_bias_predictor", False) for param in param_group.get("params", []))


def clear_predictive_optimizer_grads(optimizer) -> None:
    for param_group in optimizer.param_groups:
        if not _is_predictive_param_group(param_group):
            continue
        for param in param_group["params"]:
            if not getattr(param, "is_bias_predictor", False):
                continue
            param.grad = None
            if hasattr(param, "main_grad") and param.main_grad is not None:
                param.main_grad.zero_()


def disable_predictive_param_groups(optimizer) -> list[dict[str, object]]:
    saved_group_states = []
    for param_group in optimizer.param_groups:
        if not _is_predictive_param_group(param_group):
            continue
        saved_state = {"group": param_group}
        if "lr" in param_group:
            saved_state["lr"] = param_group["lr"]
            param_group["lr"] = 0.0
        if "weight_decay" in param_group:
            saved_state["weight_decay"] = param_group["weight_decay"]
            param_group["weight_decay"] = 0.0
        saved_group_states.append(saved_state)
    return saved_group_states


def restore_predictive_param_groups(saved_group_states: list[dict[str, object]]) -> None:
    for saved_state in saved_group_states:
        param_group = saved_state["group"]
        if "lr" in saved_state:
            param_group["lr"] = saved_state["lr"]
        if "weight_decay" in saved_state:
            param_group["weight_decay"] = saved_state["weight_decay"]


class PredictiveRouterReplayState:
    router_instances: ClassVar[list["PredictiveRouterReplayState"]] = []
    predictive_loss_tracker: ClassVar[list[tuple[int, float]]] = []
    predictive_bias_ratio_tracker: ClassVar[list[tuple[int, float]]] = []
    predictive_topk_accuracy_tracker: ClassVar[list[tuple[int, float]]] = []

    def __init__(self, layer_idx: int | None = None):
        self.layer_idx = len(self.router_instances) if layer_idx is None else layer_idx
        self.predictive_action = RouterPredictiveAction.DISABLED
        self.recorded_old_inputs: torch.Tensor | None = None
        self.recorded_old_logits: torch.Tensor | None = None
        self.predictive_valid_mask: torch.Tensor | None = None
        self.router_instances.append(self)

    @staticmethod
    def _squeeze_router_dim(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.ndim >= 3 and tensor.shape[1] == 1:
            return tensor.squeeze(1)
        return tensor

    @classmethod
    def register_router(cls, router, attr_name: str = "predictive_router_replay") -> "PredictiveRouterReplayState":
        state = cls()
        setattr(router, attr_name, state)
        return state

    @classmethod
    def reset_registry(cls) -> None:
        cls.router_instances.clear()
        cls.predictive_loss_tracker.clear()
        cls.predictive_bias_ratio_tracker.clear()
        cls.predictive_topk_accuracy_tracker.clear()

    @classmethod
    def get_router_instances(cls) -> list["PredictiveRouterReplayState"]:
        return cls.router_instances

    def has_valid_predictive_data(self) -> bool:
        return (
            self.recorded_old_inputs is not None
            and self.recorded_old_logits is not None
            and self.recorded_old_inputs.shape[0] > 0
            and self.recorded_old_logits.shape[0] > 0
        )

    def record_predictive_data(self, inputs: torch.Tensor, logits: torch.Tensor) -> None:
        self.recorded_old_inputs = self._squeeze_router_dim(inputs).detach().contiguous()
        self.recorded_old_logits = self._squeeze_router_dim(logits).detach().contiguous()
        self.predictive_valid_mask = None

    def set_predictive_data(
        self,
        *,
        inputs: torch.Tensor | None,
        logits: torch.Tensor | None,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        self.recorded_old_inputs = inputs.detach().contiguous() if inputs is not None else None
        self.recorded_old_logits = logits.detach().contiguous() if logits is not None else None
        self.predictive_valid_mask = valid_mask.detach().contiguous() if valid_mask is not None else None

    def get_predictive_data(self) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        return self.recorded_old_inputs, self.recorded_old_logits, self.predictive_valid_mask

    def clear_predictive_data(self) -> None:
        self.recorded_old_inputs = None
        self.recorded_old_logits = None
        self.predictive_valid_mask = None

    def set_predictive_action(self, action: RouterPredictiveAction) -> None:
        self.predictive_action = action

    def clear_predictive_action(self) -> None:
        self.predictive_action = RouterPredictiveAction.DISABLED

    @classmethod
    def set_global_predictive_action(cls, action: RouterPredictiveAction) -> None:
        for router in cls.router_instances:
            router.set_predictive_action(action)

    @classmethod
    def clear_global_predictive_action(cls) -> None:
        for router in cls.router_instances:
            router.clear_predictive_action()

    @classmethod
    def get_global_predictive_action(cls) -> RouterPredictiveAction:
        if not cls.router_instances:
            return RouterPredictiveAction.DISABLED
        return cls.router_instances[0].predictive_action

    @classmethod
    def clear_global_predictive_data(cls) -> None:
        for router in cls.router_instances:
            router.clear_predictive_data()

    @classmethod
    def set_global_predictive_data(
        cls,
        *,
        old_inputs_concat: torch.Tensor | None,
        old_logits_concat: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
    ) -> None:
        if old_inputs_concat is None or old_logits_concat is None:
            cls.clear_global_predictive_data()
            return

        if old_inputs_concat.ndim != 3 or old_logits_concat.ndim != 3:
            raise ValueError("Predictive tensors must have shape [num_tokens, num_layers, hidden_or_experts].")
        if old_inputs_concat.shape[:2] != old_logits_concat.shape[:2]:
            raise ValueError(
                f"Predictive tensor shape mismatch: inputs={old_inputs_concat.shape}, logits={old_logits_concat.shape}"
            )
        if old_inputs_concat.shape[1] != len(cls.router_instances):
            raise ValueError(
                f"Predictive tensor layer count {old_inputs_concat.shape[1]} does not match "
                f"registered routers {len(cls.router_instances)}."
            )

        for layer_idx, router in enumerate(cls.router_instances):
            router.set_predictive_data(
                inputs=old_inputs_concat[:, layer_idx : layer_idx + 1, :],
                logits=old_logits_concat[:, layer_idx : layer_idx + 1, :],
                valid_mask=valid_mask,
            )

    @classmethod
    def record_predictive_loss(cls, layer_idx: int, loss_value: float) -> None:
        cls.predictive_loss_tracker.append((layer_idx, loss_value))

    @classmethod
    def record_predictive_bias_ratio(cls, layer_idx: int, ratio_value: float) -> None:
        cls.predictive_bias_ratio_tracker.append((layer_idx, ratio_value))

    @classmethod
    def record_predictive_topk_accuracy(cls, layer_idx: int, accuracy_value: float) -> None:
        cls.predictive_topk_accuracy_tracker.append((layer_idx, accuracy_value))

    @classmethod
    def get_and_clear_predictive_metrics(cls) -> dict[str, float]:
        metrics = {}
        if cls.predictive_loss_tracker:
            metrics["predictive_loss"] = sum(loss for _, loss in cls.predictive_loss_tracker) / len(
                cls.predictive_loss_tracker
            )
            cls.predictive_loss_tracker.clear()
        if cls.predictive_bias_ratio_tracker:
            metrics["predictive_bias_to_logits_ratio"] = sum(
                ratio for _, ratio in cls.predictive_bias_ratio_tracker
            ) / len(cls.predictive_bias_ratio_tracker)
            cls.predictive_bias_ratio_tracker.clear()
        if cls.predictive_topk_accuracy_tracker:
            metrics["predictive_topk_accuracy"] = sum(
                accuracy for _, accuracy in cls.predictive_topk_accuracy_tracker
            ) / len(cls.predictive_topk_accuracy_tracker)
            cls.predictive_topk_accuracy_tracker.clear()
        return metrics
