import logging
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
            old_inputs = old_inputs.to(device=input.device, dtype=input.dtype)
            old_logits = old_logits.to(device=logits.device, dtype=logits.dtype)
            current_logits = logits
            if valid_mask is not None:
                valid_mask = valid_mask.to(device=input.device)
                current_logits = logits[valid_mask]
            predicted_delta_logits = bias_predictor(old_inputs)
            predictive_loss = compute_predictive_loss(
                old_logits=old_logits,
                current_logits=current_logits,
                predicted_delta_logits=predicted_delta_logits,
                loss_type=self.config.bias_predictor_loss_type,
            )
            PredictiveRouterReplayState.record_predictive_loss(predictive_state.layer_idx, predictive_loss.item())
            PredictiveRouterReplayState.record_predictive_topk_accuracy(
                predictive_state.layer_idx,
                calculate_topk_accuracy(topk=self.topk, logits1=old_logits + predicted_delta_logits, logits2=current_logits),
            )
        else:
            predictive_loss = build_synthetic_predictive_loss(bias_predictor=bias_predictor, input_tensor=input)

        probs, routing_map = self.routing(logits)
        predictive_loss.backward()
        predictive_state.clear_predictive_data()
        return probs, routing_map

    TopKRouter.forward = patched_forward
    TopKRouter._predictive_router_replay_patched = True


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
