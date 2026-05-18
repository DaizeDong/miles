import inspect
from collections import OrderedDict

import torch
from mbridge.core import register_model
from mbridge.core.safetensor_io import SafeTensorIO
from mbridge.models import Qwen2MoEBridge

from miles.utils.hf_config_utils import maybe_register_sglang_hf_configs

maybe_register_sglang_hf_configs()


def _get_text_config(hf_config):
    return hf_config.text_config if hasattr(hf_config, "text_config") else hf_config


class _Qwen3_5ExpertSafeTensorIO(SafeTensorIO):
    _EXPERT_KEY_SEP = "#expert"

    def __init__(self, hf_dir: str, max_cached_weights: int = 2):
        super().__init__(hf_dir)
        self.max_cached_weights = max_cached_weights
        self._expert_tensor_cache = OrderedDict()

    @classmethod
    def make_expert_weight_name(cls, base_name: str, expert_id: int) -> str:
        return f"{base_name}{cls._EXPERT_KEY_SEP}{expert_id}"

    @classmethod
    def parse_expert_weight_name(cls, weight_name: str):
        if cls._EXPERT_KEY_SEP not in weight_name:
            return None
        base_name, expert_id = weight_name.rsplit(cls._EXPERT_KEY_SEP, 1)
        if not expert_id.isdigit():
            return None
        return base_name, int(expert_id)

    def _get_cached_expert_tensor(self, base_name: str) -> torch.Tensor:
        tensor = self._expert_tensor_cache.pop(base_name, None)
        if tensor is None:
            tensor = super().load_one_hf_weight(base_name)
        self._expert_tensor_cache[base_name] = tensor
        while len(self._expert_tensor_cache) > self.max_cached_weights:
            self._expert_tensor_cache.popitem(last=False)
        return tensor

    def load_some_hf_weight(self, hf_weight_names: list[str]) -> dict:
        real_weight_names = []
        expert_weight_names = []
        for name in hf_weight_names:
            if self.parse_expert_weight_name(name) is None:
                real_weight_names.append(name)
            else:
                expert_weight_names.append(name)

        ret = super().load_some_hf_weight(real_weight_names) if real_weight_names else {}
        for name in expert_weight_names:
            base_name, expert_id = self.parse_expert_weight_name(name)
            ret[name] = self._get_cached_expert_tensor(base_name)[expert_id].contiguous()
        return ret


@register_model("qwen3_5moe")
@register_model("qwen3_5_moe")
class Qwen3_5MoeBridge(Qwen2MoEBridge):
    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }
    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": [
            "model.language_model.layers.{layer_number}.self_attn.o_proj.weight",
        ],
        "self_attention.linear_qkv.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.input_layernorm.weight",
        ],
        "self_attention.q_layernorm.weight": [
            "model.language_model.layers.{layer_number}.self_attn.q_norm.weight",
        ],
        "self_attention.k_layernorm.weight": [
            "model.language_model.layers.{layer_number}.self_attn.k_norm.weight",
        ],
        "self_attention.linear_qkv.weight": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
        "self_attention.input_layernorm.weight": [
            "model.language_model.layers.{layer_number}.input_layernorm.weight",
        ],
        "self_attention.linear_attn.A_log": [
            "model.language_model.layers.{layer_number}.linear_attn.A_log",
        ],
        "self_attention.linear_attn.conv1d.weight": [
            "model.language_model.layers.{layer_number}.linear_attn.conv1d.weight",
        ],
        "self_attention.linear_attn.dt_bias": [
            "model.language_model.layers.{layer_number}.linear_attn.dt_bias",
        ],
        "self_attention.linear_attn.in_proj_qkvz.weight": [
            "model.language_model.layers.{layer_number}.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.{layer_number}.linear_attn.in_proj_z.weight",
        ],
        "self_attention.linear_attn.in_proj_ba.weight": [
            "model.language_model.layers.{layer_number}.linear_attn.in_proj_b.weight",
            "model.language_model.layers.{layer_number}.linear_attn.in_proj_a.weight",
        ],
        "self_attention.linear_attn.norm.weight": [
            "model.language_model.layers.{layer_number}.linear_attn.norm.weight",
        ],
        "self_attention.linear_attn.out_proj.weight": [
            "model.language_model.layers.{layer_number}.linear_attn.out_proj.weight",
        ],
    }

    @property
    def text_config(self):
        return _get_text_config(self.hf_config)

    def _get_safetensor_io(self, weights_path: str):
        return _Qwen3_5ExpertSafeTensorIO(self._get_actual_hf_path(weights_path))

    def _get_aggregated_expert_hf_name(self, mcore_weights_name: str) -> str:
        layer_number = mcore_weights_name.split(".")[2]
        if ".mlp.experts.linear_fc1" in mcore_weights_name:
            return f"model.language_model.layers.{layer_number}.mlp.experts.gate_up_proj"
        if ".mlp.experts.linear_fc2" in mcore_weights_name:
            return f"model.language_model.layers.{layer_number}.mlp.experts.down_proj"
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _build_config(self):
        text_config = self.text_config
        kwargs = {
            "use_cpu_initialization": False,
            "moe_ffn_hidden_size": text_config.moe_intermediate_size,
            "moe_router_bias_update_rate": 0.001,
            "moe_router_topk": text_config.num_experts_per_tok,
            "num_moe_experts": text_config.num_experts,
            "moe_aux_loss_coeff": text_config.router_aux_loss_coef,
            "moe_grouped_gemm": True,
            "moe_router_score_function": "softmax",
            "moe_router_load_balancing_type": "none",
            "moe_shared_expert_intermediate_size": text_config.shared_expert_intermediate_size,
            "moe_shared_expert_gate": bool(text_config.shared_expert_intermediate_size),
            "persist_layer_norm": True,
            "bias_activation_fusion": True,
            "bias_dropout_fusion": True,
            "moe_router_pre_softmax": False,
            "qk_layernorm": True,
            "attention_output_gate": True,
            "layernorm_zero_centered_gamma": True,
        }
        if "text_config_key" in inspect.signature(self._build_base_config).parameters:
            kwargs["text_config_key"] = "text_config"
            return self._build_base_config(**kwargs)

        original_hf_config = self.hf_config
        try:
            self.hf_config = text_config
            return self._build_base_config(**kwargs)
        finally:
            self.hf_config = original_hf_config

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]
        if ".pre_mlp_layernorm.weight" in name:
            return [f"model.language_model.layers.{layer_number}.post_attention_layernorm.weight"]
        if ".mlp.router.bias_predictor.weight" in name:
            return [f"model.language_model.layers.{layer_number}.mlp.bias_predictor.weight"]
        if ".mlp.router.weight" in name:
            return [f"model.language_model.layers.{layer_number}.mlp.gate.weight"]
        if ".mlp.shared_experts.linear_fc1.weight" in name:
            return [
                f"model.language_model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight",
                f"model.language_model.layers.{layer_number}.mlp.shared_expert.up_proj.weight",
            ]
        if ".mlp.shared_experts.linear_fc2.weight" in name:
            return [f"model.language_model.layers.{layer_number}.mlp.shared_expert.down_proj.weight"]
        if ".mlp.shared_experts.gate_weight" in name:
            return [f"model.language_model.layers.{layer_number}.mlp.shared_expert_gate.weight"]
        if ".mlp.experts.linear_fc1" in name:
            expert_id = int(name.split("weight")[-1])
            return [
                _Qwen3_5ExpertSafeTensorIO.make_expert_weight_name(
                    f"model.language_model.layers.{layer_number}.mlp.experts.gate_up_proj",
                    expert_id,
                )
            ]
        if ".mlp.experts.linear_fc2" in name:
            expert_id = int(name.split("weight")[-1])
            return [
                _Qwen3_5ExpertSafeTensorIO.make_expert_weight_name(
                    f"model.language_model.layers.{layer_number}.mlp.experts.down_proj",
                    expert_id,
                )
            ]
        raise NotImplementedError(f"Unsupported parameter name: {name}")

    def _weight_to_hf_format(self, mcore_weights_name: str, mcore_weights: torch.Tensor):
        if ".mlp.experts.linear_fc" in mcore_weights_name:
            expert_id = int(mcore_weights_name.split("weight")[-1])
            cache_key = self._get_aggregated_expert_hf_name(mcore_weights_name)
            num_experts = self.text_config.num_experts
            if not hasattr(self, "_expert_export_cache"):
                self._expert_export_cache = {}
            expert_cache = self._expert_export_cache.setdefault(cache_key, {})
            expert_cache[expert_id] = mcore_weights
            if len(expert_cache) < num_experts:
                return [], []

            stacked = torch.stack([expert_cache[idx] for idx in range(num_experts)], dim=0)
            del self._expert_export_cache[cache_key]
            return [cache_key], [stacked]

        hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        text_config = self.text_config
        attention_output_gate = getattr(self.config, "attention_output_gate", False)

        if "self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            assert len(hf_names) == 3
            num_key_value_heads = text_config.num_key_value_heads
            hidden_dim = text_config.hidden_size
            num_attention_heads = text_config.num_attention_heads
            head_dim = getattr(text_config, "head_dim", hidden_dim // num_attention_heads)
            out_shape = (
                [num_key_value_heads, -1, hidden_dim] if ".bias" not in mcore_weights_name else [num_key_value_heads, -1]
            )
            qkv = mcore_weights.view(*out_shape)
            q_len = head_dim * num_attention_heads // num_key_value_heads
            k_len = head_dim
            single_out_shape = [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]

            q = qkv[:, :q_len].reshape(*single_out_shape)
            gate = None
            if attention_output_gate:
                gate = qkv[:, q_len : q_len + q_len].reshape(*single_out_shape)
                q_len += q_len
            k = qkv[:, q_len : q_len + k_len].reshape(*single_out_shape)
            v = qkv[:, q_len + k_len :].reshape(*single_out_shape)

            if attention_output_gate:
                if ".bias" in mcore_weights_name:
                    q = q.view(num_attention_heads, -1)
                    gate = gate.view(num_attention_heads, -1)
                else:
                    q = q.view(num_attention_heads, -1, hidden_dim)
                    gate = gate.view(num_attention_heads, -1, hidden_dim)
                q = torch.cat([q, gate], dim=1).reshape(*single_out_shape).contiguous()
            return hf_names, [q, k, v]

        if "self_attention.linear_attn.in_proj_qkvz.weight" in mcore_weights_name:
            assert len(hf_names) == 2
            key_dim = text_config.linear_num_key_heads * text_config.linear_key_head_dim
            value_dim = text_config.linear_num_value_heads * text_config.linear_value_head_dim
            qkv, z = mcore_weights.split([2 * key_dim + value_dim, value_dim], dim=0)
            return hf_names, [qkv.contiguous(), z.contiguous()]

        if "self_attention.linear_attn.in_proj_ba.weight" in mcore_weights_name:
            assert len(hf_names) == 2
            b, a = mcore_weights.split([text_config.linear_num_value_heads, text_config.linear_num_value_heads], dim=0)
            return hf_names, [b.contiguous(), a.contiguous()]

        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        text_config = self.text_config
        attention_output_gate = getattr(self.config, "attention_output_gate", False)

        if ".mlp.experts.linear_fc" in mcore_weights_name:
            assert len(hf_weights) == 1
            return hf_weights[0]

        if "self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            assert len(hf_weights) == 3
            num_key_value_heads = text_config.num_key_value_heads
            hidden_dim = text_config.hidden_size
            num_attention_heads = text_config.num_attention_heads
            num_queries_per_group = num_attention_heads // num_key_value_heads
            head_dim = getattr(text_config, "head_dim", hidden_dim // num_attention_heads)
            group_dim = head_dim * num_attention_heads // num_key_value_heads
            q, k, v = hf_weights

            if attention_output_gate:
                real_num_key_value_heads = q.shape[0] // (2 * group_dim)
                q = (
                    q.view(real_num_key_value_heads, num_queries_per_group, 2, head_dim, -1)
                    .transpose(1, 2)
                    .flatten(1, 3)
                )
            else:
                real_num_key_value_heads = q.shape[0] // group_dim
                q = q.view(real_num_key_value_heads, group_dim, -1)

            k = k.view(real_num_key_value_heads, head_dim, -1)
            v = v.view(real_num_key_value_heads, head_dim, -1)
            out_shape = [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]
            return torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()

        if "self_attention.linear_attn.in_proj_qkvz.weight" in mcore_weights_name:
            assert len(hf_weights) == 2
            in_proj_qkv, in_proj_z = hf_weights
            return torch.cat([in_proj_qkv, in_proj_z], dim=0).contiguous()

        if "self_attention.linear_attn.in_proj_ba.weight" in mcore_weights_name:
            assert len(hf_weights) == 2
            in_proj_b, in_proj_a = hf_weights
            return torch.cat([in_proj_b, in_proj_a], dim=0).contiguous()

        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)
