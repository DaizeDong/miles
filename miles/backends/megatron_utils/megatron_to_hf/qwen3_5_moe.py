import re

import torch

from miles.utils.hf_config_utils import load_hf_config

_TEXT_CONFIG_CACHE = {}
_EXPERT_CACHE = {}


def reset_qwen3_5_moe_export_cache():
    _TEXT_CONFIG_CACHE.clear()
    _EXPERT_CACHE.clear()


def _get_text_config(args):
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if hf_checkpoint is None:
        raise ValueError("Qwen3.5 export requires args.hf_checkpoint so the linear-attention dimensions can be read.")
    if hf_checkpoint not in _TEXT_CONFIG_CACHE:
        hf_config = load_hf_config(hf_checkpoint, trust_remote_code=True)
        _TEXT_CONFIG_CACHE[hf_checkpoint] = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
    return _TEXT_CONFIG_CACHE[hf_checkpoint]


def _stack_expert_tensor(cache_key, expert_idx, tensor, num_experts, hf_name):
    cache = _EXPERT_CACHE.setdefault(cache_key, {})
    cache[expert_idx] = tensor
    if len(cache) < num_experts:
        return []

    stacked = torch.stack([cache[idx] for idx in range(num_experts)], dim=0)
    del _EXPERT_CACHE[cache_key]
    return [(hf_name, stacked)]


def convert_qwen3_5_moe_to_hf(args, name, param):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.language_model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]

    text_config = _get_text_config(args)

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if not match:
        raise ValueError(f"Unknown parameter name: {name}")

    layer_idx, rest = match.groups()

    expert_pattern = r"mlp\.experts\.(linear_fc[12])\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, expert_idx = match.groups()
        expert_idx = int(expert_idx)
        if rest == "linear_fc1":
            return _stack_expert_tensor(
                cache_key=(layer_idx, rest),
                expert_idx=expert_idx,
                tensor=param,
                num_experts=args.num_experts,
                hf_name=f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj",
            )
        return _stack_expert_tensor(
            cache_key=(layer_idx, rest),
            expert_idx=expert_idx,
            tensor=param,
            num_experts=args.num_experts,
            hf_name=f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj",
        )

    shared_expert_pattern = r"mlp\.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest = match.group(1)
        if rest == "linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.language_model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight", gate_weight),
                (f"model.language_model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight", up_weight),
            ]
        if rest == "linear_fc2.weight":
            return [(f"model.language_model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight", param)]
        if rest == "gate_weight":
            return [(f"model.language_model.layers.{layer_idx}.mlp.shared_expert_gate.weight", param)]
        raise ValueError(f"Unknown shared expert parameter name: {name}")

    if rest == "self_attention.linear_proj.weight":
        return [(f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
    if rest == "self_attention.linear_qkv.weight":
        param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
        q_param, k_param, v_param = torch.split(
            param,
            split_size_or_sections=[2 * value_num_per_group, 1, 1],
            dim=1,
        )
        q_param = (
            q_param.reshape(args.num_query_groups, 2, value_num_per_group, head_dim, args.hidden_size)
            .transpose(1, 2)
            .reshape(-1, args.hidden_size)
        )
        k_param = k_param.reshape(-1, args.hidden_size)
        v_param = v_param.reshape(-1, args.hidden_size)
        return [
            (f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
            (f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
            (f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
        ]
    if rest == "self_attention.linear_qkv.bias":
        param = param.view(args.num_query_groups, -1)
        q_bias, k_bias, v_bias = torch.split(
            param,
            split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
            dim=1,
        )
        return [
            (f"model.language_model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias.contiguous().flatten()),
            (f"model.language_model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias.contiguous().flatten()),
            (f"model.language_model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias.contiguous().flatten()),
        ]
    if rest == "self_attention.linear_qkv.layer_norm_weight":
        return [(f"model.language_model.layers.{layer_idx}.input_layernorm.weight", param)]
    if rest == "self_attention.q_layernorm.weight":
        return [(f"model.language_model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
    if rest == "self_attention.k_layernorm.weight":
        return [(f"model.language_model.layers.{layer_idx}.self_attn.k_norm.weight", param)]
    if rest == "self_attention.linear_attn.in_proj_qkvz.weight":
        key_dim = text_config.linear_num_key_heads * text_config.linear_key_head_dim
        value_dim = text_config.linear_num_value_heads * text_config.linear_value_head_dim
        in_proj_qkv, in_proj_z = param.split([2 * key_dim + value_dim, value_dim], dim=0)
        return [
            (f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight", in_proj_qkv.contiguous()),
            (f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_z.weight", in_proj_z.contiguous()),
        ]
    if rest == "self_attention.linear_attn.in_proj_ba.weight":
        in_proj_b, in_proj_a = param.split(
            [text_config.linear_num_value_heads, text_config.linear_num_value_heads],
            dim=0,
        )
        return [
            (f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_b.weight", in_proj_b.contiguous()),
            (f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_a.weight", in_proj_a.contiguous()),
        ]
    if rest == "mlp.router.weight":
        return [(f"model.language_model.layers.{layer_idx}.mlp.gate.weight", param)]
    if rest == "mlp.router.bias_predictor.weight":
        return [(f"model.language_model.layers.{layer_idx}.mlp.bias_predictor.weight", param)]
    if rest == "pre_mlp_layernorm.weight":
        return [(f"model.language_model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
    if rest.startswith("self_attention.") and rest[len("self_attention.") :] in [
        "input_layernorm.weight",
        "linear_attn.A_log",
        "linear_attn.conv1d.weight",
        "linear_attn.dt_bias",
        "linear_attn.norm.weight",
        "linear_attn.out_proj.weight",
    ]:
        rest = rest[len("self_attention.") :]
        return [(f"model.language_model.layers.{layer_idx}.{rest}", param)]

    raise ValueError(f"Unknown parameter name: {name}")
