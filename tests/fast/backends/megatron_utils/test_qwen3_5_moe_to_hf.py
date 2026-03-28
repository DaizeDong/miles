from types import SimpleNamespace

import torch

from miles.backends.megatron_utils.megatron_to_hf.qwen3_5_moe import (
    convert_qwen3_5_moe_to_hf,
    reset_qwen3_5_moe_export_cache,
)


def _make_args(**kwargs):
    defaults = dict(
        hf_checkpoint="/tmp/qwen3.5",
        hidden_size=8,
        num_attention_heads=2,
        num_query_groups=1,
        kv_channels=4,
        num_experts=2,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_convert_qwen3_5_moe_full_attention_qkv_weight(monkeypatch):
    text_config = SimpleNamespace(
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=4,
        linear_value_head_dim=2,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.megatron_to_hf.qwen3_5_moe._get_text_config",
        lambda args: text_config,
    )

    args = _make_args()
    param = torch.arange(24 * 8, dtype=torch.float32).reshape(24, 8)

    outputs = convert_qwen3_5_moe_to_hf(
        args,
        "module.module.decoder.layers.3.self_attention.linear_qkv.weight",
        param,
    )

    assert [name for name, _ in outputs] == [
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.3.self_attn.k_proj.weight",
        "model.language_model.layers.3.self_attn.v_proj.weight",
    ]
    assert outputs[0][1].shape == (16, 8)
    assert outputs[1][1].shape == (4, 8)
    assert outputs[2][1].shape == (4, 8)


def test_convert_qwen3_5_moe_linear_attention_fused_weights(monkeypatch):
    text_config = SimpleNamespace(
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=4,
        linear_value_head_dim=2,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.megatron_to_hf.qwen3_5_moe._get_text_config",
        lambda args: text_config,
    )

    args = _make_args()
    qkvz = torch.arange(24 * 8, dtype=torch.float32).reshape(24, 8)
    ba = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)

    qkvz_outputs = convert_qwen3_5_moe_to_hf(
        args,
        "module.module.decoder.layers.1.self_attention.linear_attn.in_proj_qkvz.weight",
        qkvz,
    )
    ba_outputs = convert_qwen3_5_moe_to_hf(
        args,
        "module.module.decoder.layers.1.self_attention.linear_attn.in_proj_ba.weight",
        ba,
    )

    assert [name for name, _ in qkvz_outputs] == [
        "model.language_model.layers.1.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.1.linear_attn.in_proj_z.weight",
    ]
    assert qkvz_outputs[0][1].shape == (16, 8)
    assert qkvz_outputs[1][1].shape == (8, 8)
    assert [name for name, _ in ba_outputs] == [
        "model.language_model.layers.1.linear_attn.in_proj_b.weight",
        "model.language_model.layers.1.linear_attn.in_proj_a.weight",
    ]
    assert ba_outputs[0][1].shape == (4, 8)
    assert ba_outputs[1][1].shape == (4, 8)


def test_convert_qwen3_5_moe_experts_are_stacked(monkeypatch):
    reset_qwen3_5_moe_export_cache()
    text_config = SimpleNamespace(
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=4,
        linear_value_head_dim=2,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.megatron_to_hf.qwen3_5_moe._get_text_config",
        lambda args: text_config,
    )

    args = _make_args(num_experts=2)
    expert0 = torch.full((6, 8), 1.0)
    expert1 = torch.full((6, 8), 2.0)

    first = convert_qwen3_5_moe_to_hf(
        args,
        "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight0",
        expert0,
    )
    second = convert_qwen3_5_moe_to_hf(
        args,
        "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight1",
        expert1,
    )

    assert first == []
    assert second[0][0] == "model.language_model.layers.0.mlp.experts.gate_up_proj"
    assert torch.equal(second[0][1], torch.stack([expert0, expert1], dim=0))
