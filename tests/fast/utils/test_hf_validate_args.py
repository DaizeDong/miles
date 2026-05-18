from types import SimpleNamespace

import pytest

from miles.utils.arguments import hf_validate_args


def _make_args(**overrides):
    defaults = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_layers=40,
        ffn_hidden_size=512,
        moe_ffn_hidden_size=512,
        moe_shared_expert_intermediate_size=512,
        untie_embeddings_and_output_weights=True,
        norm_epsilon=1e-6,
        rotary_base=10000000,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_hf_validate_args_uses_moe_intermediate_size_for_moe_models():
    args = _make_args()
    hf_config = SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=40,
        intermediate_size=5632,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts=256,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
        rope_theta=10000000,
    )

    hf_validate_args(args, hf_config)


def test_hf_validate_args_checks_shared_expert_size_for_moe_models():
    args = _make_args(moe_shared_expert_intermediate_size=256)
    hf_config = SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=40,
        intermediate_size=5632,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts=256,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
        rope_theta=10000000,
    )

    with pytest.raises(AssertionError, match="shared_expert_intermediate_size"):
        hf_validate_args(args, hf_config)
