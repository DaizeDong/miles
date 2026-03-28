import torch

from miles.utils.replay_base import BaseReplayManager, RouterLogitsCacheAction


def test_router_logits_cache_records_compute_log_prob_and_predictive_bias():
    manager = BaseReplayManager()
    logits = torch.randn(3, 5)
    bias = torch.randn(3, 5)
    token_ids = torch.tensor([2, 4, 6], dtype=torch.long)

    manager.set_cache_action(RouterLogitsCacheAction.COMPUTE_LOG_PROB)
    manager.record_logits(logits, layer_idx=1)
    manager.record_predictive_bias(bias, layer_idx=1)
    manager.record_global_token_ids(token_ids)

    cache = manager.get_and_clear_logits_cache()

    assert len(cache["compute_log_prob"]) == 1
    assert cache["compute_log_prob"][0][0] == 1
    assert torch.equal(cache["compute_log_prob"][0][1], logits)

    assert len(cache["predictive_bias"]) == 1
    assert cache["predictive_bias"][0][0] == 1
    assert cache["predictive_bias"][0][1].shape == (3, 1, 5)
    assert torch.equal(cache["predictive_bias"][0][1].squeeze(1), bias)

    assert len(cache["global_token_ids"]) == 1
    assert torch.equal(cache["global_token_ids"][0], token_ids)

    cleared_cache = manager.get_and_clear_logits_cache()
    assert cleared_cache == {
        "compute_log_prob": [],
        "training": [],
        "router_weights": {},
        "global_token_ids": [],
        "predictive_bias": [],
    }


def test_router_logits_cache_records_training_phase_only():
    manager = BaseReplayManager()
    logits = torch.randn(2, 7)

    manager.set_cache_action(RouterLogitsCacheAction.TRAINING)
    manager.record_logits(logits, layer_idx=3)
    manager.record_predictive_bias(torch.randn(2, 7), layer_idx=3)
    manager.clear_cache_action()

    cache = manager.get_and_clear_logits_cache()

    assert cache["compute_log_prob"] == []
    assert len(cache["training"]) == 1
    assert cache["training"][0][0] == 3
    assert torch.equal(cache["training"][0][1], logits)
    assert cache["predictive_bias"] == []


def test_router_logits_cache_can_generate_unique_token_ids_from_logits_shape():
    manager = BaseReplayManager()

    manager.set_cache_action(RouterLogitsCacheAction.COMPUTE_LOG_PROB)
    manager.record_logits(torch.randn(3, 5), layer_idx=0)
    manager.record_global_token_ids()
    manager.record_logits(torch.randn(2, 5), layer_idx=1)
    manager.record_global_token_ids()

    cache = manager.get_and_clear_logits_cache()

    assert len(cache["global_token_ids"]) == 2
    assert torch.equal(cache["global_token_ids"][0], torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(cache["global_token_ids"][1], torch.tensor([3, 4], dtype=torch.long))


def test_router_logits_cache_auto_token_ids_support_dp_offsets():
    ids_rank0, next_base = BaseReplayManager._compute_auto_global_token_ids(
        num_tokens=3,
        base_offset=0,
        rank_counts=[3, 2, 4],
        dp_rank=0,
    )
    ids_rank1, same_next_base = BaseReplayManager._compute_auto_global_token_ids(
        num_tokens=2,
        base_offset=0,
        rank_counts=[3, 2, 4],
        dp_rank=1,
    )
    ids_rank2, final_base = BaseReplayManager._compute_auto_global_token_ids(
        num_tokens=4,
        base_offset=0,
        rank_counts=[3, 2, 4],
        dp_rank=2,
    )

    assert torch.equal(ids_rank0, torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(ids_rank1, torch.tensor([3, 4], dtype=torch.long))
    assert torch.equal(ids_rank2, torch.tensor([5, 6, 7, 8], dtype=torch.long))
    assert next_base == 9
    assert same_next_base == 9
    assert final_base == 9
