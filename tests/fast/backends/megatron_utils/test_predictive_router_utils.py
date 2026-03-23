import torch

from miles.backends.megatron_utils.predictive_router_replay import (
    PredictiveRouterReplayState,
    RouterPredictiveAction,
    calculate_topk_accuracy,
    compute_predictive_bias_ratio,
    compute_predictive_loss,
)
from miles.backends.megatron_utils.predictive_router_utils import (
    build_predictive_valid_mask,
    prepare_predictive_router_data,
    restore_predictive_samples,
    select_predictive_samples,
)


def test_build_predictive_valid_mask_prefix_lengths():
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    valid_mask, selected_lens = build_predictive_valid_mask(
        attention_mask=attention_mask,
        valid_indices=[0, 1],
        old_lengths=[2, 3],
        old_token_positions_list=None,
    )

    assert selected_lens == [2, 3]
    assert valid_mask.tolist() == [True, True, False, True, True, True, False]


def test_build_predictive_valid_mask_uses_explicit_positions():
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    old_token_positions = [
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([1, 2], dtype=torch.int32),
    ]

    valid_mask, selected_lens = build_predictive_valid_mask(
        attention_mask=attention_mask,
        valid_indices=[0, 1],
        old_lengths=[2, 2],
        old_token_positions_list=old_token_positions,
    )

    assert selected_lens == [2, 2]
    assert valid_mask.tolist() == [True, False, True, False, False, True, True, False]


def test_select_predictive_samples_downsamples_and_restores():
    generator = torch.Generator().manual_seed(1234)
    old_inputs_list = [
        torch.randn(4, 2, 3),
        torch.randn(2, 2, 3),
        torch.randn(3, 2, 3),
    ]
    old_logits_list = [
        torch.randn(4, 2, 5),
        torch.randn(2, 2, 5),
        torch.randn(3, 2, 5),
    ]

    selection = select_predictive_samples(
        old_inputs_list=old_inputs_list,
        old_logits_list=old_logits_list,
        downsample_batch_size=2,
        max_len_limit=3,
        storage_dtype="fp16",
        generator=generator,
    )

    assert selection.sampled_indices == [1, 2]
    assert selection.sampled_mask.tolist() == [False, True, True]
    assert all(tensor.dtype == torch.float16 for tensor in selection.old_inputs)
    assert all(tensor.dtype == torch.float16 for tensor in selection.old_logits)

    restored_inputs = restore_predictive_samples(selection.old_inputs, selection.sampled_mask)
    restored_logits = restore_predictive_samples(selection.old_logits, selection.sampled_mask)
    assert restored_inputs[0] is None
    assert restored_logits[0] is None
    assert restored_inputs[1].shape == torch.Size([2, 2, 3])
    assert restored_logits[2].shape == torch.Size([3, 2, 5])


def test_select_predictive_samples_falls_back_to_shortest_sequences():
    old_inputs_list = [
        torch.randn(6, 2, 3),
        torch.randn(5, 2, 3),
        torch.randn(2, 2, 3),
    ]
    old_logits_list = [
        torch.randn(6, 2, 5),
        torch.randn(5, 2, 5),
        torch.randn(2, 2, 5),
    ]

    selection = select_predictive_samples(
        old_inputs_list=old_inputs_list,
        old_logits_list=old_logits_list,
        downsample_batch_size=2,
        max_len_limit=3,
    )

    assert selection.sampled_indices == [1, 2]
    assert selection.sampled_mask.tolist() == [False, True, True]


def test_prepare_predictive_router_data_trims_to_current_lengths():
    attention_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.bool,
    )
    old_inputs_list = [
        torch.arange(24, dtype=torch.float32).reshape(4, 2, 3),
        torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
    ]
    old_logits_list = [
        torch.arange(40, dtype=torch.float32).reshape(4, 2, 5),
        torch.arange(20, dtype=torch.float32).reshape(2, 2, 5),
    ]

    prepared = prepare_predictive_router_data(
        old_inputs_list=old_inputs_list,
        old_logits_list=old_logits_list,
        attention_mask=attention_mask,
    )

    assert prepared.has_valid_samples is True
    assert prepared.valid_indices == [0, 1]
    assert prepared.selected_current_lens == [2, 2]
    assert prepared.valid_mask.tolist() == [True, True, True, True, False]
    assert prepared.old_inputs_concat.shape == torch.Size([4, 2, 3])
    assert prepared.old_logits_concat.shape == torch.Size([4, 2, 5])
    assert torch.equal(prepared.old_inputs_concat[:2], old_inputs_list[0][:2])
    assert torch.equal(prepared.old_inputs_concat[2:], old_inputs_list[1])


def test_prepare_predictive_router_data_uses_explicit_positions():
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    old_inputs_list = [
        torch.randn(2, 2, 3),
        torch.randn(2, 2, 3),
    ]
    old_logits_list = [
        torch.randn(2, 2, 5),
        torch.randn(2, 2, 5),
    ]
    old_token_positions_list = [
        torch.tensor([0, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    ]

    prepared = prepare_predictive_router_data(
        old_inputs_list=old_inputs_list,
        old_logits_list=old_logits_list,
        attention_mask=attention_mask,
        old_token_positions_list=old_token_positions_list,
    )

    assert prepared.valid_indices == [0, 1]
    assert prepared.selected_current_lens == [2, 2]
    assert prepared.valid_mask.tolist() == [True, False, True, True, True, False]


def test_predictive_router_replay_registry_and_metrics():
    PredictiveRouterReplayState.reset_registry()
    state0 = PredictiveRouterReplayState()
    state1 = PredictiveRouterReplayState()

    PredictiveRouterReplayState.set_global_predictive_action(RouterPredictiveAction.COMPUTE_PREDICTIVE_LOSS)
    assert state0.predictive_action == RouterPredictiveAction.COMPUTE_PREDICTIVE_LOSS
    assert state1.predictive_action == RouterPredictiveAction.COMPUTE_PREDICTIVE_LOSS

    old_inputs_concat = torch.randn(5, 2, 3)
    old_logits_concat = torch.randn(5, 2, 4)
    valid_mask = torch.tensor([True, False, True, True, False], dtype=torch.bool)
    PredictiveRouterReplayState.set_global_predictive_data(
        old_inputs_concat=old_inputs_concat,
        old_logits_concat=old_logits_concat,
        valid_mask=valid_mask,
    )

    state0_inputs, state0_logits, state0_mask = state0.get_predictive_data()
    state1_inputs, state1_logits, state1_mask = state1.get_predictive_data()
    assert state0_inputs.shape == torch.Size([5, 1, 3])
    assert state0_logits.shape == torch.Size([5, 1, 4])
    assert state1_inputs.shape == torch.Size([5, 1, 3])
    assert state1_logits.shape == torch.Size([5, 1, 4])
    assert torch.equal(state0_mask, valid_mask)
    assert torch.equal(state1_mask, valid_mask)

    PredictiveRouterReplayState.record_predictive_loss(0, 1.0)
    PredictiveRouterReplayState.record_predictive_loss(1, 3.0)
    PredictiveRouterReplayState.record_predictive_bias_ratio(0, 2.0)
    PredictiveRouterReplayState.record_predictive_topk_accuracy(0, 0.25)
    metrics = PredictiveRouterReplayState.get_and_clear_predictive_metrics()
    assert metrics == {
        "predictive_loss": 2.0,
        "predictive_bias_to_logits_ratio": 2.0,
        "predictive_topk_accuracy": 0.25,
    }
    assert PredictiveRouterReplayState.get_and_clear_predictive_metrics() == {}

    PredictiveRouterReplayState.clear_global_predictive_data()
    assert state0.get_predictive_data() == (None, None, None)
    assert state1.get_predictive_data() == (None, None, None)
    PredictiveRouterReplayState.clear_global_predictive_action()
    assert state0.predictive_action == RouterPredictiveAction.DISABLED
    assert state1.predictive_action == RouterPredictiveAction.DISABLED


def test_compute_predictive_loss_variants_and_metrics():
    old_logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    current_logits = torch.tensor([[0.5, 1.5], [1.5, 0.5]], dtype=torch.float32)
    predicted_delta_logits = torch.tensor([[0.25, 0.75], [0.75, 0.25]], dtype=torch.float32)

    l2_loss = compute_predictive_loss(
        old_logits=old_logits,
        current_logits=current_logits,
        predicted_delta_logits=predicted_delta_logits,
        loss_type="l2",
    )
    kl_loss = compute_predictive_loss(
        old_logits=old_logits,
        current_logits=current_logits,
        predicted_delta_logits=predicted_delta_logits,
        loss_type="kl",
    )
    kl_post_loss = compute_predictive_loss(
        old_logits=old_logits,
        current_logits=current_logits,
        predicted_delta_logits=predicted_delta_logits,
        loss_type="kl-post",
    )

    assert l2_loss.item() >= 0
    assert kl_loss.item() >= 0
    assert kl_post_loss.item() >= 0
    assert compute_predictive_bias_ratio(predicted_delta_logits, old_logits) > 0
    assert calculate_topk_accuracy(topk=1, logits1=old_logits + predicted_delta_logits, logits2=current_logits) == 1.0
