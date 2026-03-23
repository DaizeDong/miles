from miles.backends.megatron_utils.predictive_train_schedule import (
    get_effective_train_iters,
    get_predictive_actor_train_pass_count,
    get_rollout_train_step_id,
)


def test_predictive_actor_train_pass_count_matches_two_phase_schedule():
    assert get_predictive_actor_train_pass_count(role="actor", predictive_enabled=False) == 1
    assert get_predictive_actor_train_pass_count(role="critic", predictive_enabled=True) == 1
    assert get_predictive_actor_train_pass_count(role="actor", predictive_enabled=True) == 2


def test_effective_train_iters_expand_for_predictive_actor():
    assert get_effective_train_iters(base_train_iters=16, role="actor", predictive_enabled=False) == 16
    assert get_effective_train_iters(base_train_iters=16, role="critic", predictive_enabled=True) == 16
    assert get_effective_train_iters(base_train_iters=16, role="actor", predictive_enabled=True) == 32


def test_rollout_train_step_id_accounts_for_train_pass_index():
    assert get_rollout_train_step_id(
        rollout_id=3,
        step_id=1,
        num_steps_per_rollout=4,
        train_pass_index=0,
        num_train_passes=2,
    ) == 25
    assert get_rollout_train_step_id(
        rollout_id=3,
        step_id=1,
        num_steps_per_rollout=4,
        train_pass_index=1,
        num_train_passes=2,
    ) == 29
