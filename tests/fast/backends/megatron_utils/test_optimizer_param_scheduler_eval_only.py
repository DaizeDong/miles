"""Regression coverage for zero-update eval-only optimizer scheduling."""

from argparse import Namespace
from unittest.mock import MagicMock, patch


_MODEL_MODULE = "miles.backends.megatron_utils.model"


def _args(**overrides):
    values = {
        "num_rollout": 2,
        "rollout_batch_size": 32,
        "n_samples_per_prompt": 16,
        "global_batch_size": 128,
        "lr_decay_iters": None,
        "lr_wsd_decay_iters": None,
        "lr_warmup_fraction": None,
        "lr_warmup_iters": 0,
        "lr_warmup_init": 0.0,
        "lr": 3.75e-7,
        "min_lr": 0.0,
        "lr_decay_style": "constant",
        "start_weight_decay": 0.0,
        "end_weight_decay": 0.0,
        "weight_decay_incr_style": "constant",
        "use_checkpoint_opt_param_scheduler": False,
        "override_opt_param_scheduler": False,
        "lr_wsd_decay_style": "linear",
    }
    values.update(overrides)
    return Namespace(**values)


@patch(f"{_MODEL_MODULE}.OptimizerParamScheduler")
def test_positive_training_schedule_is_unchanged(mock_scheduler):
    from miles.backends.megatron_utils.model import get_optimizer_param_scheduler

    args = _args()
    optimizer = MagicMock()
    get_optimizer_param_scheduler(args, optimizer)

    assert args.train_iters == 8
    assert args.lr_decay_iters == 8
    kwargs = mock_scheduler.call_args.kwargs
    assert kwargs["lr_warmup_steps"] == 0
    assert kwargs["lr_decay_steps"] == 1024
    assert kwargs["wd_incr_steps"] == 1024
    assert kwargs["wsd_decay_steps"] is None
    assert kwargs["use_checkpoint_opt_param_scheduler"] is False
    assert kwargs["override_opt_param_scheduler"] is False


@patch(f"{_MODEL_MODULE}.OptimizerParamScheduler")
def test_eval_only_uses_dummy_scheduler_horizon_without_training_updates(mock_scheduler):
    from miles.backends.megatron_utils.model import get_optimizer_param_scheduler

    args = _args(num_rollout=0)
    optimizer = MagicMock()
    get_optimizer_param_scheduler(args, optimizer)

    assert args.train_iters == 0
    assert args.lr_decay_iters == 1
    kwargs = mock_scheduler.call_args.kwargs
    assert kwargs["lr_decay_steps"] == 128
    assert kwargs["wd_incr_steps"] == 128


@patch(f"{_MODEL_MODULE}.OptimizerParamScheduler")
def test_explicit_positive_schedule_warmup_wsd_and_checkpoint_flags_are_unchanged(mock_scheduler):
    from miles.backends.megatron_utils.model import get_optimizer_param_scheduler

    args = _args(
        lr_decay_iters=3,
        lr_wsd_decay_iters=2,
        lr_warmup_fraction=0.25,
        weight_decay_incr_style="linear",
        use_checkpoint_opt_param_scheduler=True,
        override_opt_param_scheduler=True,
    )
    get_optimizer_param_scheduler(args, MagicMock())

    assert args.train_iters == 8
    assert args.lr_decay_iters == 3
    kwargs = mock_scheduler.call_args.kwargs
    assert kwargs["lr_warmup_steps"] == 96
    assert kwargs["lr_decay_steps"] == 384
    assert kwargs["wd_incr_steps"] == 1024
    assert kwargs["wd_incr_style"] == "linear"
    assert kwargs["wsd_decay_steps"] == 256
    assert kwargs["use_checkpoint_opt_param_scheduler"] is True
    assert kwargs["override_opt_param_scheduler"] is True


@patch(f"{_MODEL_MODULE}.OptimizerParamScheduler")
def test_positive_explicit_zero_decay_is_not_silently_clamped(mock_scheduler):
    from miles.backends.megatron_utils.model import get_optimizer_param_scheduler

    args = _args(lr_decay_iters=0, lr_warmup_iters=2)
    get_optimizer_param_scheduler(args, MagicMock())

    assert args.train_iters == 8
    kwargs = mock_scheduler.call_args.kwargs
    assert kwargs["lr_warmup_steps"] == 256
    assert kwargs["lr_decay_steps"] == 0
    assert kwargs["wd_incr_steps"] == 1024


@patch(f"{_MODEL_MODULE}.OptimizerParamScheduler")
def test_positive_rollout_with_zero_computed_iterations_is_not_silently_clamped(mock_scheduler):
    from miles.backends.megatron_utils.model import get_optimizer_param_scheduler

    args = _args(num_rollout=1, rollout_batch_size=1, n_samples_per_prompt=1)
    get_optimizer_param_scheduler(args, MagicMock())

    assert args.train_iters == 0
    assert args.lr_decay_iters == 0
    kwargs = mock_scheduler.call_args.kwargs
    assert kwargs["lr_decay_steps"] == 0
    assert kwargs["wd_incr_steps"] == 0
