from argparse import Namespace
from pathlib import Path

import pytest
import torch

from miles.backends.training_utils.ci_utils import check_grad_norm, check_kl


def _make_args(**overrides):
    values = {
        "multi_latent_attention": False,
        "lora_rank": 0,
        "use_rollout_routing_replay": False,
        "ci_save_grad_norm": None,
        "ci_load_grad_norm": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_check_kl_only_treats_first_predictive_pass_as_initial_step():
    args = _make_args()

    check_kl(
        args,
        {"train/ppo_kl": 0.0, "train/pg_clipfrac": 0.0},
        step_id=0,
        accumulated_step_id=0,
        train_pass_index=0,
        num_steps_per_rollout=2,
    )

    check_kl(
        args,
        {"train/ppo_kl": 1.0, "train/pg_clipfrac": 1.0},
        step_id=0,
        accumulated_step_id=2,
        train_pass_index=1,
        num_steps_per_rollout=2,
    )


def test_check_grad_norm_uses_pass_aware_accumulated_step_id_in_path(tmp_path: Path):
    path_template = str(tmp_path / "{role}-{rollout_id}-{step_id}-{train_pass_index}-{accumulated_step_id}.pt")
    args = _make_args(ci_save_grad_norm=path_template)

    check_grad_norm(
        args=args,
        grad_norm=1.23,
        rollout_id=4,
        step_id=1,
        train_pass_index=1,
        num_steps_per_rollout=3,
        num_train_passes=2,
        role="actor",
        rank=0,
    )

    expected_path = tmp_path / "actor-4-1-1-28.pt"
    assert expected_path.exists()
    assert torch.load(expected_path, weights_only=False) == pytest.approx(1.23)
