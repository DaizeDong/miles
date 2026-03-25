from types import SimpleNamespace

from miles.utils import wandb_utils


class _FakeWandb:
    class util:
        @staticmethod
        def generate_id():
            return "abc123"


def _make_args(**overrides):
    values = {
        "wandb_group": "PREEXP-Qwen3-30B-A3B-Base",
        "wandb_project": "miles+Qwen3-30B-A3B-Base",
        "wandb_random_suffix": True,
        "wandb_run_name": None,
        "wandb_group_resolved": None,
        "rank": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_random_suffix_creates_unique_group(monkeypatch):
    monkeypatch.setattr(wandb_utils, "wandb", _FakeWandb)
    args = _make_args()

    group, run_name = wandb_utils._resolve_wandb_identity(args)

    assert group == "PREEXP-Qwen3-30B-A3B-Base_abc123"
    assert run_name == "PREEXP-Qwen3-30B-A3B-Base_abc123-RANK_0"
    assert args.wandb_group_resolved == group
    assert args.wandb_run_name == run_name


def test_disable_random_suffix_keeps_stable_group():
    args = _make_args(wandb_random_suffix=False)

    group, run_name = wandb_utils._resolve_wandb_identity(args)

    assert group == "PREEXP-Qwen3-30B-A3B-Base"
    assert run_name == "PREEXP-Qwen3-30B-A3B-Base"


def test_pre_resolved_identity_is_reused():
    args = _make_args(
        wandb_group_resolved="PREEXP-Qwen3-30B-A3B-Base_existing",
        wandb_run_name="PREEXP-Qwen3-30B-A3B-Base_existing-RANK_0",
    )

    group, run_name = wandb_utils._resolve_wandb_identity(args)

    assert group == "PREEXP-Qwen3-30B-A3B-Base_existing"
    assert run_name == "PREEXP-Qwen3-30B-A3B-Base_existing-RANK_0"
