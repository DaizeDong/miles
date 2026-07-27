from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils import dist_ckpt_compat
from miles.backends.megatron_utils import predictive_router_replay
from miles.utils import distributed_utils


def _stats(**overrides):
    values = {
        "num_predictor_params": 26,
        "num_nonzero_predictor_params": 26,
        "predictor_numel": 3_407_872,
        "all_weights_finite": True,
        "weight_sum": 1.25,
        "weight_abs_sum": 2.5,
        "weight_l2_norm": 0.75,
    }
    values.update(overrides)
    return values


def _install_single_rank_collectives(monkeypatch):
    monkeypatch.setattr(dist_ckpt_compat.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist_ckpt_compat.dist, "get_world_size", lambda group=None: 1)

    def all_gather_object(records, local_record, group=None):
        records[0] = local_record

    monkeypatch.setattr(dist_ckpt_compat.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        dist_ckpt_compat.dist,
        "broadcast_object_list",
        lambda result, src=0, group=None: None,
    )
    monkeypatch.setattr(distributed_utils, "get_gloo_group", lambda: object())


def _enable_gate(monkeypatch):
    monkeypatch.setenv("PR2_MODEL_ONLY_RELOAD_GATE", "1")
    monkeypatch.setenv("PR2_VALIDATE", "1")
    monkeypatch.setenv("PR2_EXPECT_CHECKPOINT_ITERATION", "2")
    monkeypatch.setenv("PR2_EXPECT_PREDICTOR_NUMEL", "3407872")


def _gate_args(**overrides):
    values = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("raw_expected_step", [None, "", "   \t"])
def test_optimizer_resume_probe_treats_missing_or_blank_expectation_as_disabled(
    monkeypatch, raw_expected_step
):
    monkeypatch.setenv("PR2_VALIDATE", "1")
    if raw_expected_step is None:
        monkeypatch.delenv("PR2_EXPECT_RESUME_HDO_STEP", raising=False)
    else:
        monkeypatch.setenv("PR2_EXPECT_RESUME_HDO_STEP", raw_expected_step)
    monkeypatch.setattr(
        dist_ckpt_compat,
        "_collect_local_optimizer_resume_record",
        lambda *args, **kwargs: pytest.fail("disabled optimizer-resume probe must not inspect optimizer state"),
    )

    dist_ckpt_compat.maybe_validate_pr2_optimizer_resume(SimpleNamespace(), object(), 0)


def test_optimizer_resume_probe_strips_and_validates_nonblank_expectation(monkeypatch):
    monkeypatch.setenv("PR2_VALIDATE", "1")
    monkeypatch.setenv("PR2_EXPECT_RESUME_HDO_STEP", " 8 \t")
    monkeypatch.setenv("PR2_EXPECT_CHECKPOINT_ITERATION", "2")
    monkeypatch.setenv("PR2_EXPECT_PREDICTOR_NUMEL", "3407872")
    _install_single_rank_collectives(monkeypatch)
    monkeypatch.setattr(
        dist_ckpt_compat,
        "_collect_local_optimizer_resume_record",
        lambda *args, **kwargs: {
            "rank": 0,
            "iteration": 2,
            "local_error": None,
            "lrs": [3.75e-7, 1.875e-5, 3.75e-7],
            "max_lrs": [3.75e-7, 1.875e-5, 3.75e-7],
            "children": [
                {
                    "index": 0,
                    "is_hdo": True,
                    "state_count": 1,
                    "missing_steps": False,
                    "steps": [8.0],
                    "missing_sub_steps": False,
                    "sub_steps": [8.0],
                }
            ],
            "predictor_group_count": 1,
            "predictor_state_count": 1,
            "predictor_numel": 3_407_872,
            "exp_avg_abs_sum": 1.0,
            "exp_avg_sq_abs_sum": 1.0,
        },
    )

    args = SimpleNamespace(lr=3.75e-7, bias_predictor_lr_mult=50)
    dist_ckpt_compat.maybe_validate_pr2_optimizer_resume(args, object(), 2)


def test_model_reload_probe_is_noop_unless_explicitly_enabled(monkeypatch):
    monkeypatch.delenv("PR2_MODEL_ONLY_RELOAD_GATE", raising=False)
    monkeypatch.setattr(
        predictive_router_replay,
        "collect_predictive_param_stats",
        lambda model: pytest.fail("disabled probe must not inspect the model"),
    )
    dist_ckpt_compat.maybe_validate_pr2_model_reload(SimpleNamespace(), [], 2)


def test_model_reload_probe_accepts_expected_nonzero_finite_predictor(monkeypatch):
    _enable_gate(monkeypatch)
    _install_single_rank_collectives(monkeypatch)
    monkeypatch.setattr(predictive_router_replay, "collect_predictive_param_stats", lambda model: _stats())
    dist_ckpt_compat.maybe_validate_pr2_model_reload(_gate_args(), [object()], 2)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"predictor_numel": 1}, "predictor_numel=1"),
        (
            {
                "num_nonzero_predictor_params": 0,
                "weight_abs_sum": 0.0,
                "weight_l2_norm": 0.0,
            },
            "nonzero predictor tensors=0/26",
        ),
        ({"all_weights_finite": False, "weight_sum": float("nan")}, "non-finite"),
    ],
)
def test_model_reload_probe_rejects_bad_predictor_state(monkeypatch, overrides, message):
    _enable_gate(monkeypatch)
    _install_single_rank_collectives(monkeypatch)
    monkeypatch.setattr(
        predictive_router_replay,
        "collect_predictive_param_stats",
        lambda model: _stats(**overrides),
    )
    with pytest.raises(RuntimeError, match=message):
        dist_ckpt_compat.maybe_validate_pr2_model_reload(_gate_args(), [object()], 2)


@pytest.mark.parametrize(
    ("environment_name", "environment_value", "message"),
    [
        ("PR2_EXPECT_CHECKPOINT_ITERATION", None, "requires PR2_EXPECT_CHECKPOINT_ITERATION"),
        ("PR2_EXPECT_CHECKPOINT_ITERATION", "not-an-integer", "must be a positive integer"),
        ("PR2_EXPECT_CHECKPOINT_ITERATION", "0", "must be a positive integer"),
    ],
)
def test_model_reload_probe_rejects_missing_or_invalid_iteration(
    monkeypatch, environment_name, environment_value, message
):
    _enable_gate(monkeypatch)
    if environment_value is None:
        monkeypatch.delenv(environment_name)
    else:
        monkeypatch.setenv(environment_name, environment_value)
    with pytest.raises(RuntimeError, match=message):
        dist_ckpt_compat.maybe_validate_pr2_model_reload(_gate_args(), [object()], 2)


def test_model_reload_probe_rejects_iteration_mismatch(monkeypatch):
    _enable_gate(monkeypatch)
    _install_single_rank_collectives(monkeypatch)
    monkeypatch.setattr(predictive_router_replay, "collect_predictive_param_stats", lambda model: _stats())
    with pytest.raises(RuntimeError, match="iteration=3, expected=2"):
        dist_ckpt_compat.maybe_validate_pr2_model_reload(_gate_args(), [object()], 3)


@pytest.mark.parametrize("field", ["tensor_model_parallel_size", "pipeline_model_parallel_size"])
def test_model_reload_probe_rejects_unsupported_topology(monkeypatch, field):
    _enable_gate(monkeypatch)
    with pytest.raises(RuntimeError, match="tensor/pipeline parallel size 1"):
        dist_ckpt_compat.maybe_validate_pr2_model_reload(_gate_args(**{field: 2}), [object()], 2)


def test_model_reload_probe_propagates_a_remote_rank_failure(monkeypatch):
    _enable_gate(monkeypatch)
    monkeypatch.setattr(dist_ckpt_compat.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist_ckpt_compat.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(distributed_utils, "get_gloo_group", lambda: object())
    monkeypatch.setattr(predictive_router_replay, "collect_predictive_param_stats", lambda model: _stats())

    def all_gather_object(records, local_record, group=None):
        records[0] = local_record
        records[1] = {
            "rank": 1,
            "iteration": 2,
            "local_error": "RuntimeError: malformed remote predictor",
        }

    monkeypatch.setattr(dist_ckpt_compat.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        dist_ckpt_compat.dist,
        "broadcast_object_list",
        lambda result, src=0, group=None: None,
    )
    with pytest.raises(RuntimeError, match="malformed remote predictor"):
        dist_ckpt_compat.maybe_validate_pr2_model_reload(_gate_args(), [object()], 2)
