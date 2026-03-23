import argparse
import importlib
import sys
import types
from argparse import Namespace

import pytest


def _install_argument_test_stubs():
    if "sglang_router.launch_router" not in sys.modules:
        launch_router_module = types.ModuleType("sglang_router.launch_router")

        class RouterArgs:
            @staticmethod
            def add_cli_args(parser, **kwargs):
                return parser

        launch_router_module.RouterArgs = RouterArgs
        router_module = types.ModuleType("sglang_router")
        router_module.launch_router = launch_router_module
        sys.modules["sglang_router"] = router_module
        sys.modules["sglang_router.launch_router"] = launch_router_module

    if "transformers" not in sys.modules:
        transformers_module = types.ModuleType("transformers")

        class AutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("AutoConfig.from_pretrained should not be called in predictive argument tests.")

        transformers_module.AutoConfig = AutoConfig
        sys.modules["transformers"] = transformers_module

    if "ray" not in sys.modules:
        ray_module = types.ModuleType("ray")

        def remote(*args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        ray_module.remote = remote
        ray_module.init = lambda *args, **kwargs: None
        ray_module.shutdown = lambda *args, **kwargs: None
        ray_module.get = lambda refs: refs
        ray_module.nodes = lambda: []
        ray_private_module = types.ModuleType("ray._private")
        services_module = types.ModuleType("ray._private.services")
        services_module.get_node_ip_address = lambda: "127.0.0.1"
        ray_private_module.services = services_module
        ray_module._private = ray_private_module
        sys.modules["ray"] = ray_module
        sys.modules["ray._private"] = ray_private_module
        sys.modules["ray._private.services"] = services_module

        ray_util_module = types.ModuleType("ray.util")
        scheduling_module = types.ModuleType("ray.util.scheduling_strategies")

        class NodeAffinitySchedulingStrategy:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        scheduling_module.NodeAffinitySchedulingStrategy = NodeAffinitySchedulingStrategy
        ray_util_module.scheduling_strategies = scheduling_module
        sys.modules["ray.util"] = ray_util_module
        sys.modules["ray.util.scheduling_strategies"] = scheduling_module

    if "miles.backends.sglang_utils.arguments" not in sys.modules:
        sglang_args_module = types.ModuleType("miles.backends.sglang_utils.arguments")
        sglang_args_module.add_sglang_arguments = lambda parser: parser
        sglang_args_module.validate_args = lambda args: None
        sys.modules["miles.backends.sglang_utils.arguments"] = sglang_args_module

    if "httpx" not in sys.modules:
        httpx_module = types.ModuleType("httpx")

        class Limits:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class Timeout:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class Client:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class AsyncClient(Client):
            async def get(self, *args, **kwargs):
                raise RuntimeError("httpx.AsyncClient.get should not be called in predictive argument tests.")

            async def post(self, *args, **kwargs):
                raise RuntimeError("httpx.AsyncClient.post should not be called in predictive argument tests.")

            async def delete(self, *args, **kwargs):
                raise RuntimeError("httpx.AsyncClient.delete should not be called in predictive argument tests.")

        class HTTPStatusError(Exception):
            def __init__(self, *args, response=None, **kwargs):
                super().__init__(*args)
                self.response = response

        httpx_module.Limits = Limits
        httpx_module.Timeout = Timeout
        httpx_module.Client = Client
        httpx_module.AsyncClient = AsyncClient
        httpx_module.HTTPStatusError = HTTPStatusError
        sys.modules["httpx"] = httpx_module


_install_argument_test_stubs()

arguments = importlib.import_module("miles.utils.arguments")
PREDICTIVE_ROUTING_REPLAY_LOSS_TYPES = arguments.PREDICTIVE_ROUTING_REPLAY_LOSS_TYPES
PREDICTIVE_ROUTING_REPLAY_STORAGE_DTYPES = arguments.PREDICTIVE_ROUTING_REPLAY_STORAGE_DTYPES
_validate_predictive_routing_replay_args = arguments._validate_predictive_routing_replay_args
get_miles_extra_args_provider = arguments.get_miles_extra_args_provider


def _make_parser():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    return parser


def _make_validation_args(**overrides):
    values = {
        "enable_predictive_routing_replay": False,
        "bias_predictor_loss_type": "kl-post",
        "bias_predictor_lr_mult": 1000.0,
        "predictive_downsample_batch_size": None,
        "predictive_downsample_max_len_limit": None,
        "predictive_storage_dtype": "bf16",
        "train_backend": "megatron",
        "use_routing_replay": False,
        "use_rollout_routing_replay": False,
        "allgather_cp": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_predictive_flags_parse():
    parser = _make_parser()
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "64",
            "--enable-predictive-routing-replay",
            "--bias-predictor-loss-type",
            "kl-post",
            "--bias-predictor-lr-mult",
            "321.0",
            "--predictive-downsample-batch-size",
            "4",
            "--predictive-downsample-max-len-limit",
            "1024",
            "--predictive-storage-dtype",
            "fp16",
        ]
    )

    assert args.enable_predictive_routing_replay is True
    assert args.bias_predictor_loss_type == "kl-post"
    assert args.bias_predictor_lr_mult == pytest.approx(321.0)
    assert args.predictive_downsample_batch_size == 4
    assert args.predictive_downsample_max_len_limit == 1024
    assert args.predictive_storage_dtype == "fp16"


def test_predictive_loss_type_defaults_to_kl_post():
    parser = _make_parser()
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "64",
        ]
    )

    assert args.bias_predictor_loss_type == "kl-post"


def test_predictive_validation_sets_aliases():
    args = _make_validation_args(enable_predictive_routing_replay=True, use_routing_replay=True)

    _validate_predictive_routing_replay_args(args)

    assert args.enable_bias_predictor is True
    assert args.predictive_routing_replay_mode == "R2"


def test_predictive_validation_disabled_path_sets_aliases():
    args = _make_validation_args()

    _validate_predictive_routing_replay_args(args)

    assert args.enable_bias_predictor is False
    assert args.predictive_routing_replay_mode is None


def test_predictive_validation_requires_routing_replay():
    args = _make_validation_args(enable_predictive_routing_replay=True)

    with pytest.raises(AssertionError, match="requires --use-routing-replay"):
        _validate_predictive_routing_replay_args(args)


def test_predictive_validation_rejects_rollout_routing_replay():
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        use_rollout_routing_replay=True,
    )

    with pytest.raises(AssertionError, match="actor-side R2"):
        _validate_predictive_routing_replay_args(args)


def test_predictive_validation_requires_megatron():
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        train_backend="fsdp",
    )

    with pytest.raises(AssertionError, match="megatron backend"):
        _validate_predictive_routing_replay_args(args)


def test_predictive_validation_rejects_allgather_cp():
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        allgather_cp=True,
    )

    with pytest.raises(AssertionError, match="allgather-cp"):
        _validate_predictive_routing_replay_args(args)


@pytest.mark.parametrize("loss_type", PREDICTIVE_ROUTING_REPLAY_LOSS_TYPES)
def test_predictive_validation_accepts_supported_loss_types(loss_type):
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        bias_predictor_loss_type=loss_type,
    )

    _validate_predictive_routing_replay_args(args)


@pytest.mark.parametrize("storage_dtype", PREDICTIVE_ROUTING_REPLAY_STORAGE_DTYPES)
def test_predictive_validation_accepts_supported_storage_dtypes(storage_dtype):
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        predictive_storage_dtype=storage_dtype,
    )

    _validate_predictive_routing_replay_args(args)


def test_predictive_validation_rejects_nonpositive_lr_multiplier():
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        bias_predictor_lr_mult=0,
    )

    with pytest.raises(AssertionError, match="bias-predictor-lr-mult"):
        _validate_predictive_routing_replay_args(args)


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        ("predictive_downsample_batch_size", 0, "predictive-downsample-batch-size"),
        ("predictive_downsample_max_len_limit", 0, "predictive-downsample-max-len-limit"),
    ],
)
def test_predictive_validation_rejects_nonpositive_downsample_values(field_name, field_value, message):
    args = _make_validation_args(
        enable_predictive_routing_replay=True,
        use_routing_replay=True,
        **{field_name: field_value},
    )

    with pytest.raises(AssertionError, match=message):
        _validate_predictive_routing_replay_args(args)
