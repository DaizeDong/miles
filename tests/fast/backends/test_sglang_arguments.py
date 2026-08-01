import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import patch


def _load_arguments_module():
    server_args_module = ModuleType("sglang.srt.server_args")
    server_args_module.ServerArgs = type("ServerArgs", (), {})
    sglang_module = ModuleType("sglang")
    sglang_module.__path__ = []
    srt_module = ModuleType("sglang.srt")
    srt_module.__path__ = []
    http_utils_module = ModuleType("miles.utils.http_utils")
    http_utils_module._wrap_ipv6 = lambda value: value

    source = Path(__file__).parents[3] / "miles/backends/sglang_utils/arguments.py"
    spec = importlib.util.spec_from_file_location("sglang_arguments_under_test", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "sglang": sglang_module,
            "sglang.srt": srt_module,
            "sglang.srt.server_args": server_args_module,
            "miles.utils.http_utils": http_utils_module,
        },
    ):
        spec.loader.exec_module(module)
    return module


arguments = _load_arguments_module()


def _base_args(**overrides):
    values = {
        "rollout_num_gpus_per_engine": 8,
        "true_on_policy_mode": False,
        "recompute_logprobs_via_prefill": False,
        "sglang_router_policy": None,
        "sglang_router_ip": None,
        "sglang_enable_dp_attention": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestSglangParallelSizeAliases(TestCase):
    def test_validate_args_accepts_current_short_parallel_fields(self):
        args = _base_args(
            sglang_dp_size=1,
            sglang_pp_size=2,
            sglang_ep_size=4,
        )

        arguments.validate_args(args)

        self.assertEqual(args.sglang_tp_size, 8)
        self.assertEqual(args.sglang_dp_size, 1)
        self.assertEqual(args.sglang_pp_size, 2)
        self.assertEqual(args.sglang_ep_size, 4)
        self.assertFalse(hasattr(args, "sglang_data_parallel_size"))
        self.assertFalse(hasattr(args, "sglang_pipeline_parallel_size"))
        self.assertFalse(hasattr(args, "sglang_expert_parallel_size"))

    def test_validate_args_accepts_legacy_long_parallel_fields(self):
        args = _base_args(
            sglang_data_parallel_size=1,
            sglang_pipeline_parallel_size=2,
            sglang_expert_parallel_size=4,
        )

        arguments.validate_args(args)

        self.assertEqual(args.sglang_tp_size, 8)
        self.assertEqual(args.sglang_dp_size, 1)
        self.assertEqual(args.sglang_pp_size, 2)
        self.assertEqual(args.sglang_ep_size, 4)

    def test_validate_args_accepts_matching_parallel_aliases(self):
        args = _base_args(
            sglang_dp_size=1,
            sglang_data_parallel_size=1,
            sglang_pp_size=2,
            sglang_pipeline_parallel_size=2,
            sglang_ep_size=4,
            sglang_expert_parallel_size=4,
        )

        arguments.validate_args(args)

        self.assertEqual(args.sglang_dp_size, 1)
        self.assertEqual(args.sglang_pp_size, 2)
        self.assertEqual(args.sglang_ep_size, 4)

    def test_validate_args_rejects_conflicting_parallel_aliases(self):
        aliases = [
            ("sglang_dp_size", "sglang_data_parallel_size"),
            ("sglang_pp_size", "sglang_pipeline_parallel_size"),
            ("sglang_ep_size", "sglang_expert_parallel_size"),
        ]
        for short_name, legacy_name in aliases:
            with self.subTest(short_name=short_name, legacy_name=legacy_name):
                values = {
                    "sglang_dp_size": 1,
                    "sglang_pp_size": 1,
                    "sglang_ep_size": 1,
                    short_name: 2,
                    legacy_name: 3,
                }
                args = _base_args(**values)

                with self.assertRaisesRegex(ValueError, "Conflicting SGLang aliases"):
                    arguments.validate_args(args)
