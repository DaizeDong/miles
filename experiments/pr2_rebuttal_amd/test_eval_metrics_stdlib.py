"""Login-safe stdlib tests for the frozen rebuttal eval hook."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = Path(__file__).with_name("eval_metrics.py")
SPEC = importlib.util.spec_from_file_location("pr2_frozen_eval_metrics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


def sample(name: str, rm_type: str, index: int, row: int, reward: float):
    return SimpleNamespace(
        index=index,
        response=f"response-{name}-{index}",
        reward=reward,
        status=SimpleNamespace(value="completed"),
        metadata={
            "rebuttal_eval_dataset": name,
            "rm_type": rm_type,
            "source_row_index": row,
        },
    )


class FrozenEvalMetricsTest(unittest.TestCase):
    def test_official_metadata_removes_only_explicit_null_struct_fields(self) -> None:
        class FakeInput:
            def __init__(self, key, instruction_id_list, prompt, kwargs):
                self.key = key
                self.instruction_id_list = instruction_id_list
                self.prompt = prompt
                self.kwargs = kwargs

        evaluation_lib = SimpleNamespace(InputExample=FakeInput)
        metadata = {
            "record_id": 3,
            "prompt_text": "prompt",
            "instruction_id_list": ["constraint:id"],
            "kwargs": [{"unused": None, "zero": 0, "empty": "", "keyword": "kept"}],
        }
        result = METRICS._metadata_input(evaluation_lib, metadata)
        self.assertEqual(
            result.kwargs,
            [{"zero": 0, "empty": "", "keyword": "kept"}],
        )

    def test_per_dataset_n_metrics(self) -> None:
        self.assertEqual(METRICS.dataset_reward_metrics([1, 0], 1), {"avg@1": 0.5})
        result = METRICS.dataset_reward_metrics([1, 0, 0, 0], 4)
        self.assertEqual(result["avg@1"], 0.25)
        self.assertEqual(result["pass@1"], 0.25)
        self.assertEqual(result["pass@4"], 1.0)
        with self.assertRaisesRegex(ValueError, "binary"):
            METRICS.dataset_reward_metrics([0.5], 1)

    def test_fractional_constraint_mean_is_explicit_and_avg_only(self) -> None:
        self.assertEqual(
            METRICS.dataset_reward_metrics(
                [0.5, 0.25], 1, allow_fractional=True
            ),
            {"avg@1": 0.375},
        )
        with self.assertRaisesRegex(ValueError, "group_size=1"):
            METRICS.dataset_reward_metrics(
                [0.5, 0.0], 2, allow_fractional=True
            )
        for invalid in (-0.1, 1.1):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, r"\[0, 1\]"
            ):
                METRICS.dataset_reward_metrics(
                    [invalid], 1, allow_fractional=True
                )

    def test_flattened_samples_require_exact_tag_rm_type_and_n(self) -> None:
        configs = {
            "gsm8k": SimpleNamespace(n_samples_per_eval_prompt=1),
            "math500": SimpleNamespace(n_samples_per_eval_prompt=4),
        }
        specs = {"gsm8k": {"artifact_rows": 1}, "math500": {"artifact_rows": 1}}
        samples = [
            sample("gsm8k", "gsm8k_verl", 0, 0, 1.0),
            *[sample("math500", "math", index, 0, float(index == 0)) for index in range(4)],
        ]
        grouped = METRICS._validate_and_group_saved_samples(samples, configs, specs)
        self.assertEqual({key: len(value) for key, value in grouped.items()}, {"gsm8k": 1, "math500": 4})
        samples[-1].metadata["rebuttal_eval_dataset"] = "gsm8k"
        with self.assertRaisesRegex(ValueError, "invalid dataset/rm_type pair"):
            METRICS._validate_and_group_saved_samples(samples, configs, specs)

    def test_hook_returns_metrics_and_persists_create_once_hashes(self) -> None:
        gsm = [sample("gsm8k", "gsm8k_verl", 0, 0, 1.0)]
        math = [sample("math500", "math", index, 0, float(index == 0)) for index in range(4)]
        configs = [
            SimpleNamespace(name="gsm8k", n_samples_per_eval_prompt=1),
            SimpleNamespace(name="math500", n_samples_per_eval_prompt=4),
        ]
        args = SimpleNamespace(eval_datasets=configs)
        data = {
            "gsm8k": {"rewards": [1.0], "samples": gsm, "truncated": [False]},
            "math500": {
                "rewards": [1.0, 0.0, 0.0, 0.0],
                "samples": math,
                "truncated": [False] * 4,
            },
        }

        rollout_metrics = types.ModuleType("miles.ray.rollout.metrics")
        rollout_metrics._compute_metrics_from_samples = lambda _args, _samples: {}
        utils = types.ModuleType("miles.utils")
        tracking = types.ModuleType("miles.utils.tracking_utils")
        tracking.log = lambda *_args, **_kwargs: None
        utils.tracking_utils = tracking
        metric_utils = types.ModuleType("miles.utils.metric_utils")
        metric_utils.compute_rollout_step = lambda _args, rollout_id: rollout_id
        metric_utils.dict_add_prefix = lambda values, prefix: {
            f"{prefix}{key}": value for key, value in values.items()
        }
        fake_modules = {
            "miles": types.ModuleType("miles"),
            "miles.ray": types.ModuleType("miles.ray"),
            "miles.ray.rollout": types.ModuleType("miles.ray.rollout"),
            "miles.ray.rollout.metrics": rollout_metrics,
            "miles.utils": utils,
            "miles.utils.tracking_utils": tracking,
            "miles.utils.metric_utils": metric_utils,
        }

        with tempfile.TemporaryDirectory() as temporary:
            save_root = Path(temporary) / "run"
            with (
                mock.patch.dict(sys.modules, fake_modules),
                mock.patch.dict(os.environ, {"TENSORBOARD_DIR": str(save_root / "tensorboard")}),
                mock.patch.object(
                    METRICS,
                    "_load_manifest_contract",
                    return_value=(
                        {"gsm8k": {"artifact_rows": 1}, "math500": {"artifact_rows": 1}},
                        {"eval_config_sha256": "a" * 64},
                    ),
                ),
                mock.patch.object(
                    METRICS,
                    "_load_saved_eval_samples",
                    return_value=({"gsm8k": gsm, "math500": math}, {"raw_eval_sha256": "b" * 64}),
                ),
            ):
                result = METRICS.log_eval_rollout_data(233, args, data)
                self.assertEqual(result["eval/math500-pass@4"], 1.0)
                self.assertEqual(result["eval/step"], 233)
                artifact_dir = save_root / "eval_artifacts" / "rollout-233"
                aggregate = json.loads((artifact_dir / "aggregate.json").read_text(encoding="utf-8"))
                self.assertEqual(aggregate["provenance"]["raw_eval_sha256"], "b" * 64)
                checksums = json.loads((artifact_dir / "sha256.json").read_text(encoding="utf-8"))
                for filename, expected in checksums.items():
                    actual = hashlib.sha256((artifact_dir / filename).read_bytes()).hexdigest()
                    self.assertEqual(actual, expected)
                with self.assertRaises(FileExistsError):
                    METRICS.log_eval_rollout_data(233, args, data)
                artifact_dir.chmod(0o755)

    def test_hook_keeps_fractional_ifevalg_diagnostic_separate_from_official_primary(self) -> None:
        google = [sample("google_ifeval", "ifevalg", 0, 0, 0.5)]
        ifbench = [sample("ifbench_test", "ifbench", 0, 0, 1.0)]
        for record_id, item in enumerate((*google, *ifbench)):
            item.metadata.update(
                record_id=record_id,
                prompt_text=f"prompt-{record_id}",
                instruction_id_list=["constraint-a", "constraint-b"],
                kwargs=[{}, {}],
            )
        configs = [
            SimpleNamespace(name="google_ifeval", n_samples_per_eval_prompt=1),
            SimpleNamespace(name="ifbench_test", n_samples_per_eval_prompt=1),
        ]
        args = SimpleNamespace(eval_datasets=configs)
        data = {
            "google_ifeval": {
                "rewards": [0.5],
                "samples": google,
                "truncated": [False],
            },
            "ifbench_test": {
                "rewards": [1.0],
                "samples": ifbench,
                "truncated": [False],
            },
        }

        def official_outputs(_samples, rm_type):
            if rm_type == "ifevalg":
                strict_decisions = [True, False]
                loose_decisions = [True, True]
            else:
                strict_decisions = loose_decisions = [True, True]

            def output(decisions):
                return SimpleNamespace(
                    follow_instruction_list=decisions,
                    instruction_id_list=["constraint-a", "constraint-b"],
                    follow_all_instructions=all(decisions),
                )

            return [output(strict_decisions)], [output(loose_decisions)]

        rollout_metrics = types.ModuleType("miles.ray.rollout.metrics")
        rollout_metrics._compute_metrics_from_samples = lambda _args, _samples: {}
        utils = types.ModuleType("miles.utils")
        tracking = types.ModuleType("miles.utils.tracking_utils")
        tracking.log = lambda *_args, **_kwargs: None
        utils.tracking_utils = tracking
        metric_utils = types.ModuleType("miles.utils.metric_utils")
        metric_utils.compute_rollout_step = lambda _args, rollout_id: rollout_id
        metric_utils.dict_add_prefix = lambda values, prefix: {
            f"{prefix}{key}": value for key, value in values.items()
        }
        fake_modules = {
            "miles": types.ModuleType("miles"),
            "miles.ray": types.ModuleType("miles.ray"),
            "miles.ray.rollout": types.ModuleType("miles.ray.rollout"),
            "miles.ray.rollout.metrics": rollout_metrics,
            "miles.utils": utils,
            "miles.utils.tracking_utils": tracking,
            "miles.utils.metric_utils": metric_utils,
        }

        with tempfile.TemporaryDirectory() as temporary:
            save_root = Path(temporary) / "run"
            with (
                mock.patch.dict(sys.modules, fake_modules),
                mock.patch.dict(os.environ, {"TENSORBOARD_DIR": str(save_root / "tensorboard")}),
                mock.patch.object(
                    METRICS,
                    "_load_manifest_contract",
                    return_value=(
                        {
                            "google_ifeval": {"artifact_rows": 1},
                            "ifbench_test": {"artifact_rows": 1},
                        },
                        {"eval_config_sha256": "a" * 64},
                    ),
                ),
                mock.patch.object(
                    METRICS,
                    "_load_saved_eval_samples",
                    return_value=(
                        {"google_ifeval": google, "ifbench_test": ifbench},
                        {"raw_eval_sha256": "b" * 64},
                    ),
                ),
                mock.patch.object(METRICS, "_official_outputs", side_effect=official_outputs),
            ):
                result = METRICS.log_eval_rollout_data(232, args, data)

            self.assertEqual(result["eval/google_ifeval-online_reward"], 0.5)
            self.assertEqual(result["eval/google_ifeval"], 1.0)
            self.assertEqual(result["eval/google_ifeval-prompt_strict"], 0.0)
            self.assertEqual(result["eval/google_ifeval-prompt_loose"], 1.0)
            aggregate_path = save_root / "eval_artifacts" / "rollout-232" / "aggregate.json"
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            self.assertEqual(
                aggregate["datasets"]["google_ifeval"]["online_reward_diagnostic"],
                0.5,
            )

    def test_strict_ifbench_remains_binary_fail_closed(self) -> None:
        self.assertNotIn("ifbench", METRICS._FRACTIONAL_REWARD_RM_TYPES)
        with self.assertRaisesRegex(ValueError, "binary"):
            METRICS.dataset_reward_metrics([0.5], 1)


if __name__ == "__main__":
    unittest.main()
