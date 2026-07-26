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
    def test_per_dataset_n_metrics(self) -> None:
        self.assertEqual(METRICS.dataset_reward_metrics([1, 0], 1), {"avg@1": 0.5})
        result = METRICS.dataset_reward_metrics([1, 0, 0, 0], 4)
        self.assertEqual(result["avg@1"], 0.25)
        self.assertEqual(result["pass@1"], 0.25)
        self.assertEqual(result["pass@4"], 1.0)
        with self.assertRaisesRegex(ValueError, "binary"):
            METRICS.dataset_reward_metrics([0.5], 1)

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


if __name__ == "__main__":
    unittest.main()
