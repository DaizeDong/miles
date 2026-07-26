from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from experiments.pr2_rebuttal_amd import math_reward


class MathRewardTests(unittest.TestCase):
    def sample(self, response: str, label: str, rm_type: str | None = None):
        metadata = {} if rm_type is None else {"rm_type": rm_type}
        return SimpleNamespace(response=response, label=label, metadata=metadata)

    def test_historical_gsm8k_semantics(self) -> None:
        self.assertEqual(math_reward.extract_gsm8k_solution("work\n#### 1,234"), "1234")
        self.assertIsNone(math_reward.extract_gsm8k_solution(r"\boxed{1234}"))
        self.assertEqual(math_reward.score_sample(self.sample("work\n#### -2.5", "-2.5")), 1.0)
        self.assertEqual(math_reward.score_sample(self.sample("work\n#### 3", "4", "gsm8k_verl")), 0.0)

    def test_math500_uses_boxed_grader_branch(self) -> None:
        sample = self.sample(r"work\n\boxed{7}", "7", "math")
        with mock.patch.object(math_reward, "grade_math_response", return_value=True) as grader:
            self.assertEqual(math_reward.score_sample(sample), 1.0)
        grader.assert_called_once_with(sample.response, sample.label)

    def test_unknown_rm_type_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported rebuttal math rm_type"):
            math_reward.score_sample(self.sample("x", "x", "ifeval_old"))

    def test_batched_async_entry_point_preserves_order(self) -> None:
        samples = [
            self.sample("#### 1", "1", "gsm8k_verl"),
            self.sample("#### 2", "3", "gsm8k_verl"),
        ]
        scores = asyncio.run(math_reward.reward_fn(None, samples))
        self.assertEqual(scores, [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
