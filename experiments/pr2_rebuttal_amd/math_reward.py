"""Frozen math reward dispatcher for the PR2 rebuttal experiments.

The historical Moonlight runs used Verl's strict GSM8K convention: the final
answer must appear after ``####`` in the last 300 response characters.  The
held-out MATH500 evaluation instead uses the standard boxed-answer grader.
Miles exposes one global ``--custom-rm-path`` even when an eval config contains
multiple datasets, so this module dispatches explicitly on each sample's
``metadata.rm_type`` rather than silently applying the GSM8K scorer to both.
"""

from __future__ import annotations

import re
from typing import Any


_SOLUTION_CLIP_CHARS = 300
_STRICT_GSM8K_PATTERN = re.compile(r"#### (\-?[0-9\.\,]+)")
_GSM8K_RM_TYPES = frozenset({"", "gsm8k_verl"})
_MATH_RM_TYPES = frozenset({"math"})


def extract_gsm8k_solution(solution_str: str) -> str | None:
    """Extract the strict final GSM8K answer used by the historical Verl run."""

    clipped = solution_str[-_SOLUTION_CLIP_CHARS:]
    solutions = _STRICT_GSM8K_PATTERN.findall(clipped)
    if not solutions:
        return None
    return solutions[-1].replace(",", "").replace("$", "")


def score_gsm8k_response(response: str | None, label: Any) -> float:
    answer = extract_gsm8k_solution(response or "")
    if answer is None:
        return 0.0
    return 1.0 if answer == str(label) else 0.0


def grade_math_response(response: str | None, label: Any) -> bool:
    """Load the Miles MATH grader lazily so GSM8K-only workers stay minimal."""

    from miles.rollout.rm_hub.math_utils import grade_answer_verl

    return bool(grade_answer_verl(response or "", label))


def _sample_rm_type(sample: Any) -> str:
    metadata = getattr(sample, "metadata", None)
    if metadata is None:
        return ""
    if not isinstance(metadata, dict):
        raise TypeError("math reward sample.metadata must be a mapping or None")
    value = metadata.get("rm_type", "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("math reward metadata.rm_type must be a string")
    return value.strip()


def score_sample(sample: Any) -> float:
    """Score one sample, failing closed for an unregistered dataset scorer."""

    rm_type = _sample_rm_type(sample)
    response = getattr(sample, "response", None)
    label = getattr(sample, "label", None)
    if rm_type in _GSM8K_RM_TYPES:
        return score_gsm8k_response(response, label)
    if rm_type in _MATH_RM_TYPES:
        return 1.0 if grade_math_response(response, label) else 0.0
    raise ValueError(f"unsupported rebuttal math rm_type: {rm_type!r}")


async def reward_fn(args: Any, sample_or_samples: Any, **kwargs: Any) -> float | list[float]:
    """Miles ``--custom-rm-path`` entry point for training and mixed eval."""

    del args, kwargs
    if isinstance(sample_or_samples, list):
        return [score_sample(sample) for sample in sample_or_samples]
    return score_sample(sample_or_samples)

