import json
import sys
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from miles.rollout.rm_hub import async_rm
from miles.rollout.rm_hub import ifbench
from miles.utils.async_utils import run
from miles.utils.types import Sample


class _FakeInputExample:
    def __init__(self, *, key, instruction_id_list, prompt, kwargs):
        self.key = key
        self.instruction_id_list = instruction_id_list
        self.prompt = prompt
        self.kwargs = kwargs


class _FakeEvaluationLib:
    InputExample = _FakeInputExample

    @staticmethod
    def test_instruction_following_strict(inp, prompt_to_response):
        response = prompt_to_response[inp.prompt]
        decisions = [token in response for token in inp.instruction_id_list]
        return SimpleNamespace(
            follow_all_instructions=all(decisions),
            follow_instruction_list=decisions,
        )


class _FakeLegacyVerifierModule:
    IF_FUNCTIONS_MAP = {
        "contains_keyword": lambda response, keyword: keyword in response,
        "is_lowercase": lambda response: response == response.lower(),
    }


class _ContainsInstruction:
    def __init__(self, instruction_id):
        self.instruction_id = instruction_id
        self.keyword = None

    def build_description(self, keyword=None):
        self.keyword = keyword

    def check_following(self, response):
        return self.keyword in response


class _NoKwargsInstruction:
    def __init__(self, instruction_id):
        self.instruction_id = instruction_id

    def build_description(self):
        return None

    def check_following(self, response):
        return False


class _FakeIFEvalGRegistry:
    INSTRUCTION_DICT = {
        "contains:first": _ContainsInstruction,
        "contains:second": _ContainsInstruction,
        "no_kwargs:false": _NoKwargsInstruction,
    }


@pytest.fixture
def fake_evaluation_lib(monkeypatch):
    fake = _FakeEvaluationLib()
    monkeypatch.setattr(ifbench, "_load_evaluation_lib", lambda: fake)
    return fake


@pytest.fixture
def fake_open_instruct(monkeypatch):
    modules = {
        "open_instruct.if_functions": _FakeLegacyVerifierModule(),
        "open_instruct.IFEvalG.instructions_registry": _FakeIFEvalGRegistry(),
    }
    monkeypatch.setattr(ifbench, "_load_open_instruct_module", modules.__getitem__)
    return modules


def _metadata(*instruction_ids):
    return {
        "record_id": 7,
        "prompt_text": "test prompt",
        "instruction_id_list": list(instruction_ids),
        "kwargs": [{} for _ in instruction_ids],
    }


def test_strict_reward_preserves_all_constraints_semantics(fake_evaluation_lib):
    metadata = _metadata("alpha", "beta")

    assert ifbench.compute_ifbench_reward("alpha beta", None, metadata) == 1.0
    assert ifbench.compute_ifbench_reward("alpha only", None, metadata) == 0.0


def test_per_constraint_mean_reward_is_dense(fake_evaluation_lib):
    metadata = _metadata("alpha", "beta")

    reward = ifbench.compute_ifbench_reward(
        "alpha only",
        None,
        metadata,
        aggregation="per_constraint_mean",
    )

    assert reward == 0.5


def test_invalid_aggregation_fails_before_loading_verifier(monkeypatch):
    monkeypatch.setattr(
        ifbench,
        "_load_evaluation_lib",
        MagicMock(side_effect=AssertionError("verifier should not be loaded")),
    )

    with pytest.raises(ValueError, match="Unsupported IFBench reward aggregation"):
        ifbench.compute_ifbench_reward("response", None, _metadata("alpha"), aggregation="unknown")


def test_missing_verifier_requires_explicit_or_vendored_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv("IFBENCH_REPO_PATH", raising=False)
    monkeypatch.setattr(ifbench, "_VENDORED_IFBENCH_REPO", tmp_path / "missing-ifbench")

    with pytest.raises(ImportError, match="runtime cloning and pip installation are disabled"):
        ifbench._resolve_ifbench_repo()


def test_missing_open_instruct_requires_explicit_or_vendored_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv("OPEN_INSTRUCT_REPO_PATH", raising=False)
    monkeypatch.setattr(ifbench, "_VENDORED_OPEN_INSTRUCT_REPO", tmp_path / "missing-open-instruct")

    with pytest.raises(ImportError, match="runtime cloning and pip installation are disabled"):
        ifbench._resolve_open_instruct_repo()


def test_explicit_verifier_checkout_is_loaded_without_installing(monkeypatch, tmp_path):
    repo_path = tmp_path / "IFBench"
    repo_path.mkdir()
    (repo_path / "evaluation_lib.py").write_text("SENTINEL = 'loaded'\n", encoding="utf-8")
    monkeypatch.setenv("IFBENCH_REPO_PATH", str(repo_path))

    ifbench._load_evaluation_lib.cache_clear()
    sys.modules.pop(ifbench._EVALUATION_MODULE_NAME, None)
    try:
        evaluation_lib = ifbench._load_evaluation_lib()
        assert evaluation_lib.SENTINEL == "loaded"
    finally:
        ifbench._load_evaluation_lib.cache_clear()
        sys.modules.pop(ifbench._EVALUATION_MODULE_NAME, None)


def test_legacy_language_reward_is_repeatable_after_loader_stabilization(monkeypatch, tmp_path):
    repo_path = tmp_path / "open-instruct"
    verifier_path = repo_path / "open_instruct" / "if_functions.py"

    class FakeDetectorFactory:
        seed = None

    class FakeLangdetect:
        DetectorFactory = FakeDetectorFactory
        calls = 0

        @classmethod
        def detect(cls, _text):
            cls.calls += 1
            if cls.DetectorFactory.seed == 0:
                return "en"
            return "en" if cls.calls % 2 else "fr"

    verifier_module = SimpleNamespace(
        __file__=str(verifier_path),
        IF_FUNCTIONS_MAP={
            "validate_response_language": lambda response, language: FakeLangdetect.detect(response) == language,
        },
    )

    def fake_import_module(module_name):
        if module_name == "langdetect":
            return FakeLangdetect
        if module_name == "open_instruct.if_functions":
            assert FakeDetectorFactory.seed == 0
            return verifier_module
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(ifbench, "_resolve_open_instruct_repo", lambda: repo_path)
    monkeypatch.setattr(ifbench.importlib, "import_module", fake_import_module)
    ifbench._load_open_instruct_module.cache_clear()
    try:
        label = {"func_name": "validate_response_language", "language": "en"}
        rewards = [ifbench.compute_ifeval_old_reward("A short English response.", label) for _ in range(20)]
        assert rewards == [1.0] * 20
        assert FakeDetectorFactory.seed == ifbench._LANGDETECT_SEED == 0
    finally:
        ifbench._load_open_instruct_module.cache_clear()


def test_legacy_ifeval_reward_does_not_mutate_metadata(fake_open_instruct):
    metadata = {
        "ground_truth": {
            "func_name": "contains_keyword",
            "keyword": "alpha",
            "unused": None,
        }
    }
    original = deepcopy(metadata)

    assert ifbench.compute_ifeval_old_reward("<think>x</think>alpha", None, metadata) == 1.0
    assert metadata == original


def test_legacy_ifeval_accepts_json_label(fake_open_instruct):
    label = json.dumps({"func_name": "is_lowercase"})

    assert ifbench.compute_ifeval_old_reward("all lowercase", label) == 1.0


def test_legacy_ifeval_empty_answer_is_zero_without_calling_verifier(fake_open_instruct):
    label = {"func_name": "contains_keyword", "keyword": "alpha"}

    assert ifbench.compute_ifeval_old_reward("<think>unfinished</think>", label) == 0.0


def test_legacy_ifeval_rejects_new_schema(fake_open_instruct):
    label = {"instruction_id": ["contains:first"], "kwargs": [{"keyword": "alpha"}]}

    with pytest.raises(ValueError, match="use rm_type='ifevalg'"):
        ifbench.compute_ifeval_old_reward("alpha", label)


def test_ifevalg_uses_official_per_constraint_mean(fake_open_instruct):
    label = "[{'instruction_id': ['contains:first', 'contains:second'], " \
        "'kwargs': [{'keyword': 'alpha'}, {'keyword': 'beta'}]}]"

    assert ifbench.compute_ifevalg_reward("alpha only", label) == 0.5


def test_ifevalg_rejects_legacy_schema(fake_open_instruct):
    label = {"func_name": "contains_keyword", "keyword": "alpha"}

    with pytest.raises(ValueError, match="use rm_type='ifeval_old'"):
        ifbench.compute_ifevalg_reward("alpha", label)


@pytest.mark.parametrize(
    "label,error",
    [
        (
            {"instruction_id": ["contains:first", "contains:second"], "kwargs": [{"keyword": "alpha"}]},
            "mismatched instruction_id and kwargs lengths",
        ),
        (
            {"instruction_id": ["contains:first"], "kwargs": {"keyword": "alpha"}},
            "'kwargs' must be a list",
        ),
        (
            {"instruction_id": ["contains:first"], "kwargs": ["not-a-dict"]},
            "Each IFEvalG kwargs entry must be a dict or None",
        ),
    ],
)
def test_ifevalg_rejects_malformed_constraint_alignment(fake_open_instruct, label, error):
    with pytest.raises(ValueError, match=error):
        ifbench.compute_ifevalg_reward("alpha", label)


def test_ifevalg_accepts_positionally_aligned_none_kwargs(fake_open_instruct):
    label = {
        "instruction_id": ["no_kwargs:false", "contains:second"],
        "kwargs": [None, {"keyword": "alpha"}],
    }

    assert ifbench.compute_ifevalg_reward("alpha", label) == 0.5


@pytest.mark.parametrize(
    "rm_type,expected",
    [
        ("ifbench", 0.0),
        ("ifbench_per_constraint", 0.5),
    ],
)
def test_reward_dispatch_selects_aggregation(fake_evaluation_lib, rm_type, expected):
    args = MagicMock(custom_rm_path=None, rm_type=rm_type, rm_url=None)
    sample = Sample(
        prompt="test prompt",
        response="alpha only",
        metadata=_metadata("alpha", "beta"),
    )

    assert run(async_rm(args, sample)) == expected


@pytest.mark.parametrize(
    "rm_type,label,expected",
    [
        ("ifeval_old", {"func_name": "contains_keyword", "keyword": "alpha"}, 1.0),
        (
            "ifevalg",
            [
                {
                    "instruction_id": ["contains:first", "contains:second"],
                    "kwargs": [{"keyword": "alpha"}, {"keyword": "beta"}],
                }
            ],
            0.5,
        ),
    ],
)
def test_ifeval_schema_dispatch(fake_open_instruct, rm_type, label, expected):
    args = MagicMock(custom_rm_path=None, rm_type=rm_type, rm_url=None)
    sample = Sample(prompt="test prompt", response="alpha only", label=label)

    assert run(async_rm(args, sample)) == expected
