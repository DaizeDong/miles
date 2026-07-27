#!/usr/bin/env python3
"""Network-free stdlib regression tests for data-preparation hardening."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from unittest import mock

from experiments.pr2_rebuttal_amd import prepare_rebuttal_data as prep
from experiments.pr2_rebuttal_amd import validate_rebuttal_runtime as runtime_validator


def _source(name: str) -> prep.FileSource:
    return next(source for source in prep.FILE_SOURCES if source.name == name)


class DataHardeningTests(unittest.TestCase):
    def _if_multi_row(self, key: str, ground_truth: str) -> dict:
        return {
            "key": key,
            "messages": [{"role": "user", "content": "same prompt"}],
            "ground_truth": ground_truth,
            "dataset": "ifeval",
            "constraint_type": "multi",
            "constraint": "constraints",
        }

    def test_if_multi_consolidation_preserves_constraints(self) -> None:
        rows = [
            self._if_multi_row(
                "one", "[{'instruction_id':['a:b'],'kwargs':[{'N':1}]}]"
            ),
            self._if_multi_row(
                "two", "[{'instruction_id':['a:b'],'kwargs':[{'N':1}]}]"
            ),
            self._if_multi_row(
                "three", "[{'instruction_id':['c:d','a:b'],'kwargs':[{}, {'N':1}]}]"
            ),
        ]
        converted = prep._convert_if_multi(rows, _source("if_multi_fallback_train"))
        consolidated, metrics = prep._consolidate_if_multi_prompts(converted, "if_multi")
        self.assertEqual(len(consolidated), 1)
        self.assertEqual(
            json.loads(consolidated[0]["label"]),
            [{"instruction_id": ["a:b", "c:d"], "kwargs": [{"N": 1}, {}]}],
        )
        self.assertEqual(metrics["prompt_duplicate_rows_removed"], 2)
        self.assertEqual(metrics["prompt_identical_contract_duplicates_removed"], 1)
        self.assertEqual(metrics["prompt_contract_consolidation_groups"], 1)
        self.assertEqual(metrics["constraints_deduplicated_during_consolidation"], 1)

    def test_known_pinned_duplicate_at_21355_merges_both_verifiers(self) -> None:
        prompt = (
            "Creatively image a question and justification for this answer: (C) There should be "
            "2 paragraphs. Paragraphs are separated with the markdown divider: ***"
        )
        base = {
            "messages": [{"role": "user", "content": prompt}],
            "dataset": "ifeval",
            "constraint_type": "multi",
            "constraint": "paragraph constraints",
        }
        rows = [
            {
                **base,
                "key": "ai2-adapt-dev/flan_v2_converted_16928",
                "ground_truth": "[{'instruction_id': ['paragraphs:paragraphs'], 'kwargs': [{}]}]",
            },
            {
                **base,
                "key": "ai2-adapt-dev/flan_v2_converted_78050",
                "ground_truth": (
                    "[{'instruction_id': ['length_constraints:number_paragraphs'], "
                    "'kwargs': [{'num_paragraphs': 2}]}]"
                ),
            },
        ]
        converted = prep._convert_if_multi(rows, _source("if_multi_fallback_train"))
        consolidated, metrics = prep._consolidate_if_multi_prompts(converted, "if_multi")
        self.assertEqual(
            json.loads(consolidated[0]["label"]),
            [
                {
                    "instruction_id": [
                        "paragraphs:paragraphs",
                        "length_constraints:number_paragraphs",
                    ],
                    "kwargs": [{}, {"num_paragraphs": 2}],
                }
            ],
        )
        self.assertEqual(metrics["prompt_duplicate_rows_removed"], 1)
        self.assertEqual(metrics["prompt_contract_consolidation_groups"], 1)
        self.assertEqual(metrics["prompt_identical_contract_duplicates_removed"], 0)

    def test_if_multi_exact_schema(self) -> None:
        row = self._if_multi_row("one", "[{'instruction_id':['a:b'],'kwargs':[{}]}]")
        row["extra"] = True
        with self.assertRaisesRegex(prep.ValidationError, "exact source keys"):
            prep._convert_if_multi([row], _source("if_multi_fallback_train"))

    def test_if_multi_reused_source_labels_preserve_distinct_prompts(self) -> None:
        first = self._if_multi_row("reused", "[{'instruction_id':['a:b'],'kwargs':[{}]}]")
        second = self._if_multi_row("reused", "[{'instruction_id':['c:d'],'kwargs':[{}]}]")
        second["messages"] = [{"role": "user", "content": "different prompt"}]
        converted = prep._convert_if_multi(
            [first, second], _source("if_multi_fallback_train")
        )
        self.assertEqual(len(converted), 2)
        self.assertNotEqual(converted[0]["prompt"], converted[1]["prompt"])
        self.assertEqual(
            prep._source_key_duplicate_metrics([first, second], "if_multi"),
            {"source_key_duplicate_values": 1, "source_key_duplicate_rows": 1},
        )

    def test_new_key_type_is_preserved_and_unique(self) -> None:
        rows = [
            {"key": 1, "prompt": "one", "instruction_id_list": ["a:b"], "kwargs": [{}]},
            {"key": "1", "prompt": "two", "instruction_id_list": ["a:b"], "kwargs": [{}]},
        ]
        output = prep._convert_new_ifeval(
            rows,
            _source("google_ifeval"),
            rm_type="ifevalg",
            verifier_schema=prep.GOOGLE_IFEVAL_SCHEMA,
        )
        self.assertIs(type(output[0]["metadata"]["source_key"]), int)
        self.assertIs(type(output[1]["metadata"]["source_key"]), str)
        with self.assertRaisesRegex(prep.ValidationError, "duplicate typed key"):
            prep._convert_new_ifeval(
                [rows[0], {**rows[0], "prompt": "different"}],
                _source("google_ifeval"),
                rm_type="ifevalg",
                verifier_schema=prep.GOOGLE_IFEVAL_SCHEMA,
            )

    def test_normalized_leakage_is_hard_and_casefold_is_diagnostic(self) -> None:
        with self.assertRaisesRegex(prep.ValidationError, "prompt leakage"):
            prep._check_prompt_leakage_views(
                {"train": {"Cafe\u0301 "}, "eval": {"Caf\u00e9"}}
            )
        _, _, diagnostic = prep._check_prompt_leakage_views(
            {"train": {"Mixed Case"}, "eval": {"mixed case"}}
        )
        self.assertEqual(next(iter(diagnostic.values())), 1)

    def test_compute_and_output_guards(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(prep.ValidationError, "requires SPUR_JOB_ID"):
                prep._require_compute_allocation()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(prep.ValidationError, "output root must be a child"):
                prep._require_allowed_output_root(Path(temporary) / "bundle")

    def test_dependency_lock_is_missing_only_and_excludes_common_runtime(self) -> None:
        lock_path = Path(prep.__file__).with_name("verifier_requirements.lock")
        locked = prep._parse_requirements_lock(lock_path)
        self.assertEqual(
            set(locked), {"emoji", "immutabledict", "langdetect", "nltk", "syllapy"}
        )
        self.assertEqual(locked["nltk"], "3.9.4")
        hashes = prep._requirements_lock_hashes(lock_path)
        self.assertEqual(set(hashes), set(locked))
        self.assertTrue(all(values and all(len(value) == 64 for value in values) for values in hashes.values()))
        self.assertTrue(
            {"pydantic", "httpx", "anyio", "ray", "sglang", "torch"}.isdisjoint(locked)
        )

    def test_ifbench_broad_unused_requirements_are_source_proven(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            root = stage / "third_party/IFBench"
            root.mkdir(parents=True)
            (root / "requirements.txt").write_text(
                "absl-py\nspacy\nunicodedata2\n", encoding="utf-8"
            )
            for module in prep.IFBENCH_RUNTIME_MODULES:
                (root / module).write_text("import re\n", encoding="utf-8")
            report = prep._ifbench_unused_requirement_report(stage)
            self.assertEqual(
                report["declared_but_runtime_unused"], ["spacy", "unicodedata2"]
            )
            (root / "instructions.py").write_text("import spacy\n", encoding="utf-8")
            with self.assertRaisesRegex(prep.ValidationError, "excluded dependency spacy"):
                prep._ifbench_unused_requirement_report(stage)

    def test_official_strict_and_loose_runtime_smoke_uses_fresh_inputs(self) -> None:
        calls: list[tuple[str, int]] = []

        class FakeInput:
            def __init__(self, key, instruction_id_list, prompt, kwargs):
                self.key = key
                self.instruction_id_list = instruction_id_list
                self.prompt = prompt
                self.kwargs = kwargs

        class FakeEvaluationLib:
            InputExample = FakeInput

            @staticmethod
            def test_instruction_following_strict(input_example, responses):
                calls.append(("strict", id(input_example)))
                input_example.kwargs[0]["strict_mutation"] = True
                self.assertEqual(responses, {"prompt": "smoke response"})
                return SimpleNamespace(
                    instruction_id_list=["constraint:id"],
                    follow_instruction_list=[False],
                    follow_all_instructions=False,
                )

            @staticmethod
            def test_instruction_following_loose(input_example, responses):
                calls.append(("loose", id(input_example)))
                self.assertNotIn("strict_mutation", input_example.kwargs[0])
                self.assertEqual(responses, {"prompt": "smoke response"})
                return SimpleNamespace(
                    instruction_id_list=["constraint:id"],
                    follow_instruction_list=[True],
                    follow_all_instructions=True,
                )

        row = {"metadata": {"record_id": 7, "prompt_text": "prompt"}}
        runtime_validator._official_strict_loose_smoke(
            FakeEvaluationLib, row, "constraint:id", {"N": 1}
        )
        self.assertEqual([mode for mode, _ in calls], ["strict", "loose"])
        self.assertNotEqual(calls[0][1], calls[1][1])

    def test_nltk_runtime_and_data_supply_chain_are_explicitly_pinned(self) -> None:
        self.assertEqual(prep.EXPECTED_CONTAINER_PYTHON, (3, 10))
        self.assertRegex(prep.NLTK_DATA_REVISION, r"^[0-9a-f]{40}$")
        self.assertEqual(
            set(prep.NLTK_DATA_RESOURCES),
            {"punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger_eng"},
        )
        for name, spec in prep.NLTK_DATA_RESOURCES.items():
            self.assertRegex(spec["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(spec["bytes"], 0)
            url = prep._nltk_resource_url(name, spec)
            self.assertIn(prep.NLTK_DATA_REVISION, url)
            self.assertNotIn("gh-pages", url)
        source = Path(prep.__file__).read_text(encoding="utf-8")
        self.assertNotIn("nltk.downloader", source)

    def test_verify_only_also_requires_compute_allocation(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(prep.ValidationError, "requires SPUR_JOB_ID"):
                prep.verify_dataset_bundle(Path("/home/daidong/nonexistent-test-bundle"))

    def test_eval_protocols_are_disjoint_and_fully_explicit(self) -> None:
        root = Path("/home/daidong/test-bundle")
        self.assertEqual(prep.EVAL_CONFIG_FILES, ("eval_math.yaml", "eval_nonmath.yaml"))
        protocols = prep._eval_protocols(root)
        math = {item["name"]: item for item in protocols["math"]["datasets"]}
        nonmath = {item["name"]: item for item in protocols["nonmath"]["datasets"]}
        self.assertEqual(set(math), {"gsm8k", "math500"})
        self.assertEqual(set(nonmath), {"google_ifeval", "ifbench_test"})
        self.assertTrue(set(math).isdisjoint(nonmath))
        self.assertEqual((math["gsm8k"]["rm_type"], math["gsm8k"]["n_samples_per_eval_prompt"]), ("gsm8k_verl", 1))
        self.assertEqual((math["math500"]["rm_type"], math["math500"]["n_samples_per_eval_prompt"]), ("math", 4))
        self.assertEqual(math["gsm8k"]["temperature"], 0.0)
        self.assertEqual(math["math500"]["temperature"], 1.0)
        self.assertTrue(all(item["top_p"] == 1.0 for item in math.values()))
        self.assertTrue(all(item["max_response_len"] == 1024 for item in math.values()))
        self.assertTrue(all(item["max_response_len"] == 2048 for item in nonmath.values()))
        for item in list(math.values()) + list(nonmath.values()):
            self.assertTrue(
                {"n_samples_per_eval_prompt", "temperature", "top_p", "top_k", "max_response_len"}
                <= set(item)
            )
            self.assertEqual(
                item["metadata_overrides"],
                {prep.EVAL_DATASET_METADATA_KEY: item["name"]},
            )
            self.assertEqual(item["rm_type"], prep.EVAL_DATASET_RM_TYPES[item["name"]])

        for protocol_name in ("math", "nonmath"):
            text = prep._eval_config_text(root, protocol_name)
            for item in protocols[protocol_name]["datasets"]:
                self.assertIn(
                    "      metadata_overrides:\n"
                    f"        rebuttal_eval_dataset: {item['name']}\n",
                    text,
                )

    def test_eval_dataset_reward_route_allowlist_fails_closed(self) -> None:
        with mock.patch.dict(prep.EVAL_DATASET_RM_TYPES, {"gsm8k": "wrong"}, clear=False):
            with self.assertRaisesRegex(prep.ValidationError, "frozen eval dataset/reward route"):
                prep._eval_protocols(Path("/home/daidong/test-bundle"))

    def test_gsm8k_and_math500_reward_routes_are_distinct(self) -> None:
        gsm = prep._convert_gsm8k(
            [{"question": "1+1?", "answer": "Reasoning\n#### 2"}], _source("gsm8k_test")
        )
        math = prep._convert_math500(
            [
                {
                    "problem": "1+1?",
                    "solution": "2",
                    "answer": "2",
                    "subject": "Algebra",
                    "level": 1,
                    "unique_id": "test/algebra/one.json",
                }
            ],
            _source("math500"),
        )
        self.assertEqual(gsm[0]["metadata"]["rm_type"], "gsm8k_verl")
        self.assertEqual(math[0]["metadata"]["rm_type"], "math")
        self.assertEqual(gsm[0]["metadata"]["source_revision"], prep.GSM8K_REVISION)

    def test_sbatch_requires_frozen_source_identity(self) -> None:
        text = Path(prep.__file__).with_name("prepare_rebuttal_data.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("${MILES_ROOT:?", text)
        self.assertIn("${EXPECTED_CODE_SHA:?", text)
        self.assertIn("status --porcelain --untracked-files=all", text)
        self.assertNotIn("miles-pr1362-validation-fixed", text)

    def test_python_frozen_source_identity_is_recorded(self) -> None:
        expected = "a" * 40
        repo_root = Path(prep.__file__).resolve().parents[2]
        git_results = [
            CompletedProcess(args=[], returncode=0, stdout=expected + "\n"),
            CompletedProcess(args=[], returncode=0, stdout=""),
        ]
        with mock.patch.dict(
            os.environ,
            {"MILES_ROOT": str(repo_root), "EXPECTED_CODE_SHA": expected},
            clear=True,
        ), mock.patch.object(prep.subprocess, "run", side_effect=git_results):
            identity = prep._require_frozen_source()
        self.assertEqual(identity["actual_code_sha"], expected)
        self.assertTrue(identity["git_clean"])


if __name__ == "__main__":
    unittest.main()
