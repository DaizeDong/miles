"""Fast, network-free contract tests for the rebuttal data builder."""

from __future__ import annotations

import json

import pytest

from experiments.pr2_rebuttal_amd import prepare_rebuttal_data as prep


def _source(name: str) -> prep.FileSource:
    return next(source for source in prep.FILE_SOURCES if source.name == name)


def test_old_ifeval_remains_func_name_schema() -> None:
    rows = [
        {
            "messages": [{"role": "user", "content": "answer in lowercase"}],
            "ground_truth": '{"func_name":"validate_lowercase","N":null}',
            "dataset": "ifeval",
            "constraint_type": "All Lowercase",
            "constraint": "answer in lowercase",
        }
    ]
    output = prep._convert_old_ifeval(rows, _source("rlvr_ifeval_train"))
    assert output[0]["metadata"]["rm_type"] == "ifeval_old"
    assert output[0]["metadata"]["verifier_schema"] == prep.OLD_IFEVAL_SCHEMA
    assert json.loads(output[0]["label"])["func_name"] == "validate_lowercase"


def test_old_ifeval_rejects_new_schema() -> None:
    rows = [
        {
            "messages": [{"role": "user", "content": "x"}],
            "ground_truth": '{"instruction_id_list":["a:b"],"kwargs":[{}]}',
            "dataset": "ifeval",
            "constraint_type": "x",
            "constraint": "x",
        }
    ]
    with pytest.raises(prep.ValidationError, match="func_name"):
        prep._convert_old_ifeval(rows, _source("rlvr_ifeval_train"))


def test_if_multi_preserves_one_element_list_and_null_kwargs() -> None:
    rows = [
        {
            "key": "row-1",
            "messages": [{"role": "user", "content": "do both"}],
            "ground_truth": "[{'instruction_id':['a:b','c:d'],'kwargs':[None,{'N':2}]}]",
            "dataset": "ifeval",
            "constraint_type": "multi",
            "constraint": "two constraints",
        }
    ]
    output = prep._convert_if_multi(rows, _source("if_multi_fallback_train"))
    payload = json.loads(output[0]["label"])
    assert output[0]["metadata"]["rm_type"] == "ifevalg"
    assert output[0]["metadata"]["candidate_status"] == "preregistered_fallback_only"
    assert payload == [{"instruction_id": ["a:b", "c:d"], "kwargs": [None, {"N": 2}]}]


def test_google_and_ifbench_routes_are_distinct() -> None:
    row = {"key": 1, "prompt": "x", "instruction_id_list": ["a:b"], "kwargs": [{}]}
    google = prep._convert_new_ifeval(
        [row],
        _source("google_ifeval"),
        rm_type="ifevalg",
        verifier_schema=prep.GOOGLE_IFEVAL_SCHEMA,
    )
    ifbench = prep._convert_new_ifeval(
        [row],
        _source("ifbench_test"),
        rm_type="ifbench",
        verifier_schema=prep.IFBENCH_SCHEMA,
    )
    assert google[0]["metadata"]["rm_type"] == "ifevalg"
    assert ifbench[0]["metadata"]["rm_type"] == "ifbench"
    assert google[0]["metadata"]["verifier_schema"] != ifbench[0]["metadata"]["verifier_schema"]


def test_exact_raw_deduplication_does_not_expand_rollouts() -> None:
    row = {"prompt": "one", "label": "x"}
    unique, duplicate_count = prep._deduplicate_exact_raw([row, dict(row)], "sample", 2)
    assert unique == [row]
    assert duplicate_count == 1


def test_prompt_leakage_fails_closed() -> None:
    with pytest.raises(prep.ValidationError, match="prompt leakage"):
        prep._check_disjoint({"train": {"same"}, "heldout": {"same"}})


def test_math500_uses_boxed_answer_contract() -> None:
    rows = [
        {
            "problem": "1+1?",
            "solution": "2",
            "answer": "2",
            "subject": "Algebra",
            "level": 1,
            "unique_id": "test/algebra/one.json",
        }
    ]
    output = prep._convert_math500(rows, _source("math500"))
    assert "\\boxed{$Answer}" in output[0]["prompt"][0]["content"]
    assert output[0]["metadata"]["rm_type"] == "math"


def test_eval_config_keeps_official_benchmarks_separate(tmp_path) -> None:
    text = prep._eval_config_text(tmp_path)
    assert "name: google_ifeval" in text and "rm_type: ifevalg" in text
    assert "name: ifbench_test" in text and "rm_type: ifbench" in text
    assert "if_multi_fallback_train" not in text
