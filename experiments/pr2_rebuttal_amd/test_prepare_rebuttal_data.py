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


def test_if_multi_requires_exact_source_and_contract_keys() -> None:
    row = {
        "key": "row-1",
        "messages": [{"role": "user", "content": "do it"}],
        "ground_truth": "[{'instruction_id':['a:b'],'kwargs':[{}]}]",
        "dataset": "ifeval",
        "constraint_type": "one",
        "constraint": "one constraint",
        "unexpected": "must fail",
    }
    with pytest.raises(prep.ValidationError, match="exact source keys"):
        prep._convert_if_multi([row], _source("if_multi_fallback_train"))


def test_if_multi_prompt_contracts_are_deterministically_consolidated() -> None:
    base = {
        "messages": [{"role": "user", "content": "same prompt"}],
        "dataset": "ifeval",
        "constraint_type": "multi",
        "constraint": "constraints",
    }
    rows = [
        {
            **base,
            "key": "one",
            "ground_truth": "[{'instruction_id':['a:b'],'kwargs':[{'N':1}]}]",
        },
        {
            **base,
            "key": "two",
            "ground_truth": "[{'instruction_id':['a:b'],'kwargs':[{'N':1}]}]",
        },
        {
            **base,
            "key": "three",
            "ground_truth": "[{'instruction_id':['c:d','a:b'],'kwargs':[{}, {'N':1}]}]",
        },
    ]
    converted = prep._convert_if_multi(rows, _source("if_multi_fallback_train"))
    consolidated, metrics = prep._consolidate_if_multi_prompts(converted, "if_multi")
    assert len(consolidated) == 1
    assert json.loads(consolidated[0]["label"]) == [
        {"instruction_id": ["a:b", "c:d"], "kwargs": [{"N": 1}, {}]}
    ]
    assert consolidated[0]["metadata"]["source_keys"] == ["one", "two", "three"]
    assert metrics == {
        "prompt_duplicate_rows_removed": 2,
        "prompt_identical_contract_duplicates_removed": 1,
        "prompt_contract_consolidation_groups": 1,
        "constraints_deduplicated_during_consolidation": 1,
    }


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


def test_new_schema_key_type_is_lossless_and_typed_key_is_unique() -> None:
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
    assert output[0]["metadata"]["source_key"] == 1
    assert type(output[0]["metadata"]["source_key"]) is int
    assert output[1]["metadata"]["source_key"] == "1"
    assert type(output[1]["metadata"]["source_key"]) is str

    with pytest.raises(prep.ValidationError, match="duplicate typed key"):
        prep._convert_new_ifeval(
            [rows[0], {**rows[0], "prompt": "different"}],
            _source("google_ifeval"),
            rm_type="ifevalg",
            verifier_schema=prep.GOOGLE_IFEVAL_SCHEMA,
        )


def test_exact_raw_deduplication_does_not_expand_rollouts() -> None:
    row = {"prompt": "one", "label": "x"}
    unique, duplicate_count = prep._deduplicate_exact_raw([row, dict(row)], "sample", 2)
    assert unique == [row]
    assert duplicate_count == 1


def test_prompt_leakage_fails_closed() -> None:
    with pytest.raises(prep.ValidationError, match="prompt leakage"):
        prep._check_disjoint({"train": {"same"}, "heldout": {"same"}})


def test_nfc_strip_leakage_fails_but_casefold_is_diagnostic() -> None:
    with pytest.raises(prep.ValidationError, match="prompt leakage"):
        prep._check_prompt_leakage_views({"train": {"Cafe\u0301 "}, "eval": {"Caf\u00e9"}})

    exact, normalized, casefold = prep._check_prompt_leakage_views(
        {"train": {"Mixed Case"}, "eval": {"mixed case"}}
    )
    assert next(iter(exact.values())) == 0
    assert next(iter(normalized.values())) == 0
    assert next(iter(casefold.values())) == 1


def test_create_guards_compute_allocation_and_output_root(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SPUR_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(prep.ValidationError, match="requires SPUR_JOB_ID"):
        prep._require_compute_allocation()
    with pytest.raises(prep.ValidationError, match="output root must be a child"):
        prep._require_allowed_output_root(tmp_path / "bundle")


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


def test_gsm8k_heldout_preserves_verl_reward_and_provenance() -> None:
    rows = [{"question": "1+1?", "answer": "Reasoning\n#### 2"}]
    output = prep._convert_gsm8k(rows, _source("gsm8k_test"))
    assert output[0]["label"] == "2"
    assert output[0]["metadata"]["rm_type"] == "gsm8k_verl"
    assert output[0]["metadata"]["source_revision"] == prep.GSM8K_REVISION
    assert output[0]["prompt"][0]["content"].endswith(prep.GSM8K_PROMPT_SUFFIX)


def test_eval_configs_are_disjoint_and_sampling_is_explicit(tmp_path) -> None:
    protocols = prep._eval_protocols(tmp_path)
    math = {item["name"]: item for item in protocols["math"]["datasets"]}
    nonmath = {item["name"]: item for item in protocols["nonmath"]["datasets"]}
    assert set(math) == {"gsm8k", "math500"}
    assert set(nonmath) == {"google_ifeval", "ifbench_test"}
    assert set(math).isdisjoint(nonmath)
    assert (math["gsm8k"]["rm_type"], math["gsm8k"]["n_samples_per_eval_prompt"]) == (
        "gsm8k_verl",
        1,
    )
    assert (math["math500"]["rm_type"], math["math500"]["n_samples_per_eval_prompt"]) == (
        "math",
        4,
    )
    assert math["gsm8k"]["temperature"] == 0.0
    assert math["math500"]["temperature"] == 1.0
    assert all(item["top_p"] == 1.0 for item in math.values())
    assert all(item["max_response_len"] == 1024 for item in math.values())
    assert all(item["max_response_len"] == 2048 for item in nonmath.values())
    assert all(
        "top_p" in item and "top_k" in item
        for item in list(math.values()) + list(nonmath.values())
    )
    for item in list(math.values()) + list(nonmath.values()):
        assert item["metadata_overrides"] == {
            prep.EVAL_DATASET_METADATA_KEY: item["name"]
        }
        assert item["rm_type"] == prep.EVAL_DATASET_RM_TYPES[item["name"]]

    math_text = prep._eval_config_text(tmp_path, "math")
    nonmath_text = prep._eval_config_text(tmp_path, "nonmath")
    assert "name: gsm8k" in math_text and "name: math500" in math_text
    assert "name: google_ifeval" not in math_text
    assert "name: google_ifeval" in nonmath_text and "name: ifbench_test" in nonmath_text
    assert "name: gsm8k" not in nonmath_text
    for name in ("gsm8k", "math500"):
        assert (
            "      metadata_overrides:\n"
            f"        rebuttal_eval_dataset: {name}\n"
        ) in math_text
    for name in ("google_ifeval", "ifbench_test"):
        assert (
            "      metadata_overrides:\n"
            f"        rebuttal_eval_dataset: {name}\n"
        ) in nonmath_text


def test_eval_dataset_reward_route_allowlist_fails_closed(tmp_path) -> None:
    original = prep.EVAL_DATASET_RM_TYPES["gsm8k"]
    prep.EVAL_DATASET_RM_TYPES["gsm8k"] = "wrong"
    try:
        with pytest.raises(prep.ValidationError, match="frozen eval dataset/reward route"):
            prep._eval_protocols(tmp_path)
    finally:
        prep.EVAL_DATASET_RM_TYPES["gsm8k"] = original
