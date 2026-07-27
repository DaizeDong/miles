#!/usr/bin/env python3
"""Validate prepared verifier contracts against the pinned runtime registries."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


OUTPUT_FILES = {
    "old": "if_train_rlvr_ifeval_old.jsonl",
    "if_multi": "if_multi_fallback_train.jsonl",
    "google": "ifeval_google_heldout.jsonl",
    "ifbench": "ifbench_test_heldout.jsonl",
    "math": "math500_test.jsonl",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _rows(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number}: row must be an object")
            yield row


def _module_identity(name: str, target: Path) -> dict[str, str]:
    module = importlib.import_module(name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"runtime module {name} has no filesystem identity")
    resolved = Path(module_file).resolve()
    if name in {"ray", "sglang"} and resolved.is_relative_to(target):
        raise RuntimeError(f"verifier target illegally shadows fixed-image module {name}: {resolved}")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if name == "miles":
        origin = "repository:miles/__init__.py"
        version = "repository-source"
    else:
        origin = str(resolved)
        version = importlib.metadata.version(name)
    return {"module": name, "origin": origin, "version": version, "sha256": digest}


async def _miles_reward_smoke(row: dict[str, Any]) -> float:
    from miles.rollout.rm_hub import async_rm
    from miles.utils.types import Sample

    sample = Sample(
        prompt=copy.deepcopy(row["prompt"]),
        response="smoke response",
        label=row["label"],
        metadata=copy.deepcopy(row["metadata"]),
    )
    value = await async_rm(SimpleNamespace(custom_rm_path=None, rm_type=""), sample)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Miles reward returned a non-numeric value: {value!r}")
    return float(value)


def _parse_ifevalg(label: str) -> tuple[list[str], list[dict[str, Any] | None]]:
    payload = json.loads(label)
    if isinstance(payload, list):
        if len(payload) != 1 or not isinstance(payload[0], dict):
            raise RuntimeError("IFEvalG list payload must contain exactly one object")
        payload = payload[0]
        expected_keys = {"instruction_id", "kwargs"}
        id_key = "instruction_id"
    else:
        if not isinstance(payload, dict):
            raise RuntimeError("IFEvalG payload must be an object")
        expected_keys = {"instruction_id_list", "kwargs"}
        id_key = "instruction_id_list"
    if set(payload) != expected_keys:
        raise RuntimeError(f"IFEvalG payload keys mismatch: {sorted(payload)}")
    ids = payload[id_key]
    kwargs = payload["kwargs"]
    if not isinstance(ids, list) or not ids or not isinstance(kwargs, list) or len(ids) != len(kwargs):
        raise RuntimeError("IFEvalG instruction/kwargs alignment mismatch")
    return ids, kwargs


def _official_strict_loose_smoke(
    evaluation_lib: Any,
    row: dict[str, Any],
    instruction_id: str,
    raw_kwargs: dict[str, Any] | None,
) -> None:
    """Execute the same two official functions used by the final eval hook."""

    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("official evaluation smoke requires prepared metadata")
    record_id = metadata.get("record_id")
    prompt = metadata.get("prompt_text")
    if isinstance(record_id, bool) or not isinstance(record_id, (str, int)):
        raise RuntimeError("official evaluation smoke has an invalid record_id")
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError("official evaluation smoke has an invalid prompt_text")
    if raw_kwargs is not None and not isinstance(raw_kwargs, dict):
        raise RuntimeError("official evaluation smoke kwargs must be an object or null")
    kwargs = (
        {}
        if raw_kwargs is None
        else {key: copy.deepcopy(value) for key, value in raw_kwargs.items() if value is not None}
    )

    for mode in ("strict", "loose"):
        input_example = evaluation_lib.InputExample(
            key=copy.deepcopy(record_id),
            instruction_id_list=[instruction_id],
            prompt=prompt,
            kwargs=[copy.deepcopy(kwargs)],
        )
        function = getattr(evaluation_lib, f"test_instruction_following_{mode}")
        output = function(input_example, {prompt: "smoke response"})
        decisions = list(output.follow_instruction_list)
        if (
            list(output.instruction_id_list) != [instruction_id]
            or len(decisions) != 1
            or bool(output.follow_all_instructions) != all(bool(value) for value in decisions)
        ):
            raise RuntimeError(
                f"official {mode} evaluator output contract mismatch for {instruction_id!r}"
            )


def validate(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    target = (bundle / "third_party/python").resolve()
    if not sys_path_starts_with(target):
        raise RuntimeError(f"final PYTHONPATH must prepend verifier target {target}")

    runtime_modules = [_module_identity(name, target) for name in ("miles", "ray", "sglang")]
    from miles.rollout.rm_hub import ifbench as miles_if

    old_registry = miles_if._load_open_instruct_module("open_instruct.if_functions").IF_FUNCTIONS_MAP
    ifevalg_registry = miles_if._load_open_instruct_module(
        "open_instruct.IFEvalG.instructions_registry"
    ).INSTRUCTION_DICT
    evaluation_lib = miles_if._load_evaluation_lib()
    ifbench_registry_module = getattr(evaluation_lib, "instructions_registry", None)
    if ifbench_registry_module is None or not hasattr(ifbench_registry_module, "INSTRUCTION_DICT"):
        raise RuntimeError("pinned IFBench evaluator does not expose instructions_registry.INSTRUCTION_DICT")
    ifbench_registry = ifbench_registry_module.INSTRUCTION_DICT
    google_code = Path(os.environ["GOOGLE_IFEVAL_OFFICIAL_CODE"]).resolve()
    google_package_root = google_code.parent
    sys.path.insert(0, str(google_package_root))
    google_registry_module = importlib.import_module(
        "instruction_following_eval.instructions_registry"
    )
    google_module_path = Path(google_registry_module.__file__).resolve()
    if not google_module_path.is_relative_to(google_code):
        raise RuntimeError(
            f"official Google IFEval registry loaded from {google_module_path}, expected {google_code}"
        )
    google_registry = google_registry_module.INSTRUCTION_DICT
    google_evaluation_lib = importlib.import_module(
        "instruction_following_eval.evaluation_lib"
    )
    google_evaluation_path = Path(google_evaluation_lib.__file__).resolve()
    if not google_evaluation_path.is_relative_to(google_code):
        raise RuntimeError(
            f"official Google IFEval evaluator loaded from {google_evaluation_path}, expected {google_code}"
        )

    unique_old: dict[str, dict[str, Any]] = {}
    old_contracts = 0
    for row in _rows(bundle / OUTPUT_FILES["old"]):
        payload = json.loads(row["label"])
        if not isinstance(payload, dict) or not isinstance(payload.get("func_name"), str):
            raise RuntimeError("legacy IFEval payload is malformed")
        func_name = payload["func_name"]
        if func_name not in old_registry:
            raise RuntimeError(f"unknown legacy verifier {func_name!r}")
        kwargs = {key: value for key, value in payload.items() if key != "func_name" and value is not None}
        inspect.signature(old_registry[func_name]).bind("smoke response", **kwargs)
        unique_old.setdefault(func_name, row)
        old_contracts += 1

    unique_ifevalg: dict[str, tuple[dict[str, Any], dict[str, Any] | None]] = {}
    unique_google_official: dict[str, tuple[dict[str, Any], dict[str, Any] | None]] = {}
    ifevalg_contracts = 0
    google_official_contracts = 0
    for filename in (OUTPUT_FILES["if_multi"], OUTPUT_FILES["google"]):
        for row in _rows(bundle / filename):
            instruction_ids, kwargs_list = _parse_ifevalg(row["label"])
            for instruction_id, raw_kwargs in zip(instruction_ids, kwargs_list, strict=True):
                if instruction_id not in ifevalg_registry:
                    raise RuntimeError(f"unknown IFEvalG verifier {instruction_id!r}")
                kwargs = (
                    {}
                    if raw_kwargs is None
                    else {key: value for key, value in raw_kwargs.items() if value is not None}
                )
                instruction = ifevalg_registry[instruction_id](instruction_id)
                inspect.signature(instruction.build_description).bind(**kwargs)
                instruction.build_description(**copy.deepcopy(kwargs))
                unique_ifevalg.setdefault(instruction_id, (row, raw_kwargs))
                ifevalg_contracts += 1
                if filename == OUTPUT_FILES["google"]:
                    if instruction_id not in google_registry:
                        raise RuntimeError(f"unknown official Google IFEval verifier {instruction_id!r}")
                    google_instruction = google_registry[instruction_id](instruction_id)
                    inspect.signature(google_instruction.build_description).bind(**kwargs)
                    google_instruction.build_description(**copy.deepcopy(kwargs))
                    unique_google_official.setdefault(instruction_id, (row, raw_kwargs))
                    google_official_contracts += 1

    unique_ifbench: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    ifbench_contracts = 0
    for row in _rows(bundle / OUTPUT_FILES["ifbench"]):
        instruction_ids, kwargs_list = _parse_ifevalg(row["label"])
        for instruction_id, raw_kwargs in zip(instruction_ids, kwargs_list, strict=True):
            if instruction_id not in ifbench_registry:
                raise RuntimeError(f"unknown IFBench verifier {instruction_id!r}")
            if raw_kwargs is None:
                raise RuntimeError(f"IFBench verifier {instruction_id!r} received null kwargs")
            kwargs = {key: value for key, value in raw_kwargs.items() if value is not None}
            instruction = ifbench_registry[instruction_id](instruction_id)
            inspect.signature(instruction.build_description).bind(**kwargs)
            instruction.build_description(**copy.deepcopy(kwargs))
            unique_ifbench.setdefault(instruction_id, (row, raw_kwargs))
            ifbench_contracts += 1

    smoke_calls = 0
    for row in unique_old.values():
        asyncio.run(_miles_reward_smoke(row))
        smoke_calls += 1
    for instruction_id, (row, raw_kwargs) in unique_ifevalg.items():
        smoke_row = copy.deepcopy(row)
        smoke_payload = [{"instruction_id": [instruction_id], "kwargs": [raw_kwargs]}]
        smoke_row["label"] = _canonical(smoke_payload)
        smoke_row["metadata"]["ground_truth"] = smoke_row["label"]
        smoke_row["metadata"]["rm_type"] = "ifevalg"
        asyncio.run(_miles_reward_smoke(smoke_row))
        smoke_calls += 1
    for instruction_id, (row, raw_kwargs) in unique_ifbench.items():
        smoke_row = copy.deepcopy(row)
        smoke_row["label"] = _canonical(
            {"instruction_id_list": [instruction_id], "kwargs": [raw_kwargs]}
        )
        smoke_row["metadata"]["instruction_id_list"] = [instruction_id]
        smoke_row["metadata"]["kwargs"] = [raw_kwargs]
        smoke_row["metadata"]["rm_type"] = "ifbench"
        asyncio.run(_miles_reward_smoke(smoke_row))
        smoke_calls += 1
    for instruction_id, (row, raw_kwargs) in unique_google_official.items():
        _official_strict_loose_smoke(
            google_evaluation_lib, row, instruction_id, raw_kwargs
        )
    for instruction_id, (row, raw_kwargs) in unique_ifbench.items():
        _official_strict_loose_smoke(evaluation_lib, row, instruction_id, raw_kwargs)
    math_row = next(_rows(bundle / OUTPUT_FILES["math"]), None)
    if math_row is None:
        raise RuntimeError("MATH-500 data is empty")
    asyncio.run(_miles_reward_smoke(math_row))
    smoke_calls += 1

    return {
        "runtime_modules": runtime_modules,
        "legacy_contracts_validated": old_contracts,
        "ifevalg_contracts_validated": ifevalg_contracts,
        "google_official_contracts_validated": google_official_contracts,
        "ifbench_contracts_validated": ifbench_contracts,
        "unique_legacy_verifiers": sorted(unique_old),
        "unique_ifevalg_verifiers": sorted(unique_ifevalg),
        "unique_google_official_verifiers": sorted(unique_google_official),
        "unique_ifbench_verifiers": sorted(unique_ifbench),
        "miles_reward_smoke_calls": smoke_calls,
        "google_official_strict_loose_smoke_pairs": len(unique_google_official),
        "ifbench_official_strict_loose_smoke_pairs": len(unique_ifbench),
        "result": "PASS",
    }


def sys_path_starts_with(target: Path) -> bool:
    configured = os.environ.get("PYTHONPATH", "")
    first = configured.split(os.pathsep, 1)[0]
    return bool(first) and Path(first).resolve() == target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    report = validate(parser.parse_args().bundle)
    print("RUNTIME_CONTRACT_REPORT_JSON=" + _canonical(report), flush=True)


if __name__ == "__main__":
    main()
