"""Frozen, dataset-aware evaluation metrics for PR2 rebuttal runs.

Miles' default pass@k logger groups every dataset with the global
``args.n_samples_per_eval_prompt``.  The rebuttal math protocol mixes GSM8K
at n=1 and MATH500 at n=4, so metrics must instead use each resolved
``EvalDatasetConfig``.  The non-math protocol additionally reports the four
official strict/loose IFEval aggregates from the pinned Google/IFBench
evaluators; the online scalar reward remains a clearly named diagnostic.
"""

from __future__ import annotations

import importlib
import hashlib
import json
import logging
import math
import os
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

_DATASET_RM_TYPES = {
    "gsm8k": "gsm8k_verl",
    "math500": "math",
    "google_ifeval": "ifevalg",
    "ifbench_test": "ifbench",
}
_PROTOCOL_DATASET_SETS = {
    frozenset(("gsm8k", "math500")): "math",
    frozenset(("google_ifeval", "ifbench_test")): "nonmath",
}


def dataset_reward_metrics(rewards: Iterable[float], group_size: int) -> dict[str, float]:
    """Return sample-average accuracy and unbiased pass@{1,2,4,...}."""

    raw_values = list(rewards)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_values):
        raise TypeError("evaluation rewards must be numeric and non-boolean")
    values = [float(value) for value in raw_values]
    if not values:
        raise ValueError("evaluation rewards must not be empty")
    if group_size <= 0 or len(values) % group_size:
        raise ValueError(
            f"reward count {len(values)} is not divisible by dataset group_size={group_size}"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("evaluation rewards must be finite")
    if any(value not in (0.0, 1.0) for value in values):
        raise ValueError("formal rule-based evaluation rewards must be binary 0/1")

    metrics = {"avg@1": sum(values) / len(values)}
    if group_size == 1:
        return metrics

    groups = [values[index : index + group_size] for index in range(0, len(values), group_size)]
    for k in (2**power for power in range(int(math.log2(group_size)) + 1)):
        estimates = []
        for group in groups:
            correct = sum(value == 1.0 for value in group)
            if group_size - correct < k:
                estimate = 1.0
            else:
                estimate = 1.0 - math.comb(group_size - correct, k) / math.comb(group_size, k)
            estimates.append(estimate)
        metrics[f"pass@{k}"] = sum(estimates) / len(estimates)
    return metrics


def instruction_following_metrics(
    strict_outputs: Sequence[Any], loose_outputs: Sequence[Any]
) -> dict[str, float]:
    """Aggregate the official prompt/instruction strict/loose decisions."""

    if not strict_outputs or len(strict_outputs) != len(loose_outputs):
        raise ValueError("strict/loose official output lists must be non-empty and aligned")

    totals: dict[str, tuple[int, int]] = {}
    for mode, outputs in (("strict", strict_outputs), ("loose", loose_outputs)):
        prompt_correct = 0
        instruction_correct = 0
        instruction_total = 0
        for output in outputs:
            decisions = list(output.follow_instruction_list)
            instruction_ids = list(output.instruction_id_list)
            if not decisions or len(decisions) != len(instruction_ids):
                raise ValueError(f"official {mode} output has invalid instruction cardinality")
            prompt_correct += int(bool(output.follow_all_instructions))
            instruction_correct += sum(bool(value) for value in decisions)
            instruction_total += len(decisions)
        totals[mode] = (prompt_correct, instruction_correct, instruction_total)

    result: dict[str, float] = {}
    prompt_total = len(strict_outputs)
    for mode, (prompt_correct, instruction_correct, instruction_total) in totals.items():
        result[f"prompt_{mode}"] = prompt_correct / prompt_total
        result[f"instruction_{mode}"] = instruction_correct / instruction_total
    return result


def _metadata_input(evaluation_lib: Any, metadata: Any) -> Any:
    if not isinstance(metadata, dict):
        raise TypeError("official instruction-following evaluation requires sample metadata")
    required = ("record_id", "prompt_text", "instruction_id_list", "kwargs")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"official instruction-following metadata is missing {missing}")
    instruction_ids = metadata["instruction_id_list"]
    kwargs = metadata["kwargs"]
    if (
        not isinstance(instruction_ids, list)
        or not instruction_ids
        or any(not isinstance(value, str) or not value for value in instruction_ids)
        or not isinstance(kwargs, list)
        or len(kwargs) != len(instruction_ids)
        or any(not isinstance(value, dict) for value in kwargs)
    ):
        raise ValueError("official instruction-following metadata contract is malformed")
    return evaluation_lib.InputExample(
        key=metadata["record_id"],
        instruction_id_list=list(instruction_ids),
        prompt=str(metadata["prompt_text"]),
        # Parquet list-of-struct columns materialize unused union fields as
        # explicit nulls.  IFBench strict drops them but loose does not; use
        # the effective official kwargs for both modes.
        kwargs=[
            {key: item for key, item in value.items() if item is not None}
            for value in kwargs
        ],
    )


def _load_google_evaluation_lib() -> Any:
    code_path = os.environ.get("GOOGLE_IFEVAL_OFFICIAL_CODE")
    if not code_path:
        raise ImportError("GOOGLE_IFEVAL_OFFICIAL_CODE is required for official Google IFEval metrics")
    package_dir = Path(code_path).resolve()
    required = package_dir / "evaluation_lib.py"
    if not required.is_file() or package_dir.name != "instruction_following_eval":
        raise ImportError(f"invalid pinned Google IFEval evaluator path: {package_dir}")
    parent = str(package_dir.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    evaluation_lib = importlib.import_module("instruction_following_eval.evaluation_lib")
    if Path(evaluation_lib.__file__).resolve() != required.resolve():
        raise ImportError(
            "cached Google IFEval evaluator does not match GOOGLE_IFEVAL_OFFICIAL_CODE: "
            f"{evaluation_lib.__file__}"
        )
    return evaluation_lib


def _official_outputs(samples: Sequence[Any], rm_type: str) -> tuple[list[Any], list[Any]]:
    if rm_type == "ifevalg":
        evaluation_lib = _load_google_evaluation_lib()
    elif rm_type == "ifbench":
        # Miles' loader resolves exactly IFBENCH_REPO_PATH and validates the
        # pinned checkout before importing its top-level evaluation_lib.
        from miles.rollout.rm_hub.ifbench import _load_evaluation_lib

        evaluation_lib = _load_evaluation_lib()
        expected_path = Path(os.environ.get("IFBENCH_REPO_PATH", "")) / "evaluation_lib.py"
        if not expected_path.is_file() or Path(evaluation_lib.__file__).resolve() != expected_path.resolve():
            raise ImportError("loaded IFBench evaluator does not match IFBENCH_REPO_PATH")
    else:
        raise ValueError(f"unsupported official instruction-following rm_type={rm_type!r}")

    strict_outputs = []
    loose_outputs = []
    for sample in samples:
        strict_input = _metadata_input(evaluation_lib, sample.metadata)
        loose_input = _metadata_input(evaluation_lib, sample.metadata)
        response = str(sample.response or "")
        strict_outputs.append(
            evaluation_lib.test_instruction_following_strict(
                strict_input, {strict_input.prompt: response}
            )
        )
        loose_outputs.append(
            evaluation_lib.test_instruction_following_loose(
                loose_input, {loose_input.prompt: response}
            )
        )
        for mode, output in (("strict", strict_outputs[-1]), ("loose", loose_outputs[-1])):
            decisions = list(output.follow_instruction_list)
            if (
                list(output.instruction_id_list) != list(strict_input.instruction_id_list)
                or bool(output.follow_all_instructions) != all(bool(value) for value in decisions)
            ):
                raise ValueError(f"official {mode} evaluator output contract mismatch")
    return strict_outputs, loose_outputs


def _dataset_rm_type(samples: Sequence[Any]) -> str:
    rm_types = {
        sample.metadata.get("rm_type")
        for sample in samples
        if isinstance(getattr(sample, "metadata", None), dict)
    }
    if len(rm_types) != 1:
        raise ValueError(f"evaluation dataset has mixed or missing metadata.rm_type: {sorted(map(str, rm_types))}")
    rm_type = next(iter(rm_types))
    if not isinstance(rm_type, str):
        raise TypeError("evaluation metadata.rm_type must be a string")
    return rm_type


def _sample_record_id(sample: Any) -> Any:
    metadata = sample.metadata
    for key in ("record_id", "unique_id", "source_row_index"):
        if key in metadata:
            value = metadata[key]
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                return (key, type(value).__name__, value)
            raise TypeError(f"sample metadata.{key} must be a string or integer")
    raise ValueError("sample metadata lacks record_id/unique_id/source_row_index")


def _load_manifest_contract(args: Any, configs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path_text = os.environ.get("REBUTTAL_DATA_MANIFEST")
    expected_manifest_sha = os.environ.get("EXPECTED_REBUTTAL_DATA_MANIFEST_SHA256")
    expected_config_sha = os.environ.get("EXPECTED_EVAL_CONFIG_SHA256")
    if not manifest_path_text or not expected_manifest_sha or not expected_config_sha:
        raise ValueError("formal eval requires locked data-manifest and eval-config SHA environment")
    manifest_path = Path(manifest_path_text).resolve()
    if _sha256(manifest_path) != expected_manifest_sha:
        raise ValueError("rebuttal data manifest changed before eval postprocessing")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    config_path = Path(args.eval_config).resolve() if getattr(args, "eval_config", None) else None
    if config_path is None or _sha256(config_path) != expected_config_sha:
        raise ValueError("resolved eval config is missing or differs from its locked SHA-256")
    config_names = frozenset(configs)
    try:
        protocol_name = _PROTOCOL_DATASET_SETS[config_names]
        protocol = manifest["eval_protocols"][protocol_name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unregistered rebuttal eval dataset set: {sorted(config_names)}") from exc
    if (
        Path(protocol["config_path"]).name != config_path.name
        or protocol.get("config_sha256") != expected_config_sha
        or set(protocol.get("dataset_set", [])) != set(configs)
    ):
        raise ValueError("manifest eval protocol/config identity mismatch")

    specs = {spec["name"]: spec for spec in protocol.get("datasets", [])}
    if set(specs) != set(configs):
        raise ValueError("manifest eval protocol dataset specs mismatch")
    for name, config in configs.items():
        spec = specs[name]
        expected_rm_type = _DATASET_RM_TYPES[name]
        expected_fields = {
            "rm_type": expected_rm_type,
            "n_samples_per_eval_prompt": config.n_samples_per_eval_prompt,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_response_len": config.max_response_len,
        }
        for field, expected in expected_fields.items():
            if spec.get(field) != expected:
                raise ValueError(f"manifest/config mismatch for {name}.{field}")
        if config.rm_type != expected_rm_type:
            raise ValueError(f"config {name} rm_type={config.rm_type!r}, expected={expected_rm_type!r}")
        if config.metadata_overrides != {"rebuttal_eval_dataset": name}:
            raise ValueError(f"config {name} lacks the exact rebuttal_eval_dataset metadata tag")
        data_path = Path(config.path).resolve()
        if data_path != Path(spec["path"]).resolve() or _sha256(data_path) != spec["artifact_sha256"]:
            raise ValueError(f"eval dataset artifact identity mismatch for {name}")
        with data_path.open("rb") as stream:
            rows = sum(1 for line in stream if line.strip())
        if rows != spec["artifact_rows"]:
            raise ValueError(f"eval dataset row count mismatch for {name}: {rows}")

    verifier_records = manifest.get("verifier_code", {})
    verifier_tree_sha256 = {
        name: record.get("tree_sha256")
        for name, record in verifier_records.items()
        if isinstance(record, dict)
    }
    required_evaluators = {
        "ifbench_evaluator",
        "open_instruct_verifiers",
        "google_ifeval_official_evaluator",
    }
    if set(verifier_tree_sha256) != required_evaluators or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in verifier_tree_sha256.values()
    ):
        raise ValueError("data manifest lacks the exact pinned evaluator tree SHA-256 set")
    bundle_root = manifest_path.parent
    for name in sorted(required_evaluators):
        destination = verifier_records[name].get("destination")
        if not isinstance(destination, str) or not destination or Path(destination).is_absolute():
            raise ValueError(f"invalid evaluator destination for {name}")
        evaluator_root = (bundle_root / destination).resolve()
        if not evaluator_root.is_relative_to(bundle_root) or not evaluator_root.is_dir():
            raise ValueError(f"evaluator destination escapes/is missing for {name}")
        actual_tree_sha = _tree_sha256(evaluator_root)
        if actual_tree_sha != verifier_tree_sha256[name]:
            raise ValueError(f"runtime evaluator tree SHA-256 mismatch for {name}")
    math_reward_path = Path(__file__).resolve().with_name("math_reward.py")
    math_reward_sha256 = _sha256(math_reward_path)
    provenance = {
        "data_manifest_path": str(manifest_path),
        "data_manifest_sha256": expected_manifest_sha,
        "eval_config_path": str(config_path),
        "eval_config_sha256": expected_config_sha,
        "eval_logger_sha256": _sha256(Path(__file__).resolve()),
        "math_reward_sha256": math_reward_sha256,
        "evaluator_tree_sha256": verifier_tree_sha256,
    }
    if provenance["eval_logger_sha256"] != os.environ.get("EXPECTED_EVAL_LOGGER_SHA256"):
        raise ValueError("executed eval logger snapshot differs from its formal SHA lock")
    if math_reward_sha256 != os.environ.get("EXPECTED_MATH_REWARD_SHA256"):
        raise ValueError("per-run math reward snapshot differs from its formal SHA lock")
    return specs, provenance


def _load_saved_eval_samples(
    args: Any,
    rollout_id: int,
    configs: dict[str, Any],
    specs: dict[str, Any],
) -> tuple[dict[str, list[Any]], dict[str, str]]:
    path_template = getattr(args, "save_debug_rollout_data", None)
    if not path_template:
        raise ValueError("formal eval requires saved raw rollout data")
    raw_path = Path(path_template.format(rollout_id=f"eval_{rollout_id}")).resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"saved eval rollout is missing: {raw_path}")

    import torch

    from miles.utils.types import Sample

    payload = torch.load(raw_path, weights_only=False)
    if not isinstance(payload, dict) or payload.get("rollout_id") != rollout_id:
        raise ValueError("saved eval rollout payload/rollout_id mismatch")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("saved eval rollout has no flattened sample list")
    samples = [Sample.from_dict(value) for value in raw_samples]

    grouped = _validate_and_group_saved_samples(samples, configs, specs)

    return grouped, {
        "raw_eval_path": str(raw_path),
        "raw_eval_sha256": _sha256(raw_path),
    }


def _validate_and_group_saved_samples(
    samples: Sequence[Any], configs: dict[str, Any], specs: dict[str, Any]
) -> dict[str, list[Any]]:
    """Reconstruct flattened debug samples using the frozen metadata tag."""

    grouped: dict[str, list[Any]] = {name: [] for name in configs}
    for sample in samples:
        if not isinstance(sample.metadata, dict):
            raise TypeError("saved eval sample metadata must be a mapping")
        name = sample.metadata.get("rebuttal_eval_dataset")
        rm_type = sample.metadata.get("rm_type")
        if name not in grouped or _DATASET_RM_TYPES.get(name) != rm_type:
            raise ValueError(f"saved eval sample has invalid dataset/rm_type pair: {name!r}/{rm_type!r}")
        grouped[name].append(sample)

    for name, group in grouped.items():
        n = configs[name].n_samples_per_eval_prompt
        expected_rows = specs[name]["artifact_rows"]
        expected_samples = expected_rows * n
        if len(group) != expected_samples:
            raise ValueError(
                f"saved eval {name} samples={len(group)}, expected rows({expected_rows})*n({n})={expected_samples}"
            )
        indexes = [sample.index for sample in group]
        if any(not isinstance(value, int) or isinstance(value, bool) for value in indexes):
            raise TypeError(f"saved eval {name} contains a non-integer sample index")
        if len(set(indexes)) != len(indexes) or set(indexes) != set(range(expected_samples)):
            raise ValueError(f"saved eval {name} sample indices are not exactly 0..{expected_samples - 1}")
        record_counts = Counter(_sample_record_id(sample) for sample in group)
        if len(record_counts) != expected_rows or set(record_counts.values()) != {n}:
            raise ValueError(f"saved eval {name} record identities do not occur exactly n={n} times")
        group.sort(key=lambda sample: sample.index)
    return grouped


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    entries = list(root.rglob("*"))
    for path in entries:
        if path.is_symlink():
            raise ValueError(f"symbolic links are forbidden in frozen evaluator trees: {path}")
    files = sorted(
        path
        for path in entries
        if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_eval_artifacts(
    rollout_id: int,
    aggregate: dict[str, dict[str, float | int | str]],
    official_records: dict[str, list[dict[str, Any]]],
    provenance: dict[str, Any],
) -> dict[str, str]:
    """Persist aggregate JSON plus official per-record decisions once."""

    tensorboard_dir = os.environ.get("TENSORBOARD_DIR")
    if not tensorboard_dir:
        raise ValueError("TENSORBOARD_DIR is required to anchor formal eval artifacts")
    save_root = Path(tensorboard_dir).resolve().parent
    if Path(tensorboard_dir).resolve() != save_root / "tensorboard":
        raise ValueError("formal TENSORBOARD_DIR must be the save-root tensorboard directory")
    artifact_dir = save_root / "eval_artifacts" / f"rollout-{rollout_id}"
    artifact_dir.mkdir(parents=True, exist_ok=False)

    def write_atomic(path: Path, text: str) -> None:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite eval artifact: {path}")
        os.replace(temporary, path)

    outputs: dict[str, str] = {}
    aggregate_path = artifact_dir / "aggregate.json"
    write_atomic(
        aggregate_path,
        json.dumps(
            {
                "schema_version": 1,
                "rollout_id": rollout_id,
                "provenance": provenance,
                "datasets": aggregate,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    outputs[aggregate_path.name] = _sha256(aggregate_path)

    for name, records in sorted(official_records.items()):
        record_path = artifact_dir / f"{name}.official.jsonl"
        write_atomic(
            record_path,
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            ),
        )
        outputs[record_path.name] = _sha256(record_path)

    checksums_path = artifact_dir / "sha256.json"
    write_atomic(checksums_path, json.dumps(outputs, indent=2, sort_keys=True) + "\n")
    outputs[checksums_path.name] = _sha256(checksums_path)
    for path in artifact_dir.iterdir():
        path.chmod(0o444)
    artifact_dir.chmod(0o555)
    return outputs


def log_eval_rollout_data(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log per-dataset math pass@k and official non-math metrics."""

    # Imports stay inside the runtime hook so pure aggregation remains
    # login-safe and independently unit-testable.
    from miles.ray.rollout.metrics import _compute_metrics_from_samples
    from miles.utils import tracking_utils
    from miles.utils.metric_utils import compute_rollout_step, dict_add_prefix

    configs = {config.name: config for config in args.eval_datasets}
    if len(configs) != len(args.eval_datasets):
        raise ValueError("eval dataset names must be unique")
    if set(data) != set(configs):
        raise ValueError(
            f"eval result/config dataset mismatch: results={sorted(data)}, configs={sorted(configs)}"
        )

    specs, artifact_provenance = _load_manifest_contract(args, configs)
    saved_groups, raw_provenance = _load_saved_eval_samples(
        args, rollout_id, configs, specs
    )
    artifact_provenance.update(raw_provenance)

    log_dict: dict[str, Any] = dict(extra_metrics or {})
    artifact_aggregate: dict[str, dict[str, float | int | str]] = {}
    official_records: dict[str, list[dict[str, Any]]] = {}
    for name, payload in data.items():
        rewards = payload["rewards"]
        online_samples = payload.get("samples")
        if not online_samples or len(online_samples) != len(rewards):
            raise ValueError(f"dataset {name} samples/rewards must be non-empty and aligned")
        samples = saved_groups[name]
        if len(samples) != len(online_samples):
            raise ValueError(f"dataset {name} saved/online sample cardinality mismatch")
        online_identity = [
            (sample.index, str(sample.response or ""), sample.reward)
            for sample in sorted(online_samples, key=lambda sample: sample.index)
        ]
        saved_identity = [
            (sample.index, str(sample.response or ""), sample.reward) for sample in samples
        ]
        if online_identity != saved_identity:
            raise ValueError(f"dataset {name} saved eval samples differ from online callback payload")
        group_size = configs[name].n_samples_per_eval_prompt
        if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
            raise ValueError(f"dataset {name} has invalid n_samples_per_eval_prompt={group_size!r}")

        reward_metrics = dataset_reward_metrics(rewards, group_size)
        rm_type = _dataset_rm_type(samples)
        if rm_type in {"ifevalg", "ifbench"}:
            if group_size != 1:
                raise ValueError(f"official instruction-following protocol requires n=1, got {group_size}")
            strict_outputs, loose_outputs = _official_outputs(samples, rm_type)
            official = instruction_following_metrics(strict_outputs, loose_outputs)
            # Prompt-level loose is the preregistered primary.  Preserve the
            # historical base tag, but make it equal to that official metric;
            # never alias the online scalar reward to an official score.
            log_dict[f"eval/{name}"] = official["prompt_loose"]
            for metric_name, value in official.items():
                log_dict[f"eval/{name}-{metric_name}"] = value
            online_reward = reward_metrics["avg@1"]
            log_dict[f"eval/{name}-online_reward"] = online_reward
            artifact_aggregate[name] = {
                "rm_type": rm_type,
                "n_samples_per_eval_prompt": group_size,
                "sample_count": len(samples),
                "row_count": len(samples) // group_size,
                "online_reward_diagnostic": online_reward,
                **official,
            }
            official_records[name] = [
                {
                    "record_id": sample.metadata["record_id"],
                    "sample_index": sample.index,
                    "sample_status": getattr(sample.status, "value", str(sample.status)),
                    "rm_type": rm_type,
                    "response": str(sample.response or ""),
                    "strict_follow_all": bool(strict.follow_all_instructions),
                    "strict_follow_instruction_list": [
                        bool(value) for value in strict.follow_instruction_list
                    ],
                    "loose_follow_all": bool(loose.follow_all_instructions),
                    "loose_follow_instruction_list": [
                        bool(value) for value in loose.follow_instruction_list
                    ],
                    "instruction_id_list": list(strict.instruction_id_list),
                }
                for sample, strict, loose in zip(
                    samples, strict_outputs, loose_outputs, strict=True
                )
            ]
        else:
            dataset_metrics = reward_metrics
            log_dict[f"eval/{name}"] = dataset_metrics["avg@1"]
            for metric_name, value in dataset_metrics.items():
                log_dict[f"eval/{name}-{metric_name}"] = value
            artifact_aggregate[name] = {
                "rm_type": rm_type,
                "n_samples_per_eval_prompt": group_size,
                "sample_count": len(samples),
                "row_count": len(samples) // group_size,
                **dataset_metrics,
            }
        log_dict[f"eval/{name}-n"] = group_size
        log_dict |= dict_add_prefix(_compute_metrics_from_samples(args, samples), f"eval/{name}/")

        if "truncated" in payload:
            truncated = payload["truncated"]
            if len(truncated) != len(rewards):
                raise ValueError(f"dataset {name} truncated/reward cardinality mismatch")
            log_dict[f"eval/{name}-truncated_ratio"] = sum(bool(value) for value in truncated) / len(truncated)

    artifact_hashes = _write_eval_artifacts(
        rollout_id, artifact_aggregate, official_records, artifact_provenance
    )
    log_dict["eval/artifacts_written"] = 1
    logger.info("dataset-aware eval artifacts %s: %s", rollout_id, artifact_hashes)
    logger.info("dataset-aware eval %s: %s", rollout_id, log_dict)
    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    tracking_utils.log(args, log_dict, step_key="eval/step")
    # Returning the full dict lets the rollout manager's metric checker see
    # the same metrics while bypassing the incorrect global-n fallback.
    return log_dict
