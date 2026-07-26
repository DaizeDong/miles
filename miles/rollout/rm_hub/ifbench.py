from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import logging
import os
import sys
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_IFBENCH_REPO_ENV = "IFBENCH_REPO_PATH"
_VENDORED_IFBENCH_REPO = _WORKSPACE_ROOT / "third_party" / "IFBench"
_EVALUATION_MODULE_NAME = "_miles_ifbench_evaluation_lib"
_OPEN_INSTRUCT_REPO_ENV = "OPEN_INSTRUCT_REPO_PATH"
_VENDORED_OPEN_INSTRUCT_REPO = _WORKSPACE_ROOT / "third_party" / "open-instruct"
_LANGDETECT_SEED = 0


def _resolve_ifbench_repo() -> Path:
    """Resolve a pre-provisioned IFBench checkout without network side effects."""

    configured_path = os.environ.get(_IFBENCH_REPO_ENV)
    repo_path = Path(configured_path).expanduser() if configured_path else _VENDORED_IFBENCH_REPO
    evaluation_path = repo_path / "evaluation_lib.py"
    if not evaluation_path.is_file():
        raise ImportError(
            "IFBench is not provisioned. Set "
            f"{_IFBENCH_REPO_ENV} to a pinned IFBench checkout, or vendor it at "
            f"{_VENDORED_IFBENCH_REPO}. Expected {evaluation_path}. "
            "Install its dependencies before starting Miles workers; runtime cloning and pip installation are disabled."
        )
    return repo_path.resolve()


def _resolve_open_instruct_repo() -> Path:
    """Resolve a checkout containing the two official IFEval train verifiers."""

    configured_path = os.environ.get(_OPEN_INSTRUCT_REPO_ENV)
    repo_path = Path(configured_path).expanduser() if configured_path else _VENDORED_OPEN_INSTRUCT_REPO
    required_paths = (
        repo_path / "open_instruct" / "if_functions.py",
        repo_path / "open_instruct" / "IFEvalG" / "instructions_registry.py",
    )
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        raise ImportError(
            "Open Instruct's IFEval verifiers are not provisioned. Set "
            f"{_OPEN_INSTRUCT_REPO_ENV} to a pinned open-instruct checkout, or vendor it at "
            f"{_VENDORED_OPEN_INSTRUCT_REPO}. Missing {missing_paths}. "
            "Install its verifier dependencies before starting Miles workers; runtime cloning and pip installation "
            "are disabled."
        )
    return repo_path.resolve()


def _stabilize_langdetect() -> None:
    """Make language-constraint rewards repeatable across worker processes.

    The pinned Open Instruct verifiers import ``langdetect`` but do not set its
    factory seed. Setting it before importing the verifier modules is a local
    reproducibility stabilization; it does not modify the upstream checkout.
    """

    try:
        langdetect = importlib.import_module("langdetect")
    except Exception as exc:
        raise ImportError(
            "Open Instruct's IFEval verifiers require langdetect. Install the pinned verifier dependencies "
            "before starting Miles workers."
        ) from exc

    detector_factory = getattr(langdetect, "DetectorFactory", None)
    if detector_factory is None:
        raise ImportError("The installed langdetect package does not expose DetectorFactory.")
    detector_factory.seed = _LANGDETECT_SEED


@lru_cache(maxsize=1)
def _load_evaluation_lib():
    """Load the official evaluator lazily from a pre-provisioned checkout."""

    repo_path = _resolve_ifbench_repo()
    evaluation_path = repo_path / "evaluation_lib.py"
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        # IFBench's evaluation module imports its sibling modules by their
        # top-level names, so its checkout must precede site packages.
        sys.path.insert(0, repo_str)

    spec = importlib.util.spec_from_file_location(_EVALUATION_MODULE_NAME, evaluation_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create an import spec for {evaluation_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_EVALUATION_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(_EVALUATION_MODULE_NAME, None)
        raise ImportError(
            f"Unable to import the pinned IFBench evaluator from {evaluation_path}. "
            "Install the checkout's requirements before starting Miles workers."
        ) from exc
    return module


@lru_cache(maxsize=None)
def _load_open_instruct_module(module_name: str):
    """Import an allow-listed verifier module from the configured checkout."""

    allowed_modules = {
        "open_instruct.if_functions",
        "open_instruct.IFEvalG.instructions_registry",
    }
    if module_name not in allowed_modules:
        raise ValueError(f"Unsupported Open Instruct verifier module {module_name!r}.")

    repo_path = _resolve_open_instruct_repo()
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    _stabilize_langdetect()

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ImportError(
            f"Unable to import {module_name} from the pinned checkout at {repo_path}. "
            "Install its verifier dependencies before starting Miles workers."
        ) from exc

    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(repo_path):
        raise ImportError(
            f"Imported {module_name} from {module_path}, not the configured checkout {repo_path}. "
            "Start workers in a clean Python process with the pinned checkout first on sys.path."
        )
    return module


JsonDict = dict[str, Any]
KwargsDict = dict[str, str | int | float | None]


def _normalize_instruction_ids(raw_ids: Sequence[Any]) -> list[str]:
    """Ensure instruction identifiers are clean strings."""

    normalized: list[str] = []
    for entry in raw_ids or []:
        if entry is None:
            continue
        text = str(entry).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized


def _coerce_kwargs_list(
    raw_kwargs: Any,
    num_instructions: int,
) -> list[KwargsDict]:
    """Convert stored kwargs into the list structure expected by IFBench."""

    if isinstance(raw_kwargs, list):
        processed: list[KwargsDict] = []
        for entry in raw_kwargs:
            if isinstance(entry, dict):
                processed.append(dict(entry))
            else:
                processed.append({})
    elif isinstance(raw_kwargs, dict):
        processed = [dict(raw_kwargs) for _ in range(num_instructions)]
    else:
        processed = [{} for _ in range(num_instructions)]

    if len(processed) < num_instructions:
        tail = processed[-1] if processed else {}
        processed.extend([dict(tail) for _ in range(num_instructions - len(processed))])
    elif len(processed) > num_instructions:
        processed = processed[:num_instructions]

    # Remove explicit None values to match official preprocessing.
    sanitized: list[KwargsDict] = []
    for entry in processed:
        sanitized.append({k: v for k, v in entry.items() if v is not None})
    return sanitized


def _build_input_example(metadata: JsonDict, evaluation_lib=None):
    instruction_ids = _normalize_instruction_ids(metadata.get("instruction_id_list") or [])
    if not instruction_ids:
        logger.debug("Missing instruction identifiers in metadata: %s", metadata)
        return None

    prompt_text = metadata.get("prompt_text")
    if prompt_text is None:
        prompt_text = ""
    else:
        prompt_text = str(prompt_text)

    raw_kwargs = metadata.get("kwargs")
    kwargs_list = _coerce_kwargs_list(raw_kwargs, len(instruction_ids))

    if evaluation_lib is None:
        evaluation_lib = _load_evaluation_lib()

    return evaluation_lib.InputExample(
        key=int(metadata.get("record_id") or 0),
        instruction_id_list=instruction_ids,
        prompt=prompt_text,
        kwargs=kwargs_list,
    )


def compute_ifbench_reward(
    response: str,
    label: Any,
    metadata: JsonDict | None = None,
    *,
    aggregation: str = "strict",
) -> float:
    """Score a response using official IFBench checks.

    ``strict`` preserves the historical binary all-constraints-pass reward.
    ``per_constraint_mean`` provides a denser training reward while using the
    exact same deterministic constraint decisions.
    """

    if aggregation not in {"strict", "per_constraint_mean"}:
        raise ValueError(
            f"Unsupported IFBench reward aggregation {aggregation!r}; "
            "expected 'strict' or 'per_constraint_mean'."
        )

    if metadata is None:
        logger.debug("No metadata provided for IFBench scoring.")
        return 0.0

    if response is None:
        return 0.0

    evaluation_lib = _load_evaluation_lib()
    inp = _build_input_example(metadata, evaluation_lib=evaluation_lib)
    if inp is None:
        return 0.0

    prompt_to_response = {inp.prompt: str(response or "")}
    output = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    if aggregation == "strict":
        return 1.0 if output.follow_all_instructions else 0.0

    decisions = list(output.follow_instruction_list)
    if not decisions:
        return 0.0
    if len(decisions) != len(inp.instruction_id_list):
        raise ValueError(
            "IFBench returned a different number of constraint decisions "
            f"({len(decisions)}) than instruction ids ({len(inp.instruction_id_list)})."
        )
    return sum(bool(decision) for decision in decisions) / len(decisions)


def _remove_thinking_section(response: str) -> str:
    """Match Open Instruct's preprocessing before IFEval verification."""

    answer = response.replace("<|assistant|>", "").strip()
    answer = answer.split("</think>")[-1]
    answer = answer.replace("<answer>", "").replace("</answer>", "")
    return answer.strip()


def _constraint_payload(label: Any, metadata: JsonDict | None) -> Any:
    if metadata is not None and metadata.get("ground_truth") is not None:
        return metadata["ground_truth"]
    return label


def _parse_serialized_constraint(payload: Any) -> Any:
    """Parse JSON or the Python-literal strings used by official IF-RLVR data."""

    if not isinstance(payload, str):
        return payload

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(payload)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("Unable to parse serialized IFEval ground_truth.") from exc


def compute_ifeval_old_reward(
    response: str,
    label: Any,
    metadata: JsonDict | None = None,
) -> float:
    """Score the legacy ``func_name`` schema used by ``RLVR-IFeval``.

    The official upstream verifier mutates its constraint with ``pop``. This
    implementation deliberately copies it so shared Sample metadata remains
    stable across repeated reward calls.
    """

    if response is None:
        return 0.0
    payload = _parse_serialized_constraint(_constraint_payload(label, metadata))
    if payload is None:
        logger.debug("No ground_truth provided for legacy IFEval scoring.")
        return 0.0
    if not isinstance(payload, dict):
        raise ValueError("Legacy IFEval ground_truth must be a dict with a 'func_name' key.")
    if "func_name" not in payload:
        if "instruction_id" in payload or "instruction_id_list" in payload:
            raise ValueError("Received the IFEvalG schema in ifeval_old; use rm_type='ifevalg'.")
        raise ValueError("Legacy IFEval ground_truth is missing 'func_name'.")

    constraint = dict(payload)
    func_name = constraint["func_name"]
    verifier_module = _load_open_instruct_module("open_instruct.if_functions")
    try:
        verifier = verifier_module.IF_FUNCTIONS_MAP[func_name]
    except KeyError as exc:
        raise ValueError(f"Unknown legacy IFEval verifier function {func_name!r}.") from exc

    kwargs = {key: value for key, value in constraint.items() if key != "func_name" and value is not None}
    answer = _remove_thinking_section(str(response))
    if not answer:
        return 0.0
    return float(verifier(answer, **kwargs))


def compute_ifevalg_reward(
    response: str,
    label: Any,
    metadata: JsonDict | None = None,
) -> float:
    """Score the ``instruction_id``/``kwargs`` schema used by IF-RLVR v2.

    This reproduces Open Instruct's official per-constraint-mean reward and is
    intentionally distinct from both legacy IFEval and IFBench OOD scoring.
    """

    if response is None:
        return 0.0
    payload = _parse_serialized_constraint(_constraint_payload(label, metadata))
    if payload is None:
        logger.debug("No ground_truth provided for IFEvalG scoring.")
        return 0.0
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(f"IFEvalG ground_truth must contain exactly one constraint dict; got {len(payload)}.")
        payload = _parse_serialized_constraint(payload[0])
    if not isinstance(payload, dict):
        raise ValueError("IFEvalG ground_truth must be a dict or one-element list of dicts.")
    if "func_name" in payload:
        raise ValueError("Received the legacy func_name schema in ifevalg; use rm_type='ifeval_old'.")

    instruction_ids = payload.get("instruction_id")
    if instruction_ids is None:
        instruction_ids = payload.get("instruction_id_list")
    if not isinstance(instruction_ids, list) or not instruction_ids:
        raise ValueError("IFEvalG ground_truth is missing a non-empty instruction_id list.")
    if any(not isinstance(instruction_id, str) or not instruction_id for instruction_id in instruction_ids):
        raise ValueError("IFEvalG instruction_id entries must be non-empty strings.")

    raw_kwargs = payload.get("kwargs")
    if not isinstance(raw_kwargs, list):
        raise ValueError("IFEvalG ground_truth 'kwargs' must be a list aligned with instruction_id.")
    if len(raw_kwargs) != len(instruction_ids):
        raise ValueError(
            "IFEvalG ground_truth has mismatched instruction_id and kwargs lengths: "
            f"{len(instruction_ids)} != {len(raw_kwargs)}."
        )
    if any(entry is not None and not isinstance(entry, dict) for entry in raw_kwargs):
        raise ValueError("Each IFEvalG kwargs entry must be a dict or None.")
    kwargs_list = [
        {} if entry is None else {key: value for key, value in entry.items() if value is not None}
        for entry in raw_kwargs
    ]

    registry = _load_open_instruct_module("open_instruct.IFEvalG.instructions_registry")
    answer = _remove_thinking_section(str(response))
    decisions: list[bool] = []
    for instruction_id, kwargs in zip(instruction_ids, kwargs_list, strict=True):
        try:
            instruction_cls = registry.INSTRUCTION_DICT[instruction_id]
        except KeyError as exc:
            raise ValueError(f"Unknown IFEvalG instruction id {instruction_id!r}.") from exc
        instruction = instruction_cls(instruction_id)
        instruction.build_description(**kwargs)
        decisions.append(bool(answer and instruction.check_following(answer)))

    return sum(decisions) / len(decisions)
