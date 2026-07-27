#!/usr/bin/env python3
"""Build the PR2 rebuttal datasets from immutable upstream revisions.

This program is intentionally compute-node oriented.  It downloads pinned raw
artifacts, preserves incompatible instruction-following verifier contracts,
writes one physical JSONL row per unique upstream record, validates exact
prompt disjointness, and publishes the result with an atomic directory rename.

Rollout multiplicity is *not* represented in these files.  Miles must create it
at runtime with ``--n-samples-per-prompt``.
"""

from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable


FORMAT_VERSION = 1
DEFAULT_OUTPUT_ROOT = Path("/home/daidong/rebuttal_workspace/data/pr2-rebuttal-one-copy-v1")
ALLOWED_OUTPUT_ROOT = Path("/home/daidong")
EXPECTED_CONTAINER_IMAGE = (
    "docker.io/rlsys/miles@sha256:e60a69faa831ae2a146819290e8469cf23c81b97febeaf5d11e8baec2bfca285"
)
EXPECTED_CONTAINER_PYTHON = (3, 10)
NLTK_DATA_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
NLTK_DATA_RESOURCES: dict[str, dict[str, Any]] = {
    "punkt": {
        "subdir": "tokenizers",
        "bytes": 13_905_355,
        "sha256": "51c3078994aeaf650bfc8e028be4fb42b4a0d177d41c012b6a983979653660ec",
    },
    "punkt_tab": {
        "subdir": "tokenizers",
        "bytes": 4_319_076,
        "sha256": "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106",
    },
    "stopwords": {
        "subdir": "corpora",
        "bytes": 37_733,
        "sha256": "48c0e52d8b52546e827f53761fb30300c0ab94f70660d28bd65ba0a86270946b",
    },
    "averaged_perceptron_tagger_eng": {
        "subdir": "taggers",
        "bytes": 1_539_115,
        "sha256": "6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b",
    },
}

RLVR_IFEVAL_REVISION = "47c03c73621c4aab2b824b7818681117d662770e"
IF_MULTI_REVISION = "2e3a77407b7fce69f95b248d64a884e3ae1c2423"
IFBENCH_TEST_REVISION = "2e8a48de45ff3bf41242f927254ca81b59ca3ae2"
GOOGLE_IFEVAL_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
IFBENCH_CODE_REVISION = "1091c4c3de6c1f6ed12c012ed68f11ea450b0117"
OPEN_INSTRUCT_REVISION = "5b2ebfa12381925bb431845d588dbc9ebead20a7"
GOOGLE_RESEARCH_REVISION = "ec7c3d346277b737bc2decffcd1b533d4b7ec105"

OLD_IFEVAL_SCHEMA = "open_instruct.IFEvalVerifierOld.func_name.v1"
IF_MULTI_SCHEMA = "open_instruct.IFEvalG.one_element_ground_truth_list.v1"
GOOGLE_IFEVAL_SCHEMA = "open_instruct.IFEvalG.instruction_id_list_kwargs.v1"
IFBENCH_SCHEMA = "ifbench.instruction_id_list_kwargs.v1"
MATH_SCHEMA = "miles.math.boxed_equivalence.v1"
GSM8K_SCHEMA = "verl.gsm8k.final_hash_answer.v1"

MATH_PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem."
)
MATH_PROMPT_SUFFIX = 'Remember to put your answer on its own line after "Answer:".'
GSM8K_PROMPT_SUFFIX = ' Let\'s think step by step and output the final answer after "####".'
GSM8K_ANSWER_PATTERN = re.compile(r"#### (\-?[0-9\.\,]+)")


@dataclasses.dataclass(frozen=True)
class FileSource:
    name: str
    repo_id: str
    revision: str
    relative_path: str
    expected_rows: int
    expected_size: int
    content_sha256: str | None = None
    git_blob_oid: str | None = None

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/datasets/{self.repo_id}/resolve/"
            f"{self.revision}/{self.relative_path}?download=true"
        )


@dataclasses.dataclass(frozen=True)
class CodeSource:
    name: str
    owner: str
    repository: str
    revision: str
    destination: str
    required_files: tuple[str, ...]

    @property
    def url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repository}/archive/{self.revision}.tar.gz"


@dataclasses.dataclass(frozen=True)
class GithubFile:
    relative_path: str
    expected_size: int
    git_blob_oid: str


@dataclasses.dataclass(frozen=True)
class GithubFileGroupSource:
    name: str
    owner: str
    repository: str
    revision: str
    destination: str
    files: tuple[GithubFile, ...]

    def url(self, relative_path: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.owner}/{self.repository}/"
            f"{self.revision}/{relative_path}"
        )


FILE_SOURCES: tuple[FileSource, ...] = (
    FileSource(
        name="rlvr_ifeval_train",
        repo_id="allenai/RLVR-IFeval",
        revision=RLVR_IFEVAL_REVISION,
        relative_path="data/train-00000-of-00001.parquet",
        expected_rows=14_973,
        expected_size=11_653_857,
        content_sha256="75f1b1f63039044034fe75d22f51f77f257a79291d0084262bb43695d4aac971",
    ),
    FileSource(
        name="if_multi_fallback_train",
        repo_id="allenai/IF_multi_constraints_upto5",
        revision=IF_MULTI_REVISION,
        relative_path="data/train-00000-of-00001.parquet",
        expected_rows=95_373,
        expected_size=72_198_499,
        content_sha256="edb8baed687cd85ce0602911f6c6cc8c8e657da69a7faba036082fa9618008ee",
    ),
    FileSource(
        name="ifbench_test",
        repo_id="allenai/IFBench_test",
        revision=IFBENCH_TEST_REVISION,
        relative_path="data/train-00000-of-00001.parquet",
        expected_rows=300,
        expected_size=94_446,
        content_sha256="80037e4d99c39a55c1e2e7d5a863d8d9edeb5ebe136ba8c1e849f8c015027c6a",
    ),
    FileSource(
        name="google_ifeval",
        repo_id="google/IFEval",
        revision=GOOGLE_IFEVAL_REVISION,
        relative_path="ifeval_input_data.jsonl",
        expected_rows=541,
        expected_size=207_111,
        git_blob_oid="7c875005c09550bc3fd94c7f42867f78490bc554",
    ),
    FileSource(
        name="gsm8k_test",
        repo_id="openai/gsm8k",
        revision=GSM8K_REVISION,
        relative_path="main/test-00000-of-00001.parquet",
        expected_rows=1_319,
        expected_size=419_088,
        content_sha256="ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
    ),
    FileSource(
        name="math500",
        repo_id="HuggingFaceH4/MATH-500",
        revision=MATH500_REVISION,
        relative_path="test.jsonl",
        expected_rows=500,
        expected_size=446_564,
        git_blob_oid="2376b9a194b46c0790e197c91b7249e5f88ac09b",
    ),
)

CODE_SOURCES: tuple[CodeSource, ...] = (
    CodeSource(
        name="ifbench_evaluator",
        owner="allenai",
        repository="IFBench",
        revision=IFBENCH_CODE_REVISION,
        destination="third_party/IFBench",
        required_files=("evaluation_lib.py", "requirements.txt"),
    ),
    CodeSource(
        name="open_instruct_verifiers",
        owner="allenai",
        repository="open-instruct",
        revision=OPEN_INSTRUCT_REVISION,
        destination="third_party/open-instruct",
        required_files=(
            "open_instruct/ground_truth_utils.py",
            "open_instruct/if_functions.py",
            "open_instruct/IFEvalG/instructions_registry.py",
        ),
    ),
)

GOOGLE_EVAL_SOURCE = GithubFileGroupSource(
    name="google_ifeval_official_evaluator",
    owner="google-research",
    repository="google-research",
    revision=GOOGLE_RESEARCH_REVISION,
    destination="third_party/google-research/instruction_following_eval",
    files=(
        GithubFile("instruction_following_eval/README.md", 1_457, "062d6166779d0a7ae83c3f7166ced351b0a9e97e"),
        GithubFile(
            "instruction_following_eval/evaluation_lib.py",
            6_984,
            "c87422822f9e1565ca7574ac0b3cee5389c9f44e",
        ),
        GithubFile(
            "instruction_following_eval/evaluation_main.py",
            2_449,
            "bea8f3dd18072fdc23e2f24e20c4d9599b20e06c",
        ),
        GithubFile(
            "instruction_following_eval/instructions.py",
            55_162,
            "0007359bd37c7b9c1d9557f8937cb6586d2788a7",
        ),
        GithubFile(
            "instruction_following_eval/instructions_registry.py",
            7_240,
            "6b42adcc4c8cc827810ab3efa8a06f02bb48c397",
        ),
        GithubFile(
            "instruction_following_eval/instructions_util.py",
            19_538,
            "d50702400738df4378fe286c0090a416f9a172fa",
        ),
        GithubFile(
            "instruction_following_eval/requirements.txt",
            35,
            "49a5a9136343f30d72c00df19dfb3557696697f9",
        ),
    ),
)

OUTPUT_FILES = {
    "rlvr_ifeval_train": "if_train_rlvr_ifeval_old.jsonl",
    "if_multi_fallback_train": "if_multi_fallback_train.jsonl",
    "google_ifeval": "ifeval_google_heldout.jsonl",
    "ifbench_test": "ifbench_test_heldout.jsonl",
    "gsm8k_test": "gsm8k_test.jsonl",
    "math500": "math500_test.jsonl",
}

EVAL_CONFIG_FILES = ("eval_math.yaml", "eval_nonmath.yaml")
EVAL_DATASET_RM_TYPES = {
    "gsm8k": "gsm8k_verl",
    "math500": "math",
    "google_ifeval": "ifevalg",
    "ifbench_test": "ifbench",
}
EVAL_DATASET_METADATA_KEY = "rebuttal_eval_dataset"
IFBENCH_RUNTIME_MODULES = (
    "evaluation_lib.py",
    "instructions_registry.py",
    "instructions.py",
    "instructions_util.py",
)
IFBENCH_DECLARED_BUT_RUNTIME_UNUSED = ("spacy", "unicodedata2")

IF_MULTI_SOURCE_KEYS = {
    "key",
    "messages",
    "ground_truth",
    "dataset",
    "constraint_type",
    "constraint",
}
NEW_IFEVAL_SOURCE_KEYS = {"key", "prompt", "instruction_id_list", "kwargs"}
RUNTIME_VALIDATOR = Path(__file__).with_name("validate_rebuttal_runtime.py")


class ValidationError(RuntimeError):
    """Raised when an upstream or prepared artifact violates its contract."""


def _require_allowed_output_root(output_root: Path) -> Path:
    resolved = output_root.resolve()
    allowed = ALLOWED_OUTPUT_ROOT.resolve()
    if resolved == allowed or not resolved.is_relative_to(allowed):
        raise ValidationError(
            f"output root must be a child of {allowed}; resolved output was {resolved}"
        )
    return resolved


def _require_compute_allocation() -> dict[str, str]:
    """Require scheduler evidence in the inner Python create path.

    The outer wrapper passes the scheduler variables through Docker.  Requiring
    both an allocation identifier and scheduler node metadata prevents an
    accidental create/download invocation on a login node merely by setting the
    container-image marker.
    """

    spur_job_id = os.environ.get("SPUR_JOB_ID", "").strip()
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    node_list = os.environ.get("SLURM_JOB_NODELIST", "").strip()
    spur_allocation = os.environ.get("SPUR_JOB_NODELIST", "").strip()
    if not (spur_job_id or slurm_job_id):
        raise ValidationError("create mode requires SPUR_JOB_ID or SLURM_JOB_ID")
    if not (node_list or spur_allocation):
        raise ValidationError(
            "create mode requires scheduler node metadata (SLURM_JOB_NODELIST or SPUR_JOB_NODELIST)"
        )
    return {
        "spur_job_id": spur_job_id,
        "slurm_job_id": slurm_job_id,
        "scheduler_node_list": node_list or spur_allocation,
    }


def _require_frozen_source() -> dict[str, Any]:
    miles_root_raw = os.environ.get("MILES_ROOT", "").strip()
    expected_sha = os.environ.get("EXPECTED_CODE_SHA", "").strip()
    if not miles_root_raw:
        raise ValidationError("MILES_ROOT is required for data preparation and verification")
    if re.fullmatch(r"[0-9a-f]{40}", expected_sha) is None:
        raise ValidationError("EXPECTED_CODE_SHA must be a lowercase 40-character Git SHA")
    miles_root = Path(miles_root_raw).resolve()
    generator_root = Path(__file__).resolve().parents[2]
    if miles_root != generator_root:
        raise ValidationError(
            f"data builder is running from {generator_root}, not declared MILES_ROOT {miles_root}"
        )
    if not (miles_root / ".git").exists():
        raise ValidationError(f"MILES_ROOT is not a Git worktree: {miles_root}")

    def git_output(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={miles_root}", "-C", str(miles_root), *arguments],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            raise ValidationError(
                f"Git identity check failed rc={result.returncode}: {' '.join(arguments)}: "
                f"{result.stdout.strip()}"
            )
        return result.stdout.strip()

    actual_sha = git_output("rev-parse", "HEAD")
    if actual_sha != expected_sha:
        raise ValidationError(f"MILES_ROOT HEAD={actual_sha}, expected={expected_sha}")
    dirty = git_output("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValidationError(f"MILES_ROOT must be clean; dirty entries: {dirty.splitlines()[:10]}")
    return {
        "miles_root": str(miles_root),
        "expected_code_sha": expected_sha,
        "actual_code_sha": actual_sha,
        "git_clean": True,
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_oid(path: Path) -> str:
    digest = hashlib.sha1()  # noqa: S324 - Git's object format requires SHA-1.
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    digest = hashlib.sha256()
    count = 0
    try:
        with temporary.open("wb") as stream:
            for row in rows:
                encoded = (_canonical(row) + "\n").encode("utf-8")
                stream.write(encoded)
                digest.update(encoded)
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"rows": count, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _download(url: str, destination: Path, attempts: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download-{os.getpid()}")
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "miles-pr2-rebuttal-data-prep/1.0"},
            )
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            return
        except (OSError, urllib.error.URLError) as exc:
            if temporary.exists():
                temporary.unlink()
            if attempt == attempts:
                raise RuntimeError(f"download failed after {attempts} attempts: {url}") from exc
            time.sleep(2 ** (attempt - 1))


def _validate_source_file(source: FileSource, path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size != source.expected_size:
        raise ValidationError(f"{source.name}: bytes={size}, expected={source.expected_size}")
    sha256 = _sha256(path)
    git_oid = _git_blob_oid(path)
    if source.content_sha256 is not None and sha256 != source.content_sha256:
        raise ValidationError(
            f"{source.name}: sha256={sha256}, expected immutable LFS oid={source.content_sha256}"
        )
    if source.git_blob_oid is not None and git_oid != source.git_blob_oid:
        raise ValidationError(f"{source.name}: git_blob_oid={git_oid}, expected={source.git_blob_oid}")
    return {
        "repo_id": source.repo_id,
        "revision": source.revision,
        "relative_path": source.relative_path,
        "url": source.url,
        "bytes": size,
        "sha256": sha256,
        "git_blob_oid": git_oid,
        "expected_rows": source.expected_rows,
        "identity_check": (
            {"kind": "sha256_lfs_oid", "value": source.content_sha256}
            if source.content_sha256 is not None
            else {"kind": "git_blob_sha1", "value": source.git_blob_oid}
        ),
    }


def _safe_extract_tar(archive: Path, destination_parent: Path) -> Path:
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if not members:
            raise ValidationError(f"empty source archive: {archive}")
        roots: set[str] = set()
        safe_members: list[tarfile.TarInfo] = []
        for member in members:
            pure_parts = Path(member.name).parts
            if not pure_parts or Path(member.name).is_absolute() or ".." in pure_parts:
                raise ValidationError(f"unsafe archive member: {member.name!r}")
            roots.add(pure_parts[0])
            # Source archives can contain convenience symlinks (open-instruct's
            # CLAUDE.md is one).  The evaluator does not need them, so omit all
            # links instead of recreating filesystem references from an archive.
            if member.issym() or member.islnk():
                continue
            if member.isdev():
                raise ValidationError(f"unsupported archive member type: {member.name!r}")
            safe_members.append(member)
        if len(roots) != 1:
            raise ValidationError(f"archive must contain exactly one root directory, got {sorted(roots)}")
        tar.extractall(destination_parent, members=safe_members)  # noqa: S202 - validated above.
    extracted = destination_parent / next(iter(roots))
    if not extracted.is_dir():
        raise ValidationError(f"archive root is not a directory: {extracted}")
    return extracted


def _nltk_resource_url(name: str, spec: dict[str, Any]) -> str:
    return (
        "https://raw.githubusercontent.com/nltk/nltk_data/"
        f"{NLTK_DATA_REVISION}/packages/{spec['subdir']}/{name}.zip"
    )


def _safe_extract_zip(archive: Path, destination_parent: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if not members:
            raise ValidationError(f"empty ZIP archive: {archive}")
        for member in members:
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValidationError(f"unsafe ZIP member in {archive}: {member.filename!r}")
            unix_mode = member.external_attr >> 16
            if unix_mode & 0o170000 == 0o120000:
                raise ValidationError(f"ZIP symlink is forbidden in {archive}: {member.filename!r}")
        bundle.extractall(destination_parent)


def _provision_code_source(source: CodeSource, stage: Path, raw_root: Path) -> dict[str, Any]:
    archive = raw_root / f"{source.name}.tar.gz"
    _download(source.url, archive)
    extract_parent = raw_root / f"extract-{source.name}"
    extract_parent.mkdir()
    extracted = _safe_extract_tar(archive, extract_parent)
    if source.revision not in extracted.name:
        raise ValidationError(
            f"{source.name}: archive root {extracted.name!r} does not identify revision {source.revision}"
        )
    destination = stage / source.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    extracted.rename(destination)
    for relative in source.required_files:
        path = destination / relative
        if not path.is_file():
            raise ValidationError(f"{source.name}: required evaluator file missing: {relative}")

    if source.name == "ifbench_evaluator":
        text = (destination / "evaluation_lib.py").read_text(encoding="utf-8")
        for symbol in ("test_instruction_following_strict", "test_instruction_following_loose"):
            if symbol not in text:
                raise ValidationError(f"IFBench checkout lacks required symbol {symbol}")
    elif source.name == "open_instruct_verifiers":
        text = (destination / "open_instruct/ground_truth_utils.py").read_text(encoding="utf-8")
        for symbol in ("class IFEvalVerifierOld", "class IFEvalVerifier"):
            if symbol not in text:
                raise ValidationError(f"open-instruct checkout lacks required symbol {symbol}")

    return {
        "owner": source.owner,
        "repository": source.repository,
        "revision": source.revision,
        "url": source.url,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "destination": source.destination,
        "tree_sha256": _tree_sha256(destination),
        "required_files": list(source.required_files),
        "archive_links_extracted": False,
    }


def _provision_github_file_group(
    source: GithubFileGroupSource, stage: Path, raw_root: Path
) -> dict[str, Any]:
    destination = stage / source.destination
    destination.mkdir(parents=True, exist_ok=True)
    files_manifest: dict[str, Any] = {}
    for file_spec in source.files:
        basename = Path(file_spec.relative_path).name
        raw_path = raw_root / f"{source.name}-{basename}"
        url = source.url(file_spec.relative_path)
        _download(url, raw_path)
        if raw_path.stat().st_size != file_spec.expected_size:
            raise ValidationError(
                f"{source.name}/{basename}: bytes={raw_path.stat().st_size}, expected={file_spec.expected_size}"
            )
        blob_oid = _git_blob_oid(raw_path)
        if blob_oid != file_spec.git_blob_oid:
            raise ValidationError(
                f"{source.name}/{basename}: git_blob_oid={blob_oid}, expected={file_spec.git_blob_oid}"
            )
        output_path = destination / basename
        shutil.copyfile(raw_path, output_path)
        files_manifest[basename] = {
            "upstream_path": file_spec.relative_path,
            "url": url,
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
            "git_blob_oid": blob_oid,
        }

    evaluation_lib = (destination / "evaluation_lib.py").read_text(encoding="utf-8")
    for symbol in ("test_instruction_following_strict", "test_instruction_following_loose"):
        if symbol not in evaluation_lib:
            raise ValidationError(f"official Google evaluator lacks required symbol {symbol}")
    return {
        "owner": source.owner,
        "repository": source.repository,
        "revision": source.revision,
        "destination": source.destination,
        "distribution": "pinned selected upstream files",
        "files": files_manifest,
        "tree_sha256": _tree_sha256(destination),
    }


def _files_with_hashes(root: Path, pattern: str = "*") -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(candidate for candidate in root.rglob(pattern) if candidate.is_file()):
        records[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return records


def _parse_requirements_lock(path: Path) -> dict[str, str]:
    locked: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        requirement = fields[0]
        if requirement.count("==") != 1:
            raise ValidationError(f"{path}:{line_number}: requirement must use exactly one == pin")
        name, version = requirement.split("==", 1)
        normalized_name = name.strip().lower().replace("_", "-")
        if not normalized_name or not version.strip() or normalized_name in locked:
            raise ValidationError(f"{path}:{line_number}: malformed or duplicate requirement pin")
        hashes = fields[1:]
        if not hashes or any(
            re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", value) is None for value in hashes
        ):
            raise ValidationError(
                f"{path}:{line_number}: every requirement needs one or more lowercase SHA-256 hashes"
            )
        locked[normalized_name] = version.strip()
    if not locked:
        raise ValidationError(f"empty verifier dependency lock: {path}")
    return locked


def _requirements_lock_hashes(path: Path) -> dict[str, list[str]]:
    _parse_requirements_lock(path)
    hashes: dict[str, list[str]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        name = fields[0].split("==", 1)[0].lower().replace("_", "-")
        hashes[name] = [field.removeprefix("--hash=sha256:") for field in fields[1:]]
    return hashes


def _ifbench_unused_requirement_report(stage: Path) -> dict[str, Any]:
    """Prove why two broad upstream requirements are absent from our lock.

    IFBench's repository-level requirements include spaCy and unicodedata2,
    but its pinned evaluation import closure does not reference either.  A
    spaCy wheel cannot safely be layered with ``--no-deps`` unless its large
    dependency closure is also frozen, so this exception is fail-closed over
    the exact runtime modules instead of relying on an image assumption.
    """

    root = stage / "third_party/IFBench"
    module_records: dict[str, dict[str, Any]] = {}
    import_roots: set[str] = set()
    for relative in IFBENCH_RUNTIME_MODULES:
        path = root / relative
        if not path.is_file():
            raise ValidationError(f"IFBench runtime module is missing: {relative}")
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            raise ValidationError(f"cannot parse pinned IFBench runtime module {relative}") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                import_roots.add(node.module.split(".", 1)[0])
        for package in IFBENCH_DECLARED_BUT_RUNTIME_UNUSED:
            if re.search(rf"\b{re.escape(package)}\b", text):
                raise ValidationError(
                    f"pinned IFBench runtime module {relative} now references excluded dependency {package}"
                )
        module_records[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    if set(IFBENCH_DECLARED_BUT_RUNTIME_UNUSED).intersection(import_roots):
        raise ValidationError("excluded IFBench requirement entered the runtime import closure")
    requirements_path = root / "requirements.txt"
    requirements_text = requirements_path.read_text(encoding="utf-8")
    declared = {
        line.strip().lower().replace("_", "-")
        for line in requirements_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not set(IFBENCH_DECLARED_BUT_RUNTIME_UNUSED).issubset(declared):
        raise ValidationError("IFBench upstream-unused exception no longer matches requirements.txt")
    return {
        "upstream_revision": IFBENCH_CODE_REVISION,
        "requirements_sha256": _sha256(requirements_path),
        "declared_but_runtime_unused": list(IFBENCH_DECLARED_BUT_RUNTIME_UNUSED),
        "runtime_modules": module_records,
        "runtime_import_roots": sorted(import_roots),
        "proof": (
            "No identifier reference in the pinned evaluation_lib -> instructions_registry -> "
            "instructions/instructions_util closure; real registry/build/check and Miles reward "
            "smokes remain mandatory."
        ),
    }


def _run_checked(command: list[str], *, env: dict[str, str] | None = None) -> str:
    print("RUN_SETUP " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    print(result.stdout, end="", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"setup command failed rc={result.returncode}: {' '.join(command)}")
    return result.stdout


def _provision_python_dependencies(stage: Path) -> dict[str, Any]:
    """Build only the verifier packages absent from the fixed base image.

    Installing a normal resolved requirements set into a PYTHONPATH-prepended
    target can silently replace Ray/sglang's pydantic/httpx/anyio stack.  The
    lock therefore contains verifier-only packages known to be absent from the
    immutable image, and both wheel/build and install use ``--no-deps``.  Their
    common dependencies must already be present and are checked explicitly.
    """

    requirements = Path(__file__).with_name("verifier_requirements.lock").resolve()
    if not requirements.is_file():
        raise FileNotFoundError(f"verifier dependency lock is missing: {requirements}")
    locked_requirements = _parse_requirements_lock(requirements)
    if sys.version_info[:2] != EXPECTED_CONTAINER_PYTHON:
        raise ValidationError(
            "fixed Miles container was validated with Python 3.10; refusing an unverified interpreter "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )
    verifier_only_names = ("emoji", "immutabledict", "langdetect", "nltk", "syllapy")
    protected_base_names = (
        "anyio",
        "httpx",
        "pydantic",
        "pydantic-settings",
        "ray",
        "setuptools",
        "sglang",
        "torch",
    )
    required_base_names = (
        "absl-py",
        "click",
        "joblib",
        "regex",
        "setuptools",
        "six",
        "tqdm",
        "wheel",
    )
    inventory_names = tuple(
        dict.fromkeys(verifier_only_names + protected_base_names + required_base_names)
    )
    base_image_inventory: dict[str, str | None] = {}
    for name in inventory_names:
        try:
            base_image_inventory[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            base_image_inventory[name] = None
    unexpectedly_present = [name for name in verifier_only_names if base_image_inventory[name] is not None]
    if unexpectedly_present:
        raise ValidationError(
            "fixed-image verifier inventory changed; refusing to shadow already-installed packages: "
            f"{unexpectedly_present}"
        )
    missing_base = [name for name in required_base_names if base_image_inventory[name] is None]
    if missing_base:
        raise ValidationError(f"fixed image lacks required verifier base dependencies: {missing_base}")
    third_party = stage / "third_party"
    python_target = third_party / "python"
    wheelhouse = third_party / "wheelhouse"
    nltk_data = third_party / "nltk_data"
    python_target.mkdir(parents=True)
    wheelhouse.mkdir(parents=True)
    nltk_data.mkdir(parents=True)

    pip_base = [sys.executable, "-m", "pip", "--disable-pip-version-check", "--no-cache-dir"]
    _run_checked(
        pip_base
        + [
            "download",
            "--require-hashes",
            "--no-deps",
            "--index-url",
            "https://pypi.org/simple",
            "--dest",
            str(wheelhouse),
            "--requirement",
            str(requirements),
        ]
    )
    _run_checked(
        pip_base
        + [
            "install",
            "--require-hashes",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(python_target),
            "--no-index",
            "--find-links",
            str(wheelhouse),
            "--requirement",
            str(requirements),
        ]
    )

    setup_env = dict(os.environ)
    setup_env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(python_target), setup_env.get("PYTHONPATH", "")) if part
    )
    setup_env["NLTK_DATA"] = str(nltk_data)
    setup_env["PYTHONDONTWRITEBYTECODE"] = "1"
    nltk_resource_manifest: dict[str, dict[str, Any]] = {}
    for name, spec in NLTK_DATA_RESOURCES.items():
        archive_relative = f"{spec['subdir']}/{name}.zip"
        archive = nltk_data / archive_relative
        url = _nltk_resource_url(name, spec)
        _download(url, archive)
        if archive.stat().st_size != spec["bytes"] or _sha256(archive) != spec["sha256"]:
            raise ValidationError(f"pinned NLTK resource identity mismatch: {name}")
        _safe_extract_zip(archive, archive.parent)
        nltk_resource_manifest[name] = {
            "revision": NLTK_DATA_REVISION,
            "url": url,
            "archive_path": archive_relative,
            "bytes": spec["bytes"],
            "sha256": spec["sha256"],
        }

    freeze_code = (
        "import importlib.metadata as m; "
        f"p={str(python_target)!r}; "
        "print('\\n'.join(sorted(f'{d.metadata[\"Name\"]}=={d.version}' "
        "for d in m.distributions(path=[p]))))"
    )
    freeze = _run_checked([sys.executable, "-c", freeze_code], env=setup_env)
    freeze_path = third_party / "verifier_requirements.freeze.txt"
    _atomic_write_text(freeze_path, freeze)

    nltk_probe = (
        "import nltk; "
        f"nltk.data.path.insert(0, {str(nltk_data)!r}); "
        "[nltk.data.find(x) for x in "
        "['tokenizers/punkt','tokenizers/punkt_tab','corpora/stopwords',"
        "'taggers/averaged_perceptron_tagger_eng']]; "
        "print('NLTK_RESOURCE_PROBE=PASS')"
    )
    _run_checked([sys.executable, "-c", nltk_probe], env=setup_env)

    base_probe = (
        "import absl,anyio,click,emoji,httpx,immutabledict,joblib,langdetect,nltk,pydantic,"
        "pydantic_settings,regex,setuptools,six,syllapy,tqdm,wheel; "
        "print('VERIFIER_DEPENDENCY_IMPORT_PROBE=PASS')"
    )
    _run_checked([sys.executable, "-c", base_probe], env=setup_env)

    evaluator_probes = (
        (
            [str(python_target), str(stage / "third_party/IFBench")],
            "import evaluation_lib; assert hasattr(evaluation_lib, 'test_instruction_following_loose'); "
            "print('IFBENCH_IMPORT_PROBE=PASS')",
        ),
        (
            [str(python_target), str(stage / "third_party/open-instruct")],
            "from open_instruct import if_functions; "
            "from open_instruct.IFEvalG import instructions_registry; "
            "assert if_functions.IF_FUNCTIONS_MAP and instructions_registry.INSTRUCTION_DICT; "
            "print('OPEN_INSTRUCT_IFEVAL_IMPORT_PROBE=PASS')",
        ),
        (
            [str(python_target), str(stage / "third_party/google-research")],
            "from instruction_following_eval import evaluation_lib; "
            "assert hasattr(evaluation_lib, 'test_instruction_following_strict'); "
            "assert hasattr(evaluation_lib, 'test_instruction_following_loose'); "
            "print('GOOGLE_IFEVAL_IMPORT_PROBE=PASS')",
        ),
    )
    for paths, code in evaluator_probes:
        probe_env = dict(setup_env)
        probe_env["PYTHONPATH"] = os.pathsep.join(
            paths + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
        )
        _run_checked([sys.executable, "-c", code], env=probe_env)

    target_distributions = {
        distribution.metadata["Name"].lower(): distribution.version
        for distribution in importlib.metadata.distributions(path=[str(python_target)])
    }
    if target_distributions != locked_requirements or set(target_distributions) != set(
        verifier_only_names
    ):
        raise ValidationError(
            "verifier target must exactly match the absent verifier-only lock; "
            f"got={target_distributions}, expected={locked_requirements}"
        )
    protected_collisions = sorted(set(target_distributions).intersection(protected_base_names))
    if protected_collisions:
        raise ValidationError(f"verifier target shadows protected base packages: {protected_collisions}")

    distribution_artifacts = _files_with_hashes(wheelhouse)
    if len(distribution_artifacts) != len(locked_requirements):
        raise ValidationError(
            "dependency download must produce exactly one artifact per locked requirement; "
            f"got={sorted(distribution_artifacts)}"
        )
    downloaded_hashes = {record["sha256"] for record in distribution_artifacts.values()}
    for name, allowed_hashes in _requirements_lock_hashes(requirements).items():
        if not downloaded_hashes.intersection(allowed_hashes):
            raise ValidationError(f"no downloaded artifact matches the declared PyPI hash for {name}")
    nltk_archives = _files_with_hashes(nltk_data, "*.zip")
    if len(nltk_archives) != len(NLTK_DATA_RESOURCES):
        raise ValidationError(
            f"expected exactly {len(NLTK_DATA_RESOURCES)} pinned NLTK resource archives, "
            f"got {len(nltk_archives)}"
        )
    return {
        "requirements_lock_source": str(requirements),
        "requirements_lock_sha256": _sha256(requirements),
        "requirements_index": "https://pypi.org/simple",
        "requirements_pypi_sha256": _requirements_lock_hashes(requirements),
        "nltk_compatibility": {
            "selected_version": "3.9.4",
            "official_pypi_requires_python": ">=3.10",
            "ifbench_1091c4c_requirement": "nltk (no upper bound), Python >=3.10",
            "open_instruct_5b2ebfa_package_requirement": "nltk>=3.9.1, Python ==3.12.*",
            "runtime_scope": (
                "Vendored verifier modules only; Open Instruct package installation/support is not claimed. "
                "Compatibility on container Python 3.10 is established by registry/signature validation "
                "and real Miles async_rm smoke."
            ),
        },
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "fixed_container_image": EXPECTED_CONTAINER_IMAGE,
        "base_image_package_inventory": base_image_inventory,
        "verifier_only_distributions": target_distributions,
        "protected_base_distributions": list(protected_base_names),
        "required_base_distributions": list(required_base_names),
        "ifbench_upstream_unused_requirements": _ifbench_unused_requirement_report(stage),
        "pip_dependency_resolution": (
            "disabled (--no-deps, --no-build-isolation); download and install both enforce "
            "--require-hashes"
        ),
        "python_target": "third_party/python",
        "python_target_tree_sha256": _tree_sha256(python_target),
        "freeze_path": "third_party/verifier_requirements.freeze.txt",
        "freeze_sha256": _sha256(freeze_path),
        "wheelhouse": "third_party/wheelhouse",
        "wheel_files": distribution_artifacts,
        "distribution_artifacts": distribution_artifacts,
        "wheelhouse_tree_sha256": _tree_sha256(wheelhouse),
        "nltk_data": "third_party/nltk_data",
        "nltk_data_source": {
            "repository": "nltk/nltk_data",
            "revision": NLTK_DATA_REVISION,
            "resources": nltk_resource_manifest,
        },
        "nltk_resources": list(NLTK_DATA_RESOURCES),
        "nltk_resource_archives": nltk_archives,
        "nltk_data_tree_sha256": _tree_sha256(nltk_data),
        "worker_installation_allowed": False,
        "runtime_environment": {
            "PYTHONPATH_prepend": "third_party/python",
            "NLTK_DATA": "third_party/nltk_data",
            "IFBENCH_REPO_PATH": "third_party/IFBench",
            "OPEN_INSTRUCT_REPO_PATH": "third_party/open-instruct",
            "GOOGLE_IFEVAL_OFFICIAL_CODE": "third_party/google-research/instruction_following_eval",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "dependency_scope": (
            "Only verifier-specific packages absent from the fixed image. Common runtime "
            "packages are inherited from the image and are never copied into the prepended target."
        ),
    }


def _runtime_validation_environment(bundle: Path) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    target = bundle / "third_party/python"
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                part
                for part in (str(target), str(repo_root), environment.get("PYTHONPATH", ""))
                if part
            ),
            "NLTK_DATA": str(bundle / "third_party/nltk_data"),
            "IFBENCH_REPO_PATH": str(bundle / "third_party/IFBench"),
            "OPEN_INSTRUCT_REPO_PATH": str(bundle / "third_party/open-instruct"),
            "GOOGLE_IFEVAL_OFFICIAL_CODE": str(
                bundle / "third_party/google-research/instruction_following_eval"
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _validate_runtime_contracts(bundle: Path) -> dict[str, Any]:
    if not RUNTIME_VALIDATOR.is_file():
        raise FileNotFoundError(f"runtime contract validator is missing: {RUNTIME_VALIDATOR}")
    output = _run_checked(
        [sys.executable, str(RUNTIME_VALIDATOR), str(bundle)],
        env=_runtime_validation_environment(bundle),
    )
    prefix = "RUNTIME_CONTRACT_REPORT_JSON="
    report_lines = [line[len(prefix) :] for line in output.splitlines() if line.startswith(prefix)]
    if len(report_lines) != 1:
        raise ValidationError("runtime validator did not emit exactly one machine-readable report")
    report = json.loads(report_lines[0])
    if report.get("result") != "PASS" or not report.get("miles_reward_smoke_calls"):
        raise ValidationError("runtime verifier/reward smoke did not pass")
    return report


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required in the compute-node container") from exc
    return pq.read_table(path).to_pylist()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValidationError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def _require_fields(row: dict[str, Any], required: set[str], source: str, index: int) -> None:
    missing = required.difference(row)
    if missing:
        raise ValidationError(f"{source}[{index}]: missing required fields {sorted(missing)}")


def _deduplicate_exact_raw(
    rows: list[dict[str, Any]], source: str, expected_rows: int
) -> tuple[list[dict[str, Any]], int]:
    if len(rows) != expected_rows:
        raise ValidationError(f"{source}: raw rows={len(rows)}, expected={expected_rows}")
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0
    for row in rows:
        key = _canonical(row)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, duplicates


def _one_user_prompt(messages: Any, source: str, index: int) -> list[dict[str, str]]:
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValidationError(f"{source}[{index}]: expected exactly one message")
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        raise ValidationError(f"{source}[{index}]: expected one user message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValidationError(f"{source}[{index}]: user content must be non-empty text")
    return [{"role": "user", "content": content}]


def _ensure_unique_prompts(rows: list[dict[str, Any]], dataset: str) -> set[str]:
    prompts: set[str] = set()
    normalized_prompts: set[str] = set()
    for index, row in enumerate(rows):
        prompt = _one_user_prompt(row.get("prompt"), dataset, index)[0]["content"]
        if prompt in prompts:
            raise ValidationError(f"{dataset}[{index}]: exact duplicate prompt remains after deduplication")
        normalized = unicodedata.normalize("NFC", prompt).strip()
        if normalized in normalized_prompts:
            raise ValidationError(
                f"{dataset}[{index}]: NFC+strip duplicate prompt remains after deduplication"
            )
        prompts.add(prompt)
        normalized_prompts.add(normalized)
    return prompts


def _convert_old_ifeval(rows: list[dict[str, Any]], source: FileSource) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    required = {"messages", "ground_truth", "dataset", "constraint_type", "constraint"}
    for index, row in enumerate(rows):
        _require_fields(row, required, source.name, index)
        if "instruction_id_list" in row or "kwargs" in row:
            raise ValidationError(f"{source.name}[{index}]: new verifier fields appeared in old-schema data")
        prompt = _one_user_prompt(row["messages"], source.name, index)
        ground_truth = row["ground_truth"]
        if not isinstance(ground_truth, str):
            raise ValidationError(f"{source.name}[{index}]: ground_truth must remain a JSON string")
        try:
            constraint = json.loads(ground_truth)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{source.name}[{index}]: invalid ground_truth JSON") from exc
        if not isinstance(constraint, dict):
            raise ValidationError(f"{source.name}[{index}]: ground_truth JSON must be an object")
        func_name = constraint.get("func_name")
        if not isinstance(func_name, str) or not func_name:
            raise ValidationError(f"{source.name}[{index}]: old schema requires non-empty func_name")
        if any(key in constraint for key in ("instruction_id", "instruction_id_list", "kwargs")):
            raise ValidationError(f"{source.name}[{index}]: mixed old/new verifier schema")
        for field in ("dataset", "constraint_type", "constraint"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValidationError(f"{source.name}[{index}]: {field} must be non-empty text")
        converted.append(
            {
                "prompt": prompt,
                "label": _canonical(constraint),
                "metadata": {
                    "rm_type": "ifeval_old",
                    "verifier_schema": OLD_IFEVAL_SCHEMA,
                    "data_source": source.repo_id,
                    "source_revision": source.revision,
                    "source_row_index": index,
                    "source_dataset": row["dataset"],
                    "constraint_type": row["constraint_type"],
                    "constraint": row["constraint"],
                },
            }
        )
    _ensure_unique_prompts(converted, source.name)
    return converted


def _convert_if_multi(rows: list[dict[str, Any]], source: FileSource) -> list[dict[str, Any]]:
    """Preserve the one-element IFEvalG contract used by the gated fallback."""

    converted: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for index, row in enumerate(rows):
        if set(row) != IF_MULTI_SOURCE_KEYS:
            raise ValidationError(
                f"{source.name}[{index}]: exact source keys required; "
                f"got={sorted(row)}, expected={sorted(IF_MULTI_SOURCE_KEYS)}"
            )
        prompt = _one_user_prompt(row["messages"], source.name, index)
        raw_ground_truth = row["ground_truth"]
        if not isinstance(raw_ground_truth, str):
            raise ValidationError(f"{source.name}[{index}]: ground_truth must remain serialized text")
        try:
            payload = ast.literal_eval(raw_ground_truth)
        except (SyntaxError, ValueError) as exc:
            raise ValidationError(f"{source.name}[{index}]: invalid Python-literal ground_truth") from exc
        if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
            raise ValidationError(
                f"{source.name}[{index}]: IFEvalG ground_truth must be a one-element list of objects"
            )
        contract = payload[0]
        if set(contract) != {"instruction_id", "kwargs"}:
            raise ValidationError(
                f"{source.name}[{index}]: exact IFEvalG keys required; got={sorted(contract)}"
            )
        instruction_ids = contract.get("instruction_id")
        kwargs = contract.get("kwargs")
        if (
            not isinstance(instruction_ids, list)
            or not instruction_ids
            or any(not isinstance(item, str) or not item for item in instruction_ids)
        ):
            raise ValidationError(f"{source.name}[{index}]: invalid instruction_id list")
        if not isinstance(kwargs, list) or len(kwargs) != len(instruction_ids):
            raise ValidationError(f"{source.name}[{index}]: kwargs must align exactly with instruction_id")
        if any(item is not None and not isinstance(item, dict) for item in kwargs):
            raise ValidationError(f"{source.name}[{index}]: kwargs entries must be objects or null")
        if not isinstance(row["key"], str) or not row["key"]:
            raise ValidationError(f"{source.name}[{index}]: key must be non-empty text")
        typed_key = _canonical({"type": "str", "value": row["key"]})
        if typed_key in seen_keys:
            raise ValidationError(f"{source.name}[{index}]: duplicate source key {row['key']!r}")
        seen_keys.add(typed_key)
        for field in ("dataset", "constraint_type", "constraint"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValidationError(f"{source.name}[{index}]: {field} must be non-empty text")

        canonical_payload = [
            {"instruction_id": list(instruction_ids), "kwargs": copy.deepcopy(kwargs)}
        ]
        canonical_ground_truth = _canonical(canonical_payload)
        prompt_text = prompt[0]["content"]
        converted.append(
            {
                "prompt": prompt,
                "label": canonical_ground_truth,
                "metadata": {
                    "rm_type": "ifevalg",
                    "verifier_schema": IF_MULTI_SCHEMA,
                    "data_source": source.repo_id,
                    "source_revision": source.revision,
                    "source_row_index": index,
                    "source_key": copy.deepcopy(row["key"]),
                    "source_dataset": row["dataset"],
                    "constraint_type": row["constraint_type"],
                    "constraint": row["constraint"],
                    "prompt_text": prompt_text,
                    "ground_truth": canonical_ground_truth,
                    "source_ground_truth": raw_ground_truth,
                    "instruction_id": list(instruction_ids),
                    "kwargs": copy.deepcopy(kwargs),
                    "candidate_status": "preregistered_fallback_only",
                },
            }
        )
    return converted


def _consolidate_if_multi_prompts(
    rows: list[dict[str, Any]], dataset: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deterministically consolidate IF_multi rows sharing an exact prompt.

    Exact contract repeats retain the first row. Distinct valid IFEvalG
    contracts are merged in source order, de-duplicating only byte-equivalent
    ``(instruction_id, kwargs)`` pairs. No verifier constraint is silently
    discarded.
    """

    by_prompt: dict[str, dict[str, Any]] = {}
    source_contract_identities: dict[str, set[str]] = {}
    identical_removed = 0
    consolidated_groups: set[str] = set()
    constraints_deduplicated = 0
    duplicate_rows_removed = 0
    for row in rows:
        prompt = _one_user_prompt(row.get("prompt"), dataset, 0)[0]["content"]
        if prompt not in by_prompt:
            by_prompt[prompt] = copy.deepcopy(row)
            source_contract_identities[prompt] = {row["label"]}
            continue

        duplicate_rows_removed += 1
        retained = by_prompt[prompt]
        retained_metadata = retained["metadata"]
        incoming_metadata = row["metadata"]
        retained_indices = retained_metadata.setdefault(
            "source_row_indices", [retained_metadata["source_row_index"]]
        )
        retained_keys = retained_metadata.setdefault(
            "source_keys", [copy.deepcopy(retained_metadata["source_key"])]
        )
        retained_ground_truths = retained_metadata.setdefault(
            "source_ground_truths", [retained_metadata["source_ground_truth"]]
        )
        retained_indices.append(incoming_metadata["source_row_index"])
        retained_keys.append(copy.deepcopy(incoming_metadata["source_key"]))
        retained_ground_truths.append(incoming_metadata["source_ground_truth"])

        if row["label"] in source_contract_identities[prompt]:
            identical_removed += 1
            continue

        source_contract_identities[prompt].add(row["label"])
        consolidated_groups.add(prompt)
        retained_ids = retained_metadata["instruction_id"]
        retained_kwargs = retained_metadata["kwargs"]
        incoming_ids = incoming_metadata["instruction_id"]
        incoming_kwargs = incoming_metadata["kwargs"]
        if len(retained_ids) != len(retained_kwargs) or len(incoming_ids) != len(incoming_kwargs):
            raise ValidationError(f"{dataset}: cannot consolidate malformed aligned contracts")
        seen_constraints = {
            _canonical({"instruction_id": instruction_id, "kwargs": kwargs})
            for instruction_id, kwargs in zip(retained_ids, retained_kwargs, strict=True)
        }
        for instruction_id, kwargs in zip(incoming_ids, incoming_kwargs, strict=True):
            identity = _canonical({"instruction_id": instruction_id, "kwargs": kwargs})
            if identity in seen_constraints:
                constraints_deduplicated += 1
                continue
            retained_ids.append(instruction_id)
            retained_kwargs.append(copy.deepcopy(kwargs))
            seen_constraints.add(identity)
        if not retained_ids or len(retained_ids) != len(retained_kwargs):
            raise ValidationError(f"{dataset}: consolidation produced an invalid IFEvalG contract")
        merged_payload = [{"instruction_id": retained_ids, "kwargs": retained_kwargs}]
        retained["label"] = _canonical(merged_payload)
        retained_metadata["ground_truth"] = retained["label"]
        retained_metadata["prompt_consolidated"] = True

    consolidated = list(by_prompt.values())
    _ensure_unique_prompts(consolidated, dataset)
    return consolidated, {
        "prompt_duplicate_rows_removed": duplicate_rows_removed,
        "prompt_identical_contract_duplicates_removed": identical_removed,
        "prompt_contract_consolidation_groups": len(consolidated_groups),
        "constraints_deduplicated_during_consolidation": constraints_deduplicated,
    }


def _normalize_new_contract(
    row: dict[str, Any], source: FileSource, index: int
) -> tuple[str | int, str, list[str], list[dict[str, Any]]]:
    if set(row) != NEW_IFEVAL_SOURCE_KEYS:
        raise ValidationError(
            f"{source.name}[{index}]: exact source keys required; "
            f"got={sorted(row)}, expected={sorted(NEW_IFEVAL_SOURCE_KEYS)}"
        )
    if "ground_truth" in row or "func_name" in row:
        raise ValidationError(f"{source.name}[{index}]: old verifier fields appeared in new-schema data")
    record_id = row["key"]
    if isinstance(record_id, bool) or not isinstance(record_id, (str, int)):
        raise ValidationError(f"{source.name}[{index}]: key must be a losslessly stored string or integer")
    if isinstance(record_id, str) and not record_id:
        raise ValidationError(f"{source.name}[{index}]: string key must be non-empty")
    prompt = row["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValidationError(f"{source.name}[{index}]: prompt must be non-empty text")
    instruction_ids = row["instruction_id_list"]
    kwargs = row["kwargs"]
    if (
        not isinstance(instruction_ids, list)
        or not instruction_ids
        or any(not isinstance(item, str) or not item for item in instruction_ids)
    ):
        raise ValidationError(f"{source.name}[{index}]: instruction_id_list must be non-empty strings")
    if not isinstance(kwargs, list) or len(kwargs) != len(instruction_ids):
        raise ValidationError(
            f"{source.name}[{index}]: kwargs length must equal instruction_id_list length"
        )
    if any(not isinstance(item, dict) for item in kwargs):
        raise ValidationError(f"{source.name}[{index}]: every kwargs item must be an object")
    return copy.deepcopy(record_id), prompt, list(instruction_ids), copy.deepcopy(kwargs)


def _convert_new_ifeval(
    rows: list[dict[str, Any]],
    source: FileSource,
    *,
    rm_type: str,
    verifier_schema: str,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    seen_typed_keys: set[str] = set()
    for index, row in enumerate(rows):
        record_id, prompt_text, instruction_ids, kwargs = _normalize_new_contract(row, source, index)
        typed_key = _canonical({"type": type(record_id).__name__, "value": record_id})
        if typed_key in seen_typed_keys:
            raise ValidationError(f"{source.name}[{index}]: duplicate typed key {record_id!r}")
        seen_typed_keys.add(typed_key)
        contract = {"instruction_id_list": instruction_ids, "kwargs": kwargs}
        converted.append(
            {
                "prompt": [{"role": "user", "content": prompt_text}],
                "label": _canonical(contract),
                "metadata": {
                    "rm_type": rm_type,
                    "verifier_schema": verifier_schema,
                    "data_source": source.repo_id,
                    "source_revision": source.revision,
                    "source_row_index": index,
                    "source_key": copy.deepcopy(record_id),
                    "record_id": copy.deepcopy(record_id),
                    "prompt_text": prompt_text,
                    "instruction_id_list": instruction_ids,
                    "kwargs": kwargs,
                },
            }
        )
    _ensure_unique_prompts(converted, source.name)
    return converted


def _convert_gsm8k(rows: list[dict[str, Any]], source: FileSource) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if set(row) != {"question", "answer"}:
            raise ValidationError(
                f"{source.name}[{index}]: exact GSM8K source keys required; got={sorted(row)}"
            )
        question = row["question"]
        reference = row["answer"]
        if not isinstance(question, str) or not question.strip():
            raise ValidationError(f"{source.name}[{index}]: question must be non-empty text")
        if not isinstance(reference, str):
            raise ValidationError(f"{source.name}[{index}]: answer must be text")
        match = GSM8K_ANSWER_PATTERN.search(reference)
        if match is None:
            raise ValidationError(f"{source.name}[{index}]: reference lacks a strict #### answer")
        label = match.group(1).replace(",", "")
        converted.append(
            {
                "prompt": [
                    {"role": "user", "content": f"{question}{GSM8K_PROMPT_SUFFIX}"}
                ],
                "label": label,
                "metadata": {
                    "rm_type": "gsm8k_verl",
                    "verifier_schema": GSM8K_SCHEMA,
                    "data_source": source.repo_id,
                    "source_revision": source.revision,
                    "source_row_index": index,
                    "source_split": "test",
                    "question": question,
                    "reference_answer": reference,
                    "reward_model": {"style": "rule", "ground_truth": label},
                },
            }
        )
    _ensure_unique_prompts(converted, source.name)
    return converted


def _convert_math500(rows: list[dict[str, Any]], source: FileSource) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    required = {"problem", "solution", "answer", "subject", "level", "unique_id"}
    for index, row in enumerate(rows):
        _require_fields(row, required, source.name, index)
        for field in ("problem", "solution", "answer", "subject", "unique_id"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValidationError(f"{source.name}[{index}]: {field} must be non-empty text")
        if not isinstance(row["level"], int) or not 1 <= row["level"] <= 5:
            raise ValidationError(f"{source.name}[{index}]: level must be an integer in [1, 5]")
        prompt_text = f"{MATH_PROMPT_PREFIX}\n\n{row['problem']}\n\n{MATH_PROMPT_SUFFIX}"
        converted.append(
            {
                "prompt": [{"role": "user", "content": prompt_text}],
                "label": row["answer"],
                "metadata": {
                    "rm_type": "math",
                    "verifier_schema": MATH_SCHEMA,
                    "data_source": source.repo_id,
                    "source_revision": source.revision,
                    "source_row_index": index,
                    "source_split": "test",
                    "unique_id": row["unique_id"],
                    "subject": row["subject"],
                    "level": row["level"],
                    "reference_solution": row["solution"],
                },
            }
        )
    _ensure_unique_prompts(converted, source.name)
    unique_ids = {row["metadata"]["unique_id"] for row in converted}
    if len(unique_ids) != len(converted):
        raise ValidationError(f"{source.name}: duplicate unique_id values")
    return converted


def _load_raw(source: FileSource, path: Path) -> list[dict[str, Any]]:
    return _read_parquet(path) if source.relative_path.endswith(".parquet") else _read_jsonl(path)


def _check_disjoint(prompt_sets: dict[str, set[str]]) -> dict[str, int]:
    overlaps: dict[str, int] = {}
    names = sorted(prompt_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = prompt_sets[left].intersection(prompt_sets[right])
            key = f"{left}__vs__{right}"
            overlaps[key] = len(overlap)
            if overlap:
                example = sorted(overlap)[0][:200]
                raise ValidationError(f"prompt leakage {key}: count={len(overlap)}, example={example!r}")
    return overlaps


def _check_prompt_leakage_views(
    prompt_sets: dict[str, set[str]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Hard-fail exact and NFC+strip leakage; report casefold as a diagnostic."""

    exact = _check_disjoint(prompt_sets)
    normalized_sets = {
        name: {unicodedata.normalize("NFC", prompt).strip() for prompt in prompts}
        for name, prompts in prompt_sets.items()
    }
    normalized = _check_disjoint(normalized_sets)
    casefold_sets = {
        name: {prompt.casefold() for prompt in prompts}
        for name, prompts in normalized_sets.items()
    }
    casefold_counts: dict[str, int] = {}
    names = sorted(casefold_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            key = f"{left}__vs__{right}"
            casefold_counts[key] = len(casefold_sets[left].intersection(casefold_sets[right]))
    return exact, normalized, casefold_counts


def _eval_protocols(output_root: Path) -> dict[str, dict[str, Any]]:
    def dataset(
        source_name: str,
        eval_name: str,
        rm_type: str,
        n: int,
        temperature: float,
        top_p: float,
        max_response_len: int,
    ) -> dict[str, Any]:
        expected_rm_type = EVAL_DATASET_RM_TYPES.get(eval_name)
        if expected_rm_type is None or rm_type != expected_rm_type:
            raise ValidationError(
                f"invalid frozen eval dataset/reward route: {eval_name!r} -> {rm_type!r}"
            )
        return {
            "name": eval_name,
            "source_dataset": source_name,
            "artifact_path": OUTPUT_FILES[source_name],
            "path": str(output_root / OUTPUT_FILES[source_name]),
            "rm_type": rm_type,
            # This is deliberately separate from rm_type: the latter remains
            # sample reward metadata, while this stable tag survives the
            # flattened eval_*.pt result representation.
            "metadata_overrides": {EVAL_DATASET_METADATA_KEY: eval_name},
            "n_samples_per_eval_prompt": n,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": -1,
            "max_response_len": max_response_len,
        }

    return {
        "math": {
            "config_path": "eval_math.yaml",
            "dataset_set": ["gsm8k", "math500"],
            "datasets": [
                dataset("gsm8k_test", "gsm8k", "gsm8k_verl", 1, 0.0, 1.0, 1024),
                dataset("math500", "math500", "math", 4, 1.0, 1.0, 1024),
            ],
            "scoring_dispatcher_required": (
                "Per-sample metadata.rm_type dispatcher: gsm8k_verl -> historical Verl ####; "
                "math -> Miles boxed equivalence"
            ),
        },
        "nonmath": {
            "config_path": "eval_nonmath.yaml",
            "dataset_set": ["google_ifeval", "ifbench_test"],
            "datasets": [
                dataset("google_ifeval", "google_ifeval", "ifevalg", 1, 0.0, 1.0, 2048),
                dataset("ifbench_test", "ifbench_test", "ifbench", 1, 0.0, 1.0, 2048),
            ],
        },
    }


def _eval_config_text(output_root: Path, protocol_name: str) -> str:
    protocols = _eval_protocols(output_root)
    if protocol_name not in protocols:
        raise ValidationError(f"unknown eval protocol {protocol_name!r}")
    datasets = protocols[protocol_name]["datasets"]
    lines = [
        "# Generated by prepare_rebuttal_data.py; paths are immutable after publication.",
        "eval:",
        "  defaults:",
        "    input_key: prompt",
        "    label_key: label",
        "    metadata_key: metadata",
        "  datasets:",
    ]
    for spec in datasets:
        lines.extend(
            (
                f"    - name: {spec['name']}",
                f"      path: {_canonical(spec['path'])}",
                f"      rm_type: {spec['rm_type']}",
                "      metadata_overrides:",
                f"        {EVAL_DATASET_METADATA_KEY}: {spec['name']}",
                f"      n_samples_per_eval_prompt: {spec['n_samples_per_eval_prompt']}",
                f"      temperature: {spec['temperature']}",
                f"      top_p: {spec['top_p']}",
                f"      top_k: {spec['top_k']}",
                f"      max_response_len: {spec['max_response_len']}",
            )
        )
    return "\n".join(lines) + "\n"


def _dataset_manifest_entry(
    source: FileSource,
    output_name: str,
    role: str,
    schema: str,
    rm_type: str,
    raw_rows: int,
    duplicates_removed: int,
    output_info: dict[str, Any],
    prompt_consolidation: dict[str, int] | None = None,
) -> dict[str, Any]:
    entry = {
        "role": role,
        "source": source.name,
        "source_revision": source.revision,
        "output_path": output_name,
        "raw_rows": raw_rows,
        "exact_raw_record_duplicates_removed": duplicates_removed,
        "output_rows": output_info["rows"],
        "output_exact_prompt_duplicates": 0,
        "verifier_schema": schema,
        "rm_type": rm_type,
        "sha256": output_info["sha256"],
        "bytes": output_info["bytes"],
        "physical_rows_per_prompt": 1,
        "rollout_preexpanded": False,
    }
    entry.update(
        prompt_consolidation
        or {
            "prompt_duplicate_rows_removed": 0,
            "prompt_identical_contract_duplicates_removed": 0,
            "prompt_contract_consolidation_groups": 0,
            "constraints_deduplicated_during_consolidation": 0,
        }
    )
    return entry


def _prepare_one_dataset(
    source: FileSource,
    raw_path: Path,
    stage: Path,
    converter: Callable[[list[dict[str, Any]], FileSource], list[dict[str, Any]]],
    *,
    role: str,
    schema: str,
    rm_type: str,
    consolidate_if_multi: bool = False,
) -> tuple[dict[str, Any], set[str]]:
    raw_rows = _load_raw(source, raw_path)
    unique_raw, duplicates = _deduplicate_exact_raw(raw_rows, source.name, source.expected_rows)
    converted = converter(unique_raw, source)
    prompt_consolidation: dict[str, int] | None = None
    if consolidate_if_multi:
        converted, prompt_consolidation = _consolidate_if_multi_prompts(converted, source.name)
        if prompt_consolidation["prompt_duplicate_rows_removed"] != len(unique_raw) - len(converted):
            raise ValidationError(f"{source.name}: prompt-consolidation accounting mismatch")
    elif len(converted) != len(unique_raw):
        raise ValidationError(f"{source.name}: converter changed row cardinality")
    prompts = _ensure_unique_prompts(converted, source.name)
    output_name = OUTPUT_FILES[source.name]
    output_info = _atomic_write_jsonl(stage / output_name, converted)
    entry = _dataset_manifest_entry(
        source,
        output_name,
        role,
        schema,
        rm_type,
        len(raw_rows),
        duplicates,
        output_info,
        prompt_consolidation,
    )
    return entry, prompts


def _manifest_file_records(stage: Path, relative_paths: Iterable[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative in sorted(relative_paths):
        path = stage / relative
        records[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return records


def create_dataset_bundle(output_root: Path) -> dict[str, Any]:
    output_root = _require_allowed_output_root(output_root)
    allocation = _require_compute_allocation()
    source_code = _require_frozen_source()
    reported_image = os.environ.get("MILES_CONTAINER_IMAGE")
    if reported_image != EXPECTED_CONTAINER_IMAGE:
        raise ValidationError(
            "data creation must run through the fixed-image compute wrapper: "
            f"reported={reported_image!r}, expected={EXPECTED_CONTAINER_IMAGE!r}"
        )
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output {output_root}; use --verify-only or choose a new path"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    raw_root = stage / ".raw"
    raw_root.mkdir()
    try:
        source_manifest: dict[str, Any] = {}
        raw_paths: dict[str, Path] = {}
        for source in FILE_SOURCES:
            suffix = Path(source.relative_path).suffix
            raw_path = raw_root / f"{source.name}{suffix}"
            print(f"DOWNLOAD source={source.name} revision={source.revision} url={source.url}", flush=True)
            _download(source.url, raw_path)
            source_manifest[source.name] = _validate_source_file(source, raw_path)
            raw_paths[source.name] = raw_path

        code_manifest: dict[str, Any] = {}
        for source in CODE_SOURCES:
            print(f"DOWNLOAD code={source.name} revision={source.revision} url={source.url}", flush=True)
            code_manifest[source.name] = _provision_code_source(source, stage, raw_root)
        print(
            "DOWNLOAD code="
            f"{GOOGLE_EVAL_SOURCE.name} revision={GOOGLE_EVAL_SOURCE.revision} "
            "distribution=pinned_selected_files",
            flush=True,
        )
        code_manifest[GOOGLE_EVAL_SOURCE.name] = _provision_github_file_group(
            GOOGLE_EVAL_SOURCE, stage, raw_root
        )
        dependency_manifest = _provision_python_dependencies(stage)
        source_by_name = {source.name: source for source in FILE_SOURCES}
        dataset_manifest: dict[str, Any] = {}
        prompt_sets: dict[str, set[str]] = {}

        entry, prompts = _prepare_one_dataset(
            source_by_name["rlvr_ifeval_train"],
            raw_paths["rlvr_ifeval_train"],
            stage,
            _convert_old_ifeval,
            role="train",
            schema=OLD_IFEVAL_SCHEMA,
            rm_type="ifeval_old",
        )
        dataset_manifest["rlvr_ifeval_train"] = entry
        prompt_sets["rlvr_ifeval_train"] = prompts

        entry, prompts = _prepare_one_dataset(
            source_by_name["if_multi_fallback_train"],
            raw_paths["if_multi_fallback_train"],
            stage,
            _convert_if_multi,
            role="preregistered_train_fallback",
            schema=IF_MULTI_SCHEMA,
            rm_type="ifevalg",
            consolidate_if_multi=True,
        )
        entry["activation_gate"] = (
            "Use only if the preregistered RLVR-IFeval base-difficulty pilot gate fails; "
            "never pool old and IFEvalG verifier schemas in one run."
        )
        dataset_manifest["if_multi_fallback_train"] = entry
        prompt_sets["if_multi_fallback_train"] = prompts

        entry, prompts = _prepare_one_dataset(
            source_by_name["google_ifeval"],
            raw_paths["google_ifeval"],
            stage,
            lambda rows, source: _convert_new_ifeval(
                rows,
                source,
                rm_type="ifevalg",
                verifier_schema=GOOGLE_IFEVAL_SCHEMA,
            ),
            role="heldout_eval",
            schema=GOOGLE_IFEVAL_SCHEMA,
            rm_type="ifevalg",
        )
        dataset_manifest["google_ifeval"] = entry
        prompt_sets["google_ifeval"] = prompts

        entry, prompts = _prepare_one_dataset(
            source_by_name["ifbench_test"],
            raw_paths["ifbench_test"],
            stage,
            lambda rows, source: _convert_new_ifeval(
                rows,
                source,
                rm_type="ifbench",
                verifier_schema=IFBENCH_SCHEMA,
            ),
            role="heldout_eval",
            schema=IFBENCH_SCHEMA,
            rm_type="ifbench",
        )
        dataset_manifest["ifbench_test"] = entry
        prompt_sets["ifbench_test"] = prompts

        entry, prompts = _prepare_one_dataset(
            source_by_name["gsm8k_test"],
            raw_paths["gsm8k_test"],
            stage,
            _convert_gsm8k,
            role="heldout_eval",
            schema=GSM8K_SCHEMA,
            rm_type="gsm8k_verl",
        )
        dataset_manifest["gsm8k_test"] = entry
        prompt_sets["gsm8k_test"] = prompts

        entry, prompts = _prepare_one_dataset(
            source_by_name["math500"],
            raw_paths["math500"],
            stage,
            _convert_math500,
            role="heldout_eval",
            schema=MATH_SCHEMA,
            rm_type="math",
        )
        dataset_manifest["math500"] = entry
        prompt_sets["math500"] = prompts

        exact_overlaps, normalized_overlaps, casefold_overlaps = _check_prompt_leakage_views(
            prompt_sets
        )
        eval_protocols = _eval_protocols(output_root)
        for protocol_name, protocol in eval_protocols.items():
            config_path = protocol["config_path"]
            _atomic_write_text(stage / config_path, _eval_config_text(output_root, protocol_name))
            protocol["config_sha256"] = _sha256(stage / config_path)
            protocol["config_bytes"] = (stage / config_path).stat().st_size
            for dataset in protocol["datasets"]:
                dataset_record = dataset_manifest[dataset["source_dataset"]]
                dataset["artifact_sha256"] = dataset_record["sha256"]
                dataset["artifact_rows"] = dataset_record["output_rows"]

        runtime_report = _validate_runtime_contracts(stage)
        dependency_manifest["runtime_contract_validator"] = {
            "repo_relative_path": RUNTIME_VALIDATOR.resolve()
            .relative_to(Path(__file__).resolve().parents[2])
            .as_posix(),
            "sha256": _sha256(RUNTIME_VALIDATOR),
        }
        dependency_manifest["runtime_contract_report"] = runtime_report
        # Import probes and real reward smoke can create evaluator-local cache
        # directories. Hash the final immutable evaluator trees afterwards.
        for record in code_manifest.values():
            record["tree_sha256"] = _tree_sha256(stage / record["destination"])

        shutil.rmtree(raw_root)
        generator_path = Path(__file__).resolve()
        artifact_paths = list(OUTPUT_FILES.values()) + list(EVAL_CONFIG_FILES)
        manifest: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "generator": {
                "repo_relative_path": generator_path.relative_to(generator_path.parents[2]).as_posix(),
                "sha256": _sha256(generator_path),
                "runtime_validator_repo_relative_path": RUNTIME_VALIDATOR.resolve()
                .relative_to(generator_path.parents[2])
                .as_posix(),
                "runtime_validator_sha256": _sha256(RUNTIME_VALIDATOR),
            },
            "compute_allocation": allocation,
            "source_code": source_code,
            "output_root": str(output_root),
            "policy": {
                "one_physical_row_per_prompt": True,
                "rollout_preexpanded": False,
                "rollout_multiplicity_owner": "Miles --n-samples-per-prompt at runtime",
                "schema_translation": "forbidden",
                "train_eval_prompt_overlap_allowed": False,
                "hard_leakage_normalization": "Unicode NFC then outer-whitespace strip",
                "casefold_overlap": "diagnostic only",
                "primary_if_train": "rlvr_ifeval_train",
                "fallback_if_train": "if_multi_fallback_train",
                "fallback_activation": "preregistered base-difficulty gate only; never pooled",
            },
            "metric_semantics": {
                "google_ifeval": {
                    "online_rm_type": "ifevalg",
                    "online_scalar": "per-instruction mean diagnostic",
                    "official_reporting": (
                        "Run prompt-level and instruction-level strict/loose metrics with the pinned "
                        "google-research instruction_following_eval checkout."
                    ),
                    "prohibited_claim": (
                        "Do not label the online per-instruction mean as official Google prompt-level accuracy."
                    ),
                },
                "ifbench_test": {
                    "online_rm_type": "ifbench",
                    "official_reporting": "Pinned IFBench strict/loose evaluator; prompt-level loose is primary.",
                },
            },
            "reward_stabilization": {
                "langdetect_detector_factory_seed": 0,
                "implementation_owner": "Miles verifier loader; upstream evaluator checkout is unmodified",
            },
            "sources": source_manifest,
            "verifier_code": code_manifest,
            "verifier_dependencies": dependency_manifest,
            "datasets": dataset_manifest,
            "eval_protocols": eval_protocols,
            "exact_prompt_overlap_counts": exact_overlaps,
            "nfc_strip_prompt_overlap_counts": normalized_overlaps,
            "casefold_prompt_overlap_diagnostics": casefold_overlaps,
            "files": _manifest_file_records(stage, artifact_paths),
        }
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _atomic_write_text(stage / "manifest.json", manifest_text)

        # The absent-target check above plus rename avoids replacing existing data.
        stage.rename(output_root)
        print(f"PUBLISHED output_root={output_root}", flush=True)
        print(f"MANIFEST_SHA256={_sha256(output_root / 'manifest.json')}", flush=True)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        return manifest
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _validate_prepared_old(rows: list[dict[str, Any]], expected: dict[str, Any]) -> set[str]:
    prompts = _ensure_unique_prompts(rows, "prepared.rlvr_ifeval_train")
    for index, row in enumerate(rows):
        _require_fields(row, {"prompt", "label", "metadata"}, "prepared.rlvr_ifeval_train", index)
        metadata = row["metadata"]
        if not isinstance(metadata, dict):
            raise ValidationError(f"prepared old IF row {index}: metadata must be an object")
        if metadata.get("rm_type") != "ifeval_old" or metadata.get("verifier_schema") != OLD_IFEVAL_SCHEMA:
            raise ValidationError(f"prepared old IF row {index}: verifier routing mismatch")
        label = row["label"]
        try:
            contract = json.loads(label)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"prepared old IF row {index}: invalid label JSON") from exc
        if not isinstance(contract, dict) or not isinstance(contract.get("func_name"), str):
            raise ValidationError(f"prepared old IF row {index}: func_name contract missing")
        if any(key in contract for key in ("instruction_id", "instruction_id_list", "kwargs")):
            raise ValidationError(f"prepared old IF row {index}: mixed verifier contract")
    if len(rows) != expected["output_rows"]:
        raise ValidationError("prepared old IF row count differs from manifest")
    return prompts


def _validate_prepared_if_multi(rows: list[dict[str, Any]], expected: dict[str, Any]) -> set[str]:
    name = "prepared.if_multi_fallback_train"
    prompts = _ensure_unique_prompts(rows, name)
    for index, row in enumerate(rows):
        _require_fields(row, {"prompt", "label", "metadata"}, name, index)
        metadata = row["metadata"]
        if not isinstance(metadata, dict):
            raise ValidationError(f"{name}[{index}]: metadata must be an object")
        if metadata.get("rm_type") != "ifevalg" or metadata.get("verifier_schema") != IF_MULTI_SCHEMA:
            raise ValidationError(f"{name}[{index}]: verifier routing mismatch")
        if metadata.get("candidate_status") != "preregistered_fallback_only":
            raise ValidationError(f"{name}[{index}]: fallback status missing")
        if metadata.get("ground_truth") != row["label"]:
            raise ValidationError(f"{name}[{index}]: label/ground_truth serialization mismatch")
        try:
            payload = json.loads(row["label"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"{name}[{index}]: invalid canonical ground_truth JSON") from exc
        if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
            raise ValidationError(f"{name}[{index}]: ground_truth must remain a one-element list")
        contract = payload[0]
        if "func_name" in contract or "instruction_id_list" in contract:
            raise ValidationError(f"{name}[{index}]: mixed verifier contract")
        ids = contract.get("instruction_id")
        kwargs = contract.get("kwargs")
        if not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids):
            raise ValidationError(f"{name}[{index}]: invalid instruction_id")
        if not isinstance(kwargs, list) or len(kwargs) != len(ids):
            raise ValidationError(f"{name}[{index}]: kwargs alignment mismatch")
        if any(item is not None and not isinstance(item, dict) for item in kwargs):
            raise ValidationError(f"{name}[{index}]: kwargs entries must be objects or null")
        if metadata.get("instruction_id") != ids or metadata.get("kwargs") != kwargs:
            raise ValidationError(f"{name}[{index}]: label/metadata verifier contract mismatch")
        source_keys = metadata.get("source_keys", [metadata.get("source_key")])
        source_indices = metadata.get("source_row_indices", [metadata.get("source_row_index")])
        if not isinstance(source_keys, list) or not source_keys or len(source_keys) != len(source_indices):
            raise ValidationError(f"{name}[{index}]: consolidation provenance mismatch")
        if len({_canonical({"type": type(key).__name__, "value": key}) for key in source_keys}) != len(
            source_keys
        ):
            raise ValidationError(f"{name}[{index}]: duplicate source key in consolidation provenance")
    if len(rows) != expected["output_rows"]:
        raise ValidationError("prepared IF_multi row count differs from manifest")
    return prompts


def _validate_prepared_new(
    rows: list[dict[str, Any]], expected: dict[str, Any], *, rm_type: str, schema: str, name: str
) -> set[str]:
    prompts = _ensure_unique_prompts(rows, f"prepared.{name}")
    typed_keys: set[str] = set()
    for index, row in enumerate(rows):
        _require_fields(row, {"prompt", "label", "metadata"}, f"prepared.{name}", index)
        metadata = row["metadata"]
        if not isinstance(metadata, dict):
            raise ValidationError(f"prepared {name} row {index}: metadata must be an object")
        if metadata.get("rm_type") != rm_type or metadata.get("verifier_schema") != schema:
            raise ValidationError(f"prepared {name} row {index}: verifier routing mismatch")
        if "func_name" in metadata:
            raise ValidationError(f"prepared {name} row {index}: old verifier field present")
        prompt_text = _one_user_prompt(row["prompt"], name, index)[0]["content"]
        if metadata.get("prompt_text") != prompt_text:
            raise ValidationError(f"prepared {name} row {index}: prompt_text mismatch")
        ids = metadata.get("instruction_id_list")
        kwargs = metadata.get("kwargs")
        if not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids):
            raise ValidationError(f"prepared {name} row {index}: invalid instruction ids")
        if not isinstance(kwargs, list) or len(kwargs) != len(ids) or any(not isinstance(x, dict) for x in kwargs):
            raise ValidationError(f"prepared {name} row {index}: invalid kwargs")
        if row["label"] != _canonical({"instruction_id_list": ids, "kwargs": kwargs}):
            raise ValidationError(f"prepared {name} row {index}: label/metadata contract mismatch")
        source_key = metadata.get("source_key")
        record_id = metadata.get("record_id")
        if source_key != record_id or type(source_key) is not type(record_id):
            raise ValidationError(f"prepared {name} row {index}: source key was not preserved losslessly")
        if isinstance(record_id, bool) or not isinstance(record_id, (str, int)):
            raise ValidationError(f"prepared {name} row {index}: invalid source key type")
        identity = _canonical({"type": type(record_id).__name__, "value": record_id})
        if identity in typed_keys:
            raise ValidationError(f"prepared {name} row {index}: duplicate typed source key")
        typed_keys.add(identity)
    if len(rows) != expected["output_rows"]:
        raise ValidationError(f"prepared {name} row count differs from manifest")
    return prompts


def _validate_prepared_gsm8k(rows: list[dict[str, Any]], expected: dict[str, Any]) -> set[str]:
    prompts = _ensure_unique_prompts(rows, "prepared.gsm8k_test")
    for index, row in enumerate(rows):
        if set(row) != {"prompt", "label", "metadata"}:
            raise ValidationError(f"prepared GSM8K row {index}: exact output keys required")
        metadata = row["metadata"]
        if not isinstance(metadata, dict):
            raise ValidationError(f"prepared GSM8K row {index}: metadata must be an object")
        if (
            metadata.get("rm_type") != "gsm8k_verl"
            or metadata.get("verifier_schema") != GSM8K_SCHEMA
            or metadata.get("data_source") != "openai/gsm8k"
            or metadata.get("source_split") != "test"
        ):
            raise ValidationError(f"prepared GSM8K row {index}: verifier/provenance mismatch")
        prompt_text = _one_user_prompt(row["prompt"], "gsm8k_test", index)[0]["content"]
        if not prompt_text.endswith(GSM8K_PROMPT_SUFFIX):
            raise ValidationError(f"prepared GSM8K row {index}: old-Verl prompt suffix mismatch")
        label = row["label"]
        reward_ground_truth = metadata.get("reward_model", {}).get("ground_truth")
        if not isinstance(label, str) or not label or label != reward_ground_truth or "," in label:
            raise ValidationError(f"prepared GSM8K row {index}: reward label mismatch")
        if not isinstance(metadata.get("source_row_index"), int):
            raise ValidationError(f"prepared GSM8K row {index}: source row provenance missing")
    if len(rows) != expected["output_rows"]:
        raise ValidationError("prepared GSM8K row count differs from manifest")
    return prompts


def _validate_prepared_math(rows: list[dict[str, Any]], expected: dict[str, Any]) -> set[str]:
    prompts = _ensure_unique_prompts(rows, "prepared.math500")
    unique_ids: set[str] = set()
    for index, row in enumerate(rows):
        _require_fields(row, {"prompt", "label", "metadata"}, "prepared.math500", index)
        metadata = row["metadata"]
        if not isinstance(metadata, dict):
            raise ValidationError(f"prepared MATH row {index}: metadata must be an object")
        if metadata.get("rm_type") != "math" or metadata.get("verifier_schema") != MATH_SCHEMA:
            raise ValidationError(f"prepared MATH row {index}: verifier routing mismatch")
        if (
            metadata.get("data_source") != "HuggingFaceH4/MATH-500"
            or metadata.get("source_revision") != MATH500_REVISION
            or metadata.get("source_split") != "test"
            or not isinstance(metadata.get("source_row_index"), int)
        ):
            raise ValidationError(f"prepared MATH row {index}: dataset provenance mismatch")
        if not isinstance(row["label"], str) or not row["label"]:
            raise ValidationError(f"prepared MATH row {index}: empty answer")
        unique_id = metadata.get("unique_id")
        if not isinstance(unique_id, str) or not unique_id or unique_id in unique_ids:
            raise ValidationError(f"prepared MATH row {index}: invalid or duplicate unique_id")
        unique_ids.add(unique_id)
    if len(rows) != expected["output_rows"]:
        raise ValidationError("prepared MATH row count differs from manifest")
    return prompts


def verify_dataset_bundle(output_root: Path) -> dict[str, Any]:
    output_root = _require_allowed_output_root(output_root)
    _require_compute_allocation()
    source_code = _require_frozen_source()
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValidationError(f"unsupported manifest version: {manifest.get('format_version')}")
    if manifest.get("output_root") != str(output_root):
        raise ValidationError("manifest output_root does not match requested directory")
    recorded_source_code = manifest.get("source_code", {})
    if (
        recorded_source_code.get("actual_code_sha") != source_code["actual_code_sha"]
        or recorded_source_code.get("expected_code_sha") != source_code["expected_code_sha"]
        or recorded_source_code.get("git_clean") is not True
    ):
        raise ValidationError("bundle source-code SHA differs from current clean frozen worktree")
    generator = manifest.get("generator", {})
    if (
        generator.get("repo_relative_path")
        != Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix()
        or generator.get("sha256") != _sha256(Path(__file__).resolve())
        or generator.get("runtime_validator_repo_relative_path")
        != RUNTIME_VALIDATOR.resolve()
        .relative_to(Path(__file__).resolve().parents[2])
        .as_posix()
        or not RUNTIME_VALIDATOR.is_file()
        or generator.get("runtime_validator_sha256") != _sha256(RUNTIME_VALIDATOR)
    ):
        raise ValidationError("bundle generator/validator identity differs from current frozen source")
    allocation = manifest.get("compute_allocation", {})
    if not (allocation.get("spur_job_id") or allocation.get("slurm_job_id")) or not allocation.get(
        "scheduler_node_list"
    ):
        raise ValidationError("manifest lacks compute-allocation identity")
    policy = manifest.get("policy") or {}
    if policy.get("one_physical_row_per_prompt") is not True or policy.get("rollout_preexpanded") is not False:
        raise ValidationError("manifest does not guarantee one-copy runtime-only rollout multiplicity")
    if policy.get("schema_translation") != "forbidden":
        raise ValidationError("manifest no longer forbids cross-verifier schema translation")
    if manifest.get("reward_stabilization", {}).get("langdetect_detector_factory_seed") != 0:
        raise ValidationError("deterministic langdetect seed is missing")

    expected_pins = {source.name: source.revision for source in FILE_SOURCES}
    if set(manifest.get("sources", {})) != set(expected_pins):
        raise ValidationError("source set differs from the frozen source inventory")
    for name, revision in expected_pins.items():
        source = next(item for item in FILE_SOURCES if item.name == name)
        record = manifest.get("sources", {}).get(name, {})
        if record.get("revision") != revision:
            raise ValidationError(f"source revision mismatch for {name}")
        if (
            record.get("repo_id") != source.repo_id
            or record.get("relative_path") != source.relative_path
            or record.get("url") != source.url
            or record.get("bytes") != source.expected_size
            or record.get("expected_rows") != source.expected_rows
        ):
            raise ValidationError(f"source identity mismatch for {name}")
        expected_identity = source.content_sha256 or source.git_blob_oid
        if record.get("identity_check", {}).get("value") != expected_identity:
            raise ValidationError(f"source immutable identity mismatch for {name}")
        if source.content_sha256 is not None and record.get("sha256") != source.content_sha256:
            raise ValidationError(f"source content SHA-256 mismatch for {name}")
        if source.git_blob_oid is not None and record.get("git_blob_oid") != source.git_blob_oid:
            raise ValidationError(f"source Git blob identity mismatch for {name}")
    expected_code_pins = {source.name: source.revision for source in CODE_SOURCES}
    expected_code_pins[GOOGLE_EVAL_SOURCE.name] = GOOGLE_EVAL_SOURCE.revision
    if set(manifest.get("verifier_code", {})) != set(expected_code_pins):
        raise ValidationError("verifier source set differs from the frozen code inventory")
    for name, revision in expected_code_pins.items():
        record = manifest.get("verifier_code", {}).get(name, {})
        if record.get("revision") != revision:
            raise ValidationError(f"verifier revision mismatch for {name}")
        checkout = output_root / record.get("destination", "")
        if not checkout.is_dir() or _tree_sha256(checkout) != record.get("tree_sha256"):
            raise ValidationError(f"verifier checkout hash mismatch for {name}")
        if name == GOOGLE_EVAL_SOURCE.name:
            if (
                record.get("owner") != GOOGLE_EVAL_SOURCE.owner
                or record.get("repository") != GOOGLE_EVAL_SOURCE.repository
                or record.get("destination") != GOOGLE_EVAL_SOURCE.destination
                or set(record.get("files", {}))
                != {Path(spec.relative_path).name for spec in GOOGLE_EVAL_SOURCE.files}
            ):
                raise ValidationError(f"verifier source identity mismatch for {name}")
            for file_spec in GOOGLE_EVAL_SOURCE.files:
                basename = Path(file_spec.relative_path).name
                file_record = record["files"].get(basename, {})
                if (
                    file_record.get("upstream_path") != file_spec.relative_path
                    or file_record.get("bytes") != file_spec.expected_size
                    or file_record.get("git_blob_oid") != file_spec.git_blob_oid
                    or file_record.get("url") != GOOGLE_EVAL_SOURCE.url(file_spec.relative_path)
                ):
                    raise ValidationError(f"verifier file identity mismatch for {name}/{basename}")
        else:
            source = next(item for item in CODE_SOURCES if item.name == name)
            if (
                record.get("owner") != source.owner
                or record.get("repository") != source.repository
                or record.get("destination") != source.destination
                or record.get("required_files") != list(source.required_files)
            ):
                raise ValidationError(f"verifier source identity mismatch for {name}")

    dependencies = manifest.get("verifier_dependencies", {})
    if dependencies.get("fixed_container_image") != EXPECTED_CONTAINER_IMAGE:
        raise ValidationError("dependency environment was not built in the fixed container image")
    if sys.version_info[:2] != EXPECTED_CONTAINER_PYTHON or not str(
        dependencies.get("python_version", "")
    ).startswith("3.10"):
        raise ValidationError("dependency environment is not the validated container Python 3.10")
    requirements = Path(__file__).with_name("verifier_requirements.lock").resolve()
    if not requirements.is_file() or _sha256(requirements) != dependencies.get("requirements_lock_sha256"):
        raise ValidationError("current verifier dependency lock differs from the bundle")
    expected_target_distributions = _parse_requirements_lock(requirements)
    target_distributions = dependencies.get("verifier_only_distributions", {})
    if target_distributions != expected_target_distributions:
        raise ValidationError("dependency target does not exactly match the frozen verifier-only lock")
    if dependencies.get("pip_dependency_resolution") != (
        "disabled (--no-deps, --no-build-isolation); download and install both enforce "
        "--require-hashes"
    ):
        raise ValidationError("dependency manifest does not prove hash-locked --no-deps provisioning")
    if dependencies.get("requirements_pypi_sha256") != _requirements_lock_hashes(requirements):
        raise ValidationError("dependency manifest differs from the declared official PyPI hashes")
    if dependencies.get("requirements_index") != "https://pypi.org/simple":
        raise ValidationError("dependency manifest does not pin the official PyPI simple index")
    if dependencies.get("ifbench_upstream_unused_requirements") != _ifbench_unused_requirement_report(
        output_root
    ):
        raise ValidationError("IFBench upstream-unused dependency proof differs from the frozen sources")
    if dependencies.get("nltk_compatibility") != {
        "selected_version": "3.9.4",
        "official_pypi_requires_python": ">=3.10",
        "ifbench_1091c4c_requirement": "nltk (no upper bound), Python >=3.10",
        "open_instruct_5b2ebfa_package_requirement": "nltk>=3.9.1, Python ==3.12.*",
        "runtime_scope": (
            "Vendored verifier modules only; Open Instruct package installation/support is not claimed. "
            "Compatibility on container Python 3.10 is established by registry/signature validation "
            "and real Miles async_rm smoke."
        ),
    }:
        raise ValidationError("dependency manifest lacks the frozen NLTK compatibility contract")
    protected = set(dependencies.get("protected_base_distributions", []))
    if protected.intersection(target_distributions):
        raise ValidationError("dependency target shadows protected fixed-image packages")
    validator_record = dependencies.get("runtime_contract_validator", {})
    if validator_record != {
        "repo_relative_path": RUNTIME_VALIDATOR.resolve()
        .relative_to(Path(__file__).resolve().parents[2])
        .as_posix(),
        "sha256": _sha256(RUNTIME_VALIDATOR),
    }:
        raise ValidationError("runtime validator identity mismatch in dependency manifest")
    dependency_paths = (
        ("python_target", "python_target_tree_sha256"),
        ("wheelhouse", "wheelhouse_tree_sha256"),
        ("nltk_data", "nltk_data_tree_sha256"),
    )
    for path_key, hash_key in dependency_paths:
        relative = dependencies.get(path_key)
        if not isinstance(relative, str):
            raise ValidationError(f"dependency manifest lacks {path_key}")
        path = output_root / relative
        if not path.is_dir() or _tree_sha256(path) != dependencies.get(hash_key):
            raise ValidationError(f"dependency tree hash mismatch: {path_key}")
    freeze_path = output_root / dependencies.get("freeze_path", "")
    if not freeze_path.is_file() or _sha256(freeze_path) != dependencies.get("freeze_sha256"):
        raise ValidationError("dependency freeze checksum mismatch")
    distribution_artifacts = dependencies.get("distribution_artifacts", {})
    if not distribution_artifacts or distribution_artifacts != dependencies.get("wheel_files"):
        raise ValidationError("dependency manifest has no consistent distribution-artifact hashes")
    declared_hashes = {
        digest
        for package_hashes in _requirements_lock_hashes(requirements).values()
        for digest in package_hashes
    }
    if {record.get("sha256") for record in distribution_artifacts.values()} != declared_hashes:
        raise ValidationError("downloaded dependency artifacts differ from official PyPI hash lock")
    for relative, expected in distribution_artifacts.items():
        wheel = output_root / dependencies["wheelhouse"] / relative
        if (
            not wheel.is_file()
            or wheel.stat().st_size != expected.get("bytes")
            or _sha256(wheel) != expected.get("sha256")
        ):
            raise ValidationError(f"dependency wheel checksum mismatch: {relative}")
    nltk_archives = dependencies.get("nltk_resource_archives", {})
    expected_nltk_source = {
        "repository": "nltk/nltk_data",
        "revision": NLTK_DATA_REVISION,
        "resources": {
            name: {
                "revision": NLTK_DATA_REVISION,
                "url": _nltk_resource_url(name, spec),
                "archive_path": f"{spec['subdir']}/{name}.zip",
                "bytes": spec["bytes"],
                "sha256": spec["sha256"],
            }
            for name, spec in NLTK_DATA_RESOURCES.items()
        },
    }
    if dependencies.get("nltk_data_source") != expected_nltk_source:
        raise ValidationError("NLTK data source revision/URL/hash contract mismatch")
    if set(nltk_archives) != {
        f"{spec['subdir']}/{name}.zip" for name, spec in NLTK_DATA_RESOURCES.items()
    }:
        raise ValidationError("dependency manifest lacks the exact pinned NLTK resource archive set")
    expected_nltk_archives = {
        f"{spec['subdir']}/{name}.zip": {"bytes": spec["bytes"], "sha256": spec["sha256"]}
        for name, spec in NLTK_DATA_RESOURCES.items()
    }
    if nltk_archives != expected_nltk_archives:
        raise ValidationError("NLTK archive inventory differs from predeclared SHA-256 identities")
    for relative, expected in nltk_archives.items():
        archive = output_root / dependencies["nltk_data"] / relative
        if (
            not archive.is_file()
            or archive.stat().st_size != expected.get("bytes")
            or _sha256(archive) != expected.get("sha256")
        ):
            raise ValidationError(f"NLTK resource checksum mismatch: {relative}")

    file_records = manifest.get("files", {})
    expected_files = set(OUTPUT_FILES.values()) | set(EVAL_CONFIG_FILES)
    if set(file_records) != expected_files:
        raise ValidationError("prepared artifact set differs from the frozen inventory")
    for relative, expected in file_records.items():
        path = output_root / relative
        if not path.is_file():
            raise ValidationError(f"prepared artifact missing: {relative}")
        if path.stat().st_size != expected.get("bytes") or _sha256(path) != expected.get("sha256"):
            raise ValidationError(f"prepared artifact checksum mismatch: {relative}")

    datasets = manifest.get("datasets", {})
    required_datasets = set(OUTPUT_FILES)
    if set(datasets) != required_datasets:
        raise ValidationError(f"manifest dataset keys mismatch: {sorted(datasets)}")
    for name, record in datasets.items():
        source = next(item for item in FILE_SOURCES if item.name == name)
        if (
            record.get("source") != name
            or record.get("source_revision") != source.revision
            or record.get("raw_rows") != source.expected_rows
            or record.get("output_path") != OUTPUT_FILES[name]
        ):
            raise ValidationError(f"dataset {name} source/generator identity mismatch")
        if record.get("physical_rows_per_prompt") != 1 or record.get("rollout_preexpanded") is not False:
            raise ValidationError(f"dataset {name} violates one-copy policy")
        if record.get("output_exact_prompt_duplicates") != 0:
            raise ValidationError(f"dataset {name} records output duplicates")
        integer_metrics = (
            "raw_rows",
            "exact_raw_record_duplicates_removed",
            "output_rows",
            "prompt_duplicate_rows_removed",
            "prompt_identical_contract_duplicates_removed",
            "prompt_contract_consolidation_groups",
            "constraints_deduplicated_during_consolidation",
        )
        if any(not isinstance(record.get(metric), int) or record[metric] < 0 for metric in integer_metrics):
            raise ValidationError(f"dataset {name} has malformed deduplication metrics")
        if (
            record["raw_rows"]
            - record["exact_raw_record_duplicates_removed"]
            - record["prompt_duplicate_rows_removed"]
            != record["output_rows"]
        ):
            raise ValidationError(f"dataset {name} deduplication accounting does not balance")
        output_path = record.get("output_path")
        file_record = file_records.get(output_path, {})
        if record.get("bytes") != file_record.get("bytes") or record.get("sha256") != file_record.get(
            "sha256"
        ):
            raise ValidationError(f"dataset {name} identity disagrees with file inventory")
        if name != "if_multi_fallback_train" and any(
            record.get(metric) != 0
            for metric in (
                "prompt_duplicate_rows_removed",
                "prompt_identical_contract_duplicates_removed",
                "prompt_contract_consolidation_groups",
                "constraints_deduplicated_during_consolidation",
            )
        ):
            raise ValidationError(f"dataset {name} unexpectedly consolidated prompt records")
    expected_contracts = {
        "rlvr_ifeval_train": ("ifeval_old", OLD_IFEVAL_SCHEMA, "train"),
        "if_multi_fallback_train": ("ifevalg", IF_MULTI_SCHEMA, "preregistered_train_fallback"),
        "google_ifeval": ("ifevalg", GOOGLE_IFEVAL_SCHEMA, "heldout_eval"),
        "ifbench_test": ("ifbench", IFBENCH_SCHEMA, "heldout_eval"),
        "gsm8k_test": ("gsm8k_verl", GSM8K_SCHEMA, "heldout_eval"),
        "math500": ("math", MATH_SCHEMA, "heldout_eval"),
    }
    for name, (rm_type, schema, role) in expected_contracts.items():
        record = datasets[name]
        if (record.get("rm_type"), record.get("verifier_schema"), record.get("role")) != (
            rm_type,
            schema,
            role,
        ):
            raise ValidationError(f"dataset contract mismatch: {name}")

    prompt_sets = {
        "rlvr_ifeval_train": _validate_prepared_old(
            _read_jsonl(output_root / OUTPUT_FILES["rlvr_ifeval_train"]),
            datasets["rlvr_ifeval_train"],
        ),
        "if_multi_fallback_train": _validate_prepared_if_multi(
            _read_jsonl(output_root / OUTPUT_FILES["if_multi_fallback_train"]),
            datasets["if_multi_fallback_train"],
        ),
        "google_ifeval": _validate_prepared_new(
            _read_jsonl(output_root / OUTPUT_FILES["google_ifeval"]),
            datasets["google_ifeval"],
            rm_type="ifevalg",
            schema=GOOGLE_IFEVAL_SCHEMA,
            name="google_ifeval",
        ),
        "ifbench_test": _validate_prepared_new(
            _read_jsonl(output_root / OUTPUT_FILES["ifbench_test"]),
            datasets["ifbench_test"],
            rm_type="ifbench",
            schema=IFBENCH_SCHEMA,
            name="ifbench_test",
        ),
        "gsm8k_test": _validate_prepared_gsm8k(
            _read_jsonl(output_root / OUTPUT_FILES["gsm8k_test"]),
            datasets["gsm8k_test"],
        ),
        "math500": _validate_prepared_math(
            _read_jsonl(output_root / OUTPUT_FILES["math500"]), datasets["math500"]
        ),
    }
    exact_overlaps, normalized_overlaps, casefold_overlaps = _check_prompt_leakage_views(prompt_sets)
    if exact_overlaps != manifest.get("exact_prompt_overlap_counts"):
        raise ValidationError("recomputed prompt overlap matrix differs from manifest")
    if normalized_overlaps != manifest.get("nfc_strip_prompt_overlap_counts"):
        raise ValidationError("recomputed NFC+strip leakage matrix differs from manifest")
    if casefold_overlaps != manifest.get("casefold_prompt_overlap_diagnostics"):
        raise ValidationError("recomputed casefold diagnostic matrix differs from manifest")
    expected_protocols = _eval_protocols(output_root)
    recorded_protocols = manifest.get("eval_protocols", {})
    if set(recorded_protocols) != {"math", "nonmath"}:
        raise ValidationError("manifest must contain exactly math and nonmath eval protocols")
    math_set = set(recorded_protocols["math"].get("dataset_set", []))
    nonmath_set = set(recorded_protocols["nonmath"].get("dataset_set", []))
    if math_set != {"gsm8k", "math500"} or nonmath_set != {
        "google_ifeval",
        "ifbench_test",
    }:
        raise ValidationError("eval protocol dataset sets differ from the frozen experiment contract")
    if math_set.intersection(nonmath_set):
        raise ValidationError("math and nonmath eval dataset sets must be disjoint")
    for protocol_name, expected_protocol in expected_protocols.items():
        recorded = recorded_protocols[protocol_name]
        config_path = expected_protocol["config_path"]
        config_file = output_root / config_path
        for field, expected_value in expected_protocol.items():
            if field != "datasets" and recorded.get(field) != expected_value:
                raise ValidationError(
                    f"eval protocol field mismatch: {protocol_name}.{field}"
                )
        if (
            recorded.get("config_path") != config_path
            or recorded.get("config_sha256") != _sha256(config_file)
            or recorded.get("config_bytes") != config_file.stat().st_size
        ):
            raise ValidationError(f"eval protocol config identity mismatch: {protocol_name}")
        if config_file.read_text(encoding="utf-8") != _eval_config_text(output_root, protocol_name):
            raise ValidationError(f"eval protocol config content mismatch: {protocol_name}")
        expected_datasets = expected_protocol["datasets"]
        recorded_datasets = recorded.get("datasets")
        if not isinstance(recorded_datasets, list) or len(recorded_datasets) != len(
            expected_datasets
        ):
            raise ValidationError(f"eval protocol dataset cardinality mismatch: {protocol_name}")
        for expected_dataset, recorded_dataset in zip(
            expected_datasets, recorded_datasets, strict=True
        ):
            eval_name = expected_dataset["name"]
            expected_rm_type = EVAL_DATASET_RM_TYPES.get(eval_name)
            if (
                expected_rm_type is None
                or expected_dataset.get("rm_type") != expected_rm_type
                or expected_dataset.get("metadata_overrides")
                != {EVAL_DATASET_METADATA_KEY: eval_name}
            ):
                raise ValidationError(
                    f"internal frozen eval route is invalid: {protocol_name}/{eval_name}"
                )
            if (
                recorded_dataset.get("name") != eval_name
                or recorded_dataset.get("rm_type") != expected_rm_type
                or recorded_dataset.get("metadata_overrides")
                != {EVAL_DATASET_METADATA_KEY: eval_name}
            ):
                raise ValidationError(
                    f"eval dataset attribution/reward route mismatch: "
                    f"{protocol_name}/{eval_name}"
                )
            source_dataset = expected_dataset["source_dataset"]
            expected_with_identity = {
                **expected_dataset,
                "artifact_sha256": datasets[source_dataset]["sha256"],
                "artifact_rows": datasets[source_dataset]["output_rows"],
            }
            if recorded_dataset != expected_with_identity:
                raise ValidationError(
                    f"eval protocol sampling/data identity mismatch: {protocol_name}/"
                    f"{expected_dataset['name']}"
                )
    runtime_report = _validate_runtime_contracts(output_root)
    if runtime_report != dependencies.get("runtime_contract_report"):
        raise ValidationError("recomputed runtime verifier/reward report differs from manifest")
    print(f"VERIFY_RESULT=PASS output_root={output_root}", flush=True)
    print(f"MANIFEST_SHA256={_sha256(manifest_path)}", flush=True)
    print(
        json.dumps(
            {
                "datasets": datasets,
                "exact_prompt_overlap_counts": exact_overlaps,
                "nfc_strip_prompt_overlap_counts": normalized_overlaps,
                "casefold_prompt_overlap_diagnostics": casefold_overlaps,
                "runtime_contract_report": runtime_report,
            },
            indent=2,
        ),
        flush=True,
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="perform offline checksum/schema/disjointness validation of an existing bundle",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.verify_only:
        verify_dataset_bundle(args.output_root)
    else:
        create_dataset_bundle(args.output_root)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"DATA_PREP_RESULT=FAIL type={type(exc).__name__} message={exc}", file=sys.stderr, flush=True)
        raise
    else:
        print("DATA_PREP_RESULT=PASS", flush=True)
