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
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable


FORMAT_VERSION = 1
DEFAULT_OUTPUT_ROOT = Path("/home/daidong/rebuttal_workspace/data/pr2-rebuttal-one-copy-v1")
EXPECTED_CONTAINER_IMAGE = (
    "docker.io/rlsys/miles@sha256:e60a69faa831ae2a146819290e8469cf23c81b97febeaf5d11e8baec2bfca285"
)

RLVR_IFEVAL_REVISION = "47c03c73621c4aab2b824b7818681117d662770e"
IF_MULTI_REVISION = "2e3a77407b7fce69f95b248d64a884e3ae1c2423"
IFBENCH_TEST_REVISION = "2e8a48de45ff3bf41242f927254ca81b59ca3ae2"
GOOGLE_IFEVAL_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
IFBENCH_CODE_REVISION = "1091c4c3de6c1f6ed12c012ed68f11ea450b0117"
OPEN_INSTRUCT_REVISION = "5b2ebfa12381925bb431845d588dbc9ebead20a7"
GOOGLE_RESEARCH_REVISION = "ec7c3d346277b737bc2decffcd1b533d4b7ec105"

OLD_IFEVAL_SCHEMA = "open_instruct.IFEvalVerifierOld.func_name.v1"
IF_MULTI_SCHEMA = "open_instruct.IFEvalG.one_element_ground_truth_list.v1"
GOOGLE_IFEVAL_SCHEMA = "open_instruct.IFEvalG.instruction_id_list_kwargs.v1"
IFBENCH_SCHEMA = "ifbench.instruction_id_list_kwargs.v1"
MATH_SCHEMA = "miles.math.boxed_equivalence.v1"

MATH_PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem."
)
MATH_PROMPT_SUFFIX = 'Remember to put your answer on its own line after "Answer:".'


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
    "math500": "math500_test.jsonl",
}


class ValidationError(RuntimeError):
    """Raised when an upstream or prepared artifact violates its contract."""


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
    """Build an immutable shared verifier dependency layer once, before workers."""

    requirements = Path(__file__).with_name("verifier_requirements.lock").resolve()
    if not requirements.is_file():
        raise FileNotFoundError(f"verifier dependency lock is missing: {requirements}")
    inventory_names = (
        "absl-py",
        "emoji",
        "httpx",
        "immutabledict",
        "langdetect",
        "nltk",
        "pydantic",
        "pydantic-settings",
        "setuptools",
        "spacy",
        "syllapy",
        "tqdm",
        "unicodedata2",
    )
    base_image_inventory: dict[str, str | None] = {}
    for name in inventory_names:
        try:
            base_image_inventory[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            base_image_inventory[name] = None
    third_party = stage / "third_party"
    python_target = third_party / "python"
    wheelhouse = third_party / "wheelhouse"
    nltk_data = third_party / "nltk_data"
    python_target.mkdir(parents=True)
    wheelhouse.mkdir(parents=True)
    nltk_data.mkdir(parents=True)

    pip_base = [sys.executable, "-m", "pip", "--disable-pip-version-check", "--no-cache-dir"]
    _run_checked(pip_base + ["wheel", "--wheel-dir", str(wheelhouse), "--requirement", str(requirements)])
    _run_checked(
        pip_base
        + [
            "install",
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
    setup_env["PYTHONPATH"] = str(python_target)
    setup_env["NLTK_DATA"] = str(nltk_data)
    setup_env["PYTHONDONTWRITEBYTECODE"] = "1"
    resources = ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger_eng")
    _run_checked(
        [sys.executable, "-m", "nltk.downloader", "-d", str(nltk_data), *resources],
        env=setup_env,
    )

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
        "import absl,emoji,httpx,immutabledict,langdetect,nltk,pydantic,"
        "pydantic_settings,syllapy,tqdm,unicodedata2; "
        "print('VERIFIER_DEPENDENCY_IMPORT_PROBE=PASS')"
    )
    _run_checked([sys.executable, "-c", base_probe], env=setup_env)

    evaluator_probes = (
        (
            [str(stage / "third_party/IFBench"), str(python_target)],
            "import evaluation_lib; assert hasattr(evaluation_lib, 'test_instruction_following_loose'); "
            "print('IFBENCH_IMPORT_PROBE=PASS')",
        ),
        (
            [str(stage / "third_party/open-instruct"), str(python_target)],
            "from open_instruct import if_functions; "
            "from open_instruct.IFEvalG import instructions_registry; "
            "assert if_functions.IF_FUNCTIONS_MAP and instructions_registry.INSTRUCTION_DICT; "
            "print('OPEN_INSTRUCT_IFEVAL_IMPORT_PROBE=PASS')",
        ),
        (
            [str(stage / "third_party/google-research"), str(python_target)],
            "from instruction_following_eval import evaluation_lib; "
            "assert hasattr(evaluation_lib, 'test_instruction_following_strict'); "
            "assert hasattr(evaluation_lib, 'test_instruction_following_loose'); "
            "print('GOOGLE_IFEVAL_IMPORT_PROBE=PASS')",
        ),
    )
    for paths, code in evaluator_probes:
        probe_env = dict(setup_env)
        probe_env["PYTHONPATH"] = os.pathsep.join(paths)
        _run_checked([sys.executable, "-c", code], env=probe_env)

    wheels = _files_with_hashes(wheelhouse, "*.whl")
    if not wheels:
        raise ValidationError("dependency setup produced no wheels")
    nltk_archives = _files_with_hashes(nltk_data, "*.zip")
    if len(nltk_archives) < len(resources):
        raise ValidationError(
            f"expected at least {len(resources)} pinned NLTK resource archives, got {len(nltk_archives)}"
        )
    return {
        "requirements_lock_source": str(requirements),
        "requirements_lock_sha256": _sha256(requirements),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "fixed_container_image": EXPECTED_CONTAINER_IMAGE,
        "base_image_package_inventory": base_image_inventory,
        "python_target": "third_party/python",
        "python_target_tree_sha256": _tree_sha256(python_target),
        "freeze_path": "third_party/verifier_requirements.freeze.txt",
        "freeze_sha256": _sha256(freeze_path),
        "wheelhouse": "third_party/wheelhouse",
        "wheel_files": wheels,
        "wheelhouse_tree_sha256": _tree_sha256(wheelhouse),
        "nltk_data": "third_party/nltk_data",
        "nltk_resources": list(resources),
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
            "Pinned IFBench pyproject dependencies plus all evaluator runtime imports; "
            "the legacy requirements.txt-only spacy entry is excluded because the pinned "
            "pyproject and evaluator source do not import it."
        ),
    }


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
    for index, row in enumerate(rows):
        prompt = _one_user_prompt(row.get("prompt"), dataset, index)[0]["content"]
        if prompt in prompts:
            raise ValidationError(f"{dataset}[{index}]: exact duplicate prompt remains after deduplication")
        prompts.add(prompt)
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
    required = {"key", "messages", "ground_truth", "dataset", "constraint_type", "constraint"}
    for index, row in enumerate(rows):
        _require_fields(row, required, source.name, index)
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
        if "func_name" in contract or "instruction_id_list" in contract:
            raise ValidationError(f"{source.name}[{index}]: mixed or unexpected verifier schema")
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
        for field in ("dataset", "constraint_type", "constraint"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValidationError(f"{source.name}[{index}]: {field} must be non-empty text")

        canonical_payload = [{"instruction_id": list(instruction_ids), "kwargs": list(kwargs)}]
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
                    "source_key": row["key"],
                    "source_dataset": row["dataset"],
                    "constraint_type": row["constraint_type"],
                    "constraint": row["constraint"],
                    "prompt_text": prompt_text,
                    "ground_truth": canonical_ground_truth,
                    "source_ground_truth": raw_ground_truth,
                    "instruction_id": list(instruction_ids),
                    "kwargs": list(kwargs),
                    "candidate_status": "preregistered_fallback_only",
                },
            }
        )
    _ensure_unique_prompts(converted, source.name)
    return converted


def _normalize_new_contract(
    row: dict[str, Any], source: FileSource, index: int
) -> tuple[int, str, list[str], list[dict[str, Any]]]:
    _require_fields(row, {"key", "prompt", "instruction_id_list", "kwargs"}, source.name, index)
    if "ground_truth" in row or "func_name" in row:
        raise ValidationError(f"{source.name}[{index}]: old verifier fields appeared in new-schema data")
    try:
        record_id = int(row["key"])
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{source.name}[{index}]: key must be integer-compatible") from exc
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
    return record_id, prompt, list(instruction_ids), [dict(item) for item in kwargs]


def _convert_new_ifeval(
    rows: list[dict[str, Any]],
    source: FileSource,
    *,
    rm_type: str,
    verifier_schema: str,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        record_id, prompt_text, instruction_ids, kwargs = _normalize_new_contract(row, source, index)
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
                    "source_key": str(row["key"]),
                    "record_id": record_id,
                    "prompt_text": prompt_text,
                    "instruction_id_list": instruction_ids,
                    "kwargs": kwargs,
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


def _eval_config_text(output_root: Path) -> str:
    datasets = (
        ("math500", OUTPUT_FILES["math500"], "math", 1, 1024),
        ("google_ifeval", OUTPUT_FILES["google_ifeval"], "ifevalg", 1, 1024),
        ("ifbench_test", OUTPUT_FILES["ifbench_test"], "ifbench", 1, 1024),
    )
    lines = [
        "# Generated by prepare_rebuttal_data.py; paths are immutable after publication.",
        "eval:",
        "  defaults:",
        "    input_key: prompt",
        "    label_key: label",
        "    metadata_key: metadata",
        "    n_samples_per_eval_prompt: 1",
        "    temperature: 0.0",
        "    top_p: 1.0",
        "  datasets:",
    ]
    for name, filename, rm_type, n_samples, response_len in datasets:
        lines.extend(
            (
                f"    - name: {name}",
                f"      path: {_canonical(str(output_root / filename))}",
                f"      rm_type: {rm_type}",
                f"      n_samples_per_eval_prompt: {n_samples}",
                f"      max_response_len: {response_len}",
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
) -> dict[str, Any]:
    return {
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


def _prepare_one_dataset(
    source: FileSource,
    raw_path: Path,
    stage: Path,
    converter: Callable[[list[dict[str, Any]], FileSource], list[dict[str, Any]]],
    *,
    role: str,
    schema: str,
    rm_type: str,
) -> tuple[dict[str, Any], set[str]]:
    raw_rows = _load_raw(source, raw_path)
    unique_raw, duplicates = _deduplicate_exact_raw(raw_rows, source.name, source.expected_rows)
    converted = converter(unique_raw, source)
    if len(converted) != len(unique_raw):
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
    )
    return entry, prompts


def _manifest_file_records(stage: Path, relative_paths: Iterable[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative in sorted(relative_paths):
        path = stage / relative
        records[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return records


def create_dataset_bundle(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
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
        # Import probes may create empty evaluator-local cache directories.  The
        # tree hash is recomputed after all setup so verification sees final state.
        for record in code_manifest.values():
            record["tree_sha256"] = _tree_sha256(stage / record["destination"])

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

        overlaps = _check_disjoint(prompt_sets)
        eval_config_name = "eval_config.yaml"
        _atomic_write_text(stage / eval_config_name, _eval_config_text(output_root))

        shutil.rmtree(raw_root)
        generator_path = Path(__file__).resolve()
        artifact_paths = list(OUTPUT_FILES.values()) + [eval_config_name]
        manifest: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "generator": {"path": str(generator_path), "sha256": _sha256(generator_path)},
            "output_root": str(output_root),
            "policy": {
                "one_physical_row_per_prompt": True,
                "rollout_preexpanded": False,
                "rollout_multiplicity_owner": "Miles --n-samples-per-prompt at runtime",
                "schema_translation": "forbidden",
                "train_eval_prompt_overlap_allowed": False,
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
            "exact_prompt_overlap_counts": overlaps,
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
    if len(rows) != expected["output_rows"]:
        raise ValidationError("prepared IF_multi row count differs from manifest")
    return prompts


def _validate_prepared_new(
    rows: list[dict[str, Any]], expected: dict[str, Any], *, rm_type: str, schema: str, name: str
) -> set[str]:
    prompts = _ensure_unique_prompts(rows, f"prepared.{name}")
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
    if len(rows) != expected["output_rows"]:
        raise ValidationError(f"prepared {name} row count differs from manifest")
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
    output_root = output_root.resolve()
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValidationError(f"unsupported manifest version: {manifest.get('format_version')}")
    if manifest.get("output_root") != str(output_root):
        raise ValidationError("manifest output_root does not match requested directory")
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
        if manifest.get("sources", {}).get(name, {}).get("revision") != revision:
            raise ValidationError(f"source revision mismatch for {name}")
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

    dependencies = manifest.get("verifier_dependencies", {})
    if dependencies.get("fixed_container_image") != EXPECTED_CONTAINER_IMAGE:
        raise ValidationError("dependency environment was not built in the fixed container image")
    requirements = Path(__file__).with_name("verifier_requirements.lock").resolve()
    if not requirements.is_file() or _sha256(requirements) != dependencies.get("requirements_lock_sha256"):
        raise ValidationError("current verifier dependency lock differs from the bundle")
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
    wheel_files = dependencies.get("wheel_files", {})
    if not wheel_files:
        raise ValidationError("dependency manifest has no wheel hashes")
    for relative, expected in wheel_files.items():
        wheel = output_root / dependencies["wheelhouse"] / relative
        if (
            not wheel.is_file()
            or wheel.stat().st_size != expected.get("bytes")
            or _sha256(wheel) != expected.get("sha256")
        ):
            raise ValidationError(f"dependency wheel checksum mismatch: {relative}")
    nltk_archives = dependencies.get("nltk_resource_archives", {})
    if len(nltk_archives) < 4:
        raise ValidationError("dependency manifest lacks the four required NLTK resource archives")
    for relative, expected in nltk_archives.items():
        archive = output_root / dependencies["nltk_data"] / relative
        if (
            not archive.is_file()
            or archive.stat().st_size != expected.get("bytes")
            or _sha256(archive) != expected.get("sha256")
        ):
            raise ValidationError(f"NLTK resource checksum mismatch: {relative}")

    file_records = manifest.get("files", {})
    expected_files = set(OUTPUT_FILES.values()) | {"eval_config.yaml"}
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
        if record.get("physical_rows_per_prompt") != 1 or record.get("rollout_preexpanded") is not False:
            raise ValidationError(f"dataset {name} violates one-copy policy")
        if record.get("output_exact_prompt_duplicates") != 0:
            raise ValidationError(f"dataset {name} records output duplicates")
    expected_contracts = {
        "rlvr_ifeval_train": ("ifeval_old", OLD_IFEVAL_SCHEMA, "train"),
        "if_multi_fallback_train": ("ifevalg", IF_MULTI_SCHEMA, "preregistered_train_fallback"),
        "google_ifeval": ("ifevalg", GOOGLE_IFEVAL_SCHEMA, "heldout_eval"),
        "ifbench_test": ("ifbench", IFBENCH_SCHEMA, "heldout_eval"),
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
        "math500": _validate_prepared_math(
            _read_jsonl(output_root / OUTPUT_FILES["math500"]), datasets["math500"]
        ),
    }
    overlaps = _check_disjoint(prompt_sets)
    if overlaps != manifest.get("exact_prompt_overlap_counts"):
        raise ValidationError("recomputed prompt overlap matrix differs from manifest")
    print(f"VERIFY_RESULT=PASS output_root={output_root}", flush=True)
    print(f"MANIFEST_SHA256={_sha256(manifest_path)}", flush=True)
    print(json.dumps({"datasets": datasets, "exact_prompt_overlap_counts": overlaps}, indent=2), flush=True)
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
