#!/usr/bin/env python3
import copy
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def usage() -> int:
    print(
        "Usage: update_eval_results.py <current_json> <benchmark> <step_label> "
        "<model_path> <generated_path> <eval_log> <metric_json> [legacy_root] [legacy_series_name]",
        file=sys.stderr,
    )
    return 2


def step_value_from_label(step_label: str):
    if re.fullmatch(r"\d+", step_label):
        return int(step_label)
    match = re.search(r"(\d+)$", step_label)
    if match:
        return int(match.group(1))
    return step_label


def sort_key(step):
    if isinstance(step, (int, float)):
        return (0, float(step))
    if isinstance(step, str) and step.isdigit():
        return (0, float(step))
    return (1, str(step))


def list_value(data, key, idx, default=None):
    values = data.get(key)
    if not isinstance(values, list) or idx >= len(values):
        return copy.deepcopy(default)
    return values[idx]


def rows_from_data(data, benchmark: str):
    rows = []

    entries = data.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            step_label = str(entry.get("step_label") or "")
            rows.append(
                {
                    "step": step_value_from_label(step_label),
                    "step_label": step_label,
                    "score": entry.get("score"),
                    "metric_key": entry.get("metric_key"),
                    "all_test_scores": entry.get("all_test_scores", {}),
                    "model_path": entry.get("model_path"),
                    "generated_path": entry.get("generated_path"),
                    "eval_log": entry.get("eval_log"),
                    "updated_at_utc": entry.get("updated_at_utc"),
                }
            )
        return rows

    steps = data.get("steps")
    scores = data.get(benchmark)
    if isinstance(steps, list) and isinstance(scores, list):
        count = min(len(steps), len(scores))
        for idx in range(count):
            step = steps[idx]
            step_label = list_value(data, "step_labels", idx)
            if not step_label:
                if isinstance(step, int):
                    step_label = f"rollout_{step:04d}"
                else:
                    step_label = str(step)
            rows.append(
                {
                    "step": step,
                    "step_label": step_label,
                    "score": scores[idx],
                    "metric_key": list_value(data, "metric_key", idx),
                    "all_test_scores": list_value(data, "all_test_scores", idx, {}),
                    "model_path": list_value(data, "model_path", idx),
                    "generated_path": list_value(data, "generated_path", idx),
                    "eval_log": list_value(data, "eval_log", idx),
                    "updated_at_utc": list_value(data, "updated_at_utc", idx),
                }
            )
    return rows


def write_current_result(path: Path, benchmark: str, rows):
    data = {
        "benchmark": benchmark,
        "steps": [row["step"] for row in rows],
        "step_labels": [row["step_label"] for row in rows],
        benchmark: [row["score"] for row in rows],
        "metric_key": [row["metric_key"] for row in rows],
        "all_test_scores": [row["all_test_scores"] for row in rows],
        "model_path": [row["model_path"] for row in rows],
        "generated_path": [row["generated_path"] for row in rows],
        "eval_log": [row["eval_log"] for row in rows],
        "updated_at_utc": [row["updated_at_utc"] for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_legacy_result(path: Path, benchmark: str, rows):
    data = {
        "steps": [row["step"] for row in rows],
        benchmark: [row["score"] for row in rows],
        "metric_key": [row["metric_key"] for row in rows],
        "all_test_scores": [row["all_test_scores"] for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def to_host_path(path: str) -> str:
    results_root = os.environ.get("RESULTS_ROOT", "")
    results_root_host = os.environ.get("RESULTS_ROOT_HOST", results_root)
    if results_root and path.startswith(results_root.rstrip("/") + "/"):
        suffix = path[len(results_root.rstrip("/")):]
        return results_root_host.rstrip("/") + suffix
    if results_root and path == results_root:
        return results_root_host
    return path


def main() -> int:
    if len(sys.argv) not in {8, 10}:
        return usage()

    current_json = Path(sys.argv[1]).resolve()
    benchmark = sys.argv[2]
    step_label = sys.argv[3]
    model_path = sys.argv[4]
    generated_path = to_host_path(sys.argv[5])
    eval_log = to_host_path(sys.argv[6])
    metrics = json.loads(sys.argv[7])
    legacy_root = sys.argv[8] if len(sys.argv) == 10 else ""
    legacy_series_name = sys.argv[9] if len(sys.argv) == 10 else ""

    current_data = {}
    if current_json.exists():
        current_data = json.loads(current_json.read_text(encoding="utf-8"))

    rows = rows_from_data(current_data, benchmark)
    row = {
        "step": step_value_from_label(step_label),
        "step_label": step_label,
        "score": metrics.get("score"),
        "metric_key": metrics.get("metric_key"),
        "all_test_scores": metrics.get("all_test_scores", {}),
        "model_path": model_path,
        "generated_path": generated_path,
        "eval_log": eval_log,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    replaced = False
    for idx, existing in enumerate(rows):
        if existing.get("step_label") == step_label:
            rows[idx] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)

    rows.sort(key=lambda item: sort_key(item.get("step")))
    write_current_result(current_json, benchmark, rows)

    if legacy_root and legacy_series_name:
        legacy_path = Path(legacy_root).resolve() / benchmark / "results" / f"{legacy_series_name}.json"
        write_legacy_result(legacy_path, benchmark, rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
