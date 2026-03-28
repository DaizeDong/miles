#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def list_value(data, key, idx, default=None):
    values = data.get(key)
    if not isinstance(values, list) or idx >= len(values):
        return default
    return values[idx]


def extract_latest(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))

    latest = data.get("latest")
    if isinstance(latest, dict):
        return {
            "benchmark": data.get("benchmark") or path.stem,
            "score": latest.get("score"),
            "metric_key": latest.get("metric_key"),
            "step_label": latest.get("step_label"),
            "result_json": str(path),
            "model_path": latest.get("model_path"),
            "eval_log": latest.get("eval_log"),
        }

    benchmark = data.get("benchmark") or path.stem
    steps = data.get("steps")
    scores = data.get(benchmark)
    if not isinstance(steps, list) or not isinstance(scores, list) or not steps or not scores:
        return {
            "benchmark": benchmark,
            "score": None,
            "metric_key": None,
            "step_label": None,
            "result_json": str(path),
            "model_path": None,
            "eval_log": None,
        }

    idx = min(len(steps), len(scores)) - 1
    step_label = list_value(data, "step_labels", idx, steps[idx])
    return {
        "benchmark": benchmark,
        "score": scores[idx],
        "metric_key": list_value(data, "metric_key", idx),
        "step_label": step_label,
        "result_json": str(path),
        "model_path": list_value(data, "model_path", idx),
        "eval_log": list_value(data, "eval_log", idx),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: summarize_results.py <results_root> <run_name>", file=sys.stderr)
        return 2

    results_root = Path(sys.argv[1]).resolve()
    run_name = sys.argv[2]
    run_dir = results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    benchmark_files = sorted(
        path for path in run_dir.glob("*.json") if path.name not in {"summary.json"}
    )

    rows = []
    for path in benchmark_files:
        rows.append(extract_latest(path))

    summary = {
        "run_name": run_name,
        "results_root": str(results_root),
        "benchmarks": rows,
    }

    summary_json = run_dir / "summary.json"
    summary_md = run_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        f"# {run_name}",
        "",
        "| benchmark | score | metric_key | step_label | result_json |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        score = "null" if row["score"] is None else str(row["score"])
        lines.append(
            f"| {row['benchmark']} | {score} | {row['metric_key'] or ''} | {row['step_label'] or ''} | {row['result_json']} |"
        )

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(summary_json))
    print(str(summary_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
