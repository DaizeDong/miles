#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

DATA_ROOT=${DATA_ROOT:?DATA_ROOT not set}
RESULTS_ROOT=${RESULTS_ROOT:?RESULTS_ROOT not set}
RESULTS_ROOT_HOST=${RESULTS_ROOT_HOST:-${RESULTS_ROOT}}
RUN_NAME=${RUN_NAME:?RUN_NAME not set}
BENCHMARK_NAME=${BENCHMARK_NAME:?BENCHMARK_NAME not set}
STEP_LABEL=${STEP_LABEL:?STEP_LABEL not set}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH not set}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-1.0}
EVAL_TOP_P=${EVAL_TOP_P:-0.7}
EVAL_PROMPT_LENGTH=${EVAL_PROMPT_LENGTH:-2048}
EVAL_RESPONSE_LENGTH=${EVAL_RESPONSE_LENGTH:-16384}
EVAL_TP_SIZE=${EVAL_TP_SIZE:-8}
EVAL_GPU_MEMORY_UTILIZATION=${EVAL_GPU_MEMORY_UTILIZATION:-0.9}
EVAL_ROLLOUT_N=${EVAL_ROLLOUT_N:-4}
EVAL_RAY_MEMORY_MONITOR_REFRESH_MS=${EVAL_RAY_MEMORY_MONITOR_REFRESH_MS:-0}
EVAL_RAY_MEMORY_USAGE_THRESHOLD=${EVAL_RAY_MEMORY_USAGE_THRESHOLD:-0.99}
AUTO_CKPT_EVAL_EXPORT_ROOT=${AUTO_CKPT_EVAL_EXPORT_ROOT:-}
AUTO_CKPT_EVAL_EXPORT_SERIES_NAME=${AUTO_CKPT_EVAL_EXPORT_SERIES_NAME:-}

RUN_DIR="${RESULTS_ROOT}/${RUN_NAME}/${BENCHMARK_NAME}"
RESULT_JSON="${RESULTS_ROOT}/${RUN_NAME}/${BENCHMARK_NAME}.json"

mkdir -p "${RUN_DIR}"

gen_out="${RUN_DIR}/step_${STEP_LABEL}.parquet"
eval_log="${RUN_DIR}/step_${STEP_LABEL}.eval.log"
GENERATION_MODEL_PATH="${MODEL_PATH}"

echo "[INFO] BENCHMARK_NAME=${BENCHMARK_NAME}"
echo "[INFO] STEP_LABEL=${STEP_LABEL}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[INFO] RESULTS_ROOT_HOST=${RESULTS_ROOT_HOST}"
echo "[INFO] EVAL_PROMPT_LENGTH=${EVAL_PROMPT_LENGTH}"
echo "[INFO] EVAL_RESPONSE_LENGTH=${EVAL_RESPONSE_LENGTH}"
echo "[INFO] EVAL_TP_SIZE=${EVAL_TP_SIZE}"
echo "[INFO] EVAL_GPU_MEMORY_UTILIZATION=${EVAL_GPU_MEMORY_UTILIZATION}"
echo "[INFO] EVAL_ROLLOUT_N=${EVAL_ROLLOUT_N}"
echo "[INFO] EVAL_RAY_MEMORY_MONITOR_REFRESH_MS=${EVAL_RAY_MEMORY_MONITOR_REFRESH_MS}"
echo "[INFO] EVAL_RAY_MEMORY_USAGE_THRESHOLD=${EVAL_RAY_MEMORY_USAGE_THRESHOLD}"
echo "[INFO] AUTO_CKPT_EVAL_EXPORT_ROOT=${AUTO_CKPT_EVAL_EXPORT_ROOT:-disabled}"
echo "[INFO] AUTO_CKPT_EVAL_EXPORT_SERIES_NAME=${AUTO_CKPT_EVAL_EXPORT_SERIES_NAME:-disabled}"

GENERATION_MODEL_PATH="$(
  python3 - "${MODEL_PATH}" "${RUN_DIR}" "${STEP_LABEL}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

try:
    from safetensors import safe_open
except Exception:
    print(sys.argv[1])
    raise SystemExit(0)

model_path = Path(sys.argv[1]).resolve()
run_dir = Path(sys.argv[2]).resolve()
step_label = sys.argv[3]
config_path = model_path / "config.json"
index_path = model_path / "model.safetensors.index.json"

if not config_path.is_file() or not index_path.is_file():
    print(str(model_path))
    raise SystemExit(0)

config = json.loads(config_path.read_text(encoding="utf-8"))
config_vocab_size = config.get("vocab_size")
index_data = json.loads(index_path.read_text(encoding="utf-8"))
weight_map = index_data.get("weight_map", {})

weight_file = None
weight_key = None
for candidate in ("model.embed_tokens.weight", "lm_head.weight"):
    mapped = weight_map.get(candidate)
    if mapped:
        weight_file = model_path / mapped
        weight_key = candidate
        break

if weight_file is None or not weight_file.is_file():
    print(str(model_path))
    raise SystemExit(0)

with safe_open(str(weight_file), framework="np") as f:
    actual_vocab_size = int(f.get_slice(weight_key).get_shape()[0])

if config_vocab_size == actual_vocab_size:
    print(str(model_path))
    raise SystemExit(0)

patched_root = run_dir / f"model_for_eval_{step_label}"
if patched_root.exists():
    shutil.rmtree(patched_root)
patched_root.mkdir(parents=True, exist_ok=True)

for src in model_path.iterdir():
    dst = patched_root / src.name
    if src.name == "config.json":
        continue
    if src.is_dir():
        os_method = getattr(src, "symlink_to", None)
        dst.symlink_to(src, target_is_directory=True)
    else:
        dst.symlink_to(src)

config["vocab_size"] = actual_vocab_size
(patched_root / "config.json").write_text(
    json.dumps(config, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print(str(patched_root))
PY
)"

echo "[INFO] GENERATION_MODEL_PATH=${GENERATION_MODEL_PATH}"

export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=INFO
export PYTHONPATH="/root/verl:/root/out-Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"

cd /root/verl

echo "[INFO] Stopping any existing Ray clusters..."
ray stop --force 2>/dev/null || true
sleep 2

export HEAD_NODE="$(hostname -s)"
export RAY_PORT="${RAY_PORT:-6379}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
export RAY_ADDRESS="${HEAD_NODE}:${RAY_PORT}"
export TARGET_GPUS=8
export RAY_memory_monitor_refresh_ms="${EVAL_RAY_MEMORY_MONITOR_REFRESH_MS}"
export RAY_memory_usage_threshold="${EVAL_RAY_MEMORY_USAGE_THRESHOLD}"

echo "[INFO] RAY_PORT=${RAY_PORT}"
echo "[INFO] DASHBOARD_PORT=${DASHBOARD_PORT}"
echo "[INFO] expecting ${TARGET_GPUS} GPUs on single node"

ray start --head \
  --node-ip-address "${HEAD_NODE}" \
  --port "${RAY_PORT}" \
  --num-gpus 8 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port "${DASHBOARD_PORT}"

sleep 3

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"VERL_LOGGING_LEVEL\": \"${VERL_LOGGING_LEVEL}\"
  }
}"

if [ "${FORCE_REGENERATE:-0}" = "1" ] || [ ! -s "${gen_out}" ]; then
  echo "[HEAD] submitting ray job for benchmark ${BENCHMARK_NAME} ..."
  ray job submit --address="http://${HEAD_NODE}:${DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- \
    python3 -m verl.trainer.main_generation_server \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    actor_rollout_ref.model.path="${GENERATION_MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.temperature="${EVAL_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${EVAL_TOP_P}" \
    actor_rollout_ref.rollout.prompt_length="${EVAL_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${EVAL_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${EVAL_TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${EVAL_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n="${EVAL_ROLLOUT_N}" \
    actor_rollout_ref.rollout.skip_tokenizer_init=false \
    data.train_files="${DATA_ROOT}/test.parquet" \
    data.prompt_key=prompt \
    +data.output_path="${gen_out}"

  sleep 10
else
  echo "[INFO] Reusing existing generation output: ${gen_out}"
fi

python3 -m verl.trainer.main_eval \
  data.path="${gen_out}" \
  data.prompt_key=prompt \
  data.response_key=responses \
  custom_reward_function.path=/root/verl/recipe/r1/reward_score.py \
  custom_reward_function.name=reward_func \
  2>&1 | tee "${eval_log}"

metric_json=$(
  python3 - "${eval_log}" "${BENCHMARK_NAME}" <<'PY'
import ast
import json
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
benchmark = sys.argv[2]
text = log_path.read_text(errors="ignore")

def to_jsonable(x):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    try:
        return float(x)
    except Exception:
        return str(x)

candidates = []
for line in text.splitlines():
    if "test_score" not in line:
        continue
    lb = line.find("{")
    rb = line.rfind("}")
    if lb == -1 or rb == -1 or rb <= lb:
        continue
    candidates.append(line[lb:rb + 1])

if not candidates:
    candidates = re.findall(r"\{[^{}\n]*test_score[^{}\n]*\}", text)

data = None
for s in reversed(candidates):
    try:
        obj = ast.literal_eval(s)
    except Exception:
        continue
    if isinstance(obj, dict):
        data = obj
        break

if not isinstance(data, dict):
    print(json.dumps({"benchmark": benchmark, "score": None, "metric_key": None, "all_test_scores": {}}, ensure_ascii=False))
    raise SystemExit(0)

test_scores = {}
for k, v in data.items():
    if not isinstance(k, str):
        continue
    if k.startswith("test_score/") or ("test_score" in k):
        test_scores[k] = to_jsonable(v)

if not test_scores:
    print(json.dumps({"benchmark": benchmark, "score": None, "metric_key": None, "all_test_scores": {}}, ensure_ascii=False))
    raise SystemExit(0)

keys = sorted(test_scores.keys())
primary_key = keys[0]
for k in keys:
    lowered = k.lower()
    if benchmark.lower() in lowered:
        primary_key = k
        break
    if "test_score/" + benchmark.lower() == lowered:
        primary_key = k
        break

print(
    json.dumps(
        {
            "benchmark": benchmark,
            "score": test_scores.get(primary_key),
            "metric_key": primary_key,
            "all_test_scores": test_scores,
        },
        ensure_ascii=False,
    )
)
PY
)

update_args=(
  "${RESULT_JSON}"
  "${BENCHMARK_NAME}"
  "${STEP_LABEL}"
  "${MODEL_PATH}"
  "${gen_out}"
  "${eval_log}"
  "${metric_json}"
)
if [ -n "${AUTO_CKPT_EVAL_EXPORT_ROOT}" ] && [ -n "${AUTO_CKPT_EVAL_EXPORT_SERIES_NAME}" ]; then
  update_args+=("${AUTO_CKPT_EVAL_EXPORT_ROOT}" "${AUTO_CKPT_EVAL_EXPORT_SERIES_NAME}")
fi
python3 "${SCRIPT_DIR}/update_eval_results.py" "${update_args[@]}"

echo "[INFO] Stopping Ray cluster..."
ray stop --force 2>/dev/null || true
