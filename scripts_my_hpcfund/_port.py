#!/usr/bin/env python3
"""
Mirror scripts_mine/ -> scripts_my_hpcfund/, mechanically rewriting for the
AMD HPC Fund cluster (account=hwang, partition=mi3008x, apptainer SIF,
/work1 scratch). See logs/_recon/PORTING_INVENTORY.md for the rationale.

Reproducible: re-run any time scripts_mine/ changes. Idempotent.
"""
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

SRC = Path("/home1/dzdong/workspace/miles/scripts_mine")
DST = Path("/home1/dzdong/workspace/miles/scripts_my_hpcfund")

# Skip qwen3.5 — user said drop it.
SKIP_PATTERNS = ("qwen3.5-35B",)

# Longest match first. Each entry is (search, replace).
# Order matters: more specific paths must come before their prefixes.
PATH_SUBS = [
    # Specific data files (jsonl-not-parquet on this cluster)
    ("/mnt/weka/home/hongyi.wang/workspace/rlhf/verl/data/DAPO-Math-17k/train.parquet",
     "/work1/hwang/dzdong/data/DAPO-Math-17k/dapo-math-17k.jsonl"),
    ("/mnt/weka/home/hongyi.wang/workspace/rlhf/verl/data/aime-2024-full/train.parquet",
     "/work1/hwang/dzdong/data/aime-2024/aime-2024.jsonl"),
    # AUTO_CKPT_EVAL_EXPORT_ROOT was wrongly nested under verl/scripts_mine/val
    # on the source cluster. The actual val/ tree lives in miles/. Point it
    # at the ported tree on host (consumed by a host-side monitor, not in
    # the container).
    ("/mnt/weka/home/hongyi.wang/workspace/rlhf/verl/scripts_mine/val",
     "/home1/dzdong/workspace/miles/scripts_my_hpcfund/val"),
    # Repos
    ("/mnt/weka/home/hongyi.wang/workspace/rlhf/miles",
     "/home1/dzdong/workspace/miles"),
    ("/mnt/weka/home/hongyi.wang/workspace/rlhf/Megatron-LM-verl",
     "/work1/hwang/dzdong/Megatron-LM-verl"),
    ("/mnt/weka/home/hongyi.wang/workspace/rlhf/verl",
     "/work1/hwang/dzdong/verl"),
    ("/mnt/weka/home/hongyi.wang/workspace/transformers-main",
     "/work1/hwang/dzdong/transformers-main"),
    ("/mnt/weka/home/hongyi.wang/workspace/xllm-sglang",
     "/work1/hwang/dzdong/xllm-sglang"),
    # Container images (sqsh -> sif, both flavors map to the same SIF here)
    ("/mnt/weka/shrd/k2m/hongyi.wang/containers/slimerl+slime+latest.sqsh",
     "/work1/hwang/dzdong/images/miles-mi300.sif"),
    ("/mnt/weka/shrd/k2m/hongyi.wang/containers/verlai+verl+sgl055.latest.sqsh",
     "/work1/hwang/dzdong/images/miles-mi300.sif"),
    # Shared checkpoint paths from haolong.jia
    ("/mnt/weka/shrd/k2m/haolong.jia/checkpoint_torch_dist",
     "/work1/hwang/dzdong/checkpoints_torch_dist"),
    ("/mnt/weka/shrd/k2m/haolong.jia/checkpoint",
     "/work1/hwang/dzdong/checkpoints_hf"),
    ("/mnt/weka/shrd/k2m/haolong.jia",
     "/work1/hwang/dzdong"),
    # Per-user dirs under hongyi.wang
    ("/mnt/weka/shrd/k2m/hongyi.wang/datasets_miles",
     "/work1/hwang/dzdong/datasets_miles"),
    ("/mnt/weka/shrd/k2m/hongyi.wang/checkpoint_miles",
     "/work1/hwang/dzdong/checkpoint_miles"),
    ("/mnt/weka/shrd/k2m/hongyi.wang",
     "/work1/hwang/dzdong"),
    # Generic fallback (anything left under /mnt/weka/shrd/k2m or /mnt/weka/home)
    ("/mnt/weka/shrd/k2m", "/work1/hwang/dzdong"),
    ("/mnt/weka/home/hongyi.wang", "/work1/hwang/dzdong"),
    # Post-pass: keep big run output off the 24 GB $HOME. Repo lives on $HOME
    # but logs and tmp must go to $WORK.
    ("/home1/dzdong/workspace/miles/logs_tmp",
     "/work1/hwang/dzdong/miles_logs_tmp"),
    ("/home1/dzdong/workspace/miles/logs",
     "/work1/hwang/dzdong/miles_logs"),
    # Wrappers that hardcoded a path into scripts_mine/ should reference the
    # ported tree instead.
    ("scripts_mine/", "scripts_my_hpcfund/"),
    # Flatten SLURM --output / --error paths: this cluster's SLURM cannot
    # auto-create the `%x-%J/` subdirectory and the job FAILs with signal 53
    # ("cannot open output file") before our script runs.
    ("/sbatch/%x-%J/%N.%J.%t.log", "/sbatch/%x.%J.%N.%t.log"),
    ("/sbatch/%x-%J/%N.%J.%t.err", "/sbatch/%x.%J.%N.%t.err"),
    ("/sbatch/%x-%J.%t.err", "/sbatch/%x.%J.%t.err"),
]

# SLURM directive line-level substitutions.
# Each entry is (regex, replacement_or_None). None = delete the matched line.
HEADER_REGEX_SUBS = [
    (re.compile(r"^#SBATCH\s+--account=k2m\s*$"), "#SBATCH --account=hwang"),
    (re.compile(r"^#SBATCH\s+--qos=lowprio\s*$"), None),
    (re.compile(r"^#SBATCH\s+--partition=lowprio\s*$"),
     "#SBATCH --partition=mi3008x"),
    (re.compile(r"^#SBATCH\s+--gres=gpu:\S+\s*$"), None),
    # Drop the obsolete moe reservation marker — it was already commented.
    (re.compile(r"^##SBATCH\s+--reservation=moe\s*$"), None),
]

# Container-invocation transformation:
#  srun \
#    ... \
#    --container-image="${CONTAINER_IMAGE}" \
#    --container-mounts="${CONTAINER_MOUNTS}" \
#    --export=ALL \
#    bash -c '...'
# becomes:
#  srun \
#    ... \
#    --export=ALL \
#    apptainer exec --rocm --writable-tmpfs \
#      --bind /work1/hwang/dzdong:/work1/hwang/dzdong \
#      --bind /home1/dzdong/workspace/miles:/root/miles \
#      "${CONTAINER_IMAGE}" \
#      bash -c '...'
APPTAINER_INSERT = (
    '  apptainer exec --rocm --writable-tmpfs \\\n'
    '    --bind /work1/hwang/dzdong:/work1/hwang/dzdong \\\n'
    '    --bind /home1/dzdong/workspace/miles:/root/miles \\\n'
    '    "${CONTAINER_IMAGE}" \\'
)


def rewrite_text(path: Path, text: str) -> str:
    # 1) Verbatim path substitutions — longest first.
    for src, dst in PATH_SUBS:
        text = text.replace(src, dst)

    # 2) SLURM header regex substitutions (line-by-line).
    out_lines = []
    for line in text.splitlines(keepends=True):
        stripped_nl = line.rstrip("\n")
        replaced = False
        for rx, repl in HEADER_REGEX_SUBS:
            if rx.match(stripped_nl):
                if repl is None:
                    pass  # drop line
                else:
                    out_lines.append(repl + "\n")
                replaced = True
                break
        if not replaced:
            out_lines.append(line)
    text = "".join(out_lines)

    # 3) Drop the pyxis --container-image and --container-mounts srun args,
    #    and inject an apptainer exec wrapper before `bash -c '...'`.
    is_sh = path.suffix == ".sh"
    if is_sh and "--container-image=" in text:
        # Remove both pyxis flags (entire line including trailing backslash).
        text = re.sub(
            r"^\s*--container-image=.*\\\s*\n", "", text, flags=re.M)
        text = re.sub(
            r"^\s*--container-mounts=.*\\\s*\n", "", text, flags=re.M)
        # Insert apptainer exec line right before `bash -c '...'` (must be
        # indented continuation of the srun \ block).
        # Use a function-form replacement so the literal newline survives —
        # `r"\n"` in a regex replacement is backslash+n, not LF.
        def insert_apptainer(m: re.Match) -> str:
            return m.group(1) + APPTAINER_INSERT + "\n" + m.group(2)
        text = re.sub(
            r"(^\s*--export=ALL\s*\\\s*\n)(\s*bash -c)",
            insert_apptainer, text, count=0, flags=re.M)

    # 4) Megatron lives at /app/Megatron-LM in the rlsys/miles SIF, not
    #    /root/Megatron-LM as in the original sqsh. Rewrite all path refs
    #    (RUNTIME_PYTHONPATH default, dir-existence guard, embedded JSON).
    text = text.replace('/root/Megatron-LM', '/app/Megatron-LM')

    # 4b) `pip install -e .` produces a cp313-tagged wheel (host Py3.13) that
    # doesn't register miles/miles_plugins inside the cp310 container venv.
    # Belt-and-suspenders: keep /root/miles on PYTHONPATH so source imports
    # work regardless. Also prepend /work1/hwang/dzdong/python_extras for the
    # site-packages we manually installed there (see 4c). Patch both the
    # RUNTIME_PYTHONPATH default and any embedded ray runtime_env PYTHONPATH
    # literal.
    text = text.replace(
        'RUNTIME_PYTHONPATH:-/app/Megatron-LM/',
        'RUNTIME_PYTHONPATH:-/work1/hwang/dzdong/python_extras:/root/miles:/app/Megatron-LM/')
    text = text.replace(
        '\\"PYTHONPATH\\": \\"/app/Megatron-LM/\\"',
        '\\"PYTHONPATH\\": \\"/work1/hwang/dzdong/python_extras:/root/miles:/app/Megatron-LM/\\"')

    # 4c) The rlsys/miles:MI300-latest SIF ships a broken wandb (wandb.errors
    # submodule missing → ModuleNotFoundError at import time in
    # miles/utils/tracking_utils.py). The apptainer overlay (--writable-tmpfs)
    # is too small to repair via in-job `pip install --force-reinstall wandb`
    # — that fails with ENOSPC. Instead, wandb 0.19.11 has been pre-installed
    # one-time to /work1/hwang/dzdong/python_extras/ and added to PYTHONPATH
    # above. So no in-job pip install is needed.
    #
    # 4d) The runtime_env PYTHONPATH in (4b) only reaches ray-spawned workers.
    # The driver `python3 train.py` is invoked from bash with bash env, so we
    # also need PYTHONPATH at bash level right before it. The bash-level
    # exports are injected by step 5a (the ray-job-submit bypass) — see
    # below — so the PYTHONPATH export is added in that same block.

    # 5) AMD HPC Fund submission filter REQUIRES `#SBATCH --time=` and caps
    #    user runtime at 12h (partition MaxTime=4d is shadowed by QoS limit).
    #    The source cluster (k2m) had a sensible default and the scripts
    #    omitted it. Inject a 12h max before the --output line if absent.
    if is_sh and "#SBATCH" in text and "#SBATCH --time=" not in text:
        text = re.sub(
            r"(^#SBATCH --output=)",
            r"#SBATCH --time=12:00:00\n\1",
            text, count=1, flags=re.M)

    # 5a) Bypass `ray job submit`. On this cluster the dashboard agent
    # (port 8265) takes the job submission request, queues it on an internal
    # async coroutine, and never responds — `ray job submit` returns
    # 504 after 300s. Runtime env is only env_vars (no pip deps), which we
    # can just `export` at bash level so the ray-spawned workers inherit
    # them. Replace `ray job submit --runtime-env-json=... -- python3 ...`
    # with a direct `python3 ...` invocation; train.py's own `ray.init()`
    # attaches to the running head via RAY_ADDRESS.
    if is_sh and "ray job submit" in text:
        text = re.sub(
            r'  ray job submit --address="http://\$\{HEAD_IP\}:\$\{DASHBOARD_PORT\}" \\\n'
            r'    --runtime-env-json="\$\{RUNTIME_ENV_JSON\}" \\\n'
            r'    -- \\\n'
            r'    python3 "\$\{TRAIN_ENTRYPOINT\}" \\',
            r'  # Bypass `ray job submit` (dashboard job-submission agent hangs on\n'
            r'  # this cluster — 504 after 300s). The Ray worker processes are\n'
            r'  # spawned by raylet (which already inherited our exported env),\n'
            r'  # so we only need to ensure the runtime-env keys are exported.\n'
            r'  export PYTHONPATH="/work1/hwang/dzdong/python_extras:/root/miles:/app/Megatron-LM/${PYTHONPATH:+:${PYTHONPATH}}"\n'
            r'  export CUDA_DEVICE_MAX_CONNECTIONS=1\n'
            r'  export NCCL_NVLS_ENABLE="${HAS_NVLINK}"\n'
            r'  # 309779 hung 9.5h after `Connected to Ray cluster` with no\n'
            r'  # output. python3 default buffering hides exactly the kind of\n'
            r'  # log we need (placement-group, sglang boot, actor init). Run\n'
            r'  # unbuffered; also widen ray verbosity so worker-side errors\n'
            r'  # (e.g. raylet registration failures) reach our redirected log.\n'
            r'  export PYTHONUNBUFFERED=1\n'
            r'  export RAY_DEDUP_LOGS=0\n'
            r'  export RAY_BACKEND_LOG_LEVEL=info\n'
            r'  python3 -u "${TRAIN_ENTRYPOINT}" \\',
            text, count=1)

    # 5b) The launch script hardcodes `source /root/miles/scripts/models/
    # qwen3-30B-A3B.sh`. Parameterize via MODEL_CONFIG_SCRIPT env so any
    # MoE config can be plugged in (e.g. moonlight.sh) without per-model
    # script clones. Default stays Qwen3-30B-A3B for backward compat.
    #
    # Also append a MOE_GROUPED_GEMM filter: the model config hardcodes
    # `--moe-grouped-gemm`, which routes MoE experts through Transformer
    # Engine's `general_grouped_gemm`. On this ROCm/MI300X that kernel
    # deadlocks deterministically — faulthandler (job 312259) caught rank 0
    # permanently stuck in TE general_grouped_gemm while ranks 1-7 waited in
    # the MoE combine. `--moe-grouped-gemm` is argparse store_true so it
    # cannot be un-set by appending; instead, when MOE_GROUPED_GEMM=0, strip
    # the flag from MODEL_ARGS right after the model config is sourced →
    # Megatron falls back to SequentialMLP (per-expert GEMM loop), avoiding
    # the TE grouped-GEMM kernel. Numerically equivalent; PR² routing
    # unaffected (this only changes the expert-compute implementation).
    text = re.sub(
        r'SCRIPT_DIR="\$\(cd -- "\$\(dirname -- "/root/miles/scripts/run-qwen3-30B-A3B\.sh"\)" &>/dev/null && pwd\)"\nsource "\$\{SCRIPT_DIR\}/models/qwen3-30B-A3B\.sh"',
        'MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-/root/miles/scripts/models/qwen3-30B-A3B.sh}"\n'
        'source "${MODEL_CONFIG_SCRIPT}"\n'
        'if [ "${MOE_GROUPED_GEMM:-1}" = "0" ]; then\n'
        '  _MA_FILTERED=()\n'
        '  for _ma in "${MODEL_ARGS[@]}"; do\n'
        '    [ "${_ma}" = "--moe-grouped-gemm" ] || _MA_FILTERED+=("${_ma}")\n'
        '  done\n'
        '  MODEL_ARGS=("${_MA_FILTERED[@]}")\n'
        '  echo "[INFO] MOE_GROUPED_GEMM=0 -> stripped --moe-grouped-gemm (SequentialMLP)"\n'
        'fi',
        text)

    # 5c) On this cluster, the wait-for-workers Python heredoc fires
    # `ray.init(address=...)` shortly after `ray start --head` and either
    # crashes with "Failed to register worker to Raylet: End of file" or
    # hangs indefinitely (309312 / 309514 / 309766). With `set -e` the
    # whole script then dies before the loop can retry. For single-node
    # runs (ACTOR_NUM_NODES == SLURM_NNODES) the wait is unnecessary —
    # ray head already has all GPUs registered. Insert a fast-path that
    # bypasses the loop entirely for single-node. The original loop body
    # is kept for genuine multi-node runs (where workers truly need to
    # register from sibling tasks).
    if is_sh and "--dashboard-port" in text:
        # `ray start --head` returns once the GCS is up, but the dashboard
        # job-submission agent (port 8265) takes longer to be ready to accept
        # POSTs. A plain `sleep 15` isn't enough — `ray job submit` then
        # hangs 300s and returns 504 (309767). Poll /api/version until 200.
        ray_dashboard_wait = (
            '  echo "[HEAD] waiting for ray dashboard at '
            '${HEAD_IP}:${DASHBOARD_PORT}..."\n'
            '  for _i in $(seq 1 120); do\n'
            '    _code=$(curl -s -m 2 -o /dev/null -w "%{http_code}" '
            '"http://${HEAD_IP}:${DASHBOARD_PORT}/api/version" '
            '2>/dev/null || true)\n'
            '    if [ "${_code}" = "200" ]; then\n'
            '      echo "[HEAD] dashboard ready (after ${_i}*2s)"\n'
            '      break\n'
            '    fi\n'
            '    sleep 2\n'
            '  done\n'
        )
        text = re.sub(
            r"(    --dashboard-port \"\$\{DASHBOARD_PORT\}\"\n)(\n  export RAY_ADDRESS=)",
            r"\1\n" + ray_dashboard_wait + r"\2",
            text, count=0)
        text = re.sub(
            r"(  WAIT_BEGIN=\$\(date \+%s\)\n)(  while true; do)",
            r'\1  if [ "${SLURM_NNODES}" -eq 1 ]; then\n'
            r'    echo "[HEAD] single-node run; ray head already has '
            r'${TARGET_GPUS} GPUs, skipping wait-for-workers loop"\n'
            r'    AVAILABLE_GPUS="${TARGET_GPUS}"\n'
            r'  else\n'
            r'\2',
            text, count=1)
        text = re.sub(
            r"(  done\n)(\n  RUNTIME_ENV_JSON=)",
            r'\1  fi\n\2',
            text, count=1)
        text = text.replace(
            'AVAILABLE_GPUS=$(python - <<PY',
            'AVAILABLE_GPUS=$(timeout 30 python - 2>/dev/null <<PY')

    # 5d) Ray worker-prestart crash-loop (root cause of the 12h silent hangs
    #     in 309779 / 309999). The raylet prestarts `num_cpus` Python workers
    #     (= 192 on these nodes). 192 cold `import ray` from the SIF squashfs
    #     overrun the 60s worker-registration timeout, so the raylet declares
    #     the whole pool dead and crash-loops it forever — never servicing the
    #     driver's CoreWorker registration (driver hangs at worker.py:2542).
    #     Verified fix (diag 310366): cap ray's logical CPUs at 32 (prestart
    #     192->32; physical cores are unaffected, this is only ray scheduling
    #     accounting) and raise the registration grace to 600s. The raylet
    #     reads RAY_worker_register_timeout_seconds at start, so export it
    #     before every `ray start`.
    #
    # 5e) Ray actors (RolloutManager etc.) are spawned by the raylet and
    #     inherit the RAYLET's environment, not the driver's. wandb 0.21.1
    #     lives in /work1/hwang/dzdong/python_extras (see 4c) and is only on
    #     PYTHONPATH via the bash export placed before `python3 train.py` —
    #     which runs AFTER `ray start`, so the raylet (and every actor) misses
    #     it. The actor's `import wandb` then resolves to a stale/partial
    #     module → `AttributeError: module 'wandb' has no attribute 'login'`.
    #     Fix: also export PYTHONPATH before `ray start` so the raylet — and
    #     thus all actors — inherit /work1/.../python_extras + /root/miles.
    # 5f) Ray 2.47's AMDGPUAcceleratorManager masks GPUs via HIP_VISIBLE_DEVICES
    #     and only skips it when RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES is
    #     set. miles' NOSET_VISIBLE_DEVICES_ENV_VARS_LIST (miles/ray/utils.py)
    #     predates that ray change — it lists the ROCR NOSET var, not the HIP
    #     one — so ray still masks every GPU actor. The SGLangEngine actor is
    #     created with num_gpus=0.2, so ray masks it to a SINGLE GPU; sglang
    #     then addresses GPUs absolutely via base_gpu_id and a TP=4 engine asks
    #     for ordinals 1..3 that don't exist → `HIP error: invalid device
    #     ordinal`. miles' design intent is NOSET-everywhere (it does its own
    #     base_gpu_id placement), so exporting the missing HIP NOSET var before
    #     `ray start` — so the raylet and every actor inherit it — restores
    #     that intent without editing miles source.
    if is_sh and "ray start" in text:
        text = text.replace(
            '  ray start \\',
            '  export PYTHONPATH="/work1/hwang/dzdong/python_extras:/root/miles:'
            '/app/Megatron-LM/${PYTHONPATH:+:${PYTHONPATH}}"\n'
            '  export RAY_worker_register_timeout_seconds=600\n'
            '  export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1\n  ray start \\')
        text = text.replace(
            '    --num-gpus "${NUM_GPUS_PER_NODE}" \\',
            '    --num-gpus "${NUM_GPUS_PER_NODE}" \\\n    --num-cpus 32 \\')

    # 5g) sglang's CUDA-graph capture crashes inside capture_one_batch_size
    #     with `TypeError: cannot unpack non-iterable ForwardMetadata object`
    #     — the triton attention backend in the SIF's sglang
    #     (0.5.6.post3.dev1938) returns a bare ForwardMetadata where the
    #     capture path expects a tuple. Allow disabling CUDA graph via the
    #     SGLANG_DISABLE_CUDA_GRAPH env knob (miles auto-exposes sglang's
    #     --disable-cuda-graph as --sglang-disable-cuda-graph). Off by default
    #     so other models keep CUDA graph; the Moonlight wrapper sets it to 1.
    # 5h) The SIF's sglang triton attention backend is internally inconsistent
    #     for DeepSeek-V2-style MLA models (Moonlight): deepseek_v2.py's
    #     forward_absorb_fused_mla_rope_prepare unpacks
    #     attn_backend.forward_metadata as a 7-tuple, but the triton backend
    #     now stores a ForwardMetadata dataclass → `TypeError: cannot unpack
    #     non-iterable ForwardMetadata object` on the first decode. Allow
    #     overriding the attention backend (e.g. to `aiter`) via the
    #     SGLANG_ATTENTION_BACKEND env knob — miles auto-exposes sglang's
    #     --attention-backend as --sglang-attention-backend.
    if is_sh and "SGLANG_ARGS=(" in text:
        text = text.replace(
            '  --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 "${SGLANG_CUDA_GRAPH_MAX}")\n)',
            '  --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 "${SGLANG_CUDA_GRAPH_MAX}")\n)\n'
            '\n'
            'if [ "${SGLANG_DISABLE_CUDA_GRAPH:-0}" = "1" ]; then\n'
            '  SGLANG_ARGS+=(--sglang-disable-cuda-graph)\n'
            'fi\n'
            '\n'
            'if [ -n "${SGLANG_ATTENTION_BACKEND:-}" ]; then\n'
            '  SGLANG_ARGS+=(--sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}")\n'
            'fi')

    # 5i) Megatron's training-side attention backend. The base script hardcodes
    #     `--attention-backend flash`, but Transformer Engine's flash/fused DPA
    #     on this ROCm has no backend for DeepSeek-V2-style MLA's asymmetric
    #     head dims (Q/K 192 vs V 128) → `ValueError: No dot product attention
    #     backend is available`. Make it an env knob (MEGATRON_ATTENTION_BACKEND)
    #     so the Moonlight wrapper can select `unfused` (the general fallback).
    if is_sh and "--attention-backend flash" in text:
        text = text.replace(
            '  --attention-backend flash',
            '  --attention-backend "${MEGATRON_ATTENTION_BACKEND:-flash}"')

    # 5j) Transformer Engine on this ROCm (gfx942) has NO usable DPA backend
    #     for MLA: NVTE_DEBUG shows FlashAttention/FusedAttention are
    #     unavailable (TE-determined, kernels absent) and the pure-PyTorch
    #     UnfusedDotProductAttention is disabled for `qkv_format = thd`. The
    #     `thd` (packed varlen) layout comes from `--use-dynamic-batch-size`.
    #     Gate it behind USE_DYNAMIC_BATCH_SIZE so the Moonlight wrapper can
    #     turn it off — then miles uses `bshd`, which UnfusedDPA accepts.
    if is_sh and "  --use-dynamic-batch-size\n" in text:
        text = text.replace(
            '  --use-dynamic-batch-size\n'
            '  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"\n'
            ')\n',
            ')\n'
            'if [ "${USE_DYNAMIC_BATCH_SIZE:-1}" = "1" ]; then\n'
            '  PERF_ARGS+=(--use-dynamic-batch-size '
            '--max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}")\n'
            'fi\n')

    # 5k) RCCL's alltoall_single deadlocks deterministically on this MI300X
    #     node: the MoE expert-parallel token dispatch (OpType=ALLTOALL_BASE)
    #     hangs all 8 EP ranks for the full 600s NCCL watchdog window (jobs
    #     311034 + 311039, both at SeqNum=3, every rank posted the collective).
    #     The model config hardcodes `--moe-token-dispatcher-type alltoall`.
    #     Expose an override (MOE_TOKEN_DISPATCHER) appended to PERF_ARGS;
    #     argparse applies it last-wins over MODEL_ARGS. `allgather` avoids the
    #     alltoall path entirely. The dispatcher only moves tokens to experts —
    #     routing decisions are unchanged, so PR² routing-replay is preserved.
    if is_sh and "--moe-enable-deepep\n" in text:
        text = text.replace(
            '    --moe-enable-deepep\n'
            '  )\n'
            'fi\n',
            '    --moe-enable-deepep\n'
            '  )\n'
            'fi\n'
            'if [ -n "${MOE_TOKEN_DISPATCHER:-}" ]; then\n'
            '  PERF_ARGS+=(--moe-token-dispatcher-type '
            '"${MOE_TOKEN_DISPATCHER}")\n'
            'fi\n')

    # 5m) Megatron->HF weight conversion mode. The 2-node launch script never
    #     passes --megatron-to-hf-mode, so it defaults to `raw`
    #     (HfWeightIteratorDirect), which deadlocks update_weights under
    #     SequentialMLP (MOE_GROUPED_GEMM=0) — the per-expert weight broadcast
    #     desyncs across ranks (job 312260). Expose MEGATRON_TO_HF_MODE so the
    #     wrapper can select `bridge` (HfWeightIteratorBridge, mbridge-based).
    #     Inject into CKPT_ARGS only where the launch script doesn't already
    #     pass the flag (1-node launcher already has it).
    if (is_sh and "--megatron-to-hf-mode" not in text
            and '  --save-interval "${SAVE_INTERVAL}"\n)\n' in text):
        text = text.replace(
            '  --save-interval "${SAVE_INTERVAL}"\n)\n',
            '  --save-interval "${SAVE_INTERVAL}"\n'
            '  --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE:-raw}"\n)\n')

    # 5n) Allow bumping NUM_ROLLOUT between resumes (smoke 20 → full 150)
    #     without Megatron's OptimizerParamScheduler asserting on the saved
    #     total_iters mismatch (313122 died: "class input value 76800 and
    #     checkpoint value 10240 do not match"). Pass
    #     --override-opt_param-scheduler so the command-line LR schedule
    #     replaces the checkpoint's. Safe on fresh runs (no checkpoint loaded)
    #     and on resumes with unchanged schedule (overrides with same values).
    if is_sh and "--save-interval" in text and "--override-opt_param-scheduler" not in text:
        text = text.replace(
            '  --save-interval "${SAVE_INTERVAL}"\n',
            '  --save-interval "${SAVE_INTERVAL}"\n'
            '  --override-opt_param-scheduler\n', 1)

    # 5o) Upstream radixark/main removed --prompt-truncation from the argparse
    #     surface; the legacy launch ROLLOUT_ARGS still passes it
    #     (`--prompt-truncation "${PROMPT_TRUNCATION}"`). train.py rejects
    #     unrecognized args, so the run exits with code 2. Strip the line.
    if is_sh and '  --prompt-truncation "${PROMPT_TRUNCATION}"\n' in text:
        text = text.replace('  --prompt-truncation "${PROMPT_TRUNCATION}"\n', '')

    # 6) Commit 47f0319 ("downsize off2 PR² ablation grid from 8 to 2 nodes")
    #    intended to downsize the A0..A7 cells but the diff only renamed the
    #    files. The body still says --nodes=8 / ACTOR_NUM_NODES=4 /
    #    launch/hy-sbatch-8nodes.sh, contradicting the commit description and
    #    the filename. Apply the missed downsizing here for any A?-...-2nodes
    #    wrapper. Idempotent (no-op if already 2-nodes).
    if re.search(r"hy-sbatch-pr2-A[0-9]+-.*-2nodes\.sh$", str(path)):
        text = re.sub(r"^(#SBATCH --nodes=)8\s*$", r"\g<1>2", text, flags=re.M)
        text = text.replace(
            'ACTOR_NUM_NODES:-4', 'ACTOR_NUM_NODES:-1')
        text = text.replace(
            'launch/hy-sbatch-8nodes.sh', 'launch/hy-sbatch-2nodes.sh')

    return text


def port_file(src: Path) -> Path | None:
    rel = src.relative_to(SRC)
    if any(p in str(rel) for p in SKIP_PATTERNS):
        return None
    dst = DST / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix in (".sh", ".py"):
        text = src.read_text()
        new_text = rewrite_text(src, text)
        dst.write_text(new_text)
        # Preserve exec bit.
        if src.stat().st_mode & stat.S_IEXEC:
            dst.chmod(dst.stat().st_mode | stat.S_IEXEC)
    else:
        shutil.copy2(src, dst)
    return dst


def main():
    # Non-destructive: overwrite files that mirror scripts_mine/, leave
    # locally-added files (e.g. cluster-specific sbatches) alone.
    DST.mkdir(parents=True, exist_ok=True)

    ported = []
    for src in sorted(SRC.rglob("*")):
        if src.is_dir():
            continue
        dst = port_file(src)
        if dst is not None:
            ported.append(dst)

    print(f"[INFO] ported {len(ported)} files -> {DST}")

    # syntax check on shell scripts
    bad = []
    for p in ported:
        if p.suffix == ".sh":
            r = subprocess.run(["bash", "-n", str(p)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                bad.append((p, r.stderr.strip()))
    if bad:
        print(f"[FAIL] {len(bad)} script(s) failed bash -n syntax check:")
        for p, err in bad:
            print(f"  {p}:")
            print(f"    {err}")
        sys.exit(1)
    print(f"[OK] all .sh scripts pass bash -n")


if __name__ == "__main__":
    main()
