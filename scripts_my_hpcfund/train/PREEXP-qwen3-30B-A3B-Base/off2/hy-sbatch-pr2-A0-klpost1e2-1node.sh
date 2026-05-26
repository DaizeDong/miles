#!/bin/bash
# === Qwen3-30B-A3B PR² A0 (paper-faithful), single-node colocate ===
# The original target model. Switched back from Moonlight-16B-A3B: Moonlight
# uses DeepSeek-V2-style MLA, and this SIF's Transformer Engine on ROCm
# (gfx942) has no DPA backend for MLA. Qwen3-30B-A3B is standard GQA
# (32 heads / 4 KV groups, no MLA), so it avoids the TE-MLA wall.
#
# Algorithm (A0 = paper-faithful PR²):
#   - kl-post bias predictor, lr_mult=1e2
#   - downsample batch=2, NO additional stabilization (paper baseline)
#   - off2 (NUM_STEPS_PER_ROLLOUT=2)
#
# Resources:
#   - 1 node, 8× MI300X, colocate actor + rollout (forced by mi3008x QoS)
#   - ROLLOUT_NUM_GPUS_PER_ENGINE=1 (TP=1, 8 sglang engines); EP=8, TP=1, PP=1
#
# NUM_ROLLOUT default 20 for cold-boot smoke. Ramp via env when stable.
#SBATCH --job-name=qwen3-off2-pr2-A0-klpost1e2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --account=hwang
#SBATCH --partition=mi3008x
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/work1/hwang/dzdong/miles_logs_tmp/sbatch/%x.%J.%N.%t.log
#SBATCH --error=/work1/hwang/dzdong/miles_logs_tmp/sbatch/%x.%J.%N.%t.err
#SBATCH --open-mode=append

set -euo pipefail

# Model & config
export MODEL_NAME="${MODEL_NAME:-Qwen3-30B-A3B-Base}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-/root/miles/scripts/models/qwen3-30B-A3B.sh}"
export RUN_POSTFIX="${RUN_POSTFIX:-off2-pr2-A0-klpost1e2-1node}"

# Resource layout (QoS forces 1-node, colocate)
export RESOURCE_LAYOUT="${RESOURCE_LAYOUT:-colocate}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
# TP=1 sglang engine per GPU (8 engines). Qwen3 has 32 attention heads so the
# aiter FMHA num_head%16 constraint is satisfied at TP=1 (and TP=2).
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"

# Parallelism
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
export EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-8}"
export EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"

# Sequence lengths (modest for cold-boot smoke)
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export EVAL_MAX_RESPONSE_LENGTH="${EVAL_MAX_RESPONSE_LENGTH:-1024}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"

# 30B model: leave more GPU room for the colocated Megatron actor by
# shrinking sglang's static memory pool (default 0.7 is tuned for 16B).
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.5}"

# Rollout / optimizer
export NUM_ROLLOUT="${NUM_ROLLOUT:-10}"           # smoke on amd_pr2_on_main
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
# GLOBAL_BATCH_SIZE = 32 * 16 / 2 = 256
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export EVAL_TOP_P="${EVAL_TOP_P:-0.7}"

# PR² mechanism (paper-faithful A0 — matches the Qwen3 paper A0 2-node wrapper)
export USE_MILES_ROUTER="${USE_MILES_ROUTER:-1}"
export USE_ROUTING_REPLAY="${USE_ROUTING_REPLAY:-1}"
export USE_ROLLOUT_ROUTING_REPLAY="${USE_ROLLOUT_ROUTING_REPLAY:-0}"
export ENABLE_PREDICTIVE_ROUTING_REPLAY="${ENABLE_PREDICTIVE_ROUTING_REPLAY:-1}"
export BIAS_PREDICTOR_LOSS_TYPE="${BIAS_PREDICTOR_LOSS_TYPE:-kl-post}"
export BIAS_PREDICTOR_LR_MULT="${BIAS_PREDICTOR_LR_MULT:-1e2}"
export PREDICTIVE_DOWNSAMPLE_BATCH_SIZE="${PREDICTIVE_DOWNSAMPLE_BATCH_SIZE:-2}"
export PREDICTIVE_DOWNSAMPLE_MAX_LEN_LIMIT="${PREDICTIVE_DOWNSAMPLE_MAX_LEN_LIMIT:-8192}"
export PREDICTIVE_MAX_TOTAL_TOKENS="${PREDICTIVE_MAX_TOTAL_TOKENS:-4096}"
export PREDICTIVE_STORAGE_DTYPE="${PREDICTIVE_STORAGE_DTYPE:-fp32}"
# All enhancement vars left UNSET (paper A0 baseline; defaults disable them).

# --- AMD HPC Fund portability fixes (model-agnostic, carried from Moonlight) ---
# PyTorch expandable_segments allocator hands hipMemcpy virtual-mapped pointers.
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:False}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
# sglang CUDA-graph capture crashes (ForwardMetadata unpack) on the SIF sglang.
export SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-1}"
# aiter CK 2-stage MoE gemm rejects some expert shapes; triton MoE is general.
export SGLANG_MOE_RUNNER_BACKEND="${SGLANG_MOE_RUNNER_BACKEND:-triton}"
# MoE expert compute: the model config hardcodes --moe-grouped-gemm, which
# routes experts through Transformer Engine's general_grouped_gemm. On this
# ROCm/MI300X that kernel deadlocks deterministically — faulthandler (312259)
# caught rank 0 permanently stuck in TE general_grouped_gemm while ranks 1-7
# waited in the MoE combine. MOE_GROUPED_GEMM=0 strips the flag (see _port.py
# rule 5b) → Megatron uses SequentialMLP (per-expert GEMM loop). Numerically
# equivalent; PR² routing-replay unaffected.
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-0}"

# Megatron->HF weight conversion for the sglang weight sync. Stays `raw`
# (HfWeightIteratorDirect) — `bridge` needs the megatron-bridge package which
# is absent from this SIF (312270). The `raw` path's SequentialMLP deadlock
# (312260: per-expert weight broadcast desync) is fixed at the source: a
# miles patch in update_weight/common.py:_named_params_and_buffers_global
# now rewrites SequentialMLP's per-EP-rank-local expert names
# (mlp.experts.local_experts.{local}.linear_fcN.weight) to the canonical
# global grouped name (mlp.experts.linear_fcN.weight{global}), so the EP
# broadcast picks a consistent src rank.
export MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-raw}"

# Megatron MoE expert-parallel token dispatch: the model config hardcodes
# `alltoall`, but RCCL's alltoall_single deadlocks deterministically on this
# MI300X node (311034 + 311039 both hung all 8 EP ranks at SeqNum=3
# ALLTOALL_BASE for the full 600s watchdog window). `allgather` dispatches via
# allgather collectives instead — it only changes HOW tokens reach experts,
# not WHICH experts, so PR² routing-replay semantics are unchanged.
export MOE_TOKEN_DISPATCHER="${MOE_TOKEN_DISPATCHER:-allgather}"
# aiter attention backend for the sglang rollout engine (TP=1 → 32 heads OK).
export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-aiter}"
# TE attention: miles routes ALL models through TE's DotProductAttention,
# and TE reported flash/fused "disabled due to NVTE_FLASH_ATTN=0 /
# NVTE_FUSED_ATTN=0" while unfused rejects qkv_format=thd (inherent to RL
# varlen). Force-enable TE's flash + fused (ROCm CK) backends — the CK fused
# path supports varlen thd — and let Megatron auto-select.
export MEGATRON_ATTENTION_BACKEND="${MEGATRON_ATTENTION_BACKEND:-auto}"
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-1}"
export NVTE_FUSED_ATTN_CK="${NVTE_FUSED_ATTN_CK:-1}"

# --- NCCL flight recorder (DIAGNOSTIC: MoE alltoall deadlock) ---
# 311034/311039/311043 all deadlock at PG 30 SeqNum=3 OpType=ALLTOALL_BASE
# (the EP=8 MoE token dispatch); the dispatcher knob cannot avoid it
# (miles/utils/arguments.py:1895 forces alltoall for varlen RL). Enable the
# torch NCCL flight recorder so the watchdog dumps every rank's per-collective
# trace on timeout — this tells transport-hang (all ranks posted, consistent
# sizes, none completed) apart from cross-rank size mismatch. Default temp
# path is /tmp (lost on node teardown); redirect to /work1 so it survives.
NCCL_TRACE_DIR="/work1/hwang/dzdong/nccl_trace/${SLURM_JOB_ID:-manual}"
mkdir -p "${NCCL_TRACE_DIR}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:-${NCCL_TRACE_DIR}/rank_}"
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"
# faulthandler: every actor periodically dumps all-thread Python stacks to
# stderr. 311922's flight recorder showed rank 0 completes the MoE dispatch
# alltoall but never issues the combine — it hangs in non-NCCL code. This
# captures a hung rank's Python frame. Kept on at a low-noise 600s period as
# a safety net in case the SequentialMLP fix exposes a different hang.
export MILES_FAULTHANDLER_PERIOD="${MILES_FAULTHANDLER_PERIOD:-600}"

# Training infra
export ENABLE_ASYNC_TRAIN="${ENABLE_ASYNC_TRAIN:-0}"
export ENABLE_KEEP_OLD_ACTOR="${ENABLE_KEEP_OLD_ACTOR:-1}"
export ENABLE_AUTO_CKPT_EVAL="${ENABLE_AUTO_CKPT_EVAL:-0}"  # disable for first smoke
export UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-1}"

BASE_SCRIPT="/home1/dzdong/workspace/miles/scripts_my_hpcfund/train/PREEXP-qwen3-30B-A3B-Base/off2/launch/hy-sbatch-2nodes.sh"

exec bash "${BASE_SCRIPT}"
