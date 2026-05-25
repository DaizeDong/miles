#!/bin/bash
# === Moonlight 16B-A3B PR² A0 (paper-faithful), single-node colocate ===
# Adapted from user's verl-based "moon-base-gsm-pr2-1e2" 8-node disagg run.
# Mapped to Miles env vars + scaled to mi3008x QoS limit (1 node × 8 MI300X × 12h).
#
# Algorithm (= verl version):
#   - kl-post bias predictor, lr_mult=1e2
#   - downsample batch=4, NO additional stabilization (paper baseline)
#   - LR=5e-7 (verl actor_lr), temperature=1.0, val_top_p=0.7
#   - max_prompt=1024, max_response=1024 (verl values, much smaller than Miles default)
#   - rollout n=16, batch_size=32, off2 (NUM_STEPS_PER_ROLLOUT=2)
#
# Resources:
#   - 1 node, 8× MI300X, colocate actor + rollout (forced by QoS)
#   - ROLLOUT_NUM_GPUS_PER_ENGINE=4 (verl INFER_TP=4) → 2 sglang engines / node
#   - EP=8, TP=1, PP=1 (verl COMMON_* defaults)
#
# NUM_ROLLOUT default lowered to 20 for cold-boot smoke. Ramp via env when stable.
#SBATCH --job-name=moonlight-off2-pr2-A0-klpost1e2
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
export MODEL_NAME="${MODEL_NAME:-Moonlight-16B-A3B}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-/root/miles/scripts/models/moonlight.sh}"
export RUN_POSTFIX="${RUN_POSTFIX:-off2-pr2-A0-klpost1e2-1node}"

# Resource layout (QoS forces 1-node, colocate)
export RESOURCE_LAYOUT="${RESOURCE_LAYOUT:-colocate}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
# verl used INFER_TP=4, but the aiter FMHA kernel asserts num_head_qo % 16 == 0.
# Moonlight has 16 attention heads, so TP>1 gives <16 heads/rank and fails.
# Use TP=1 (one full-model sglang engine per GPU, 8 engines).
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"

# Parallelism (verl COMMON_* matches Miles defaults)
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
export EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-8}"
export EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"

# Sequence lengths (verl values; much smaller than Miles default 2048/16384)
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export EVAL_MAX_RESPONSE_LENGTH="${EVAL_MAX_RESPONSE_LENGTH:-1024}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"

# Rollout / optimizer
export NUM_ROLLOUT="${NUM_ROLLOUT:-20}"            # smoke; raise to 270+ once stable
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
# GLOBAL_BATCH_SIZE = 32 * 16 / 2 = 256  (matches Miles default formula)
export LR="${LR:-5e-7}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export EVAL_TOP_P="${EVAL_TOP_P:-0.7}"

# PR² mechanism (paper-faithful, no extra stabilization → A0)
export USE_MILES_ROUTER="${USE_MILES_ROUTER:-1}"
export USE_ROUTING_REPLAY="${USE_ROUTING_REPLAY:-1}"
export USE_ROLLOUT_ROUTING_REPLAY="${USE_ROLLOUT_ROUTING_REPLAY:-0}"
export ENABLE_PREDICTIVE_ROUTING_REPLAY="${ENABLE_PREDICTIVE_ROUTING_REPLAY:-1}"
export BIAS_PREDICTOR_LOSS_TYPE="${BIAS_PREDICTOR_LOSS_TYPE:-kl-post}"
export BIAS_PREDICTOR_LR_MULT="${BIAS_PREDICTOR_LR_MULT:-1e2}"
export PREDICTIVE_DOWNSAMPLE_BATCH_SIZE="${PREDICTIVE_DOWNSAMPLE_BATCH_SIZE:-4}"  # verl: 4
export PREDICTIVE_DOWNSAMPLE_MAX_LEN_LIMIT="${PREDICTIVE_DOWNSAMPLE_MAX_LEN_LIMIT:-2048}"  # ≥ prompt+resp
export PREDICTIVE_STORAGE_DTYPE="${PREDICTIVE_STORAGE_DTYPE:-fp32}"
# All enhancement vars left UNSET (paper A0 baseline; defaults disable them).

# A plain contiguous D2H copy of a normal Parameter (word_embeddings.weight)
# raises `HIP error: invalid argument` in tensor_backper. Strong suspect:
# PyTorch's expandable_segments allocator hands out virtual-mapped GPU
# pointers that hipMemcpy rejects. Force plain hipMalloc segments.
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:False}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"

# sglang CUDA-graph capture crashes on the SIF's sglang triton backend
# (TypeError: cannot unpack non-iterable ForwardMetadata object). Disable it.
export SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-1}"

# The auto MoE runner picks aiter's CK 2-stage gemm, which rejects Moonlight's
# expert GEMM shape ("device_gemm ... does not support this GEMM problem").
# Force the triton MoE runner, which handles arbitrary expert dims.
export SGLANG_MOE_RUNNER_BACKEND="${SGLANG_MOE_RUNNER_BACKEND:-triton}"

# Transformer Engine on this ROCm (gfx942) has no usable MLA attention:
# NVTE_DEBUG showed FlashAttention/FusedAttention unavailable and the pure
# UnfusedDotProductAttention disabled for qkv_format=thd. thd comes from
# --use-dynamic-batch-size; turning it off makes miles use bshd, which the
# unfused backend accepts. Slower (pure-PyTorch attention) but correct.
export MEGATRON_ATTENTION_BACKEND="${MEGATRON_ATTENTION_BACKEND:-unfused}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-0}"

# sglang's triton attention backend is broken for DeepSeek-V2-style MLA
# (Moonlight): deepseek_v2.forward_absorb_fused_mla_rope_prepare unpacks
# forward_metadata as a tuple but triton now stores a ForwardMetadata object.
# Use the aiter attention backend instead.
export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-aiter}"

# Training infra
export ENABLE_ASYNC_TRAIN="${ENABLE_ASYNC_TRAIN:-0}"
export ENABLE_KEEP_OLD_ACTOR="${ENABLE_KEEP_OLD_ACTOR:-1}"
export ENABLE_AUTO_CKPT_EVAL="${ENABLE_AUTO_CKPT_EVAL:-0}"  # disable for first smoke; eval path needs verification
export UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-1}"

BASE_SCRIPT="/home1/dzdong/workspace/miles/scripts_my_hpcfund/train/PREEXP-qwen3-30B-A3B-Base/off2/launch/hy-sbatch-2nodes.sh"

exec bash "${BASE_SCRIPT}"
