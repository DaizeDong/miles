#!/bin/bash

build_replay_args() {
  local replay_args_var="${1:-REPLAY_ARGS}"
  declare -n replay_args_ref="${replay_args_var}"

  replay_args_ref=()
  if [ "${USE_ROUTING_REPLAY}" = "1" ] || [ "${USE_ROLLOUT_ROUTING_REPLAY}" = "1" ]; then
    replay_args_ref+=(--use-routing-replay)
  fi

  if [ "${ENABLE_PREDICTIVE_ROUTING_REPLAY}" != "1" ]; then
    return
  fi

  replay_args_ref+=(
    --enable-predictive-routing-replay
    --bias-predictor-loss-type "${BIAS_PREDICTOR_LOSS_TYPE}"
    --bias-predictor-lr-mult "${BIAS_PREDICTOR_LR_MULT}"
    --predictive-storage-dtype "${PREDICTIVE_STORAGE_DTYPE}"
  )

  if [ -n "${PREDICTIVE_DOWNSAMPLE_BATCH_SIZE}" ] && [ "${PREDICTIVE_DOWNSAMPLE_BATCH_SIZE}" != "null" ]; then
    replay_args_ref+=(--predictive-downsample-batch-size "${PREDICTIVE_DOWNSAMPLE_BATCH_SIZE}")
  fi

  if [ -n "${PREDICTIVE_DOWNSAMPLE_MAX_LEN_LIMIT}" ] && [ "${PREDICTIVE_DOWNSAMPLE_MAX_LEN_LIMIT}" != "null" ]; then
    replay_args_ref+=(--predictive-downsample-max-len-limit "${PREDICTIVE_DOWNSAMPLE_MAX_LEN_LIMIT}")
  fi

  if [ -n "${PREDICTIVE_MAX_TOTAL_TOKENS:-}" ] && [ "${PREDICTIVE_MAX_TOTAL_TOKENS}" != "null" ]; then
    replay_args_ref+=(--predictive-max-total-tokens "${PREDICTIVE_MAX_TOTAL_TOKENS}")
  fi

  if [ -n "${PREDICTIVE_MAX_HIDDEN_SHIFT_RELATIVE_NORM:-}" ] && [ "${PREDICTIVE_MAX_HIDDEN_SHIFT_RELATIVE_NORM}" != "null" ]; then
    replay_args_ref+=(--predictive-max-hidden-shift-relative-norm "${PREDICTIVE_MAX_HIDDEN_SHIFT_RELATIVE_NORM}")
  fi

  if [ -n "${PREDICTIVE_HIDDEN_SHIFT_WEIGHT_MODE:-}" ] && [ "${PREDICTIVE_HIDDEN_SHIFT_WEIGHT_MODE}" != "binary" ]; then
    replay_args_ref+=(--predictive-hidden-shift-weight-mode "${PREDICTIVE_HIDDEN_SHIFT_WEIGHT_MODE}")
  fi

  if [ -n "${PREDICTIVE_BOUNDARY_LOSS_MAX_WEIGHT:-}" ] && [ "${PREDICTIVE_BOUNDARY_LOSS_MAX_WEIGHT}" != "null" ]; then
    replay_args_ref+=(--predictive-boundary-loss-max-weight "${PREDICTIVE_BOUNDARY_LOSS_MAX_WEIGHT}")
  fi

  if [ -n "${PREDICTIVE_BOUNDARY_LOSS_MIN_MARGIN:-}" ] && [ "${PREDICTIVE_BOUNDARY_LOSS_MIN_MARGIN}" != "null" ]; then
    replay_args_ref+=(--predictive-boundary-loss-min-margin "${PREDICTIVE_BOUNDARY_LOSS_MIN_MARGIN}")
  fi

  if [ -n "${PREDICTIVE_MIN_POST_TOPK_MARGIN_FOR_FLIP:-}" ] && [ "${PREDICTIVE_MIN_POST_TOPK_MARGIN_FOR_FLIP}" != "null" ]; then
    replay_args_ref+=(--predictive-min-post-topk-margin-for-flip "${PREDICTIVE_MIN_POST_TOPK_MARGIN_FOR_FLIP}")
  fi

  if [ -n "${PREDICTIVE_LAYER_SCALE_SCHEDULE:-}" ] && [ "${PREDICTIVE_LAYER_SCALE_SCHEDULE}" != "none" ]; then
    replay_args_ref+=(--predictive-layer-scale-schedule "${PREDICTIVE_LAYER_SCALE_SCHEDULE}")
  fi

  if [ -n "${PREDICTIVE_LAYER_SCALE_MIN:-}" ] && [ "${PREDICTIVE_LAYER_SCALE_MIN}" != "1.0" ]; then
    replay_args_ref+=(--predictive-layer-scale-min "${PREDICTIVE_LAYER_SCALE_MIN}")
  fi

  if [ -n "${PREDICTIVE_MAX_DELTA_TO_OLD_RATIO:-}" ] && [ "${PREDICTIVE_MAX_DELTA_TO_OLD_RATIO}" != "null" ]; then
    replay_args_ref+=(--predictive-max-delta-to-old-ratio "${PREDICTIVE_MAX_DELTA_TO_OLD_RATIO}")
  fi

  if [ -n "${PREDICTIVE_MAX_DELTA_TO_TOPK_MARGIN_RATIO:-}" ] && [ "${PREDICTIVE_MAX_DELTA_TO_TOPK_MARGIN_RATIO}" != "null" ]; then
    replay_args_ref+=(--predictive-max-delta-to-topk-margin-ratio "${PREDICTIVE_MAX_DELTA_TO_TOPK_MARGIN_RATIO}")
  fi

  if [ -n "${PREDICTIVE_MAX_DELTA_TO_TOPK_MARGIN_RATIO_FINAL:-}" ] && [ "${PREDICTIVE_MAX_DELTA_TO_TOPK_MARGIN_RATIO_FINAL}" != "null" ]; then
    replay_args_ref+=(--predictive-max-delta-to-topk-margin-ratio-final "${PREDICTIVE_MAX_DELTA_TO_TOPK_MARGIN_RATIO_FINAL}")
  fi

  if [ -n "${PREDICTIVE_TOPK_MARGIN_RATIO_ANNEAL_START_ROLLOUT:-}" ] && [ "${PREDICTIVE_TOPK_MARGIN_RATIO_ANNEAL_START_ROLLOUT}" != "null" ]; then
    replay_args_ref+=(--predictive-topk-margin-ratio-anneal-start-rollout "${PREDICTIVE_TOPK_MARGIN_RATIO_ANNEAL_START_ROLLOUT}")
  fi

  if [ -n "${PREDICTIVE_TOPK_MARGIN_RATIO_ANNEAL_END_ROLLOUT:-}" ] && [ "${PREDICTIVE_TOPK_MARGIN_RATIO_ANNEAL_END_ROLLOUT}" != "null" ]; then
    replay_args_ref+=(--predictive-topk-margin-ratio-anneal-end-rollout "${PREDICTIVE_TOPK_MARGIN_RATIO_ANNEAL_END_ROLLOUT}")
  fi
}
