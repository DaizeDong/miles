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
}
