# Predictive Routing Replay Runtime

This note documents the current predictive routing replay structure after the controller-centered cleanup.

## Runtime layout

- `miles/backends/megatron_utils/predictive_router_replay.py`
  - Owns the predictive runtime patch on `TopKRouter.forward`.
  - Exposes `PredictiveReplayController` as the single control-plane object for:
    - registered router states
    - global predictive action
    - recorded predictive microbatch queue
    - per-train-step usage flag
    - predictive metrics and metric tensor capture
  - Keeps `PredictiveRouterReplayState`, `PredictiveRouterReplayBuffer`, and `PredictiveTrainStepState` as compatibility wrappers.
- `miles/backends/megatron_utils/predictive_router_utils.py`
  - Only keeps the production packing path.
  - `RecordedPredictiveMicrobatch` is the cross-pass transport object.
  - `pack_recorded_predictive_microbatch(...)` is the only public packing API that the runtime depends on.
- `miles/backends/megatron_utils/model.py`
  - Collects recorded predictive tensors after old-logprob forward.
  - Delegates train-pass mode changes to the controller.
  - Gates predictor optimizer groups when a train step does not consume valid predictive data.
- `miles/backends/megatron_utils/actor.py`
  - Builds an explicit actor train-pass plan:
    - baseline: `["compute"]`
    - predictive: `["skip", "compute"]`
  - Resets replay read indices and predictive microbatch cursor between predictive passes.

## Data flow

1. Old actor log-prob pass runs with `RouterPredictiveAction.RECORD`.
2. Each patched router records local old inputs/logits and logs predicted bias into the routing replay artifact.
3. `model.forward_only(...)` collects per-layer recorded tensors and packs them into a `RecordedPredictiveMicrobatch`.
4. Actor training reuses the same rollout batch twice when predictive replay is enabled:
   - pass 0: `skip`
   - pass 1: `compute`
5. During compute pass, the controller loads the next recorded predictive microbatch into router-local state.
6. Patched routers compute predictor loss and top-k metrics, then normal actor optimization proceeds.

## Artifact layout

- `miles/backends/megatron_utils/router_replay_artifacts.py` is the shared naming and loading layer.
- Main payload:
  - `{step}_tp{tp_rank}_pp{pp_rank}.pt`
- Predictive sidecars:
  - `{step}_tp{tp_rank}_pp{pp_rank}_predictive_metrics.json`
  - `{step}_tp{tp_rank}_pp{pp_rank}_predictive_metric_tensors.pt`
- `router_replay_saver.py` and `examples/router_replay/analyze_saved_logits.py` both use this shared protocol.

## Launcher layout

- `scripts_mine/train/PREEXP-qwen3-30B-A3B-Base/off2/launch/_predictive_replay_args.sh`
  - Shared shell helper for constructing `REPLAY_ARGS`.
- Node-specific launchers keep resource/topology logic local and source the helper for predictive routing replay flags.

## Remaining cleanup targets

- Remove compatibility wrappers once all call sites stop importing `PredictiveRouterReplayBuffer` and `PredictiveTrainStepState`.
- Fold predictive-specific debug/test helpers that still mirror old abstractions.
- If predictive becomes stable, move this note into the main runtime docs and document the analysis outputs more formally.
