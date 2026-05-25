# Predictive Routing Replay R2 Port Plan

Status: in progress
Branch: `prr-r2-port`
Baseline commit: `d0f6d88` (`chore: start PRR R2 port plan`)
Date: 2026-03-23

## Progress Tracker

- Phase A: completed
- Phase B: completed
- Phase C: completed
- Phase D: pending
- Phase E: pending

## Progress Log

- 2026-03-23: Started full implementation pass. Re-validated `verl -> miles`
  source mapping, confirmed current `miles` worktree is dirty, and locked the
  phase order to avoid mixing predictive changes with unrelated local edits.
- 2026-03-23: Completed Phase A. Added predictive routing replay CLI flags,
  a dedicated argument validator, and a focused fast test module with local
  dependency stubs so the argument layer can be exercised without full SGLang
  or Ray runtime dependencies.
- 2026-03-23: Phase A verification status:
  - `python -m compileall miles/utils/arguments.py tests/fast/utils/test_predictive_arguments.py`
  - local smoke import with stubbed `transformers` / `ray` / `sglang_router`
    passed
  - `pytest` is not installed in the current shell environment, so the new
    pytest module was verified statically and with direct smoke execution only
- 2026-03-23: Completed Phase B. Added a pure predictive runtime state module
  and a pure data utility module for valid-mask construction, downsampling,
  restore-to-full-batch, and train-time tensor preparation.
- 2026-03-23: Phase B verification status:
  - `python -m compileall miles/backends/megatron_utils/predictive_router_replay.py miles/backends/megatron_utils/predictive_router_utils.py tests/fast/backends/megatron_utils/test_predictive_router_utils.py`
  - current shell Python is missing `torch`, so the new tensor-level fast test
    module could not be executed locally and was only syntax-checked
- 2026-03-23: Completed Phase C. Wired predictive router replay into
  `miles` model setup with a runtime `TopKRouter.forward` overlay, per-router
  bias predictor attachment, and Megatron optimizer `config_overrides` keyed by
  `ParamKey(attr=\"is_bias_predictor\")`.
- 2026-03-23: Phase C design update:
  - instead of extending `docker/patch/latest/megatron.patch` for predictive
    logic, the port now layers a runtime patch on top of the existing Miles
    routing replay patch
  - this keeps the predictive delta isolated from the large Megatron patch file
    and avoids requiring a new base patch rebuild just to iterate on PRR R2
- 2026-03-23: Phase C verification status:
  - `python -m compileall miles/backends/megatron_utils/model.py miles/backends/megatron_utils/predictive_router_replay.py`
  - no Megatron runtime or `torch` package is available in the current shell,
    so integration could only be syntax-checked locally

## 1. Goal

Port the minimum usable `predictive routing replay` implementation from `verl`
into `miles`, limited to:

- `R2` only
- no `union` mode
- no `predictive R3`
- no rollout-side router-state replay

The target is a working actor-side predictive training pipeline in `miles`
that records router states during old-logprob computation and trains a router
bias predictor during actor training.

## 2. Scope

### In scope

- predictive config and CLI flags in `miles`
- Megatron router runtime support for bias predictor
- actor-side predictive state recording for `R2`
- actor-side predictive loss computation during training
- separate optimizer treatment for predictor params
- predictor metrics logging
- enough tests and smoke coverage to keep future implementation stable

### Explicitly out of scope

- predictive `R3`
- `union` mode
- rollout-side router-state capture and decoding
- SGLang predictive router-state transport
- checkpoint export/import support for predictor weights
- HF conversion support for predictor weights

## 3. Source Mapping

### Main `verl` source files

- `verl/verl/workers/config/actor.py`
- `verl/verl/workers/megatron_workers.py`
- `verl/verl/workers/actor/megatron_actor.py`
- `verl/verl/utils/megatron/router_replay_patch.py`
- `verl/verl/utils/megatron/router_replay_utils.py`
- `verl/tests/utils/megatron/test_router_replay_utils.py`

### Main `miles` target files

- `miles/utils/arguments.py`
- `miles/utils/replay_base.py`
- `miles/backends/megatron_utils/model_provider.py`
- `miles/backends/megatron_utils/model.py`
- `miles/backends/megatron_utils/actor.py`
- `miles/backends/megatron_utils/replay_utils.py`
- `miles/backends/training_utils/data.py`
- `miles/backends/training_utils/log_utils.py`
- `miles/backends/megatron_utils/update_weight/common.py`
- `docker/patch/latest/megatron.patch`

## 4. Key Architectural Difference

This port is not a direct file copy.

`verl` predictive `R2` is integrated into a PPO actor update flow that has
explicit mini-step scheduling inside `update_policy`.

`miles` currently uses a different actor loop:

1. compute old logprobs
2. compute advantages / returns
3. run one actor train step pipeline

Because of this difference:

- the predictive router patch logic can be reused with adaptation
- the `verl` control flow cannot be copied verbatim
- the `miles` implementation must translate the predictive phases into the
  existing `forward_only -> train_one_step` structure

This is the main reason the plan is implementation-driven rather than a blind
sync from `verl`.

## 5. Design Decision For Phase 1

For the first port, predictive support will be actor-side only.

Recommended phase-1 rule:

- rollout engines do not need to own or execute the predictor
- rollout weight sync should skip predictor-only parameters

Reason:

- `R2` predictive training uses actor-side recorded router states
- rollout does not need predictor behavior for this phase
- this avoids having to patch SGLang model structure immediately
- this keeps the first implementation small and testable

If later we want predictor-aware rollout or full structural parity between
actor and rollout models, that will be a follow-up phase.

## 6. Functional Target

The final phase-1 behavior should be:

1. Enable standard `R2` replay in `miles`.
2. When predictive mode is enabled, each router also owns a bias predictor.
3. During old-logprob computation, router inputs and router logits are recorded.
4. The recorded predictive tensors are downsampled and attached to the local
   rollout/training data path.
5. During actor training, the predictor computes a loss against the delta
   between old logits and current logits.
6. Predictor gradients are applied only on predictive-enabled train passes.
7. Predictor-specific metrics are logged.

## 7. Work Breakdown

### Phase A: Configuration Surface

Add predictive-specific flags to `miles`.

Status:

- completed on 2026-03-23

Target file:

- `miles/utils/arguments.py`

Required args:

- `--enable-predictive-routing-replay`
- `--bias-predictor-loss-type`
- `--bias-predictor-lr-mult`
- `--predictive-downsample-batch-size`
- `--predictive-downsample-max-len-limit`
- `--predictive-storage-dtype`

Validation rules:

- predictive mode requires `--use-routing-replay`
- supported loss types are `l2`, `kl`, `kl-post`
- supported storage dtypes are `fp32`, `bf16`, `fp16`
- `union` flags are not exposed in phase 1

Acceptance criteria:

- args parse successfully
- invalid combinations fail early with clear errors

### Phase B: Predictive Runtime Primitive

Create a dedicated predictive runtime layer for `miles`.

Status:

- completed on 2026-03-23

Recommended new file:

- `miles/backends/megatron_utils/predictive_router_replay.py`

Responsibilities:

- define predictive action enum
- hold per-router predictive state
- expose helper methods to:
  - record old inputs/logits
  - set predictive data for train-time use
  - clear predictive data
  - record predictive metrics
- provide patch logic for router forward

Content to adapt from `verl`:

- `RouterPredictiveAction`
- per-layer predictive storage
- predictive metric trackers
- predictive loss computation
- top-k accuracy helper

Phase-1 simplification:

- keep only R2 paths
- remove R3-only fields:
  - `old_bias`
  - `old_token_positions`
  - `router_request_id`
  - `R3_COLLECT_STATS`
- remove all union-mode code paths

Acceptance criteria:

- runtime module can be imported independently
- all predictive state transitions are explicit and testable

### Phase C: Megatron Patch Integration

Extend the existing `miles` Megatron patch.

Status:

- completed on 2026-03-23

Implementation note:

- the predictive port now uses a runtime `TopKRouter.forward` overlay plus
  post-build router attachment instead of expanding `docker/patch/latest/megatron.patch`

Target file:

- `docker/patch/latest/megatron.patch`

Current patch already:

- wraps router top-k through `routing_replay_manager.get_topk_fn(...)`
- registers replay object on router instances

Required additions:

- attach predictor module to MoE router when predictive mode is enabled
- mark predictor params with an attribute such as `is_bias_predictor=True`
- expose enough router-local state so predictive runtime can access:
  - input tensor
  - router logits
  - replay action object
- allow router forward to branch between:
  - record predictive state
  - skip predictive loss
  - compute predictive loss

Important:

- keep the existing R2/R3 replay behavior unchanged when predictive is off
- keep patch idempotent

Acceptance criteria:

- predictive off path is behaviorally unchanged
- predictive on path creates predictor params on each router

### Phase D: Optimizer Integration

Predictor parameters need a different learning rate regime.

Target files:

- `miles/backends/megatron_utils/model.py`
- possibly a small helper under `miles/backends/megatron_utils/`

Implementation target:

- mirror the `verl` idea of using `ParamKey(attr='is_bias_predictor')`
- pass `config_overrides` into `get_megatron_optimizer(...)`
- predictor param-group LR should be:
  - `base_lr * bias_predictor_lr_mult`

Acceptance criteria:

- predictor params land in a dedicated optimizer override path
- logged param-group LRs clearly show predictor multiplier was applied

### Phase E: Old-Logprob Recording Path

Add predictive recording during old-logprob computation.

Target files:

- `miles/backends/megatron_utils/actor.py`
- `miles/backends/megatron_utils/model.py`
- `miles/backends/training_utils/log_utils.py`

Required behavior:

- when predictive mode is enabled, `compute_log_prob(...)` runs routers in
  predictive `RECORD` mode
- the forward pass should return:
  - standard old logprobs
  - standard replay data
  - predictive old inputs / old logits

Needed support:

- a forward collection path analogous to `verl`'s `layers_predictive_states`
- aggregation over micro-batches in the order expected by `miles`

Important adaptation:

- this data path must fit `miles`'s `forward_only(...)` and
  `aggregate_forward_results(...)` structure
- values are best carried as per-sample Python lists of variable-shape arrays,
  not padded tensors

Acceptance criteria:

- after old-logprob pass, predictive data exists for each sampled sequence
- no predictive data is generated when predictive mode is disabled

### Phase F: Predictive Data Packing And Transport

Adapt the `verl` predictive helpers into `miles`.

Recommended new file:

- `miles/backends/megatron_utils/predictive_router_utils.py`

Content to bring over in simplified form:

- `build_predictive_valid_mask(...)`
- downsample helpers
- per-sample concat / split helpers
- `set_router_predictive_data(...)`
- merge logic for old inputs / logits

Phase-1 removals:

- all `old_bias` logic
- all `old_token_positions` logic
- all `router_request_id` logic
- all `predictive_max_total_tokens` logic unless needed for stability later

Data model for phase 1:

- `old_inputs`: list of arrays `[num_tokens_i, num_layers, hidden]`
- `old_logits`: list of arrays `[num_tokens_i, num_layers, num_experts]`

Transport path inside `miles`:

1. attach to rollout/training dict after old-logprob
2. preserve through `DataIterator`
3. split into micro-batches aligned with training order
4. load into local router instances before actor train forward

Acceptance criteria:

- predictive data stays aligned with samples after dynamic batching
- valid mask matches current packed-token layout

### Phase G: Actor Train-Time Predictive Loss

Integrate predictive loss into the actor train pipeline.

Target files:

- `miles/backends/megatron_utils/actor.py`
- `miles/backends/megatron_utils/model.py`

Required train-stage control:

- before training forward:
  - set predictive action to `COMPUTE_PREDICTIVE_LOSS` if data is present
  - otherwise set to `SKIP_PREDICTIVE`
- after forward:
  - clear predictive state

Predictor safety rules:

- clear predictor grad state before each train step
- do not let stale Adam state move predictor on skipped steps
- if predictive loss is skipped, predictor param-group should be temporarily
  disabled or predictor grads explicitly scrubbed before optimizer step

This part must be adapted carefully because `miles` does not use the same
mini-step schedule as `verl`.

Phase-1 rule:

- compute predictive loss on the actor train pass that consumes the recorded
  old logprob states
- do not attempt to recreate `verl`'s full multi-mini-step schedule

Acceptance criteria:

- training runs with predictive enabled and no shape mismatch
- predictor weights only change on predictive-enabled train passes

### Phase H: Logging

Add predictor metrics to `miles` logging.

Target files:

- `miles/backends/training_utils/log_utils.py`
- predictive runtime module

Metrics to expose:

- `predictive_loss`
- `predictive_topk_accuracy`
- `predictive_bias_to_logits_ratio`

Acceptance criteria:

- metrics appear only when predictive is enabled
- reductions are consistent with current logging conventions

### Phase I: Rollout Weight Sync Compatibility

Phase-1 recommended implementation:

- filter predictor-only params out of rollout weight sync

Target file:

- `miles/backends/megatron_utils/update_weight/common.py`

Reason:

- rollout engines do not need predictor params for actor-side-only R2 predictive
- current rollout model schema likely does not contain predictor weights
- sending unmatched names to rollout engines is an unnecessary failure mode

Acceptance criteria:

- rollout weight sync still succeeds when predictive is enabled
- actor retains full predictor params locally

## 8. Files Likely To Be Touched

Primary implementation files:

- `miles/utils/arguments.py`
- `miles/utils/replay_base.py`
- `miles/backends/megatron_utils/actor.py`
- `miles/backends/megatron_utils/model.py`
- `miles/backends/megatron_utils/model_provider.py`
- `miles/backends/megatron_utils/replay_utils.py`
- `miles/backends/training_utils/data.py`
- `miles/backends/training_utils/log_utils.py`
- `miles/backends/megatron_utils/update_weight/common.py`
- `docker/patch/latest/megatron.patch`

Recommended new files:

- `miles/backends/megatron_utils/predictive_router_replay.py`
- `miles/backends/megatron_utils/predictive_router_utils.py`
- `tests/fast/backends/megatron_utils/test_predictive_router_utils.py`
- `tests/fast/utils/test_predictive_arguments.py`

## 9. Current Local Risk

The current branch is based on a dirty working tree. The following existing user
modified files must be treated as hot files during implementation:

- `miles/utils/arguments.py`
- `miles/utils/types.py`
- `miles/rollout/sglang_rollout.py`
- `miles/utils/data.py`

Implementation rule:

- do not overwrite existing user edits
- re-read those files immediately before each patch
- keep changes narrow and additive

## 10. Validation Plan

### Unit tests

Add or update tests for:

- predictive argument parsing and validation
- predictive valid-mask construction
- downsample behavior
- predictive data alignment under dynamic batching
- predictor param-group LR override selection

### Fast integration tests

Add a narrow Megatron-side smoke test that verifies:

- predictive recording path runs
- predictive train path runs
- predictor param-group exists

### Manual smoke run

Before broader training:

1. run a tiny MoE training smoke with `--use-routing-replay`
2. enable predictive with small batch and short context
3. verify:
   - no router-state shape errors
   - no weight-sync failure
   - predictor metrics are logged

### Acceptance checkpoint for phase 1

Phase 1 is complete only if all of the following are true:

- `miles` trains with `R2 + predictive` enabled
- predictive metrics are emitted
- rollout sync still works
- predictor params are not accidentally applied to rollout engines
- non-predictive R2 and existing R3 paths do not regress

## 11. Implementation Order

Recommended execution order:

1. add CLI/config surface
2. add predictive runtime module
3. patch Megatron router structure and forward
4. add optimizer override for predictor params
5. add old-logprob predictive recording path
6. add predictive data packing and micro-batch transport
7. add train-time predictive loss control
8. add logging
9. add rollout weight-sync filtering
10. add tests
11. run smoke validation

This order minimizes time spent debugging mixed failures from multiple layers.

## 12. Non-Goals For The First Implementation Pass

The first implementation pass should not attempt to solve:

- predictive rollout behavior
- predictor-aware SGLang model structure
- predictor checkpoint export/import
- union-mode replay semantics
- R3 tensor-position alignment

If any of these becomes necessary during implementation, stop and treat it as a
scope change instead of silently extending phase 1.

## 13. Follow-Up Phases After Phase 1

### Follow-up A

Predictive `R3` using rollout-side router states:

- `router_inputs`
- `router_logits`
- `router_bias`
- `router_token_positions`

### Follow-up B

Predictor-aware rollout model structure and weight sync.

### Follow-up C

Checkpoint and HF export support for predictor parameters.

### Follow-up D

Union mode.

## 14. Working Rules For Future Implementation

When implementing against this document:

- preserve existing user modifications
- do not broaden scope without updating this document
- prefer additive files over invasive rewrites
- keep `R2` predictive logic isolated from `R3`
- add tests before enabling optional extensions

If implementation reality forces a deviation from this plan, update this
document in the same change set that introduces the deviation.
