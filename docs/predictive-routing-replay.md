# Predictive Routing Replay (PR²)

This document describes Miles's implementation of Predictive Routing Replay
(PR²): the runtime architecture, the algorithm (and where Miles diverges
from the paper), and the user-facing configuration parameters.

Core source:

- `miles/backends/megatron_utils/predictive_router_replay.py` — runtime, controller, patched router forward, all algorithmic helpers (stabilization, flip-fallback, loss reweighting, synthetic-loss synchronization).
- `miles/backends/megatron_utils/predictive_router_utils.py` — packing/transport of recorded predictive microbatches.
- `miles/backends/megatron_utils/router_replay_artifacts.py` — on-disk artifact naming/loading.
- `miles/backends/megatron_utils/router_replay_saver.py` — saver protocol.
- `miles/backends/megatron_utils/model.py`, `model_provider.py`, `actor.py` — integration with the Megatron training loop and actor train-pass plan.
- `miles/utils/arguments.py` — CLI flag definitions.
- `scripts_mine/train/PREEXP-*/off*/launch/_predictive_replay_args.sh` — shell helper that translates env vars to CLI args.

---

## 1. Runtime layout

`predictive_router_replay.py` owns the patch on `TopKRouter.forward` and exposes `PredictiveReplayController` as the single control-plane object. The controller holds:

- registered per-layer router states,
- the global predictive action (RECORD / SKIP / COMPUTE_PREDICTIVE_LOSS / DISABLED),
- the recorded predictive microbatch queue,
- per-train-step usage flag,
- predictive metrics + metric-tensor capture.

`PredictiveRouterReplayState`, `PredictiveRouterReplayBuffer`, and `PredictiveTrainStepState` remain as compatibility wrappers around the controller.

`model.py` collects recorded predictive tensors after the old log-prob forward, delegates train-pass mode changes to the controller, and gates predictor optimizer groups when a train step has no valid predictive data.

`actor.py` builds an explicit actor train-pass plan: `["compute"]` baseline, `["skip", "compute"]` with predictive replay. It resets the replay read indices and predictive microbatch cursor between predictive passes.

## 2. Data flow

1. Old actor log-prob pass runs with `RouterPredictiveAction.RECORD`.
2. Each patched router records local old inputs/logits and logs the predicted bias into the routing replay artifact.
3. `model.forward_only(...)` collects per-layer recorded tensors and packs them into a `RecordedPredictiveMicrobatch`.
4. Actor training reuses the same rollout batch twice when predictive replay is enabled:
   - pass 0: `skip` — pure on-policy update, no predictive loss
   - pass 1: `compute` — loads the recorded predictive microbatch into router-local state
5. During the compute pass the patched routers compute the predictor loss and top-k metrics, then normal actor optimization proceeds.

## 3. Artifact layout

`router_replay_artifacts.py` is the shared naming/loading layer.

| File | Purpose |
|---|---|
| `{step}_tp{tp_rank}_pp{pp_rank}.pt` | Main routing-replay payload |
| `{step}_tp{tp_rank}_pp{pp_rank}_predictive_metrics.json` | Per-layer predictive metrics |
| `{step}_tp{tp_rank}_pp{pp_rank}_predictive_metric_tensors.pt` | Per-layer captured tensors (input/logits/applied_delta) |

`router_replay_saver.py` and `examples/router_replay/analyze_saved_logits.py` use this shared protocol.

---

## 4. Algorithm

### 4.1 Paper PR² (reference)

**Rollout phase** (paper Eq. 4–6):

$$\hat{\rho}^{(l)}_t = \mathrm{Softmax}\!\left(p^{(l)}_{\mathrm{old},t} + h^{(l)}_{\mathrm{old},t} W^{(l)}_p\right), \quad \hat{\mathcal{I}}^{(l)}_t = \mathrm{TopK}(\hat{\rho}^{(l)}_t, k)$$

The predictor delta is added to base logits and routed via softmax + top-k. The predicted index $\hat{\mathcal{I}}$ is cached for replay.

**Training phase** (paper Eq. 7–8 + Algorithm 3):

$$\mathcal{L}_{\mathrm{PR}^2} = \sum_{l=1}^L \mathbb{E}_t\!\left[ D_{\mathrm{KL}}\!\left(\langle\rho^{(l)}_t\rangle \,\big\|\, \hat{\rho}^{(l)}_t\right) \right]$$

Uniform token weighting, stop-grad on the current router's $\rho^{(l)}_t$ as teacher. Mini-step $i=1$ skips the predictive loss; predictor uses a dedicated learning-rate multiplier $\alpha$.

### 4.2 Miles enhancements

Miles adds a stabilization/safety layer + sample reweighting that the paper does not specify. See §6 for parameter semantics.

**Rollout-side stabilization** (`stabilize_predictive_delta_logits`):

Form $\tilde{b}^{(l)}_t = h^{(l)}_{\mathrm{old},t} W^{(l)}_p$, then apply:

(i) **Depth-aware layer gating**

$$\tilde{b}^{(l)}_t \leftarrow \gamma_l \cdot \tilde{b}^{(l)}_t, \quad \gamma_l \in \{\mathrm{none}, \mathrm{linear}_\downarrow, \sqrt{}_\downarrow, \cos_\downarrow\}(l/L)$$

(ii) **Magnitude clip** (`PREDICTIVE_MAX_DELTA_TO_OLD_RATIO=r_max`)

$$\tilde{b}^{(l)}_t \leftarrow \tilde{b}^{(l)}_t \cdot \min\!\left(1,\ r_{\max} \cdot \overline{|p^{(l)}_{\mathrm{old},t}|}\Big/\overline{|\tilde{b}^{(l)}_t|}\right)$$

(iii) **Top-k boundary-margin clip** (`PREDICTIVE_MAX_DELTA_TO_TOPK_MARGIN_RATIO`, with cross-rollout anneal)

$$b^{(l)}_t \leftarrow \tilde{b}^{(l)}_t \cdot \min\!\left(1,\ \frac{\eta_t \cdot m^{(l)}_t}{2 \max_j |\tilde{b}^{(l)}_{t,j}|}\right)$$

where $m^{(l)}_t = p^{(l)}_{\mathrm{old},t,(k)} - p^{(l)}_{\mathrm{old},t,(k+1)}$ is the top-k boundary margin.

**Flip-fallback** (`apply_predictive_flip_fallback`, gated by `PREDICTIVE_MIN_POST_TOPK_MARGIN_FOR_FLIP=τ`):

When $\hat{\mathcal{I}}^{(l)}_t \neq \mathcal{I}^{(l)}_{\mathrm{old},t}$ and the post-correction margin $\hat{m}^{(l)}_t < \tau$, revert the token to original logits ($b^{(l)}_t \leftarrow 0$) and set its execution weight $w^{\mathrm{exec},(l)}_t = 0$. Confidently predicted tokens keep $w^{\mathrm{exec},(l)}_t = 1$.

**Training loss with sample reweighting**:

$$\mathcal{L}^{\mathrm{Miles}}_{\mathrm{PR}^2} = \sum_{l=1}^L \frac{\sum_t w^{(l)}_t \cdot D_{\mathrm{KL}}\!\bigl(\langle\rho^{(l)}_t\rangle \,\big\|\, \hat{\rho}^{(l)}_t\bigr)}{\sum_t w^{(l)}_t}, \quad w^{(l)}_t = w^{\mathrm{shift},(l)}_t \cdot w^{\mathrm{bdry},(l)}_t \cdot w^{\mathrm{exec},(l)}_t$$

where:

- **Hidden-shift weight** (`PREDICTIVE_MAX_HIDDEN_SHIFT_RELATIVE_NORM=τ_h`, mode `binary`/`linear`/`quadratic`): supervise only on tokens whose router input has drifted from the route-recording snapshot.
- **Boundary weight** (`PREDICTIVE_BOUNDARY_LOSS_MAX_WEIGHT=w_max`, `PREDICTIVE_BOUNDARY_LOSS_MIN_MARGIN=m_min`): $\min(w_{\max}, 1/\max(m^{(l)}_t, m_{\min}))$.
- **Execution weight**: from flip-fallback.

`compute_predictive_loss` then computes the chosen loss form (`l2`, `kl`, `kl-post`) over the stabilized $\hat{\rho}$.

**Synchronization loss** (`build_synthetic_predictive_loss`): when a DP rank holds no valid predictive tokens, construct $(W_p \cdot x_{\mathrm{any}}).\mathrm{sum}() \cdot 0$ so $W_p$ retains a backward graph and the collective stays in sync.

---

## 5. Configuration parameters

All parameters below are surfaced through `scripts_mine/train/PREEXP-*/off*/launch/_predictive_replay_args.sh`, which translates env vars to CLI flags consumed by `miles/utils/arguments.py`. When an env var is left empty / set to `null` / equals its default, the corresponding CLI flag is not emitted and the feature is disabled.

### 5.1 Required (when PR² is on)

| Env var | CLI flag | Meaning |
|---|---|---|
| `ENABLE_PREDICTIVE_ROUTING_REPLAY=1` | `--enable-predictive-routing-replay` | Master switch |
| `BIAS_PREDICTOR_LOSS_TYPE` | `--bias-predictor-loss-type` | `l2` / `kl` / `kl-post` (default: `kl-post`) |
| `BIAS_PREDICTOR_LR_MULT` | `--bias-predictor-lr-mult` | Predictor LR multiplier $\alpha$ (paper uses $10^4$; Miles ablation grid covers $10^2$–$10^4$) |
| `PREDICTIVE_STORAGE_DTYPE` | `--predictive-storage-dtype` | `fp32` / `bf16` / `fp16` / `fp8` for cached features |

### 5.2 Data sub-sampling

| Env var | CLI flag | Meaning |
|---|---|---|
| `PREDICTIVE_DOWNSAMPLE_BATCH_SIZE` | `--predictive-downsample-batch-size` | Sub-sample rate over rollout batch |
| `PREDICTIVE_DOWNSAMPLE_MAX_LEN_LIMIT` | `--predictive-downsample-max-len-limit` | Per-sample token cap |
| `PREDICTIVE_MAX_TOTAL_TOKENS` | `--predictive-max-total-tokens` | Hard cap on packed-microbatch token count |

### 5.3 Miles enhancements (all default OFF — paper-faithful when omitted)

| Env var | CLI flag | Meaning |
|---|---|---|
| `PREDICTIVE_LAYER_SCALE_SCHEDULE` | `--predictive-layer-scale-schedule` | `none` / `linear_decay` / `sqrt_decay` / `cosine_decay` — depth-aware gate $\gamma_l$ |
| `PREDICTIVE_LAYER_SCALE_MIN` | `--predictive-layer-scale-min` | Floor for $\gamma_l$ (default 1.0 = no decay) |
| `PREDICTIVE_MAX_DELTA_TO_OLD_RATIO` | `--predictive-max-delta-to-old-ratio` | Magnitude clip $r_{\max}$, e.g. `0.015` |
| `PREDICTIVE_MAX_DELTA_TO_TOPK_MARGIN_RATIO` | `--predictive-max-delta-to-topk-margin-ratio` | Initial top-k margin cap, e.g. `1.0` |
| `PREDICTIVE_MAX_DELTA_TO_TOPK_MARGIN_RATIO_FINAL` | `--predictive-max-delta-to-topk-margin-ratio-final` | Final cap after anneal, e.g. `1.5`–`2.0` |
| `PREDICTIVE_TOPK_MARGIN_RATIO_ANNEAL_START_ROLLOUT` | `--predictive-topk-margin-ratio-anneal-start-rollout` | Rollout id at which anneal starts |
| `PREDICTIVE_TOPK_MARGIN_RATIO_ANNEAL_END_ROLLOUT` | `--predictive-topk-margin-ratio-anneal-end-rollout` | Rollout id at which anneal ends |
| `PREDICTIVE_MIN_POST_TOPK_MARGIN_FOR_FLIP` | `--predictive-min-post-topk-margin-for-flip` | Flip-fallback threshold $\tau$, e.g. `0.05` |
| `PREDICTIVE_MAX_HIDDEN_SHIFT_RELATIVE_NORM` | `--predictive-max-hidden-shift-relative-norm` | Hidden-shift cutoff $\tau_h$, e.g. `0.02` |
| `PREDICTIVE_HIDDEN_SHIFT_WEIGHT_MODE` | `--predictive-hidden-shift-weight-mode` | `binary` (default) / `linear` / `quadratic` |
| `PREDICTIVE_BOUNDARY_LOSS_MAX_WEIGHT` | `--predictive-boundary-loss-max-weight` | Boundary weight cap $w_{\max}$, e.g. `4.0` |
| `PREDICTIVE_BOUNDARY_LOSS_MIN_MARGIN` | `--predictive-boundary-loss-min-margin` | Denominator floor $m_{\min}$, default `1e-4` |

### 5.4 Logged metrics (per layer, in `predictive_metrics.json`)

| Metric | Source |
|---|---|
| `predictive_loss` | The KL/L2/kl-post value (after sample reweighting) |
| `predictive_topk_accuracy` | $|\hat{\mathcal{I}} \cap \mathcal{I}_{\mathrm{current}}|/k$ |
| `predictive_stabilizer_scale` | Net scale applied by §4.2 (i)–(iii) |
| `predictive_ratio_clip_scale` | Scale from magnitude clip |
| `predictive_margin_clip_scale_{mean,min}` | Scale from top-k margin clip |
| `predictive_topk_boundary_margin_mean` | Pre-correction margin |
| `predictive_post_topk_boundary_margin_{mean,changed_mean}` | Post-correction margins |
| `predictive_flip_fallback_fraction` | Fraction of tokens reverted by flip-fallback |
| `predictive_confident_flip_fraction` | Fraction of tokens with predictor-driven flips kept |
| `predictive_stabilized_bias_to_logits_ratio` | $\overline{\|b\|}/\overline{\|p_{\mathrm{old}}\|}$ after stabilization |

These metrics are essential for the ablation grid in `scripts_mine/train/PREEXP-qwen3-30B-A3B-Base/off2/hy-sbatch-pr2-A{0..7}-*.sh` — each enhancement should produce a measurable change in one of them.

---

## 6. Comparison with the PR² paper

| Aspect | Paper PR² | Miles |
|---|---|---|
| Rollout delta handling | Use $hW_p$ as-is | Multi-stage stabilization + flip-fallback |
| Token-level KL expectation | Uniform $\mathbb{E}_t$ | $\mathbb{E}_t[w_t \cdot D_{\mathrm{KL}}] / \mathbb{E}_t[w_t]$ |
| Layer weighting | Sum over $l$ | Sum + depth schedule $\gamma_l$ |
| Skip $i=1$ predictive loss | Yes (Algorithm 3) | Yes (state-machine controlled) |
| Loss form | `kl-post` (Appendix H ablates delta-matching) | All three (`l2` / `kl` / `kl-post`); default `kl-post` |
| Empty-rank synchronization | Not specified | Synthetic-loss path |

When all Miles enhancements are off (`PREDICTIVE_LAYER_SCALE_SCHEDULE=none`, others unset), Miles reduces to the paper-faithful implementation. The `A0` ablation cell exercises this configuration.
