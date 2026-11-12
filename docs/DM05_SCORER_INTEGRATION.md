# DM05 IKV and local scorer integration

This reconciliation starts from main `ae9b8a1bc8daa7d078bea9e7af82d76216025eea`
(2026-10-17), the published archive/Qwen interface, and the October 27 deployed
parallel-scorer bridge. Source digests for imported formula modules are in
`DM05_RUNTIME_RECONCILIATION.json`. A deployment's later usage date is not proof
that every file is newer than main.

## Retained and recovered paths

- `opendm/opendm/model/dm05/persistent_kv.py`: retain main's RoPE identity handling,
  unchanged-current/text key protection, and inference sliding-window restoration.
  Do not replace it wholesale with the older deployed copy.
- `qwen_archive.py`: the deployed file equals main. `QwenArchiveBank` keeps detached
  historical KV on CPU and supports recall; `PersistentKVBank` is the distinct
  bounded, irreversible-eviction policy. Both use transactional plans/commits.
- `formula_archive.py`, `formula_content.py`: recover the optional deployed
  formula archive, without selecting it by default. Its formula is exactly
  `wT*exp(-age/sT) + wC*(1-exp(-contact/sC)) + wR*class_recency + wQ*relevance`.
  Q here is the supplied SigLIP text/patch relevance, not measured query usage.
  This is a recorded experimental formula, not a claim to implement the latest
  paper's formula or the later V92 diagnostic.
- `parallel_scorer.py`: extract the local model-facing bridge from the October 27
  HTTP deployment. No host, port, checkpoint path, credentials or diagnostic
  prompt is built into it. RuntimeScorer scores are **logits**, not 0–100 labels.
  Preserve the deployed mapping: mean each spatial 2x2 block of the 8x8 scorer
  logits to DM05's 4x4 native history grid. Mean(sigmoid(logits)) is different.

The unreachable former experiment host was not independently audited. Its V92
non-history VLM reread/primacy experiments are not silently enabled or represented
as part of this release. The deployed oracle prompts, pinned demonstration IDs,
manual success adjustments and machine-specific launch wrappers are not imported.

## Local inference bridge

For episode-local inference with either `window_refresh` (default) or
`incremental` KV reuse, use [DM05KVSession](DM05_KV_UPDATE_MODES.md). The update
mode is independent of the bank/scorer and does not pin the first frame.

Install/pin the separate `genalyu/ikv` scorer source on PYTHONPATH, and load the
matching frozen encoder revision and trained scorer checkpoint:

```python
from training.parallel_scorer.encoders import QwenEncoders
from training.parallel_scorer.runtime import RuntimeScorer
from opendm.model.dm05.parallel_scorer import ParallelScorerClient
from opendm.model.dm05.qwen_archive import QwenArchiveBank

scorer = RuntimeScorer(scorer_checkpoint, device="cuda:0")
encoder = QwenEncoders(encoder_checkpoint, scorer.info["encoder_revision"],
                       device="cuda:0", image_batch=1)
client = ParallelScorerClient(scorer, encoder, task_text, sensor_id=0)
bank = QwenArchiveBank(320, client, task_text, protect_current=False)
# Actual measured head-camera images, stable frame IDs, times in seconds:
client.append(frame_ids, rgb_images, observation_times_seconds)
# Supply this bank through the existing ikv_kv_request deployment/forward path.
```

The caller creates a new client and bank for every episode and must maintain a
consistent frame-ID/time contract. `register()` also accepts already encoded
`[64,D]` features. Only context-independent image features are cached. Every
score call recomputes contextual scores; future frames are excluded and future
candidates fail. Candidate time/grid/modality and checkpoint/encoder revisions
are checked. All registered causal head frames provide context; register the
current observed head frame explicitly when using that protocol.

This bridge selects only head-camera historical KV. It does not replace current
three-camera observations or create tactile tokens. It is not by itself an
XPolicyLab launcher or multi-camera scorer-training data adapter. In particular,
QwenArchiveBank defaults its query time to the latest candidate time. Pass
`history_query_time` through `DM05KVSession` (or `query_time` in a raw
`ikv_kv_request`) for a separate causal current-observation query time, rather
than relabeling a historical frame as current.
The October 27 deployment used only historical head features, a known mismatch
with the scorer's multi-camera training input; preserve that distinction when
comparing old results with a new current-observation-conditioned protocol.

## Flow-matching boundary for scorer RL

`DM05ForConditionalGeneration.forward()` already computes the native action FM
loss after `_compute_prefix_cache()` and `splice_dm05_kv()`. It samples
`epsilon ~ Normal(0,I)` and `t = 0.001 + 0.999*Beta(1.5,1)`, constructs
`x_t=t*epsilon+(1-t)*a`, predicts velocity `v`, and reduces
`sum(mask*(v-(epsilon-a))**2)/sum(mask)` per sample. Use the released DM05-MEM
Robodojo checkpoint's native 14 active joint channels, normalization and action
mask; padded channels must not become supervised targets. This is an **action**
FM reward, not N0's video/tactile/action reward.

`streaming_training.train_episode()` trains the DM05 policy. It is **not GRPO
for a frozen DM05 reward and a trainable scorer**. A scorer-RL driver must still:

1. Freeze DM05 and fork the same causal archive state for every candidate.
2. Sample four distinct candidate paths with one group budget and one shared
   action/noise/time target. Current observations and text stay fixed.
3. Evaluate each path without candidate-to-candidate archive commits leaking.
4. Recompute scorer path probabilities with gradients, apply group-relative
   advantages/sensitivity weighting, and update only scorer parameters.
5. Verify all-kept parity, deterministic common-noise comparisons, real
   counterfactual loss sensitivity, causal data alignment and 16-rank sync.

The scorer GRPO implementation is now in `robodojo_training/scorer_rl/`; see its
README for matched FM draws, isolated archive rollout and distributed checks.
CPU regressions pass; real-checkpoint/cloud validation is recorded per run.
Default native FM sampling and policy weights remain unchanged.

## Validation scope

CPU regression tests cover 8x8→4x4 mapping, archive payload selection, causality,
immutable IDs, encoder revision validation, and formula plan/commit rollback.
Existing archive/RGB-Qwen tests remain part of the check. These tests establish
interface behavior, not real-checkpoint parity, GPU performance, FM reward
usefulness, or robot task success. Historical oracle/pinned-memory diagnostic
results must never be reported as normal scorer-policy evaluation results.

Reconciliation validation (2026-11-12): **26 passed, 1 skipped** with Python 3.12,
PyTorch 2.11 and the repository-pinned Transformers 5.3.0. Run from opendm/:

    PYTHONPATH=. OMP_NUM_THREADS=4 python -m pytest -q tests/test_parallel_scorer.py tests/test_qwen_archive.py tests/test_rgb_max.py tests/test_dm05_persistent_kv.py tests/test_qwen_baseline_parity.py tests/test_dm05_online_history.py

The skip is the optional external XPolicyLab deployment adapter: set
XPOLICYLAB_PARENT to the parent of its checkout to exercise it. The passing set
includes tiny-DM05 training forward/backward and native-KV baseline comparisons.
