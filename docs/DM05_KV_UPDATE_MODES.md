# DM05: window refresh or incremental KV reuse

`DM05KVSession` provides an episode-local inference entry for both historical
KV update modes. `window_refresh` is the default, preserving the existing
window-refresh path. The switch controls which historical observations are
encoded; it does not select the ranking algorithm, change the KV budget, or
protect the first frame.

| Mode | Historical input encoded this call | Older historical KV |
| --- | --- | --- |
| `window_refresh` | Every frame in the supplied bounded window | Same IDs are replaced with refreshed KV; rows outside the window follow the bank's existing policy |
| `incremental` | Only complete frames newer than the last committed frame ID | Reused directly, with the same bank ranking/merge path |

For example, after frames 0–19 have been committed, a window containing frames
1–20 supplies 320 history tokens in refresh mode and only frame 20's 16 tokens
in incremental mode. Both modes select at most 320 history rows. Overlapping IDs
never create duplicate cached rows. Incremental mode accepts an initial window
or a backlog of multiple unseen frames, and accepts calls with no new frame.
The input window is at most the configured capacity / 16 frames.

## Inference

Use the official checkpoint's processor and normalization to prepare the actual
current observation. Keep the model weights and task fixed for a session:

```python
from opendm.model.dm05.kv_session import DM05KVSession

model.eval()
session = DM05KVSession(model, update_mode="incremental")

# prepared_step contains the actual current prefix and the history fields below.
actions = session.inference_action(prepared_step, diffusion_steps=10)

# The property can be changed without clearing committed memory.
session.update_mode = "window_refresh"
actions = session.inference_action(next_prepared_step, diffusion_steps=10)

# Reset at every environment/episode boundary. Also clear the caller's raw ring.
session.reset()
```

This entry consumes prepared tensors; it is not an XPolicyLab HTTP launcher or
a raw RoboDojo observation/action converter. Use one session per environment,
batch size one, with `model.config.ikv_history_capacity` matching the reserved
history slots (normally 320) and the checkpoint's IKV prefix configuration.
Do not pass a separate `ikv_kv_request`: the session owns the transaction.

Required step fields:

- `input_ids`, `attention_mask`, `token_type_ids`: the actual current multimodal
  prefix, with the model's reserved history placeholders. Include native current
  `pixel_values`, state/prompt inputs and `action_mask` as required by the model.
- `history_patch_ids`: a flat integer tensor, `frame_id * 16 + patch_id` for
  patch IDs 0–15. IDs are absolute within the episode, ordered by frame then
  patch, and never renumbered when the raw ring slides. Only complete frames
  are accepted. Empty history uses an empty integer tensor.
- `history_times`: seconds since episode start, repeated 16 times per frame.
  Times must agree within a frame and increase across frames. The caller must
  keep each frame ID's observation/time identity stable.
- `history_pixel_values`: processed historical head-camera pixels, one row per
  frame; or `history_features`: already encoded `[16 * frames, hidden_size]`
  features. Supply tensors on the same devices expected by the native model.
- `history_rgb_values`: historical RGB `[frames, 3, H, W]` in [0,255] for automatic
  DINO/motion metadata, unless those descriptors/motion values are supplied.

Only eligible **past** observations belong in the history fields. Current
cameras/text/state are computed normally and remain outside the historical
budget. Filtering happens before policy vision and automatic DINO encoding.
If the caller precomputes all window features/descriptors, those caller-side
computations are not saved. Motion is measured on the supplied raw sequence
before filtering; supply overlap or explicit per-patch `history_motion` when
motion across a frame boundary matters. Optional `history_contact` is per patch.

The committed bank is the cursor. A failed forward does not consume new frames;
retrying processes them again. No-new-frame calls can still rerank old candidates.
Set optional `history_query_time` to the causal current-observation time when
that is the scorer protocol. It must be finite and no earlier than any candidate.
Otherwise query time defaults to the latest candidate time, including on empty
updates; it does not reset to zero. The raw `ikv_kv_request` equivalent is
`query_time`.

## Keep the existing scoring and memory policy

The default session uses `PersistentKVBank`, with bounded resident memory and
irreversible eviction. Its `history_descriptors` are per-patch DINO descriptors;
when omitted for nonempty history they are computed from `history_rgb_values`.
The committed watermark prevents old evicted frames from being encoded again in
incremental mode. Window refresh retains the existing restriction that evicted
old IDs cannot reenter this bounded bank.

To reproduce the formula archive or use a scorer, supply a fresh bank factory:

```python
from opendm.model.dm05.formula_archive import FormulaArchiveBank
from opendm.model.dm05.qwen_archive import QwenArchiveBank

formula_session = DM05KVSession(
    model, update_mode="incremental",
    bank_factory=lambda: FormulaArchiveBank(320),
)
# Formula history_descriptors: [patches, DINO_dimension + 1], last column is
# the existing SigLIP text/patch relevance Q, not an extra DINO component.

scorer_session = DM05KVSession(
    model, update_mode="incremental",
    bank_factory=lambda: QwenArchiveBank(
        320, make_fresh_episode_client(), task_text, protect_current=False,
    ),
)
# Qwen/ParallelScorer history_descriptors: [patches, 1] frame IDs, aligned with
# history_patch_ids and the actual RGB observations registered in the client.
# Register frames through scorer_session.bank.client as required by that client.
```

Both archive banks require explicit descriptors for nonempty history. They keep
the existing recallable CPU archive; incremental mode avoids refreshing archived
rows but still lets the scorer select any archived candidate. The GPU read
budget remains 320, while the CPU archive has its separate existing byte limit.
The factory must create a fresh bank on reset, and a fresh scorer client when
that client stores episode observations. Model weights/encoder/task changes
also require a reset. Reset does not clear caller-owned raw rings or clients
outside that factory.

Neither mode pins the first frame. `protect_current` in `QwenArchiveBank` protects
the newest historical frame when enabled, not the episode's first frame. The
example explicitly disables it, matching the unrestricted scorer selection
protocol. Preserve the same bank settings when comparing update modes.

## Ordered policy training

The existing prepared-episode trainer uses the same selection helper:

```bash
cd opendm
PYTHONPATH=. python scripts/train_dm05_persistent_kv.py \
  --checkpoint /path/to/DM05-MEM-Robodojo-Sim \
  --dino /path/to/dinov2-base \
  --episodes /path/to/prepared_episodes \
  --output /path/to/output \
  --kv-update-mode incremental
```

Use `--kv-update-mode window_refresh` for the previous default. The mode is saved
in `persistent_kv_training.json`. Python callers can use
`train_episode(model, steps, optimizer, update_mode="incremental")` or
`prepare_stream_step(model, step, bank, update_mode="incremental")`.
The trainer updates the DM05 policy using its existing bounded bank; it does not
train an external scorer. Historical KV stays detached across steps, and a bank
is reset at each episode/weight update.

## Semantics and validation limits

In both modes, the native VLM first computes the current prefix with the supplied
fresh history rows; the bank is then merged before action-expert denoising.
Cached old history does not become an extra VLM-prefill attention input. An old
row retains the contextual representation from its last encoding. Consequently,
window refresh and incremental reuse are different recurrent inference policies,
not numerically equivalent ways to evaluate one dense history prefix.

Reserved prefix placeholders and the current observation still go through the
native prefix path. Fewer historical image encodings do not establish an equal
reduction in dense Transformer work or end-to-end latency. Compare real latency
and task success using the same checkpoints, episodes, scorer/budget, time
protocol and diffusion settings; start a fresh session for each mode.

`opendm/tests/test_dm05_kv_modes.py` exercises both modes with tiny native DM05,
all three banks, exact retained K/V reuse, duplicate replacement, archive recall,
empty updates, retry/reset and ordered training. The existing RoPE/current-key,
sliding-cache and baseline-parity tests remain applicable. These are tiny-model/interface
regressions; no new real-checkpoint performance or task-success result is claimed.


Validation on 2026-11-12: **61 passed, 1 skipped**, using Python 3.12,
PyTorch 2.11 and Transformers 5.3.0. The skip requires the optional external
XPolicyLab checkout. Reproduce from `opendm/`:

```bash
PYTHONPATH=.:.. OMP_NUM_THREADS=4 python -m pytest -q \
  tests/test_dm05_kv_modes.py tests/test_dm05_persistent_kv.py \
  tests/test_qwen_archive.py tests/test_parallel_scorer.py \
  tests/test_scorer_rl_reward.py tests/test_rgb_max.py \
  tests/test_qwen_baseline_parity.py tests/test_dm05_online_history.py \
  tests/test_dm05_ikv_rgb.py tests/test_dm05_adarms_cond.py
```
