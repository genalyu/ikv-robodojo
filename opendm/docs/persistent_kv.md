# Persistent DM05 / OpenWAM IKV — 2026-10-04

All changes are on 2Kuanfan4090, rooted at `/home/ubuntu/genalyu/ikv-robodojo`.
This implementation has not been evaluated for task success or fine-tuned yet.
The stopped 40% result belongs to the previous feature-memory implementation.

## DM05 data flow

1. The RoboDojo adapter retains a bounded 21-frame raw ring: 20 eligible historical
   frames plus the current sampled frame. Sampling is 1 Hz, observations/control
   are configured at 25 Hz, and a prediction executes 25 of its 50 generated actions.
   Thus simulated replan interval is 1 second; wall-clock latency is separate.
2. Each prediction encodes the latest eligible <=20 frames into <=320 visual rows.
   DINO descriptors are cached only for this bounded window; VLM features refresh.
3. VLM prefill encodes current cameras, text/state and the refreshed window.
   All 34 prefix layers use full KV storage even when their attention is sliding.
   This changes storage, not the layer's original sliding attention mask.
4. Extract refreshed history K/V at each layer. Use absolute episode patch IDs
   (`sample_index * 16 + patch_index`) to replace duplicate survivors. An ID below
   the watermark which is no longer resident is ineligible to reenter the bank.
5. Merge those refreshed eligible rows with resident rows outside the window,
   select up to 320 IDs with one shared selection across all layers, and install
   them into the reserved history region. Current camera/text/state positions
   remain separate from this 320-history-position budget.
6. Store historical K before RoPE in float32 and historical V in model dtype. Undo current prefix RoPE and
   apply the appropriate local/global Gemma RoPE at destination positions. This
   prevents stale physical-slot rotations when selection changes.
7. Action expert denoising reads this merged prefix. Action suffix K/V is ephemeral
   at each noise level and is not persisted between observations.
8. Commit detached layer K/V only when inference succeeds with finite actions.
   Reset clears both the raw window and the bank for the relevant episode/env.

No evicted KV archive or tombstone table exists. Recent evicted patches may still
be contextual inputs during the 20-frame refresh, but cannot become persistent
cache rows again. Older surviving KV contains the contextual representation from
its last refresh. This is a recurrent memory model, not an exact recomputation of
an infinitely long baseline prefix. Fine-tuning must use the same recurrent path.

## OpenWAM data flow

- Input only the current composite observation to the causal VAE. Preserve the
  original future-video horizon and joint video/action generation; no IDM.
- Derive N from the actual latent patch grid, not a hardcoded 512 or 128. The
  production 384x320 input was verified to produce a 12x10 grid: N=120.
- Each layer contributes N current clean-frame K/V rows. Merge them with the N
  resident rows, select N with shared indices, and let action/future-video queries
  read only these N memory rows plus their ordinary future/action suffix K/V.
  There is no extra full current frame appended to the expert's memory budget.
- The current clean frame itself attends densely to its own N rows. Its t=0,
  first-frame-causal branch is independent of action/future diffusion noise.
  Captured clean KV is therefore stable across noise levels for the same context.
- All observed keys use temporal RoPE anchor zero, retaining spatial RoPE; age is
  selector metadata. Distinct history ages are not separately encoded in RoPE.
- A generation commits exactly once after all denoising and output conversion
  succeed. Motion is a read gate only, applied after retention selection.
- Each environment has a separate bank. Async execution and CFG>1 are rejected;
  prediction caching and compiled joint loops are disabled on this path. Outer
  activation checkpointing is disabled for capture, avoiding replay mutations.

## Importance score and the all-new-token failure

For position i, with birth time t_i, current time t, latest matched-class time l_i,
contact duration c_i, and r_i matching **other** live slots (cosine >=0.9):

    T_i = exp(-max(t-t_i,0)/8)
    C_i = 1-exp(-c_i/8)
    R_i = 1[valid descriptor] * exp(-max(l_i-t_i,0)/8) / (1+r_i)
    S_i = T_i + C_i + R_i

Stable descending top-K is followed by chronological/spatial ordering. Ties keep
older candidate order. Matching statistics refer only to the live bank; no
unbounded semantic archive is retained. Empty contact input means C=0 (the current
deployment does not supply contact sensors). DM05 time units are seconds at the
configured 1 Hz sampling. OpenWAM currently uses generation count as time units;
its eight-update scale is not necessarily eight physical seconds.

The previous R omitted the repetition divisor. With N fresh tokens and C=0,
every fresh token scored 2 and every older token scored less than 2: an N-slot
OpenWAM cache would always discard all history. The repetition correction gives
older unique events a chance to survive, while staying strictly score-based.
It does not guarantee old-event retention forever or higher task success. Tune
threshold/decay on validation episodes and include a no-correction ablation.

## DM05 training interface

`scripts/train_dm05_online_ikv.sh` now invokes the persistent-KV trainer, not the
legacy random-sample feature-memory experiment:

```bash
cd /home/ubuntu/genalyu/ikv-robodojo/opendm
bash scripts/train_dm05_online_ikv.sh   --episodes /absolute/path/to/prepared_episode_pt_files   --output /absolute/path/to/training_output
```

Each .pt file contains a list of tensor-only step dictionaries in chronological
order. Use `torch.save(steps, path)`. Required fields for each step:

- `input_ids`, `attention_mask`, `token_type_ids`: current observation's processed
  prefix with 320 reserved history placeholders, batch size one.
- `pixel_values`: the actual current camera tensors, when present in the prefix.
- `history_pixel_values`, `history_rgb_values`: <=20 eligible historical frames,
  aligned with the same episode-anchored sampling as deployment. RGB is CHW in
  [0,255]; processed pixels follow the checkpoint image processor.
- `history_patch_ids`: flattened absolute sample_index*16+patch_index, length
  16 * historical_frame_count; `history_times`: same length, seconds.
- `action` and `action_mask`: normalized supervised actions in checkpoint model
  dimensions (not merely the robot's 14 unpadded dimensions).
- Optional `history_descriptors` avoids frozen DINO recomputation; optional
  `history_motion` and `history_contact` provide per-position scoring metadata.

`prepare_stream_step` encodes differentiable vision features, packs the prefix,
and passes the same explicit KV transaction as deployment. `train_episode`
backpropagates each step immediately, detaches previous KV, and updates weights
only at the end of the episode. A new bank starts for every episode/weight update.
Default CLI freezes the VLM and trains the remaining policy; `--train-vlm` enables
VLM training at a substantially higher memory cost. This is truncated recurrent
training, not backpropagation through the entire historical cache.

The CLI expects prepared chronological data; it is not a raw RoboDojo dataset
converter. Never fabricate past state/wrist views from the final observation.
The old feature-replay helpers remain available for historical ablations, but
are not the new persistent training launcher. No training run was launched.

## OpenWAM training interface

`OpenWAM/openwam/model/streaming_training.py::train_episode` takes chronological
`architecture.prepare_inputs` outputs plus `ikv_current_image` and
`ikv_dino_features`. Each step must have one `first_frame_latents` frame, actual
current proprio/prompt, future video targets and action targets. It invokes the
ordinary joint flow-matching loss with the exact persistent session used at
inference, performs step-wise backward, commits detached KV, and updates weights
only at the episode end. Call `init_training_schedulers` as in the usual trainer.
No IDM loss is introduced. Dataset conversion/distributed integration is not
provided by this small ordered-episode interface.

## Main implementation files

- DM05: `opendm/opendm/model/dm05/persistent_kv.py`, `dm05_arch.py`,
  `streaming_training.py`; deployment adapter:
  `/home/ubuntu/genalyu/RoboDojo/XPolicyLab/policy/OpenDM/model.py`.
- OpenWAM: `OpenWAM/openwam/model/persistent_kv.py`, `streaming_training.py`,
  `architectures/dual_system/mot_driver.py`, `video_backbone/wan_backbone.py`,
  `architectures/base.py`, `deploy/engine.py`, `deploy/policy.py`.
- Raw comparison: 3Kuanfan4090:
  `/home/ubuntu/genalyu/eval_dm05_online_ikv/paired_comparison_stopped.json`.

## Verification completed

- 23 DM05 tests: bounded recurrence, irreversible eviction, duplicate refresh,
  rollback, real tiny Gemma with sliding/full layers, forward/backward/inference,
  and ordered training optimizer updates; legacy feature-selector tests retained.
- 10 OpenWAM tests: exact first-frame attention parity, read budget, motion gating,
  real tiny Wan+action joint transformer, invariance across denoising times/noise,
  and ordered joint training backward/optimizer updates.
- Real checkpoint smoke checks on 2Kuanfan4090, two predictions each, **2 denoising
  steps** on synthetic images (not task evaluation, not production latency tests):
  DM05: 34 layers, 320 slots/layer, actions 50x14, all finite.
  OpenWAM: 30 layers, 120 slots/layer, actions 32x20, all finite.
  Repetitive synthetic OpenWAM images selected no old positions on the second
  call; the distinct-old-event retention test separately verifies historical reads.
- Smoke outputs: `/home/ubuntu/genalyu/logs/dm05_persistent_kv_smoke.json` and
  `/home/ubuntu/genalyu/logs/openwam_persistent_kv_smoke.json`.
- Evaluation is stopped and its heartbeat paused. No new campaign or full training
  run was launched. Fine-tuning/validation is still needed to establish quality.
