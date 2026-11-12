# Frozen DM05 scorer GRPO

Run with both this repository (including `opendm/`) and the pinned `genalyu/ikv`
source on PYTHONPATH:

    torchrun --nnodes=2 --nproc_per_node=8 ... -m robodojo_training.scorer_rl.train --config run.json

The scorer starts at 10M SFT step 16000. DM05-MEM-Robodojo-Sim remains frozen.
Qwen3.5 image/text encoding is frozen and cached per source episode SHA256 and
encoder revision. Only the small scorer receives gradients.

Each query uses current three-camera RGB and state, historical head observations
at 1 Hz, and the following native 50-action chunk at 25 Hz, 14 absolute joint
channels. The matching released quantile statistics normalize state and action;
only 14 channels contribute to loss after padding to 32. Current observations
are context, never selectable history. RGB is decoded with JPEG color markers.
No tactile or predicted-video reward is invented for this model.

History is replayed chronologically at 1 Hz. The latest 20 head frames refresh
native prefix KV; an episode-local CPU archive retains older observed keys.
The fixed physical prefix has 320 slots. Patch IDs are `sample_index*16+patch`;
raw control-frame indices are `sample_index*25`, and times are seconds. A candidate can choose older archived
keys and newer refreshed keys. All four candidates share one prefix, history
snapshot, selection budget, Gaussian action noise and Beta(1.5,1) time draw.
Their cache copies are isolated. Only after all evaluations does a deterministic
latest-320 roll-in commit the observation into the full archive. The scorer never
sees future observations, although independent image features may be precomputed.

Native scorer 8x8 head logits are averaged in spatial 2x2 blocks to native 4x4 KV
logits. Four ordered Plackett-Luce samples without replacement are evaluated:

    x_t = t * epsilon + (1-t) * a
    L_FM = sum(mask * (v(x_t,t,KV) - (epsilon-a))^2) / sum(mask)
    reward = -L_FM
    A_g = (reward_g - group_mean) / max(group_std, 1e-4)

The original IKV `group_loss` computes the clipped ratio of exact ordered-path
probabilities. It does not call an unordered-subset probability a likelihood.
A group budget is one of 64/160/256, capped strictly below candidate count; early
queries use half the candidate count if below the smallest configured budget.
Duplicates are logged, not silently resampled into an unmodeled distribution.

Every update accumulates 4 queries per rank, 64 queries and 256 FM evaluations
across 16 ranks. Per-query weight is the reward standard deviation divided by a
shared previous-update EMA, clipped to [0.25,4], then normalized globally.
The centered-logit SFT anchor has coefficient 0.01. It is not a KL divergence.
All ranks synchronize scorer gradients and verify exact parameter hashes at
update 1 and each save. AdamW learning rate is 1e-5, gradient norm cap 1.
Default launch runs 1000 optimizer updates and saves at 1, every 10, and the end.

Preflight compares the public native forward, shared-prefix FM and all-kept KV,
checks repeated candidate equality and archive isolation, and requires nonzero
counterfactual sensitivity across ranks. Unit tests use a tiny CPU DM05; passing
them is not evidence of real-checkpoint quality or robot success. Durable
per-update JSON files support a separate CPU-side live W&B uploader. A stable run
requires real-checkpoint preflight on every rank, finite nonzero gradients,
matching rank hashes, sustained updates, readable checkpoints and live W&B.

## Input manifest and run config

`episodes` points to a JSON list. Each row includes id, episode_id, task, prompt,
split (`train`/`validation`), fps=25, path, source_revision, source_path and sha256.
The path is an NPZ containing `rgb[N,3,H,W,3]` uint8 RGB, `frames[N]` (0,25,50,...),
`states[T,14]`, `actions[T,14]`. Keep only sampled queries with a full 50-action
target. Preserve the original SFT episode split; validation episodes are excluded
from training, not used as additional reward examples. This driver does not yet
report held-out robot success or automatically select the best checkpoint.

Config keys: checkpoint, encoder, dm05_checkpoint, dm05_revision, episodes,
feature_dir, output, seed, max_updates, save_every, budgets,
gradient_accumulation_frames, learning_rate, reference_logit_weight.
An output directory identifies a fresh attempt; the launcher must reject reuse
of an attempt with existing updates. Optimizer artifacts are saved, but exact
mid-episode resume is not implemented. Preserve failed attempts separately.

CPU regressions (2026-11-12): 28 passed, 2 skipped (CUDA parity and external
XPolicyLab). The native cache construction fix preserves configured layer counts
for VLADynamicCache, preventing an all-kept FM mask mismatch. Inference
DynamicCache full-prefix retention is unchanged.
