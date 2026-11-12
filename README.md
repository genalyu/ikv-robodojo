# RoboDojo IKV: DM05, π0.5 and OpenWAM

DM05 supports real-observation historical KV selection, transactional persistent
banks, a recallable CPU archive, and optional external/local scorer interfaces.
The current integration guide is [DM05 scorer integration](docs/DM05_SCORER_INTEGRATION.md).

## Version and behavior

The November 12, 2026 reconciliation preserves main's existing RoPE/current-key
and sliding-window safeguards, recovers the deployed formula archive, and makes
the October 27 local parallel-scorer interface portable. It does not copy an
older deployment over main or enable diagnostic oracle/primacy interventions.
[Source reconciliation](docs/DM05_RUNTIME_RECONCILIATION.json) records the boundary.

- [Switchable DM05 window refresh / incremental KV reuse](docs/DM05_KV_UPDATE_MODES.md)
- [Persistent KV, detached recurrent training and validation scope](opendm/docs/persistent_kv.md)
- [RGB Qwen-Max archive scoring](opendm/docs/rgb_max.md)
- [Local parallel scorer and native FM loss](docs/DM05_SCORER_INTEGRATION.md)
- [Original RGB motion/DINO experiments](docs/LEGACY_RGB_MOTION.md) — historical
  design for DM05, π0.5 and OpenWAM; not the default current scoring recipe.

DM05's `forward()` supports native action flow-matching loss after historical KV
selection. The existing streaming trainer updates DM05 itself; it is not a
scorer-GRPO trainer. The separate scorer RL entry uses frozen DM05 reward,
matched candidate noise/time and causal joint-action data; distributed GPU
verification is recorded per run. No success-rate or cloud-training claim follows from CPU tests.

Model weights, credentials, machine-specific launchers and evaluation artifacts
are not distributed in this repository. Use the official model's matching
normalization, observation/action and image-processing contracts.

Scorer GRPO with frozen DM05 action FM: [workflow and contracts](robodojo_training/scorer_rl/README.md).
