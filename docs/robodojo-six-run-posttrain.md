# RoboDojo six-run post-training on neosim A100

Server workspace: `/mnt/cfs/9wt59p/genalyu/ikv-robodojo`.
Shared data/weights/runs: `/mnt/cfs/9wt59p/genalyu/robodojo-posttrain`.
Queue: `scripts/run_robodojo_queue.py`, currently deliberately gated by `preflight_ready.json`.
Four A100 80 GB GPUs, `CUDA_VISIBLE_DEVICES=0,1,2,3`. Do not disturb the current neosim pure IKV run.

## Scope and provenance

The user selected **all 35 publicly available RoboDojo simulation training tasks**, not only the six memory evaluation tasks. The authoritative task list is in the server's `expected_tasks.json`, fetched from the ModelScope `RoboDojo-Benchmark/RoboDojo` tree under `data/RoboDojo`. Use identical task sets and corresponding demonstration episodes for each method's baseline and pure IKV pair. Methods retain their native action spaces and observation preprocessing.

| Method | Starting weights | Native training data | Official schedule | Four-GPU effective batch target | IKV budget |
|---|---|---|---|---|---|
| DM05-MEM | Dexmal/DM05-MEM general base | Dexmal/robodojo-sim JSONL/video; 25 FPS, 20 head history frames at 1 FPS, three cameras, absolute 14-D joints, action chunk 25 | 30,000 optimizer steps | 4 per GPU × accumulation 2 = 32 | 320 history VLM K/V positions |
| OpenWAM-α | OpenWAM-Alpha-Pretrain-Foundation-Model | RoboDojo HDF5; EEF20 mapped into 80-D, 33 state/action frames, video stride 4, three-camera composite 384×320 | 5 epochs | 4 per GPU × accumulation 8 = 128 | 120 clean-frame Wan K/V positions |
| PI0.5 | gs://openpi-assets/checkpoints/pi05_base/params | RoboDojo LeRobot v3 video; 3 cameras, absolute joint actions, official ALOHA transforms | 60,000 optimizer steps | global batch 256, FSDP devices 2 on four GPUs | 768 visual PaliGemma K/V positions |

Sources within this server repository: `opendm/docs/en/dm05_robodojo.md`, `OpenWAM/assets/openwam_usage_docs/openwam-alpha-finetuning.md`, `OpenWAM/configs/dataloader/robodojo.yaml`, and `openpi/src/openpi/training/config.py`. The official DM05 example documents a single `cover_blocks` entry; this work extends its dataset registration to all public task directories. Four-GPU gradient accumulation changes device layout but preserves the examples' effective batch and training length.

## Six runs and safety gate

The fixed queue order is DM05 baseline, DM05 IKV, OpenWAM baseline, OpenWAM IKV, PI05 baseline, PI05 IKV. Never train motion-only IKV or motion+IKV. Baseline and IKV for each method must share the same base weights, source episodes, action space, normalization statistics, optimizer settings, and official schedule. The intended variable is persistent RGB K/V memory with its fixed method-specific budget.

**Do not create `preflight_ready.json` yet.** The baseline launchers exist, but all three IKV training paths still need integration into their official distributed trainers. Existing streaming helper functions are not sufficient: they are not wired to official epoch/step semantics; PI05's current `compute_loss` does not use the recurrent inference bank. `scripts/launch_*.sh` deliberately exit for IKV until validated. The queue also requires neosim checkpoint step 2000 and four idle GPUs.

## Remaining checks before readiness

1. Complete and verify all downloads and isolated environments. Confirm the complete 35-task HDF5 and DM05 JSONL/video coverage, full LeRobot v3 metadata, and model checkpoint integrity. Build `data_manifest.json` only after this audit.
2. Wire ordered episode sampling and detached per-episode K/V state into the official four-GPU trainers. Reset the bank at episode boundaries and after optimizer updates; no future frame may be used in a current prediction. PI05 must train through the same bounded 768-token K/V path used at inference, not merely set an inference flag. Ensure no motion gating is enabled.
3. Verify shared normalization per baseline/IKV pair. Run CPU unit tests and one real four-GPU smoke per method after neosim releases the GPUs. Confirm at least one optimizer update, finite loss, resume behavior, and checkpoint format.
4. Commit and publish the A100 server working tree to `genalyu/ikv-robodojo`; then write `preflight_ready.json` containing the exact committed HEAD and `all_six_runs_validated=true` so the queue may start.

## GitHub push route

The GitHub SSH identity is configured on the jump host `a100-baiduyun-yinx-copy1`, not inside the training container. The jump host accesses the same shared repository path. Its key `/root/.ssh/id_ed25519_github_genalyu` has fingerprint `SHA256:MRwYZ46nDhqasXJPe9u5l/wG/kssfolQTuQluzb7YHY` and authenticates to GitHub as `genalyu`. Push from that host with `GIT_SSH_COMMAND="ssh -o BatchMode=yes -o IdentitiesOnly=yes -i /root/.ssh/id_ed25519_github_genalyu"`; origin's push URL is `git@github.com:genalyu/ikv-robodojo.git`. Keep the private key on the jump host. The deployment branch is `robodojo-six-posttrain-20261006`.
