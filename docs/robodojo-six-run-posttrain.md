# RoboDojo six-run post-training on neosim A100

Server workspace: `/mnt/cfs/9wt59p/genalyu/ikv-robodojo`.
Shared data/weights/runs: `/mnt/cfs/9wt59p/genalyu/robodojo-posttrain`.
Queue: `scripts/run_robodojo_queue.py`, waiting for neosim completion, full data audit, final weights, and four idle GPUs.
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

The queue waits for neosim checkpoint step 2000 and four idle GPUs, then checks readiness before each run. It audits the complete 35-task DM05 JSONL/video data before the first DM05 job; it audits the full HDF5 and LeRobot sources before OpenWAM and PI05. This lets DM05 start while the larger sources continue downloading. Every run requires a clean main checkout and matching assets. Four-GPU optimizer-step validation remains pending.

## Remaining checks before readiness

1. Complete and verify all downloads and isolated environments. Confirm the complete 35-task HDF5 and DM05 JSONL/video coverage, full LeRobot v3 metadata, and model checkpoint integrity. Build `data_manifest.json` only after this audit.
2. Validate the integrated ordered episode sampling and detached per-episode K/V state in the official four-GPU trainers. The recurrent code resets memory at episode boundaries and after optimizer updates and uses the bounded 768-token PI05 inference path. Ensure no motion gating is enabled.
3. Verify shared normalization per baseline/IKV pair. Run CPU unit tests and one real four-GPU smoke per method after neosim releases the GPUs. Confirm at least one optimizer update, finite loss, resume behavior, and checkpoint format.
4. Publish the A100 server working tree directly to `genalyu/ikv-robodojo` main. The queue checks clean main and reruns the data audit automatically.

## GitHub push route

The GitHub SSH identity is configured on the jump host `a100-baiduyun-yinx-copy1`, not inside the training container. The jump host accesses the same shared repository path. Its key `/root/.ssh/id_ed25519_github_genalyu` has fingerprint `SHA256:MRwYZ46nDhqasXJPe9u5l/wG/kssfolQTuQluzb7YHY` and authenticates to GitHub as `genalyu`. Push from that host with `GIT_SSH_COMMAND="ssh -o BatchMode=yes -o IdentitiesOnly=yes -i /root/.ssh/id_ed25519_github_genalyu"`; origin's push URL is `git@github.com:genalyu/ikv-robodojo.git`. Keep the private key on the jump host. The deployment branch is `main`, as requested by the user.

## Domestic mirror installation (2026-10-06)

`/mnt/cfs/9wt59p/genalyu/ikv-robodojo/scripts/install_robodojo_envs.py` prefers Tsinghua PyPI and SJTU CUDA wheels. OpenWAM follows its committed Docker requirements lock, including Python 3.12 (NumPy 2.5.3 requires >=3.12), torch 2.7.1+cu128, and a separate DeepSpeed build after torch is installed. OpenPI exports its official uv lock without modifying it; its CUDA torch build is 2.7.1+cu126. OpenDM retains its pyproject pins with CUDA torch 2.11.0+cu128. Downloads use existing proxy only for upstream sources unavailable domestically.

DM05 and OpenWAM now have persistent IKV integration in the official trainers; neither is validated end-to-end yet. Their distributed microbatch sampler keeps chronological chunks per rank and resets memory on episode/optimizer boundaries. DM05 IKV uses batch 1 x accumulation 8 x four GPUs (32 targets/update); OpenWAM uses 1 x 32 x four GPUs (128 targets/update). PI05 has a four-frame recurrent loss path in the official JAX trainer, with a v3 LeRobot adapter. The queue rechecks readiness and records the data audit automatically; full four-GPU training validation remains pending.

## PI05 LeRobot v3 and pure IKV training

The public RoboDojo LeRobot dataset is v3.0 (3,500 episodes, 1,856,102 frames, 35 tasks). OpenPI's pinned LeRobot commit reads v2 episode files and cannot open this dataset. The OpenPI environment therefore installs LeRobot 0.4.4 from the Aliyun mirror and imports its `lerobot.datasets` reader. It also adapts the v3 task table (`prompt -> task_index`) to OpenPI's `task_index -> prompt` transform. Other OpenPI locked packages are retained where possible. LeRobot's optional Rerun viewer requires NumPy >=2, while OpenPI requires NumPy <2; the environment keeps NumPy 1.26.4 and Rerun 0.23.1, which is not involved in training. All other environment dependencies pass `uv pip check` after this intentional viewer mismatch.

The PI05 baseline retains the official random frame batch 256 and 60,000 updates. Pure IKV uses the same base checkpoint, camera views, actions, normalization assets, optimizer, and 60,000 updates. Each IKV update processes 64 episode-safe four-frame streams (up to 256 valid targets), with the frozen SigLIP/Pali visual tokens supplying the 768-row recurrent KV index. Its action loss is the native flow-matching objective with the same prefill, candidate selection, text suffix, and action suffix path as recurrent inference. The prior K/V is detached between frames and reset at every update. Episode tails are padded to four frames but masked out of the loss. No motion-only or motion+IKV mode is used.

A server-side v3 fixture using episode 0 confirmed the reader returns three 480x640 views, 14D state, and 50x14 action targets. The baseline and IKV OpenPI transforms both produced three 224x224 views, 50x32 padded targets, and 200 prompt tokens. The episode-safe sequence index was checked across an episode boundary. Full dataset and four-GPU training validation remain gates before the queue can launch.

The PI05 normalization asset is the released RoboDojo PI05 checkpoint's `assets/arx_x5_sim/norm_stats.json`, exactly as referenced by the official RoboDojo OpenPI config. It was streamed from the 2K server into `/mnt/cfs/9wt59p/genalyu/robodojo-posttrain/norm/pi05/official-release/arx_x5_sim/norm_stats.json`; SHA-256 is `ad7dea3e3d2bcdb348945fe03422ab1adccd03baf67318b1a1d153dfe8694db5`. The launcher copies the same bytes into baseline and IKV asset directories.

DM05's official RGB vision config requests FlashAttention 2, but the deployed torch 2.11+cu128 environment has no official prebuilt `flash_attn` wheel. Both baseline and IKV launchers therefore select the model's supported SDPA vision attention backend. This changes the attention implementation for both runs together; model weights, attention visibility, source data, optimizer, and 30,000-update schedule remain identical. Revisit performance only after the official wheel becomes available or a verified build succeeds.

## DM05 release gap and conversion

The Dexmal RoboDojo simulation archive contains only 34 task directories (3,400 episodes) and omits `dlc`. The 35-task RoboDojo source and PI05 LeRobot v3 metadata both include it. To keep all three methods on the selected 35-task scope, `scripts/convert_robodojo_dm05.py` derives `dlc` JSONL and H.264 video from the original RoboDojo HDF5 episodes. Its state/action joint order, prompt, camera references, and terminal extra frame were checked against a released `arrange_largest_number` episode: all 579 JSONL records matched exactly, and the generated videos decoded as 579 frames at 25 FPS. The encoding differs from the prebuilt archive; inputs are the same source frames. `scripts/prepare_dm05_dlc.py` waits for 100 `dlc` HDF5 episodes, converts them, then builds OpenDM's 3,500-episode index cache. The server-side `scripts/follow_dm05_dlc.py` converts finished HDF5 episodes during the download, and the queue calls the preparation script before its DM05-specific audit. The full-source audit for later jobs also checks every DM05 video's existence and nonempty size as well as episode counts.
