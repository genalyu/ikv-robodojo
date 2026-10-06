#!/usr/bin/env bash
set -euo pipefail
mode="${1:?baseline or ikv}"
source /mnt/cfs/9wt59p/genalyu/ikv-robodojo/scripts/robodojo_env.sh
if [[ "$mode" != baseline && "$mode" != ikv ]]; then exit 2; fi
batch=4
accum=8
extra=()
if [[ "$mode" == ikv ]]; then
  batch=1
  accum=32
  extra+=(+training.persistent_ikv_training=true "+training.ikv_dino_model_path=$ROBODOJO_ROOT/models/dinov2-base")
fi
export PATH="$ROBODOJO_ROOT/envs/openwam/bin:$PATH"
export NPROC_PER_NODE=4
export HYDRA_FULL_ERROR=1
data_root="$ROBODOJO_ROOT/raw/data/RoboDojo"
model_root="$ROBODOJO_ROOT/models/openwam-alpha-foundation"
run_dir="$ROBODOJO_ROOT/runs/openwam_$mode"
test -d "$data_root"
test -f "$ROBODOJO_ROOT/data_manifest.json"
test -d "$model_root"
mkdir -p "$run_dir"
cd "$ROBODOJO_REPO/OpenWAM"
bash scripts/train.sh \
  dataloader=robodojo \
  "dataloader.dataset_dir=$data_root" \
  "training.finetune_ckpt_path=$model_root" \
  training.num_epochs=5 \
  "training.batch_size=$batch" \
  "training.gradient_accumulation_steps=$accum" \
  "training.output_path=$run_dir" \
  "project.output_dir=$run_dir" \
  project.wandb.project=ikv-robodojo \
  "project.wandb.run_name=openwam_$mode" \
  "${extra[@]}"
python - "$run_dir" <<'PY'
import json, pathlib, sys
base = pathlib.Path(sys.argv[1])
checkpoints = list(base.glob('*/checkpoint_step_*.safetensors'))
if not checkpoints:
    raise SystemExit('OpenWAM final checkpoint missing')
chosen = max(checkpoints, key=lambda x: (x.stat().st_mtime, x.name))
(chosen.parent / 'run_complete.json').write_text(json.dumps({'validated': True, 'epochs': 5, 'checkpoint': str(chosen)}, indent=2))
(base / 'run_complete.json').write_text(json.dumps({'validated': True, 'epochs': 5, 'checkpoint': str(chosen)}, indent=2))
PY
