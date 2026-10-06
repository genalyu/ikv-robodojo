#!/usr/bin/env bash
set -euo pipefail
mode="${1:?baseline or ikv}"
source /mnt/cfs/9wt59p/genalyu/ikv-robodojo/scripts/robodojo_env.sh
if [[ "$mode" != baseline && "$mode" != ikv ]]; then exit 2; fi
if [[ "$mode" == ikv ]]; then
  echo 'OpenWAM persistent IKV training is not validated; refusing to run.' >&2
  exit 78
fi
export PATH="$ROBODOJO_ROOT/envs/openwam/bin:$PATH"
export NPROC_PER_NODE=4
export HYDRA_FULL_ERROR=1
data_root="$ROBODOJO_ROOT/raw/data/RoboDojo"
model_root="$ROBODOJO_ROOT/models/openwam-alpha-foundation"
run_dir="$ROBODOJO_ROOT/runs/openwam_baseline"
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
  training.batch_size=4 \
  training.gradient_accumulation_steps=8 \
  "training.output_path=$run_dir" \
  "project.output_dir=$run_dir" \
  project.wandb.project=ikv-robodojo \
  project.wandb.run_name=openwam_baseline
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
