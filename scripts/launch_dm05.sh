#!/usr/bin/env bash
set -euo pipefail
mode="${1:?baseline or ikv}"
source /mnt/cfs/9wt59p/genalyu/ikv-robodojo/scripts/robodojo_env.sh
if [[ "$mode" != baseline && "$mode" != ikv ]]; then exit 2; fi
exp=playground/dm05_mem_sft_robodojo_cover_blocks.py
extra=()
batch=4
accum=2
if [[ "$mode" == ikv ]]; then
  exp=playground/dm05_mem_ikv_rgb_robodojo_cover_blocks.py
  batch=1
  accum=8
  extra=(--trainer-config.persistent-ikv-training
    --model-config.ikv-dino-model-path "$ROBODOJO_ROOT/models/dinov2-base")
fi
export ROBODOJO_DM05_DATA_ROOT="$ROBODOJO_ROOT/dm05-official-dataset/robodojo_sim"
export PATH="$ROBODOJO_ROOT/envs/opendm/bin:$PATH"
test -f "$ROBODOJO_DM05_DATA_ROOT/jsonl/index_cache.json"
test -f "$ROBODOJO_ROOT/norm/dm05/norm_stats.json"
test -f "$ROBODOJO_ROOT/models/dm05-mem-base/config.json"
run_dir="$ROBODOJO_ROOT/runs/dm05_$mode"
mkdir -p "$run_dir"
cd "$ROBODOJO_REPO/opendm"
bash script/dm05_launcher.sh \
  --exp "$exp" \
  --task train --nproc_per_node 4 \
  --data-config.dataset-name robodojo_sim_all \
  --data-config.norm-stats-root "$ROBODOJO_ROOT/norm/dm05" \
  --model-config.model-name-or-path "$ROBODOJO_ROOT/models/dm05-mem-base" \
  --trainer-config.output-dir "$run_dir" \
  --trainer-config.num-train-steps 30000 \
  --trainer-config.per-device-train-batch-size "$batch" \
  --trainer-config.gradient-accumulation-steps "$accum" "${extra[@]}"
python - "$run_dir" <<'PY'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
checkpoint = run / 'checkpoint-30000'
if not checkpoint.is_dir() or not (checkpoint / 'norm_stats.json').is_file():
    raise SystemExit('DM05 final checkpoint validation failed')
(run / 'run_complete.json').write_text(json.dumps({'validated': True, 'steps': 30000, 'checkpoint': str(checkpoint)}, indent=2))
PY
