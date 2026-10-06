#!/usr/bin/env bash
set -euo pipefail
mode="${1:?baseline or ikv}"
source /mnt/cfs/9wt59p/genalyu/ikv-robodojo/scripts/robodojo_env.sh
if [[ "$mode" != baseline && "$mode" != ikv ]]; then exit 2; fi
if [[ "$mode" == ikv ]]; then
  echo 'PI05 persistent IKV training is not validated; refusing to run.' >&2
  exit 78
fi
data_root="$ROBODOJO_ROOT/lerobot-v30/data/RoboDojo_lerobot_v30_video"
test -f "$data_root/meta/info.json"
test -f "$ROBODOJO_ROOT/data_manifest.json"
test -f "$ROBODOJO_ROOT/models/pi05-base/source_manifest.json"
link="$HF_LEROBOT_HOME/RoboDojo_sim_arx-x5_v30"
if [[ -e "$link" && ! -L "$link" ]]; then
  echo "unexpected LeRobot dataset at $link" >&2
  exit 1
fi
ln -sfn "$data_root" "$link"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_COMPILATION_CACHE_DIR="$ROBODOJO_ROOT/jax-cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$ROBODOJO_ROOT/runs/pi05_baseline"
cd "$ROBODOJO_REPO/openpi"
python="$ROBODOJO_ROOT/envs/openpi/bin/python"
stats="$ROBODOJO_PI05_ASSETS_BASE_DIR/pi05_robodojo_all_baseline/RoboDojo_sim_arx-x5_v30/norm_stats.json"
if [[ ! -f "$stats" ]]; then
  "$python" scripts/compute_norm_stats.py --config-name pi05_robodojo_all_baseline
fi
test -f "$stats"
"$python" scripts/train.py pi05_robodojo_all_baseline \
  --exp-name all_tasks_seed_0 \
  --checkpoint-base-dir "$ROBODOJO_ROOT/runs" \
  --project-name ikv-robodojo
python - "$ROBODOJO_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
run = root / 'runs/pi05_robodojo_all_baseline/all_tasks_seed_0'
checkpoint = run / '59999'
if not (checkpoint / 'params').is_dir():
    raise SystemExit('PI05 final step 59999 checkpoint missing')
result = root / 'runs/pi05_baseline/run_complete.json'
result.write_text(json.dumps({'validated': True, 'steps': 60000, 'checkpoint': str(checkpoint)}, indent=2))
PY
