#!/usr/bin/env bash
set -euo pipefail
mode="${1:?baseline or ikv}"
source /mnt/cfs/9wt59p/genalyu/ikv-robodojo/scripts/robodojo_env.sh
if [[ "$mode" != baseline && "$mode" != ikv ]]; then exit 2; fi
config="pi05_robodojo_all_${mode}"
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
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_COMPILATION_CACHE_DIR="$ROBODOJO_ROOT/jax-cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$ROBODOJO_ROOT/runs/pi05_$mode"
cd "$ROBODOJO_REPO/openpi"
python="$ROBODOJO_ROOT/envs/openpi/bin/python"
official_stats="$ROBODOJO_PI05_ASSETS_BASE_DIR/official-release/arx_x5_sim/norm_stats.json"
test -f "$official_stats"
stats_dir="$ROBODOJO_PI05_ASSETS_BASE_DIR/$config/arx_x5_sim"
mkdir -p "$stats_dir"
cp "$official_stats" "$stats_dir/norm_stats.json"
test "$(sha256sum "$official_stats" | cut -d " " -f 1)" = "$(sha256sum "$stats_dir/norm_stats.json" | cut -d " " -f 1)"
"$python" scripts/train.py "$config" \
  --exp-name all_tasks_seed_0 \
  --checkpoint-base-dir "$ROBODOJO_ROOT/runs" \
  --project-name ikv-robodojo
python - "$ROBODOJO_ROOT" "$mode" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
mode = sys.argv[2]
run = root / f'runs/pi05_robodojo_all_{mode}/all_tasks_seed_0'
checkpoint = run / '59999'
if not (checkpoint / 'params').is_dir():
    raise SystemExit('PI05 final step 59999 checkpoint missing')
result = root / f'runs/pi05_{mode}/run_complete.json'
result.write_text(json.dumps({'validated': True, 'steps': 60000, 'checkpoint': str(checkpoint)}, indent=2))
PY
