#!/usr/bin/env bash
set -euo pipefail
export ROBODOJO_ROOT=/mnt/cfs/9wt59p/genalyu/robodojo-posttrain
export ROBODOJO_REPO=/mnt/cfs/9wt59p/genalyu/ikv-robodojo
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME="$ROBODOJO_ROOT/hf-cache"
export HF_DATASETS_CACHE="$ROBODOJO_ROOT/hf-cache/datasets"
export OPENPI_DATA_HOME="$ROBODOJO_ROOT/openpi-cache"
export HF_LEROBOT_HOME="$ROBODOJO_ROOT/lerobot-cache"
export ROBODOJO_PI05_ASSETS_BASE_DIR="$ROBODOJO_ROOT/norm/pi05"
export ROBODOJO_PI05_BASE_PARAMS="$ROBODOJO_ROOT/models/pi05-base/params"
export UV_CACHE_DIR="$ROBODOJO_ROOT/uv-cache"
export TMPDIR="$ROBODOJO_ROOT/tmp"
export WANDB_DIR="$ROBODOJO_ROOT/wandb"
export WANDB_CACHE_DIR="$ROBODOJO_ROOT/wandb-cache"
export WANDB_CONFIG_DIR="$ROBODOJO_ROOT/wandb-config"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$OPENPI_DATA_HOME" "$HF_LEROBOT_HOME" "$ROBODOJO_PI05_ASSETS_BASE_DIR" "$TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
if [[ -f /mnt/cfs/9wt59p/genalyu/.config/ikv/wandb.env ]]; then
  source /mnt/cfs/9wt59p/genalyu/.config/ikv/wandb.env
fi
