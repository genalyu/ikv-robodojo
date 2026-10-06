#!/usr/bin/env bash
# Persistent per-layer KV training requires ordered, prepared episodes.
# Example: bash scripts/train_dm05_online_ikv.sh --episodes /path/to/episodes --output /path/to/output
set -euo pipefail
cd /home/ubuntu/genalyu/ikv-robodojo/opendm
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec /home/ubuntu/genalyu/envs/opendm/bin/python scripts/train_dm05_persistent_kv.py "$@"
