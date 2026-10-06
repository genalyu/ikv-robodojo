"""Fail-closed coverage audit before any of the six post-training jobs can run."""
import json
from pathlib import Path

ROOT = Path('/mnt/cfs/9wt59p/genalyu/robodojo-posttrain')
expected = json.loads((ROOT / 'expected_tasks.json').read_text())['tasks']
assert len(expected) == 35 and len(set(expected)) == 35

hdf5_root = ROOT / 'raw/data/RoboDojo'
dm_root = ROOT / 'dm05-official-dataset/robodojo_sim'
pi_root = ROOT / 'lerobot-v30/data/RoboDojo_lerobot_v30_video'

raw = {task: len(list((hdf5_root / task / 'arx_x5/data').glob('episode_*.hdf5'))) for task in expected}
dm = {task: len(list((dm_root / 'jsonl' / task).glob('episode_*.jsonl'))) for task in expected}
raw_extra = sorted(p.name for p in hdf5_root.iterdir() if p.is_dir() and p.name not in expected)
dm_extra = sorted(p.name for p in (dm_root / 'jsonl').iterdir() if p.is_dir() and p.name not in expected)
pi_info = json.loads((pi_root / 'meta/info.json').read_text())
pi_total = int(pi_info['total_episodes'])
errors = []
for task in expected:
    if raw[task] != 100:
        errors.append(f'HDF5 {task}: {raw[task]}/100 episodes')
    if dm[task] != 100:
        errors.append(f'DM05 JSONL {task}: {dm[task]}/100 episodes')
if raw_extra or dm_extra:
    errors.append(f'unexpected task directories: HDF5={raw_extra} DM05={dm_extra}')
if pi_total != 3500:
    errors.append(f'PI05 LeRobot episodes: {pi_total}/3500')
if errors:
    raise SystemExit('\n'.join(errors))

manifest = {
    'source_tasks': expected,
    'source_revision': 'RoboDojo-Benchmark/RoboDojo master',
    'raw_hdf5_episodes_per_task': raw,
    'dm05_jsonl_episodes_per_task': dm,
    'pi05_lerobot_total_episodes': pi_total,
    'validated': True,
}
target = ROOT / 'data_manifest.json'
tmp = target.with_suffix('.tmp')
tmp.write_text(json.dumps(manifest, indent=2))
tmp.replace(target)
print(f'validated {len(expected)} tasks, {sum(raw.values())} HDF5 episodes, {sum(dm.values())} DM05 episodes, {pi_total} LeRobot episodes')
EOF'