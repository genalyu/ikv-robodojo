"""Fail-closed coverage audit before any of the six post-training jobs can run."""
import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path('/mnt/cfs/9wt59p/genalyu/robodojo-posttrain')
expected = json.loads((ROOT / 'expected_tasks.json').read_text())['tasks']
assert len(expected) == 35 and len(set(expected)) == 35

hdf5_root = ROOT / 'raw/data/RoboDojo'
dm_root = ROOT / 'dm05-official-dataset/robodojo_sim'
pi_root = ROOT / 'lerobot-v30/data/RoboDojo_lerobot_v30_video'

raw = {task: len(list((hdf5_root / task / 'arx_x5/data').glob('episode_*.hdf5'))) for task in expected}
dm = {task: len(list((dm_root / 'jsonl' / task).glob('episode_*.jsonl'))) for task in expected}
raw_extra = sorted(p.name for p in hdf5_root.iterdir() if p.is_dir() and p.name not in expected) if hdf5_root.is_dir() else []
dm_extra = sorted(p.name for p in (dm_root / 'jsonl').iterdir() if p.is_dir() and p.name not in expected) if (dm_root / 'jsonl').is_dir() else []
pi_info = json.loads((pi_root / 'meta/info.json').read_text())
pi_total = int(pi_info['total_episodes'])
pi_episodes = pq.read_table(pi_root / 'meta/episodes/chunk-000/file-000.parquet')
pi_video_keys = [key for key, feature in pi_info['features'].items() if feature['dtype'] == 'video']
pi_required_data = set()
pi_required_video = set()
for row in pi_episodes.to_pylist():
    pi_required_data.add(pi_info['data_path'].format(
        chunk_index=row['data/chunk_index'], file_index=row['data/file_index']))
    for key in pi_video_keys:
        pi_required_video.add(pi_info['video_path'].format(
            video_key=key, chunk_index=row[f'videos/{key}/chunk_index'],
            file_index=row[f'videos/{key}/file_index']))
errors = []
for task in expected:
    if raw[task] != 100:
        errors.append(f'HDF5 {task}: {raw[task]}/100 episodes')
    if dm[task] != 100:
        errors.append(f'DM05 JSONL {task}: {dm[task]}/100 episodes')
if raw_extra or dm_extra:
    errors.append(f'unexpected task directories: HDF5={raw_extra} DM05={dm_extra}')
if pi_total != 3500 or pi_episodes.num_rows != 3500:
    errors.append(f'PI05 LeRobot episodes: info={pi_total} metadata={pi_episodes.num_rows}/3500')
pi_missing = [name for name in sorted(pi_required_data | pi_required_video)
              if not (pi_root / name).is_file() or (pi_root / name).stat().st_size == 0]
if pi_missing:
    errors.append(f'PI05 missing/empty shards: {len(pi_missing)} of {len(pi_required_data | pi_required_video)}; first: {pi_missing[:5]}')
pi_rows = sum(pq.read_metadata(pi_root / name).num_rows for name in sorted(pi_required_data)
              if (pi_root / name).is_file())
if pi_rows != int(pi_info['total_frames']):
    errors.append(f'PI05 parquet rows: {pi_rows}/{pi_info["total_frames"]}')
if errors:
    raise SystemExit('\n'.join(errors))

manifest = {
    'source_tasks': expected,
    'source_revision': 'RoboDojo-Benchmark/RoboDojo master',
    'raw_hdf5_episodes_per_task': raw,
    'dm05_jsonl_episodes_per_task': dm,
    'pi05_lerobot_total_episodes': pi_total,
    'pi05_required_data_shards': len(pi_required_data),
    'pi05_required_video_shards': len(pi_required_video),
    'pi05_parquet_rows': pi_rows,
    'validated': True,
}
target = ROOT / 'data_manifest.json'
tmp = target.with_suffix('.tmp')
tmp.write_text(json.dumps(manifest, indent=2))
tmp.replace(target)
print(f'validated {len(expected)} tasks, {sum(raw.values())} HDF5 episodes, {sum(dm.values())} DM05 episodes, {pi_total} LeRobot episodes')
