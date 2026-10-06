"""Verify the 35-task DM05 dataset before either DM05 run begins."""
import json
from pathlib import Path

ROOT = Path("/mnt/cfs/9wt59p/genalyu/robodojo-posttrain")
DM = ROOT / "dm05-official-dataset/robodojo_sim"
tasks = json.loads((ROOT / "expected_tasks.json").read_text())["tasks"]
assert len(tasks) == 35 and len(set(tasks)) == 35
cache_path = DM / "jsonl/index_cache.json"
if not cache_path.is_file():
    raise SystemExit("DM05 index cache missing")
cache = json.loads(cache_path.read_text())["data"]
if len(cache) != 3500:
    raise SystemExit(f"DM05 index cache has {len(cache)}/3500 episodes")
errors = []
for task in tasks:
    episodes = sorted((DM / "jsonl" / task).glob("episode_*.jsonl"))
    if len(episodes) != 100:
        errors.append(f"{task}: {len(episodes)}/100 JSONL episodes")
    for episode in episodes:
        if str(episode) not in cache or cache[str(episode)] <= 1:
            errors.append(f"{episode}: missing/invalid index")
        for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            video = DM / "video" / task / episode.stem / f"{cam}.mp4"
            if not video.is_file() or video.stat().st_size == 0:
                errors.append(f"{video}: missing/empty video")
if errors:
    raise SystemExit("\n".join(errors[:20]))
manifest = {"validated": True, "tasks": tasks, "episodes": 3500, "videos": 10500}
target = ROOT / "dm05_manifest.json"
tmp = target.with_suffix(".tmp")
tmp.write_text(json.dumps(manifest, indent=2))
tmp.replace(target)
print("validated 35 DM05 tasks, 3500 episodes, 10500 videos")
