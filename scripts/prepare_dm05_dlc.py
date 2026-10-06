"""Fill the DM05 release's missing dlc task from the RoboDojo HDF5 source."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/mnt/cfs/9wt59p/genalyu/robodojo-posttrain")
REPO = Path("/mnt/cfs/9wt59p/genalyu/ikv-robodojo")
DM = ROOT / "dm05-official-dataset/robodojo_sim"
TASK = "dlc"


def main() -> int:
    candidates = [
        ROOT / "raw-dlc/data/RoboDojo",
        ROOT / "raw/data/RoboDojo",
    ]
    source = next((p for p in candidates if len(files := list(
        (p / TASK / "arx_x5/data").glob("episode_*.hdf5"))) == 100
        and all(file.stat().st_size > 0 for file in files)), None)
    if source is None:
        print("dlc source HDF5 is incomplete", flush=True)
        return 1
    expected = json.loads((ROOT / "expected_tasks.json").read_text())["tasks"]
    if len(expected) != 35 or TASK not in expected:
        raise RuntimeError("RoboDojo task manifest mismatch")
    output = DM / "jsonl" / TASK
    videos = DM / "video" / TASK
    def complete() -> bool:
        episodes = sorted(output.glob("episode_*.jsonl"))
        return len(episodes) == 100 and all(
            (videos / ep.stem / f"{cam}.mp4").is_file() and
            (videos / ep.stem / f"{cam}.mp4").stat().st_size > 0
            for ep in episodes for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist")
        )
    if not complete():
        follower = ROOT / "dlc_convert_follow.pid"
        if follower.is_file():
            try:
                os.kill(int(follower.read_text()), 0)
            except (OSError, ValueError):
                pass
            else:
                print("dlc incremental conversion is still running", flush=True)
                return 1
        subprocess.run([
            "python3", str(REPO / "scripts/convert_robodojo_dm05.py"),
            "--task", TASK, "--source-root", str(source),
            "--output-root", str(DM), "--episodes", "100", "--workers", "2",
        ], check=True)
    if not complete():
        raise RuntimeError("dlc conversion incomplete")
    if all(len(list((DM / "jsonl" / task).glob("episode_*.jsonl"))) == 100
           for task in expected):
        cache = DM / "jsonl/index_cache.json"
        if not cache.is_file() or len(json.loads(cache.read_text())["data"]) != 3500:
            subprocess.run([
                str(ROOT / "envs/opendm/bin/python"), "-c",
                "from opendm.data.dataset import build_index_cache; "
                f"build_index_cache({str(DM / 'jsonl')!r})",
            ], cwd=REPO / "opendm", check=True)
        return 0
    print("other DM05 task episodes incomplete", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
