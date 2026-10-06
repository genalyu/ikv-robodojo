"""Convert each complete dlc HDF5 episode while the mirror download continues."""
import json
import os
import time
from pathlib import Path

import h5py

from convert_robodojo_dm05 import CAMS, convert_episode

ROOT = Path("/mnt/cfs/9wt59p/genalyu/robodojo-posttrain")
SOURCE = ROOT / "raw-dlc/data/RoboDojo/dlc/arx_x5/data"
OUTPUT = ROOT / "dm05-official-dataset/robodojo_sim"
PID = ROOT / "dlc_convert_follow.pid"


def complete(source: Path) -> bool:
    episode = source.stem
    jsonl = OUTPUT / "jsonl/dlc" / f"{episode}.jsonl"
    video = OUTPUT / "video/dlc" / episode
    return jsonl.is_file() and jsonl.stat().st_size > 0 and all(
        (video / f"{cam}.mp4").is_file() and
        (video / f"{cam}.mp4").stat().st_size > 0 for cam in CAMS)


def main():
    PID.write_text(str(os.getpid()))
    try:
        while True:
            files = sorted(SOURCE.glob("episode_*.hdf5"))
            for file in files:
                if complete(file) or file.stat().st_size == 0 or not h5py.is_hdf5(file):
                    continue
                episode, frames = convert_episode(file, OUTPUT, "dlc")
                print(episode, frames, flush=True)
            done = sum(complete(file) for file in files)
            if len(files) == 100 and done == 100:
                print(json.dumps({"episodes": done, "converted": True}), flush=True)
                return
            print(f"waiting for dlc source: {len(files)}/100 downloaded, {done}/100 converted", flush=True)
            time.sleep(60)
    finally:
        PID.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
