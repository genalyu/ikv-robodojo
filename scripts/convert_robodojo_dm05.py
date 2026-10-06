"""Convert RoboDojo raw HDF5 episodes to the OpenDM RoboDojo JSONL/video layout.

The released OpenDM archive omits dlc. This converter reproduces its
state/action/image record mapping and adds the terminal record used by the
released episodes: terminal state and action are the last raw action.
"""
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np

CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
JOINTS = ("left_arm_joint_states", "left_ee_joint_states",
          "right_arm_joint_states", "right_ee_joint_states")


def convert_episode(source: Path, output: Path, task: str) -> tuple[str, int]:
    episode = source.stem
    json_path = output / "jsonl" / task / f"{episode}.jsonl"
    video_dir = output / "video" / task / episode
    video_dir.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source) as h:
        states = np.concatenate([h[f"state/{key}"][:] for key in JOINTS], axis=1)
        actions = np.concatenate([h[f"action/{key}"][:] for key in JOINTS], axis=1)
        count = len(actions)
        assert states.shape == actions.shape == (count, 14)
        fps = int(h["additional_info/frequency"][()])
        assert fps == 25, (source, fps)
        instruction = h["instruction"][()].decode("utf-8")
        images = {cam: h[f"vision/{cam}/colors"][:] for cam in CAMS}
        assert all(len(images[cam]) == count for cam in CAMS)
        for cam in CAMS:
            dest = video_dir / f"{cam}.mp4"
            if dest.is_file() and dest.stat().st_size > 0:
                continue
            tmp = dest.with_suffix(".mp4.tmp")
            process = subprocess.Popen(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                 "-f", "rawvideo", "-pixel_format", "bgr24",
                 "-video_size", "640x480", "-framerate", str(fps), "-i", "pipe:0",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "30",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(tmp)],
                stdin=subprocess.PIPE)
            last = None
            try:
                for payload in images[cam]:
                    frame = cv2.imdecode(np.frombuffer(payload.tobytes(), np.uint8),
                                         cv2.IMREAD_COLOR)
                    assert frame is not None and frame.shape == (480, 640, 3), source
                    process.stdin.write(frame.tobytes())
                    last = frame
                process.stdin.write(last.tobytes())
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError(f"ffmpeg failed: {source} {cam}")
                tmp.replace(dest)
            except Exception:
                process.kill()
                process.wait()
                tmp.unlink(missing_ok=True)
                raise
        tmp = json_path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as file:
            for i in range(count + 1):
                state = states[i].tolist() if i < count else actions[-1].tolist()
                action = actions[i].tolist() if i < count else actions[-1].tolist()
                row = {
                    "images_1": {"type": "video", "url": f"./{task}/{episode}/cam_head.mp4", "frame_idx": i},
                    "images_2": {"type": "video", "url": f"./{task}/{episode}/cam_left_wrist.mp4", "frame_idx": i},
                    "images_3": {"type": "video", "url": f"./{task}/{episode}/cam_right_wrist.mp4", "frame_idx": i},
                    "state": state, "action": action, "prompt": instruction,
                    "is_robot": True, "task_name": f"robodojo_{task}",
                }
                file.write(json.dumps(row, separators=(",", ":")) + "\n")
        tmp.replace(json_path)
    return episode, count + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    files = sorted((args.source_root / args.task / "arx_x5/data").glob("episode_*.hdf5"))
    if len(files) < args.episodes:
        raise SystemExit(f"{args.task}: only {len(files)}/{args.episodes} HDF5 episodes")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert_episode, path, args.output_root, args.task)
                   for path in files[:args.episodes]]
        for future in as_completed(futures):
            episode, frames = future.result()
            print(episode, frames, flush=True)


if __name__ == "__main__":
    main()
