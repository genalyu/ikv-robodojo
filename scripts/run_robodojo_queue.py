"""Run six four-GPU RoboDojo jobs after the existing neosim IKV job."""
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/mnt/cfs/9wt59p/genalyu/robodojo-posttrain")
REPO = Path("/mnt/cfs/9wt59p/genalyu/ikv-robodojo")
NEOSIM = Path("/mnt/cfs/9wt59p/genalyu/ikv-task-data/neosim_mem_tactile_chip_selection")
NEOSIM_DONE = NEOSIM / "runs/ikv/checkpoints/checkpoint_step_2000/training_state/complete.json"
ORDER = (
    ("dm05_baseline", "launch_dm05.sh", "baseline"),
    ("dm05_ikv", "launch_dm05.sh", "ikv"),
    ("openwam_baseline", "launch_openwam.sh", "baseline"),
    ("openwam_ikv", "launch_openwam.sh", "ikv"),
    ("pi05_baseline", "launch_pi05.sh", "baseline"),
    ("pi05_ikv", "launch_pi05.sh", "ikv"),
)


def write_status(**fields):
    path = ROOT / "queue_status.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(fields, updated_at=datetime.now(timezone.utc).isoformat())
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=2))
    tmp.replace(path)


def live_neosim():
    result = subprocess.run(
        ["pgrep", "-f", "python -u -m n0_twam.train.*neosim_mem_tactile_chip_selection"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def gpu_free():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    used = [int(line.strip()) for line in result.stdout.splitlines()]
    return len(used) == 4 and all(x < 2000 for x in used)


def ready():
    # The official DM05 archive contains 34 tasks; make its omitted dlc task
    # from the same RoboDojo HDF5 demonstrations before the full-source audit.
    cache = ROOT / "dm05-official-dataset/robodojo_sim/jsonl/index_cache.json"
    if not cache.is_file():
        prepared = subprocess.run(
            ["python3", str(REPO / "scripts/prepare_dm05_dlc.py")],
            capture_output=True, text=True, check=False,
        )
        (ROOT / "prepare_dm05_dlc.log").write_text(
            prepared.stdout + prepared.stderr)
        if prepared.returncode:
            return False
    # Run the complete source audit whenever the downloads may have finished.
    manifest = ROOT / "data_manifest.json"
    if not manifest.is_file():
        last = getattr(ready, "_last_audit", 0)
        if time.monotonic() - last >= 600:
            ready._last_audit = time.monotonic()
            audit = subprocess.run(
                ["python3", str(REPO / "scripts/verify_robodojo_data.py")],
                capture_output=True, text=True, check=False,
            )
            (ROOT / "preflight_data.log").write_text(audit.stdout + audit.stderr)
    if not manifest.is_file() or json.loads(manifest.read_text()).get("validated") is not True:
        return False
    if subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip() != "main":
        return False
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        return False
    required = [
        ROOT / "envs/opendm/bin/python",
        ROOT / "envs/openwam/bin/python",
        ROOT / "envs/openpi/bin/python",
        ROOT / "models/dm05-mem-base/config.json",
        ROOT / "models/dinov2-base/config.json",
        ROOT / "models/openwam-alpha-foundation/config.yaml",
        ROOT / "models/pi05-base/source_manifest.json",
        ROOT / "norm/dm05/norm_stats.json",
        ROOT / "norm/pi05/official-release/arx_x5_sim/norm_stats.json",
        ROOT / "dm05-official-dataset/robodojo_sim/jsonl/index_cache.json",
    ]
    if not all(path.is_file() for path in required):
        return False
    if not list((ROOT / "models/openwam-alpha-foundation").rglob("*.safetensors")):
        return False
    for launcher in {item[1] for item in ORDER}:
        script = REPO / "scripts" / launcher
        if not script.is_file() or not os.access(script, os.X_OK):
            return False
        if "training is not validated; refusing to run" in script.read_text():
            return False
    return True


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (NEOSIM_DONE.is_file() and not live_neosim() and gpu_free() and ready()):
            if not NEOSIM_DONE.is_file() or live_neosim():
                state = "waiting_for_neosim_ikv"
            elif not ready():
                state = "waiting_for_preflight"
            else:
                state = "waiting_for_gpus"
            write_status(state=state, order=[x[0] for x in ORDER])
            time.sleep(60)
        for run_id, launcher, mode in ORDER:
            result_path = ROOT / "runs" / run_id / "run_complete.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                if result.get("validated") is True:
                    continue
                raise RuntimeError(f"existing unvalidated completion: {run_id}")
            if not gpu_free():
                raise RuntimeError("four GPUs not free before " + run_id)
            write_status(state="running", current=run_id)
            log = ROOT / "logs" / f"{run_id}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as output:
                code = subprocess.run(
                    ["bash", str(REPO / "scripts" / launcher), mode],
                    cwd=REPO,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=False,
                ).returncode
            if code:
                write_status(state="failed", current=run_id, exit_code=code)
                raise SystemExit(code)
            if not result_path.is_file() or json.loads(result_path.read_text()).get("validated") is not True:
                write_status(state="failed_validation", current=run_id)
                raise RuntimeError(f"launcher did not validate: {run_id}")
        write_status(state="complete", current=None)


if __name__ == "__main__":
    main()
