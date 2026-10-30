"""Run the frozen R9 matrix once; three regions concurrently, seeds sequentially."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/agent_forecasting/r9_development_20261030.json"
REPO_PY = "/root/workspace/Code/DualTimesField/.venv/bin/python"
CHRONOS_PY = "/root/workspace/Code/DualTimesField-foundation-baselines/.venvs/chronos2/bin/python"


def main():
    cfg = json.loads(CONFIG.read_text())
    out = ROOT / cfg["output_root"]
    logs = out / "commands"
    logs.mkdir(exist_ok=False)
    started = time.monotonic()
    records = []

    def run(stage, region, seed, arm=None):
        name = f"{seed}_{region}_{arm or stage}"
        cmd = [CHRONOS_PY if stage == "base" else REPO_PY, "-B", "-m",
               "forecasting.agentic_dev_eval.r9", stage, "--region", region, "--seed", str(seed)]
        if arm:
            cmd += ["--arm", arm]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   PYTHONDONTWRITEBYTECODE="1")
        start = time.monotonic()
        record = {"name": name, "command": cmd, "started_unix": time.time()}
        (logs / f"{name}.start.json").write_text(json.dumps(record, indent=2) + "\n")
        with (logs / f"{name}.txt").open("x") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                timeout = (cfg["base_deadline_seconds"] + 60 if stage == "base" else
                           cfg["origins_per_region"] * cfg["decision_deadline_seconds"] + 90)
                returncode = process.wait(timeout=timeout)
            except BaseException:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        record.update(returncode=returncode, wall_seconds=time.monotonic() - start)
        (logs / f"{name}.exit.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(json.dumps(record), flush=True)
        if returncode:
            raise RuntimeError(f"{name} failed; not retried")

    def region_run(region, seed):
        run("base", region, seed)
        for arm in ("B", "C", "D"):
            run("decide", region, seed, arm)

    for seed in cfg["seeds"]:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(region_run, region, seed) for region in cfg["regions"]]
            for future in futures:
                future.result()
    (logs / "matrix.json").write_text(json.dumps({"wall_seconds": time.monotonic() - started,
        "commands": sorted(records, key=lambda r: r["name"])}, indent=2) + "\n")


if __name__ == "__main__":
    main()
