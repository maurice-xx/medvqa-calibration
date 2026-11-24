#!/usr/bin/env python3
"""顺序执行主actual TF、3×3矩阵与外部TF，各阶段独立可恢复且不重新生成。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(".")
PYTHONS = {"qwen": "$HOME/miniconda3/envs/merlin/bin/python",
           "healthgpt": "$HOME/miniconda3/envs/merlin/bin/python",
           "gemma4": "$HOME/venvs/tf5/bin/python"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", required=True)
    parser.add_argument("--recover-on-unfinished", action="store_true")
    args = parser.parse_args()
    smoke = json.loads(Path(args.smoke).read_text())
    if not smoke.get("pass") or smoke.get("max_abs_logprob_diff", 1) > 1e-5:
        raise RuntimeError("Qwen双次完整输出TF冒烟未通过")
    audit = ROOT / "audit/repair_20261116" / args.run_id
    audit.mkdir(parents=True, exist_ok=True)
    lock = (audit / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX if args.recover_on_unfinished else fcntl.LOCK_EX | fcntl.LOCK_NB)
    old_status_path = audit / "pipeline_status.json"
    if args.recover_on_unfinished and old_status_path.exists():
        old = json.loads(old_status_path.read_text())
        if old.get("finished"):
            print("原流水线已正常走完全部阶段，备用恢复入口退出", flush=True)
            return
        for stage in old.get("stages", []):
            pid = stage.get("pid")
            cmdline = Path(f"/proc/{pid}/cmdline") if pid else None
            if stage.get("returncode") is None and cmdline and cmdline.exists():
                command = cmdline.read_bytes().replace(b"\x00", b" ").decode(errors="replace")
                if "fixed_output_tf.py" in command or "run_repaired_matrix.py" in command:
                    print(f"等待既有评分子进程退出: {pid}", flush=True)
                    subprocess.run(["tail", "--pid=" + str(pid), "-f", "/dev/null"], check=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "TOKENIZERS_PARALLELISM": "false"}
    stages = []
    for model in PYTHONS:
        out = ROOT / "results/corrected" / args.run_id / "actual" / f"{model}.jsonl"
        stages.append((f"actual_{model}", model, [str(ROOT / "code/fixed_output_tf.py"),
                       "--model", model, "--cohort", "main", "--out", str(out), "--smoke-then-full"]))
    for model in PYTHONS:
        out = ROOT / "results/corrected" / args.run_id / "rescore" / f"{model}.jsonl"
        source = ROOT / "results/raw" / f"{dict(qwen='qwen500', healthgpt='healthgpt500', gemma4='gemma500')[model]}.jsonl"
        stages.append((f"matrix_{model}", model, [str(ROOT / "code/run_repaired_matrix.py"),
                       "--model", model, "--source", str(source), "--out", str(out)]))
    for model in ("qwen", "healthgpt"):
        out = ROOT / "results/corrected" / args.run_id / "external" / f"{model}.jsonl"
        stages.append((f"external_{model}", model, [str(ROOT / "code/fixed_output_tf.py"),
                       "--model", model, "--cohort", "external", "--out", str(out), "--smoke-then-full"]))
    status = {"pipeline_pid": os.getpid(), "run_id": args.run_id,
              "stage_order": [stage[0] for stage in stages], "stages": []}
    status_path = audit / "pipeline_status.json"
    for name, model, command in stages:
        logpath = audit / f"{name}.log"
        item = {"name": name, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "command": [PYTHONS[model], "-u", *command], "log": str(logpath)}
        status["stages"].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        with logpath.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(item["command"], cwd=ROOT, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT)
            item["pid"] = process.pid
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
            item["returncode"] = process.wait()
        item["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(item, ensure_ascii=False), flush=True)
    status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    status["failed_stages"] = [item["name"] for item in status["stages"] if item.get("returncode") != 0]
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
