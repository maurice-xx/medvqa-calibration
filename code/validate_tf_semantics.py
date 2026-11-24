#!/usr/bin/env python3
"""独立比较全logits参考与整数尾窗，复核原prompt下A/B/C完整输出。"""
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import signal
import select
import time
import traceback

import torch
from PIL import Image
import fixed_output_tf as ft
import rescore_3x3 as rs


def synchronize():
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    report = {"protocol": "full_logits_vs_integer_tail_window_reference_v1",
              "status": "running", "wait_pid": args.wait_pid, "parent_pid": args.parent_pid,
              "tolerance_max_abs_logprob": 1e-4, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "source": args.source, "source_sha256": rs.sha256(Path(args.source).read_bytes())}
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    libc = ctypes.CDLL(None, use_errno=True)
    pidfd = libc.pidfd_open(args.wait_pid, 0)
    if pidfd < 0:
        error = ctypes.get_errno()
        if error == 3:
            pidfd = None
        else:
            raise OSError(error, os.strerror(error))
    if pidfd is not None:
        poller = select.poll()
        poller.register(pidfd, select.POLLIN)
        poller.poll()
        os.close(pidfd)
    print("原Qwen评分进程结束，开始独立单模型语义验证", flush=True)
    model = None
    try:
        rows = ft.read_rows(args.source)
        shortest_a = min((r for r in rows if r.get("status") == "ok" and r["format"] == "A"),
                         key=lambda r: len(r.get("raw_full", r["raw"])))
        first_qid = rows[0]["qid"]
        chosen = [shortest_a]
        chosen += [next(r for r in rows if r["qid"] == first_qid and r["format"] == fmt)
                   for fmt in ("B", "C")]
        expected = 1296
        actual_output = rs.ROOT / "results/corrected/fixed_tf_bf16_20261116/actual/qwen.jsonl"
        records = ft.read_rows(actual_output)
        report["prior_qwen_phase"] = {"expected": expected,
            "rows": len(records), "success": sum(r.get("status") == "ok" for r in records),
            "errors": [{k: r.get(k) for k in ("qid", "format", "status", "reason")}
                       for r in records if r.get("status") != "ok"]}
        proc, model = rs.load_model("qwen")
        report["execution"] = model._rescore_execution
        report["code_sha256"] = {p.name: rs.sha256(p.read_bytes()) for p in
                                  (Path(__file__), Path(rs.__file__), Path(ft.__file__),
                                   Path(rs.__file__).with_name("answer_spans.py"),
                                   Path(rs.__file__).with_name("run_calib.py"))}
        report["cases"] = []
        for row in chosen:
            image = Image.open(row["image_path"]).convert("RGB")
            enc, meta = ft.prepare_actual(proc, row, image, "qwen")
            start = time.time()
            tail_logps = rs.score_encoded(model, enc, meta["full_output_positions"], meta["full_output_token_ids"])
            synchronize()
            tail_elapsed = time.time() - start
            start = time.time()
            enc_reference = enc.to(rs.input_device(model))
            with torch.inference_mode():
                reference = model(**enc_reference, logits_to_keep=0, use_cache=False)
                positions = torch.tensor([p - 1 for p in meta["full_output_positions"]], device=reference.logits.device)
                full_logits = reference.logits[0].index_select(0, positions).float()
                target = torch.tensor(meta["full_output_token_ids"], device=full_logits.device)
                full_logps = (full_logits.gather(1, target[:, None]).squeeze(1)
                              - torch.logsumexp(full_logits, dim=-1)).cpu().tolist()
            synchronize()
            full_elapsed = time.time() - start
            differences = [abs(a - b) for a, b in zip(tail_logps, full_logps)]
            maxdiff = max(differences)
            case = {"qid": row["qid"], "format": row["format"], "raw": meta["raw"],
                    "raw_sha256": meta["raw_sha256"], "prompt_sha256": meta["prompt_sha256"],
                    "prompt_len": meta["prompt_len"], "sequence_len": enc["input_ids"].shape[-1],
                    "full_output_positions": meta["full_output_positions"],
                    "full_output_token_ids": meta["full_output_token_ids"],
                    "span_token_indices": meta["span_token_indices"],
                    "tail_window_logprobs": tail_logps, "reference_full_logits_logprobs": full_logps,
                    "max_abs_logprob_diff": maxdiff,
                    "original_prompt_token_ids_match": meta["original_prompt_token_ids_match"],
                    "full_output_roundtrip_verified": meta["full_output_roundtrip_verified"],
                    "tail_window_elapsed_s": tail_elapsed, "full_logits_elapsed_s": full_elapsed,
                    "pass": len(tail_logps) == len(full_logps) and maxdiff <= report["tolerance_max_abs_logprob"]}
            report["cases"].append(case)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps(case, ensure_ascii=False), flush=True)
            del reference, full_logits, target, positions
            if not case["pass"]:
                raise RuntimeError(f"全logits语义参考不一致: {row['format']} {maxdiff}")
        report["memory_final"] = ft.memory_snapshot()
        report["pass"] = all(r["pass"] for r in report["cases"]) and {r["format"] for r in report["cases"]} == {"A", "B", "C"}
        report["status"] = "pass" if report["pass"] else "fail"
    except Exception as error:
        report.update(status="fail", **{"pass": False}, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    finally:
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        print("独立验证完成，释放模型并恢复主流水线", flush=True)
        os.kill(args.parent_pid, signal.SIGCONT)
    if not report.get("pass"):
        raise RuntimeError(f"语义验证未通过，正式产物保持provisional: {output}")


if __name__ == "__main__":
    main()
