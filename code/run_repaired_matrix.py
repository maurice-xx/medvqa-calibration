#!/usr/bin/env python3
"""运行修复矩阵或先进行无GPU的全题token边界审计。"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import rescore_3x3 as rs


def cpu_audit(args):
    source = Path(args.source) if args.source else rs.ROOT / "results/raw" / f"{rs.BASE_FILES[args.model]}.jsonl"
    rows = rs.load_base_answers(source)
    questions = rs.validate_seed(args.seed)
    if args.limit:
        questions = questions[:args.limit]
    proc = rs.AutoProcessor.from_pretrained(rs.MODEL_PATHS[args.model], local_files_only=True)
    proc.image_processor.max_pixels = rs.MAX_PIXELS
    proc.image_processor.min_pixels = 4 * 28 * 28
    counts = Counter()
    failures = []
    image_cache = {}
    max_prompt = 0
    max_answer = 0
    examples = []
    start = time.time()
    for i, q in enumerate(questions, 1):
        path = q["image_path"]
        if path not in image_cache:
            image_cache[path] = rs.Image.open(path).convert("RGB")
        for g in rs.FORMAT_SET:
            for s in rs.FORMAT_SET:
                try:
                    enc, meta = rs.prepare_score(proc, image_cache[path], q["question"],
                                                 rows[q["qid"]][g]["answer"], s, args.model)
                    assert not set(meta["answer_token_ids"]) & set(meta["special_token_ids"])
                    assert not set(meta["prefix_token_ids"]) & set(meta["special_token_ids"])
                    assert meta["answer_token_ids"] == [enc["input_ids"][0, j].item()
                                                        for j in meta["scored_positions"]]
                    max_prompt = max(max_prompt, meta["prompt_len"])
                    max_answer = max(max_answer, len(meta["answer_token_ids"]))
                    counts["ok"] += 1
                    counts["answer_boundary_crossing"] += bool(meta["answer_boundary_crossing_tokens"])
                    if len(examples) < 9:
                        examples.append({"qid": q["qid"], "g": g, "s": s, **meta})
                except Exception as exc:
                    failures.append({"qid": q["qid"], "g": g, "s": s,
                                     "reason": f"{type(exc).__name__}:{exc}"})
        if i % 25 == 0 or i == 1:
            rs.log(f"CPU边界审计 {args.model}: {i}/{len(questions)}题，{dict(counts)}")
    result = {"model": args.model, "protocol": rs.PROTOCOL, "source": str(source),
              "source_sha256": rs.sha256(source.read_bytes()),
              "code_sha256": rs.CODE_SHA256,
              "questions": len(questions), "expected_cells": len(questions) * 9,
              "counts": dict(counts), "failures": failures, "max_prompt_tokens": max_prompt,
              "max_answer_tokens": max_answer, "elapsed_s": time.time() - start,
              "examples": examples, "image_processor_config": rs.jsonable(proc.image_processor.to_dict())}
    out = rs.ROOT / "audit/repair_20261116" / f"boundary_audit_{args.model}.json"
    if out.exists():
        raise FileExistsError(f"不覆盖已有审计: {out}")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    rs.log(f"审计完成: {out}，{dict(counts)}，失败{len(failures)}")
    if failures or counts["ok"] != len(questions) * 9:
        raise RuntimeError("完整token边界审计失败")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(rs.MODEL_PATHS))
    ap.add_argument("--seed", default=str(rs.ROOT / "data/seed500.jsonl"))
    ap.add_argument("--source")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cpu-audit", action="store_true")
    ap.add_argument("--with-aprime", action="store_true")
    args = ap.parse_args()
    if args.cpu_audit:
        cpu_audit(args)
        return
    forwarded = ["--model", args.model, "--seed", args.seed, "--smoke-then-full"]
    if args.source:
        forwarded += ["--source", args.source]
    if args.out:
        forwarded += ["--out", args.out]
    if args.limit:
        forwarded += ["--limit", str(args.limit)]
    if args.with_aprime:
        forwarded.append("--with-aprime")
    sys.argv = [rs.__file__] + forwarded
    rs.main()


if __name__ == "__main__":
    main()
