#!/usr/bin/env python3
"""实际原提示下 greedy 重新生成；不是 teacher forcing。

保存本次真实生成的完整 raw、答案、生成和答案 token ids/logprobs。
历史 stored 缺失的 token 证据显式标为不可用，不用 stored 文本冒充本次输出。
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor
from answer_spans import (SPAN_VERSION, legacy_span_indices, score_generation)
from run_calib import FORMATS, MODEL_PATHS, SYS, normalize, validate_regeneration_jobs

ROOT = Path(".")
BASE = {"qwen": "qwen500", "healthgpt": "healthgpt500", "gemma4": "gemma500"}
MAX_PIXELS = 448 * 448
MAX_NEW_TOKENS = 128


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def span_conf(scores, gen_ids, processor, fmt):
    """兼容旧入口；实际逻辑统一由 answer_spans 实现。"""
    result = score_generation(scores, gen_ids, processor, fmt)
    return (result["conf_token_geo"], result["conf_token_first"],
            result["span_len"], result["span_token_logprobs"])


def load_model(name):
    import transformers as tf
    path = MODEL_PATHS[name]
    proc = AutoProcessor.from_pretrained(path, local_files_only=True)
    if hasattr(proc, "image_processor"):
        try:
            proc.image_processor.max_pixels = MAX_PIXELS
            proc.image_processor.min_pixels = 4 * 28 * 28
        except Exception as error:
            log(f"pixel warn: {error}")
    if "gemma" in path.lower():
        cls = getattr(tf, "AutoModelForMultimodalLM", None) or tf.Gemma4ForConditionalGeneration
    else:
        cls = tf.Qwen3VLForConditionalGeneration
    free, _ = torch.cuda.mem_get_info(0)
    if free / 2**30 < 15.5:
        raise RuntimeError(f"insufficient GPU: {free/2**30:.1f} GiB")
    model = cls.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                device_map="cuda:0", local_files_only=True)
    model.eval()
    return proc, model


def compare_evidence(evidence, stored, tokenizer, fmt):
    stored_ids = stored.get("gen_ids")
    stored_span_ids = stored.get("span_token_ids")
    same_definition = stored.get("span_version") == SPAN_VERSION
    conf, stored_conf = evidence["conf_token_geo"], stored.get("conf_token_geo")
    old_span = legacy_span_indices(evidence["gen_ids"], tokenizer, fmt)
    old_logps = [evidence["gen_token_logprobs"][i] for i in old_span]
    old_conf = math.exp(sum(old_logps) / len(old_logps)) if old_logps else None
    return {
        "stored_raw_match": evidence["raw"] == stored.get("raw", ""),
        "stored_raw_match_after_strip_diagnostic": evidence["raw"].strip() == stored.get("raw", "").strip(),
        "stored_answer_match": evidence["ans_raw"] == stored.get("ans_raw"),
        "stored_gen_token_ids_available": stored_ids is not None,
        "generated_token_ids_match_stored": evidence["gen_ids"] == stored_ids if stored_ids is not None else None,
        "stored_span_token_ids_available": stored_span_ids is not None,
        "span_token_ids_match_stored": evidence["span_token_ids"] == stored_span_ids if stored_span_ids is not None else None,
        "stored_token_ids_source": stored.get("token_ids_source", "not_recorded"),
        "conf_stored": stored_conf,
        "stored_conf_span_compatible": same_definition,
        "abs_diff": abs(conf - stored_conf) if same_definition and conf is not None and stored_conf is not None else None,
        "legacy_span_conf_recomputed": old_conf,
        "legacy_conf_abs_diff": abs(old_conf - stored_conf) if not same_definition and old_conf is not None and stored_conf is not None else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_PATHS))
    parser.add_argument("--seed", default=str(ROOT / "data/seed500.jsonl"))
    parser.add_argument("--raw", help="用于对照的 stored JSONL")
    parser.add_argument("--out", help="独立复验输出；默认 results/corrected/rescore_actual")
    parser.add_argument("--jobs-jsonl", help="只复验所列 qid/format")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    raw_path = Path(args.raw) if args.raw else ROOT / "results/raw" / f"{BASE[args.model]}.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    by_key = {(r["qid"], r["format"]): r for r in rows if r.get("status") == "ok"}
    if args.jobs_jsonl:
        requested = [json.loads(line) for line in Path(args.jobs_jsonl).read_text().splitlines() if line.strip()]
        requested = [r for r in requested if r.get("model", args.model) == args.model]
        validate_regeneration_jobs(requested)
        for row in requested:
            key = (row["qid"], row.get("format", row.get("g")))
            stored = by_key.get(key)
            if stored is not None:
                validate_regeneration_jobs([dict(stored, action="regenerate")])
            else:
                by_key[key] = row
        keys = [(r["qid"], r.get("format", r.get("g"))) for r in requested]
        if args.limit:
            keys = keys[:args.limit]
    else:
        questions = [json.loads(line) for line in Path(args.seed).read_text().splitlines() if line.strip()]
        if args.limit:
            questions = questions[:args.limit]
        keys = [(q["qid"], fmt) for q in questions for fmt in ("A", "B", "C")]
    out = Path(args.out) if args.out else ROOT / "results/corrected/rescore_actual" / f"{args.model}.jsonl"
    protected = ROOT / "results"
    if (out.resolve().parent == protected / "rescore_actual" or
            (out.resolve().parent.name in {"raw", "raw_vqarad", "raw_vqamed", "raw_wording"}
             and out.resolve().parent.parent == protected)):
        raise ValueError("不得覆盖历史 raw/actual；请使用独立 corrected 输出")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            if (row.get("status") == "ok" and row.get("span_version") == SPAN_VERSION
                    and row.get("teacher_forcing") is False
                    and row.get("recovery_status") == "regenerated_with_token_evidence"
                    and isinstance(row.get("gen_ids"), list)):
                done.add((row["qid"], row.get("format", row.get("g"))))
    jobs = [key for key in keys if key in by_key and key not in done]
    if not jobs:
        log("no pending jobs")
        return
    proc, model = load_model(args.model)
    log("model loaded")
    n_ok = n_fail = 0
    with out.open("a", encoding="utf-8") as handle:
        for index, (qid, fmt) in enumerate(jobs, 1):
            stored = by_key[(qid, fmt)]
            question = stored
            rec = {"model": args.model, "qid": qid, "g": fmt, "format": fmt,
                   "category": question["category"], "image_path": question["image_path"],
                   "generation_protocol": "greedy_actual_prompt_regeneration",
                   "teacher_forcing": False, "teacher_forcing_verified": False,
                   "stored_source": str(raw_path), "source_raw_file": str(raw_path),
                   "original_raw": stored.get("raw"),
                   "saved_raw": stored.get("saved_raw", stored.get("raw")),
                   "original_output_raw": stored.get("original_output_raw", stored.get("raw")),
                   "original_token_identity_verified": False,
                   "raw_text_integrity": stored.get("raw_text_integrity", {
                       "status": "missing" if not stored.get("raw") else "saved_output_available",
                       "basis": "independent_generation_comparison", "original_completeness_proven": False}),
                   "prompt_system": SYS,
                   "prompt_user_text": f"{FORMATS[fmt]['instr']}\n\nQuestion: {question['question']}",
                   "dataset": stored.get("dataset", "main"),
                   "schema_version": "saved-output-tf-v2-provisional",
                   "formal_analysis_eligible": False,
                   "question": question["question"], "prompt_instruction": FORMATS[fmt]["instr"],
                   "system_prompt": SYS,
                   "prompt_text": f"{FORMATS[fmt]['instr']}\n\nQuestion: {question['question']}"}
            try:
                image = Image.open(question["image_path"]).convert("RGB")
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": SYS}]},
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"{FORMATS[fmt]['instr']}\n\nQuestion: {question['question']}"}]},
                ]
                kwargs = dict(tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)
                if args.model == "gemma4":
                    kwargs["enable_thinking"] = False
                inputs = proc.apply_chat_template(messages, **kwargs).to("cuda:0")
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                               do_sample=False, output_scores=True,
                                               return_dict_in_generate=True)
                ids = generated.sequences[0, inputs["input_ids"].shape[-1]:].tolist()
                evidence = score_generation(generated.scores, ids, proc, fmt)
                rec.update(evidence)
                rec.update(compare_evidence(evidence, stored, proc.tokenizer, fmt))
                rec["answer"] = evidence["ans_raw"]
                rec["conf_recomputed"] = evidence["conf_token_geo"]
                rec["ans_norm"] = normalize(evidence["ans_raw"], question["category"])
                rec["gold"] = question["gold"]
                rec["gold_norm"] = normalize(question["gold"], question["category"])
                rec["correct"] = bool(rec["ans_norm"]) and rec["ans_norm"] == rec["gold_norm"]
                rec["recovery_status"] = "regenerated_with_token_evidence"
                rec["status"] = "ok"
                rec["formal_analysis_eligible"] = evidence["span_scoreable"]
                n_ok += rec["status"] == "ok"
                n_fail += rec["status"] != "ok"
            except Exception as error:
                rec.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
                n_fail += 1
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 50 == 0 or index == len(jobs):
                log(f"{index}/{len(jobs)} ok={n_ok} fail={n_fail}")
    log(f"FINISHED {args.model}: ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    main()
