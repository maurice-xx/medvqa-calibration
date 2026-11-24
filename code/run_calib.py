#!/usr/bin/env python3
"""校准研究推理脚本：3 种回答格式 × 逐 token 置信度提取。

用法：
  CUDA_VISIBLE_DEVICES=0 python run_calib.py --model qwen \
      --seed data/seed100.jsonl --out results/raw/qwen.jsonl [--limit N]

置信度定义（互不混用，逐条记录）：
- conf_token_geo：生成答案 span 内 token logprob 的几何均值（exp(mean logp)）
- conf_token_first：答案 span 首 token 概率
- conf_verbalized：仅格式 C，模型自报 0–100 归一化到 [0,1]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor
from answer_spans import (extract_answer, extract_verbalized, score_generation,
                          SPAN_VERSION)

MODEL_PATHS = {
    "qwen": "assets/Qwen3-VL-8B-Instruct",
    "healthgpt": "assets/.runtime/modelscope/TianweiLin--HealthGPT-Pro-8B/TianweiLin/HealthGPT-Pro-8B",
    "gemma4": "assets/models_cache/google/gemma-4-E4B-it",
}
if os.environ.get("GEMMA_MODEL_PATH"):
    MODEL_PATHS["gemma4"] = os.environ["GEMMA_MODEL_PATH"]
MAX_PIXELS = 448 * 448
MIN_FREE_GIB = 16.5

FORMATS = {
    "A": {
        "instr": "Answer the question with a short answer only.",
        "marker": None, "end_marker": None,
    },
    "B": {
        "instr": ("First describe in one short sentence what you see in the image that is relevant "
                  "to the question. Then give your final answer on a new line in the exact "
                  "format 'Answer: <answer>'."),
        "marker": "Answer:", "end_marker": None,
    },
    "C": {
        "instr": ("Answer the question with a short answer on a new line in the exact format "
                  "'Answer: <answer>'. Then on a new line state how confident you are that this "
                  "answer is correct as a number from 0 to 100 in the exact format "
                  "'Confidence: <number>'."),
        "marker": "Answer:", "end_marker": "Confidence:",
    },
}
SYS = "You are a medical imaging assistant. Answer the question based on the image."

VARIANTS = {
    "A2": {"instr": "Give the answer directly. Use a short phrase.", "marker": None, "end_marker": None},
    "A3": {"instr": "Respond with the answer in a few words. No explanation.", "marker": None, "end_marker": None},
    "B2": {"instr": "Briefly state what you observe in one sentence, then write 'Answer: ' followed by the final answer on a new line.", "marker": "Answer:", "end_marker": None},
    "B3": {"instr": "First, in one short sentence, describe the relevant image finding. On a new line, provide your final answer as 'Answer: <answer>'.", "marker": "Answer:", "end_marker": None},
    "C2": {"instr": "Write 'Answer: <answer>' on one line, then on the next line write 'Confidence: <0-100>' indicating how likely your answer is correct.", "marker": "Answer:", "end_marker": "Confidence:"},
    "C3": {"instr": "State your answer on its own line as 'Answer: <answer>'. Then state a confidence score from 0 to 100 as 'Confidence: <number>'.", "marker": "Answer:", "end_marker": "Confidence:"},
}
FORMATS.update(VARIANTS)

YES = {"yes", "y", "true", "present", "positive"}
NO = {"no", "n", "false", "absent", "negative", "none"}
MODALITY = {"ct": "ct", "computed tomography": "ct", "mri": "mri", "magnetic resonance": "mri",
            "x-ray": "xray", "xray": "xray", "radiograph": "xray", "ultrasound": "ultrasound"}
ORGAN_ALIAS = {"lungs": "lung", "brains": "brain", "livers": "liver", "the heart": "heart"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize(ans: str, cat: str) -> str:
    a = (ans or "").strip().lower().strip(".\"'` ")
    a = re.sub(r"^(the|a|an)\s+", "", a)
    if cat == "abnormality":
        if a in YES:
            return "yes"
        if a in NO:
            return "no"
        m = re.match(r"^(yes|no)\b", a) or re.search(r"\b(yes|no)\b", a)
        return m.group(1) if m else a[:40]
    if cat == "modality":
        for k, v in sorted(MODALITY.items(), key=lambda x: -len(x[0])):
            if k in a:
                return v
        return a[:40]
    if cat == "vqamed_modality":
        table = [("ct angiography", "ct"), ("cta", "ct"), ("ct venography", "ct"),
                 ("ct", "ct"), ("mri", "mri"), ("magnetic resonance", "mri"),
                 ("mra", "mri"), ("doppler ultrasound", "ultrasound"),
                 ("ultrasound", "ultrasound"), ("us", "ultrasound"),
                 ("x-ray", "xray"), ("radiograph", "xray"), ("xray", "xray"),
                 ("pet", "pet"), ("contrast", "contrast"), ("iv", "contrast"),
                 ("iodine", "contrast")]
        for k, v in table:
            if a.startswith(k) or a == k:
                return v
        return a[:40]
    if cat == "plane":
        a = "axial" if a in ("transverse", "ax") else a
        return a
    if cat == "organ":
        return ORGAN_ALIAS.get(a, a.rstrip("s") if len(a) > 4 and a.endswith("s") else a)
    return a


def span_conf(scores, gen_ids, processor, fmt):
    """兼容旧调用入口；所有答案跨度规则由 answer_spans 统一实现。"""
    result = score_generation(scores, gen_ids, processor, fmt)
    return result["conf_token_geo"], result["conf_token_first"], result["span_len"]


def load_model(path: str):
    from transformers import AutoProcessor
    import transformers as _tf
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    if hasattr(processor, "image_processor"):
        try:
            processor.image_processor.max_pixels = MAX_PIXELS
            processor.image_processor.min_pixels = 4 * 28 * 28
        except Exception as e:
            log(f"pixel warn: {e}")
    if "gemma" in path.lower():
        cls = getattr(_tf, "AutoModelForMultimodalLM", None) or _tf.Gemma4ForConditionalGeneration
    else:
        cls = _tf.Qwen3VLForConditionalGeneration
    free, total = torch.cuda.mem_get_info(0)
    log(f"GPU free {free/2**30:.1f}/{total/2**30:.1f} GiB")
    if free / 2**30 < MIN_FREE_GIB:
        raise RuntimeError(f"insufficient GPU memory: {free/2**30:.1f} GiB < {MIN_FREE_GIB}")
    model = cls.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                device_map="cuda:0", local_files_only=True)
    model.eval()
    return processor, model


def validate_regeneration_jobs(records):
    """jobs 入口只接受明确缺失或截断的重生成任务。"""
    for row in records:
        if row.get("action") != "regenerate" or row.get("teacher_forcing") is True:
            raise ValueError("jobs-jsonl 仅接受显式 regenerate；完整原输出必须执行 teacher forcing")
        for key in ("saved_raw", "original_output_raw"):
            if key in row and row.get("raw") is not None and row[key] != row["raw"]:
                raise ValueError("重生成任务的历史 raw 字段不一致")
        raw = row.get("saved_raw", row.get("raw"))
        known_cut = (row.get("model") == "gemma4" and row.get("qid") == "slake:validate:10458"
                     and row.get("format", row.get("g")) == "A" and isinstance(raw, str) and len(raw) == 600)
        if (isinstance(raw, str) and raw.strip() and not known_cut
                and row.get("raw_truncated") is not True and row.get("raw_complete") is not False):
            raise ValueError("完整已存 raw 不得作为 regenerate 任务")
        if row.get("format", row.get("g")) not in FORMATS:
            raise ValueError("任务包含未知 format")
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PATHS), required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--seed")
    source.add_argument("--jobs-jsonl", help="原记录或待重新推理记录；逐条使用其 format")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--formats", default=None, help="逗号分隔，如 A,B,C 或 A2,B2")
    args = ap.parse_args()

    if args.jobs_jsonl:
        records = [json.loads(l) for l in Path(args.jobs_jsonl).read_text().splitlines() if l.strip()]
        records = [q for q in records if q.get("model", args.model) == args.model]
        validate_regeneration_jobs(records)
        jobs = [(q, q.get("format", q.get("g"))) for q in records]
    else:
        questions = [json.loads(l) for l in Path(args.seed).read_text().splitlines() if l.strip()]
        fmts = args.formats.split(",") if args.formats else ["A", "B", "C"]
        jobs = [(q, fmt) for q in questions for fmt in fmts]
    if any(fmt not in FORMATS for _, fmt in jobs):
        raise ValueError("任务包含未知 format")
    if args.limit:
        jobs = jobs[: args.limit]

    out_path = Path(args.out)
    protected = Path(__file__).resolve().parents[1] / "results"
    if out_path.resolve().parent.name in {"raw", "raw_vqarad", "raw_vqamed", "raw_wording"} and out_path.resolve().parent.parent == protected:
        raise ValueError("修复运行不得覆盖历史 raw；请写入 results/corrected 下")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                if (r.get("status") == "ok" and r.get("span_version") == SPAN_VERSION
                        and r.get("recovery_status") == "regenerated_with_token_evidence"
                        and isinstance(r.get("gen_ids"), list)):
                    done.add(f"{r['model']}|{r['qid']}|{r['format']}")
            except Exception:
                continue
    jobs = [(q, f) for q, f in jobs if f"{args.model}|{q['qid']}|{f}" not in done]
    log(f"model={args.model} jobs={len(jobs)} (done={len(done)})")

    if not jobs:
        return
    processor, model = load_model(MODEL_PATHS[args.model])
    log("model loaded")

    n_ok = n_err = 0
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, (q, fmt) in enumerate(jobs, 1):
            rec = {"model": args.model, "qid": q["qid"], "category": q["category"],
                   "gold": q["gold"], "question": q["question"], "format": fmt,
                   "base_format": fmt[0], "image_path": q["image_path"],
                   "source_raw_file": q.get("source_raw_file"),
                   "original_raw": q.get("raw"),
                   "saved_raw": q.get("saved_raw", q.get("raw")),
                   "original_output_raw": q.get("original_output_raw", q.get("raw")),
                   "original_token_identity_verified": False,
                   "raw_text_integrity": q.get("raw_text_integrity", {
                       "status": "missing" if not q.get("raw") else "truncation_suspected",
                       "basis": "new_generation" if not args.jobs_jsonl else "validated_regeneration_job",
                       "original_completeness_proven": False}),
                   "prompt_system": SYS,
                   "prompt_user_text": f"{FORMATS[fmt]['instr']}\n\nQuestion: {q['question']}",
                   "dataset": q.get("dataset", "main"),
                   "schema_version": "saved-output-tf-v2-provisional",
                   "formal_analysis_eligible": False,
                   "action": "regenerate",
                   "prompt_instruction": FORMATS[fmt]["instr"],
                   "system_prompt": SYS,
                   "prompt_text": f"{FORMATS[fmt]['instr']}\n\nQuestion: {q['question']}"}
            try:
                image = Image.open(q["image_path"]).convert("RGB")
                instr = FORMATS[fmt]["instr"]
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": SYS}]},
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"{instr}\n\nQuestion: {q['question']}"}]},
                ]
                kw = dict(tokenize=True, return_dict=True, return_tensors="pt",
                          add_generation_prompt=True)
                if args.model == "gemma4":
                    kw["enable_thinking"] = False
                inputs = processor.apply_chat_template(messages, **kw).to("cuda:0")
                with torch.inference_mode():
                    gen = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                         output_scores=True, return_dict_in_generate=True)
                gen_ids = gen.sequences[0, inputs["input_ids"].shape[-1]:].tolist()
                evidence = score_generation(gen.scores, gen_ids, processor, fmt)
                ans_norm = normalize(evidence["ans_raw"], q["category"])
                gold_n = normalize(q["gold"], q["category"])
                rec.update(evidence)
                rec.update({
                    "ans_norm": ans_norm, "gold_norm": gold_n,
                    "correct": bool(ans_norm) and ans_norm == gold_n,
                    "status": "ok", "error": None,
                    "formal_analysis_eligible": evidence["span_scoreable"],
                    "generation_protocol": "greedy_actual_prompt_regeneration",
                    "teacher_forcing": False,
                    "teacher_forcing_verified": False,
                    "recovery_status": "regenerated_with_token_evidence",
                    "model_path": MODEL_PATHS[args.model],
                    "generation_config": {"max_new_tokens": 128, "do_sample": False},
                })
                n_ok += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rec.update({"raw": "", "status": "error", "error": "cuda_oom"})
                n_err += 1
            except Exception as e:  # noqa: BLE001
                rec.update({"raw": "", "status": "error", "error": f"{type(e).__name__}: {e}"})
                n_err += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(jobs):
                el = time.time() - t0
                log(f"{i}/{len(jobs)} ok={n_ok} err={n_err} avg={el/i:.1f}s "
                    f"eta={(el/i)*(len(jobs)-i)/60:.0f}min")
    log(f"FINISHED {args.model}: ok={n_ok} err={n_err}")


if __name__ == "__main__":
    main()
