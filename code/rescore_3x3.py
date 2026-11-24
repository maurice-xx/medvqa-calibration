#!/usr/bin/env python3
"""固定完整答案的 assistant-prefill 3×3 重评分；旧 user-side 结果仅保留为 legacy。"""
import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from answer_spans import text_span, score_generation
from run_calib import normalize

ROOT = Path(".")
MODEL_PATHS = {
    "qwen": "assets/Qwen3-VL-8B-Instruct",
    "healthgpt": "assets/.runtime/modelscope/TianweiLin--HealthGPT-Pro-8B/TianweiLin/HealthGPT-Pro-8B",
    "gemma4": "assets/models_cache/google/gemma-4-E4B-it",
}
for _name in MODEL_PATHS:
    MODEL_PATHS[_name] = os.environ.get(f"{_name.upper()}_MODEL_PATH", MODEL_PATHS[_name])
BASE_FILES = {"qwen": "qwen500", "healthgpt": "healthgpt500", "gemma4": "gemma500"}
SYS = "You are a medical imaging assistant. Answer the question based on the image."
NEUTRAL_INSTR = "Answer the question based on the image."
SUFFIX = {
    "A": "",
    "B": "Observation: The image shows a medical scan.\nAnswer: ",
    "C": "Answer: ",
}
APRIME = ("Answer the question with a short answer on a new line in the exact format "
          "'Answer: <answer>'.")
FORMAT_SET = ("A", "B", "C")
PROTOCOL = "assistant_prefill_v2"
MAX_PIXELS = 448 * 448
MAX_PROMPT_TOKENS = 8192
MIN_FREE_GIB = 15.5
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def jsonable(value):
    return json.loads(json.dumps(value, default=str))


def validate_seed(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    seen = set()
    for row in rows:
        for key in ("qid", "question", "image_path", "gold", "category"):
            if not row.get(key):
                raise ValueError(f"seed 缺字段 {key}: {row}")
        if row["qid"] in seen:
            raise ValueError(f"重复qid: {row['qid']}")
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
        seen.add(row["qid"])
    return rows


def load_base_answers(path, allow_truncated=False):
    """完整生成文本优先；不沿用已截断 ans_raw 或旧 correct。"""
    answers = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        fmt = row.get("format")
        if row.get("status") != "ok" or fmt not in FORMAT_SET:
            continue
        raw = row.get("raw_full", row.get("raw", ""))
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"缺少完整生成文本: {row['qid']} {fmt}")
        extracted = text_span(raw, fmt)
        answer = extracted["ans_raw"]
        if not answer:
            raise ValueError(f"空提取答案: {row['qid']} {fmt}")
        truncated = len(raw) == 600 and not row.get("raw_full")
        if truncated and not allow_truncated:
            raise ValueError(f"旧raw达到600字符存储上限，需恢复: {row['qid']} {fmt}")
        ans_norm = normalize(answer, row["category"])
        gold_norm = normalize(row["gold"], row["category"])
        key = (row["qid"], fmt)
        if fmt in answers.get(row["qid"], {}):
            raise ValueError(f"重复答案: {key}")
        answers.setdefault(row["qid"], {})[fmt] = {
            "answer": answer, "ans_norm": ans_norm, "gold_norm": gold_norm,
            "correct": bool(ans_norm) and ans_norm == gold_norm,
            "answer_source": "raw_full" if row.get("raw_full") else "raw",
            "source_raw_truncated": truncated,
            "source_raw_sha256": sha256(raw), "extraction": extracted,
            "source_answer_changed": answer != row.get("ans_raw"),
            "source_label_changed": (bool(ans_norm) and ans_norm == gold_norm) != row.get("correct"),
            "source_raw_complete_verified": not truncated,
            "source_recovery_status": row.get("recovery_status"),
            "source_recovery_reason": row.get("recovery_reason"),
        }
    return answers


def load_model(name):
    import transformers as tf
    path = MODEL_PATHS[name]
    proc = AutoProcessor.from_pretrained(path, local_files_only=True)
    if hasattr(proc, "image_processor"):
        # 保留主生成管线原设置；实际处理配置及输出张量另外完整记录。
        proc.image_processor.max_pixels = MAX_PIXELS
        proc.image_processor.min_pixels = 4 * 28 * 28
    cls = tf.Gemma4ForConditionalGeneration if name == "gemma4" else tf.Qwen3VLForConditionalGeneration
    if torch.cuda.device_count() != 2:
        raise RuntimeError("本执行协议要求CUDA_VISIBLE_DEVICES=0,1，仅使用两卡")
    gib = 2 ** 30
    budgets, available = {}, []
    for index, cap in enumerate((10, 9)):
        free, total = torch.cuda.mem_get_info(index)
        budget = min(free - 2 * gib, cap * gib)
        if budget <= 0:
            raise RuntimeError(f"GPU{index}真实空闲不足2GiB: {free / gib:.3f}GiB")
        budgets[index] = int(budget)
        available.append({"device": index, "free_bytes": free, "total_bytes": total,
                          "max_memory_bytes": budgets[index]})
    budgets["cpu"] = 24 * gib
    log(f"BF16自动分片预算: {budgets}; 实际空闲: {available}")
    started = time.time()
    model = cls.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                device_map="auto", max_memory=budgets,
                                offload_state_dict=True, local_files_only=True)
    model.eval()
    if any(str(p.dtype) != "torch.bfloat16" for p in model.parameters() if p.is_floating_point()):
        raise RuntimeError("模型包含非BF16浮点权重，拒绝静默改变精度")
    model._rescore_execution = {
        "device_map": jsonable(model.hf_device_map), "max_memory_bytes": jsonable(budgets),
        "gpu_memory_before_load": available, "cpu_backup_bytes": 24 * gib,
        "load_elapsed_s": time.time() - started, "quantization": None,
        "dtype": "torch.bfloat16", "device_map_strategy": "auto",
    }
    log(f"模型设备映射: {model._rescore_execution}")
    return proc, model


def input_device(model):
    embedding = model.get_input_embeddings()
    hook = getattr(embedding, "_hf_hook", None)
    execution = getattr(hook, "execution_device", None)
    return torch.device(execution) if execution is not None else embedding.weight.device


def score_encoded(model, enc, positions, token_ids):
    # 使用整数尾部窗口，避免accelerate把索引tensor搬到第一卡而末层位于第二卡。
    sequence_len = enc["input_ids"].shape[-1]
    window_start = min(positions) - 1
    keep = sequence_len - window_start
    enc = enc.to(input_device(model))
    with torch.inference_mode():
        out = model(**enc, logits_to_keep=keep, use_cache=False)
        indices = torch.tensor([p - 1 - window_start for p in positions], device=out.logits.device)
        logits = out.logits[0].index_select(0, indices).float()
        target = torch.tensor(token_ids, device=logits.device)
        lps = logits.gather(1, target[:, None]).squeeze(1) - torch.logsumexp(logits, dim=-1)
        logps = lps.cpu().tolist()
    if len(logps) != len(positions) or not all(math.isfinite(v) for v in logps):
        raise ValueError("输出logprob无效")
    return logps


def build_messages(question, instr, image):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYS}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"{instr}\n\nQuestion: {question}"}]},
    ]


def build_text(proc, question, instr, name, image):
    kw = dict(tokenize=False, add_generation_prompt=True)
    if name == "gemma4":
        kw["enable_thinking"] = False
    return proc.apply_chat_template(build_messages(question, instr, image), **kw)


def tensor_metadata(enc, image):
    result = {"original_size": list(image.size), "mode": image.mode,
              "tensor_shapes": {k: list(v.shape) for k, v in enc.items() if hasattr(v, "shape")}}
    if "image_grid_thw" in enc:
        result["image_grid_thw"] = enc["image_grid_thw"].tolist()
    if "image_position_ids" in enc:
        values = enc["image_position_ids"].tolist()
        result["image_position_ids_sha256"] = sha256(json.dumps(values))
        result["image_position_ids_bounds"] = [enc["image_position_ids"].amin().item(),
                                                 enc["image_position_ids"].amax().item()]
    return result


def prepare_score(proc, image, question, answer, s, name):
    """整串token化，按offset定位答案，显式验证图像展开后的token对应。"""
    base = build_text(proc, question, NEUTRAL_INSTR, name, image)
    prompt = base + SUFFIX[s]
    full = prompt + answer
    enc = proc(text=full, images=[image], return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    txt = proc.tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    text_ids, offsets = txt["input_ids"], txt["offset_mapping"]
    answer_text_indices = [i for i, (start, end) in enumerate(offsets)
                           if end > len(prompt) and start < len(full)]
    if not answer_text_indices:
        raise ValueError("答案没有对应token")
    first = answer_text_indices[0]
    # 图像占位符的token扩展发生在答案前，答案后缀必须逐token完全相同。
    tail = text_ids[first:]
    delta = len(ids) - len(text_ids)
    start = first + delta
    if start < 1 or ids[start:] != tail:
        raise ValueError("完整文本与多模态processor答案后缀不一致")
    special = set(proc.tokenizer.all_special_ids)
    positions = [i + delta for i in answer_text_indices if text_ids[i] not in special]
    if not positions:
        raise ValueError("排除special后答案为空")
    if start > MAX_PROMPT_TOKENS:
        raise ValueError(f"提示过长: {start}")
    prefix_indices = [i for i, (a, b) in enumerate(offsets)
                      if b > len(base) and a < len(prompt) and text_ids[i] not in special
                      and i not in answer_text_indices]
    crossing = [{"id": text_ids[i], "offset": list(offsets[i])}
                for i in answer_text_indices if offsets[i][0] < len(prompt)]
    non_special_prefix = [text_ids[i] for i in prefix_indices]
    meta = {
        "protocol": PROTOCOL, "assistant_prefill": SUFFIX[s], "prompt_text": prompt,
        "prompt_sha256": sha256(prompt), "base_prompt_sha256": sha256(base),
        "prompt_token_ids": ids[:positions[0]], "prompt_len": positions[0],
        "effective_prompt_decoded": proc.tokenizer.decode(ids[:positions[0]], skip_special_tokens=False),
        "token_boundary_policy": "full_text_tokenization_include_answer_overlapping_non_special_tokens",
        "prefix_token_ids": non_special_prefix,
        "prefix_standalone_token_ids": proc.tokenizer.encode(SUFFIX[s], add_special_tokens=False),
        "prefix_token_positions": [i + delta for i in prefix_indices],
        "answer_token_ids": [ids[i] for i in positions], "scored_token_ids": [ids[i] for i in positions],
        "scored_positions": positions, "answer_char_range": [len(prompt), len(full)],
        "answer_boundary_crossing_tokens": crossing,
        "special_token_ids": sorted(special),
        "excluded_special_positions": [i + delta for i in answer_text_indices if text_ids[i] in special],
        "image_settings": tensor_metadata(enc, image),
    }
    return enc, meta


def score(proc, model, image, question, answer, s, name):
    enc, meta = prepare_score(proc, image, question, answer, s, name)
    positions = meta["scored_positions"]
    logps = score_encoded(model, enc, positions, meta["answer_token_ids"])
    if len(logps) != len(positions) or not all(math.isfinite(v) for v in logps):
        raise ValueError("答案logprob无效")
    meta.update(conf=math.exp(sum(logps) / len(logps)), span=len(logps),
                conf_first=math.exp(logps[0]), span_token_logprobs=logps,
                status="ok", reason="ok")
    return meta


def safe_score(*args):
    for attempt in range(2):
        try:
            return score(*args)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt:
                return {"conf": None, "span": 0, "status": "fail", "reason": "cuda_oom"}
        except Exception as exc:
            return {"conf": None, "span": 0, "status": "fail",
                    "reason": f"{type(exc).__name__}:{exc}"}


def provenance(proc, model, args, source):
    import transformers
    path = Path(MODEL_PATHS[args.model])
    hashes = {p.name: sha256(p.read_bytes()) for p in sorted(path.iterdir())
              if p.is_file() and (p.name.endswith("config.json") or p.name == "chat_template.jinja")}
    checkpoint_inventory = [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                            for p in sorted(path.glob("*.safetensors"))]
    code_files = [Path(__file__), Path(__file__).with_name("answer_spans.py"),
                  Path(__file__).with_name("run_calib.py")]
    return {
        "protocol": PROTOCOL, "model": args.model, "checkpoint_path": str(path),
        "checkpoint_config_sha256": sha256((path / "config.json").read_bytes()),
        "checkpoint_file_hashes": hashes,
        "checkpoint_weight_inventory": checkpoint_inventory,
        "code_sha256": CODE_SHA256,
        "code_file_hashes": {p.name: sha256(p.read_bytes()) for p in code_files},
        "source_path": str(source), "source_file_sha256": sha256(source.read_bytes()),
        "seed_file_sha256": sha256(Path(args.seed).read_bytes()),
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_dtype": str(next(model.parameters()).dtype),
        "execution": getattr(model, "_rescore_execution", {}),
        "image_processor_config": jsonable(proc.image_processor.to_dict()),
        "image_pixel_settings_requested": {"max_pixels": MAX_PIXELS, "min_pixels": 4 * 28 * 28},
        "generation_parameters": {"max_new_tokens": 128, "do_sample": False},
        "score_forward_parameters": {"use_cache": False, "logits_to_keep": "answer_positions_minus_one"},
        "suffixes": SUFFIX, "neutral_instruction": NEUTRAL_INSTR,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def generate_aprime(proc, model, questions, name, prov, out_path):
    """与run_calib主生成管线相同，仅替换用户指令；不作label-only因果解释。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("code_sha256") != prov["code_sha256"] or row.get("status") != "ok":
                raise ValueError("已有A_prime记录不兼容，拒绝静默混合")
            if row["qid"] in done:
                raise ValueError("A_prime重复qid")
            done.add(row["qid"])
    with out_path.open("a", encoding="utf-8") as writer:
        for i, q in enumerate(questions, 1):
            if q["qid"] in done:
                continue
            image = Image.open(q["image_path"]).convert("RGB")
            kw = dict(tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)
            if name == "gemma4":
                kw["enable_thinking"] = False
            inputs = proc.apply_chat_template(build_messages(q["question"], APRIME, image), **kw)
            image_meta = tensor_metadata(inputs, image)
            prompt_ids = inputs["input_ids"][0].tolist()
            inputs = inputs.to("cuda:0")
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                     output_scores=True, return_dict_in_generate=True)
            gen_ids = gen.sequences[0, len(prompt_ids):].tolist()
            span_meta = score_generation(gen.scores, gen_ids, proc, "B")
            raw = span_meta["raw"]
            answer = span_meta["ans_raw"]
            ans_norm = normalize(answer, q["category"])
            gold_norm = normalize(q["gold"], q["category"])
            logps = span_meta["gen_token_logprobs"]
            prompt_text = build_text(proc, q["question"], APRIME, name, image)
            record = dict(prov, qid=q["qid"], question=q["question"], gold=q["gold"],
                          category=q["category"], image_path=q["image_path"],
                          format="A_prime", raw=raw, raw_full=raw, ans_raw=answer,
                          ans_norm=ans_norm, gold_norm=gold_norm,
                          correct=bool(ans_norm) and ans_norm == gold_norm,
                          generation_token_ids=gen_ids, generation_token_logprobs=logps,
                          prompt_token_ids=prompt_ids, prompt_sha256=sha256(prompt_text),
                          image_settings=image_meta, generation_instruction=APRIME,
                          control_interpretation="instruction_variant_not_label_only_causal_control",
                          status="ok")
            record.update(span_meta)
            record["protocol"] = "aprime_same_generation_pipeline_v2"
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.flush()
            if i % 10 == 0 or i == 1:
                log(f"A_prime {name}: {i}/{len(questions)}")
    log(f"A_PRIME_FINISHED {name}: {len(questions)}题")


def read_done(path, prov):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            for key in ("protocol", "code_sha256", "source_file_sha256", "checkpoint_config_sha256"):
                if row.get(key) != prov.get(key):
                    raise ValueError(f"拒绝混合不同运行来源 {key}: {path}")
            if row.get("status") != "ok":
                continue
            key = (row["qid"], row["g"], row["s"])
            if key in done:
                raise ValueError(f"重复评分: {key}")
            done.add(key)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_PATHS))
    ap.add_argument("--seed", default=str(ROOT / "data/seed500.jsonl"))
    ap.add_argument("--source")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip-aprime", action="store_true", help="兼容旧命令；此脚本只重评分")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke-then-full", action="store_true")
    ap.add_argument("--with-aprime", action="store_true", help="矩阵完成后运行同生成管线A_prime")
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit必须为正数")
    qs = validate_seed(args.seed)
    if args.limit:
        qs = qs[:args.limit]
    source = Path(args.source) if args.source else ROOT / "results/raw" / f"{BASE_FILES[args.model]}.jsonl"
    if not source.exists():
        raise FileNotFoundError(f"必须先生成完整校正raw: {source}")
    source_hash = sha256(source.read_bytes())
    answers = load_base_answers(source, allow_truncated=True)
    if sha256(source.read_bytes()) != source_hash:
        raise ValueError("读取期间source变化，拒绝混合运行")
    complete = [q for q in qs if len(answers.get(q["qid"], {})) == 3]
    if len(complete) != len(qs):
        raise ValueError(f"完整三格式答案不足: {len(complete)}/{len(qs)}")
    log(f"{args.model}: {len(qs)}题, {len(qs) * 9}评分，source={source}")
    if args.dry_run:
        return
    proc, model = load_model(args.model)
    prov = provenance(proc, model, args, source)
    if prov["source_file_sha256"] != source_hash:
        raise ValueError("模型加载期间source变化，须重新运行")
    out_path = Path(args.out) if args.out else ROOT / "results/corrected/rescore" / f"{args.model}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = out_path.with_suffix(".manifest.json")
    if not manifest.exists():
        manifest.write_text(json.dumps(prov, ensure_ascii=False, indent=2) + "\n")
    done = read_done(out_path, prov)
    image_cache = {}

    def run_question(q, writer=None):
        path = q["image_path"]
        if path not in image_cache:
            image_cache[path] = Image.open(path).convert("RGB")
        image = image_cache[path]
        records = []
        for g in FORMAT_SET:
            a = answers[q["qid"]][g]
            for s in FORMAT_SET:
                if writer is not None and (q["qid"], g, s) in done:
                    continue
                if a.get("source_raw_truncated"):
                    result = {"conf": None, "span": 0, "status": "blocked",
                              "reason": "stored_raw_600_char_cap_requires_separate_regeneration_cohort"}
                else:
                    result = safe_score(proc, model, image, q["question"], a["answer"], s, args.model)
                record = dict(prov, qid=q["qid"], g=g, s=s, image_path=path,
                              question=q["question"], category=q["category"], gold=q["gold"], **a)
                record.update(result)
                if writer is not None:
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    writer.flush()
                records.append(record)
        return records

    if args.smoke_then_full:
        smoke_path = out_path.with_suffix(".smoke.jsonl")
        if smoke_path.exists():
            smoke = [json.loads(line) for line in smoke_path.read_text().splitlines()]
            if len(smoke) != 9 or any(row.get("code_sha256") != prov["code_sha256"]
                                      or row.get("source_file_sha256") != prov["source_file_sha256"]
                                      for row in smoke):
                raise ValueError("已有smoke来源不兼容")
        else:
            smoke = run_question(qs[0])
        for row in smoke:
            if row["status"] != "ok":
                raise RuntimeError(f"smoke失败: {row['reason']}")
            if set(row["answer_token_ids"]) & set(row["special_token_ids"]):
                raise RuntimeError("smoke答案含special")
            if row["span"] != len(row["span_token_logprobs"]):
                raise RuntimeError("smoke答案跨度不一致")
        if not smoke_path.exists():
            smoke_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in smoke))
        log(f"SMOKE PASS {args.model}: 9/9，已验证assistant prefill、答案token与special排除")
    reasons = Counter()
    count = 0
    with out_path.open("a", encoding="utf-8") as writer:
        for i, q in enumerate(qs, 1):
            for row in run_question(q, writer):
                reasons[row["reason"]] += 1
                count += 1
            if i % 10 == 0 or i == 1:
                log(f"rescore {i}/{len(qs)}题，本轮{count}格，{dict(reasons)}")
    final_done = read_done(out_path, prov)
    expected = len(qs) * 9
    blocked = sum(a.get("source_raw_truncated", False) * 3
                  for q in qs for a in answers[q["qid"]].values())
    summary = {"expected_cells": expected, "success_cells": len(final_done),
               "blocked_truncated_source_cells": blocked, "this_run_reasons": dict(reasons)}
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if len(final_done) + blocked != expected:
        raise RuntimeError(f"矩阵执行失败: 成功{len(final_done)}+已知截断阻塞{blocked}/{expected}")
    log(f"FINISHED {args.model}: {len(final_done)}格，阻塞{blocked}，{dict(reasons)}")
    if args.with_aprime:
        generate_aprime(proc, model, qs, args.model, prov,
                        ROOT / "results/corrected/raw_aprime" / f"{args.model}.jsonl")


if __name__ == "__main__":
    main()
