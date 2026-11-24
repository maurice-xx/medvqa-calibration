#!/usr/bin/env python3
"""原prompt对完整stored raw做teacher forcing，不重新生成或声称恢复历史token身份。"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch
from PIL import Image

import answer_spans as spans
import rescore_3x3 as rs
from run_calib import FORMATS, normalize

PROTOCOL = "original_prompt_fixed_full_output_tf_v1"
ROOT = rs.ROOT
BASE = rs.BASE_FILES
CODE = Path(__file__)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def source_paths(model, cohort):
    if cohort == "main":
        return [ROOT / "results/raw" / f"{BASE[model]}.jsonl"]
    if model == "gemma4":
        raise ValueError("Gemma没有外部队列")
    return [ROOT / f"results/{name}/{model}.jsonl"
            for name in ("raw_wording", "raw_vqarad", "raw_vqamed")]


def memory_snapshot():
    return [{"device": i, "free_bytes": torch.cuda.mem_get_info(i)[0],
             "allocated_bytes": torch.cuda.memory_allocated(i),
             "reserved_bytes": torch.cuda.memory_reserved(i),
             "peak_allocated_bytes": torch.cuda.max_memory_allocated(i),
             "peak_reserved_bytes": torch.cuda.max_memory_reserved(i)}
            for i in range(torch.cuda.device_count())]


def prepare_actual(proc, row, image, name):
    fmt, raw = row["format"], row.get("raw_full", row.get("raw", ""))
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("stored raw为空")
    if len(raw) == 600 and not row.get("raw_full"):
        raise ValueError("原raw达到600字符存储上限；仅允许另cohort定向再生成")
    prompt = rs.build_text(proc, row["question"], FORMATS[fmt]["instr"], name, image)
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)
    if name == "gemma4":
        kw["enable_thinking"] = False
    original = proc.apply_chat_template(
        rs.build_messages(row["question"], FORMATS[fmt]["instr"], image), **kw)
    full = prompt + raw
    enc = proc(text=full, images=[image], return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    text = proc.tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    text_ids, offsets = text["input_ids"], text["offset_mapping"]
    selected = [i for i, (a, b) in enumerate(offsets) if b > len(prompt) and a < len(full)]
    if not selected:
        raise ValueError("完整输出没有token")
    delta = len(ids) - len(text_ids)
    start = selected[0] + delta
    if start < 1 or ids[start:] != text_ids[selected[0]:]:
        raise ValueError("整串token与图像展开后输出token不一致")
    original_ids = original["input_ids"][0].tolist()
    if ids[:start] != original_ids:
        raise ValueError("完整stored输出改变原prompt token边界，拒绝静默重定义prompt")
    special = set(proc.tokenizer.all_special_ids)
    output_indices = [i for i in selected if text_ids[i] not in special]
    positions = [i + delta for i in output_indices]
    output_ids = [ids[i] for i in positions]
    decoded = proc.tokenizer.decode(output_ids, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    if decoded != raw:
        raise ValueError(f"完整stored raw无法无损重token化: {decoded!r} != {raw!r}")
    extraction = spans.text_span(raw, fmt)
    char_start = len(prompt) + extraction["span_char_start"]
    char_end = len(prompt) + extraction["span_char_end"]
    answer_local = [j for j, i in enumerate(output_indices)
                    if offsets[i][1] > char_start and offsets[i][0] < char_end]
    partial = []
    for i in output_indices:
        left, right = offsets[i]
        if right > char_start and left < char_end:
            before = full[left:char_start] if left < char_start else ""
            after = full[char_end:right] if right > char_end else ""
            if before.strip(' \t\r\n"') or after.strip(' \t\r\n"'):
                partial.append(i)
    if not answer_local:
        raise ValueError("最终答案跨度没有token")
    if partial:
        raise ValueError("答案跨度边界token包含答案外非空白文本")
    if start > rs.MAX_PROMPT_TOKENS:
        raise ValueError(f"prompt过长: {start}")
    return enc, {
        **extraction, "raw": raw, "raw_full": raw,
        "prompt_text": prompt, "prompt_sha256": rs.sha256(prompt),
        "prompt_token_ids": original_ids, "prompt_len": start,
        "original_prompt_token_ids_match": True,
        "full_output_token_ids": output_ids, "gen_ids": output_ids,
        "full_output_positions": positions, "span_token_indices": answer_local,
        "span_token_ids": [output_ids[i] for i in answer_local],
        "span_len": len(answer_local), "span_scoreable": True,
        "full_output_roundtrip_verified": True,
        "raw_sha256": rs.sha256(raw), "source_raw_sha256": rs.sha256(raw),
        "token_ids_source": "retokenized_stored_full_output",
        "historical_actual_generation_identity_verified": False,
        "historical_gen_token_ids_available": isinstance(row.get("gen_ids"), list),
        "source_raw_complete_verified": True,
        "completeness_basis": "stored_raw_below_600_char_cap_or_explicit_raw_full",
        "token_boundary_policy": "original_prompt_unchanged_full_text_offsets_include_answer_overlap",
        "image_settings": rs.tensor_metadata(enc, image),
    }


def score_actual(proc, model, row):
    started = time.time()
    image = Image.open(row["image_path"]).convert("RGB")
    enc, meta = prepare_actual(proc, row, image, row["model"])
    logps = rs.score_encoded(model, enc, meta["full_output_positions"], meta["full_output_token_ids"])
    answer_logps = [logps[i] for i in meta["span_token_indices"]]
    meta.update({
        "gen_token_logprobs": logps, "full_output_token_logprobs": logps,
        "span_token_logprobs": answer_logps,
        "conf_token_geo": math.exp(sum(answer_logps) / len(answer_logps)),
        "conf_token_first": math.exp(answer_logps[0]),
        "conf_verbalized": spans.extract_verbalized(meta["raw"], row["format"]),
        "score_elapsed_s": time.time() - started,
        "ans_norm": normalize(meta["ans_raw"], row["category"]),
        "gold_norm": normalize(row["gold"], row["category"]),
        "status": "ok", "reason": "ok",
    })
    meta["correct"] = bool(meta["ans_norm"]) and meta["ans_norm"] == meta["gold_norm"]
    return meta


def manifest(proc, model, args, paths):
    import transformers
    files = [CODE, Path(rs.__file__), Path(spans.__file__), Path(rs.__file__).with_name("run_calib.py")]
    path = Path(rs.MODEL_PATHS[args.model])
    return {
        "protocol": PROTOCOL, "model": args.model, "cohort": args.cohort,
        "teacher_forcing": True, "regeneration": False,
        "checkpoint_path": str(path), "dtype": "torch.bfloat16", "quantization": None,
        "checkpoint_config_sha256": rs.sha256((path / "config.json").read_bytes()),
        "checkpoint_files": [{"name": p.name, "bytes": p.stat().st_size,
                              "mtime_ns": p.stat().st_mtime_ns} for p in path.glob("*.safetensors")],
        "code_sha256": {p.name: rs.sha256(p.read_bytes()) for p in files},
        "source_sha256": {str(p): rs.sha256(p.read_bytes()) for p in paths},
        "execution": model._rescore_execution,
        "processor_config": rs.jsonable(proc.to_dict()),
        "image_processor_config": rs.jsonable(proc.image_processor.to_dict()),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "span_version": spans.SPAN_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def resumable_done(out, run):
    done = set()
    for record in read_rows(out) if out.exists() else []:
        if record.get("run_signature") != run:
            raise ValueError(f"拒绝混用不同来源或代码: {out}")
        key = (record["dataset"], record["qid"], record["format"])
        if record.get("status") == "ok":
            if key in done:
                raise ValueError(f"重复成功记录: {key}")
            done.add(key)
    return done


def smoke(proc, model, row, out, base):
    if out.exists():
        existing = json.loads(out.read_text())
        if existing.get("run_signature") != base["run_signature"] or not existing.get("pass"):
            raise ValueError("已有冒烟来源不兼容或失败")
        return
    before = memory_snapshot()
    first = score_actual(proc, model, row)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    second = score_actual(proc, model, row)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    maxdiff = max(abs(a - b) for a, b in zip(first["full_output_token_logprobs"], second["full_output_token_logprobs"]))
    passed = (first["full_output_token_ids"] == second["full_output_token_ids"]
              and first["prompt_token_ids"] == second["prompt_token_ids"] and maxdiff <= 1e-5)
    record = {**base, "qid": row["qid"], "format": row["format"],
              "first": first, "second": second,
              "max_abs_logprob_diff": maxdiff,
              "abs_confidence_diff": abs(first["conf_token_geo"] - second["conf_token_geo"]),
              "memory_before": before, "memory_after": memory_snapshot(), "pass": passed}
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    rs.log(f"FULL_OUTPUT_TF_SMOKE pass={passed} max_abs_logprob_diff={maxdiff} "
           f"elapsed={first['score_elapsed_s']:.3f}/{second['score_elapsed_s']:.3f}s output={out}")
    if not passed:
        raise RuntimeError("完整stored输出双次TF冒烟不一致")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(BASE))
    parser.add_argument("--cohort", default="main", choices=("main", "external"))
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-then-full", action="store_true")
    args = parser.parse_args()
    paths = source_paths(args.model, args.cohort)
    source_hashes = {str(p): rs.sha256(p.read_bytes()) for p in paths}
    jobs, seen = [], set()
    for path in paths:
        for row in read_rows(path):
            if row.get("status") != "ok":
                raise ValueError(f"源raw非ok: {path} {row.get('qid')}")
            row = {**row, "model": args.model, "dataset": path.parent.name, "stored_source": str(path)}
            key = (row["dataset"], row["qid"], row["format"])
            if key in seen:
                raise ValueError(f"重复源raw: {key}")
            seen.add(key)
            jobs.append(row)
    if args.limit:
        jobs = jobs[:args.limit]
    out = Path(args.out) if args.out else ROOT / "results/corrected/actual_fixed_output_tf_v1" / args.cohort / f"{args.model}.jsonl"
    if "results/corrected" not in str(out.resolve()):
        raise ValueError("必须使用独立corrected目录")
    out.parent.mkdir(parents=True, exist_ok=True)
    proc, model = rs.load_model(args.model)
    prov = manifest(proc, model, args, paths)
    if prov["source_sha256"] != source_hashes:
        raise ValueError("模型加载期间源文件变化")
    stable = {k: prov[k] for k in ("protocol", "model", "cohort", "checkpoint_config_sha256", "code_sha256", "source_sha256", "span_version")}
    signature = rs.sha256(json.dumps(stable, sort_keys=True))
    prov["run_signature"] = signature
    manpath = out.with_suffix(".manifest.json")
    if manpath.exists():
        if json.loads(manpath.read_text()).get("run_signature") != signature:
            raise ValueError("已有manifest来源不兼容")
    else:
        manpath.write_text(json.dumps(prov, ensure_ascii=False, indent=2) + "\n")
    base = {"protocol": PROTOCOL, "model": args.model, "teacher_forcing": True,
            "regeneration": False, "run_signature": signature,
            "span_version": spans.SPAN_VERSION, "manifest_path": str(manpath)}
    if args.smoke_only or args.smoke_then_full:
        example = next((r for r in jobs if r["format"] == "B"), jobs[0])
        smoke(proc, model, example, out.with_suffix(".smoke.json"), base)
    if args.smoke_only:
        return
    done = resumable_done(out, signature)
    pending = [r for r in jobs if (r["dataset"], r["qid"], r["format"]) not in done]
    counts = Counter()
    rs.log(f"FIXED_OUTPUT_TF {args.model}/{args.cohort}: total={len(jobs)} done={len(done)} pending={len(pending)}")
    with out.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(pending, 1):
            identity = {k: row[k] for k in ("model", "qid", "format", "dataset", "stored_source", "question", "image_path", "category", "gold")}
            record = {**identity, **base}
            try:
                record.update(score_actual(proc, model, row))
            except Exception as error:
                record.update(status="blocked" if len(row.get("raw", "")) == 600 else "error",
                              reason=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                rs.log(f"TF失败 {row['qid']}/{row['format']}: {record['reason']}")
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            counts[record["status"]] += 1
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 25 == 0 or index == 1 or index == len(pending):
                rs.log(f"TF {args.model}/{args.cohort} {index}/{len(pending)} {dict(counts)}")
            if counts["error"] >= 3:
                raise RuntimeError("累计3条TF执行错误，停止以保留可恢复队列")
    final = resumable_done(out, signature)
    summary = {**base, "expected": len(jobs), "success": len(final),
               "pending_or_blocked": len(jobs) - len(final), "this_run": dict(counts),
               "gpu_memory_final": memory_snapshot()}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    rs.log(f"TF_FINISHED {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
