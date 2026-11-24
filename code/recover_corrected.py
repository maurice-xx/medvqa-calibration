#!/usr/bin/env python3
"""隔离历史候选概率，安排原输出 TF，并合并经验证的新证据；不加载模型。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from transformers import AutoTokenizer
from answer_spans import (SPAN_VERSION, answer_span, base_format, confidence_from_logps, decode_ids,
                          extract_verbalized, legacy_span_indices, text_span)
from run_calib import FORMATS, MODEL_PATHS, SYS, normalize

ROOT = Path(".")
BASE = {"qwen": "qwen500", "healthgpt": "healthgpt500", "gemma4": "gemma500"}
DATASET = {"raw": "main", "raw_vqarad": "vqarad", "raw_vqamed": "vqamed", "raw_wording": "wording"}
SCHEMA_VERSION = "saved-output-tf-v2-provisional"


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def raw_hash(raw):
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def finite_logps(values):
    return isinstance(values, list) and bool(values) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value <= 1e-6 for value in values)


def required_action(row):
    raw = row.get("raw")
    if not isinstance(raw, str) or not raw.strip():
        return "regenerate", "historical_raw_missing"
    known_cut = (row.get("model") == "gemma4" and
                 row.get("qid") == "slake:validate:10458" and
                 row.get("format") == "A" and len(raw) == 600)
    if known_cut or row.get("raw_truncated") is True or row.get("raw_complete") is False:
        return "regenerate", "historical_raw_truncated"
    return "teacher_force_saved_output", "original_generated_token_evidence_not_recorded"


def candidate_ids(raw, tokenizer, model):
    """只枚举可逆的首尾空白与停止 token；候选从不晋升正式概率。"""
    stop_ids = [1, 106, 50] if model == "gemma4" else [151645, 151643]
    seen = set()
    for prefix in ("", "\n", "\n\n", " "):
        for suffix in ("", "\n", "\n\n", " "):
            ids = tokenizer.encode(prefix + raw + suffix, add_special_tokens=False)
            if decode_ids(tokenizer, ids).strip() != raw.strip():
                continue
            for stop in (None, *stop_ids):
                candidate = ids + ([] if stop is None else [stop])
                signature = tuple(candidate)
                if signature in seen:
                    continue
                seen.add(signature)
                if len(candidate) <= 128 and decode_ids(tokenizer, candidate).strip() == raw.strip():
                    yield candidate


def output_integrity(row):
    action, reason = required_action(row)
    raw = row.get("raw")
    return action, {
        "status": "saved_output_available" if action == "teacher_force_saved_output" else (
            "missing" if reason == "historical_raw_missing" else "truncation_suspected"),
        "basis": reason if action == "regenerate" else "nonempty_saved_raw_no_known_truncation",
        "saved_character_count": len(raw) if isinstance(raw, str) else 0,
        "original_completeness_proven": False,
        "original_token_identity_verified": False,
    }


def provisional_tf(result, evidence, evidence_path):
    """完整保存待独立语义验收的 TF，候选视图不承载正式概率。"""
    merged = dict(result)
    previous = evidence.get("provisional_tf_evidence")
    merged["provisional_tf_evidence"] = previous or {
        "status": "provisional_requires_independent_semantic_acceptance",
        "evidence_file": evidence_path,
        "formal_analysis_eligible": False,
        "record": dict(evidence),
    }
    merged.update({"formal_analysis_eligible": False, "teacher_forcing": False,
                   "teacher_forcing_verified": False,
                   "confidence_evidence_status": "provisional_tf_quarantined",
                   "conf_token_geo": None, "conf_token_first": None,
                   "span_scoreable": False, "span_len": None,
                   "gen_ids": None, "gen_token_logprobs": None,
                   "span_token_ids": [], "span_token_indices": [], "span_token_logprobs": []})
    return merged


def repair_record(row, tokenizer, actual=None, previous=None):
    result = dict(row)
    raw, fmt = row.get("raw"), row["format"]
    raw_text = raw if isinstance(raw, str) else ""
    meta = text_span(raw_text, fmt)
    result.update(meta)
    result["base_format"] = base_format(fmt)
    result["ans_norm"] = normalize(meta["ans_raw"], row["category"])
    result["gold_norm"] = normalize(row["gold"], row["category"])
    result["correct"] = bool(result["ans_norm"]) and result["ans_norm"] == result["gold_norm"]
    result["conf_verbalized"] = extract_verbalized(raw_text, fmt)
    for key in ("conf_token_geo", "conf_token_first", "span_len", "ans_raw"):
        result["legacy_" + key] = row.get(key)
    result.update({
        "conf_token_geo": None, "conf_token_first": None, "span_len": None,
        "span_token_logprobs": [], "span_token_ids": [], "span_token_indices": [],
        "gen_ids": None, "gen_token_logprobs": None, "span_scoreable": False,
        "teacher_forcing": False, "regeneration": False,
        "original_generation_token_ids": None, "original_token_identity_verified": False,
        "schema_version": SCHEMA_VERSION,
        "confidence_evidence_status": "awaiting_real_teacher_forcing",
        "token_ids_source": "not_recorded", "legacy_candidate_conf": None,
        "historical_actual_generation_identity_verified": False,
        "raw_complete_verified": False, "teacher_forcing_verified": False,
        "formal_analysis_eligible": False,
    })
    action, reason = required_action(row)
    result.update({"action": action, "recovery_reason": reason,
                   "recovery_status": "needs_regeneration" if action == "regenerate" else "needs_teacher_forcing",
                   "raw_integrity_status": "missing_or_truncated" if action == "regenerate" else "saved_output_no_known_truncation",
                   "raw_sha256": raw_hash(raw_text),
                   "prompt_instruction": FORMATS[fmt]["instr"], "system_prompt": SYS,
                   "prompt_text": f"{FORMATS[fmt]['instr']}\n\nQuestion: {row['question']}"})
    result.update({"saved_raw": raw, "original_output_raw": raw,
                   "raw_text_integrity": output_integrity(row)[1],
                   "prompt_system": SYS, "prompt_user_text": result["prompt_text"]})
    if previous and (previous.get("provisional_tf_evidence") or previous.get("teacher_forcing") is True):
        result = provisional_tf(result, previous, previous.get("verified_evidence_file"))
    if action != "regenerate" and previous and previous.get("legacy_candidate_conf"):
        result["legacy_candidate_conf"] = previous["legacy_candidate_conf"]
        result["legacy_candidate_reason"] = previous.get("legacy_candidate_reason", "historical_actual_identity_unverified")
        result["recovery_evidence"] = previous.get("recovery_evidence")
        result["recovery_status"] = "diagnostic_candidate"
        return result
    if action == "regenerate":
        return result
    if not actual:
        result["legacy_candidate_reason"] = "no_stored_token_logprobs"
        return result
    logps = actual.get("span_token_logprobs")
    if not finite_logps(logps):
        result["legacy_candidate_reason"] = "invalid_or_missing_stored_token_logprobs"
        return result
    if actual.get("span_len") != len(logps) or row.get("span_len") != len(logps):
        result["legacy_candidate_reason"] = "stored_span_length_logprob_mismatch"
        return result
    comparisons = [
        (math.exp(sum(logps) / len(logps)), actual.get("conf_recomputed")),
        (math.exp(sum(logps) / len(logps)), row.get("conf_token_geo")),
        (math.exp(logps[0]), row.get("conf_token_first")),
    ]
    if any(not isinstance(value, (int, float)) or not math.isclose(calc, value, abs_tol=1e-6, rel_tol=2e-6)
           for calc, value in comparisons):
        result["legacy_candidate_reason"] = "legacy_probability_evidence_mismatch"
        return result
    matches, length_matches = [], 0
    for ids in candidate_ids(raw_text, tokenizer, row["model"]):
        old = legacy_span_indices(ids, tokenizer, fmt)
        if len(old) != len(logps):
            continue
        length_matches += 1
        current = answer_span(ids, tokenizer, fmt)
        if current["span_boundary_partial"] or not current["span_token_indices"]:
            continue
        lookup = dict(zip(old, logps))
        if any(index not in lookup for index in current["span_token_indices"]):
            continue
        selected = [lookup[index] for index in current["span_token_indices"]]
        signature = (tuple(current["span_token_ids"]), tuple(selected), current["ans_raw"])
        matches.append((signature, current, selected, old))
    result["legacy_candidate_length_matches"] = length_matches
    signatures = {match[0] for match in matches}
    if len(signatures) != 1:
        result["legacy_candidate_reason"] = "ambiguous_token_logprob_alignment" if signatures else "no_token_span_candidate"
        return result
    _, current, selected, old = matches[0]
    result["legacy_candidate_conf"] = {
        "conf_token_geo": math.exp(sum(selected) / len(selected)),
        "conf_token_first": math.exp(selected[0]),
        "span_token_logprobs": selected, "span_token_ids": current["span_token_ids"],
        "span_token_indices": current["span_token_indices"], "gen_ids": current["gen_ids"],
        "span_len": current["span_len"], "ans_raw": current["ans_raw"],
        "token_ids_source": "tokenizer_reconstruction_not_original_generation",
        "formal_analysis_eligible": False,
    }
    result["recovery_status"] = "diagnostic_candidate"
    result["legacy_candidate_reason"] = "historical_actual_identity_unverified"
    result["recovery_evidence"] = {
        "legacy_span_length": len(old), "stored_logprob_count": len(logps),
        "candidate_alignments": len(matches), "distinct_answer_alignments": 1,
        "old_confidence_matches_stored_rounded_logps": True,
        "historical_actual_raw_was_copied_from_stored": True,
        "original_generated_token_ids_recorded": False, "logprobs_decimal_precision": 6,
    }
    return result


def verified_tf(row, original, source, tokenizer):
    source_value = row.get("source_raw_file", row.get("stored_source", row.get("source")))
    if source_value is None or not str(source_value).endswith(source):
        return False
    if row.get("status") != "ok" or row.get("span_version") != SPAN_VERSION:
        return False
    protocol_verified = (row.get("protocol") == "original_prompt_fixed_full_output_tf_v1"
                         and row.get("original_prompt_token_ids_match") is True
                         and row.get("full_output_roundtrip_verified") is True
                         and row.get("source_raw_complete_verified") is True
                         and row.get("regeneration") is False)
    if row.get("teacher_forcing") is not True or not (
            row.get("teacher_forcing_verified") is True or protocol_verified):
        return False
    if row.get("raw") != original.get("raw"):
        return False
    if row.get("raw_sha256") and row["raw_sha256"] != raw_hash(original["raw"]):
        return False
    return verified_token_evidence(row, original, tokenizer)


def verified_token_evidence(row, original, tokenizer):
    ids, logps = row.get("gen_ids"), row.get("gen_token_logprobs")
    if (not isinstance(ids, list) or not ids or any(not isinstance(i, int) or i < 0 for i in ids)
            or not finite_logps(logps) or len(ids) != len(logps)):
        return False
    span = row.get("span_token_indices")
    if not isinstance(span, list) or any(not isinstance(i, int) or not 0 <= i < len(ids) for i in span):
        return False
    expected = answer_span(ids, tokenizer, original["format"])
    expected.update(confidence_from_logps(expected, logps))
    if expected["raw"] != original["raw"]:
        return False
    for key in ("ans_raw", "span_char_start", "span_char_end", "span_token_indices",
                "span_token_ids", "span_len", "span_scoreable", "span_boundary_partial"):
        if key == "span_boundary_partial" and key not in row:
            if expected[key]:
                return False
        elif row.get(key) != expected[key]:
            return False
    for key in ("conf_token_geo", "conf_token_first"):
        left, right = row.get(key), expected[key]
        if (left is None) != (right is None):
            return False
        if right is not None and (not isinstance(left, (int, float)) or
                                  not math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-9)):
            return False
    return True


def merge_verified(result, evidence, evidence_path, tokenizer=None):
    if tokenizer is not None:
        evidence = dict(evidence)
        span = answer_span(evidence["gen_ids"], tokenizer, result["format"])
        span.update(confidence_from_logps(span, evidence["gen_token_logprobs"]))
        evidence.update(span)
    # 原始题目、原始完整输出和具体措辞保留，新的 token 证据独立覆盖候选。
    protected = {key: result.get(key) for key in (
        "model", "qid", "format", "base_format", "category", "gold", "question", "image_path",
        "raw", "raw_sha256", "source_raw_file", "prompt_instruction", "prompt_text", "system_prompt")}
    merged = dict(result)
    merged.update(evidence)
    merged.update(protected)
    merged["ans_norm"] = normalize(merged["ans_raw"], merged["category"])
    merged["gold_norm"] = normalize(merged["gold"], merged["category"])
    merged["correct"] = bool(merged["ans_norm"]) and merged["ans_norm"] == merged["gold_norm"]
    merged["conf_verbalized"] = extract_verbalized(evidence.get("regenerated_raw", evidence["raw"]), merged["format"])
    merged.update({"teacher_forcing_verified": True,
                   "recovery_status": "teacher_forced_saved_output_verified",
                   "recovery_reason": None, "action": "none",
                   "verified_evidence_file": evidence_path,
                   "formal_analysis_eligible": bool(evidence.get("span_scoreable"))})
    return merged


def verified_regeneration(row, original, source, tokenizer):
    if required_action(original)[0] != "regenerate":
        return False
    source_value = row.get("source_raw_file", row.get("stored_source"))
    if source_value is None or not str(source_value).endswith(source):
        return False
    if (row.get("status") != "ok" or row.get("span_version") != SPAN_VERSION
            or row.get("recovery_status") != "regenerated_with_token_evidence"
            or row.get("teacher_forcing") is not False):
        return False
    generated_raw = row.get("regenerated_raw", row.get("raw"))
    if not isinstance(generated_raw, str) or not generated_raw.strip():
        return False
    expected = dict(original, raw=generated_raw)
    return verified_token_evidence(row, expected, tokenizer)


def merge_regeneration(result, evidence, evidence_path):
    merged = merge_verified(result, evidence, evidence_path)
    merged.update({"regenerated_raw": evidence.get("regenerated_raw", evidence["raw"]),
                   "scoring_raw_source": "regenerated_raw", "teacher_forcing": False,
                   "teacher_forcing_verified": False, "regeneration_verified": True,
                   "recovery_status": "regenerated_with_token_evidence"})
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(BASE), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--verified-tf", type=Path, action="append", default=[],
                        help="显式 TF 证据仅收录为 provisional；正式验收由独立 gate 负责")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[args.model], local_files_only=True)
    actual = {(r["qid"], r.get("format", r.get("g"))): r for r in read_rows(
        args.root / "results/rescore_actual" / f"{args.model}.jsonl")}
    # 不自动扫描仍在写入的 TF 目录，防止未验收概率进入候选正式字段。
    tf_paths = list(args.verified_tf)
    sources = [("raw", BASE[args.model])]
    if not args.main_only:
        sources += [(name, args.model) for name in ("raw_vqarad", "raw_vqamed", "raw_wording")]
    report = {
        "model": args.model, "span_version": SPAN_VERSION, "schema_version": SCHEMA_VERSION,
        "method": "CPU diagnostic candidates; TF evidence quarantined pending independent semantic acceptance",
        "historical_limitations": ["原始生成ids未保存，候选概率仅供诊断",
            "历史actual.raw由stored复制，候选不能证明历史生成身份",
            "完整raw只补算原输出概率；仅缺失或真截断输出重新生成"],
        "files": [], "pending_actions": [], "needs_regeneration": [],
    }
    evidence_rows = [(str(path), row) for path in dict.fromkeys(tf_paths) for row in read_rows(path)
                     if row.get("model") == args.model]
    for directory, filename in sources:
        source = args.root / "results" / directory / f"{filename}.jsonl"
        if not source.exists():
            continue
        source_rel = str(source.relative_to(args.root))
        output = args.root / "results/corrected" / directory / f"{filename}.jsonl"
        regenerated_path = args.root / "results/corrected/regenerated" / directory / f"{filename}.jsonl"
        candidates = evidence_rows + [(str(output), row) for row in read_rows(output)]
        regenerated = [(str(regenerated_path), row) for row in read_rows(regenerated_path)]
        regenerated += [(str(output), row) for row in read_rows(output)]
        by_key = {}
        for path, row in candidates:
            by_key.setdefault((row.get("qid"), row.get("format", row.get("g"))), []).append((path, row))
        rows, pending = [], []
        previous = {(r["qid"], r["format"]): r for r in read_rows(output)}
        for index, row in enumerate(read_rows(source)):
            key = (row["qid"], row["format"])
            repaired = repair_record(row, tokenizer, actual.get(key) if directory == "raw" else None, previous.get(key))
            repaired.update({"source_raw_file": source_rel, "dataset": DATASET[directory],
                             "source_row_index": index})
            for path, evidence in by_key.get(key, []):
                if verified_tf(evidence, row, source_rel, tokenizer):
                    repaired = provisional_tf(repaired, evidence, path)
                    break
            if repaired["action"] == "regenerate":
                for path, evidence in regenerated:
                    if (evidence.get("qid"), evidence.get("format")) != key:
                        continue
                    if verified_regeneration(evidence, row, source_rel, tokenizer):
                        repaired = merge_regeneration(repaired, evidence, path)
                        break
            rows.append(repaired)
            if repaired["action"] != "none":
                job = dict(repaired)
                job.pop("provisional_tf_evidence", None)
                job["provisional_tf_evidence_available"] = bool(repaired.get("provisional_tf_evidence"))
                pending.append(job)
                summary = {"model": args.model, "qid": row["qid"], "format": row["format"],
                           "source": source_rel, "source_raw_file": source_rel,
                           "action": repaired["action"], "reason": repaired["recovery_reason"]}
                report["pending_actions"].append(summary)
                if repaired["action"] == "regenerate":
                    report["needs_regeneration"].append(summary)
        dump_rows(output, rows)
        jobs_path = args.root / "results/corrected/jobs" / f"{directory}_{args.model}.jsonl"
        dump_rows(jobs_path, pending)
        item = {"source": source_rel, "output": str(output.relative_to(args.root)), "total": len(rows),
                "counts": dict(Counter(r["recovery_status"] for r in rows)),
                "action_counts": dict(Counter(r["action"] for r in rows)),
                "dataset": DATASET[directory],
                "legacy_candidate_count": sum(r.get("legacy_candidate_conf") is not None for r in rows),
                "provisional_tf_count": sum(r.get("provisional_tf_evidence") is not None for r in rows),
                "formal_confidence_count": sum(r.get("conf_token_geo") is not None for r in rows),
                "pending_reasons": dict(Counter(r["recovery_reason"] for r in rows if r.get("recovery_reason"))),
                "jobs": str(jobs_path.relative_to(args.root))}
        report["files"].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    report["action_counts"] = dict(Counter(r["action"] for r in report["pending_actions"]))
    manifest = args.root / "results/corrected" / f"recovery_manifest_{args.model}.json"
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    manifests = [json.loads(path.read_text()) for path in sorted(
        (args.root / "results/corrected").glob("recovery_manifest_*.json"))]
    combined = {key: report[key] for key in ("span_version", "schema_version", "method", "historical_limitations")}
    combined.update({"models": [item["model"] for item in manifests],
                     "files": [item for data in manifests for item in data["files"]],
                     "pending_actions": [item for data in manifests for item in data.get("pending_actions", [])],
                     "needs_regeneration": [item for data in manifests for item in data.get("needs_regeneration", [])]})
    combined["action_counts"] = dict(Counter(r["action"] for r in combined["pending_actions"]))
    temp = args.root / "results/corrected" / f"recovery_manifest.{args.model}.tmp"
    temp.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n")
    temp.replace(args.root / "results/corrected/recovery_manifest.json")


if __name__ == "__main__":
    main()
