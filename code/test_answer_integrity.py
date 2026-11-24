#!/usr/bin/env python3
"""只在CPU验证答案边界、候选隔离和TF合并门槛。"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from answer_spans import (answer_span, base_format, confidence_from_logps,
                          extract_verbalized, text_span)
from recover_corrected import (merge_regeneration, merge_verified, repair_record, required_action,
                               verified_regeneration, verified_tf)


class ByteTokenizer:
    all_special_ids = [106, 999]
    eos_token_id = 999
    pad_token_id = 999

    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        values = [self.pieces.get(i, b"") for i in ids
                  if not skip_special_tokens or i not in self.all_special_ids]
        return b"".join(values).decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checks = []

    def check(name, condition, details=None):
        checks.append({"name": name, "pass": bool(condition), "details": details})
        if not condition:
            raise AssertionError(name + ": " + repr(details))

    for fmt in ("", None, 1, "D", "A1", "C4", "BB", " C"):
        try:
            base_format(fmt)
        except ValueError:
            check("invalid_format_" + repr(fmt), True)
        else:
            check("invalid_format_" + repr(fmt), False)
    for fmt in ("A", "B", "C", "A2", "A3", "B2", "B3", "C2", "C3"):
        check("valid_format_" + fmt, base_format(fmt) == fmt[0])
    cases = [
        ("Answer: no\nConfidence: 1\nAnswer: yes\nConfidence: 97", "C", "yes"),
        ("Observation. Answer: no\nAnswer: CT", "B", "CT"),
        ("\" CT scan \"", "A3", "CT scan"),
        ("Answer: \nConfidence: 88", "C2", ""),
        ("Answer:", "B3", ""),
        ("CT\nConfidence: 91", "C", "CT"),
        ("Confidence: 12\nAnswer: CT\nConfidence: 89", "C3", "CT"),
        ("", "A", ""),
        ("Answer: 脑\nConfidence: 92", "C", "脑"),
    ]
    for raw, fmt, expected in cases:
        meta = text_span(raw, fmt)
        check("text_" + fmt + "_" + repr(raw), meta["ans_raw"] == expected, meta)
        check("text_offset_exact", raw[meta["span_char_start"]:meta["span_char_end"]] == expected)
    check("last_confidence_verbalized", extract_verbalized(cases[0][0], "C") == .97)
    byte_tok = ByteTokenizer({1: b"Answer: ", 2: b"\xe8", 3: b"\x84", 4: b"\x91",
                              5: b"\nConfidence: 92", 106: b"<end_of_turn>"})
    meta = answer_span([1, 2, 3, 4, 5, 106], byte_tok, "C")
    check("utf8_bytes_all_included", meta["span_token_indices"] == [1, 2, 3], meta)
    check("gemma106_filtered", meta["filtered_special_token_ids"] == [106] and 106 not in meta["span_token_ids"])
    score = confidence_from_logps(meta, [-.2] * 6)
    check("utf8_scoreable", score["span_scoreable"] and math.isclose(score["conf_token_geo"], math.exp(-.2)))
    for pieces, ids in [({1: b"Answer: CT", 2: b"\nConfidence: 90"}, [1, 2]),
                         ({1: b"Answer: ", 2: b"CT\nConfidence:", 3: b"90"}, [1, 2, 3])]:
        meta = answer_span(ids, ByteTokenizer(pieces), "C")
        score = confidence_from_logps(meta, [-.4] * len(ids))
        check("cross_marker_partial", meta["span_boundary_partial"] and not score["span_scoreable"], meta)
        check("cross_marker_formal_conf_null", score["conf_token_geo"] is None)
        check("cross_marker_sensitivity_available", score["sensitivity_conf_token_geo_inclusive"] is not None)
    meta = answer_span([1, 106], ByteTokenizer({1: b"Answer: \nConfidence: 90"}), "C")
    score = confidence_from_logps(meta, [-.2, -.3])
    check("empty_answer_no_tokens", meta["span_token_indices"] == [] and score["conf_token_geo"] is None)
    for values in ([-.2], [float("nan"), -.2], [1., -.2]):
        try:
            confidence_from_logps(answer_span([1, 2], ByteTokenizer({1: b"CT", 2: b" scan"}), "A"), values)
        except ValueError:
            check("invalid_logps_rejected", True)
        else:
            check("invalid_logps_rejected", False)

    from transformers import AutoTokenizer
    from run_calib import MODEL_PATHS
    from recover_corrected import BASE, ROOT, read_rows
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[args.model], local_files_only=True)
    legacy_actual = {(row['qid'], row.get('format', row.get('g'))): row for row in read_rows(
        ROOT / 'results/rescore_actual' / f'{args.model}.jsonl')}
    diagnostic = None
    for historical in read_rows(ROOT / 'results/raw' / f'{BASE[args.model]}.jsonl')[:12]:
        candidate = repair_record(historical, tokenizer, legacy_actual.get((historical['qid'], historical['format'])))
        if candidate.get('legacy_candidate_conf') is not None:
            diagnostic = candidate
            break
    check('actual_legacy_candidate_found', diagnostic is not None)
    check('actual_legacy_candidate_formal_conf_null', diagnostic['conf_token_geo'] is None and
          diagnostic['conf_token_first'] is None and diagnostic['gen_ids'] is None)
    check('actual_legacy_candidate_requires_TF', diagnostic['action'] == 'teacher_force_saved_output' and
          diagnostic['recovery_status'] == 'diagnostic_candidate' and not diagnostic['formal_analysis_eligible'])
    for raw, fmt, expected in cases:
        ids = tokenizer.encode(raw, add_special_tokens=False)
        meta = answer_span(ids, tokenizer, fmt)
        check("real_tokenizer_" + fmt + "_" + repr(raw), meta["ans_raw"] == expected, meta)
        check("real_special_filter", not set(meta["span_token_ids"]).intersection(tokenizer.all_special_ids))
    if args.model == "gemma4":
        ids = tokenizer.encode("CT", add_special_tokens=False) + [106]
        meta = answer_span(ids, tokenizer, "A")
        check("real_gemma106_filtered", 106 in meta["filtered_special_token_ids"] and 106 not in meta["span_token_ids"])
    original = {"model": args.model, "qid": "test", "format": "A", "raw": "CT", "status": "ok",
                "category": "modality", "gold": "CT", "question": "Which modality?", "span_len": 1,
                "conf_token_geo": .8, "conf_token_first": .8}
    repaired = repair_record(original, tokenizer)
    check("no_logps_requires_TF", repaired["action"] == "teacher_force_saved_output" and repaired["conf_token_geo"] is None)
    check("complete_long_raw_requires_TF", required_action(dict(original, raw="x" * 700))[0] == "teacher_force_saved_output")
    check("known_cut_regenerates", required_action(dict(original, model="gemma4", qid="slake:validate:10458", raw="x" * 600))[0] == "regenerate")
    check("missing_raw_regenerates", required_action(dict(original, raw=None))[0] == "regenerate")
    ids = tokenizer.encode("CT", add_special_tokens=False)
    evidence = answer_span(ids, tokenizer, "A")
    evidence.update(confidence_from_logps(evidence, [-.2] * len(ids)))
    evidence.update({"status": "ok", "teacher_forcing": True, "teacher_forcing_verified": True,
                     "gen_token_logprobs": [-.2] * len(ids), "source_raw_file": "results/raw/test.jsonl"})
    check("verified_TF_accepted", verified_tf(evidence, original, "results/raw/test.jsonl", tokenizer))
    protocol_evidence = dict(evidence)
    protocol_evidence.pop("teacher_forcing_verified")
    protocol_evidence.pop("span_boundary_partial")
    protocol_evidence.update({"protocol": "original_prompt_fixed_full_output_tf_v1", "regeneration": False,
                              "original_prompt_token_ids_match": True, "full_output_roundtrip_verified": True,
                              "source_raw_complete_verified": True})
    check("fixed_output_protocol_verified", verified_tf(protocol_evidence, original, "results/raw/test.jsonl", tokenizer))
    protocol_merged = merge_verified(dict(repaired, source_raw_file="results/raw/test.jsonl"), protocol_evidence, "fixed.jsonl", tokenizer)
    check("fixed_output_boundary_enriched", protocol_merged.get("span_boundary_partial") is False and
          protocol_merged.get("sensitivity_conf_token_geo_inclusive") is not None)
    for patch in ({"teacher_forcing_verified": False}, {"teacher_forcing": False}, {"raw": "CT "},
                  {"gen_ids": None}, {"gen_token_logprobs": []}, {"source_raw_file": "results/raw_wording/test.jsonl"},
                  {"source_raw_file": None}, {"span_token_indices": []}, {"conf_token_geo": .99}):
        invalid = dict(evidence, **patch)
        check("unverified_TF_rejected_" + str(patch), not verified_tf(invalid, original, "results/raw/test.jsonl", tokenizer))
    repaired["source_raw_file"] = "results/raw/test.jsonl"
    merged = merge_verified(repaired, dict(evidence, question="WRONG", raw="WRONG"), "tf.jsonl")
    check("merge_keeps_original_raw_and_prompt", merged["raw"] == original["raw"] and merged["question"] == original["question"])
    check("verified_over_candidate", merged["recovery_status"] == "teacher_forced_saved_output_verified" and merged["action"] == "none")
    cut_original = dict(original, model="gemma4", qid="slake:validate:10458", raw="x" * 600)
    regenerated = dict(evidence, teacher_forcing=False, recovery_status="regenerated_with_token_evidence")
    check("verified_regeneration_cut_only", verified_regeneration(regenerated, cut_original, "results/raw/test.jsonl", tokenizer))
    check("regeneration_not_replace_complete_raw", not verified_regeneration(regenerated, original, "results/raw/test.jsonl", tokenizer))
    cut_repaired = repair_record(cut_original, tokenizer)
    cut_repaired["source_raw_file"] = "results/raw/test.jsonl"
    cut_merged = merge_regeneration(cut_repaired, regenerated, "regenerated.jsonl")
    check("regeneration_preserves_original_raw", cut_merged["raw"] == cut_original["raw"] and cut_merged["regenerated_raw"] == "CT")
    check("regeneration_safe_rerun", verified_regeneration(cut_merged, cut_original, "results/raw/test.jsonl", tokenizer))
    report = {"model": args.model, "cpu_only": True, "tests": checks,
              "passed": len(checks), "failed": 0,
              "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("answer_spans.py", "recover_corrected.py", "run_calib.py", "rescore_actual.py")}}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"model": args.model, "passed": len(checks), "failed": 0, "out": str(args.out)}))


if __name__ == "__main__":
    main()
