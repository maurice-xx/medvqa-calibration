#!/usr/bin/env python3
"""只读CPU检查完整raw的原prompt边界、可逆分词和答案跨度。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import fixed_output_tf as ft
import rescore_3x3 as rs
from run_calib import FORMATS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    proc = rs.AutoProcessor.from_pretrained(rs.MODEL_PATHS[args.model], local_files_only=True)
    counts, cases = Counter(), []
    started = time.time()
    paths = ft.source_paths(args.model, "main")
    if args.model != "gemma4":
        paths += ft.source_paths(args.model, "external")
    for path in paths:
        for row in ft.read_rows(path):
            raw = row.get("raw_full", row.get("raw", ""))
            reason = []
            if len(raw) == 600 and not row.get("raw_full"):
                counts["blocked_known_truncation"] += 1
                continue
            prompt = rs.build_text(proc, row["question"], FORMATS[row["format"]]["instr"], args.model, row["image_path"])
            full = prompt + raw
            enc = proc.tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = enc["input_ids"], enc["offset_mapping"]
            selected = [i for i, (left, right) in enumerate(offsets) if right > len(prompt) and left < len(full)]
            if not selected:
                reason.append("empty_output")
            else:
                original = proc.tokenizer.encode(prompt, add_special_tokens=False)
                if ids[:selected[0]] != original:
                    reason.append("prompt_output_crossing")
                output_indices = [i for i in selected if ids[i] not in set(proc.tokenizer.all_special_ids)]
                output_ids = [ids[i] for i in output_indices]
                decoded = proc.tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if decoded != raw:
                    reason.append("raw_roundtrip_mismatch")
                span = ft.spans.text_span(raw, row["format"])
                left, right = len(prompt) + span["span_char_start"], len(prompt) + span["span_char_end"]
                answer = [i for i in output_indices if offsets[i][1] > left and offsets[i][0] < right]
                if not answer:
                    reason.append("empty_answer_span")
                for i in answer:
                    a, b = offsets[i]
                    before = full[a:left] if a < left else ""
                    after = full[right:b] if b > right else ""
                    if before.strip(' \t\r\n"') or after.strip(' \t\r\n"'):
                        reason.append("partial_answer_boundary")
                        break
            counts["checked"] += 1
            if reason:
                for item in reason:
                    counts[item] += 1
                cases.append({"source": str(path), "qid": row["qid"], "format": row["format"], "raw": raw, "reasons": reason})
            else:
                counts["strict_preconditions_ok"] += 1
    result = {"model": args.model, "counts": dict(counts), "cases": cases,
              "elapsed_s": time.time() - started,
              "code_sha256": {p.name: rs.sha256(p.read_bytes()) for p in
                              (Path(__file__), Path(ft.__file__), Path(rs.__file__), Path(ft.spans.__file__))}}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
