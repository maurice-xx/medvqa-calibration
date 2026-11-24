#!/usr/bin/env python3
"""校准研究抽样：SLAKE 多 split 英文可枚举题。

用法：
  python sample_data.py --out data/seed500.jsonl --n 500 --splits test,validate --seed 20261022
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SLAKE_ROOT = Path("assets/data/Slake/Slake1.0")
CATEGORIES = {"Abnormality": "abnormality", "Organ": "organ", "Modality": "modality", "Position": "position"}
PROPS = {"abnormality": 0.40, "organ": 0.20, "modality": 0.20, "position": 0.20}
EXCLUDE = re.compile(
    r"\b(treat|treatment|management|manage|next step|purpose|indication|why|"
    r"cause|history|symptom|recommend|prognosis|complication|risk)\b", re.I,
)
MODALITY_KEEP = {"ct", "mri", "xray", "ultrasound"}


def norm_light(s: str) -> str:
    return (s or "").strip().lower()


def load_candidates(splits) -> list[dict]:
    test_imgs = set()
    if "test" in splits and "validate" in splits:
        # validate 与 test 图像不相交（排除 test 图像上的 validate 题，防泄漏）
        test_imgs = {r["img_name"] for r in json.loads((SLAKE_ROOT / "test.json").read_text())}
    out = []
    for split in splits:
        rows = json.loads((SLAKE_ROOT / f"{split}.json").read_text())
        for r in rows:
            if r.get("q_lang") != "en":
                continue
            if split == "validate" and r["img_name"] in test_imgs:
                continue
            cat = CATEGORIES.get(r.get("content_type"))
            if cat is None:
                continue
            q = (r.get("question") or "").strip()
            if not q or EXCLUDE.search(q):
                continue
            ans = (r.get("answer") or "").strip()
            if not ans or "," in ans:
                continue
            at = r.get("answer_type")
            if cat == "abnormality":
                if at != "CLOSED" or norm_light(ans) not in {"yes", "no"}:
                    continue
            elif at != "OPEN":
                continue
            if cat == "modality" and norm_light(ans) not in MODALITY_KEEP:
                continue
            img = SLAKE_ROOT / "imgs" / r["img_name"]
            if not img.exists():
                continue
            out.append({
                "dataset": "slake", "split": split,
                "qid": f"slake:{split}:{r['qid']}",
                "image_path": str(img), "question": q, "gold": ans,
                "category": cat, "content_type": r.get("content_type"),
                "answer_type": at,
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--splits", default="test,validate")
    ap.add_argument("--max-per-image", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    rng = random.Random(args.seed)
    cands = load_candidates(splits)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for c in cands:
        by_cat[c["category"]].append(c)
    print("splits:", splits, "| candidates:", {k: len(v) for k, v in by_cat.items()})

    quota = {k: int(round(args.n * p)) for k, p in PROPS.items()}
    quota["abnormality"] += args.n - sum(quota.values())

    sampled: list[dict] = []
    used: Counter = Counter()
    for cat, qn in quota.items():
        pool = by_cat[cat][:]
        rng.shuffle(pool)
        took, ids = [], set()
        for limit in range(1, args.max_per_image + 1):
            for c in pool:
                if len(took) >= qn:
                    break
                if c["qid"] in ids or used[c["image_path"]] >= limit:
                    continue
                took.append(c)
                ids.add(c["qid"])
        for c in took:
            used[c["image_path"]] += 1
        sampled.extend(took)
        if len(took) < qn:
            print(f"WARN {cat}: {len(took)}/{qn}")

    sampled.sort(key=lambda x: x["qid"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for c in sampled:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("sampled", len(sampled), dict(Counter(c["category"] for c in sampled)),
          "| images:", len({c["image_path"] for c in sampled}),
          "| by split:", dict(Counter(c["split"] for c in sampled)))


if __name__ == "__main__":
    main()
