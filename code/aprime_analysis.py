#!/usr/bin/env python3
"""P1: A' 控制条件分析（标签效应 vs 观察效应）。

对比
  A        : 直接短答
  A'       : 仅加 "Answer:" 标签（无观察、无自报）
  B        : 标签 + 观察要求

  A→A' 的准确率差 = 标签效应；A'→B 的差 = 观察效应

边界处理
  - A' 记录状态非 ok 或答案为空 → 计入 invalid，不参与准确率
  - 若存在 {model}_conf.jsonl（A' 在 sA 下的置信度），额外报告 A' 的 ECE
  - 主实验文件缺失时明确报错

输出
  results/aprime/aprime_summary.json / .md
"""
import argparse
import json
from pathlib import Path

import numpy as np

from calib_metrics import RESULTS_ROOT
ROOT = Path(__file__).resolve().parent.parent
BASEMAP = {"qwen": "qwen500", "healthgpt": "healthgpt500", "gemma4": "gemma500"}
MODELS = ("qwen", "healthgpt", "gemma4")


def log(m):
    print(m, flush=True)


def ece(conf, corr, n_bins=10):
    conf = np.asarray(conf, float)
    corr = np.asarray(corr, float)
    if len(conf) == 0:
        return None
    idx = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def load_jsonl(path, required=True):
    p = Path(path)
    if not p.exists():
        if required:
            raise SystemExit(f"缺少输入文件: {p}")
        return None
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RESULTS_ROOT / "raw"))
    ap.add_argument("--aprime-dir", default=str(RESULTS_ROOT / "raw_aprime"))
    ap.add_argument("--out-dir", default=str(RESULTS_ROOT / "aprime"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(ROOT / "code"))
    from run_calib import normalize  # 复用与主实验一致的答案归一化

    summary, lines = {}, ["# A′ 控制：标签效应 vs 观察效应", ""]
    for model in MODELS:
        base = load_jsonl(Path(args.raw_dir) / f"{BASEMAP[model]}.jsonl")
        apr = load_jsonl(Path(args.aprime_dir) / f"{model}.jsonl", required=False)
        if apr is None:
            log(f"跳过 {model}: 无 A′ 数据")
            continue

        res = {"n_base": len(base), "n_aprime": len(apr), "invalid_aprime": 0}
        # 主实验 A / B
        for f in ("A", "B"):
            rs = [r for r in base if r.get("status") == "ok"
                  and r.get("conf_token_geo") is not None and r.get("format") == f]
            acc = sum(int(r["correct"]) for r in rs) / len(rs) if rs else None
            res[f] = {"n": len(rs), "acc": round(acc, 4) if acc is not None else None,
                      "conf": round(float(np.mean([r["conf_token_geo"] for r in rs])), 4) if rs else None,
                      "ece": ece([r["conf_token_geo"] for r in rs], [int(r["correct"]) for r in rs])}
        # A'
        ok = []
        for r in apr:
            if r.get("status") != "ok" or not r.get("ans_raw"):
                res["invalid_aprime"] += 1
                continue
            gold = normalize(r.get("gold", ""), r.get("category", ""))
            pred = normalize(r["ans_raw"], r.get("category", ""))
            ok.append(int(bool(pred) and pred == gold))
        res["A_prime"] = {"n": len(ok), "acc": round(sum(ok) / len(ok), 4) if ok else None}
        conf_rows = load_jsonl(Path(args.aprime_dir) / f"{model}_conf.jsonl", required=False)
        if conf_rows:
            cmap = {r["qid"]: r["conf"] for r in conf_rows if r.get("conf") is not None}
            confs, corrs = [], []
            for r, y in zip([x for x in apr if x.get("status") == "ok" and x.get("ans_raw")], ok):
                if r["qid"] in cmap:
                    confs.append(cmap[r["qid"]])
                    corrs.append(y)
            if confs:
                res["A_prime"].update({"conf": round(float(np.mean(confs)), 4),
                                       "ece": ece(confs, corrs), "n_conf": len(confs)})

        summary[model] = res
        delta_label = (res["A_prime"]["acc"] - res["A"]["acc"]) if res["A_prime"]["acc"] is not None and res["A"]["acc"] is not None else None
        delta_obs = (res["B"]["acc"] - res["A_prime"]["acc"]) if res["A_prime"]["acc"] is not None and res["B"]["acc"] is not None else None
        res["delta_label_pp"] = round(100 * delta_label, 1) if delta_label is not None else None
        res["delta_observation_pp"] = round(100 * delta_obs, 1) if delta_obs is not None else None
        log(f"== {model}: A={res['A']['acc']} A'={res['A_prime']['acc']} B={res['B']['acc']} "
            f"| 标签 {res['delta_label_pp']}pp, 观察 {res['delta_observation_pp']}pp")

        lines += [f"## {model}", "",
                  "| 条件 | n | acc | conf | ECE |", "|---|---|---|---|---|"]
        for f in ("A", "A_prime", "B"):
            v = res[f]
            lines.append(f"| {f} | {v['n']} | {v['acc']} | {v.get('conf', '—')} | {v.get('ece', '—')} |")
        lines += ["", f"- 标签效应（A→A′）：{res['delta_label_pp']} pp",
                  f"- 观察效应（A′→B）：{res['delta_observation_pp']} pp",
                  f"- A′ 无效记录：{res['invalid_aprime']}", ""]

    (out_dir / "aprime_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "aprime_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"saved {out_dir}/aprime_summary.{{json,md}}")


if __name__ == '__main__':
    from analysis_gate import require_analysis_inputs
    require_analysis_inputs('aprime_analysis', RESULTS_ROOT)
    main()
