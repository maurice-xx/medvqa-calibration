#!/usr/bin/env python3
"""P1: 3x3 重评分分析（生成效应 vs 评分效应 + 规范上下文 ECE）。

核心量
  - scoring spread  : 固定答案，换评分前缀的置信度跨度均值
  - generation spread: 固定前缀，换答案来源的置信度跨度均值
  - 两者均给图像聚类 bootstrap 95% CI
  - canonical ECE   : 所有答案在同一前缀下评分时的 ECE（行=答案来源，列=评分前缀）
  - 对角相关 rho    : 重评分对角 vs 生成时置信度（校验"前缀匹配"的有效性）

边界处理
  - 只保留 9 格齐全的题目；缺格计入 dropped
  - 零方差/除零时 rho 记为 None
  - 输入文件缺失或字段不全时明确报错

输出
  results/rescore_analysis/rescore_summary.json
  results/rescore_analysis/rescore_summary.md
"""
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from calib_metrics import RESULTS_ROOT, cluster_mean_ci
ROOT = Path(__file__).resolve().parent.parent
FM = ("A", "B", "C")
BASEMAP = {"qwen": "qwen500", "healthgpt": "healthgpt500", "gemma4": "gemma500"}
N_BOOT = 1000
SEED = 20261103


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


def pearson(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return cov / (sx * sy)


def load_jsonl(path):
    p = Path(path)
    if not p.exists():
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


def build_matrix(recs, base_rows):
    """返回 full{qid: {(g,s): conf}}, corr{(qid,g): 0/1}, gen_conf{(qid,g): conf}。"""
    corr, gen_conf = {}, {}
    for r in base_rows:
        if r.get("status") != "ok" or r.get("conf_token_geo") is None:
            continue
        f = r.get("format")
        if f in FM:
            corr[(r["qid"], f)] = int(bool(r.get("correct")))
            gen_conf[(r["qid"], f)] = float(r["conf_token_geo"])
    mat = defaultdict(dict)
    for r in recs:
        if r.get("conf") is None:
            continue
        if r.get("g") not in FM or r.get("s") not in FM:
            continue
        mat[r["qid"]][(r["g"], r["s"])] = float(r["conf"])
    full = {q: m for q, m in mat.items() if len(m) == 9}
    return full, corr, gen_conf


def spreads_by_question(full):
    """每个题目的 (scoring_spread, generation_spread, image_path)。"""
    out = []
    for q, m in full.items():
        sc, gn = [], []
        for g in FM:
            vals = [m[(g, s)] for s in FM]
            sc.append(max(vals) - min(vals))
        for s in FM:
            vals = [m[(g, s)] for g in FM]
            gn.append(max(vals) - min(vals))
        out.append((statistics.mean(sc), statistics.mean(gn)))
    return out


def boot_ci(values, images, n_boot=N_BOOT, seed=SEED):
    return cluster_mean_ci(values, images, n_boot=n_boot, seed=seed)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore-dir", default=str(RESULTS_ROOT / "rescore"))
    ap.add_argument("--raw-dir", default=str(RESULTS_ROOT / "raw"))
    ap.add_argument("--out-dir", default=str(RESULTS_ROOT / "rescore_analysis"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, lines = {}, ["# 3×3 重评分分析", ""]

    for model in ("qwen", "healthgpt", "gemma4"):
        recs = load_jsonl(Path(args.rescore_dir) / f"{model}.jsonl")
        base = load_jsonl(Path(args.raw_dir) / f"{BASEMAP[model]}.jsonl")
        if recs is None or base is None:
            log(f"跳过 {model}: 数据文件缺失")
            continue
        full, corr, gen_conf = build_matrix(recs, base)
        if not full:
            log(f"跳过 {model}: 无完整 9 格题目")
            continue
        image_by_question = {r['qid']: r['image_path'] for r in base}
        if any(q not in image_by_question for q in full):
            raise ValueError('完整矩阵缺少可验证的图像聚类映射')
        images = [image_by_question[q] for q in full]
        sp = spreads_by_question(full)
        sc_vals = [a for a, _ in sp]
        gn_vals = [b for _, b in sp]

        pairs = [(m[(g, g)], gen_conf[(q, g)]) for q, m in full.items() for g in FM
                 if (q, g) in gen_conf]
        rho = pearson([a for a, _ in pairs], [b for _, b in pairs]) if pairs else None

        diag_ece, canon = {}, {}
        for g in FM:
            confs = [full[q][(g, g)] for q in full if (q, g) in corr]
            corrs = [corr[(q, g)] for q in full if (q, g) in corr]
            diag_ece[g] = ece(confs, corrs) if confs else None
        for s in FM:
            canon[s] = {}
            for g in FM:
                confs = [full[q][(g, s)] for q in full if (q, g) in corr]
                corrs = [corr[(q, g)] for q in full if (q, g) in corr]
                canon[s][g] = ece(confs, corrs) if confs else None

        summary[model] = {
            'n_images_full': len(set(images)),
            'bootstrap': {'unit': 'image', 'replicates': N_BOOT, 'seed': SEED},
            'interpretation': 'descriptive scoring-context spread; not a causal decomposition',
            'input_semantics': 'read companion provenance; historical matrices used user-side suffix scoring and are not reproducible from the former scoring code',
            "n_questions_full": len(full),
            "dropped_incomplete": len({r["qid"] for r in recs}) - len(full),
            "diag_vs_gen_pearson": round(rho, 4) if rho is not None else None,
            "scoring_spread_mean": round(statistics.mean(sc_vals), 4),
            "scoring_spread_ci95": boot_ci(sc_vals, images),
            "generation_spread_mean": round(statistics.mean(gn_vals), 4),
            "generation_spread_ci95": boot_ci(gn_vals, images),
            "ece_diagonal_scoring": diag_ece,
            "ece_canonical": canon,
            'matrix_minimum_policy': '保存未舍入ECE；仅显示时舍入，精确行最小加粗，精确相等共享加粗',
        }
        s = summary[model]
        log(f"== {model} n={len(full)} rho={s['diag_vs_gen_pearson']}")
        log(f"   scoring {s['scoring_spread_mean']} {s['scoring_spread_ci95']}")
        log(f"   generation {s['generation_spread_mean']} {s['generation_spread_ci95']}")
        log(f"   diag {diag_ece}  canonical {canon}")

        lines += [f"## {model}", "",
                  f"- 完整 9 格题目：{len(full)}（丢弃不完整 {s['dropped_incomplete']}）",
                  f"- 对角-生成相关 ρ = {s['diag_vs_gen_pearson']}",
                  f"- **评分跨度** = {s['scoring_spread_mean']} {s['scoring_spread_ci95']}",
                  f"- **答案来源跨度** = {s['generation_spread_mean']} {s['generation_spread_ci95']}",
                  "",
                  "| 答案来源 | sA | sB | sC |", "|---|---|---|---|"]
        for g in FM:
            lines.append(f"| {g} | " + " | ".join(str(canon[s_][g]) for s_ in FM) + " |")
        lines.append("")

    (out_dir / "rescore_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "rescore_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"saved {out_dir}/rescore_summary.{{json,md}}")


if __name__ == '__main__':
    from analysis_gate import require_analysis_inputs
    require_analysis_inputs('rescore_analysis', RESULTS_ROOT)
    main()
