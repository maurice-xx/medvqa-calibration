#!/usr/bin/env python3
"""P1: 格式条件校准与 few-shot 曲线。

策略
  global           : 所有格式共用 (a, b)
  cond_intercept   : 共享斜率 b，每格式一个截距 a_f
  cond_full        : 每格式独立 (a_f, b_f)，L2 正则
  independent      : 每格式单独拟合（与 cond_full 等价，作为对照）

评价
  - 在 fit 半区拟合，test 半区算 ECE（按格式与总体）
  - few-shot：每格式抽 N ∈ {10,25,50,100,200} 条，比较 independent vs cond_intercept
  - 样本不足 2 条的格式回退到 global 参数（不丢样本、不写 0）

边界处理
  - 设计矩阵奇异时用伪逆，仍失败则回退 global
  - 空格式、空 test 集不参与 ECE
  - 输入文件缺失报错

输出
  results/format_cond/format_cond.json / .md / fig6_fewshot_calibration.png
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calib_metrics import RESULTS_ROOT
ROOT = Path(__file__).resolve().parent.parent
FM = ("A", "B", "C")
MODELS = (("qwen500", "Qwen3-VL-8B"), ("healthgpt500", "HealthGPT-Pro-8B"), ("gemma500", "Gemma-4-E4B"))
SIZES = (10, 25, 50, 100, 200)
REPS = 5
L2 = 1e-2
RNG = np.random.default_rng(20261103)


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


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_lr(X, y, l2=L2):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    penalty = np.ones(X.shape[1])
    if X.shape[1] == 2:
        penalty[0] = 0.0
    elif X.shape[1] in (4, 6):
        penalty[:3] = 0.0
    w = np.zeros(X.shape[1])
    for _ in range(60):
        p = sigmoid(X @ w)
        W = p * (1 - p) + 1e-6
        g = X.T @ (p - y) + l2 * penalty * w
        H = (X * W[:, None]).T @ X + l2 * np.diag(penalty)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ g
        w -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w


def design(conf, fmt, mode):
    z = logit(conf).reshape(-1, 1)
    if mode == "global":
        return np.hstack([np.ones((len(z), 1)), z])
    oh = np.array([[1.0 if f == k else 0.0 for k in FM] for f in fmt]).reshape(-1, len(FM))
    if mode == "cond_intercept":
        return np.hstack([oh, z])
    if mode == "cond_full":
        return np.hstack([oh, z * oh])
    raise ValueError(mode)


def fit_predict(tr, te, mode):
    if len(tr['correct']) == 0:
        return None
    w = fit_lr(design(tr["conf"], tr["fmt"], mode), tr["correct"])
    return sigmoid(design(te["conf"], te["fmt"], mode) @ w)


def aggregate_ece(pred, fmt, correct):
    per = per_format_ece(pred, fmt, correct)
    available = [v for v in per.values() if v is not None]
    pooled = ece(pred, correct)
    return {'ece_per_format': per, 'ece_overall': pooled, 'ece_pooled': pooled,
            'ece_macro': float(np.mean(available)) if available else None}


def per_format_ece(pred, fmt, correct):
    out = {}
    for f in FM:
        m = [i for i, ff in enumerate(fmt) if ff == f]
        out[f] = ece(pred[m], correct[m]) if m else None
    return out


def load_rows(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"缺少输入文件: {p}")
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") != "ok" or r.get("conf_token_geo") is None:
            continue
        if r.get("format") not in FM:
            continue
        rows.append({"conf": float(r["conf_token_geo"]), "correct": int(bool(r["correct"])),
                     "fmt": r["format"], "img": r["image_path"]})
    return rows


def half(r):
    return int(hashlib.md5(r["img"].encode()).hexdigest(), 16) % 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RESULTS_ROOT / "raw"))
    ap.add_argument("--out-dir", default=str(RESULTS_ROOT / "format_cond"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, lines = {}, ["# 格式条件校准", ""]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.6), sharey=True)
    for ax, (model, title) in zip(axes, MODELS):
        rows = load_rows(Path(args.raw_dir) / f"{model}.jsonl")
        tr_rows = [r for r in rows if half(r) == 0]
        te_rows = [r for r in rows if half(r) == 1]
        tr = {"conf": np.array([r["conf"] for r in tr_rows]),
              "fmt": [r["fmt"] for r in tr_rows],
              "correct": np.array([r["correct"] for r in tr_rows], float)}
        te = {"conf": np.array([r["conf"] for r in te_rows]),
              "fmt": [r["fmt"] for r in te_rows],
              "correct": np.array([r["correct"] for r in te_rows], float)}
        res = {'calibrator_config': {'input': 'logit(conf)', 'logit_clip': [1e-4, 1 - 1e-4],
                'sigmoid_z_clip': [-30, 30], 'l2_slopes': L2, 'intercepts_penalized': False,
                'optimizer': 'Newton IRLS', 'max_iterations': 60,
                'aggregation': 'pooled ECE, macro arithmetic mean of per-format ECE, and per-format ECE'},
               'n_fit_per_format': {f: sum(r['fmt'] == f for r in tr_rows) for f in FM},
               'n_test_per_format': {f: sum(r['fmt'] == f for r in te_rows) for f in FM},
               "n_fit": len(tr["correct"]), "n_test": len(te["correct"]), "strategies": {}}
        if not tr_rows or not te_rows:
            res['status'] = 'fit_unavailable' if not tr_rows else 'test_empty'
            res['raw'] = per_format_ece(te['conf'], te['fmt'], te['correct'])
            raw_values = [v for v in res['raw'].values() if v is not None]
            res['raw_pooled'] = ece(te['conf'], te['correct'])
            res['raw_macro'] = float(np.mean(raw_values)) if raw_values else None
            res['few_shot'] = {'sizes': list(SIZES), 'status': res['status'],
                               'independent': [None] * len(SIZES), 'cond_intercept': [None] * len(SIZES)}
            summary[model] = res
            lines += [f'## {title}', '', f"状态: {res['status']}；fit={len(tr_rows)}，test={len(te_rows)}", '']
            continue
        res['status'] = 'ok'
        for mode in ("global", "cond_intercept", "cond_full"):
            pred = fit_predict(tr, te, mode)
            res["strategies"][mode] = aggregate_ece(pred, te['fmt'], te['correct'])
        # independent：逐格式拟合，样本不足回退 global
        wg = fit_lr(design(tr["conf"], tr["fmt"], "global"), tr["correct"])
        pred_ind = np.zeros(len(te["correct"]))
        per_ind = {}
        for f in FM:
            mtr = [i for i, ff in enumerate(tr["fmt"]) if ff == f]
            mte = [i for i, ff in enumerate(te["fmt"]) if ff == f]
            if not mte:
                continue
            if len(mtr) >= 2:
                w = fit_lr(np.hstack([np.ones((len(mtr), 1)),
                                      logit(tr["conf"][mtr]).reshape(-1, 1)]), tr["correct"][mtr])
                pred_ind[mte] = sigmoid(w[0] + w[1] * logit(te["conf"][mte]))
            else:
                pred_ind[mte] = sigmoid(design(te["conf"][mte], [f] * len(mte), "global") @ wg)
            per_ind[f] = ece(pred_ind[mte], te["correct"][mte])
        res["strategies"]["independent"] = aggregate_ece(pred_ind, te['fmt'], te['correct'])
        res["raw"] = per_format_ece(te["conf"], te["fmt"], te["correct"])

        res['raw_pooled'] = ece(te['conf'], te['correct'])
        res['raw_macro'] = float(np.mean([v for v in res['raw'].values() if v is not None]))
        for strategy in res['strategies'].values():
            strategy['ece_pooled'] = strategy['ece_overall']
            strategy['ece_macro'] = float(np.mean([v for v in strategy['ece_per_format'].values() if v is not None]))
        # few-shot：independent vs cond_intercept
        curves = {"independent": [], "cond_intercept": []}
        for N in SIZES:
            e_ind, e_cond = [], []
            for _ in range(REPS):
                pick = []
                for f in FM:
                    pool = [r for r in tr_rows if r["fmt"] == f]
                    if not pool:
                        continue
                    idx = RNG.choice(len(pool), size=min(N, len(pool)), replace=False)
                    pick.extend([pool[i] for i in idx])
                if len(pick) < 6:
                    continue
                tr_s = {"conf": np.array([r["conf"] for r in pick]),
                        "fmt": [r["fmt"] for r in pick],
                        "correct": np.array([r["correct"] for r in pick], float)}
                e_cond.append(ece(fit_predict(tr_s, te, "cond_intercept"), te["correct"]))
                wg_s = fit_lr(design(tr_s["conf"], tr_s["fmt"], "global"), tr_s["correct"])
                pred_s = np.zeros(len(te["correct"]))
                for f in FM:
                    mtr = [i for i, ff in enumerate(tr_s["fmt"]) if ff == f]
                    mte = [i for i, ff in enumerate(te["fmt"]) if ff == f]
                    if not mte:
                        continue
                    if len(mtr) >= 2:
                        w = fit_lr(np.hstack([np.ones((len(mtr), 1)),
                                              logit(tr_s["conf"][mtr]).reshape(-1, 1)]),
                                   tr_s["correct"][mtr])
                        pred_s[mte] = sigmoid(w[0] + w[1] * logit(te["conf"][mte]))
                    else:
                        pred_s[mte] = sigmoid(design(te["conf"][mte], [f] * len(mte), "global") @ wg_s)
                e_ind.append(ece(pred_s, te["correct"]))
            curves["independent"].append(None if not e_ind else round(float(np.mean(e_ind)), 4))
            curves["cond_intercept"].append(None if not e_cond else round(float(np.mean(e_cond)), 4))
        res["few_shot"] = {"sizes": list(SIZES), 'actual_samples_per_format': {f: [min(n, sum(r['fmt'] == f for r in tr_rows)) for n in SIZES] for f in FM},
                           'repetitions': REPS, 'sampling_unit': 'question', 'split_unit': 'image', **curves}
        summary[model] = res

        log(f"== {model} raw={res['raw']}")
        for k, v in res["strategies"].items():
            log(f"   {k}: overall={v['ece_overall']} per={v['ece_per_format']}")

        xs = [n for n, v in zip(SIZES, curves["independent"]) if v is not None]
        ys = [v for v in curves["independent"] if v is not None]
        ys2 = [v for v in curves["cond_intercept"] if v is not None]
        if xs:
            ax.plot(xs, ys, "o-", ms=4, lw=1.2, label="independent")
            ax.plot(xs, ys2, "s--", ms=4, lw=1.2, label="format-conditioned")
        ax.set_xscale("log")
        ax.set_xticks(list(SIZES))
        ax.set_xticklabels([str(n) for n in SIZES])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("labeled samples per format")
        lines += [f"## {title}", "",
                  f"- fit {res['n_fit']} / test {res['n_test']}",
                  "",
                  "| 策略 | A | B | C | pooled | macro |", "|---|---|---|---|---|---|"]
        for k, v in res["strategies"].items():
            pf = v["ece_per_format"]
            lines.append(f"| {k} | {pf.get('A')} | {pf.get('B')} | {pf.get('C')} | {v['ece_pooled']} | {v['ece_macro']:.4f} |")
        lines.append("")
    axes[0].set_ylabel("test ECE")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_fewshot_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    (out_dir / "format_cond.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "format_cond.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"saved {out_dir}/format_cond.{{json,md,png}}")


if __name__ == '__main__':
    from analysis_gate import require_analysis_inputs
    require_analysis_inputs('format_cond_calib', RESULTS_ROOT)
    main()
