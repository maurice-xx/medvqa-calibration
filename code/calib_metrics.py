#!/usr/bin/env python3
"""校准指标与跨格式迁移矩阵。

用法：
  python calib_metrics.py --raw-dir results/raw --out results/summary

输出：
  summary.json / summary.md
  - 每模型×格式的 acc / ECE(10bin) / Brier / AUROC / 平均置信度
  - 3×3 温度缩放迁移矩阵（源格式拟合 T → 目标格式 ECE）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = Path(os.environ.get('MEDVQA_RESULTS_ROOT', 'results/corrected'))
if not RESULTS_ROOT.is_absolute():
    RESULTS_ROOT = PROJECT_ROOT / RESULTS_ROOT
POWER_GRID = tuple(math.exp(math.log(0.05) + i * math.log(400) / 79) for i in range(80))
CALIBRATION_CONFIG = {'method': 'power', 'transform': 'c**(1/T)',
                      'objective': 'ECE10', 'grid': list(POWER_GRID),
                      'input_clip': [0.0, 1.0], 'regularization': None,
                      'nll_clip': [1e-6, 1 - 1e-6]}

def distinct_curve(conf, correct):
    pairs = sorted(zip(conf, correct), key=lambda x: -x[0])
    total, hits, out = len(pairs), 0, []
    for i, (score, y) in enumerate(pairs):
        hits += y
        if i + 1 == total or pairs[i + 1][0] != score:
            out.append({'threshold': float(score), 'coverage': (i + 1) / total,
                        'risk': ((i + 1) - hits) / (i + 1), 'n_selected': i + 1})
    return out

def threshold_at_risk(conf, correct, target=0.05):
    valid = [p for p in distinct_curve(conf, correct) if p['risk'] <= target]
    return valid[-1]['threshold'] if valid else None

def selection_status(threshold, rows, risk, n_selected, target=0.05):
    if threshold is None:
        return 'fit_threshold_unavailable'
    if not rows:
        return 'test_empty'
    if not n_selected:
        return 'test_selection_empty'
    return 'test_risk_exceeds_target' if risk > target else 'test_risk_at_or_below_target'

def cluster_mean_ci(values, images, n_boot=1000, seed=20261103):
    import numpy as np
    by = defaultdict(list)
    for value, image in zip(values, images):
        by[image].append(value)
    keys = sorted(by)
    if not keys:
        return None
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = [v for i in rng.integers(0, len(keys), len(keys)) for v in by[keys[i]]]
        means.append(float(np.mean(sample)))
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def cohort_metadata(rows):
    """以题目、图像、格式、评分及源文件/执行身份确定分析总体。"""
    source_fields = ['stored_source', 'source_raw_sha256', 'run_signature', 'protocol']
    records = sorted((str(r['qid']), r['image_path'], r['format'],
                      int(r['correct']), float(r['conf_token_geo']),
                      *[str(r.get(field, '')) for field in source_fields]) for r in rows)
    encoded = json.dumps(records, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return {'n_records': len(records), 'n_images': len({r['image_path'] for r in rows}),
            'sha256': hashlib.sha256(encoded).hexdigest(),
            'identity_fields': ['qid', 'image_path', 'format', 'correct', 'conf_token_geo', *source_fields]}


def ece(conf: list[float], correct: list[int], n_bins: int = 10) -> float:
    if len(conf) == 0:
        return float("nan")
    bins = [[] for _ in range(n_bins)]
    for c, y in zip(conf, correct):
        b = min(int(c * n_bins), n_bins - 1)
        bins[b].append((c, y))
    e = 0.0
    for b in bins:
        if not b:
            continue
        acc = sum(y for _, y in b) / len(b)
        avg = sum(c for c, _ in b) / len(b)
        e += len(b) / len(conf) * abs(acc - avg)
    return e


def brier(conf: list[float], correct: list[int]) -> float:
    if len(conf) == 0:
        return float("nan")
    return sum((c - y) ** 2 for c, y in zip(conf, correct)) / len(conf)


def auroc(scores: list[float], labels: list[int]) -> float:
    """标签 1 = 正类（此处：错误样本）。并列取平均秩。"""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    allv = sorted(set(scores))
    rank = {}
    for v in allv:
        less = sum(1 for s in scores if s < v)
        eq = sum(1 for s in scores if s == v)
        rank[v] = less + (eq + 1) / 2  # 平均秩（1-indexed）
    r_pos = sum(rank[s] for s in pos)
    r_neg = sum(rank[s] for s in neg)
    n1, n0 = len(pos), len(neg)
    return (r_pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def temp_scale(conf: list[float], T: float) -> list[float]:
    return [max(min(c ** (1.0 / T), 1.0), 0.0) for c in conf]


def fit_temperature(conf: list[float], correct: list[int]) -> float:
    best_T, best_e = 1.0, float("inf")
    if len(conf) == 0:
        return None
    for T in POWER_GRID:
        e = ece(temp_scale(conf, T), correct)
        if e < best_e:
            best_e, best_T = e, T
    return float(best_T)


def img_half(r: dict) -> int:
    h = hashlib.md5(r["image_path"].encode()).hexdigest()
    return int(h, 16) % 2  # 0=fit, 1=test


def finite_round(value, digits=4):
    return round(value, digits) if math.isfinite(value) else None


def per_format_metrics(rows: list[dict], fmt: str) -> dict:
    rs = [r for r in rows if r["format"] == fmt and r.get("status") == "ok"
          and r.get("conf_token_geo") is not None]
    conf = [r["conf_token_geo"] for r in rs]
    correct = [int(r["correct"]) for r in rs]
    labels_err = [1 - c for c in correct]
    verb = [(r["conf_verbalized"], int(r["correct"])) for r in rs
            if r.get("conf_verbalized") is not None]
    return {
        "format": fmt, "n": len(rs), "cohort": cohort_metadata(rs),
        "n_invalid": sum(1 for r in rows if r["format"] == fmt and r.get("status") != "ok"),
        "acc": round(sum(correct) / len(correct), 4) if correct else None,
        "mean_conf": round(sum(conf) / len(conf), 4) if conf else None,
        "ece10": ece(conf, correct) if conf else None,
        "brier": round(brier(conf, correct), 4) if conf else None,
        "auroc_error_detection": finite_round(auroc([1 - c for c in conf], labels_err)) if conf else None,
        "verbalized": {
            "n": len(verb),
            "mean": round(sum(v for v, _ in verb) / len(verb), 4) if verb else None,
            "ece10": ece([v for v, _ in verb], [c for _, c in verb]) if verb else None,
            "auroc_error_detection": finite_round(auroc([1 - v for v, _ in verb], [1 - c for _, c in verb])) if verb else None,
        },
    }


def transfer_matrix(rows: list[dict]) -> dict:
    """源格式拟合 T（fit 半区）→ 各格式 test 半区 ECE。返回 3×3 矩阵。"""
    fmts = ["A", "B", "C"]
    data = {}
    for fmt in fmts:
        rs = [r for r in rows if r["format"] == fmt and r.get("status") == "ok"
              and r.get("conf_token_geo") is not None]
        fit = [r for r in rs if img_half(r) == 0]
        test = [r for r in rs if img_half(r) == 1]
        data[fmt] = (fit, test)
    mat_uncal, mat_cal, temps = {}, {}, {}
    for src in fmts:
        fit, _ = data[src]
        conf_f = [r["conf_token_geo"] for r in fit]
        corr_f = [int(r["correct"]) for r in fit]
        T = fit_temperature(conf_f, corr_f) if conf_f else None
        temps[src] = T
        for tgt in fmts:
            _, test = data[tgt]
            conf_t = [r["conf_token_geo"] for r in test]
            corr_t = [int(r["correct"]) for r in test]
            if not conf_t or T is None:
                continue
            mat_uncal[(src, tgt)] = ece(conf_t, corr_t)
            mat_cal[(src, tgt)] = ece(temp_scale(conf_t, T), corr_t)
    return {"temperatures": temps, "calibration_config": CALIBRATION_CONFIG,
            "uncalibrated": {f"{s}->{t}": v for (s, t), v in mat_uncal.items()},
            "temperature_scaled": {f"{s}->{t}": v for (s, t), v in mat_cal.items()},
            "n_test_per_format": {f: len(data[f][1]) for f in fmts}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RESULTS_ROOT / 'raw'))
    ap.add_argument("--out", default=str(RESULTS_ROOT / 'summary'))
    args = ap.parse_args()

    result = {}
    for p in sorted(Path(args.raw_dir).glob("*.jsonl")):
        rows = load(p)
        result[p.stem] = {
            "n_records": len(rows),
            "formats": {f: per_format_metrics(rows, f) for f in ["A", "B", "C"]},
            "transfer": transfer_matrix(rows),
        }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    lines = ["# MedVQA 跨格式校准：阶段 1b 结果", ""]
    for model, r in result.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| 格式 | n | acc | mean conf | ECE10 | Brier | AUROC(err) | 自报 n | 自报 ECE |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for f, fm in r["formats"].items():
            lines.append(
                f"| {f} | {fm['n']} | {fm['acc']} | {fm['mean_conf']} | {fm['ece10']} | "
                f"{fm['brier']} | {fm['auroc_error_detection']} | {fm['verbalized']['n']} | "
                f"{fm['verbalized']['ece10']} |")
        tr = r["transfer"]
        lines.append("")
        lines.append(f"拟合温度：{tr['temperatures']}（fit 半区 n/test 半区 n：{tr['n_test_per_format']}）")
        lines.append("")
        lines.append("**未校准 ECE（行=拟合源格式，列=目标格式）**")
        lines.append("")
        lines.append("| src\\tgt | A | B | C |")
        lines.append("|---|---|---|---|")
        for s in ["A", "B", "C"]:
            lines.append(f"| {s} | " + " | ".join(
                str(tr["uncalibrated"].get(f"{s}->{t}", "—")) for t in ["A", "B", "C"]) + " |")
        lines.append("")
        lines.append("**温度缩放后 ECE（对角线=格式内重校准，非对角线=跨格式迁移）**")
        lines.append("")
        lines.append("| src\\tgt | A | B | C |")
        lines.append("|---|---|---|---|")
        for s in ["A", "B", "C"]:
            lines.append(f"| {s} | " + " | ".join(
                str(tr["temperature_scaled"].get(f"{s}->{t}", "—")) for t in ["A", "B", "C"]) + " |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == '__main__':
    from analysis_gate import require_analysis_inputs
    require_analysis_inputs('calib_metrics', RESULTS_ROOT)
    main()
