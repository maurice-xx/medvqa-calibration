#!/usr/bin/env python3
"""P0-#1: 公平的校准迁移对照。

对每个目标格式 t，在同一批 test 样本上比较 5 种条件：
  uncalibrated        原始置信度
  in-format           用 t 自身 fit 半区拟合的 power scaling
  from-A / from-B / from-C   用源格式 fit 半区拟合的 T，直接套到 t
所有拟合使用相同样本量（该格式 fit 半区全部）、相同网格搜索、相同方法。

指标：ECE / Brier / NLL，均给图像聚类 bootstrap 95%CI（T 固定为点估计，条件 bootstrap）。
输出 results/fair_transfer/{model}.json + summary.md
"""
import json, math
from pathlib import Path
import numpy as np

from calib_metrics import RESULTS_ROOT, POWER_GRID, CALIBRATION_CONFIG
ROOT = Path(__file__).resolve().parent.parent
OUT = RESULTS_ROOT / 'fair_transfer'
FM = ['A', 'B', 'C']
BASEMAP = {'qwen': 'qwen500', 'healthgpt': 'healthgpt500', 'gemma4': 'gemma500'}
N_BOOT = 500
SEED = 20261103


def half(img):
    import hashlib
    return int(hashlib.md5(img.encode()).hexdigest(), 16) % 2


def ece(conf, corr, n=10):
    conf = np.asarray(conf, float); corr = np.asarray(corr, float)
    if len(conf) == 0:
        return None
    idx = np.clip((conf * n).astype(int), 0, n - 1)
    e = 0.0
    for b in range(n):
        m = idx == b
        if m.sum():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def brier(conf, corr):
    c = np.asarray(conf, float); y = np.asarray(corr, float)
    return float(np.mean((c - y) ** 2))


def nll(conf, corr):
    p = np.clip(np.asarray(conf, float), 1e-6, 1 - 1e-6)
    y = np.asarray(corr, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_T(conf, corr):
    if len(conf) == 0:
        return None
    best_T, best = 1.0, 9e9
    for T in POWER_GRID:
        e = ece(np.clip(np.power(conf, 1 / T), 0, 1), corr)
        if e < best:
            best, best_T = e, float(T)
    return best_T


def load(model):
    rows = [json.loads(l) for l in open(RESULTS_ROOT / f'raw/{BASEMAP[model]}.jsonl')]
    out = [r for r in rows if r.get('status') == 'ok' and r.get('conf_token_geo') is not None
           and r.get('format') in FM]
    for r in out:
        r['h'] = half(r['image_path'])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, lines = {}, ['# 公平迁移对照（ECE / Brier / NLL，图像聚类 95%CI）', '']
    for model in ['qwen', 'healthgpt', 'gemma4']:
        rows = load(model)
        if not rows:
            continue
        res = {'_metadata': {'calibration_config': CALIBRATION_CONFIG,
                              'bootstrap_unit': 'image', 'bootstrap_replicates': N_BOOT,
                              'interval': 'conditional bootstrap with fixed fitted calibrator',
                              'seed': SEED, 'temperatures': {}}}
        # 先在各格式 fit 半区拟合 T
        temps = {}
        for f in FM:
            fit = [r for r in rows if r['format'] == f and r['h'] == 0]
            temps[f] = fit_T([r['conf_token_geo'] for r in fit], [int(r['correct']) for r in fit])
        res['_metadata']['temperatures'] = temps
        for tgt in FM:
            test = [r for r in rows if r['format'] == tgt and r['h'] == 1]
            conf = [r['conf_token_geo'] for r in test]
            corr = [int(r['correct']) for r in test]
            imgs = [r['image_path'] for r in test]
            conds = {'uncalibrated': conf}
            for src in FM:
                if temps[src] is not None:
                    conds[f'from_{src}'] = list(np.clip(np.power(conf, 1 / temps[src]), 0, 1))
            if temps[tgt] is not None:
                conds['in_format'] = conds[f'from_{tgt}']

            rng = np.random.default_rng(SEED)
            uniq = sorted(set(imgs))
            idx_by_img = {im: [i for i, x in enumerate(imgs) if x == im] for im in uniq}
            boots = {k: {'ece': [], 'brier': [], 'nll': []} for k in conds}
            for _ in range(N_BOOT):
                pick = rng.choice(len(uniq), len(uniq), replace=True)
                sel = np.concatenate([idx_by_img[uniq[i]] for i in pick]) if pick.size else []
                if len(sel) == 0:
                    continue
                for k, c in conds.items():
                    cc = np.asarray(c)[sel]; yy = np.asarray(corr)[sel]
                    boots[k]['ece'].append(ece(cc, yy))
                    boots[k]['brier'].append(brier(cc, yy))
                    boots[k]['nll'].append(nll(cc, yy))

            def ci(v):
                v = np.asarray(v, float)
                return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] if len(v) else None

            cell = {}
            for k, c in conds.items():
                cell[k] = {
                    'ece': ece(c, corr), 'ece_ci95': ci(boots[k]['ece']),
                    'brier': brier(c, corr) if corr else None, 'brier_ci95': ci(boots[k]['brier']),
                    'nll': nll(c, corr) if corr else None, 'nll_ci95': ci(boots[k]['nll']),
                }
            # 迁移额外损失 = 迁移 - 格式内
            for src in FM:
                if src == tgt or not test or f'from_{src}' not in cell or 'in_format' not in cell:
                    continue
                cell[f'penalty_from_{src}'] = {
                    m: round(cell[f'from_{src}'][m] - cell['in_format'][m], 4)
                    for m in ('ece', 'brier', 'nll')}
            res[tgt] = {'n_test': len(test), 'n_fit': sum(1 for r in rows if r['format'] == tgt and r['h'] == 0),
                        'temperature_in_format': temps[tgt], 'conditions': cell,
                        'status': 'ok' if test and temps[tgt] is not None else ('test_empty' if not test else 'fit_unavailable')}
            print(model, tgt, res[tgt]['status'], {k: v.get('ece') for k, v in cell.items()})
        summary[model] = res
        lines += [f'## {model}', '']
        for tgt in FM:
            c = res[tgt]['conditions']
            lines += [f'### target {tgt} (n_test={res[tgt]["n_test"]})', '',
                      '| 条件 | ECE [CI] | Brier | NLL |', '|---|---|---|---|']
            for k in ('uncalibrated', 'in_format', 'from_A', 'from_B', 'from_C'):
                if k not in c:
                    continue
                v = c[k]
                lines.append(f"| {k} | {v['ece']} {v['ece_ci95']} | {v['brier']} | {v['nll']} |")
            lines.append('')
    (OUT / 'fair_transfer.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (OUT / 'fair_transfer.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('saved', OUT)


if __name__ == '__main__':
    from analysis_gate import require_analysis_inputs
    require_analysis_inputs('fair_transfer', RESULTS_ROOT)
    main()
