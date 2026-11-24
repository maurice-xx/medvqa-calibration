#!/usr/bin/env python3
"""Distinct-score risk curves and image-separated threshold transfer."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calib_metrics import RESULTS_ROOT, distinct_curve, threshold_at_risk, selection_status
from selective_full import half, apply as apply_threshold, boot_ci

FM = ('A', 'B', 'C')
MODELS = (('qwen500', 'Qwen3-VL-8B'), ('healthgpt500', 'HealthGPT-Pro-8B'), ('gemma500', 'Gemma-4-E4B'))
N_BOOT = 1000
SEED = 20261103

def load(path):
    source = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    rows = [{'conf': float(r['conf_token_geo']), 'correct': int(r['correct']),
             'img': r['image_path'], 'fmt': r['format']} for r in source
            if r.get('status') == 'ok' and r.get('conf_token_geo') is not None and r.get('format') in FM]
    return rows, len(source) - len(rows)

def risk_curve(conf, corr, grid=None):
    points = distinct_curve(conf, corr)
    return np.array([p['coverage'] for p in points]), np.array([p['risk'] for p in points])

def risk_at(conf, corr, coverage):
    points = distinct_curve(conf, corr)
    p = next((p for p in points if p['coverage'] >= coverage), None)
    return p['risk'] if p else None

def cov_at_risk(conf, corr, target):
    valid = [p for p in distinct_curve(conf, corr) if p['risk'] <= target]
    return valid[-1]['coverage'] if valid else 0.0

def boot_transfer(rows_a, rows_b, n_boot=N_BOOT, seed=SEED):
    # 每次独立重采样图像级 fit 与 test；own 格式也不以字典成员关系判断来源。
    from collections import defaultdict
    def grouped(rows):
        out = defaultdict(list)
        for r in rows:
            out[r['img']].append(r)
        return out, sorted(out)
    a, ai = grouped(rows_a); b, bi = grouped(rows_b)
    if not ai or not bi:
        return None
    rng = np.random.default_rng(seed)
    covs, risks, available = [], [], 0
    for _ in range(n_boot):
        fit = [r for i in rng.integers(0, len(ai), len(ai)) for r in a[ai[i]]]
        test = [r for i in rng.integers(0, len(bi), len(bi)) for r in b[bi[i]]]
        thr = threshold_at_risk([r['conf'] for r in fit], [r['correct'] for r in fit])
        available += thr is not None
        cov, risk, _ = apply_threshold(test, thr)
        covs.append(cov)
        if risk is not None:
            risks.append(risk)
    def ci(v):
        return [float(x) for x in np.percentile(v, [2.5, 97.5])] if v else None
    return {'coverage_ci95': ci(covs), 'risk_ci95': ci(risks),
            'n_boot_coverage': len(covs), 'n_boot_nonempty_risk': len(risks),
            'n_boot_fit_threshold_available': available}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', default=str(RESULTS_ROOT / 'raw'))
    ap.add_argument('--out-dir', default=str(RESULTS_ROOT / 'selective_risk'))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    summary, lines = {}, ['# 选择性风险（test半区）与A格式fit阈值迁移', '',
        '所有曲线只含 distinct-score 阈值边界；ties不拆分。阈值在fit选，test评估。',
        '迁移bootstrap重新拟合fit阈值并独立重采样test图像；coverage含零选择，risk仅非空选择。', '']
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    for ax, (model, title) in zip(axes, MODELS):
        path = Path(args.raw_dir) / f'{model}.jsonl'
        if not path.exists():
            continue
        rows, dropped = load(path)
        fit = [r for r in rows if half(r['img']) == 0]
        test = [r for r in rows if half(r['img']) == 1]
        a = [r for r in fit if r['fmt'] == 'A']
        thr = threshold_at_risk([r['conf'] for r in a], [r['correct'] for r in a])
        res = {'n': len(rows), 'n_fit': len(fit), 'n_test': len(test), 'dropped': dropped,
               'per_format': {}, 'threshold_from_A_at_5pct': thr, 'threshold_reachable': thr is not None,
               'split': 'image md5 parity: fit=0 test=1', 'curve_ties': 'complete threshold groups'}
        for f in FM:
            rs = [r for r in test if r['fmt'] == f]
            c = [r['conf'] for r in rs]; y = [r['correct'] for r in rs]
            x, z = risk_curve(c, y); ax.plot(x, z, label=f'format {f}')
            high = [r for r in rs if r['conf'] >= 0.8]
            res['per_format'][f] = {'n': len(rs), 'acc': sum(y) / len(y) if y else None,
                'risk_at_cov50': risk_at(c, y, .5), 'risk_at_cov70': risk_at(c, y, .7),
                'risk_at_cov90': risk_at(c, y, .9), 'cov_at_risk5': cov_at_risk(c, y, .05),
                'cov_at_risk10': cov_at_risk(c, y, .1), 'curve': distinct_curve(c, y),
                'high_conf_error_80': 1 - np.mean([r['correct'] for r in high]) if high else None,
                'high_conf_share_80': len(high) / len(rs) if rs else None}
            cov, risk, n = apply_threshold(rs, thr)
            cell = {'coverage': cov, 'risk': risk, 'n_selected': n,
                    'fit_threshold_available': thr is not None,
                    'test_target_met': risk <= .05 if risk is not None else None,
                    'status': selection_status(thr, rs, risk, n)}
            cell.update(boot_transfer(a, rs) or {})
            res[f'apply_A_thr_to_{f}'] = cell
        summary[model] = res
        lines += [f'## {title}', '', f"A fit阈值: {thr}; fit/test={len(fit)}/{len(test)}", '',
                  '| 目标 | coverage | risk | 状态 |', '|---|---|---|---|']
        for f in FM:
            v = res[f'apply_A_thr_to_{f}']
            lines.append(f"| {f} | {v['coverage']:.4f} | {v['risk']} | {v['status']} |")
        ax.set_title(title, fontsize=9); ax.set_xlabel('coverage'); ax.axhline(.05, ls=':', c='k', lw=.8)
    axes[0].set_ylabel('selective risk'); axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out / 'fig5_risk_coverage.png', dpi=300); plt.close(fig)
    (out / 'selective_risk.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'selective_risk.md').write_text('\n'.join(lines), encoding='utf-8')
    print('saved', out)

if __name__ == '__main__':
    from analysis_gate import require_analysis_inputs
    require_analysis_inputs('selective_risk', RESULTS_ROOT)
    main()
