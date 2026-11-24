#!/usr/bin/env python3
"""图像级 fit/test 阈值迁移；固定阈值的 test 图像聚类 bootstrap。"""
import json
from collections import defaultdict
import numpy as np
from calib_metrics import (RESULTS_ROOT, POWER_GRID, CALIBRATION_CONFIG,
                           distinct_curve, threshold_at_risk, selection_status,
                           fit_temperature, temp_scale)

OUT = RESULTS_ROOT / 'selective_full'
FM = ['A', 'B', 'C']
BASEMAP = {'qwen': 'qwen500', 'healthgpt': 'healthgpt500', 'gemma': 'gemma500'}
N_BOOT = 1000
SEED = 20261103

def half(img):
    import hashlib
    return int(hashlib.md5(img.encode()).hexdigest(), 16) % 2

def curve(conf, corr, grid=None):
    return distinct_curve(conf, corr)

def thr_at_risk(rows, target=0.05):
    return threshold_at_risk([r['conf'] for r in rows], [r['correct'] for r in rows], target)

def apply(rows, thr):
    if thr is None or not rows:
        return 0.0, None, 0
    sel = [r for r in rows if r['conf'] >= thr]
    return (len(sel) / len(rows),
            sum(1 - r['correct'] for r in sel) / len(sel) if sel else None, len(sel))

def boot_ci(test_rows, thr, n_boot=N_BOOT, seed=SEED):
    by_img = defaultdict(list)
    for r in test_rows:
        by_img[r['img']].append(r)
    imgs = sorted(by_img)
    if not imgs or thr is None:
        return None
    rng = np.random.default_rng(seed)
    covs, risks = [], []
    for _ in range(n_boot):
        sample = [r for i in rng.integers(0, len(imgs), len(imgs)) for r in by_img[imgs[i]]]
        cov, risk, _ = apply(sample, thr)
        covs.append(cov)
        if risk is not None:
            risks.append(risk)
    def ci(v):
        return [float(x) for x in np.percentile(v, [2.5, 97.5])] if v else None
    return {'coverage_ci95': ci(covs), 'risk_ci95': ci(risks),
            'n_boot_coverage': len(covs), 'n_boot_nonempty_risk': len(risks),
            'n_boot_empty_selection': len(covs) - len(risks)}

def condition(rows, threshold):
    cov, risk, n = apply(rows, threshold)
    result = {'coverage': cov, 'risk': risk, 'n_selected': n,
              'reachable': threshold is not None,
              'fit_threshold_available': threshold is not None,
              'test_target_met': risk <= 0.05 if risk is not None else None,
              'status': selection_status(threshold, rows, risk, n)}
    if threshold is not None:
        result.update(boot_ci(rows, threshold) or {})
    return result

def transform(rows, T):
    if T is None:
        return []
    return [{**r, 'conf': c} for r, c in zip(rows, temp_scale([r['conf'] for r in rows], T))]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, lines = {}, ['# 选择性风险与阈值迁移', '',
        '阈值仅在图像级 fit 半区选取；test 图像聚类 bootstrap，阈值固定。',
        'coverage 分母为每次抽样的总题数，零选择仍计 coverage=0；空选择的 risk 未定义。',
        '严格单调 power scaling 保持排序与全部可达 risk-coverage 点；不能据此声称改善排序。', '']
    for model, stem in BASEMAP.items():
        path = RESULTS_ROOT / 'raw' / f'{stem}.jsonl'
        if not path.exists():
            continue
        source = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        rows = [{'conf': float(r['conf_token_geo']), 'correct': int(r['correct']),
                 'img': r['image_path'], 'fmt': r['format']} for r in source
                if r.get('status') == 'ok' and r.get('conf_token_geo') is not None and r.get('format') in FM]
        fit = {f: [r for r in rows if r['fmt'] == f and half(r['img']) == 0] for f in FM}
        test = {f: [r for r in rows if r['fmt'] == f and half(r['img']) == 1] for f in FM}
        thresholds = {f: thr_at_risk(fit[f]) for f in FM}
        temps = {f: fit_temperature([r['conf'] for r in fit[f]], [r['correct'] for r in fit[f]]) for f in FM}
        calibrated_thresholds = {f: thr_at_risk(transform(fit[f], temps[f])) for f in FM}
        res = {'thresholds_from_fit': thresholds, 'targets': {},
               'n_fit_per_format': {f: len(fit[f]) for f in FM},
               'n_test_per_format': {f: len(test[f]) for f in FM},
               'temperatures': temps, 'calibration_config': CALIBRATION_CONFIG,
               'calibrated_thresholds_from_fit': calibrated_thresholds,
               'curve_definition': 'all distinct-score threshold boundaries; no tie splitting',
               'bootstrap': {'unit': 'image', 'threshold': 'fixed', 'replicates': N_BOOT, 'seed': SEED}}
        for t in FM:
            cell = {'curve': curve([r['conf'] for r in test[t]], [r['correct'] for r in test[t]]),
                    'conditions': {}, 'calibrated_conditions': {}}
            for s in FM:
                key = 'own' if s == t else f'from_{s}'
                cell['conditions'][key] = condition(test[t], thresholds[s])
                # 源格式校准器与源阈值共同迁移；另报目标自身校准器阈值。
                cell['calibrated_conditions'][key] = condition(transform(test[t], temps[s]), calibrated_thresholds[s])
            res['targets'][t] = cell
        summary[model] = res
        lines += [f'## {model}', '', f"fit/test n: {res['n_fit_per_format']} / {res['n_test_per_format']}", '',
                  '| 目标 | 条件 | coverage | risk | 状态 |', '|---|---|---|---|---|']
        for t in FM:
            for k, v in res['targets'][t]['conditions'].items():
                lines.append(f"| {t} | {k} | {v['coverage']:.4f} | {v['risk']} | {v['status']} |")
        lines.append('')
    (OUT / 'selective_full.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (OUT / 'selective_full.md').write_text('\n'.join(lines), encoding='utf-8')
    print('saved', OUT)

if __name__ == '__main__':
    from analysis_gate import require_analysis_inputs
    require_analysis_inputs('selective_full', RESULTS_ROOT)
    main()
