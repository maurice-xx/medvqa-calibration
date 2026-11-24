#!/usr/bin/env python3
"""Image-cluster GEE and tie-aware monotone calibration comparison."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from calib_metrics import RESULTS_ROOT, CALIBRATION_CONFIG, fit_temperature, temp_scale, ece, img_half
from stability import fit_isotonic

OUT = RESULTS_ROOT / 'formal_stats'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for ds, folder, models in [('slake', 'raw', ['qwen500', 'healthgpt500', 'gemma500']),
                               ('vqarad', 'raw_vqarad', ['qwen', 'healthgpt']),
                               ('vqamed', 'raw_vqamed', ['qwen', 'healthgpt'])]:
        for m in models:
            path = RESULTS_ROOT / folder / f'{m}.jsonl'
            if not path.exists():
                continue
            rows = [r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())
                    if r.get('status') == 'ok' and r.get('conf_token_geo') is not None and r.get('format') in ['A', 'B', 'C']]
            df = pd.DataFrame([{'correct': int(r['correct']), 'fmt': r['format'], 'img': r['image_path']} for r in rows])
            X = pd.DataFrame({'fmt_B': (df['fmt'] == 'B').astype(float),
                              'fmt_C': (df['fmt'] == 'C').astype(float), 'const': 1.0})
            try:
                fit = sm.GEE(df['correct'].values, X.values, groups=df['img'].values,
                             family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
                coefs = {n: {'beta': float(fit.params[i]), 'OR': float(np.exp(fit.params[i])),
                             'p': float(fit.pvalues[i]),
                             'ci95_OR': [float(v) for v in np.exp(fit.conf_int()[i])]}
                         for i, n in enumerate(['fmt_B', 'fmt_C', 'const'])}
            except Exception as error:
                coefs = {'error': f'{type(error).__name__}: {error}'}
            metrics = {k: {} for k in ['raw', 'temperature', 'isotonic']}
            for f in ['A', 'B', 'C']:
                fr = [r for r in rows if r['format'] == f and img_half(r) == 0]
                tr = [r for r in rows if r['format'] == f and img_half(r) == 1]
                fc = [r['conf_token_geo'] for r in fr]; fy = [int(r['correct']) for r in fr]
                tc = [r['conf_token_geo'] for r in tr]; ty = [int(r['correct']) for r in tr]
                for k in metrics:
                    metrics[k][f] = None
                if not fr or not tr:
                    continue
                metrics['raw'][f] = ece(tc, ty)
                metrics['temperature'][f] = ece(temp_scale(tc, fit_temperature(fc, fy)), ty)
                metrics['isotonic'][f] = ece(list(fit_isotonic(fc, fy)(tc)), ty)
            out[f'{ds}:{m}'] = {'gee_logistic': coefs, 'test_ece': metrics,
                'calibration_config': CALIBRATION_CONFIG,
                'isotonic_config': {'implementation': 'sklearn.IsotonicRegression', 'increasing': True, 'out_of_bounds': 'clip'},
                'gee': {'cluster': 'image_path', 'reference_format': 'A', 'family': 'binomial', 'correlation': 'exchangeable'}}
            print(ds, m, metrics)
    (OUT / 'gee_isotonic.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('saved', OUT)

if __name__ == '__main__':
    from analysis_gate import require_analysis_inputs
    require_analysis_inputs('gee_stats', RESULTS_ROOT)
    main()
