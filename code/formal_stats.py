#!/usr/bin/env python3
"""Matched full/test image-cluster bootstrap summaries; no import-time writes."""
import json
import numpy as np
from calib_metrics import RESULTS_ROOT, POWER_GRID, CALIBRATION_CONFIG, fit_temperature, temp_scale, ece, img_half, cohort_metadata

RAW = RESULTS_ROOT / 'raw'
RAWV = RESULTS_ROOT / 'raw_vqarad'
OUT = RESULTS_ROOT / 'formal_stats'
FMTS = ['A', 'B', 'C']
BOOT = 1000
SEED = 20261025

def load(path):
    return [r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())
            if r.get('status') == 'ok' and r.get('conf_token_geo') is not None and r.get('format') in FMTS]

def ci(values):
    return [float(v) for v in np.percentile(values, [2.5, 97.5])] if values else None

def evaluate(rows, temps, seed):
    images = sorted({r['image_path'] for r in rows})
    by = {im: [r for r in rows if r['image_path'] == im] for im in images}
    rng = np.random.default_rng(seed)
    def values(sample):
        stats, penalty = {}, {}
        for f in FMTS:
            rs = [r for r in sample if r['format'] == f]
            c = [r['conf_token_geo'] for r in rs]; y = [int(r['correct']) for r in rs]
            stats[f] = ece(c, y) if c else None
            for s in FMTS:
                if s != f and c and temps[s] is not None and temps[f] is not None:
                    penalty[f'{s}->{f}'] = ece(temp_scale(c, temps[s]), y) - ece(temp_scale(c, temps[f]), y)
        return stats, penalty
    points, point_pen = values(rows)
    boot_ece = {f: [] for f in FMTS}; boot_pen = {f'{s}->{t}': [] for s in FMTS for t in FMTS if s != t}
    pairwise = {f'{s}-{t}': [] for i, s in enumerate(FMTS) for t in FMTS[i + 1:]}
    for _ in range(BOOT):
        sample = [r for i in rng.integers(0, len(images), len(images)) for r in by[images[i]]] if images else []
        vals, pens = values(sample)
        for f, v in vals.items():
            if v is not None:
                boot_ece[f].append(v)
        for k in pairwise:
            s, t = k.split('-')
            if vals[s] is not None and vals[t] is not None:
                pairwise[k].append(vals[s] - vals[t])
        for k, v in pens.items():
            boot_pen[k].append(v)
    stats = {f: {'point': points[f], 'mean': points[f], 'bootstrap_mean': float(np.mean(boot_ece[f])) if boot_ece[f] else None,
                  'ci95': ci(boot_ece[f]), 'n_boot': len(boot_ece[f])} for f in FMTS}
    pens = {k: {'point': point_pen.get(k), 'mean': point_pen.get(k), 'ci95': ci(v)} for k, v in boot_pen.items()}
    pairs = {k: {'point': points[k[0]] - points[k[2]] if points[k[0]] is not None and points[k[2]] is not None else None, 'ci95': ci(v)} for k, v in pairwise.items()}
    return stats, pens, pairs

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    datasets = [('slake', RAW, ['qwen500', 'healthgpt500', 'gemma500']),
                ('vqarad', RAWV, ['qwen', 'healthgpt']),
                ('vqamed', RESULTS_ROOT / 'raw_vqamed', ['qwen', 'healthgpt'])]
    for ds, root, models in datasets:
        for model in models:
            path = root / f'{model}.jsonl'
            if not path.exists():
                continue
            rows = load(path)
            temps = {f: fit_temperature([r['conf_token_geo'] for r in rows if r['format'] == f and img_half(r) == 0],
                                        [int(r['correct']) for r in rows if r['format'] == f and img_half(r) == 0]) for f in FMTS}
            test = [r for r in rows if img_half(r) == 1]
            full_stats, _, full_pairs = evaluate(rows, temps, SEED)
            test_stats, pens, pairs = evaluate(test, temps, SEED)
            res = {'raw_ece_full': {f: full_stats[f]['point'] for f in FMTS},
                   'ci_ece_full': full_stats, 'ci_ece': test_stats, 'ci_pairwise_dECE': pairs,
                   'ci_pairwise_dECE_full': full_pairs, 'transfer_penalty_ci': pens, 'temps': temps,
                   'n': {f: sum(r['format'] == f for r in test) for f in FMTS},
                   'n_full': {f: sum(r['format'] == f for r in rows) for f in FMTS},
                   'n_fit': {f: sum(r['format'] == f and img_half(r) == 0 for r in rows) for f in FMTS},
                   'cohort_full': {f: cohort_metadata([r for r in rows if r['format'] == f]) for f in FMTS},
                   'cohort_test': {f: cohort_metadata([r for r in test if r['format'] == f]) for f in FMTS},
                   'metadata': {'calibration_config': CALIBRATION_CONFIG, 'bootstrap_unit': 'image',
                      'bootstrap_replicates': BOOT, 'seed': SEED, 'ci_ece_population': 'test half',
                      'ci_ece_full_population': 'full valid sample, matching summary/summary.json',
                      'center': 'observed statistic, not bootstrap mean', 'transfer_interval': 'conditional on fitted calibrators'}}
            out[f'{ds}:{model}'] = res
            print(ds, model, 'fit/test', res['n_fit'], res['n'])
    (OUT / 'bootstrap_stats.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('saved', OUT)

if __name__ == '__main__':
    from analysis_gate import require_analysis_inputs
    require_analysis_inputs('formal_stats', RESULTS_ROOT)
    main()
