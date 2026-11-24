#!/usr/bin/env python3
"""Question-subsampled learning curves with paired source-direction penalties."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calib_metrics import RESULTS_ROOT, CALIBRATION_CONFIG, fit_temperature, temp_scale, ece
from selective_full import half

OUT = RESULTS_ROOT / 'learning_curve'
FM = ['A', 'B', 'C']
BASEMAP = {'qwen': 'qwen500', 'healthgpt': 'healthgpt500', 'gemma': 'gemma500'}
NAMES = {'qwen': 'Qwen3-VL-8B', 'healthgpt': 'HealthGPT-Pro-8B', 'gemma': 'Gemma-4-E4B'}
SIZES = [10, 25, 50, 100, 200]
REPS = 8
SEED = 20261103

def summary(v):
    return [float(np.mean(v)), *[float(x) for x in np.percentile(v, [2.5, 97.5])]] if v else None

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result, lines = {}, ['# 校准样本量曲线', '',
        '每点8次无放回question抽样；fit/test按image隔离。区间为8次重复的经验2.5–97.5分位，非置信区间。',
        '每rep先计算各source→target与同rep目标自身校准的配对差；平均source后再跨rep求分位。', '']
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 4.6), sharex=True)
    for col, (model, stem) in enumerate(BASEMAP.items()):
        path = RESULTS_ROOT / 'raw' / f'{stem}.jsonl'
        if not path.exists():
            continue
        rows = [r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())
                if r.get('status') == 'ok' and r.get('conf_token_geo') is not None and r.get('format') in FM]
        fit = {f: sorted([r for r in rows if r['format'] == f and half(r['image_path']) == 0], key=lambda r: str(r['qid'])) for f in FM}
        test = {f: [r for r in rows if r['format'] == f and half(r['image_path']) == 1] for f in FM}
        res = {f: {'in': [], 'pen': {}, 'source_direction_penalties': {}, 'replicate_details': {}} for f in FM}
        res['_metadata'] = {'n_fit_per_format': {f: len(fit[f]) for f in FM},
            'n_test_per_format': {f: len(test[f]) for f in FM}, 'sampling_unit': 'question',
            'split_unit': 'image', 'repetitions': REPS, 'seed': SEED,
            'interval': 'empirical 2.5–97.5 percentile across 8 replicates; not a confidence interval',
            'calibration_config': CALIBRATION_CONFIG}
        rng = np.random.default_rng(SEED)
        for N in SIZES:
            reps = []
            for rep in range(REPS):
                temps, ids = {}, {}
                for f in FM:
                    pool = fit[f]
                    idx = rng.choice(len(pool), N, replace=False) if len(pool) >= N else []
                    picked = [pool[i] for i in idx]
                    temps[f] = fit_temperature([r['conf_token_geo'] for r in picked], [int(r['correct']) for r in picked]) if picked else None
                    ids[f] = [r['qid'] for r in picked]
                scores = {}
                for t in FM:
                    scores[t] = {}
                    for s in FM:
                        scores[t][s] = ece(temp_scale([r['conf_token_geo'] for r in test[t]], temps[s]), [int(r['correct']) for r in test[t]]) if temps[s] is not None and test[t] else None
                reps.append({'replicate': rep, 'temperatures': temps, 'selected_qids': ids, 'ece_by_target_source': scores})
            for t in FM:
                inside = [r['ece_by_target_source'][t][t] for r in reps if r['ece_by_target_source'][t][t] is not None]
                by_source = {s: [] for s in FM if s != t}
                average_pen = []
                details = []
                for r in reps:
                    own = r['ece_by_target_source'][t][t]
                    penalties = {}
                    for s in by_source:
                        transferred = r['ece_by_target_source'][t][s]
                        if own is not None and transferred is not None:
                            penalties[s] = transferred - own
                            by_source[s].append(transferred - own)
                    avg = float(np.mean(list(penalties.values()))) if len(penalties) == len(by_source) else None
                    if avg is not None:
                        average_pen.append(avg)
                    details.append({**r, 'paired_source_penalties': penalties, 'source_mean_penalty': avg})
                res[t]['in'].append(summary(inside))
                res[t]['pen'][N] = summary(average_pen)
                res[t]['source_direction_penalties'][N] = {s: {'summary': summary(v), 'replicates': v, 'n_replicates': len(v)} for s, v in by_source.items()}
                res[t]['replicate_details'][N] = details
        result[model] = res
        lines += [f'## {NAMES[model]}', '', f"实际fit/test题数: {res['_metadata']['n_fit_per_format']} / {res['_metadata']['n_test_per_format']}", '',
                  '| 目标 | N | 格式内ECE [经验区间] | 源平均迁移惩罚 [经验区间] |', '|---|---|---|---|']
        for t in FM:
            for i, N in enumerate(SIZES):
                lines.append(f"| {t} | {N} | {res[t]['in'][i]} | {res[t]['pen'][N]} |")
            for row, values in enumerate(([v[0] if v else np.nan for v in res[t]['in']], [res[t]['pen'][N][0] if res[t]['pen'][N] else np.nan for N in SIZES])):
                axes[row, col].plot(SIZES, values, 'o-', ms=3, lw=1, label=f'format {t}')
        axes[0, col].set_title(NAMES[model], fontsize=9)
        axes[0, col].set_ylabel('in-format ECE'); axes[1, col].set_ylabel('source-mean transfer penalty')
        axes[1, col].set_xlabel('questions per format')
        for ax in axes[:, col]:
            ax.set_xscale('log'); ax.set_xticks(SIZES); ax.set_xticklabels(SIZES); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / 'fig_learning_curve.png', dpi=300); plt.close(fig)
    (OUT / 'learning_curve.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (OUT / 'learning_curve.md').write_text('\n'.join(lines), encoding='utf-8')
    print('saved', OUT)

if __name__ == '__main__':
    from analysis_gate import require_analysis_inputs
    require_analysis_inputs('learning_curve', RESULTS_ROOT)
    main()
