#!/usr/bin/env python3
"""Paper figures generated only from corrected authoritative JSON files."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calib_metrics import RESULTS_ROOT

R = RESULTS_ROOT
OUT = R / 'paper_figs'
FM = ['A', 'B', 'C']
MODELS = [('qwen500', 'Qwen3-VL-8B', '#2E6FB7'), ('healthgpt500', 'HealthGPT-Pro', '#C0504D'), ('gemma500', 'Gemma-4-E4B', '#7F7F7F')]

def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def row_minimum_mask(values):
    values = np.asarray(values, float)
    return np.isfinite(values) & (values == np.nanmin(values, axis=1, keepdims=True))


def matched_main_cohort(summary_format, bootstrap_model, fmt):
    summary_cohort = summary_format.get('cohort')
    boot_cohort = bootstrap_model.get('cohort_full', {}).get(fmt)
    if not summary_cohort or not boot_cohort or summary_cohort != boot_cohort:
        raise ValueError(f'ECE table/figure cohort identity mismatch: {fmt}')
    return summary_cohort


def matrix():
    path = R / 'rescore_analysis/rescore_summary.json'
    if not path.exists():
        return
    data = json.loads(path.read_text())
    models = [(m, title) for m, title in [('qwen', 'Qwen3-VL-8B'), ('healthgpt', 'HealthGPT-Pro')] if m in data]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(3.4, 2.05), sharey=True, squeeze=False)
    for ax, (m, title) in zip(axes[0], models):
        c = data[m]['ece_canonical']
        M = np.array([[c[s][g] for s in FM] for g in FM], dtype=float)
        minimum = row_minimum_mask(M)
        im = ax.imshow(M, cmap='RdYlGn_r', vmin=0, vmax=max(.5, float(np.nanmax(M))))
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6,
                        fontweight='bold' if minimum[i, j] else 'normal',
                        color='white' if M[i, j] > .36 else '#222222')
        ax.set_xticks(range(3), ['sA', 'sB', 'sC']); ax.set_yticks(range(3), [f'{f} ans' for f in FM])
        ax.set_title(title, fontsize=7); ax.set_xlabel('historical scoring condition', fontsize=6)
    axes[0, 0].set_ylabel('answer source')
    fig.colorbar(im, ax=list(axes[0]), fraction=.028, pad=.03, label='ECE')
    fig.text(.5, -.06, 'Bold: exact row minimum before display rounding; exact ties share bold. Historical descriptive matrix.', ha='center', fontsize=5.5)
    save(fig, 'fig_matrix')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'serif'})
    boot = json.loads((R / 'formal_stats/bootstrap_stats.json').read_text())
    summary = json.loads((R / 'summary/summary.json').read_text())
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    x = np.arange(3); w = .26
    manifest = {'input_root': str(R), 'ece': {}, 'matrix_note': 'row minimum bold; no universal diagonal optimum',
                'actual_note': 'descriptive; no causal attribution when spans/context are not comparable'}
    for i, (key, name, color) in enumerate(MODELS):
        vals, errs = [], []
        if f'slake:{key}' not in boot or key not in summary:
            continue
        manifest['ece'][key] = {}
        for f in FM:
            cohort = matched_main_cohort(summary[key]['formats'][f], boot[f'slake:{key}'], f)
            v = boot[f'slake:{key}']['ci_ece_full'][f]
            point = summary[key]['formats'][f]['ece10']
            if point is None or v['ci95'] is None:
                vals.append(np.nan); errs.append(None)
                manifest['ece'][key][f] = {'point': None, 'ci95': None, 'status': 'no_valid_confidence'}
                continue
            if not np.isclose(point, v['point'], atol=5.1e-5):
                raise ValueError(f'ECE table/figure population mismatch: {key}/{f}')
            vals.append(point)
            # CI端点直接画线，避免偏差校正区间不含中心时使用负yerr。
            lo, hi = v['ci95']; errs.append((lo, hi))
            manifest['ece'][key][f] = {'point': point, 'ci95': [lo, hi], 'population': 'full valid sample', 'cohort': cohort}
        xx = x + (i - 1) * w
        ax.bar(xx, vals, w, label=name, color=color)
        for pos, bounds in zip(xx, errs):
            if bounds is None:
                ax.text(pos, .01, 'NA', rotation=90, ha='center', fontsize=5)
                continue
            lo, hi = bounds
            ax.vlines(pos, lo, hi, color='black', lw=.6); ax.hlines([lo, hi], pos-.025, pos+.025, color='black', lw=.6)
    ax.set_xticks(x, ['A direct', 'B reason+\nanswer', 'C answer+\nconfidence'])
    ax.set_ylabel('ECE (10-bin, full sample)'); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); save(fig, 'fig_ece')
    matrix()
    sf = json.loads((R / 'selective_full/selective_full.json').read_text())
    if 'healthgpt' in sf:
        fig, ax = plt.subplots(figsize=(3.3, 2))
        for f, color in zip(FM, ['#4E79A7', '#E15759', '#59A14F']):
            cur = sf['healthgpt']['targets'][f]['curve']
            ax.plot([p['coverage'] for p in cur], [p['risk'] for p in cur], c=color, label=f'format {f}', lw=1.2)
        ax.axhline(.05, c='k', ls=':', lw=.8); ax.set_xlabel('coverage'); ax.set_ylabel('selective risk')
        ax.set_ylim(0, 1); ax.legend(fontsize=6, frameon=False); fig.tight_layout(); save(fig, 'fig_risk')
    (OUT / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('saved', OUT)

if __name__ == '__main__':
    from analysis_gate import require_analysis_inputs
    require_analysis_inputs('make_paper_figs', RESULTS_ROOT)
    main()
