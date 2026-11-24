#!/usr/bin/env python3
"""Supplemental descriptive actual/historical score figure."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calib_metrics import RESULTS_ROOT
from make_paper_figs import matrix

OUT = RESULTS_ROOT / 'paper_figs'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = RESULTS_ROOT / 'rescore_actual/actual_vs_parallel.json'
    if not path.exists():
        return
    data = json.loads(path.read_text())
    names = {'qwen': 'Qwen3-VL-8B', 'healthgpt': 'HealthGPT-Pro', 'gemma4': 'Gemma-4-E4B'}
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.5), sharey=True)
    for ax, m in zip(axes, names):
        per = data.get(m, {}).get('per_format', {})
        fm = [f for f in ['A', 'B', 'C'] if f in per]
        x = np.arange(len(fm)); w = .36
        ax.bar(x-w/2, [per[f]['mean_conf_actual'] for f in fm], w, color='#C0504D', label='actual scorer')
        ax.bar(x+w/2, [per[f]['mean_conf_parallel_diag'] for f in fm], w, color='#4E79A7', label='historical scorer')
        ax.set_xticks(x, fm); ax.set_title(names[m], fontsize=8); ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel('mean confidence', fontsize=8)
    axes[0].legend(fontsize=6, frameon=False, loc='upper left')
    fig.text(.5, .01, 'Descriptive only: historical scoring contexts / token spans are not verified comparable; no causal attribution.', ha='center', fontsize=6)
    fig.tight_layout(rect=[0, .08, 1, 1])
    fig.savefig(OUT / 'fig_actual.pdf'); fig.savefig(OUT / 'fig_actual.png', dpi=300); plt.close(fig)
    matrix()
    print('saved', OUT)

if __name__ == '__main__':
    from analysis_gate import require_analysis_inputs
    require_analysis_inputs('fix_figs', RESULTS_ROOT)
    main()
