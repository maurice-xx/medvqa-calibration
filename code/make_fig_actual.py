import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

R = Path('./results')
OUT = Path('./docs/paper_figs')
plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 300, 'font.family': 'serif'})
d = json.load(open(R / 'rescore_actual/actual_vs_parallel.json'))
FM = ['A', 'B', 'C']
names = {'qwen': 'Qwen3-VL-8B', 'healthgpt': 'HealthGPT-Pro', 'gemma4': 'Gemma-4-E4B'}

fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.75), sharey=True)
x = np.arange(3); w = 0.36
for ax, m in zip(axes, ['qwen', 'healthgpt', 'gemma4']):
    per = d[m]['per_format']
    a = [per[f]['mean_conf_actual'] for f in FM]
    p = [per[f]['mean_conf_parallel_diag'] for f in FM]
    ax.bar(x - w/2, a, w, color='#C0504D', label='actual')
    ax.bar(x + w/2, p, w, color='#4E79A7', label='parallel')
    ax.set_xticks(x); ax.set_xticklabels(FM)
    ax.set_title(names[m], fontsize=6.5)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=6)
axes[0].set_ylabel('mean confidence', fontsize=6.5)
axes[0].legend(fontsize=6, frameon=False, loc='upper right')
fig.tight_layout(pad=0.3)
fig.savefig(OUT / 'fig_actual.pdf'); fig.savefig(OUT / 'fig_actual.png')
print('saved fig_actual')
