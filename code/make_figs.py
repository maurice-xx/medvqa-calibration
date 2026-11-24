import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

R = Path('./results')
FIG = R / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
slake = json.load(open(R / 'summary/summary.json'))
vqarad = json.load(open(R / 'summary_vqarad/summary.json'))
boot = json.load(open(R / 'formal_stats/bootstrap_stats.json'))
sens = json.load(open(R / 'sensitivity/fit_sensitivity.json'))
FMTS = ['A', 'B', 'C']

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 300})
colors = {'qwen500': '#2E6FB7', 'healthgpt500': '#C0504D', 'gemma500': '#7F7F7F',
          'qwen': '#2E6FB7', 'healthgpt': '#C0504D'}
labels = {'qwen500': 'Qwen3-VL-8B', 'healthgpt500': 'HealthGPT-Pro-8B',
          'gemma500': 'Gemma-4-E4B', 'qwen': 'Qwen3-VL-8B', 'healthgpt': 'HealthGPT-Pro-8B'}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for ax, (name, data, keymap, title) in zip(axes, [
    ('slake', slake, ['qwen500', 'healthgpt500', 'gemma500'], '(a) SLAKE (432 q)'),
    ('vqarad', vqarad, ['qwen', 'healthgpt'], '(b) VQA-RAD (150 q)')]):
    x = np.arange(len(FMTS)); w = 0.8 / len(keymap)
    for i, m in enumerate(keymap):
        vals, errs = [], []
        for f in FMTS:
            v = data[m]['formats'][f]['ece10']
            ci = boot[f'{name}:{m}']['ci_ece'][f]['ci95']
            vals.append(v); errs.append([v - ci[0], ci[1] - v])
        ax.bar(x + i * w - 0.4 + w/2, vals, w, label=labels[m], color=colors[m],
               yerr=np.array(errs).T, capsize=2, error_kw={'lw': 0.8})
    ax.set_xticks(x); ax.set_xticklabels(['A direct', 'B reason+\nanswer', 'C answer+\nconfidence'])
    ax.set_title(title, fontsize=9)
    ax.set_ylabel('ECE (10-bin)')
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(FIG / 'fig2_ece_by_format.png', bbox_inches='tight')
plt.close(fig)

pen = boot['slake:healthgpt500']['transfer_penalty_ci']
M = np.zeros((3, 3))
for i, s in enumerate(FMTS):
    for j, t in enumerate(FMTS):
        if s == t:
            M[i, j] = 0.0
        else:
            M[i, j] = pen[f'{s}->{t}']['mean']
fig, ax = plt.subplots(figsize=(3.2, 2.6))
im = ax.imshow(M, cmap='Reds', vmin=0, vmax=max(0.25, M.max()))
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{M[i,j]:.3f}', ha='center', va='center', fontsize=8,
                color='white' if M[i, j] > 0.15 else '#333333')
ax.set_xticks(range(3)); ax.set_xticklabels(FMTS)
ax.set_yticks(range(3)); ax.set_yticklabels(FMTS)
ax.set_xlabel('target format'); ax.set_ylabel('calibrator fit on')
ax.set_title('HealthGPT-Pro, SLAKE\ntransfer penalty ΔECE', fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(FIG / 'fig3_transfer_penalty.png', bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.6, 2.6))
sizes = [10, 25, 50, 100, 200]
mk = {'A': 'o', 'B': 's', 'C': '^'}
col = {'A': '#4E79A7', 'B': '#E15759', 'C': '#59A14F'}
raw = sens['healthgpt500']['raw_ece_test']
for f in FMTS:
    ys = [sens['healthgpt500']['in_format'].get(f'{f}@N={n}', {}).get('mean') for n in sizes]
    ax.plot(sizes, ys, marker=mk[f], ms=4, lw=1.2, color=col[f], label=f'format {f}')
    ax.axhline(raw[f], color=col[f], ls=':', lw=0.9)
ax.set_xscale('log'); ax.set_xticks(sizes); ax.set_xticklabels(sizes)
ax.set_xlabel('calibration samples per format'); ax.set_ylabel('test ECE')
ax.set_title('HealthGPT-Pro, SLAKE: recalibration convergence\n(dotted = uncalibrated)', fontsize=9)
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(FIG / 'fig4_fit_sensitivity.png', bbox_inches='tight')
plt.close(fig)
print('figures:', [p.name for p in sorted(FIG.glob('*.png'))])
