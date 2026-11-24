import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = './results'
s = json.load(open(f'{R}/rescore_analysis/rescore_summary.json'))
FM = ['A', 'B', 'C']
names = {'qwen': 'Qwen3-VL-8B', 'healthgpt': 'HealthGPT-Pro-8B', 'gemma4': 'Gemma-4-E4B'}

fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.7), sharey=True)
for ax, m in zip(axes, ['qwen', 'healthgpt', 'gemma4']):
    if m not in s:
        continue
    canon = s[m]['ece_canonical']
    M = np.array([[canon[s_][g] for s_ in FM] for g in FM])  # rows=g (answer source), cols=s (context)
    im = ax.imshow(M, cmap='RdYlGn_r', vmin=0, vmax=0.55)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center', fontsize=8,
                    fontweight='bold' if i == j else 'normal',
                    color='white' if M[i, j] > 0.38 else '#222222')
    ax.set_xticks(range(3)); ax.set_xticklabels(["sA", "sB", "sC"])
    ax.set_yticks(range(3)); ax.set_yticklabels([f'{f} ans' for f in FM])
    ax.set_title(names[m], fontsize=9)
    ax.set_xlabel('scoring context')
axes[0].set_ylabel('answer from format')
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label('ECE', fontsize=8)
cb.ax.tick_params(labelsize=7)
fig.savefig(f'{R}/figures/fig3_rescore_matrix.png', dpi=300, bbox_inches='tight')
print('saved fig3')
