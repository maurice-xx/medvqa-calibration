#!/usr/bin/env python3
"""P0-#3: 校准修复的稳定性（重复图像级划分 × power/logistic/isotonic）。

每个重复：随机把图像对半拆 fit/test（图像级，同图多题不跨折），
在 fit 上拟合，在 test 上算 ECE / Brier / NLL。
报告 10 次重复的均值与 2.5–97.5 分位；并输出可靠性图（跨重复平均的分箱曲线）。

输出 results/stability/{stability.json, stability.md, fig_reliability.png}
"""
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from calib_metrics import RESULTS_ROOT, POWER_GRID, CALIBRATION_CONFIG
ROOT = Path(__file__).resolve().parent.parent
OUT = RESULTS_ROOT / 'stability'
FM = ['A', 'B', 'C']
BASEMAP = {'qwen': 'qwen500', 'healthgpt': 'healthgpt500', 'gemma': 'gemma500'}
NAMES = {'qwen': 'Qwen3-VL-8B', 'healthgpt': 'HealthGPT-Pro-8B', 'gemma': 'Gemma-4-E4B'}
REPS = 10
SEED = 20261103


def ece(conf, corr, n=10):
    conf = np.asarray(conf, float); corr = np.asarray(corr, float)
    if len(conf) == 0:
        return float('nan')
    idx = np.clip((conf * n).astype(int), 0, n - 1)
    e = 0.0
    for b in range(n):
        m = idx == b
        if m.sum():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def brier(conf, corr):
    c = np.asarray(conf, float); y = np.asarray(corr, float)
    return float(np.mean((c - y) ** 2))


def nll(conf, corr):
    p = np.clip(np.asarray(conf, float), 1e-6, 1 - 1e-6)
    y = np.asarray(corr, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_power(c, y):
    if len(c) == 0:
        return None
    best_T, best = None, float('inf')
    for T in POWER_GRID:
        e = ece(np.clip(np.power(c, 1 / T), 0, 1), y)
        if e < best:
            best, best_T = e, float(T)
    return lambda q, T=best_T: np.clip(np.power(np.asarray(q, float), 1 / T), 0, 1)


def fit_logistic(c, y):
    X = np.column_stack([np.ones(len(c)), logit(c)])
    y = np.asarray(y, float)
    ridge = np.diag([0.0, 1e-2])
    w = np.zeros(2)
    for _ in range(50):
        p = sigmoid(X @ w)
        W = p * (1 - p) + 1e-6
        g = X.T @ (p - y) + ridge @ w
        H = (X * W[:, None]).T @ X + ridge
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return lambda q, w=w: sigmoid(w[0] + w[1] * logit(q))


def fit_isotonic(c, y):
    from sklearn.isotonic import IsotonicRegression
    model = IsotonicRegression(increasing=True, out_of_bounds='clip', y_min=0.0, y_max=1.0)
    model.fit(np.asarray(c, float), np.asarray(y, float))
    return model.predict


def load(model):
    rows = [json.loads(l) for l in open(RESULTS_ROOT / f'raw/{BASEMAP[model]}.jsonl')]
    return [r for r in rows if r.get('status') == 'ok' and r.get('conf_token_geo') is not None
            and r.get('format') in FM]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary, lines = {}, ['# 校准修复稳定性（10 次图像级划分）', '',
        '区间为10次重复划分的经验2.5–97.5分位，非置信区间。', '']
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    for ax, (model, name) in zip(axes, [(m, NAMES[m]) for m in ('qwen', 'healthgpt', 'gemma')]):
        rows = load(model)
        imgs = sorted({r['image_path'] for r in rows})
        idx_by_img = {im: [r for r in rows if r['image_path'] == im] for im in imgs}
        res = {f: {k: {'ece': [], 'brier': [], 'nll': []} for k in
                   ('raw', 'power', 'logistic', 'isotonic')} for f in FM}
        rel = {k: np.zeros((10, 3)) for k in ('raw', 'power', 'logistic', 'isotonic')}
        rel_cnt = {k: np.zeros(10) for k in rel}
        for rep in range(REPS):
            perm = rng.permutation(len(imgs))
            cut = len(imgs) // 2
            fit_imgs = {imgs[i] for i in perm[:cut]}
            fit_rows = [r for im in sorted(fit_imgs) for r in idx_by_img[im]]
            test_rows = [r for im in imgs if im not in fit_imgs for r in idx_by_img[im]]
            for f in FM:
                fr = [r for r in fit_rows if r['format'] == f]
                tr = [r for r in test_rows if r['format'] == f]
                if len(fr) < 10 or len(tr) < 10:
                    continue
                fc = [r['conf_token_geo'] for r in fr]; fy = [int(r['correct']) for r in fr]
                tc = np.array([r['conf_token_geo'] for r in tr]); ty = np.array([int(r['correct']) for r in tr])
                fns = {'raw': lambda q: np.asarray(q, float),
                       'power': fit_power(fc, fy),
                       'logistic': fit_logistic(fc, fy),
                       'isotonic': fit_isotonic(fc, fy)}
                for k, fn in fns.items():
                    pred = np.clip(fn(tc), 0, 1)
                    res[f][k]['ece'].append(ece(pred, ty))
                    res[f][k]['brier'].append(brier(pred, ty))
                    res[f][k]['nll'].append(nll(pred, ty))
                    bins = np.clip((pred * 10).astype(int), 0, 9)
                    for b in range(10):
                        m = bins == b
                        if m.sum():
                            rel[k][b, 0] += pred[m].mean(); rel[k][b, 1] += ty[m].mean(); rel[k][b, 2] += 1
        summary[model] = {'_metadata': {'calibration_config': CALIBRATION_CONFIG,
            'logistic': {'logit_clip': [1e-4, 1 - 1e-4], 'sigmoid_z_clip': [-30, 30],
                         'l2_slope': 1e-2, 'intercept_penalized': False, 'max_iterations': 50},
            'isotonic': {'implementation': 'sklearn.IsotonicRegression', 'increasing': True, 'out_of_bounds': 'clip'},
            'interval': 'empirical 2.5–97.5 percentile over 10 repeated image splits; not a confidence interval',
            'repetitions': REPS, 'seed': SEED,
            'rank_note': 'strict monotonic power preserves ranked risk coverage; isotonic can merge ties; logistic slope is unconstrained'}}
        for f in FM:
            summary[model][f] = {}
            for k in ('raw', 'power', 'logistic', 'isotonic'):
                v = res[f][k]
                if not v['ece']:
                    continue
                summary[model][f][k] = {
                    'status': 'ok', 'n_repetitions': len(v['ece']),
                    'ece_replicates': v['ece'],
                    'brier_replicates': v['brier'], 'nll_replicates': v['nll'],
                    'ece_mean': round(float(np.mean(v['ece'])), 4),
                    'ece_empirical_interval95': [round(float(np.percentile(v['ece'], 2.5)), 4),
                                 round(float(np.percentile(v['ece'], 97.5)), 4)],
                    'brier_mean': round(float(np.mean(v['brier'])), 4),
                    'nll_mean': round(float(np.mean(v['nll'])), 4),
                }
        # 可靠性图（跨重复平均）
        ax.plot([0, 1], [0, 1], 'k:', lw=0.8)
        for k, style in [('raw', 'o-'), ('power', 's--'), ('logistic', '^:'), ('isotonic', 'd-.')]:
            xs, ys = [], []
            for b in range(10):
                if rel[k][b, 2] > 0:
                    xs.append(rel[k][b, 0] / rel[k][b, 2]); ys.append(rel[k][b, 1] / rel[k][b, 2])
            if xs:
                ax.plot(xs, ys, style, ms=3, lw=1, label=k)
        ax.set_title(name, fontsize=9); ax.set_xlabel('predicted confidence')
        lines += [f'## {name}', '']
        for f in FM:
            lines.append(f'**格式 {f}**')
            lines.append('')
            lines.append('| 方法 | ECE [经验95%区间] | Brier | NLL |')
            lines.append('|---|---|---|---|')
            for k, v in summary[model][f].items():
                lines.append(f"| {k} | {v['ece_mean']} [{v['ece_empirical_interval95'][0]}, {v['ece_empirical_interval95'][1]}] | "
                             f"{v['brier_mean']} | {v['nll_mean']} |")
            lines.append('')
    axes[0].set_ylabel('empirical accuracy')
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_reliability.png', dpi=300, bbox_inches='tight')
    (OUT / 'stability.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (OUT / 'stability.md').write_text('\n'.join(lines), encoding='utf-8')
    print('saved', OUT)


if __name__ == '__main__':
    from analysis_gate import require_analysis_inputs
    require_analysis_inputs('stability', RESULTS_ROOT)
    main()
