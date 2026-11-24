import json, math, hashlib, random
from pathlib import Path

RAW = Path('./results/raw')
OUT = Path('./results/sensitivity')
MODELS = ['qwen500', 'healthgpt500', 'gemma500']
FMTS = ['A', 'B', 'C']
SIZES = [10, 25, 50, 100, 200, 400]
REPS = 20
SEED = 20261022

def ece(conf, correct, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for c, y in zip(conf, correct):
        bins[min(int(c*n_bins), n_bins-1)].append((c, y))
    e = 0.0
    for b in bins:
        if b:
            e += len(b)/len(conf) * abs(sum(y for _,y in b)/len(b) - sum(c for c,_ in b)/len(b))
    return e

def temp_scale(conf, T):
    return [max(min(c ** (1.0/T), 1.0), 0.0) for c in conf]

def fit_T(conf, correct):
    best_T, best_e = 1.0, float('inf')
    T = 0.05
    while T <= 20.0:
        e = ece(temp_scale(conf, T), correct)
        if e < best_e: best_e, best_T = e, T
        T *= 1.15
    return best_T

def half(r):
    return int(hashlib.md5(r['image_path'].encode()).hexdigest(), 16) % 2

OUT.mkdir(parents=True, exist_ok=True)
summary = {}
for model in MODELS:
    rows = [json.loads(l) for l in open(RAW / f'{model}.jsonl')]
    fmt_data = {}
    for fmt in FMTS:
        rs = [r for r in rows if r['format'] == fmt and r.get('status') == 'ok' and r.get('conf_token_geo') is not None]
        fit = [r for r in rs if half(r) == 0]
        test = [r for r in rs if half(r) == 1]
        fmt_data[fmt] = (fit, test,
                         [r['conf_token_geo'] for r in test],
                         [int(r['correct']) for r in test])
    test_conf = {f: fmt_data[f][2] for f in FMTS}
    test_corr = {f: fmt_data[f][3] for f in FMTS}
    raw_ece = {f: round(ece(test_conf[f], test_corr[f]), 4) for f in FMTS}

    res = {'raw_ece_test': raw_ece, 'in_format': {}, 'cross_format': {}}
    rng = random.Random(SEED)
    for src in FMTS:
        fit_pool, _, _, _ = fmt_data[src]
        conf_f = [r['conf_token_geo'] for r in fit_pool]
        corr_f = [int(r['correct']) for r in fit_pool]
        # 温度上限随拟合集大小缩放（小集拟合的 T 方差大，网格上限 20 保持一致）
        for N in SIZES:
            if N > len(fit_pool):
                continue
            in_fmt, cross = [], {t: [] for t in FMTS if t != src}
            for rep in range(REPS):
                idx = rng.sample(range(len(fit_pool)), min(N, len(fit_pool)))
                T = fit_T([conf_f[i] for i in idx], [corr_f[i] for i in idx])
                in_fmt.append(ece(temp_scale(test_conf[src], T), test_corr[src]))
                for tgt in FMTS:
                    if tgt != src:
                        cross[tgt].append(ece(temp_scale(test_conf[tgt], T), test_corr[tgt]))
            def ms(v):
                m = sum(v)/len(v)
                s = (sum((x-m)**2 for x in v)/len(v)) ** 0.5
                return round(m, 4), round(s, 4)
            m_in, s_in = ms(in_fmt)
            res['in_format'][f'{src}@N={N}'] = {'mean': m_in, 'std': s_in}
            for tgt, v in cross.items():
                m_c, s_c = ms(v)
                res['cross_format'][f'{src}->{tgt}@N={N}'] = {'mean': m_c, 'std': s_c}
    summary[model] = res
    print(f'== {model} raw_ece_test: {raw_ece}')
    for k, v in res['in_format'].items():
        print(f'  {k}: {v["mean"]} ± {v["std"]}')
(OUT / 'fit_sensitivity.json').write_text(json.dumps(summary, indent=2))
print('saved', OUT / 'fit_sensitivity.json')
