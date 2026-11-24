#!/usr/bin/env python3
"""Actual versus historical parallel scoring, explicitly noncausal if spans differ."""
import json
import numpy as np
from calib_metrics import RESULTS_ROOT
from rescore_analysis import pearson

FM = ['A', 'B', 'C']
BASEMAP = {'qwen': 'qwen500', 'healthgpt': 'healthgpt500', 'gemma4': 'gemma500'}

def read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else None

def main():
    out = RESULTS_ROOT / 'rescore_actual'; out.mkdir(parents=True, exist_ok=True)
    summary, lines = {}, ['# 实际提示与历史平行评分诊断', '',
        '历史平行评分为user-side suffix条件。除非逐题token span一致且评分上下文验证通过，否则均值差仅为描述性对比，不能归因为上下文因果效应。', '']
    for model, stem in BASEMAP.items():
        act = read(out / f'{model}.jsonl')
        par = read(RESULTS_ROOT / 'rescore' / f'{model}.jsonl')
        base = read(RESULTS_ROOT / 'raw' / f'{stem}.jsonl')
        if act is None or par is None or base is None:
            summary[model] = {'status': 'missing_input', 'per_format': {}}
            continue
        stored = {(r['qid'], r['format']): r for r in base if r.get('status') == 'ok'}
        actual = {(r['qid'], r['g']): r for r in act if r.get('status') == 'ok' and r.get('conf_recomputed') is not None}
        parallel = {(r['qid'], r['g'], r['s']): r for r in par if r.get('status', 'ok') == 'ok' and r.get('conf') is not None}
        per = {}
        for f in FM:
            keys = [k for k in actual if k[1] == f and k in stored and (k[0], f, f) in parallel and stored[k].get('conf_token_geo') is not None]
            if not keys:
                continue
            a = [actual[k]['conf_recomputed'] for k in keys]
            p = [parallel[(k[0], f, f)]['conf'] for k in keys]
            s = [stored[k]['conf_token_geo'] for k in keys]
            pa = [parallel[(k[0], f, 'A')]['conf'] for k in keys if (k[0], f, 'A') in parallel]
            comparable = []
            for k in keys:
                ar = actual[k]; pr = parallel[(k[0], f, f)]
                # 不将两个缺失字段误判为相同span。
                comparable.append(ar.get('span_token_ids') is not None and ar.get('span_token_ids') == pr.get('span_token_ids'))
            per[f] = {'n': len(keys), 'mean_conf_actual': float(np.mean(a)), 'mean_conf_stored': float(np.mean(s)),
                'mean_conf_parallel_diag': float(np.mean(p)), 'mean_conf_parallel_sA': float(np.mean(pa)) if pa else None,
                'rho_actual_vs_parallel_diag': pearson(a, p), 'mean_abs_gap_actual_vs_parallel_diag': float(np.mean(np.abs(np.array(a) - p))),
                'n_verified_comparable_spans': sum(comparable), 'all_spans_verified_comparable': all(comparable),
                'interpretation': 'descriptive comparison only; span/context mismatch not a causal effect'}
        diffs = [abs(r['conf_recomputed'] - stored[k]['conf_token_geo']) for k, r in actual.items() if k in stored and stored[k].get('conf_token_geo') is not None]
        # 字段标注为历史生成结果复算差；corrected span改变后不再叫确定性验证。
        summary[model] = {'status': 'descriptive_only', 'per_format': per,
            'determinism': {'n': len(diffs), 'max_abs_diff': max(diffs) if diffs else None,
                            'mean_abs_diff': float(np.mean(diffs)) if diffs else None,
                            'interpretation': 'difference against corrected stored span; not a determinism test unless spans match'},
            'causal_interpretation_allowed': False}
        lines += [f'## {model}', '', '| 格式 | n | 实际 | 校正存储 | 历史平行 | 可验证同span数 |', '|---|---|---|---|---|---|']
        for f, v in per.items():
            lines.append(f"| {f} | {v['n']} | {v['mean_conf_actual']:.4f} | {v['mean_conf_stored']:.4f} | {v['mean_conf_parallel_diag']:.4f} | {v['n_verified_comparable_spans']} |")
    (out / 'actual_vs_parallel.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'actual_vs_parallel.md').write_text('\n'.join(lines), encoding='utf-8')
    print('saved', out)

if __name__ == '__main__':
    from analysis_gate import require_analysis_inputs
    require_analysis_inputs('actual_analysis', RESULTS_ROOT)
    main()
