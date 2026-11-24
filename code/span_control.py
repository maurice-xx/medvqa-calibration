#!/usr/bin/env python3
"""Short-span descriptive sensitivity; not a causal length control."""
import json
from calib_metrics import RESULTS_ROOT, ece

OUT = RESULTS_ROOT / 'span_control'
FM = ['A', 'B', 'C']
BASEMAP = {'qwen': 'qwen500', 'healthgpt': 'healthgpt500', 'gemma4': 'gemma500'}

def span_length(r):
    for field in ('answer_span_token_count', 'conf_span_len', 'span_len', 'answer_span_len', 'n_answer_tokens'):
        if isinstance(r.get(field), (int, float)):
            return r[field]
    if isinstance(r.get('span_token_ids'), list) and r['span_token_ids']:
        return len(r['span_token_ids'])
    return None

def stats(rows):
    if not rows:
        return [None, None, None]
    c = [r['conf_token_geo'] for r in rows]; y = [int(r['correct']) for r in rows]
    return [sum(y) / len(y), sum(c) / len(c), ece(c, y)]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, lines = {}, ['# 短答案span敏感性（≤5 token）', '',
        '子集由答案长度条件选择，组间题目组成可能不同；此描述性比较不能排除长度混杂或证明因果效应。', '']
    for m, stem in BASEMAP.items():
        path = RESULTS_ROOT / 'raw' / f'{stem}.jsonl'
        if not path.exists():
            continue
        rows = [r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip()) if r.get('status') == 'ok' and r.get('conf_token_geo') is not None]
        res = {}
        lines += [f'## {m}', '', '| 格式 | full acc/conf/ECE | ≤5 acc/conf/ECE | n full/sub | 缺长度 |', '|---|---|---|---|---|']
        for f in FM:
            full = [r for r in rows if r['format'] == f]
            sub = [r for r in full if span_length(r) is not None and span_length(r) <= 5]
            res[f] = {'full': stats(full), 'sub': stats(sub), 'n_full': len(full), 'n_sub': len(sub),
                      'n_missing_length': sum(span_length(r) is None for r in full)}
            v = res[f]
            lines.append(f"| {f} | {v['full']} | {v['sub']} | {len(full)}/{len(sub)} | {v['n_missing_length']} |")
        summary[m] = res
    (OUT / 'span_control.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (OUT / 'span_control.md').write_text('\n'.join(lines), encoding='utf-8')
    print('saved', OUT)

if __name__ == '__main__':
    from analysis_gate import require_analysis_inputs
    require_analysis_inputs('span_control', RESULTS_ROOT)
    main()
