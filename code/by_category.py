#!/usr/bin/env python3
"""Per-category descriptive metrics; small strata are exploratory."""
import json
from calib_metrics import RESULTS_ROOT, ece

OUT = RESULTS_ROOT / 'by_category'
FM = ['A', 'B', 'C']
MODELS = [('qwen500', 'Qwen3-VL-8B'), ('healthgpt500', 'HealthGPT-Pro-8B'), ('gemma500', 'Gemma-4-E4B')]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result, lines = [], ['# 分题型格式效应（描述性）', '', 'n<30标为探索性；此处不作因果归因。', '']
    for key, name in MODELS:
        path = RESULTS_ROOT / 'raw' / f'{key}.jsonl'
        if not path.exists():
            continue
        rows = [r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip()) if r.get('status') == 'ok' and r.get('conf_token_geo') is not None]
        lines += [f'## {name}', '', '| 题型 | 格式 | n | Acc | Conf | ECE | 探索性 |', '|---|---|---|---|---|---|---|']
        for cat in sorted({r.get('category', 'unknown') for r in rows}):
            for f in FM:
                rs = [r for r in rows if r.get('category', 'unknown') == cat and r['format'] == f]
                if not rs:
                    continue
                y = [int(r['correct']) for r in rs]; c = [r['conf_token_geo'] for r in rs]
                v = {'model': name, 'category': cat, 'format': f, 'n': len(rs), 'acc': sum(y) / len(y),
                     'conf': sum(c) / len(c), 'ece': ece(c, y), 'exploratory': len(rs) < 30}
                result.append(v)
                lines.append(f"| {cat} | {f} | {len(rs)} | {v['acc']:.4f} | {v['conf']:.4f} | {v['ece']:.4f} | {v['exploratory']} |")
        lines.append('')
    (OUT / 'by_category.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (OUT / 'by_category.md').write_text('\n'.join(lines), encoding='utf-8')
    print('saved', OUT)

if __name__ == '__main__':
    from analysis_gate import require_analysis_inputs
    require_analysis_inputs('by_category', RESULTS_ROOT)
    main()
