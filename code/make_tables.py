#!/usr/bin/env python3
"""Build corrected tables exclusively from authoritative analysis JSON."""
import json
from calib_metrics import RESULTS_ROOT
from make_paper_figs import matched_main_cohort

OUT = RESULTS_ROOT / 'tables'
FM = ['A', 'B', 'C']
MODELS = [('qwen500', 'Qwen3-VL-8B'), ('healthgpt500', 'HealthGPT-Pro-8B'), ('gemma500', 'Gemma-4-E4B')]

def read(relative):
    p = RESULTS_ROOT / relative
    return json.loads(p.read_text()) if p.exists() else {}

def fmt(value):
    return '—' if value is None else f'{value:.3f}'

def interval(value):
    return '—' if not value else f'[{value[0]:.3f}, {value[1]:.3f}]'

def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|'] + ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])

def latex(caption, label, header, rows):
    def escape(x):
        return str(x).replace('_', r'\_').replace('%', r'\%').replace('→', r'$\to$').replace('—', '--')
    return '\n'.join([r'\begin{table}[t]', r'\centering\small', f'\\caption{{{caption}}}', f'\\label{{{label}}}',
        r'\begin{tabular}{' + 'l' * len(header) + '}', r'\toprule', ' & '.join(map(escape, header)) + r' \\', r'\midrule',
        *[' & '.join(map(escape, row)) + r' \\' for row in rows], r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = read('summary/summary.json'); boot = read('formal_stats/bootstrap_stats.json')
    manifest = {'input_root': str(RESULTS_ROOT), 'tables': {}, 'main_cohorts': {}}
    tables = []
    rows = [
        ['划分', '按image_path的md5奇偶fit/test；同图题目不跨折'],
        ['校准器', 'power scaling c**(1/T)；80点对数网格0.05–20，fit ECE10最小'],
        ['置信度', 'corrected/raw答案span token几何均值；字段与provenance见输入审计'],
        ['阈值', '仅distinct score边界；fit选阈值，test报告coverage/risk与状态'],
        ['区间', 'bootstrap以图像聚类；学习曲线/重复划分为经验区间而非CI'],
        ['历史重评分', 'user-side suffix语义；原矩阵与旧评分代码不可复现，不作因果分解']]
    tables.append(('T1', '数据与分析协议', ['项目', '设置'], rows, ''))
    rows = []
    for key, name in MODELS:
        if key not in summary:
            continue
        manifest['main_cohorts'][key] = {}
        for f in FM:
            v = summary[key]['formats'][f]
            bootstrap = boot.get(f'slake:{key}', {})
            manifest['main_cohorts'][key][f] = matched_main_cohort(v, bootstrap, f)
            point = bootstrap.get('ci_ece_full', {}).get(f, {}).get('point')
            if (v['ece10'] is None) != (point is None) or (point is not None and abs(v['ece10'] - point) > 5.1e-5):
                raise ValueError(f'ECE table/bootstrap center mismatch: {key}/{f}')
            ci = bootstrap.get('ci_ece_full', {}).get(f, {}).get('ci95')
            rows.append([name, f, v['n'], fmt(v['acc']), fmt(v['mean_conf']), fmt(v['ece10']), interval(ci), fmt(v['brier']), fmt(v['auroc_error_detection'])])
    tables.append(('T2', 'SLAKE全量有效样本主结果', ['模型', '格式', 'n', 'Acc', 'Mean conf', 'ECE', 'ECE 95%CI', 'Brier', 'AUROC(error)'], rows,
                   'AUROC的正类为错误，评分为1-confidence。ECE中心和CI均来自全量有效样本，与fig_ece相同。'))
    ext = []
    for ds, path in [('VQA-RAD', 'summary_vqarad/summary.json'), ('VQA-Med', 'summary_vqamed/summary.json')]:
        for model, info in read(path).items():
            for f, v in info['formats'].items():
                ext.append([ds, model, f, v['n'], fmt(v['acc']), fmt(v['mean_conf']), fmt(v['ece10']), fmt(v['auroc_error_detection'])])
    if ext:
        tables.append(('T2_external', '外部数据集结果', ['数据集', '模型', '格式', 'n', 'Acc', 'Mean conf', 'ECE', 'AUROC(error)'], ext, ''))
    rescore = read('rescore_analysis/rescore_summary.json'); rows = []
    for model, r in rescore.items():
        rows.append([model, r['n_questions_full'], r.get('n_images_full'), fmt(r['scoring_spread_mean']), interval(r['scoring_spread_ci95']),
                     fmt(r['generation_spread_mean']), interval(r['generation_spread_ci95'])])
    tables.append(('T3', '历史重评分描述性跨度', ['模型', 'n题', 'n图', '评分跨度', '95%CI', '答案来源跨度', '95%CI'], rows,
                   '图像聚类bootstrap。历史矩阵为user-side suffix评分语义；span/context未验证可比，不作上下文因果归因。矩阵图加粗真正行最小，非普遍对角最优。'))
    selective = read('selective_full/selective_full.json'); rows = []
    for model, r in selective.items():
        for t in FM:
            for key, v in r['targets'][t]['conditions'].items():
                rows.append([model, t, key, fmt(v['coverage']), interval(v.get('coverage_ci95')), fmt(v['risk']), interval(v.get('risk_ci95')), v['n_selected'], v['status']])
    tables.append(('T4', 'fit阈值在test的自身与迁移表现', ['模型', '目标', '来源', 'Coverage', '95%CI', 'Risk', '95%CI', 'n选择', '状态'], rows,
                   'fit无阈值、test无数据/无选择、test风险超过目标分别列示；固定阈值test图像bootstrap，零选择coverage=0，risk未定义。'))
    cond = read('format_cond/format_cond.json'); rows = []
    for model, r in cond.items():
        rows.append([model, 'raw', *[fmt(r['raw'].get(f)) for f in FM], fmt(r.get('raw_pooled')), fmt(r.get('raw_macro'))])
        for k, v in r['strategies'].items():
            rows.append([model, k, *[fmt(v['ece_per_format'].get(f)) for f in FM], fmt(v['ece_pooled']), fmt(v['ece_macro'])])
    tables.append(('T5', '格式条件校准test ECE', ['模型', '策略', 'A', 'B', 'C', 'Pooled', 'Macro'], rows,
                   'Pooled先合并样本再算ECE；Macro为三格式ECE等权平均。二者与per-format并报，避免池化抵消掩盖分组误差。正则/clipping配置保存在format_cond.json。'))
    tex = []
    for name, title, header, rows, note in tables:
        text = f'# {title}\n\n' + (note + '\n\n' if note else '') + table(header, rows) + '\n'
        (OUT / f'{name}.md').write_text(text, encoding='utf-8')
        tex.append(latex(title, f'tab:{name}', header, rows))
        manifest['tables'][name] = {'header': header, 'rows': rows, 'note': note}
    (OUT / 'tables.tex').write_text('\n'.join(tex), encoding='utf-8')
    (OUT / 'table_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved', OUT)

if __name__ == '__main__':
    from analysis_gate import require_analysis_inputs
    require_analysis_inputs('make_tables', RESULTS_ROOT)
    main()
