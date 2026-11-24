#!/usr/bin/env python3
"""独立运行统计边界回归；只写审计输出和临时合成数据。"""
import argparse
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from unittest.mock import patch

ap = argparse.ArgumentParser()
ap.add_argument('--project-root', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
root = args.project_root.resolve()
sys.path.insert(0, str(root / 'code'))
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
import calib_metrics as cm
import selective_full as sf
import selective_risk as sr
import formal_stats as fs
import stability as st
import rescore_analysis as ra
import format_cond_calib as fc
import make_paper_figs as pf

# 合成测试只验证统计输出，禁止保存任何论文图像。
from matplotlib.figure import Figure
Figure.savefig = lambda self, *args, **kwargs: None

checks = []
def check(name, fn):
    try:
        detail = fn()
        checks.append({'name': name, 'status': 'passed', 'detail': detail})
    except Exception:
        checks.append({'name': name, 'status': 'failed', 'traceback': traceback.format_exc()})

def close(a, b):
    assert np.allclose(a, b, atol=1e-12, rtol=1e-12), (a, b)


def auc_boundaries():
    cases = [([.9, .8, .2, .1], [1, 1, 0, 0], 1.),
             ([.1, .2, .8, .9], [1, 1, 0, 0], 0.),
             ([.5, .5, .5, .5], [1, 1, 0, 0], .5),
             ([.8, .8, .2, .2], [1, 0, 1, 0], .5)]
    for scores, labels, expected in cases:
        close(cm.auroc(scores, labels), expected)
        close(cm.auroc(scores, labels), roc_auc_score(labels, scores))
    for labels in ([1, 1], [0, 0]):
        assert math.isnan(cm.auroc([.2, .5], labels))
    rng = np.random.default_rng(717)
    for _ in range(50):
        scores = rng.integers(0, 8, 31) / 8
        labels = rng.integers(0, 2, 31)
        close(cm.auroc(scores.tolist(), labels.tolist()), roc_auc_score(labels, scores))
    rows = [{'qid': str(i), 'image_path': str(i), 'format': 'A', 'status': 'ok',
             'conf_token_geo': c, 'correct': y} for i, (c, y) in enumerate(zip([.9, .8, .2, .1], [1, 1, 0, 0]))]
    close(cm.per_format_metrics(rows, 'A')['auroc_error_detection'], 1.)
    for label in (0, 1):
        assert cm.per_format_metrics([{**r, 'correct': label} for r in rows], 'A')['auroc_error_detection'] is None
    return {'cases': len(cases) + 51, 'single_class': '底层undefined，主结果为null'}


def threshold_groups():
    c, y = [.9, .9, .8, .7], [1, 0, 1, 1]
    curve = cm.distinct_curve(c, y)
    assert [p['n_selected'] for p in curve] == [2, 3, 4]
    assert cm.threshold_at_risk(c, y, .3) == .7
    assert cm.threshold_at_risk([.9, .9], [1, 0], .05) is None
    assert cm.distinct_curve([], []) == []
    assert cm.threshold_at_risk([], []) is None
    assert cm.threshold_at_risk([.9] * 20, [1] * 19 + [0], .05) == .9
    assert cm.distinct_curve([.9] * 20, [1] * 19 + [0])[0]['risk'] == .05
    for p in curve:
        selected = [v for v, score in zip(y, c) if score >= p['threshold']]
        close(p['coverage'], len(selected) / len(y))
        close(p['risk'], 1 - np.mean(selected))
    return {'complete_group_sizes': [p['n_selected'] for p in curve], 'no_tie_splitting': True}


def selection_boundaries():
    rows = [{'conf': .9, 'correct': 0, 'img': 'a'}]
    assert sf.condition(rows, None)['status'] == 'fit_threshold_unavailable'
    assert sf.condition([], .8)['status'] == 'test_empty'
    assert sf.condition(rows, 1.)['status'] == 'test_selection_empty'
    assert sf.condition(rows, .8)['status'] == 'test_risk_exceeds_target'
    ok = sf.condition([{**rows[0], 'correct': 1}], .8)
    assert ok['status'] == 'test_risk_at_or_below_target'
    assert sf.apply(rows, 1.) == (0., None, 0)
    boundary = [{'conf': .9, 'correct': y, 'img': 'a'} for y in [1] * 19 + [0]]
    exact = sf.condition(boundary, .9)
    assert exact['risk'] == .05 and exact['status'] == 'test_risk_at_or_below_target'
    return {'states_checked': 5, 'empty_risk': None, 'exact_5pct_target_accepted': True}


def exact_twenty_boundaries():
    cases = []
    for n_correct in (19, 20, 0):
        labels = [1] * n_correct + [0] * (20 - n_correct)
        conf = [.9] * 20
        expected_risk = (20 - n_correct) / 20
        rows = [{'conf': .9, 'correct': y, 'img': f'im-{i}'} for i, y in enumerate(labels)]
        curve = cm.distinct_curve(conf, labels)
        assert curve == [{'threshold': .9, 'coverage': 1., 'risk': expected_risk, 'n_selected': 20}]
        threshold = cm.threshold_at_risk(conf, labels, .05)
        assert threshold == (.9 if n_correct >= 19 else None)
        close(sf.apply(rows, .9), (1., expected_risk, 20))
        condition = sf.condition(rows, .9)
        assert condition['risk'] == expected_risk
        assert condition['test_target_met'] == (n_correct >= 19)
        assert condition['status'] == ('test_risk_at_or_below_target' if n_correct >= 19 else 'test_risk_exceeds_target')
        close(sr.cov_at_risk(conf, labels, .05), 1. if n_correct >= 19 else 0.)
        cases.append({'correct': n_correct, 'total': 20, 'risk': expected_risk, 'fit_threshold': threshold,
                      'test_status': condition['status']})
    return {'cases': cases, 'risk_computed_from_integer_error_count': True}


def variable_cluster_bootstrap():
    rows = [{'img': 'a', 'conf': .9, 'correct': 1}] + [
        {'img': 'b', 'conf': .1, 'correct': 0} for _ in range(9)]
    seed, n_boot = 122, 200
    got = sf.boot_ci(rows, .8, n_boot=n_boot, seed=seed)
    rng = np.random.default_rng(seed)
    cov, risk = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, 2, 2)
        n_a = int(np.sum(pick == 0)); n_b = 2 - n_a
        cov.append(n_a / (n_a + 9 * n_b))
        if n_a:
            risk.append(0.)
    close(got['coverage_ci95'], np.percentile(cov, [2.5, 97.5]))
    close(got['risk_ci95'], [0., 0.])
    assert got['n_boot_empty_selection'] == n_boot - len(risk)
    assert got['n_boot_coverage'] == n_boot
    assert got['n_boot_nonempty_risk'] == len(risk)
    zero = sf.boot_ci(rows, 1., n_boot=n_boot, seed=seed)
    close(zero['coverage_ci95'], [0., 0.])
    assert zero['risk_ci95'] is None
    assert zero['n_boot_empty_selection'] == n_boot
    refit = sr.boot_transfer(rows, rows, n_boot=n_boot, seed=seed)
    assert refit['n_boot_coverage'] == n_boot
    return {'unequal_cluster_sizes': [1, 9], 'bootstrap': got, 'all_empty_selection': zero}


def refitted_bootstrap_reference():
    fit = [{'img': 'f-a', 'conf': .9, 'correct': 1}] + [
        {'img': 'f-b', 'conf': .8, 'correct': 0} for _ in range(9)]
    test = [{'img': 't-a', 'conf': .95, 'correct': 1}] + [
        {'img': 't-b', 'conf': .1, 'correct': 0} for _ in range(9)]
    seed, n_boot = 722, 300
    got = sr.boot_transfer(fit, test, n_boot=n_boot, seed=seed)
    rng = np.random.default_rng(seed)
    cov, risks, available = [], [], 0
    for _ in range(n_boot):
        f_draw = rng.integers(0, 2, 2)
        t_draw = rng.integers(0, 2, 2)
        n_fit_correct = int(np.sum(f_draw == 0))
        n_test_correct = int(np.sum(t_draw == 0))
        threshold_available = n_fit_correct > 0
        available += threshold_available
        selected = n_test_correct if threshold_available else 0
        total = n_test_correct + 9 * (2 - n_test_correct)
        cov.append(selected / total)
        if selected:
            risks.append(0.)
    close(got['coverage_ci95'], np.percentile(cov, [2.5, 97.5]))
    close(got['risk_ci95'], np.percentile(risks, [2.5, 97.5]))
    assert got['n_boot_coverage'] == n_boot
    assert got['n_boot_fit_threshold_available'] == available
    assert got['n_boot_nonempty_risk'] == len(risks)
    missing = sr.boot_transfer([{**r, 'correct': 0} for r in fit], test, n_boot=n_boot, seed=seed)
    assert missing['coverage_ci95'] == [0., 0.]
    assert missing['risk_ci95'] is None and missing['n_boot_fit_threshold_available'] == 0
    assert missing['n_boot_coverage'] == n_boot and missing['n_boot_nonempty_risk'] == 0
    return {'independent_fit_test_image_draws': True, 'unequal_cluster_sizes': [1, 9],
            'reference_bootstrap': got, 'unavailable_fit_threshold': missing}


def spread_image_cluster():
    values = [0.] + [1.] * 9
    images = ['a'] + ['b'] * 9
    seed, n_boot = 322, 200
    rng = np.random.default_rng(seed)
    expected = []
    for _ in range(n_boot):
        draw = rng.integers(0, 2, 2)
        a = int(np.sum(draw == 0)); b = 2 - a
        expected.append(9 * b / (a + 9 * b))
    close(cm.cluster_mean_ci(values, images, n_boot, seed), np.percentile(expected, [2.5, 97.5]))
    close(ra.boot_ci(values, images, n_boot, seed), np.percentile(expected, [2.5, 97.5]))
    assert cm.cluster_mean_ci([], [], n_boot, seed) is None
    full = {'q': {(g, s): (i * 3 + j) / 10 for i, g in enumerate(ra.FM) for j, s in enumerate(ra.FM)}}
    close(ra.spreads_by_question(full), [[.2, .6]])
    return {'cluster_unit': 'image', 'question_weighted_draw_mean': True}


def power_grid():
    assert len(cm.POWER_GRID) == len(set(cm.POWER_GRID)) == 80
    close([cm.POWER_GRID[0], cm.POWER_GRID[-1]], [.05, 20.])
    close(np.diff(np.log(cm.POWER_GRID)), [math.log(400) / 79] * 79)
    c, y = [.1, .3, .6, .9], [0, 0, 1, 1]
    expected = min(cm.POWER_GRID, key=lambda t: cm.ece(cm.temp_scale(c, t), y))
    close(cm.fit_temperature(c, y), expected)
    import fair_transfer
    close(fair_transfer.fit_T(c, y), expected)
    close(st.fit_power(c, y)(c), cm.temp_scale(c, expected))
    assert cm.fit_temperature([], []) is None
    assert fair_transfer.fit_T([], []) is None
    assert st.fit_power([], []) is None
    return {'grid_count': 80, 'bounds': [.05, 20.], 'empty_fit': None}


def isotonic_ties():
    c = np.array([.2, .2, .7, .7, .9])
    y = np.array([1, 0, 0, 1, 1])
    query = np.array([0., .2, .4, .7, .9, 1.])
    expected = IsotonicRegression(increasing=True, out_of_bounds='clip', y_min=0., y_max=1.).fit(c, y).predict(query)
    for order in itertools.permutations(range(5)):
        index = list(order)
        close(st.fit_isotonic(c[index], y[index])(query), expected)
    close(st.fit_isotonic([.2, .2], [0, 1])([.1, .2, .9]), [.5, .5, .5])
    return {'tie_orderings_checked': 120, 'reference': 'sklearn.IsotonicRegression'}


def pooled_macro():
    pred = np.array([.9] * 10)
    labels = np.array([1] * 9 + [0])
    formats = ['A'] * 9 + ['B']
    got = fc.aggregate_ece(pred, formats, labels)
    close(got['ece_pooled'], 0.)
    close(got['ece_macro'], .5)
    close(got['ece_per_format']['A'], .1)
    close(got['ece_per_format']['B'], .9)
    assert got['ece_per_format']['C'] is None
    empty = {'conf': np.array([]), 'fmt': [], 'correct': np.array([])}
    assert fc.fit_predict(empty, empty, 'cond_full') is None
    assert fc.design([], [], 'cond_full').shape == (0, 6)
    assert fc.aggregate_ece(np.array([]), [], np.array([]))['ece_macro'] is None
    return {'cancellation_case': got, 'empty_fit_returns': None}


def ece_precision_and_serialization():
    import aprime_analysis as apm
    import fair_transfer as ft
    conf, correct = [.20000001, .20000003], [0, 0]
    expected = .20000002
    for module in (cm, fc, apm, ft, ra, st):
        close(module.ece(conf, correct), expected)
    got = fc.aggregate_ece(np.array(conf), ['A', 'B'], np.array(correct))
    close(got['ece_per_format']['A'], conf[0])
    close(got['ece_per_format']['B'], conf[1])
    close(got['ece_macro'], expected)
    rows = [{'qid': str(i), 'image_path': f'im-{i}', 'format': 'A', 'status': 'ok',
             'conf_token_geo': c, 'conf_verbalized': c, 'correct': 0} for i, c in enumerate(conf)]
    per_format = cm.per_format_metrics(rows, 'A')
    close(per_format['ece10'], expected)
    close(per_format['verbalized']['ece10'], expected)
    images = {h: next(str(i) for i in range(100) if sf.half(str(i)) == h) for h in (0, 1)}
    transfer_rows = [{'qid': f'{h}-{f}', 'image_path': images[h], 'format': f, 'status': 'ok',
                      'conf_token_geo': .7 if h == 0 else conf[0], 'correct': int(h == 0)}
                     for h in (0, 1) for f in fc.FM]
    transfer = cm.transfer_matrix(transfer_rows)
    for source in fc.FM:
        for target in fc.FM:
            key = f'{source}->{target}'
            close(transfer['uncalibrated'][key], conf[0])
            close(transfer['temperature_scaled'][key], cm.ece(cm.temp_scale([conf[0]], transfer['temperatures'][source]), [0]))
    serialized = json.loads(json.dumps({'summary': per_format, 'transfer': transfer, 'aggregate': got}, allow_nan=False))
    close(serialized['summary']['ece10'], expected)
    assert serialized['aggregate']['ece_macro'] != round(expected, 4)
    return {'ece_modules_without_early_rounding': [m.__name__ for m in (cm, fc, apm, ft, ra, st)],
            'micro_difference_preserved': expected, 'summary_transfer_and_macro_full_precision': True}


def formal_bootstrap_reference():
    rows = []
    for image, conf, count in [('a', .123456789, 1), ('b', .923456789, 9)]:
        for f in fs.FMTS:
            for i in range(count):
                rows.append({'qid': f'{image}-{i}', 'image_path': image, 'format': f,
                             'correct': 0, 'conf_token_geo': conf})
    seed, n_boot = 941, 300
    temps = {f: 1. for f in fs.FMTS}
    with patch.object(fs, 'BOOT', n_boot):
        stats, penalties, pairs = fs.evaluate(rows, temps, seed)
        empty_stats, empty_penalties, empty_pairs = fs.evaluate([], temps, seed)
    rng = np.random.default_rng(seed)
    reference = []
    for _ in range(n_boot):
        draw = rng.integers(0, 2, 2)
        n_a = int(np.sum(draw == 0)); n_b = 2 - n_a
        reference.append((n_a * .123456789 + 9 * n_b * .923456789) / (n_a + 9 * n_b))
    observed = (.123456789 + 9 * .923456789) / 10
    for f, value in stats.items():
        close(value['point'], observed)
        close(value['mean'], observed)
        close(value['bootstrap_mean'], np.mean(reference))
        close(value['ci95'], np.percentile(reference, [2.5, 97.5]))
        assert value['n_boot'] == n_boot
        assert abs(value['point'] - value['bootstrap_mean']) > .01
        assert empty_stats[f]['point'] is None and empty_stats[f]['ci95'] is None
    for value in penalties.values():
        close(value['point'], 0.)
        close(value['ci95'], [0., 0.])
    for value in pairs.values():
        close(value['point'], 0.)
        close(value['ci95'], [0., 0.])
    assert all(v['point'] is None and v['ci95'] is None for v in empty_penalties.values())
    assert all(v['point'] is None and v['ci95'] is None for v in empty_pairs.values())
    return {'observed_center': observed, 'bootstrap_mean': float(np.mean(reference)),
            'image_cluster_reference_matches': True, 'same_draw_pairwise_zero': True}


def figure_cohort_and_minimum():
    mask = pf.row_minimum_mask([[.2, .20000001, .3], [.4, .1, .1], [.5, .4, .3]])
    assert mask.tolist() == [[True, False, False], [False, True, True], [False, False, True]]
    row = {'qid': 1, 'image_path': 'a', 'format': 'A', 'correct': 1, 'conf_token_geo': .9}
    same = cm.cohort_metadata([row])
    pf.matched_main_cohort({'cohort': same}, {'cohort_full': {'A': same}}, 'A')
    changed = cm.cohort_metadata([{**row, 'qid': 2}])
    try:
        pf.matched_main_cohort({'cohort': same}, {'cohort_full': {'A': changed}}, 'A')
    except ValueError:
        pass
    else:
        raise AssertionError('同ECE但不同qid总体必须拒绝')
    assert cm.cohort_metadata([row, {**row, 'qid': 2}]) == cm.cohort_metadata([{**row, 'qid': 2}, row])
    for field in ('format', 'stored_source', 'source_raw_sha256', 'run_signature', 'protocol'):
        changed = cm.cohort_metadata([{**row, field: 'changed_source_or_format'}])
        try:
            pf.matched_main_cohort({'cohort': same}, {'cohort_full': {'A': changed}}, 'A')
        except ValueError:
            pass
        else:
            raise AssertionError(f'必须拒绝不同{field}的同分数总体')
    import make_tables as mt
    with tempfile.TemporaryDirectory(prefix='medvqa_table_cohort_') as directory:
        temp = Path(directory)
        (temp / 'summary').mkdir(); (temp / 'formal_stats').mkdir()
        summary = {'qwen500': {'formats': {f: {'cohort': same} for f in pf.FM}}}
        boot = {'slake:qwen500': {'cohort_full': {'A': changed}}}
        (temp / 'summary/summary.json').write_text(json.dumps(summary))
        (temp / 'formal_stats/bootstrap_stats.json').write_text(json.dumps(boot))
        old_root, old_out = mt.RESULTS_ROOT, mt.OUT
        try:
            mt.RESULTS_ROOT, mt.OUT = temp, temp / 'tables'
            try:
                mt.main()
            except ValueError as error:
                assert 'cohort identity mismatch' in str(error)
            else:
                raise AssertionError('主表必须在输出前拒绝异源cohort')
            assert not list(mt.OUT.glob('T*.md'))
        finally:
            mt.RESULTS_ROOT, mt.OUT = old_root, old_out
    with tempfile.TemporaryDirectory(prefix='medvqa_matrix_precision_') as directory:
        temp = Path(directory)
        raw, scores, target = temp / 'raw', temp / 'rescore', temp / 'summary'
        raw.mkdir(); scores.mkdir()
        base = [{'qid': 'q', 'image_path': 'im', 'format': g, 'status': 'ok',
                 'correct': 0, 'conf_token_geo': .3} for g in ra.FM]
        cells = [{'qid': 'q', 'g': g, 's': s, 'conf': [.20000001, .2, .3][j]}
                 for g in ra.FM for j, s in enumerate(ra.FM)]
        (raw / 'qwen500.jsonl').write_text('\n'.join(json.dumps(r) for r in base))
        (scores / 'qwen.jsonl').write_text('\n'.join(json.dumps(r) for r in cells))
        old_argv, old_boot = sys.argv, ra.N_BOOT
        try:
            sys.argv = ['rescore_analysis.py', '--rescore-dir', str(scores), '--raw-dir', str(raw), '--out-dir', str(target)]
            ra.N_BOOT = 2
            ra.main()
            data = json.loads((target / 'rescore_summary.json').read_text())['qwen']
            matrix = [[data['ece_canonical'][s][g] for s in ra.FM] for g in ra.FM]
            assert matrix[0][0] > matrix[0][1]
            assert pf.row_minimum_mask(matrix).tolist() == [[False, True, False]] * 3
            assert round(matrix[0][0], 4) == round(matrix[0][1], 4)
        finally:
            sys.argv, ra.N_BOOT = old_argv, old_boot
    return {'near_minimum_not_bold': True, 'exact_ties_bold': True,
            'same_ECE_different_identity_rejected': True, 'source_and_format_identity_checked': True,
            'table_rejects_mismatch_before_output': True, 'matrix_serialization_preserves_true_minimum': True}


def learning_replicate_aggregation():
    import learning_curve as lc
    with tempfile.TemporaryDirectory(prefix='medvqa_stats_learning_') as directory:
        temp = Path(directory)
        raw = temp / 'raw'; raw.mkdir()
        images = {h: [str(i) for i in range(100) if sf.half(str(i)) == h][:3] for h in (0, 1)}
        rows = []
        for f in lc.FM:
            for h in (0, 1):
                for i, image in enumerate(images[h]):
                    rows.append({'qid': f'{h}-{i}', 'image_path': image, 'format': f, 'status': 'ok',
                                 'correct': int((i + lc.FM.index(f)) % 2 == 0),
                                 'conf_token_geo': [.2, .6, .9][i]})
        (raw / 'qwen500.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        old = (lc.RESULTS_ROOT, lc.OUT, lc.SIZES, lc.REPS, lc.BASEMAP)
        try:
            lc.RESULTS_ROOT, lc.OUT, lc.SIZES, lc.REPS = temp, temp / 'learning_curve', [2], 3
            lc.BASEMAP = {'qwen': 'qwen500'}
            lc.main()
            result = json.loads((lc.OUT / 'learning_curve.json').read_text())['qwen']
        finally:
            lc.RESULTS_ROOT, lc.OUT, lc.SIZES, lc.REPS, lc.BASEMAP = old
        for f in lc.FM:
            details = result[f]['replicate_details']['2']
            expected = []
            assert len(details) == 3
            for detail in details:
                own = detail['ece_by_target_source'][f][f]
                penalties = [detail['ece_by_target_source'][f][s] - own for s in lc.FM if s != f]
                close(detail['source_mean_penalty'], np.mean(penalties))
                expected.append(np.mean(penalties))
            close(result[f]['pen']['2'], [np.mean(expected), *np.percentile(expected, [2.5, 97.5])])
        incomplete = [r for r in rows if not (r['format'] == 'C' and sf.half(r['image_path']) == 0)]
        (raw / 'qwen500.jsonl').write_text('\n'.join(json.dumps(r) for r in incomplete))
        old = (lc.RESULTS_ROOT, lc.OUT, lc.SIZES, lc.REPS, lc.BASEMAP)
        try:
            lc.RESULTS_ROOT, lc.OUT, lc.SIZES, lc.REPS = temp, temp / 'learning_incomplete', [2], 3
            lc.BASEMAP = {'qwen': 'qwen500'}
            lc.main()
            missing = json.loads((lc.OUT / 'learning_curve.json').read_text())['qwen']
        finally:
            lc.RESULTS_ROOT, lc.OUT, lc.SIZES, lc.REPS, lc.BASEMAP = old
        for f in lc.FM:
            assert missing[f]['pen']['2'] is None
            assert all(r['source_mean_penalty'] is None for r in missing[f]['replicate_details']['2'])
        return {'source_first_then_replicate': True, 'tested_replicates': 3,
                'incomplete_source_mean_unavailable': True}


def empty_main_process():
    with tempfile.TemporaryDirectory(prefix='medvqa_stats_empty_') as directory:
        temp = Path(directory)
        raw = temp / 'raw'; raw.mkdir()
        for stem, _ in fc.MODELS:
            (raw / f'{stem}.jsonl').write_text('')
        output = temp / 'format_cond'
        result = subprocess.run([sys.executable, str(root / 'code/format_cond_calib.py'),
                                 '--raw-dir', str(raw), '--out-dir', str(output)], capture_output=True, text=True)
        if result.returncode != 0:
            assert 'FormalEvidenceError' in result.stderr, result.stderr
            assert not (output / 'format_cond.json').exists()
        else:
            data = json.loads((output / 'format_cond.json').read_text())
            assert all(r['status'] == 'fit_unavailable' for r in data.values())
            assert all(r['raw_macro'] is None for r in data.values())
        split_checks = []
        images = {h: next(str(i) for i in range(100) if sf.half(str(i)) == h) for h in (0, 1)}
        old_argv = sys.argv
        try:
            for population, parity, expected in [('all_empty', None, 'fit_unavailable'),
                                                 ('fit_empty', 1, 'fit_unavailable'),
                                                 ('test_empty', 0, 'test_empty')]:
                rows = [] if parity is None else [{'qid': population, 'image_path': images[parity],
                    'format': f, 'status': 'ok', 'correct': 1, 'conf_token_geo': .8} for f in fc.FM]
                for stem, _ in fc.MODELS:
                    (raw / f'{stem}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
                isolated = temp / population
                sys.argv = ['format_cond_calib.py', '--raw-dir', str(raw), '--out-dir', str(isolated)]
                fc.main()
                data = json.loads((isolated / 'format_cond.json').read_text())
                assert all(r['status'] == expected and r['strategies'] == {} for r in data.values())
                assert 'NaN' not in (isolated / 'format_cond.json').read_text()
                split_checks.append({'population': population, 'status': expected, 'models': len(data),
                                     'fabricated_calibrators': 0})
        finally:
            sys.argv = old_argv
        return {'process_exit': result.returncode, 'formal_gate_refused_synthetic_cli': result.returncode != 0,
                'isolated_algorithm_empty_splits': split_checks, 'fabricated_calibrators': 0}

for name, fn in [('auroc_direction_and_ties', auc_boundaries),
                 ('distinct_score_complete_groups', threshold_groups),
                 ('fit_test_selection_states', selection_boundaries),
                 ('exact_19_20_0_of_20_risk_boundaries', exact_twenty_boundaries),
                 ('variable_cluster_denominator_zero_selection', variable_cluster_bootstrap),
                 ('refitted_threshold_bootstrap_reference', refitted_bootstrap_reference),
                 ('SG_image_cluster_bootstrap', spread_image_cluster),
                 ('power_grid_80_and_empty_fit', power_grid),
                 ('sklearn_isotonic_tie_invariance', isotonic_ties),
                 ('pooled_macro_performat_and_empty', pooled_macro),
                 ('ece_precision_and_json_serialization', ece_precision_and_serialization),
                 ('formal_image_cluster_bootstrap_reference', formal_bootstrap_reference),
                 ('matrix_exact_row_min_and_main_cohort', figure_cohort_and_minimum),
                 ('learning_source_then_replicate', learning_replicate_aggregation),
                 ('format_cond_empty_process', empty_main_process)]:
    check(name, fn)

modules = ['actual_analysis', 'aprime_analysis', 'by_category', 'calib_metrics', 'fair_transfer',
           'fix_figs', 'formal_stats', 'format_cond_calib', 'gee_stats', 'learning_curve',
           'make_paper_figs', 'make_tables', 'rescore_analysis', 'selective_full',
           'selective_risk', 'span_control', 'stability']
def syntax_check():
    for module in modules:
        compile((root / 'code' / f'{module}.py').read_text(), str(root / 'code' / f'{module}.py'), 'exec')
    return {'scripts_compiled': len(modules)}
check('all_17_analysis_scripts_compile', syntax_check)

report = {'purpose': '合成边界回归验证；不是正式医学VQA结果', 'project_root': str(root),
          'python': sys.executable, 'numpy': np.__version__,
          'test_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'source_sha256': {module: hashlib.sha256((root / 'code' / f'{module}.py').read_bytes()).hexdigest() for module in modules},
          'cli_guards': {module: "require_analysis_inputs(" in (root / 'code' / f'{module}.py').read_text() for module in modules},
          'checks': checks, 'passed': sum(c['status'] == 'passed' for c in checks),
          'failed': sum(c['status'] == 'failed' for c in checks)}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({'report': str(args.out), 'passed': report['passed'], 'failed': report['failed']}, ensure_ascii=False))
for entry in checks:
    if entry['status'] == 'failed':
        print(entry['name'], entry['traceback'])
sys.exit(bool(report['failed']))
