#!/usr/bin/env python3
"""只读核对修复数据、完整原输出、正式TF概率与任务清单。"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('.')
sys.path.insert(0, str(ROOT / 'code'))
from recover_corrected import required_action


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main():
    corrected = ROOT / 'results/corrected'
    counts, actions, requirements = Counter(), Counter(), Counter()
    sources, problems = [], []
    total_jobs = candidates = formal_conf = verified = 0
    for directory in ('raw', 'raw_vqarad', 'raw_vqamed', 'raw_wording'):
        for path in sorted((corrected / directory).glob('*.jsonl')):
            source = ROOT / 'results' / directory / path.name
            original_rows = read_rows(source)
            original = {(r['qid'], r['format']): r for r in original_rows}
            rows = read_rows(path)
            requirements.update(required_action(row)[0] for row in original_rows)
            counts.update(row['recovery_status'] for row in rows)
            actions.update(row['action'] for row in rows)
            keys = [(row['qid'], row['format']) for row in rows]
            if len(keys) != len(set(keys)) or set(keys) != set(original):
                problems.append([str(path), 'identity_or_count_mismatch'])
            for row in rows:
                key = (row['qid'], row['format'])
                old = original[key]
                context = [directory, row['model'], *key]
                if row.get('raw') != old.get('raw'):
                    problems.append(context + ['raw_changed'])
                if row.get('raw_sha256') != digest(old['raw']):
                    problems.append(context + ['raw_hash_mismatch'])
                if row.get('saved_raw', old['raw']) != old['raw'] or row.get('original_output_raw', old['raw']) != old['raw']:
                    problems.append(context + ['raw_alias_mismatch'])
                if row.get('legacy_candidate_conf') is not None:
                    candidates += 1
                is_verified = row.get('teacher_forcing_verified') is True or row.get('regeneration_verified') is True
                if is_verified:
                    verified += 1
                    if row.get('action') != 'none':
                        problems.append(context + ['verified_still_pending'])
                if row.get('conf_token_geo') is not None:
                    formal_conf += 1
                    if not is_verified or not row.get('span_scoreable'):
                        problems.append(context + ['unverified_formal_conf'])
                if row.get('span_boundary_partial') and row.get('conf_token_geo') is not None:
                    problems.append(context + ['partial_formal_conf'])
                if row.get('prompt_sha256') and digest(row.get('prompt_text', '')) != row['prompt_sha256']:
                    problems.append(context + ['prompt_hash_mismatch'])
                if row.get('teacher_forcing_verified') and row.get('original_token_identity_verified') is True:
                    problems.append(context + ['TF_not_original_generation_identity'])
            sources.append({'source': str(source.relative_to(ROOT)), 'rows': len(rows),
                            'statuses': dict(Counter(row['recovery_status'] for row in rows)),
                            'actions': dict(Counter(row['action'] for row in rows))})
    job_actions = Counter()
    for path in sorted((corrected / 'jobs').glob('*.jsonl')):
        jobs = read_rows(path)
        total_jobs += len(jobs)
        job_actions.update(row.get('action') for row in jobs)
        for row in jobs:
            for field in ('action', 'source_raw_file', 'raw', 'raw_sha256', 'prompt_instruction', 'prompt_text', 'system_prompt'):
                if field not in row or row[field] is None:
                    problems.append([str(path), row.get('qid'), 'missing_job_field', field])
            if row.get('conf_token_geo') is not None or row.get('conf_token_first') is not None:
                problems.append([str(path), row.get('qid'), 'job_has_formal_conf'])
    manifest = json.loads((corrected / 'recovery_manifest.json').read_text())
    expected_pending = {key: value for key, value in actions.items() if key != 'none'}
    if dict(job_actions) != expected_pending or manifest['action_counts'] != expected_pending:
        problems.append(['jobs_manifest_actions_inconsistent', dict(job_actions), expected_pending, manifest['action_counts']])
    tests = []
    for path in sorted((ROOT / 'audit/repair_20261116').glob('integrity_cpu_tests_*.json')):
        report = json.loads(path.read_text())
        mismatch = [name for name, checksum in report['code_sha256'].items()
                    if hashlib.sha256((ROOT / 'code' / name).read_bytes()).hexdigest() != checksum]
        tests.append({'file': str(path.relative_to(ROOT)), 'passed': report['passed'],
                      'failed': report['failed'], 'code_hash_mismatch': mismatch})
        if mismatch or report['failed']:
            problems.append(['test_evidence_not_final', str(path), mismatch])
    report = {'rows': sum(source['rows'] for source in sources), 'source_counts': sources,
              'required_action_baseline': dict(requirements), 'current_actions': dict(actions),
              'statuses': dict(counts), 'diagnostic_candidates_retained': candidates,
              'verified_records': verified, 'formal_conf_nonnull': formal_conf,
              'pending_jobs': total_jobs, 'job_actions': dict(job_actions),
              'manifest_actions': manifest['action_counts'],
              'needs_regeneration': manifest['needs_regeneration'],
              'cpu_tests': tests, 'violations': problems,
              'code_sha256': {name: hashlib.sha256((ROOT / 'code' / name).read_bytes()).hexdigest()
                              for name in ('answer_spans.py', 'recover_corrected.py', 'run_calib.py', 'rescore_actual.py')}}
    out = ROOT / 'audit/repair_20261116/recovery_integrity_audit_final.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
