#!/usr/bin/env python3
"""独立检查候选TF证据；完整cohort通过后才在新目录运行CPU分析。"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from analysis_gate import (FormalEvidenceError, inspect_file, inspect_matrix_file,
                           read_jsonl, sha256, PROTOCOL, MATRIX_PROTOCOL)

ROOT=Path(__file__).resolve().parent.parent
MODELS={'qwen':'qwen500','healthgpt':'healthgpt500','gemma4':'gemma500'}
ANALYSES=['calib_metrics.py','formal_stats.py','gee_stats.py','selective_full.py','selective_risk.py',
          'stability.py','fair_transfer.py','learning_curve.py','format_cond_calib.py','by_category.py',
          'make_tables.py','make_paper_figs.py']


def source_definitions(root, candidate_root=None, include_matrix=False):
    candidate=Path(candidate_root) if candidate_root is not None else root/'results/corrected/actual_fixed_output_tf_v1'
    main=candidate/'actual' if (candidate/'actual').exists() else candidate/'main'
    for model,stem in MODELS.items():
        yield {'dataset':'slake','model':model,'stem':stem,'folder':'raw','protocol':PROTOCOL,
               'original':root/'results/raw'/f'{stem}.jsonl','formal':main/f'{model}.jsonl'}
    for folder in ['raw_vqarad','raw_vqamed','raw_wording']:
        for model in ['qwen','healthgpt']:
            yield {'dataset':folder.removeprefix('raw_'),'model':model,'stem':model,'folder':folder,'protocol':PROTOCOL,
                   'original':root/'results'/folder/f'{model}.jsonl','formal':candidate/'external'/f'{model}.jsonl'}
    if include_matrix:
        for model,stem in MODELS.items():
            yield {'dataset':'assistant_prefill','model':model,'stem':model,'folder':'rescore','protocol':MATRIX_PROTOCOL,
                   'original':root/'results/raw'/f'{stem}.jsonl','formal':candidate/'rescore'/f'{model}.jsonl'}


def inspect_definition(definition, project_root):
    definition={k:str(v) if isinstance(v,Path) else v for k,v in definition.items()}
    formal=Path(definition['formal']);original=Path(definition['original'])
    try:
        if definition.get('protocol')==MATRIX_PROTOCOL:
            audit=inspect_matrix_file(formal,project_root,original)
        elif formal.exists() and definition['dataset']!='slake':
            selected=[r for r in read_jsonl(formal) if r.get('dataset')==definition['folder']]
            audit=inspect_file(formal,project_root,original,rows=selected)
            audit['filtered_dataset']=definition['folder']
        else:audit=inspect_file(formal,project_root,original)
        if audit['ready']:
            rows=read_jsonl(formal)
            if definition['dataset'] not in ['slake','assistant_prefill']:
                rows=[r for r in rows if r.get('dataset')==definition['folder']]
            if any(r.get('model')!=definition['model'] for r in rows):
                audit['ready']=False;audit['blocking_reasons']['expected_model_mismatch']=1
    except (FormalEvidenceError,ValueError,TypeError,OSError,KeyError) as error:
        audit={'path':str(formal),'ready':False,'exists':formal.exists(),'n_records':0,'n_accepted':0,
               'blocking_reasons':{'unreadable_or_malformed_evidence':1},'error':f'{type(error).__name__}: {error}'}
    return {**definition,'audit':audit}


def save_evidence(path,evidence):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def file_index(root):
    rows=[]
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ['.json','.md','.tex','.png','.pdf'] and path.name!='paper_evidence.json':
            rows.append({'path':str(path),'relative_path':str(path.relative_to(root)),
                         'bytes':path.stat().st_size,'sha256':sha256(path.read_bytes())})
    return rows


def stage(cohorts,root,compliant_only=False):
    # 发布前重新检查，拒绝初查后发生变化或新增的不完整输入。
    verified=[]
    for c in cohorts:
        current=inspect_definition(c,ROOT)
        if not current['audit']['ready'] or current['audit'].get('sha256')!=c['audit'].get('sha256'):
            raise FormalEvidenceError('候选在检查后发生变化或未完整验证: '+c['formal'])
        verified.append(current)
    staged=[];eligible={}
    for c in verified:
        if c['protocol']==MATRIX_PROTOCOL:continue
        rows=read_jsonl(c['formal'])
        if c['dataset']!='slake':rows=[r for r in rows if r.get('dataset')==c['folder']]
        if compliant_only:rows=[r for r in rows if r.get('marker_missing_fallback') is not True]
        if c['dataset']=='slake':
            by_qid={}
            for r in rows:by_qid.setdefault(str(r['qid']),set()).add(r['format'])
            eligible[c['model']]={q for q,fmts in by_qid.items() if fmts=={'A','B','C'}}
        staged.append((c,rows,None))
    for c in verified:
        if c['protocol']!=MATRIX_PROTOCOL:continue
        rows=read_jsonl(c['formal']);qids=eligible.get(c['model'],set())
        if compliant_only:rows=[r for r in rows if str(r['qid']) in qids]
        staged.append((c,rows,sorted(qids) if compliant_only else None))
    if any(not rows for c,rows,qids in staged):
        raise FormalEvidenceError('派生cohort为空，拒绝发布')
    snapshot={'protocol':PROTOCOL,'cohort':'compliant_only' if compliant_only else 'all_formal',
              'file_sha256':{},'source_formal_sha256':{}}
    for c,rows,qids in staged:
        path=root/c['folder']/f'{c["stem"]}.jsonl'
        if path.exists():raise FileExistsError('拒绝覆盖已有分析快照: '+str(path))
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        relative=str(path.relative_to(root))
        snapshot['file_sha256'][relative]=sha256(path.read_bytes())
        snapshot['source_formal_sha256'][relative]={'path':c['formal'],'sha256':c['audit']['sha256'],
            'expected_source':c['original'],'protocol':c['protocol'],'n_staged':len(rows),
            'dataset':None if c['dataset'] in ['slake','assistant_prefill'] else c['folder']}
        if qids is not None:snapshot['source_formal_sha256'][relative]['eligible_qids']=qids
    save_evidence(root/'cohort_snapshot.json',snapshot)


def run_script(root,script,args=None):
    env=dict(os.environ,MEDVQA_RESULTS_ROOT=str(root),CUDA_VISIBLE_DEVICES='')
    cmd=[sys.executable,str(ROOT/'code'/script),*(args or [])]
    started=time.time()
    process=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True)
    log=root/'analysis_logs'/f'{Path(script).stem}.log';log.parent.mkdir(parents=True,exist_ok=True)
    log.write_text(process.stdout+'\nSTDERR\n'+process.stderr,encoding='utf-8')
    result={'script':script,'args':args or [],'returncode':process.returncode,'seconds':time.time()-started,
            'log':str(log),'source_sha256':sha256((ROOT/'code'/script).read_bytes())}
    if process.returncode!=0:raise RuntimeError(json.dumps(result,ensure_ascii=False))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--run',action='store_true')
    mode.add_argument('--inspect-only',action='store_true')
    mode.add_argument('--plan',action='store_true')
    parser.add_argument('--candidate-root',default=str(ROOT/'results/corrected/actual_fixed_output_tf_v1'))
    parser.add_argument('--publish-root','--out-root',dest='publish_root',default=str(ROOT/'results/corrected/paper'))
    parser.add_argument('--require-external',action='store_true')
    parser.add_argument('--require-matrix',action='store_true')
    args=parser.parse_args(argv)
    candidate=Path(args.candidate_root).resolve();out=Path(args.publish_root).resolve()
    corrected=(ROOT/'results/corrected').resolve()
    if not out.is_relative_to(corrected) or out in [corrected,corrected/'raw']:
        parser.error('发布输出必须使用results/corrected内的独立子目录')
    if out.is_relative_to(candidate) or candidate.is_relative_to(out):
        parser.error('候选目录与发布目录不得相同或互相包含')
    cohorts=[inspect_definition(d,ROOT) for d in source_definitions(ROOT,candidate,args.require_matrix)]
    required=[c for c in cohorts if c['dataset']=='slake' or
              (args.require_external and c['protocol']==PROTOCOL) or
              (args.require_matrix and c['protocol']==MATRIX_PROTOCOL)]
    ready=bool(required) and all(c['audit']['ready'] for c in required)
    # 外部cohort只作为完整的已请求分支纳入，不能随文件出现而悄然改变主分析。
    selected=required
    evidence={'version':'formal_paper_evidence_v2','timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'protocol':PROTOCOL,'status':'ready_not_run' if ready else 'blocked_missing_formal_evidence',
              'ready':ready,'published':False,'requested_run':args.run,'candidate_root':str(candidate),
              'output_root':str(out),'inspection_has_no_writes':not args.run,'python':sys.version,
              'platform':platform.platform(),'cohorts':cohorts,
              'required_scope':{'main':True,'external':args.require_external,'assistant_prefill_matrix':args.require_matrix},
              'cohort_policy':{'all_formal':'完整原始固定输出TF；缺行、旧概率及未验证来源一律阻断',
                               'compliant_only':'独立排除fallback；矩阵仅保留三种生成格式均合规的整题',
                               'external':'仅显式要求且全部完整时纳入',
                               'assistant_prefill':'assistant_prefill_v2独立协议；不得当作original fixed TF'},
              'tables':[],'figures':[],'analyses':[],'blocking_reasons':{},
              'planned_analyses':ANALYSES+(['rescore_analysis.py'] if args.require_matrix else [])}
    for c in required:
        if not c['audit']['ready']:
            evidence['blocking_reasons'][f'{c["dataset"]}:{c["model"]}']=c['audit']['blocking_reasons']
    if not args.run or not ready:
        print(json.dumps(evidence,ensure_ascii=False,allow_nan=False))
        return 0 if ready else 2
    # 不覆盖旧证据或既有运行输出；失败也只留下显式失败的独立发布目录。
    if out.exists():raise FileExistsError('发布目录已存在，拒绝覆盖: '+str(out))
    evidence['source_code_sha256']={p.name:sha256(p.read_bytes()) for p in sorted((ROOT/'code').glob('*.py'))}
    import importlib.metadata
    evidence['package_versions']={name:importlib.metadata.version(name) for name in ['numpy','scikit-learn','statsmodels','matplotlib','pandas']}
    try:
        for label,compliant in [('all_formal',False),('compliant_only',True)]:
            runroot=out/label;stage(selected,runroot,compliant)
            runs=[]
            for folder,summaryfolder in [('raw_vqarad','summary_vqarad'),('raw_vqamed','summary_vqamed')]:
                if (runroot/folder).exists():
                    runs.append(run_script(runroot,'calib_metrics.py',['--raw-dir',str(runroot/folder),'--out',str(runroot/summaryfolder)]))
            for script in ANALYSES:runs.append(run_script(runroot,script))
            if args.require_matrix:runs.append(run_script(runroot,'rescore_analysis.py'))
            evidence['analyses'].append({'cohort':label,'runs':runs})
            evidence[label]={'root':str(runroot),'files':file_index(runroot)}
        for label in ['all_formal','compliant_only']:
            files=evidence[label]['files']
            evidence['tables'] += [{'cohort':label,**v} for v in files if '/tables/' in v['path']]
            evidence['figures'] += [{'cohort':label,**v} for v in files if v['path'].endswith(('.pdf','.png'))]
        for c in selected:
            checked=inspect_definition(c,ROOT)
            if not checked['audit']['ready'] or checked['audit'].get('sha256')!=c['audit'].get('sha256'):
                raise FormalEvidenceError('分析期间候选来源发生变化: '+c['formal'])
        evidence['status']='completed';evidence['published']=True
    except Exception as error:
        evidence['status']='failed_closed_analysis_error';evidence['analysis_error']=f'{type(error).__name__}: {error}'
        save_evidence(out/'paper_evidence.json',evidence)
        raise
    save_evidence(out/'paper_evidence.json',evidence)
    print(json.dumps({'status':evidence['status'],'evidence':str(out/'paper_evidence.json')},ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
