#!/usr/bin/env python3
"""论文分析的正式固定输出TF证据门禁；候选概率不能获得准入。"""
from collections import Counter
import hashlib
import json
import math
import re
from pathlib import Path

PROTOCOL = 'original_prompt_fixed_full_output_tf_v1'
SPAN_VERSION = 'final-answer-v2'
FORMAL_RECOVERY_STATUS = {'verified_saved_output_teacher_forcing', 'regenerated_with_token_evidence'}

class FormalEvidenceError(RuntimeError):
    pass

def sha256(value):
    if isinstance(value,str): value=value.encode('utf-8')
    return hashlib.sha256(value).hexdigest()

def read_jsonl(path):
    path=Path(path)
    if not path.exists():
        raise FormalEvidenceError(f'缺少输入: {path}')
    rows=[]
    for i,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        try:
            row=json.loads(line)
            if not isinstance(row,dict):raise ValueError('记录必须为JSON对象')
            rows.append(row)
        except Exception as error:
            raise FormalEvidenceError(f'非法JSONL {path}:{i}: {error}') from error
    return rows

def finite_probability(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and 0<=value<=1

def token_ids(value, allow_empty=False):
    return (isinstance(value,list) and (allow_empty or bool(value)) and
            all(isinstance(v,int) and not isinstance(v,bool) and v>=0 for v in value))

def resolve_path(path, project_root=None):
    path=Path(path)
    return path if path.is_absolute() or project_root is None else Path(project_root)/path

def row_reasons(row, project_root=None, manifest_cache=None):
    """拒绝缺证据或自相矛盾的行；不修改记录，不生成占位概率。"""
    reasons=[]
    if row.get('status')!='ok':reasons.append('status_not_ok')
    if row.get('protocol')!=PROTOCOL:reasons.append('not_original_prompt_fixed_output_tf')
    if row.get('teacher_forcing') is not True:reasons.append('teacher_forcing_not_verified')
    if row.get('regeneration') is not False:reasons.append('saved_output_cohort_not_verified')
    for field in ['original_prompt_token_ids_match','full_output_roundtrip_verified','source_raw_complete_verified']:
        if row.get(field) is not True:reasons.append(field+'_missing_or_false')
    if row.get('span_version')!=SPAN_VERSION:reasons.append('span_version_unverified')
    allowed_formats=['A','B','C','A2','A3','B2','B3','C2','C3'] if row.get('dataset')=='raw_wording' else ['A','B','C']
    if row.get('format') not in allowed_formats:reasons.append('invalid_format')
    if isinstance(row.get('qid'),bool) or not isinstance(row.get('qid'),(str,int)) or row.get('qid')=='':reasons.append('missing_qid')
    if not row.get('image_path'):reasons.append('missing_image_cluster')
    if not finite_probability(row.get('conf_token_geo')):reasons.append('missing_or_invalid_probability')
    if not isinstance(row.get('correct'),(bool,int)) or row.get('correct') not in [0,1]:
        reasons.append('invalid_correctness_label')
    elif not isinstance(row.get('ans_norm'),str) or not isinstance(row.get('gold_norm'),str):
        reasons.append('missing_normalized_final_answer')
    elif bool(row['correct']) != (bool(row['ans_norm']) and row['ans_norm']==row['gold_norm']):
        reasons.append('final_answer_correctness_mismatch')
    if not isinstance(row.get('marker_missing_fallback'),bool):reasons.append('missing_format_compliance_evidence')
    raw=row.get('raw_full',row.get('raw'))
    if not isinstance(raw,str) or not raw.strip():reasons.append('missing_saved_full_output')
    elif any(row.get(key)!=sha256(raw) for key in ['raw_sha256','source_raw_sha256']):
        reasons.append('saved_output_hash_mismatch')
    ids=row.get('full_output_token_ids'); lp=row.get('full_output_token_logprobs')
    spanids=row.get('span_token_ids'); spanlp=row.get('span_token_logprobs'); idx=row.get('span_token_indices')
    valid_lp=lambda values:isinstance(values,list) and bool(values) and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v<=1e-6 for v in values)
    if not token_ids(ids) or not valid_lp(lp) or len(ids)!=len(lp):
        reasons.append('invalid_full_output_token_evidence')
    if not token_ids(spanids) or not valid_lp(spanlp) or len(spanids)!=len(spanlp):
        reasons.append('invalid_answer_token_evidence')
    elif row.get('span_len')!=len(spanids):reasons.append('span_length_mismatch')
    else:
        conf=math.exp(sum(spanlp)/len(spanlp))
        if not finite_probability(row.get('conf_token_geo')) or not math.isclose(conf,row['conf_token_geo'],rel_tol=1e-7,abs_tol=1e-10):
            reasons.append('span_probability_mismatch')
        if not finite_probability(row.get('conf_token_first')) or not math.isclose(math.exp(spanlp[0]),row['conf_token_first'],rel_tol=1e-7,abs_tol=1e-10):
            reasons.append('first_token_probability_mismatch')
    if not isinstance(idx,list) or not idx or any(not isinstance(i,int) or isinstance(i,bool) for i in idx):
        reasons.append('invalid_answer_span_indices')
    elif isinstance(ids,list) and isinstance(lp,list):
        if idx!=sorted(set(idx)) or min(idx)<0 or max(idx)>=min(len(ids),len(lp)):
            reasons.append('answer_span_indices_out_of_bounds')
        elif [ids[i] for i in idx]!=spanids or [lp[i] for i in idx]!=spanlp:
            reasons.append('answer_span_not_part_of_full_output')
    prompt=row.get('prompt_text'); promptids=row.get('prompt_token_ids')
    if not isinstance(prompt,str) or row.get('prompt_sha256')!=sha256(prompt):
        reasons.append('prompt_hash_mismatch')
    if not token_ids(promptids) or row.get('prompt_len')!=len(promptids):
        reasons.append('prompt_token_evidence_missing')
    manpath=row.get('manifest_path')
    cache={} if manifest_cache is None else manifest_cache
    if not manpath:
        reasons.append('missing_manifest')
    else:
        path=Path(manpath)
        if not path.is_absolute() and project_root:path=Path(project_root)/path
        key=str(path)
        if key not in cache:
            try:
                manifest=json.loads(path.read_text(encoding='utf-8'))
                stable={k:manifest[k] for k in ('protocol','model','cohort','checkpoint_config_sha256','code_sha256','source_sha256','span_version')}
                signature=sha256(json.dumps(stable,sort_keys=True))
                errors=[]
                if manifest.get('run_signature')!=signature:errors.append('manifest_run_signature_mismatch')
                if manifest.get('protocol')!=PROTOCOL or manifest.get('teacher_forcing') is not True or manifest.get('regeneration') is not False:
                    errors.append('manifest_protocol_unverified')
                source_hashes=manifest.get('source_sha256',{})
                if not source_hashes:errors.append('manifest_missing_source_hashes')
                for source,digest in source_hashes.items():
                    sourcepath=Path(source)
                    if not sourcepath.is_absolute() and project_root:sourcepath=Path(project_root)/sourcepath
                    if not sourcepath.exists() or sha256(sourcepath.read_bytes())!=digest:
                        errors.append('manifest_source_file_hash_mismatch')
                cache[key]=(manifest,errors)
            except Exception:
                cache[key]=({},['manifest_missing_or_unreadable'])
        manifest,errors=cache[key]
        reasons.extend(errors)
        if row.get('run_signature')!=manifest.get('run_signature') or not row.get('run_signature'):
            reasons.append('row_manifest_signature_mismatch')
        if row.get('model')!=manifest.get('model'):reasons.append('row_manifest_model_mismatch')
        if row.get('stored_source') not in manifest.get('source_sha256',{}):
            reasons.append('row_source_not_in_manifest')
    return sorted(set(reasons))

def cohort_key(row):
    return (str(row.get('qid')),row.get('format'))

def inspect_file(path, project_root=None, expected_source=None, rows=None):
    path=Path(path); cache={}; reasons=Counter(); accepted=[]; rejected=[]; duplicates=[]; seen=set(); excluded=[]; declared={}; ex_path=Path(str(path)+'.exclusions.json')
    if expected_source is not None:expected_source=resolve_path(expected_source,project_root)
    if not path.exists():
        source=Path(expected_source) if expected_source is not None else None
        expected=read_jsonl(source) if source is not None and source.exists() else []
        return {'path':str(path),'ready':False,'exists':False,'n_records':0,'n_accepted':0,
                'n_rejected':0,'blocking_reasons':{'missing_input_file':1,'missing_formal_expected_rows':len(expected)},
                'missing_expected_keys':[list(cohort_key(r)) for r in expected],
                'unexpected_keys':[],'counts_by_format':{},'accepted_keys':[],
                'source':{'path':str(source),'sha256':sha256(source.read_bytes()),'n_records':len(expected),
                          'counts_by_format':dict(Counter(r.get('format') for r in expected)),
                          'n_images':len({r.get('image_path') for r in expected})} if expected else None}
    rows=read_jsonl(path) if rows is None else rows
    if ex_path.exists():
        try:
            document=json.loads(ex_path.read_text(encoding='utf-8'))
            if document.get('policy')!='declared_blocked_record_excluded_from_denominator':
                reasons['invalid_exclusion_policy']+=1
            for item in document.get('exclusions',[]):
                if not item.get('reason') or not item.get('evidence'):
                    reasons['exclusion_without_reason_or_evidence']+=1
                    continue
                declared[(str(item.get('qid')),item.get('format'))]=item
        except Exception:
            reasons['invalid_exclusion_file']+=1
    if expected_source is None:
        sources={r.get('stored_source') for r in rows}
        if len(sources)==1 and None not in sources:
            expected_source=resolve_path(next(iter(sources)),project_root)
        else:reasons['missing_expected_source']+=1
    for row in rows:
        key=cohort_key(row)
        if key in seen:duplicates.append(list(key))
        seen.add(key)
        why=row_reasons(row,project_root,cache)
        if why:
            if key in declared:
                excluded.append({'key':list(key),'reasons':why,'declaration':declared[key]})
            else:
                reasons.update(why); rejected.append({'key':list(key),'reasons':why})
        else:
            if key in declared:reasons['declared_exclusion_but_row_is_valid']+=1
            accepted.append(row)
    missing=[]; unexpected=[]; source_metadata=None
    if expected_source is not None:
        source=Path(expected_source)
        if source.exists():
            source_rows=read_jsonl(source); expected={cohort_key(r) for r in source_rows}
            valid={cohort_key(r) for r in accepted}
            missing=sorted(expected-valid-set(declared)); unexpected=sorted(seen-expected)
            unknown=[list(k) for k in declared if k not in expected or k not in seen]
            if unknown:reasons['declared_exclusion_key_unknown']+=len(unknown)
            original={cohort_key(r):r for r in source_rows}
            if not source_rows:reasons['empty_expected_source']+=1
            for r in accepted:
                if resolve_path(r.get('stored_source',''),project_root).resolve()!=source.resolve():
                    reasons['expected_source_path_mismatch']+=1
                if r.get('dataset') is not None and r.get('dataset')!=source.parent.name:
                    reasons['expected_source_dataset_mismatch']+=1
                old=original.get(cohort_key(r))
                if old is None:continue
                full=old.get('raw_full',old.get('raw'))
                if r.get('raw_full',r.get('raw'))!=full:
                    reasons['original_saved_output_mismatch']+=1
                for f in ['question','image_path','category','gold']:
                    if r.get(f)!=old.get(f):reasons['original_identity_mismatch_'+f]+=1
            source_metadata={'path':str(source),'sha256':sha256(source.read_bytes()),'n_records':len(source_rows),
                             'n_unique_keys':len(expected),'n_images':len({r.get('image_path') for r in source_rows})}
            if len(expected)!=len(source_rows):reasons['duplicate_original_source_keys']+=1
        else:reasons['missing_expected_source']+=1
    if duplicates:reasons['duplicate_input_keys']+=len(duplicates)
    if missing:reasons['missing_formal_expected_rows']+=len(missing)
    if unexpected:reasons['unexpected_cohort_keys']+=len(unexpected)
    if not accepted:reasons['no_verified_formal_rows']+=1
    by_format={f:{'n':sum(r.get('format')==f for r in accepted),
                  'n_images':len({r['image_path'] for r in accepted if r.get('format')==f}),
                  'n_fallback':sum(r.get('format')==f and r.get('marker_missing_fallback') is True for r in accepted),
                  'n_compliant':sum(r.get('format')==f and r.get('marker_missing_fallback') is not True for r in accepted)} for f in sorted({r['format'] for r in accepted})}
    ready=bool(accepted) and not rejected and not reasons and not duplicates and len(excluded)==len(declared)
    return {'path':str(path),'exists':True,'ready':ready,'sha256':sha256(path.read_bytes()),
            'n_records':len(rows),'n_accepted':len(accepted),'n_rejected':len(rejected),
            'n_excluded':len(excluded),'excluded_rows':excluded,
            'exclusions_file':str(ex_path) if ex_path.exists() else None,
            'n_images':len({r['image_path'] for r in accepted}),
            'counts_by_format':by_format,'blocking_reasons':dict(reasons),
            'rejected_rows':rejected,'duplicate_keys':duplicates,
            'missing_expected_keys':[list(k) for k in missing],'unexpected_keys':[list(k) for k in unexpected],
            'accepted_keys':[list(cohort_key(r)) for r in accepted],
            'source':source_metadata,'protocols':dict(Counter(r.get('protocol','missing') for r in rows)),
            'span_versions':dict(Counter(r.get('span_version','missing') for r in rows)),
            'run_signatures':sorted({r['run_signature'] for r in accepted})}

def require_formal_file(path, project_root=None, expected_source=None):
    evidence=inspect_file(path,project_root,expected_source)
    if not evidence['ready']:
        raise FormalEvidenceError(f'正式论文分析被阻断: {path}; {evidence["blocking_reasons"]}')
    return evidence

MATRIX_PROTOCOL = 'assistant_prefill_v2'
MATRIX_SUFFIXES = {'A':'','B':'Observation: The image shows a medical scan.\nAnswer: ','C':'Answer: '}

def inspect_matrix_file(path, project_root=None, expected_source=None):
    """逐格核对 assistant-prefill 证据；不沿用旧矩阵或缺格子集。"""
    path=Path(path); reasons=Counter(); rejected=[]; accepted=[]; seen=set(); duplicates=[]; excluded=[]; declared={}; ex_path=Path(str(path)+'.exclusions.json')
    if not path.exists():
        return {'path':str(path),'exists':False,'ready':False,'n_records':0,'n_accepted':0,
                'blocking_reasons':{'missing_input_file':1},'accepted_keys':[]}
    rows=read_jsonl(path)
    if ex_path.exists():
        try:
            document=json.loads(ex_path.read_text(encoding='utf-8'))
            if document.get('policy')!='declared_blocked_record_excluded_from_denominator':
                reasons['invalid_exclusion_policy']+=1
            for item in document.get('exclusions',[]):
                if not item.get('reason') or not item.get('evidence'):
                    reasons['exclusion_without_reason_or_evidence']+=1
                    continue
                declared[(str(item.get('qid')),item.get('g'),item.get('s'))]=item
        except Exception:
            reasons['invalid_exclusion_file']+=1
    sources={r.get('source_path') for r in rows}
    if expected_source is None:
        if len(sources)==1 and None not in sources:expected_source=next(iter(sources))
        else:reasons['missing_expected_source']+=1
    source=resolve_path(expected_source,project_root) if expected_source is not None else None
    original=read_jsonl(source) if source is not None and source.exists() else []
    if not original:reasons['missing_or_empty_expected_source']+=1
    source_rows={cohort_key(r):r for r in original}
    if len(source_rows)!=len(original):reasons['duplicate_original_source_keys']+=1
    qids={str(r.get('qid')) for r in original}
    expected={(q,g,s) for q in qids for g in MATRIX_SUFFIXES for s in MATRIX_SUFFIXES}
    if set(source_rows)!={(q,g) for q in qids for g in MATRIX_SUFFIXES}:
        reasons['incomplete_original_three_format_cohort']+=1
    digest=sha256(source.read_bytes()) if source is not None and source.exists() else None
    manifest_path=path.with_suffix('.manifest.json')
    try:
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if not isinstance(manifest,dict):raise ValueError('manifest必须为对象')
    except Exception:manifest={};reasons['matrix_manifest_missing_or_unreadable']+=1
    provenance=['protocol','model','checkpoint_config_sha256','code_sha256','code_file_hashes',
                'source_path','source_file_sha256','seed_file_sha256','suffixes','neutral_instruction']
    if manifest.get('protocol')!=MATRIX_PROTOCOL:reasons['matrix_manifest_protocol_unverified']+=1
    if manifest.get('suffixes')!=MATRIX_SUFFIXES:reasons['matrix_manifest_suffix_mismatch']+=1
    if manifest.get('source_file_sha256')!=digest:reasons['matrix_manifest_source_hash_mismatch']+=1
    if any(not manifest.get(field) for field in provenance if field!='suffixes'):
        reasons['matrix_manifest_incomplete_provenance']+=1
    from answer_spans import text_span
    for row in rows:
        why=[];key=(str(row.get('qid')),row.get('g'),row.get('s'))
        if key in seen:duplicates.append(list(key))
        seen.add(key)
        if row.get('protocol')!=MATRIX_PROTOCOL:why.append('not_assistant_prefill_v2')
        if row.get('status')!='ok':why.append('status_not_ok')
        if key not in expected:why.append('unexpected_matrix_cell')
        if row.get('g') not in MATRIX_SUFFIXES or row.get('s') not in MATRIX_SUFFIXES:why.append('invalid_matrix_format')
        if not row.get('image_path'):why.append('missing_image_cluster')
        for field in provenance:
            if row.get(field) is None or row.get(field)!=manifest.get(field):why.append('matrix_manifest_mismatch_'+field)
        if source is not None and resolve_path(row.get('source_path',''),project_root).resolve()!=source.resolve():why.append('expected_source_path_mismatch')
        if not digest or row.get('source_file_sha256')!=digest:why.append('manifest_source_file_hash_mismatch')
        if row.get('source_raw_complete_verified') is not True or row.get('source_raw_truncated') is not False:why.append('matrix_saved_output_incomplete')
        old=source_rows.get((key[0],row.get('g')))
        if old is not None:
            raw=old.get('raw_full',old.get('raw'))
            if not isinstance(raw,str) or not raw.strip() or row.get('source_raw_sha256')!=sha256(raw):why.append('saved_output_hash_mismatch')
            elif row.get('answer')!=text_span(raw,row['g'])['ans_raw']:why.append('matrix_answer_source_mismatch')
            for field in ['question','image_path','category','gold']:
                if row.get(field)!=old.get(field):why.append('original_identity_mismatch_'+field)
        ans=row.get('ans_norm');gold=row.get('gold_norm')
        if not isinstance(ans,str) or not isinstance(gold,str) or not isinstance(row.get('correct'),(bool,int)) or row.get('correct') not in [0,1] or bool(row.get('correct'))!=(bool(ans) and ans==gold):why.append('final_answer_correctness_mismatch')
        ids=row.get('answer_token_ids');lp=row.get('span_token_logprobs');positions=row.get('scored_positions')
        if not token_ids(ids) or row.get('scored_token_ids')!=ids or not isinstance(lp,list) or len(lp)!=len(ids or []) or not lp or any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v>1e-6 for v in lp):why.append('invalid_matrix_token_evidence')
        else:
            if row.get('span')!=len(ids):why.append('span_length_mismatch')
            if not finite_probability(row.get('conf')) or not math.isclose(row['conf'],math.exp(sum(lp)/len(lp)),rel_tol=1e-7,abs_tol=1e-10):why.append('span_probability_mismatch')
            if not finite_probability(row.get('conf_first')) or not math.isclose(row['conf_first'],math.exp(lp[0]),rel_tol=1e-7,abs_tol=1e-10):why.append('first_token_probability_mismatch')
        promptids=row.get('prompt_token_ids');prompt=row.get('prompt_text');suffix=MATRIX_SUFFIXES.get(row.get('s'))
        if not token_ids(promptids) or row.get('prompt_len')!=len(promptids):why.append('prompt_token_evidence_missing')
        if not isinstance(prompt,str) or row.get('prompt_sha256')!=sha256(prompt):why.append('prompt_hash_mismatch')
        if suffix is None or row.get('assistant_prefill')!=suffix or not isinstance(prompt,str) or not prompt.endswith(suffix):why.append('assistant_prefill_mismatch')
        elif row.get('base_prompt_sha256')!=sha256(prompt[:-len(suffix)] if suffix else prompt):why.append('base_prompt_hash_mismatch')
        elif re.search(r'(?:<\|im_start\|>assistant|<start_of_turn>model|<\|turn>model)\n(?:<channel>[^<]*</channel>\s*|<\|channel\|>[^<]*<\|message\|>\s*)?$',prompt[:-len(suffix)] if suffix else prompt) is None:why.append('assistant_role_boundary_unverified')
        if row.get('token_boundary_policy')!='full_text_tokenization_include_answer_overlapping_non_special_tokens':why.append('matrix_boundary_policy_unverified')
        if not token_ids(positions) or not token_ids(ids) or len(positions)!=len(ids) or positions!=sorted(set(positions)) or positions[0]!=row.get('prompt_len'):why.append('invalid_matrix_scored_positions')
        special=row.get('special_token_ids');prefix=row.get('prefix_token_ids');prefixpos=row.get('prefix_token_positions')
        if not token_ids(special,True) or not token_ids(prefix,True) or not token_ids(prefixpos,True):why.append('missing_special_or_prefix_evidence')
        elif token_ids(ids) and (set(ids)&set(special) or set(prefix)&set(special) or (isinstance(positions,list) and set(prefixpos)&set(positions))):why.append('matrix_prefix_or_special_scored')
        if why:
            if key in declared:
                excluded.append({'key':list(key),'reasons':sorted(set(why)),'declaration':declared[key]})
            else:
                reasons.update(why);rejected.append({'key':list(key),'reasons':sorted(set(why))})
        else:
            if key in declared:reasons['declared_exclusion_but_cell_is_valid']+=1
            accepted.append(row)
    valid={(str(r['qid']),r['g'],r['s']) for r in accepted}
    missing=sorted(expected-valid-set(declared))
    unknown=[list(k) for k in declared if k not in expected or k not in seen]
    if unknown:reasons['declared_exclusion_key_unknown']+=len(unknown)
    if missing:reasons['missing_expected_matrix_cells']+=len(missing)
    if duplicates:reasons['duplicate_matrix_cells']+=len(duplicates)
    if not accepted:reasons['no_verified_matrix_cells']+=1
    return {'path':str(path),'exists':True,'ready':bool(accepted) and not reasons and not rejected and len(excluded)==len(declared),'sha256':sha256(path.read_bytes()),
            'n_records':len(rows),'n_accepted':len(accepted),'n_rejected':len(rejected),'n_expected':len(expected),
            'n_excluded':len(excluded),'excluded_rows':excluded,
            'exclusions_file':str(ex_path) if ex_path.exists() else None,
            'n_questions':len(qids),'blocking_reasons':dict(reasons),'rejected_rows':rejected,
            'missing_expected_keys':[list(k) for k in missing],'duplicate_keys':duplicates,
            'accepted_keys':[list((str(r['qid']),r['g'],r['s'])) for r in accepted],
            'source':{'path':str(source),'sha256':digest,'n_records':len(original)}}

def require_matrix_file(path, project_root=None, expected_source=None):
    evidence=inspect_matrix_file(path,project_root,expected_source)
    if not evidence['ready']:raise FormalEvidenceError(f'正式矩阵分析被阻断: {path}; {evidence["blocking_reasons"]}')
    return evidence

def require_snapshot_file(path, root, snapshot, project_root):
    """派生子集必须与一个已完整验证的来源逐行一致。"""
    relative=str(Path(path).relative_to(root))
    if snapshot.get('protocol')!=PROTOCOL or snapshot.get('file_sha256',{}).get(relative)!=sha256(Path(path).read_bytes()):
        raise FormalEvidenceError('正式分析快照与源文件不一致: '+str(path))
    origin=snapshot.get('source_formal_sha256',{}).get(relative,{})
    source=resolve_path(origin.get('path',''),project_root)
    if not source.is_file() or origin.get('sha256')!=sha256(source.read_bytes()):
        raise FormalEvidenceError('快照完整来源不可验证: '+str(path))
    original_rows=read_jsonl(source)
    dataset=origin.get('dataset')
    if dataset is not None:original_rows=[r for r in original_rows if r.get('dataset')==dataset]
    if origin.get('protocol',PROTOCOL)==MATRIX_PROTOCOL:
        evidence=require_matrix_file(source,project_root,origin.get('expected_source'))
        if snapshot.get('cohort')=='compliant_only':
            stems={'qwen':'qwen500','healthgpt':'healthgpt500','gemma4':'gemma500'}
            model=original_rows[0].get('model') if original_rows else None
            if model not in stems:raise FormalEvidenceError('矩阵缺少主模型身份')
            base=Path(root)/'raw'/(stems[model]+'.jsonl')
            require_snapshot_file(base,root,snapshot,project_root)
            formats={}
            for row in read_jsonl(base):formats.setdefault(str(row['qid']),set()).add(row['format'])
            qids={q for q,fmts in formats.items() if fmts=={'A','B','C'}}
            if set(origin.get('eligible_qids',[]))!=qids:raise FormalEvidenceError('矩阵合规整题cohort不一致')
            original_rows=[r for r in original_rows if str(r.get('qid')) in qids]
    else:
        evidence=inspect_file(source,project_root,origin.get('expected_source'),rows=original_rows)
        if not evidence['ready']:raise FormalEvidenceError('快照完整来源未通过门禁: '+str(source))
        if snapshot.get('cohort')=='compliant_only':original_rows=[r for r in original_rows if r.get('marker_missing_fallback') is not True]
    if snapshot.get('cohort') not in ['all_formal','compliant_only'] or read_jsonl(path)!=original_rows:
        raise FormalEvidenceError('快照不是所声明的完整cohort派生: '+str(path))
    return evidence

def require_analysis_inputs(script, results_root):
    """保护直接CLI入口；导入统计函数及合成边界测试不产生论文分析。"""
    import sys
    project_root=Path(__file__).resolve().parent.parent
    root=Path(results_root).resolve()
    args=sys.argv[1:]
    path_flags=['--raw-dir','--out-dir','--out','--rescore-dir','--aprime-dir']
    for arg in args:
        option=arg.split('=',1)[0]
        if option.startswith('--') and option not in path_flags and any(flag.startswith(option) for flag in path_flags):
            raise FormalEvidenceError('路径参数不得缩写: '+option)
    def argvalue(flag,default):
        value=default
        for index,arg in enumerate(args):
            if arg==flag:
                if index+1>=len(args) or args[index+1].startswith('--'):raise FormalEvidenceError('缺少CLI参数: '+flag)
                value=args[index+1]
            elif arg.startswith(flag+'='):value=arg.split('=',1)[1]
        return Path(value).resolve()
    raw=argvalue('--raw-dir',root/'raw')
    output=argvalue('--out-dir',argvalue('--out',root))
    corrected=(project_root/'results/corrected').resolve()
    if not output.is_relative_to(corrected):
        raise FormalEvidenceError('正式分析输出必须保存在results/corrected内')
    if not raw.exists():raise FormalEvidenceError('缺少正式输入目录: '+str(raw))
    paths=sorted(raw.glob('*.jsonl'))
    if not paths:raise FormalEvidenceError('正式输入为空: '+str(raw))
    if script!='calib_metrics':
        missing=[name for name in ['qwen500','healthgpt500','gemma500'] if not (raw/(name+'.jsonl')).exists()]
        if missing:raise FormalEvidenceError('缺少主模型cohort: '+','.join(missing))
    reports=[]
    snapshot_path=root/'cohort_snapshot.json'
    snapshot=json.loads(snapshot_path.read_text(encoding='utf-8')) if snapshot_path.exists() else None
    for path in paths:
        if snapshot is not None and path.is_relative_to(root):
            reports.append(require_snapshot_file(path,root,snapshot,project_root))
        else:
            sources={r.get('stored_source') for r in read_jsonl(path)}
            if len(sources)!=1 or None in sources:
                raise FormalEvidenceError('缺少可验证的原始cohort清单: '+str(path))
            source=Path(next(iter(sources)))
            if raw.name in ['raw_vqarad','raw_vqamed'] and source.parent.name!=raw.name:
                raise FormalEvidenceError('外部数据集与原始cohort不匹配: '+str(path))
            reports.append(require_formal_file(path,project_root,source))
    if script in ['formal_stats','gee_stats']:
        for folder in ['raw_vqarad','raw_vqamed']:
            for path in sorted((root/folder).glob('*.jsonl')):
                if snapshot is not None:
                    reports.append(require_snapshot_file(path,root,snapshot,project_root))
                else:
                    sources={r.get('stored_source') for r in read_jsonl(path)}
                    if len(sources)!=1 or None in sources:
                        raise FormalEvidenceError('缺少外部原始cohort清单: '+str(path))
                    reports.append(require_formal_file(path,project_root,next(iter(sources))))
    optional={'rescore_analysis':'--rescore-dir','actual_analysis':None,'aprime_analysis':'--aprime-dir','fix_figs':None}
    if script in optional or script=='make_paper_figs' or script=='make_tables':
        directories=[]
        if script=='rescore_analysis':directories=[argvalue('--rescore-dir',root/'rescore')]
        if script=='aprime_analysis':directories=[argvalue('--aprime-dir',root/'raw_aprime')]
        if script=='actual_analysis':directories=[root/'rescore_actual',root/'rescore']
        for directory in directories:
            matrixpaths=sorted(directory.glob('*.jsonl'))
            if not matrixpaths:raise FormalEvidenceError('缺少已验证对照数据: '+str(directory))
            if script=='rescore_analysis':
                missing=[name for name in ['qwen','healthgpt','gemma4'] if not (directory/(name+'.jsonl')).exists()]
                if missing:raise FormalEvidenceError('缺少主模型矩阵cohort: '+','.join(missing))
            for path in matrixpaths:
                if snapshot is not None and path.is_relative_to(root):reports.append(require_snapshot_file(path,root,snapshot,project_root))
                elif script=='rescore_analysis':reports.append(require_matrix_file(path,project_root))
                else:reports.append(require_formal_file(path,project_root))
        if script in ['make_paper_figs','make_tables','fix_figs']:
            for relative in ['rescore_analysis/rescore_summary.json','rescore_actual/actual_vs_parallel.json']:
                path=root/relative
                if path.exists() and json.loads(path.read_text(encoding='utf-8')):
                    raise FormalEvidenceError('历史重评分摘要未通过正式TF矩阵门禁，不可加入正式图表: '+str(path))
    return reports

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+');p.add_argument('--project-root',default=str(Path(__file__).resolve().parent.parent));p.add_argument('--out');a=p.parse_args()
    files=[inspect_file(path,a.project_root) for path in a.inputs]
    report={'protocol':PROTOCOL,'ready':all(f['ready'] for f in files),'files':files}
    if a.out:
        out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'ready':report['ready'],'files':[{k:f[k] for k in ['path','ready','n_records','n_accepted','blocking_reasons']} for f in files]},ensure_ascii=False))
    raise SystemExit(0 if report['ready'] else 2)

if __name__=='__main__':main()
