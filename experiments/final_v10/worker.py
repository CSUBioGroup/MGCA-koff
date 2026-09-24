"""Isolated CUDA lifecycle, content-addressed cache, scientific audits and seals."""
import argparse
import csv
import inspect
import itertools
import math
import os
from pathlib import Path
import sys
import time
import traceback
from common import (HERE, RUNTIME, atomic_json, digest, file_lock, read_json, uid, verify_files)
from workflow import VARIANTS


def load_runtime():
    os.environ.setdefault('MPLBACKEND','Agg')
    sys.path.insert(0,str(RUNTIME))
    import trainer
    import torch
    if 'weights_only' in inspect.signature(torch.load).parameters:
        original=torch.load
        def trusted(*a,**kw):
            kw.setdefault('weights_only',False) # only local, hash-checked checkpoints
            return original(*a,**kw)
        torch.load=trusted
    return trainer


def build_args(plan,t,concurrency):
    cfg=plan['scientific']['settings']; a=cfg['architecture']
    values=dict(train_csv=t['train_csv'],val_csv=t['val_csv'],output_dir=t['output_dir'],
        dataset=t['dataset'],protocol=t['protocol'],run=t['run'],seed=t['seed'],
        ablation=VARIANTS[t['variant']],esm2_path=cfg['esm2_path'],device=cfg['device'],
        **t['params'],window_layout='even_span_v2',epochs=cfg['epochs'],patience=cfg['patience'],
        joint_rank=a['joint_rank'],protein_aux_weight=a['protein_aux_weight'],
        drug_utility_weight=a['drug_utility_weight'],joint_utility_weight=a['joint_utility_weight'],
        branch_margin=a['branch_margin'],joint_branch_dropout=a['joint_branch_dropout'],
        protein_warmup_epochs=cfg['protein_warmup_epochs'],fixed_shrinkage_epochs=cfg['fixed_shrinkage_epochs'],
        save_best_model='true',save_resume_state='true',save_cgrs_outputs='true',amp=cfg['amp'],
        concurrency=concurrency,config_id=t['task_id'])
    args=[]
    for k,v in values.items(): args += ['--'+k.replace('_','-'),str(v)]
    if t['selection_only']: args.append('--selection-only')
    else: args += ['--test-csv',t['test_csv']]
    return args


def audit_pairs(trainer,entries,formal=False):
    from rdkit import Chem
    reports=[]; seen=set()
    for t in entries:
        paths=[t['train_csv'],t['val_csv']]
        if formal: paths.append(t['test_csv'])
        key=tuple(paths)
        if key in seen: continue
        seen.add(key); sets=[]
        for path in paths:
            rows=trainer.mgca.read_labeled_rows(path)
            if len(rows)<2 or not all(seq and smi and math.isfinite(y) for seq,smi,y in rows):
                raise RuntimeError('Invalid labeled split '+path)
            canonical=[]
            for seq,smi,_ in rows:
                mol=Chem.MolFromSmiles(smi)
                if mol is None: raise RuntimeError('Invalid molecule '+path)
                canonical.append((seq,Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)))
            sets.append(dict(raw={(s,d) for s,d,_ in rows},canonical=set(canonical),
                protein={s for s,d in canonical},drug={d for s,d in canonical}))
        for i,j in itertools.combinations(range(len(paths)),2):
            a,b=sets[i],sets[j]
            r=dict(dataset=t['dataset'],protocol=t['protocol'],fold=t['fold'],
                split_i=paths[i],split_j=paths[j],test_accessed=formal,
                raw_overlap=len(a['raw']&b['raw']),canonical_overlap=len(a['canonical']&b['canonical']),
                protein_overlap=len(a['protein']&b['protein']),drug_overlap=len(a['drug']&b['drug']))
            if r['raw_overlap']: raise RuntimeError('Raw-pair overlap: '+str(r))
            if t['protocol']=='drug_cold' and r['drug_overlap']: raise RuntimeError('Drug-cold entity overlap: '+str(r))
            if t['protocol']=='protein_cold' and r['protein_overlap']: raise RuntimeError('Protein-cold sequence overlap: '+str(r))
            if t['protocol']=='warm' and r['canonical_overlap']:
                r['warning']='Existing canonical-equivalent warm pairs retained, not strict canonical-pair-disjoint evaluation.'
                print('WARNING',r,flush=True)
            reports.append(r)
    return reports


def pinned_json(path,payload):
    if path.exists() and read_json(path)!=payload:
        raise RuntimeError('Immutable runtime identity changed: '+str(path))
    if not path.exists(): atomic_json(path,payload)


def preflight(plan,tr):
    import torch
    cfg=plan['scientific']['settings']; root=Path(cfg['output_root'])
    verify_files(plan['scientific']['inputs'])
    esm=Path(cfg['esm2_path'])
    if not (esm/'config.json').is_file(): raise RuntimeError('ESM2 config missing: '+str(esm))
    assets=sorted(p for p in esm.iterdir() if p.is_file() and p.suffix in ('.json','.txt','.bin','.safetensors','.model'))
    if not any(p.suffix in ('.bin','.safetensors') for p in assets): raise RuntimeError('ESM2 weights missing')
    print('Hashing ESM2 assets...',flush=True)
    pinned_json(root/'esm_identity.json',{str(p):digest(p) for p in assets})
    snapshot=tr.environment_snapshot(torch)
    pinned_json(root/'runtime_identity.json',{k:snapshot[k] for k in ('python','torch','cuda_runtime','cudnn',
        'rdkit','transformers','numpy','scipy','sklearn','optuna')})
    reports=audit_pairs(tr,plan['scientific']['data_pairs'])
    counts={}
    device=torch.device(cfg['device'])
    for variant,ablation in VARIANTS.items():
        torch.manual_seed(42)
        model=tr.mgca.FullRegressionTransformer(ablation=ablation).to(device).eval()
        with torch.no_grad():
            y,aux=model(torch.randn(2,4,2560,device=device),torch.randn(2,4,2048,device=device))
            if not torch.isfinite(y).all(): raise RuntimeError('Model smoke nonfinite')
            if not math.isclose(float(aux['drug_gate'][0,0]),.10,rel_tol=1e-5): raise RuntimeError('Gate init')
        counts[variant]=tr.mgca.parameter_count(model); del model
    if len(set(counts.values()))!=1: raise RuntimeError('Ablation parameter mismatch')
    atomic_json(root/'preflight.json',dict(plan_id=plan['plan_id'],ok=True,counts=counts,
        audit=reports,environment=snapshot,test_accessed=False,
        homology_note='Exact-sequence audit only, not a homology-cluster guarantee'))
    print('Preflight complete: no test input opened.',flush=True)


def cache_key(plan,t):
    cfg=plan['scientific']['settings']; root=Path(cfg['output_root'])
    return uid(dict(inputs=t['input_sha256'],window_size=t['params']['window_size'],
        layout='even_span_v2',esm=read_json(root/'esm_identity.json'),
        legacy=digest(HERE/'runtime/local/ESM_Morgan_Hybrid_Fusion.py'),esm_batch_size=cfg['esm_batch_size']))


def install_cache(plan,t,tr,mode):
    import torch
    cfg=plan['scientific']['settings']; key=cache_key(plan,t)
    folder=Path(cfg['cache_root'])/key; folder.mkdir(parents=True,exist_ok=True)
    feature=tr.mgca.legacy._esm_cache_path_for_window(str(folder/'features.pt'),t['params']['window_size'],'even_span_v2')
    tr.mgca.get_combined_esm_cache_path=lambda *a,**kw:feature
    original=tr.prepare_data
    def prepare(args,device):
        if args.selection_only!=t['selection_only'] or (t['selection_only'] and args.test_csv is not None):
            raise RuntimeError('Test firewall violation')
        with file_lock(folder/'.cache.lock',blocking=True):
            marker=folder/'verified.json'
            if marker.exists():
                seal=read_json(marker)
                if seal['key']!=key: raise RuntimeError('Cache key mismatch')
                verify_files(seal['files'],folder)
            elif mode!='cache': raise RuntimeError('Cache not prepared; use the workflow')
            else:
                # Preserve any interrupted/unsealed tensors in this exact cache directory.
                for p in folder.glob('*.pt'):
                    p.rename(p.with_name(p.name+'.unsealed.'+str(time.time_ns())))
            data,rows,paths=original(args,device)
            for name,width in (('esm2',2560),('morgan',2048)):
                value=torch.load(paths[name],map_location='cpu')
                if tuple(value.shape)!=(sum(map(len,rows)),4,width) or not torch.isfinite(value).all():
                    raise RuntimeError('Invalid cache tensor '+name)
            pinned_json(marker,dict(key=key,files={Path(paths[k]).name:digest(paths[k]) for k in ('esm2','morgan')}))
            return data,rows,paths
    tr.prepare_data=prepare


def seal_run(plan,t,tr):
    import numpy as np
    out=Path(t['output_dir']); m=read_json(out/'metrics.json'); cfg=plan['scientific']['settings']
    if m['selection_only']!=t['selection_only'] or m['test_accessed']==t['selection_only'] or m['identity']['config_id']!=t['task_id']:
        raise RuntimeError('Result identity/test policy failure')
    if not cfg['protein_warmup_epochs']+cfg['fixed_shrinkage_epochs'] < m['best_epoch'] <= m['epochs_ran']:
        raise RuntimeError('Ineligible best epoch')
    required=['metrics.json','identity.json','environment.json','history.csv','best_model.pt','.complete']
    splits=[('train',t['train_csv'],'train_metrics'),('validation',t['val_csv'],'val_metrics')]
    if not t['selection_only']: splits.append(('test',t['test_csv'],'test_metrics'))
    for label,source,metric_key in splits:
        file=out/(label+'_predictions.csv'); required.append(file.name)
        with file.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
        source_rows=tr.mgca.read_labeled_rows(source)
        if len(rows)!=len(source_rows): raise RuntimeError('Prediction coverage mismatch')
        for i,(r,s) in enumerate(zip(rows,source_rows)):
            if int(r['source_row'])!=i or r['sample_id']!=tr.sample_id(s): raise RuntimeError('Sample ID mismatch')
            if not math.isclose(float(r['y_true']),s[2],abs_tol=1e-6,rel_tol=1e-6): raise RuntimeError('Label mismatch')
            if not all(math.isfinite(float(r[k])) for k in ('y_true','y_pred','error','abs_error')): raise RuntimeError('Nonfinite prediction')
        y=np.array([float(r['y_true']) for r in rows]); p=np.array([float(r['y_pred']) for r in rows])
        recalculated=tr.regression_metrics(y,p)
        for name in ('mse','rmse','mae','r2','pearson','spearman'):
            a,b=recalculated[name],m[metric_key][name]
            if a is None or b is None:
                if a is not b: raise RuntimeError('Undefined metric mismatch')
            elif not math.isclose(a,b,abs_tol=3e-6,rel_tol=3e-5): raise RuntimeError('Metric differs from predictions: '+name)
    archive='validation_cgrs_outputs.npz' if t['selection_only'] else 'cgrs_outputs.npz'
    tr.validate_cgrs_archive(out/archive,len(tr.mgca.read_labeled_rows(t['val_csv'] if t['selection_only'] else t['test_csv'])))
    required.append(archive)
    atomic_json(out/'verified.complete.json',dict(task_id=t['task_id'],verified_unix=time.time(),
        files={name:digest(out/name) for name in required}))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--mode',choices=['preflight','formal-audit','cache','train'],required=True)
    p.add_argument('--task',type=Path); p.add_argument('--concurrency',type=int,default=1)
    args=p.parse_args(); plan=read_json(args.plan); cfg=plan['scientific']['settings']; root=Path(cfg['output_root'])
    if uid(plan['scientific'])!=plan['plan_id']: raise RuntimeError('Corrupt plan')
    verify_files(plan['scientific']['code'],HERE)
    tr=load_runtime()
    import torch
    device=torch.device(cfg['device'])
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA not available')
        torch.cuda.set_device(device); torch.cuda.init()
    if args.mode=='preflight': preflight(plan,tr); return
    if read_json(root/'preflight.json')['plan_id']!=plan['plan_id']: raise RuntimeError('Run preflight first')
    if args.mode=='formal-audit':
        entries=[]
        for name in ('formal_manifest.json','ablation_manifest.json'):
            if (root/name).exists():
                manifest=read_json(root/name)
                if manifest['plan_id']!=plan['plan_id']: raise RuntimeError('Formal manifest changed')
                for dataset,h in manifest['frozen_sha256'].items():
                    if digest(root/'frozen'/dataset/'best_params.json')!=h: raise RuntimeError('Frozen params changed')
                entries+=manifest['tasks']
        if not entries: raise RuntimeError('No frozen formal manifest')
        for t in entries: verify_files(t['input_sha256'])
        atomic_json(root/'formal_split_audit.json',dict(plan_id=plan['plan_id'],reports=audit_pairs(tr,entries,True)))
        return
    t=read_json(args.task)
    if t['plan_id']!=plan['plan_id']: raise RuntimeError('Task plan mismatch')
    identity={k:v for k,v in t.items() if k not in ('task_id','output_dir')}
    if uid(identity)!=t['task_id']: raise RuntimeError('Task altered')
    if not t['selection_only']:
        frozen=root/'frozen'/t['dataset']/'best_params.json'
        freeze=read_json(frozen)
        if freeze['plan_id']!=plan['plan_id'] or freeze['selection']['params']!=t['params']:
            raise RuntimeError('Formal worker requires the frozen Full hyperparameters')
        manifest_path=root/('formal_manifest.json' if t['variant']=='full' else 'ablation_manifest.json')
        manifest=read_json(manifest_path)
        if manifest['plan_id']!=plan['plan_id'] or manifest['frozen_sha256'][t['dataset']]!=digest(frozen):
            raise RuntimeError('Formal manifest/frozen hash mismatch')
        if t not in manifest['tasks']: raise RuntimeError('Unregistered formal task')
    verify_files(t['input_sha256'])
    install_cache(plan,t,tr,args.mode)
    sys.argv=[str(RUNTIME/'trainer.py')]+build_args(plan,t,args.concurrency)
    if args.mode=='cache': tr.prepare_data(tr.parse_args(),device); return
    out=Path(t['output_dir']); out.mkdir(parents=True,exist_ok=True)
    with file_lock(out/'.worker.lock'):
        if (out/'identity.json').exists() and read_json(out/'identity.json')['config_id']!=t['task_id']:
            raise RuntimeError('Refusing to overwrite a different run')
        event=dict(task_id=t['task_id'],started_unix=time.time(),command=sys.argv,pid=os.getpid(),
            configured_concurrency=args.concurrency,
            environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','ESM_BATCH_SIZE','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONHASHSEED')})
        attempt=out/('attempt_%d.json'%time.time_ns()); atomic_json(attempt,event)
        try:
            tr.main(); seal_run(plan,t,tr); event['exit_code']=0
        except BaseException:
            event['exit_code']=1; event['failure']=traceback.format_exc(); raise
        finally:
            event['ended_unix']=time.time(); event['duration_sec']=event['ended_unix']-event['started_unix']
            atomic_json(attempt,event)


if __name__=='__main__': main()
