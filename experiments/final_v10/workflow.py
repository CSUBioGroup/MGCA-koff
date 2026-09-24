"""Immutable warm-only HPO, deterministic replay, isolated workers, formal evaluation.

SQLite stores completed Optuna trials; proposal JSON journals retain pending/failed
training. Replaying the same ask/tell sequence reconstructs the TPE RNG on resume.
No new trial replaces a failed trial and no performance gate aborts valid runs.
"""
import argparse
import concurrent.futures
import csv
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from common import (HERE, atomic_json, atomic_text, digest, file_lock, read_json,
                    scientific_sources, split_paths, uid, verify_files, write_csv)

VARIANTS = {'full': 'no', 'without_protein': 'protein_anchor',
            'without_drug': 'drug_correction', 'without_joint': 'joint_interaction'}
FLOAT_CHOICES = {'weight_decay': [0., 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                 'dropout': [.05, .1, .15, .17, .2, .25, .3]}


def settings():
    env = os.environ
    def positive(name, default):
        v = int(env.get(name, default))
        if v < 1: raise ValueError(name + ' must be positive')
        return v
    def choices(name, default):
        v = [float(x) for x in env.get(name, default).split()]
        if not v or len(set(v)) != len(v) or any(not math.isfinite(x) or x <= 0 for x in v):
            raise ValueError('Invalid initialization choices: ' + name)
        return v
    root = Path(env.get('PROJECT_ROOT', str(HERE.parents[1]))).resolve()
    out = Path(env.get('OUTPUT_ROOT', str(HERE/'outputs/unbounded_warm_hpo_v1'))).resolve()
    cfg = dict(project_root=str(root), output_root=str(out),
        cache_root=str(Path(env.get('CACHE_ROOT', str(out/'cache'))).resolve()),
        esm2_path=str(Path(env.get('ESM2_PATH', str(root.parent/'pretrained_model/esm2_t36'))).resolve()),
        device=env.get('DEVICE', 'cuda:0'), datasets=env.get('DATASETS', '2773 KinetX').split(),
        protocols=env.get('PROTOCOLS', 'warm drug_cold protein_cold').split(),
        ablation_datasets=env.get('ABLATION_DATASETS', '2773 KinetX').split(),
        trials=positive('TUNING_TRIALS', 30), top_k=positive('TOP_K', 5),
        epochs=positive('EPOCHS', 100), patience=positive('PATIENCE', 15),
        seed=int(env.get('SEED', 42)), seed_step=positive('SEED_STEP', 100),
        sampler_seed=int(env.get('SAMPLER_SEED', 2026)), n_runs=5,
        protein_warmup_epochs=int(env.get('PROTEIN_WARMUP_EPOCHS', 5)),
        fixed_shrinkage_epochs=int(env.get('FIXED_SHRINKAGE_EPOCHS', 5)),
        esm_batch_size=positive('ESM_BATCH_SIZE', 1), amp=env.get('AMP', 'false').lower(),
        drug_init_choices=choices('DRUG_INIT_CHOICES', '0.05 0.10 0.20'),
        joint_init_choices=choices('JOINT_INIT_CHOICES', '0.015 0.03 0.06'),
        objective='warm_run1_validation_mse_then_topk_five_split_mean',
        architecture=dict(protein_coefficient=1., positive_gate='softplus', gate_prior=None,
            scalar_weight_decay=0., hidden_dim=512, joint_rank=128, protein_aux_weight=.1,
            drug_utility_weight=.02, joint_utility_weight=.02, branch_margin=.01,
            joint_branch_dropout=.15, gradient_clip=5.),
        training_environment={k:env.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS',
            'PYTHONHASHSEED','CUDA_VISIBLE_DEVICES','PYTORCH_CUDA_ALLOC_CONF')})
    if int(env.get('OPTUNA_JOBS', 1)) != 1: raise ValueError('Optuna must be serial')
    if not cfg['datasets'] or not set(cfg['datasets']) <= {'2773','KinetX'}: raise ValueError('datasets')
    if not cfg['protocols'] or not set(cfg['protocols']) <= {'warm','drug_cold','protein_cold'}: raise ValueError('protocols')
    if not set(cfg['ablation_datasets']) <= set(cfg['datasets']): raise ValueError('ABLATION_DATASETS must be in DATASETS')
    if cfg['amp'] not in ('true','false'): raise ValueError('AMP must be true/false')
    if cfg['top_k'] > cfg['trials']: raise ValueError('TOP_K exceeds trial budget')
    if min(cfg['protein_warmup_epochs'],cfg['fixed_shrinkage_epochs']) < 0 or cfg['epochs'] <= cfg['protein_warmup_epochs']+cfg['fixed_shrinkage_epochs']:
        raise ValueError('Training must include learned-coefficient epochs')
    return cfg


def build_plan(cfg):
    pairs = []
    for dataset in cfg['datasets']:
        for protocol in sorted(set(cfg['protocols']) | {'warm'}):
            for fold in ([1] if protocol == 'protein_cold' else range(1,6)):
                train, val = split_paths(cfg['project_root'], dataset, protocol, fold)
                pairs.append(dict(dataset=dataset, protocol=protocol, fold=fold,
                                  train_csv=str(train), val_csv=str(val)))
    inputs = {p:digest(p) for t in pairs for p in (t['train_csv'],t['val_csv'])}
    scientific = dict(settings=cfg, code=scientific_sources(), inputs=inputs,
        search=dict(lr=[1e-5,5e-4], **FLOAT_CHOICES, batch_size=[16,32,64],
            window_size=[1,2,4,6,8], drug_gate_init=cfg['drug_init_choices'],
            joint_gate_init=cfg['joint_init_choices']), data_pairs=pairs,
        original_v10_sha256=digest(HERE/'sources/original_v10_model.py'),
        test_accessed=False)
    plan = dict(plan_id=uid(scientific), scientific=scientific)
    dest = Path(cfg['output_root'])/'plan.json'
    if dest.exists():
        if read_json(dest) != plan: raise RuntimeError('Code/data/config changed: use a NEW OUTPUT_ROOT')
    else: atomic_json(dest, plan)
    return plan


def test_path(cfg, dataset, protocol, fold):
    # Only called after best_params is frozen, never during tuning/preflight.
    train, _ = split_paths(cfg['project_root'], dataset, protocol, fold)
    return train.with_name(train.name.replace('train', 'test', 1))


def task(plan, dataset, protocol, run, params, variant='full', selection=True):
    cfg = plan['scientific']['settings']
    fold = 1 if protocol == 'protein_cold' else run
    seed = cfg['seed'] + (run-1)*cfg['seed_step'] if protocol == 'protein_cold' else cfg['seed']
    train, val = split_paths(cfg['project_root'], dataset, protocol, fold)
    paths = [str(train), str(val)]
    if not selection: paths.append(str(test_path(cfg,dataset,protocol,fold)))
    item = dict(plan_id=plan['plan_id'], dataset=dataset, protocol=protocol, run=run,
        fold=fold, seed=seed, params=params, variant=variant, selection_only=selection,
        train_csv=paths[0], val_csv=paths[1], test_csv=None if selection else paths[2],
        input_sha256={p:digest(p) for p in paths})
    for p in paths[:2]:
        if item['input_sha256'][p] != plan['scientific']['inputs'][p]: raise RuntimeError('Split changed: '+p)
    item['task_id'] = uid(item)
    group = Path('tuning')/dataset/'runs'/uid(params) if selection else Path('formal')/dataset/protocol/variant
    item['output_dir'] = str(Path(cfg['output_root'])/group/('run%d'%run))
    return item


def persist_task(plan, t):
    path = Path(plan['scientific']['settings']['output_root'])/'tasks'/(t['task_id']+'.json')
    if path.exists() and read_json(path) != t: raise RuntimeError('Task collision')
    if not path.exists(): atomic_json(path,t)
    return path


def verified(t):
    out = Path(t['output_dir']); seal = read_json(out/'verified.complete.json')
    if seal['task_id'] != t['task_id']: raise RuntimeError('Run identity changed')
    verify_files(seal['files'], out)
    m = read_json(out/'metrics.json')
    if m['identity']['config_id'] != t['task_id'] or m['selection_only'] != t['selection_only'] or m['test_accessed'] == t['selection_only']:
        raise RuntimeError('Invalid result identity/test policy')
    return m


def check_disk(cfg):
    root=Path(cfg['output_root']); root.mkdir(parents=True,exist_ok=True)
    required=float(os.environ.get('MIN_FREE_GIB',5))
    for p in (root,Path(cfg['cache_root'])):
        p.mkdir(parents=True,exist_ok=True)
        free=shutil.disk_usage(p).free/1024**3
        if free < required: raise RuntimeError('Free %.2f GiB < %.2f on %s; no automatic deletion'%(free,required,p))


def worker_call(plan, mode, t=None, concurrency=1):
    cfg=plan['scientific']['settings']; root=Path(cfg['output_root'])
    command=[sys.executable,'-u',str(HERE/'worker.py'),'--plan',str(root/'plan.json'),
             '--mode',mode,'--concurrency',str(concurrency)]
    if t: command += ['--task',str(persist_task(plan,t))]
    out=Path(t['output_dir']) if t else root
    out.mkdir(parents=True,exist_ok=True)
    log=out/('%s_%d.log'%(mode,time.time_ns()))
    print('%s: %s; log=%s'%(mode, t['task_id'][:12] if t else 'experiment',log),flush=True)
    env=dict(os.environ,ESM_BATCH_SIZE=str(cfg['esm_batch_size']),MPLBACKEND='Agg')
    with log.open('w',encoding='utf-8') as f:
        result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env)
    if result.returncode:
        tail='\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-65:])
        raise RuntimeError('%s failed; log=%s\n%s'%(mode,log,tail))


def launch_group(plan, tasks, jobs):
    if jobs < 1: raise ValueError('Concurrency must be positive')
    pending=[]
    for t in tasks:
        if (Path(t['output_dir'])/'verified.complete.json').exists(): verified(t)
        else: pending.append(t)
    cfg=plan['scientific']['settings']
    # ESM is extracted serially before any concurrent model training.
    for t in pending:
        check_disk(cfg); worker_call(plan,'cache',t)
    failures=[]
    def run_one(t):
        check_disk(cfg); worker_call(plan,'train',t,min(jobs,len(pending)))
        verified(t)
        print('completed %s/%s/%s/run%d seed%d'%(t['dataset'],t['protocol'],t['variant'],t['run'],t['seed']),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures={pool.submit(run_one,t):t for t in pending}
        for f in concurrent.futures.as_completed(futures):
            try: f.result()
            except Exception as exc:
                failures.append(str(exc)); print(str(exc),file=sys.stderr,flush=True)
    if failures: raise RuntimeError('Failed tasks retained for exact resume; no replacement trials:\n'+'\n'.join(failures))
    return [verified(t) for t in tasks]


def suggest(trial, search):
    p={'lr':trial.suggest_float('lr',*search['lr'],log=True)}
    for key in ('weight_decay','batch_size','dropout','window_size','drug_gate_init','joint_gate_init'):
        p[key]=trial.suggest_categorical(key,search[key])
    return p


def tune(plan,dataset):
    import optuna
    cfg=plan['scientific']['settings']; root=Path(cfg['output_root'])/'tuning'/dataset
    root.mkdir(parents=True,exist_ok=True)
    storage=optuna.storages.RDBStorage(url='sqlite:///'+(root/'study.sqlite3').as_posix(),
        engine_kwargs={'connect_args':{'timeout':60}})
    db=optuna.create_study(study_name='unbounded_'+dataset,storage=storage,direction='minimize',load_if_exists=True)
    if db.user_attrs.get('plan_id') not in (None,plan['plan_id']): raise RuntimeError('Study plan mismatch')
    if db.trials and db.user_attrs.get('plan_id') is None: raise RuntimeError('Unsigned study')
    db.set_user_attr('plan_id',plan['plan_id'])
    # Deterministically replay ALL past ask/tell operations, including duplicate
    # configurations. Recreating only the sampler seed on a populated DB is not equivalent.
    replay=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(
        seed=cfg['sampler_seed'],n_startup_trials=10))
    records=[]
    try:
        for number in range(cfg['trials']):
            trial=replay.ask(); params=suggest(trial,plan['scientific']['search'])
            proposal=dict(number=number,params=params,plan_id=plan['plan_id'])
            prop_path=root/'proposals'/('%04d.json'%number)
            if prop_path.exists() and read_json(prop_path)!=proposal: raise RuntimeError('TPE replay changed (check Optuna version)')
            if not prop_path.exists(): atomic_json(prop_path,proposal)
            t=task(plan,dataset,'warm',1,params)
            m=launch_group(plan,[t],1)[0]
            value=float(m['val_metrics']['mse'])
            replay.tell(trial,value)
            existing=db.get_trials(deepcopy=False)
            if len(existing)>number:
                old=existing[number]
                if old.state!=optuna.trial.TrialState.COMPLETE or old.params!=params or old.value!=value:
                    raise RuntimeError('SQLite trial disagrees with verified artifacts')
            else:
                db.add_trial(optuna.trial.create_trial(params=params,distributions=trial.distributions,
                    value=value,user_attrs={'output_dir':t['output_dir'],'task_id':t['task_id']}))
            records.append(dict(trial=number,val_mse=value,output_dir=t['output_dir'],**params))
            write_csv(root/'trials.csv',records)
        seen=set(); top=[]
        for r in sorted(records,key=lambda r:(r['val_mse'],r['trial'])):
            params={k:r[k] for k in ('lr','weight_decay','batch_size','dropout','window_size','drug_gate_init','joint_gate_init')}
            signature=uid(params)
            if signature in seen: continue
            seen.add(signature); top.append((r,params))
            if len(top)==cfg['top_k']: break
        if len(top)!=cfg['top_k']: raise RuntimeError('Insufficient unique configs; no automatic search expansion')
        reviews=[]
        for r,params in top:
            tasks=[task(plan,dataset,'warm',n,params) for n in range(1,6)]
            ms=launch_group(plan,tasks,int(os.environ.get('CANDIDATE_REVIEW_JOBS',5)))
            values=[m['val_metrics']['mse'] for m in ms]
            review=dict(trial=r['trial'],config_id=uid(params),params=params,
                val_mse_mean=statistics.mean(values),val_mse_sd=statistics.stdev(values),
                run_values=values,task_ids=[t['task_id'] for t in tasks])
            reviews.append(review)
            atomic_json(root/'top_k_candidates.json',reviews)
            write_csv(root/'top_k_candidates.csv',[dict(trial=x['trial'],config_id=x['config_id'],
                mean=x['val_mse_mean'],sd=x['val_mse_sd'],**{'run%d'%(i+1):v for i,v in enumerate(x['run_values'])},
                **x['params']) for x in reviews])
        winner=min(reviews,key=lambda x:(x['val_mse_mean'],x['val_mse_sd'],x['trial']))
        frozen=dict(plan_id=plan['plan_id'],dataset=dataset,selection=winner,test_accessed=False,
                    criterion=cfg['objective'])
        frozen['config_id']=uid(frozen)
        dest=Path(cfg['output_root'])/'frozen'/dataset/'best_params.json'
        if dest.exists() and read_json(dest)!=frozen: raise RuntimeError('Frozen params cannot be overwritten')
        if not dest.exists(): atomic_json(dest,frozen)
        atomic_text(dest.with_suffix('.sh'),'# Frozen; do not edit\n'+''.join(
            'export %s=%s\n'%(k.upper(),v) for k,v in winner['params'].items())+'export CONFIG_ID='+frozen['config_id']+'\n')
    finally:
        # Journal remains usable even if disk/OOM interrupts a pending trial.
        try:
            atomic_json(root/'progress.json',dict(plan_id=plan['plan_id'],completed=len(records),expected=cfg['trials']))
        finally:
            storage.remove_session()
            storage.engine.dispose()


def formal_tasks(plan, ablation=False):
    cfg=plan['scientific']['settings']; tasks=[]; frozen_hashes={}
    for dataset in cfg['datasets']:
        fpath=Path(cfg['output_root'])/'frozen'/dataset/'best_params.json'
        frozen=read_json(fpath)
        if frozen['plan_id']!=plan['plan_id'] or frozen['config_id']!=uid({k:v for k,v in frozen.items() if k!='config_id'}):
            raise RuntimeError('Frozen configuration mismatch')
        frozen_hashes[dataset]=digest(fpath)
        if ablation and dataset not in cfg['ablation_datasets']: continue
        for protocol in cfg['protocols']:
            for variant in (list(VARIANTS)[1:] if ablation else ['full']):
                for run in range(1,6):
                    tasks.append(task(plan,dataset,protocol,run,frozen['selection']['params'],variant,False))
    manifest=dict(plan_id=plan['plan_id'],frozen_sha256=frozen_hashes,tasks=tasks)
    path=Path(cfg['output_root'])/('ablation_manifest.json' if ablation else 'formal_manifest.json')
    if path.exists() and read_json(path)!=manifest: raise RuntimeError('Formal test data/config changed; refusing reuse')
    if not path.exists(): atomic_json(path,manifest)
    return tasks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['all','plan','preflight','tuning','formal','ablation','summary'],default=os.environ.get('PHASE','all'))
    args=parser.parse_args(); cfg=settings()
    with file_lock(Path(cfg['output_root'])/'.workflow.lock'):
        plan=build_plan(cfg)
        print('Plan:',plan['plan_id'],'phase=',args.phase,flush=True)
        if args.phase=='plan': return
        if args.phase=='summary':
            from summarize import summarize
            summarize(plan); return
        check_disk(cfg)
        worker_call(plan,'preflight')
        if args.phase=='preflight': return
        if args.phase in ('all','tuning'):
            for dataset in cfg['datasets']: tune(plan,dataset)
        for ablation,label in ((False,'formal'),(True,'ablation')):
            if args.phase not in ('all',label): continue
            tasks=formal_tasks(plan,ablation)
            if tasks: worker_call(plan,'formal-audit')
            keys=list(dict.fromkeys((t['dataset'],t['protocol'],t['variant']) for t in tasks))
            for key in keys:
                launch_group(plan,[t for t in tasks if (t['dataset'],t['protocol'],t['variant'])==key],int(os.environ.get('RUN_JOBS',5)))
        if args.phase in ('all','formal','ablation'):
            from summarize import summarize
            summarize(plan)


if __name__=='__main__': main()
