"""Frozen final MGCA case study. Numerical failures stop; no performance gate."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import common_v10 as c

HERE=Path(__file__).resolve().parent
MODEL=c.V10_MODEL
PHASES=('plan','preflight','train','four_target','k4dd','factor_xa','dpp4','plots','structure','summary')

def verify_release():
    manifest=c.read_json(HERE/'release_manifest.json')
    for name,h in manifest['files'].items():
        # This file contains deployment paths/scheduling only; scientific values
        # are validated against the immutable JSON and model snapshots below.
        if name=='experiment_config.sh':continue
        p=HERE/name
        if not p.is_file() or c.sha256_file(p)!=h: raise RuntimeError('Frozen package changed: '+name)
    c.validate_frozen_config(c.read_json(c.FROZEN_PATH))
    lock=c.read_json(HERE/'frozen/experiment_lock.json')
    if lock['ablation_runs']!=5 or lock['repeat_hpo'] or lock['repeat_benchmark']:
        raise RuntimeError('Benchmark/ablation freeze policy changed')
    return lock

@contextmanager
def lock_output(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        if os.name=='nt':
            import msvcrt
            f.seek(0);f.write(b'0');f.flush();f.seek(0)
            try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:raise RuntimeError('Output directory is already locked')
        else:
            import fcntl
            try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError:raise RuntimeError('Output directory is already locked')
        try:yield
        finally:
            if os.name=='nt':f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def call(out,label,script,args,python=None):
    logs=out/'logs';logs.mkdir(parents=True,exist_ok=True)
    attempt=logs/(label+'_'+str(time.time_ns()))
    cmd=[python or sys.executable,'-u',str(HERE/script)]+list(map(str,args))
    started=time.time()
    print('START',label,flush=True)
    with attempt.with_suffix('.log').open('w',encoding='utf-8') as log:
        proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=HERE)
    record={'command':cmd,'start_unix':started,'end_unix':time.time(),'exit_code':proc.returncode,
            'log':str(attempt.with_suffix('.log'))}
    c.atomic_json(attempt.with_suffix('.json'),record)
    if proc.returncode:
        print(attempt.with_suffix('.log').read_text(encoding='utf-8',errors='replace')[-5000:],flush=True)
        raise RuntimeError(f'{label} failed; see {attempt.with_suffix(".log")}')
    print('DONE',label,flush=True)

def seal_stage(out,name,folder):
    files={p.relative_to(out).as_posix():c.sha256_file(p) for p in folder.rglob('*')
           if p.is_file() and 'cache' not in p.relative_to(folder).parts and p.name!='last_state.pt'}
    if not files: raise RuntimeError('Empty stage: '+name)
    c.atomic_json(out/'stages'/f'{name}.json',{'package':c.sha256_file(HERE/'release_manifest.json'),'files':files})

def stage_complete(out,name):
    p=out/'stages'/f'{name}.json'
    if not p.exists(): return False
    d=c.read_json(p)
    if d['package']!=c.sha256_file(HERE/'release_manifest.json'):raise RuntimeError('Stage code identity changed')
    for name,h in d['files'].items():
        if not (out/name).is_file() or c.sha256_file(out/name)!=h:
            raise RuntimeError('Completed stage artifact missing/corrupt: '+name)
    return bool(d['files'])

def inputs(out):
    folder=out/'cohort'
    return folder/'training.csv',folder/'manifest.json'

def prepare(out):
    if stage_complete(out,'cohort'):return
    train,manifest=inputs(out)
    call(out,'cohort','case_study/prepare_full_training_2773.py',[
        '--train-csv',HERE/'inputs/train_run1.csv','--val-csv',HERE/'inputs/val_run1.csv',
        '--test-csv',HERE/'inputs/test_run1.csv','--reference-csv',HERE/'inputs/koff.csv',
        '--case-csv',HERE/'inputs/four_target.csv','--output-csv',train,
        '--manifest',manifest,'--exclusions-csv',out/'cohort/exclusions.csv'])
    seal_stage(out,'cohort',out/'cohort')

def preflight(out,esm,device):
    prepare(out)
    # Catch plotting prerequisites before spending time on five GPU refits.
    import matplotlib
    import PIL
    if importlib.util.find_spec('gemmi') is None and importlib.util.find_spec('Bio') is None:
        raise RuntimeError('Structure analysis requires gemmi or biopython; install before training')
    if shutil.disk_usage(out).free/2**30<float(os.environ.get('MIN_FREE_GIB','5')):
        raise RuntimeError('Insufficient free space on OUTPUT_ROOT filesystem')
    expected=c.read_json(HERE/'frozen/esm_identity.json')
    print('Validating pinned ESM2 asset hashes...',flush=True)
    for name,h in expected.items():
        if not (esm/name).is_file() or c.sha256_file(esm/name)!=h:raise RuntimeError('ESM2 identity mismatch: '+name)
    c.atomic_json(out/'esm_identity.json',expected)
    import torch
    if device.startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
        torch.cuda.set_device(torch.device(device));torch.cuda.init()
    train,manifest=inputs(out)
    call(out,'preflight','preflight_case_study_v10.py',[
        '--data-csv',train,'--data-manifest',manifest,'--config-json',c.FROZEN_PATH,
        '--esm2-path',esm,'--four-target-csv',HERE/'inputs/four_target.csv','--k4dd-csv',HERE/'inputs/k4dd.csv',
        '--reference-csv',HERE/'inputs/koff.csv','--train-split',HERE/'inputs/train_run1.csv',
        '--val-split',HERE/'inputs/val_run1.csv','--test-split',HERE/'inputs/test_run1.csv',
        '--output',out/'preflight.json'])
    import platform
    versions={name:__import__(name).__version__ for name in ['numpy','scipy','pandas','rdkit','transformers','sklearn','matplotlib','PIL']}
    c.atomic_json(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
        'cuda':torch.version.cuda,'device':device,'gpu':torch.cuda.get_device_name(torch.device(device)) if device.startswith('cuda') else None,
        'versions':versions,'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),
        'MKL_NUM_THREADS':os.environ.get('MKL_NUM_THREADS'),'PYTHONHASHSEED':os.environ.get('PYTHONHASHSEED')})

def train_all(out,esm,device,jobs):
    if stage_complete(out,'checkpoints'):return
    train,manifest=inputs(out)
    base=['--data-csv',train,'--data-manifest',manifest,'--config-json',c.FROZEN_PATH,'--esm2-path',esm,'--device',device]
    call(out,'cache','precompute_full_features_v10.py',base+['--output',out/'full_feature_cache_audit.json'])
    def one(seed):
        call(out,'train_seed_'+str(seed),'train_full_refit_v10.py',base+[
            '--output-dir',out/'checkpoints'/f'seed_{seed}','--seed',seed,'--epochs',c.DEFAULT_EPOCHS,'--concurrency',jobs])
    errors=[]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for future in as_completed([pool.submit(one,s) for s in c.DEFAULT_SEEDS]):
            try:future.result()
            except Exception as exc:errors.append(str(exc))
    if errors:raise RuntimeError('\n'.join(errors))
    c.checkpoint_records(out/'checkpoints')
    seal_stage(out,'checkpoints',out/'checkpoints')

def panel(out,esm,device,name):
    if stage_complete(out,name):return
    c.checkpoint_records(out/'checkpoints')
    call(out,name,'infer_case_panel_v10.py',[
        '--case-csv',HERE/'inputs'/f'{name}.csv','--training-csv',inputs(out)[0],
        '--checkpoint-root',out/'checkpoints','--esm2-path',esm,'--output-dir',out/'predictions'/name,
        '--panel-name',name,'--device',device,'--batch-size',64,'--esm-batch-size',1,'--check-batch-invariance'])
    seal_stage(out,name,out/'predictions'/name)

def occlude(out,esm,device,name):
    if stage_complete(out,name):return
    panelname,uniprot=('k4dd','P00742') if name=='factor_xa' else ('four_target','P27487|Q53TN1')
    if not stage_complete(out,panelname):raise RuntimeError('Run panel first: '+panelname)
    args=['--model-file',MODEL,'--compound-id-prefix',name,'--dataset','2773',
        '--case-csv',HERE/'inputs'/f'{panelname}.csv','--checkpoint-root',out/'checkpoints',
        '--esm2-path',esm,'--output-dir',out/'occlusion'/name,'--target-uniprot',uniprot,
        '--seeds',*c.DEFAULT_SEEDS,'--epochs',c.DEFAULT_EPOCHS,'--expected-config-id',c.EXPECTED_CONFIG_ID,
        '--device',device,'--batch-size',256,'--protein-window-size',16,'--protein-stride',4,
        '--sensitivity-window-sizes',8,32,'--protein-baseline','sequence_mean',
        '--include-zero-baseline-sensitivity','--top-features',10,'--stability-top-k',10,
        '--reference-predictions',out/'predictions'/panelname/'unique_compound_five_seed_predictions.csv']
    args+=['--representative-compound-id','factor_xa_05' if name=='factor_xa' else 'dpp4_12']
    call(out,name,'run_occlusion_v10.py',args)
    seal_stage(out,name,out/'occlusion'/name)

def plots(out):
    if stage_complete(out,'plots'):return
    for name in ['factor_xa','dpp4']:
        if not stage_complete(out,name):raise RuntimeError('Missing occlusion stage '+name)
        call(out,'plot_'+name,'plot_occlusion_headless.py',[
            '--stage2-dir',out/'occlusion'/name,'--output-dir',out/'figures'/name,'--label',name])
    call(out,'panel_plots','plot_case_panels.py',['--output-root',out])
    seal_stage(out,'plots',out/'figures')

def structure(out):
    if stage_complete(out,'structure'):return
    if not stage_complete(out,'factor_xa'):raise RuntimeError('Factor Xa occlusion missing')
    call(out,'structure','structure_analysis.py',['--output-root',out])
    seal_stage(out,'structure',out/'structure')

def summary(out):
    required=['cohort','checkpoints','four_target','k4dd','factor_xa','dpp4','plots','structure']
    for stage in required:
        if not stage_complete(out,stage):raise RuntimeError('Incomplete case study: '+stage)
    call(out,'summary','summarize_final.py',['--output-root',out])

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['all',*PHASES],default='all')
    args=parser.parse_args();frozen=verify_release()
    out=Path(os.environ.get('OUTPUT_ROOT',HERE.parent/'case_study_outputs/mgca_final_unbounded_2773_v1')).resolve()
    if out in [HERE,HERE.parent] or out.name=='v10_2773' or HERE in out.parents:
        raise RuntimeError('OUTPUT_ROOT must be a new directory outside the frozen package')
    esm=Path(os.environ.get('ESM2_PATH',HERE.parent.parent/'pretrained_model/esm2_t36')).resolve()
    device=os.environ.get('DEVICE','cuda:0');jobs=int(os.environ.get('RUN_JOBS','5'))
    if jobs<1 or jobs>5:raise RuntimeError('RUN_JOBS must be 1..5')
    if os.environ.get('ESM_BATCH_SIZE','1')!='1':raise RuntimeError('Frozen ESM batch size is 1')
    out.mkdir(parents=True,exist_ok=True)
    with lock_output(out/'.workflow.lock'):
        identity={'protocol':c.PROTOCOL,'package':c.sha256_file(HERE/'release_manifest.json'),
            'refit_config':c.sha256_file(c.FROZEN_PATH),'esm_expected':c.read_json(HERE/'frozen/esm_identity.json')}
        p=out/'case_plan.json'
        if p.exists() and c.read_json(p)!=identity:raise RuntimeError('Output belongs to a different frozen protocol')
        if not p.exists():c.atomic_json(p,identity)
        if args.phase=='plan':
            prepare(out)
            print('PLAN verified; 5 checkpoints x 33 epochs; no tuning or benchmark/ablation training.');return
        phases=list(PHASES[1:]) if args.phase=='all' else [args.phase]
        if args.phase in ['four_target','k4dd','factor_xa','dpp4']:
            preflight(out,esm,device)
        for phase in phases:
            if phase=='preflight':preflight(out,esm,device)
            elif phase=='train':
                if args.phase!='all':preflight(out,esm,device)
                train_all(out,esm,device,jobs)
            elif phase in ['four_target','k4dd']:panel(out,esm,device,phase)
            elif phase in ['factor_xa','dpp4']:occlude(out,esm,device,phase)
            elif phase=='plots':plots(out)
            elif phase=='structure':structure(out)
            elif phase=='summary':summary(out)
        print('Completed requested phase:',args.phase,'Output:',out,flush=True)

if __name__=='__main__':main()
