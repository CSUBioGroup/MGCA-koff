"""One-command validation-only sensitivity study. No ranking-based promotion/gates."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from common import (HERE, atomic_json, configurations, digest, file_lock, read_json,
                    scientific_sources, split_paths, uid, verify_run, write_csv)


def env_bool(name):
    value = os.environ.get(name, 'false').lower()
    if value not in ('true', 'false', '1', '0'):
        raise ValueError(name + ' must be true/false')
    return value in ('true', '1')


def settings():
    return dict(project_root=str(Path(os.environ.get('PROJECT_ROOT', str(HERE.parents[1]))).resolve()),
                esm2_path=str(Path(os.environ.get('ESM2_PATH', str(HERE.parents[2]/'pretrained_model/esm2_t36'))).resolve()),
                datasets=os.environ.get('DATASETS', '2773 KinetX').split(),
                protocols=os.environ.get('PROTOCOLS', 'warm drug_cold protein_cold').split(),
                n_runs=int(os.environ.get('N_RUNS', '5')), seed=int(os.environ.get('SEED', '42')),
                seed_step=int(os.environ.get('SEED_STEP', '100')),
                seed_policy=os.environ.get('SEED_POLICY', 'benchmark'),
                cap_multipliers=list(map(float, os.environ.get('CAP_MULTIPLIERS', '.5 1 2').split())),
                init_multipliers=list(map(float, os.environ.get('INIT_MULTIPLIERS', '.5 1 2').split())),
                equal_caps=env_bool('INCLUDE_EQUAL_CAPS'), amp=env_bool('AMP'),
                device=os.environ.get('DEVICE', 'cuda:0'),
                runtime_env={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONHASHSEED', 'PYTORCH_CUDA_ALLOC_CONF')})


def make_plan(cfg, out, cache):
    if not cfg['datasets'] or len(set(cfg['datasets'])) != len(cfg['datasets']) or not set(cfg['datasets']) <= {'2773', 'KinetX'}:
        raise ValueError('Invalid/duplicate datasets')
    if not cfg['protocols'] or len(set(cfg['protocols'])) != len(cfg['protocols']) or not set(cfg['protocols']) <= {'warm', 'drug_cold', 'protein_cold'}:
        raise ValueError('Invalid/duplicate protocols')
    if cfg['seed_policy'] not in ('benchmark', 'paired_seeds') or not 1 <= cfg['n_runs'] <= 100:
        raise ValueError('Invalid seed policy/n_runs')
    if cfg['seed_policy'] == 'benchmark' and cfg['n_runs'] > 5:
        raise ValueError('benchmark >5 would duplicate training; use SEED_POLICY=paired_seeds')
    configs = configurations(cfg['cap_multipliers'], cfg['init_multipliers'], cfg['equal_caps'])
    inputs, frozen, tasks = {}, {}, []
    for dataset in cfg['datasets']:
        frozen[dataset] = read_json(HERE/'frozen'/dataset/'best_params.json')
        f = frozen[dataset]
        if f['dataset'] != dataset or f['test_accessed_during_selection'] is not False:
            raise RuntimeError('Invalid frozen configuration')
        model = HERE/'runtime/mgca_hyperparameter_tuning/v10_cgrs/ESM_Morgan_Hybrid_Fusion_cgrs_v10.py'
        if digest(model) != f['model_sha256']:
            raise RuntimeError('Frozen v10 model changed')
        for c in configs:
            for protocol in cfg['protocols']:
                for run in range(1, cfg['n_runs'] + 1):
                    fold = 1 if protocol == 'protein_cold' else 1+(run-1)%5
                    seed = cfg['seed'] + (cfg['seed_step']*(run-1) if protocol == 'protein_cold' or cfg['seed_policy'] == 'paired_seeds' else 0)
                    train, val = split_paths(cfg['project_root'], dataset, protocol, fold)
                    for p in (train, val):
                        if str(p) not in inputs:
                            inputs[str(p)] = digest(p)
                    task = dict(dataset=dataset, protocol=protocol, run=run, fold=fold,
                                seed=seed, config=c, train_csv=str(train), val_csv=str(val))
                    task['relative_dir'] = '%s/%s/%s/run%d'%(dataset, protocol, c['config_id'], run)
                    tasks.append(task)
    scientific = dict(version='v10_sensitivity_validation_only_v1', settings=cfg,
                      output_root=str(out), cache_root=str(cache), configurations=configs,
                      frozen=frozen, inputs=inputs, code=scientific_sources(),
                      protocol_notes=['No test files opened, no automatic best-configuration selection.',
                                      'caps sweep keeps original normalized gate prior; cap changes also change its effective strength.',
                                      'Historical test results informed v10; this is retrospective validation sensitivity, not an untouched test.',
                                      'Early-stopped validation is the selection metric, not an unbiased performance estimate.'])
    plan_id = uid(scientific)
    for t in tasks:
        t['task_id'] = uid([plan_id, t])
    return dict(plan_id=plan_id, scientific=scientific, tasks=tasks)


def build_args(plan, task, concurrency):
    s = plan['scientific']; cfg = s['settings']; f = s['frozen'][task['dataset']]
    options = dict(f['params'])
    allowed = ('hidden_dim', 'joint_rank', 'protein_aux_weight', 'drug_utility_weight',
               'joint_utility_weight', 'gate_prior_weight', 'branch_margin', 'drug_gate_init',
               'drug_gate_cap', 'joint_gate_init', 'joint_gate_cap', 'joint_branch_dropout',
               'protein_warmup_epochs', 'fixed_shrinkage_epochs')
    options.update({k: f['architecture'][k] for k in allowed})
    options.update({k: task['config'][k] for k in ('drug_gate_cap','joint_gate_cap','drug_gate_init','joint_gate_init')})
    options.update(epochs=f['training']['epochs'], patience=f['training']['patience'],
                   train_csv=task['train_csv'], val_csv=task['val_csv'],
                   output_dir=str(Path(s['output_root'])/'runs'/task['relative_dir']),
                   dataset=task['dataset'], protocol=task['protocol'], run=task['run'], seed=task['seed'],
                   esm2_path=cfg['esm2_path'], device=cfg['device'], ablation='no',
                   config_id=task['task_id'], concurrency=concurrency, amp=str(cfg['amp']).lower(),
                   save_best_model='true', save_resume_state='true', save_cgrs_outputs='true')
    args = ['--selection-only']
    for k, v in options.items():
        args += ['--'+k.replace('_','-'), str(v)]
    assert '--test-csv' not in args
    return args


def disk_guard(out, cache):
    minimum = float(os.environ.get('MIN_FREE_GIB', '5'))
    if minimum < 0:
        raise ValueError('MIN_FREE_GIB must be >= 0')
    for p in (out, cache):
        free = shutil.disk_usage(str(p)).free/1024**3
        if free < minimum:
            raise RuntimeError('%s has %.2f GiB free; requires %.2f. Nothing deleted.'%(p, free, minimum))


def launch(plan_path, task, mode, jobs):
    plan = read_json(plan_path)
    root = Path(plan['scientific']['output_root'])
    dest = root/'runs'/task['relative_dir'] if mode == 'train' else root/'cache_logs'
    dest.mkdir(parents=True, exist_ok=True)
    if mode == 'train' and (dest/'verified.complete.json').exists():
        verify_run(dest, task['task_id'])
        print('verified skip:', task['relative_dir'], flush=True)
        return True
    cmd = [sys.executable, '-u', str(HERE/'worker.py'), '--plan', str(plan_path),
           '--task-id', task['task_id'], '--mode', mode, '--concurrency', str(jobs)]
    stamp = str(time.time_ns())
    log = dest/('%s_%s.log'%(mode, stamp))
    atomic_json(dest/('%s_%s.command.json'%(mode, stamp)), dict(command=cmd, started_unix=time.time(), task_id=task['task_id']))
    with log.open('w', encoding='utf-8') as f:
        code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    atomic_json(dest/('%s_%s.exit.json'%(mode, stamp)), dict(exit_code=code, ended_unix=time.time(), log=str(log)))
    print(('completed ' if code == 0 else 'FAILED ') + task['relative_dir'] + ' log=' + str(log), flush=True)
    return code == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('all','plan','preflight','cache','run','summary'), default='all')
    args = parser.parse_args()
    out = Path(os.environ.get('OUTPUT_ROOT', str(HERE/'outputs/sensitivity_v1'))).resolve()
    cache = Path(os.environ.get('CACHE_ROOT', str(out/'cache'))).resolve()
    out.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    jobs = int(os.environ.get('RUN_JOBS', '5'))
    if jobs < 1:
        raise ValueError('RUN_JOBS must be positive')
    with file_lock(out/'.workflow.lock'):
        plan = make_plan(settings(), out, cache)
        pp = out/'plan.json'
        if pp.exists():
            if read_json(pp) != plan:
                raise RuntimeError('Code/data/scientific settings changed. Use a NEW OUTPUT_ROOT; old results not reused.')
        else:
            atomic_json(pp, plan)
        write_csv(out/'configuration_grid.csv', [{**c, 'families': ';'.join(c['families'])} for c in plan['scientific']['configurations']])
        print('Frozen plan:', plan['plan_id'], 'configurations:', len(plan['scientific']['configurations']), 'runs:', len(plan['tasks']), flush=True)
        if args.phase == 'plan':
            return
        if args.phase != 'summary':
            disk_guard(out, cache)
            subprocess.run([sys.executable, '-u', str(HERE/'worker.py'), '--plan', str(pp), '--mode', 'preflight'], check=True)
        if args.phase == 'preflight':
            return
        if args.phase in ('all', 'cache', 'run'):
            # Precompute each train+val feature cache once, serially; no concurrent ESM extraction.
            unique = {}
            for t in plan['tasks']:
                unique.setdefault((t['train_csv'], t['val_csv']), t)
            for t in unique.values():
                disk_guard(out, cache)
                if not launch(pp, t, 'cache', 1):
                    raise RuntimeError('Feature preparation failed; see cache_logs')
        if args.phase == 'cache':
            return
        failures = []
        if args.phase in ('all', 'run'):
            groups = {}
            for t in plan['tasks']:
                groups.setdefault((t['dataset'], t['protocol'], t['config']['config_id']), []).append(t)
            for group in groups.values():
                disk_guard(out, cache)
                with ThreadPoolExecutor(max_workers=jobs) as pool:
                    futures = {pool.submit(launch, pp, t, 'train', jobs): t for t in group}
                    for future in as_completed(futures):
                        if not future.result():
                            failures.append(futures[future]['relative_dir'])
            # Execution failures are recorded, never used to choose a better setting.
        from summarize import summarize
        summarize(plan)
        if failures:
            raise RuntimeError('%d failed runs, preserved for resume. Lower RUN_JOBS if OOM; no hyperparameters were changed.'%len(failures))


if __name__ == '__main__':
    main()
