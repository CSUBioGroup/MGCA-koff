"""Isolated worker using an unchanged v10 model and a minimally hardened trainer copy."""
import argparse
import csv
import inspect
import math
import os
from pathlib import Path
import sys
import time
import traceback
from common import (HERE, RUNTIME, METRICS, atomic_json, digest, file_lock, read_json,
                    uid, verify_files)
from workflow import build_args


def load_runtime():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    sys.path.insert(0, str(RUNTIME))
    import train_v10_cgrs as trainer
    import torch
    # Explicitly trusted, locally generated checkpoints; supports torch 1.12 and 2.6+.
    if 'weights_only' in inspect.signature(torch.load).parameters:
        original_load = torch.load
        def trusted_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return original_load(*args, **kwargs)
        torch.load = trusted_load
    return trainer


def preflight(plan, trainer):
    import numpy as np
    import torch
    from rdkit import Chem
    s = plan['scientific']; root = Path(s['output_root']); cfg = s['settings']
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; CPU is not substituted automatically')
        torch.cuda.set_device(device); torch.cuda.init()
    esm = Path(cfg['esm2_path'])
    if not esm.is_dir() or not (esm/'config.json').is_file():
        raise FileNotFoundError('ESM2 directory/config.json missing: ' + str(esm))
    # Only model/tokenizer assets; no unrelated cache files in a recursive broad scan.
    assets = sorted(p for p in esm.iterdir() if p.is_file() and p.suffix in ('.json','.txt','.bin','.safetensors','.model'))
    if not any(p.suffix in ('.bin','.safetensors') for p in assets):
        raise FileNotFoundError('No ESM2 weights in ' + str(esm))
    print('Hashing ESM2 assets (read-only)...', flush=True)
    esm_identity = {str(p): digest(p) for p in assets}
    identity_path = root/'esm_identity.json'
    if identity_path.exists() and read_json(identity_path) != esm_identity:
        raise RuntimeError('ESM2 contents changed; use a new experiment output root')
    atomic_json(identity_path, esm_identity)
    counts, audit, seen = [], [], set()
    for t in plan['tasks']:
        pair = (t['train_csv'], t['val_csv'])
        if pair in seen:
            continue
        seen.add(pair)
        entries = []
        for p in pair:
            rows = trainer.mgca.read_labeled_rows(p)
            if len(rows) < 2 or not all(seq and smi and math.isfinite(y) for seq,smi,y in rows):
                raise ValueError('Invalid/empty data: ' + p)
            canonical = []
            for seq, smi, _ in rows:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError('Invalid SMILES: ' + p)
                canonical.append((seq, Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)))
            entries.append(dict(raw={(seq,smi) for seq,smi,_ in rows}, pairs=set(canonical),
                                protein={seq for seq,_ in canonical}, drug={smi for _,smi in canonical}, n=len(rows)))
        a,b = entries
        r = dict(dataset=t['dataset'], protocol=t['protocol'], fold=t['fold'],
                 train_rows=a['n'], val_rows=b['n'], raw_pair_overlap=len(a['raw'] & b['raw']),
                 canonical_pair_overlap=len(a['pairs'] & b['pairs']),
                 protein_overlap=len(a['protein'] & b['protein']), drug_overlap=len(a['drug'] & b['drug']), test_accessed=False)
        if r['raw_pair_overlap']:
            raise RuntimeError('Train/validation raw-pair leakage: ' + str(r))
        if t['protocol'] == 'drug_cold' and r['drug_overlap']:
            raise RuntimeError('Drug-cold train/validation entity leakage: ' + str(r))
        if t['protocol'] == 'protein_cold' and r['protein_overlap']:
            raise RuntimeError('Protein-cold exact-sequence leakage: ' + str(r))
        r['warning'] = 'canonical-equivalent warm pairs retained; not strict canonical pair disjoint' if t['protocol']=='warm' and r['canonical_pair_overlap'] else None
        audit.append(r)
    for c in s['configurations']:
        torch.manual_seed(42)
        kwargs = {k:c[k] for k in ('drug_gate_cap','joint_gate_cap','drug_gate_init','joint_gate_init')}
        model = trainer.mgca.FullRegressionTransformer(**kwargs).to(device)
        model.set_corrections_enabled(True); model.set_shrinkage_learnable(True); model.eval()
        with torch.no_grad():
            y, aux = model(torch.randn(2,4,2560,device=device), torch.randn(2,4,2048,device=device))
        if not torch.isfinite(y).all():
            raise RuntimeError('Non-finite model smoke test')
        for name in ('drug', 'joint'):
            actual = float(aux[name+'_gate'][0,0])
            if not np.isclose(actual, c[name+'_gate_init'], rtol=1e-5, atol=1e-7):
                raise RuntimeError('Gate initialization mismatch')
        counts.append(trainer.mgca.parameter_count(model))
        del model
    if len(set(counts)) != 1:
        raise RuntimeError('Parameter counts differ across sensitivity settings')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    atomic_json(root/'preflight.json', dict(plan_id=plan['plan_id'], ok=True, parameter_count=counts[0],
               split_audit=audit, environment=trainer.environment_snapshot(torch), test_accessed=False,
               protein_audit_scope='exact sequences only; does not independently establish homology-disjoint splits'))
    print('Preflight passed. No test CSV was opened.', flush=True)


def setup_cache(plan, task, trainer):
    import torch
    s = plan['scientific']; root = Path(s['output_root']); cache = Path(s['cache_root'])
    f = s['frozen'][task['dataset']]
    key = uid(dict(inputs=[s['inputs'][task['train_csv']], s['inputs'][task['val_csv']]],
                   esm=read_json(root/'esm_identity.json'), params={k:f['params'][k] for k in ('window_size','window_layout')},
                   legacy=digest(Path(trainer.mgca.LEGACY_SCRIPT))))
    folder = cache/key
    folder.mkdir(parents=True, exist_ok=True)
    esm_path = Path(trainer.mgca.legacy._esm_cache_path_for_window(
        str(folder/'features.pt'), f['params']['window_size'], f['params']['window_layout']))
    trainer.mgca.get_combined_esm_cache_path = lambda *a, **kw: str(esm_path)
    original_prepare = trainer.prepare_data
    def prepare(args, device):
        if not args.selection_only or args.test_csv is not None:
            raise RuntimeError('Sensitivity worker forbids test access')
        with file_lock(folder/'.cache.lock', blocking=True):
            marker = folder/'verified.json'
            if marker.exists():
                verify_files(read_json(marker)['files'])
            elif any(folder.glob('*.pt')):
                # Orphaned cache from interrupted creation: fail rather than trusting it.
                # Move aside (not delete) only this exact experiment cache's unsealed tensors.
                for p in folder.glob('*.pt'):
                    p.rename(p.with_name(p.name + '.unsealed.' + str(time.time_ns())))
            data, rows, paths = original_prepare(args, device)
            for dataset in data:
                for tensor in dataset.tensors if hasattr(dataset, 'tensors') else ():
                    if not torch.isfinite(tensor).all():
                        raise FloatingPointError('Non-finite feature cache')
            for name in ('esm2', 'morgan'):
                tensor = torch.load(paths[name], map_location='cpu')
                if not torch.isfinite(tensor).all():
                    raise FloatingPointError('Non-finite ' + name + ' cache')
            atomic_json(marker, dict(key=key, files={str(paths[n]):digest(paths[n]) for n in ('esm2','morgan')}))
            return data, rows, paths
    trainer.prepare_data = prepare


def seal_run(out, task, trainer):
    m = read_json(out/'metrics.json')
    if m['selection_only'] is not True or m['test_accessed'] is not False or m['identity']['config_id'] != task['task_id']:
        raise RuntimeError('Unexpected result identity / test access')
    if m['best_epoch'] <= 10 or m['best_epoch'] > m['epochs_ran']:
        raise RuntimeError('Ineligible selected checkpoint')
    for name in ('mse','rmse','mae'):
        if not math.isfinite(m['val_metrics'][name]):
            raise RuntimeError('Invalid validation metric')
    for split in ('train','validation'):
        p = out/(split+'_predictions.csv')
        with p.open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        source = task['train_csv'] if split=='train' else task['val_csv']
        source_rows = trainer.mgca.read_labeled_rows(source)
        if len(rows) != len(source_rows):
            raise RuntimeError('Prediction row count mismatch')
        for i, (row, source_row) in enumerate(zip(rows, source_rows)):
            if int(row['source_row']) != i or row['sample_id'] != trainer.sample_id(source_row):
                raise RuntimeError('Prediction sample identity mismatch')
            if not math.isclose(float(row['y_true']), source_row[2], rel_tol=1e-6, abs_tol=1e-6):
                raise RuntimeError('Prediction label mismatch')
            if not all(math.isfinite(float(row[k])) for k in ('y_true','y_pred','error','abs_error')):
                raise RuntimeError('Non-finite prediction')
        if split == 'validation':
            mse = sum((float(r['y_pred'])-float(r['y_true']))**2 for r in rows)/len(rows)
            if not math.isclose(mse, m['val_metrics']['mse'], rel_tol=1e-5, abs_tol=1e-7):
                raise RuntimeError('MSE disagrees with saved predictions')
            trainer.validate_cgrs_archive(out/'validation_cgrs_outputs.npz', len(rows))
    required = ('metrics.json','identity.json','environment.json','history.csv','best_model.pt',
                'train_predictions.csv','validation_predictions.csv','validation_cgrs_outputs.npz','.complete')
    atomic_json(out/'verified.complete.json', dict(task_id=task['task_id'], verified_unix=time.time(),
               files={name:digest(out/name) for name in required}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--task-id')
    parser.add_argument('--mode', choices=('preflight','cache','train'), required=True)
    parser.add_argument('--concurrency', type=int, default=1)
    args = parser.parse_args(); plan = read_json(args.plan); s = plan['scientific']
    verify_files(s['code'], HERE)
    trainer = load_runtime()
    if args.mode == 'preflight':
        verify_files(s['inputs'])
        preflight(plan, trainer)
        return
    if read_json(Path(s['output_root'])/'preflight.json')['plan_id'] != plan['plan_id']:
        raise RuntimeError('Preflight belongs to a different plan')
    task = next(t for t in plan['tasks'] if t['task_id']==args.task_id)
    verify_files({p:s['inputs'][p] for p in (task['train_csv'],task['val_csv'])})
    setup_cache(plan, task, trainer)
    sys.argv = [str(RUNTIME/'train_v10_cgrs.py')] + build_args(plan, task, args.concurrency)
    import torch
    device = torch.device(s['settings']['device'])
    if device.type == 'cuda':
        torch.cuda.set_device(device); torch.cuda.init()
    if args.mode == 'cache':
        trainer.prepare_data(trainer.parse_args(), device)
        return
    out = Path(s['output_root'])/'runs'/task['relative_dir']
    out.mkdir(parents=True, exist_ok=True)
    with file_lock(out/'.worker.lock'):
        if (out/'identity.json').exists() and read_json(out/'identity.json')['config_id'] != task['task_id']:
            raise RuntimeError('Existing run has incompatible identity; refusing overwrite')
        attempt = out/('attempt_%s.json'%time.time_ns())
        event = dict(task_id=task['task_id'], started_unix=time.time(),
                     configured_concurrency=args.concurrency, command=sys.argv, pid=os.getpid(),
                     environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONHASHSEED')})
        atomic_json(attempt,event)
        try:
            trainer.main()
            seal_run(out, task, trainer)
            event['exit_code'] = 0
        except BaseException:
            event['exit_code'] = 1
            event['failure'] = traceback.format_exc()
            raise
        finally:
            event['ended_unix'] = time.time()
            event['process_duration_sec'] = event['ended_unix']-event['started_unix']
            atomic_json(attempt,event)


if __name__ == '__main__':
    main()
