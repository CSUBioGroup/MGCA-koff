from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT/'runtime/mgca_hyperparameter_tuning/v10_unbounded/model.py'
SEED = 43

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def key(text): return hashlib.sha256(text.encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def frozen(): return read(ROOT/'frozen/refit_config.json')

def atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=str(path.parent),delete=False) as f:
        f.write(text);tmp=f.name
    os.replace(tmp,path)

def atomic_json(path,value): atomic_text(path,json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def atomic_csv(path,rows):
    import io
    if not rows: raise ValueError('Empty CSV output')
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_text(path,buf.getvalue())

def save(path,payload):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.%d'%os.getpid())
    torch.save(payload,temp);os.replace(temp,path)

def load(path):
    import torch
    try:return torch.load(path,map_location='cpu',weights_only=False)
    except TypeError as e:
        if 'weights_only' not in str(e):raise
        return torch.load(path,map_location='cpu')

@contextmanager
def lock(path):
    import fcntl
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Directory is locked: '+str(path))
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def verify_files(root,files):
    for name,h in files.items():
        if sha(Path(root)/name)!=h:raise RuntimeError('Hash mismatch: '+name)

def verify_release():
    verify_files(ROOT,read(ROOT/'release_manifest.json')['files'])
    return sha(ROOT/'release_manifest.json')

def verify_esm(path):
    print('Checking pinned ESM2 files...',flush=True)
    verify_files(path,read(ROOT/'frozen/esm_identity.json'))
    # The frozen extractor used the pinned PyTorch shards, not optional alternatives.
    if list(Path(path).glob('*.safetensors')):
        raise RuntimeError('ESM2 directory contains alternative safetensors; use the original pinned shard directory')

def module():
    if sha(MODEL)!=frozen()['model_sha256']:raise RuntimeError('Frozen model changed')
    spec=importlib.util.spec_from_file_location('mgca_kinetx_model',MODEL)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def kwargs():
    f=frozen();p=f['params'];a=f['architecture']
    return dict(proj_dim1=2560,proj_dim2=2048,hidden_dim=512,dropout=p['dropout'],nums_of_experts=4,
                ablation='no',joint_rank=128,protein_aux_weight=a['protein_aux_weight'],
                drug_utility_weight=a['drug_utility_weight'],joint_utility_weight=a['joint_utility_weight'],
                branch_margin=a['branch_margin'],drug_gate_init=p['drug_gate_init'],
                joint_gate_init=p['joint_gate_init'],joint_branch_dropout=a['joint_branch_dropout'])

def configure_device(value):
    import torch
    d=torch.device(value)
    if d.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.cuda.set_device(d);torch.cuda.init()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    elif d.type!='cpu':raise ValueError('Only cuda/cpu supported')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.set_num_threads(1);return d

def sequence(value):
    seq=''.join(value.split()).upper()
    if not seq or not set(seq)<=set('ACDEFGHIKLMNPQRSTVWYXBZUO'):raise ValueError('Invalid amino-acid sequence')
    return seq

def cohort():
    rows=[]
    for part in ('train','val','test'):
        with (ROOT/'inputs'/('%s_run1.csv'%part)).open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                y=float(r['pkoff'])
                if not math.isfinite(y):raise ValueError('Non-finite label')
                rows.append(dict(FASTA=sequence(r['FASTA']),SMILES=r['SMILES'].strip(),pkoff=y))
    if len(rows)!=5446:raise RuntimeError('Expected complete cleaned KinetX cohort of 5446 rows')
    # Deliberately no deduplication: all benchmark measurements are retained.
    return rows

def checkpoint_name(seed):return 'mgca_fullKinetX_seed_%d_epoch%d.pt'%(seed,frozen()['refit_epochs'])

def checkpoint_valid(payload,seed,data_sha=None):
    if seed!=SEED:raise RuntimeError('Only the frozen seed-43 checkpoint is accepted')
    if payload.get('checkpoint_type')!='mgca_KinetX_full_refit_final_state':raise RuntimeError('Wrong checkpoint type')
    if payload.get('frozen')!=frozen() or payload.get('seed')!=seed:raise RuntimeError('Checkpoint configuration mismatch')
    if data_sha and payload.get('data_sha256')!=data_sha:raise RuntimeError('Checkpoint cohort mismatch')
    if payload.get('epochs')!=frozen()['refit_epochs']:raise RuntimeError('Wrong final epoch')
    if not isinstance(payload.get('model_state_dict'),dict):raise RuntimeError('Checkpoint lacks model state')
    if payload.get('parameter_count')!=13392395 or payload.get('amp') is not False:raise RuntimeError('Checkpoint architecture/training mode mismatch')

def completed(folder):
    folder=Path(folder)
    if not (folder/'.complete').exists():return False
    m=read(folder/'manifest.json')
    if (folder/'.complete').read_text().strip()!=sha(folder/'manifest.json'):raise RuntimeError('Invalid completion marker')
    verify_files(folder,m['files']);return m
