from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import core as c

def prepare(out,esm,device):
    import torch
    release=c.verify_release();c.verify_esm(esm)
    identity=dict(release=release,esm=c.read(c.ROOT/'frozen/esm_identity.json'))
    existing=out/'features_manifest.json'
    if existing.exists():
        old=c.read(existing)
        if old['identity']!=identity:raise RuntimeError('Feature identity changed')
        c.verify_files(out,old['files']);print('Verified full-cohort features');return
    rows=c.cohort();module=c.module();legacy=module.legacy
    molecules=[legacy.Chem.MolFromSmiles(r['SMILES']) for r in rows]
    if any(m is None for m in molecules):raise ValueError('Invalid cohort SMILES; no silent exclusions')
    device=c.configure_device(device);sequences=list(dict.fromkeys(r['FASTA'] for r in rows));index={s:i for i,s in enumerate(sequences)}
    started=time.perf_counter()
    tokenizer=legacy.AutoTokenizer.from_pretrained(str(esm),local_files_only=True)
    model=legacy.AutoModelForMaskedLM.from_pretrained(str(esm),local_files_only=True).to(device).eval()
    with torch.inference_mode():
        unique=legacy.batch_extract_esm2(sequences,tokenizer,model,device,batch_size=1,window_size=8,window_layout='even_span_v2').cpu().float()
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
    parts=[]
    for radius in range(4):
        fp,valid=legacy.get_fingerprint(radius,molecules,device='cpu',fingerprint_type='morgan')
        if not valid.all():raise ValueError('Invalid Morgan channel')
        parts.append(fp.unsqueeze(1))
    protein=unique[torch.tensor([index[r['FASTA']] for r in rows])];drug=torch.cat(parts,dim=1).float()
    if tuple(protein.shape)!=(5446,4,2560) or tuple(drug.shape)!=(5446,4,2048):raise RuntimeError('Wrong feature dimensions')
    if not torch.isfinite(protein).all() or not torch.isfinite(drug).all():raise RuntimeError('Non-finite features')
    c.atomic_csv(out/'cohort.csv',rows)
    c.save(out/'features.pt',dict(protein=protein,drug=drug,labels=torch.tensor([r['pkoff'] for r in rows],dtype=torch.float32)))
    c.save(out/'protein_warm_cache.pt',dict(sequences=sequences,features=unique,esm_identity=identity['esm'],window_size=8,window_layout='even_span_v2'))
    c.atomic_json(existing,dict(identity=identity,files={n:c.sha(out/n) for n in ('cohort.csv','features.pt','protein_warm_cache.pt')},
                               rows=len(rows),unique_proteins=len(sequences),truncated_rows=sum(len(r['FASTA'])>1022 for r in rows),
                               extraction_seconds=time.perf_counter()-started,
                               note='Complete cleaned benchmark cohort; original tokenizer truncation retained and reported; not raw 5624-row table.'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','prepare','train','package'],default='all');args=p.parse_args()
    out=Path(os.environ.get('OUTPUT_ROOT',c.ROOT/'outputs')).resolve();out.mkdir(parents=True,exist_ok=True)
    esm=Path(os.environ['ESM2_PATH']).resolve();device=os.environ.get('DEVICE','cuda:0')
    c.verify_release()
    with c.lock(out/'.workflow.lock'):
        # ESM is unloaded before training subprocesses are launched.
        if args.phase in ('all','prepare','train'):prepare(out,esm,device)
        if args.phase in ('all','train'):
            seed=c.SEED
            path=out/'logs'/('seed_%d_%d.log'%(seed,time.time_ns()));path.parent.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,str(c.ROOT/'train.py'),'--seed',str(seed),'--output-root',str(out),'--device',device,'--concurrency','1']
            with path.open('w',encoding='utf-8') as log:rc=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
            print('seed',seed,'exit',rc,'log',path,flush=True)
            if rc:raise RuntimeError('Training failed; retained for resume. No automatic batch-size changes')
        if args.phase in ('all','package'):
            import package_release
            package_release.build(out,esm,os.environ.get('INCLUDE_ESM','0')=='1')

if __name__=='__main__':main()
