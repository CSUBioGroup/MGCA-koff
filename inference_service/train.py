"""One fixed-budget full-clean-KinetX refit; never constructs val/test loaders."""
from __future__ import annotations
import argparse
import os
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
import core as c

def rng(g):return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,loader=g.get_state())
def restore(s,g):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);g.set_state(s['loader'])
    if s['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(s['cuda'])

def run(args):
    if args.seed != c.SEED:raise ValueError('Frozen full-refit seed must be 43')
    release=c.verify_release();f=c.frozen();out=args.output_root/('checkpoints/seed_%d'%args.seed)
    out.mkdir(parents=True,exist_ok=True)
    with c.lock(out/'.lock'):
        feature_manifest=c.read(args.output_root/'features_manifest.json')
        c.verify_files(args.output_root,feature_manifest['files'])
        identity=dict(release=release,frozen=f,seed=args.seed,features=feature_manifest['files'],
                      python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=args.device)
        done=c.completed(out)
        if done:
            if done['identity']!=identity:raise RuntimeError('Completed run identity changed')
            print('Verified completed seed',args.seed);return
        if (out/'identity.json').exists() and c.read(out/'identity.json')!=identity:raise RuntimeError('Run identity changed')
        c.atomic_json(out/'identity.json',identity)
        device=c.configure_device(args.device);module=c.module()
        data=c.load(args.output_root/'features.pt')
        p,d,y=data['protein'],data['drug'],data['labels']
        if len(y)!=5446:raise RuntimeError('Wrong training cohort size')
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(args.seed);torch.cuda.reset_peak_memory_stats(device)
        model=module.FullRegressionTransformer(**c.kwargs()).to(device)
        scalar=[model.drug_shrinkage.logit,model.joint_shrinkage.logit];ids={id(x) for x in scalar}
        params=f['params'];a=f['architecture']
        opt=torch.optim.AdamW([dict(params=[x for x in model.parameters() if id(x) not in ids],weight_decay=params['weight_decay']),dict(params=scalar,weight_decay=0.0)],lr=params['lr'])
        g=torch.Generator().manual_seed(args.seed)
        loader=DataLoader(TensorDataset(p,d,y),batch_size=params['batch_size'],shuffle=True,generator=g,num_workers=0)
        start=1;history=[];last=out/'last_state.pt'
        if last.exists():
            s=c.load(last)
            if s['identity']!=identity:raise RuntimeError('Resume identity mismatch')
            model.load_state_dict(s['model'],strict=True);opt.load_state_dict(s['optimizer'])
            start=s['epoch']+1;history=s['history'];restore(s['rng'],g);del s
        for epoch in range(start,f['refit_epochs']+1):
            model.train();model.set_corrections_enabled(epoch>a['protein_warmup_epochs'])
            model.set_shrinkage_learnable(epoch>a['protein_warmup_epochs']+a['fixed_shrinkage_epochs'])
            t=time.perf_counter();total=main=0.0
            for pb,db,yb in loader:
                pb,db,yb=pb.to(device),db.to(device),yb.to(device)
                opt.zero_grad(set_to_none=True);pred,aux=model(pb,db)
                mse=(pred.reshape(-1)-yb.reshape(-1)).square().mean()
                loss=mse+model.loss_components(yb,aux)['total']
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.0,error_if_nonfinite=True);opt.step()
                total+=float(loss.detach())*len(yb);main+=float(mse.detach())*len(yb)
            if device.type=='cuda':torch.cuda.synchronize(device)
            history.append(dict(epoch=epoch,mse=main/len(y),loss=total/len(y),seconds=time.perf_counter()-t,
                                drug_coefficient=float(torch.nn.functional.softplus(model.drug_shrinkage.logit.detach())),
                                joint_coefficient=float(torch.nn.functional.softplus(model.joint_shrinkage.logit.detach())),
                                gradient_norm_last=float(norm),learning_rate=params['lr']))
            c.save(last,dict(identity=identity,epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng(g),history=history))
            c.atomic_csv(out/'history.csv',history)
            print('seed=%d epoch=%d/%d mse=%.6f'%(args.seed,epoch,f['refit_epochs'],main/len(y)),flush=True)
        model.eval();model.set_corrections_enabled(True);model.set_shrinkage_learnable(True)
        payload=dict(checkpoint_type='mgca_KinetX_full_refit_final_state',frozen=f,seed=args.seed,epochs=f['refit_epochs'],
                     model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                     data_sha256=feature_manifest['files']['cohort.csv'],identity=identity,
                     train_seconds=sum(r['seconds'] for r in history),configured_concurrency=args.concurrency,
                     parameter_count=sum(x.numel() for x in model.parameters()),amp=False,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0)
        name=c.checkpoint_name(args.seed);c.save(out/name,payload)
        saved=c.load(out/name);c.checkpoint_valid(saved,args.seed,payload['data_sha256'])
        check=module.FullRegressionTransformer(**c.kwargs()).to(device);check.load_state_dict(saved['model_state_dict'],strict=True)
        check.eval();check.set_corrections_enabled(True);check.set_shrinkage_learnable(True)
        with torch.inference_mode():
            before=model(p[:8].to(device),d[:8].to(device))[0];after=check(p[:8].to(device),d[:8].to(device))[0]
        delta=float((before-after).abs().max())
        if not np.isfinite(delta) or delta>1e-6:raise RuntimeError('Reload mismatch')
        del check,saved
        predictions=[]
        with torch.inference_mode():
            for i in range(0,len(y),64):
                pred,aux=model(p[i:i+64].to(device),d[i:i+64].to(device))
                if not torch.isfinite(pred).all():raise RuntimeError('Non-finite final predictions')
                for j,v in enumerate(pred.reshape(-1).cpu().tolist()):
                    predictions.append(dict(row_index=i+j,y_true=float(y[i+j]),y_pred=v,error=v-float(y[i+j])))
        c.atomic_csv(out/'training_predictions.csv',predictions)
        m=dict(identity=identity,files={n:c.sha(out/n) for n in (name,'history.csv','training_predictions.csv')},
               reload_max_abs_difference=delta,training_only_mse=float(np.mean([r['error']**2 for r in predictions])),
               validation_test_loader_constructed=False)
        c.atomic_json(out/'manifest.json',m);c.atomic_text(out/'.complete',c.sha(out/'manifest.json')+'\n')
        print('Completed',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--concurrency',type=int,default=5);args=p.parse_args()
    try:run(args)
    except Exception:
        import traceback
        c.atomic_json(args.output_root/('checkpoints/seed_%d/failure.json'%args.seed),dict(traceback=traceback.format_exc()))
        raise
