"""Read-only result analysis; writes only derived summary files, never chooses HPs."""
import csv
import itertools
import math
from pathlib import Path
import statistics
import numpy as np
from common import atomic_json, atomic_text, read_json, write_csv
from workflow import VARIANTS, verified

METRICS=('mse','rmse','mae','r2','pearson','spearman')


def rankdata(x):
    order=np.argsort(x,kind='stable'); result=np.empty(len(x),float); i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]: j+=1
        result[order[i:j]]=(i+j-1)/2+1; i=j
    return result


def corr(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<2 or np.std(a)==0 or np.std(b)==0: return None
    return float(np.corrcoef(a,b)[0,1])


def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); err=p-y; mse=float(np.mean(err**2))
    den=float(np.sum((y-y.mean())**2))
    return dict(mse=mse,rmse=math.sqrt(mse),mae=float(np.mean(abs(err))),
        r2=1-float(np.sum(err**2))/den if den else None,
        pearson=corr(y,p),spearman=corr(rankdata(y),rankdata(p)))


def predictions(t):
    with (Path(t['output_dir'])/'test_predictions.csv').open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def signflip(d):
    d=np.asarray(d,float); observed=abs(float(d.mean()))
    return sum(abs(float(np.mean(d*np.asarray(s))))>=observed-1e-15
        for s in itertools.product((-1,1),repeat=len(d)))/(2**len(d))


def bootstrap(d,rng,n=10000):
    d=np.asarray(d,float)
    # Chunked to bound memory on pooled sample-level arrays.
    values=[]
    for begin in range(0,n,100):
        ix=rng.integers(0,len(d),size=(min(100,n-begin),len(d)))
        values.extend(d[ix].mean(axis=1).tolist())
    return [float(x) for x in np.quantile(values,[.025,.975])]


def holm(ps):
    order=np.argsort(ps); out=[None]*len(ps); previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,ps[index]*(len(ps)-rank))); out[int(index)]=previous
    return out


def summarize(plan):
    cfg=plan['scientific']['settings']; root=Path(cfg['output_root']); dest=root/'summary'
    matrix=[]; good={}; all_tasks=[]
    # Expected matrix exists even when a phase has not yet started.
    manifests={}
    for name in ('formal_manifest.json','ablation_manifest.json'):
        if (root/name).exists():
            m=read_json(root/name)
            if m['plan_id']!=plan['plan_id']: raise RuntimeError('Summary manifest mismatch')
            for t in m['tasks']: manifests[(t['dataset'],t['protocol'],t['variant'],t['run'])]=t
    for dataset in cfg['datasets']:
        variants=list(VARIANTS) if dataset in cfg['ablation_datasets'] else ['full']
        for protocol in cfg['protocols']:
            for variant in variants:
                for run in range(1,6):
                    key=(dataset,protocol,variant,run); t=manifests.get(key)
                    r=dict(dataset=dataset,protocol=protocol,variant=variant,run=run,status='not_started')
                    if t:
                        all_tasks.append(t); r['output_dir']=t['output_dir']
                        try:
                            m=verified(t); good[key]=(t,m); r['status']='verified'
                        except Exception as exc: r.update(status='incomplete_or_invalid',reason=str(exc))
                    matrix.append(r)
    write_csv(dest/'completion_matrix.csv',matrix)
    per_run=[]; history=[]; groups=[]; aggregated=[]; coverage=[]; pairing=[]; tests=[]
    for key,(t,m) in sorted(good.items()):
        per_run.append(dict(dataset=key[0],protocol=key[1],variant=key[2],run=key[3],
            fold=t['fold'],seed=t['seed'],config_id=m['identity']['config_id'],
            best_epoch=m['best_epoch'],epochs=m['epochs_ran'],parameter_count=m['parameter_count'],
            **m['test_metrics'],**{'timing_'+k:v for k,v in m['timing'].items()},
            **m['test_branch_diagnostics']))
        with (Path(t['output_dir'])/'history.csv').open(encoding='utf-8',newline='') as f:
            history += [dict(dataset=key[0],protocol=key[1],variant=key[2],run=key[3],**r) for r in csv.DictReader(f)]
    write_csv(dest/'test_per_run.csv',per_run); write_csv(dest/'gate_epoch_trajectories.csv',history)
    group_keys=sorted(set(k[:3] for k in good))
    for dataset,protocol,variant in group_keys:
        items=[good[(dataset,protocol,variant,n)] for n in range(1,6) if (dataset,protocol,variant,n) in good]
        row=dict(dataset=dataset,protocol=protocol,variant=variant,n=len(items),complete=len(items)==5)
        for metric in METRICS:
            vals=[m['test_metrics'][metric] for t,m in items if m['test_metrics'][metric] is not None]
            row.update({metric+'_mean':statistics.mean(vals) if vals else None,
                        metric+'_sd':statistics.stdev(vals) if len(vals)>1 else None,
                        metric+'_defined_n':len(vals)})
        groups.append(row)
        if len(items)!=5: continue
        panels=[predictions(t) for t,m in items]
        if protocol=='protein_cold':
            ids=[(r['source_row'],r['sample_id'],float(r['y_true'])) for r in panels[0]]
            if any([(r['source_row'],r['sample_id'],float(r['y_true'])) for r in panel]!=ids for panel in panels[1:]):
                raise RuntimeError('Seed ensemble sample mismatch')
            y=[v[2] for v in ids]; pred=np.mean([[float(r['y_pred']) for r in panel] for panel in panels],axis=0)
            method='five_seed_prediction_ensemble'
            merged=[dict(source_row=i[0],sample_id=i[1],y_true=i[2],y_pred=float(p)) for i,p in zip(ids,pred)]
        else:
            merged=[dict(run=n+1,**r) for n,panel in enumerate(panels) for r in panel]
            ids=[r['sample_id'] for r in merged]; duplicate=len(ids)-len(set(ids))
            coverage.append(dict(dataset=dataset,protocol=protocol,variant=variant,test_records=len(ids),
                unique_raw_pairs=len(set(ids)),duplicate_raw_pair_coverage=duplicate,
                canonical_disjoint_guaranteed=False))
            method='non_strict_pooled_test_predictions' if duplicate else 'raw_pair_unique_pooled_test_predictions'
            # Existing warm canonical-equivalent overlaps prevent an unqualified strict-OOF claim.
            y=[float(r['y_true']) for r in merged]; pred=[float(r['y_pred']) for r in merged]
        write_csv(dest/'predictions'/('%s_%s_%s.csv'%(dataset,protocol,variant)),merged)
        aggregated.append(dict(dataset=dataset,protocol=protocol,variant=variant,method=method,**metrics(y,pred)))
    write_csv(dest/'test_mean_sd.csv',groups); write_csv(dest/'pooled_ensemble_metrics.csv',aggregated)
    write_csv(dest/'test_coverage_audit.csv',coverage)
    rng=np.random.default_rng(20260910)
    for dataset in cfg['ablation_datasets']:
        for protocol in cfg['protocols']:
            for variant in list(VARIANTS)[1:]:
                if any((dataset,protocol,v,n) not in good for v in ('full',variant) for n in range(1,6)): continue
                differences=[]; samples={}
                for n in range(1,6):
                    tf,mf=good[(dataset,protocol,'full',n)]; ta,ma=good[(dataset,protocol,variant,n)]
                    if tf['input_sha256']!=ta['input_sha256'] or tf['seed']!=ta['seed'] or tf['params']!=ta['params']:
                        raise RuntimeError('Invalid paired comparison')
                    d=ma['test_metrics']['mse']-mf['test_metrics']['mse']; differences.append(d)
                    pairing.append(dict(dataset=dataset,protocol=protocol,variant=variant,run=n,
                        full_mse=mf['test_metrics']['mse'],ablation_mse=ma['test_metrics']['mse'],ablation_minus_full=d))
                    pf,pa=predictions(tf),predictions(ta)
                    if len(pf)!=len(pa): raise RuntimeError('Paired row count mismatch')
                    for f,a in zip(pf,pa):
                        if (f['source_row'],f['sample_id'],f['y_true'])!=(a['source_row'],a['sample_id'],a['y_true']):
                            raise RuntimeError('Paired sample mismatch')
                        y=float(f['y_true']); delta=(float(a['y_pred'])-y)**2-(float(f['y_pred'])-y)**2
                        samples.setdefault(f['sample_id'],[]).append(delta)
                fullmean=statistics.mean(good[(dataset,protocol,'full',n)][1]['test_metrics']['mse'] for n in range(1,6))
                ablmean=fullmean+statistics.mean(differences)
                tests.append(dict(dataset=dataset,protocol=protocol,variant=variant,n=5,
                    ablation_minus_full_mean=statistics.mean(differences),
                    full_mse_reduction_pct=100*(ablmean-fullmean)/ablmean if ablmean else None,
                    ablation_increase_vs_full_pct=100*(ablmean-fullmean)/fullmean if fullmean else None,
                    exact_signflip_p=signflip(differences),
                    paired_run_bootstrap_ci=bootstrap(differences,rng),
                    sample_pair_cluster_bootstrap_ci=bootstrap([np.mean(v) for v in samples.values()],rng),
                    bootstrap_note='Conditional on fitted models; raw-pair-cluster means. Not a new training-seed or protein-cluster CI.'))
    for r,p in zip(tests,holm([r['exact_signflip_p'] for r in tests])): r['holm_p_all_mse_ablations']=p
    write_csv(dest/'paired_mse_differences.csv',pairing); atomic_json(dest/'paired_statistics.json',tests)
    audit=dict(plan_id=plan['plan_id'],expected=len(matrix),verified=len(good),complete=len(good)==len(matrix),
        failures=[r for r in matrix if r['status']!='verified'],test_used_for_hpo=False,
        caveats=['Historical test feedback makes this a retrospective development experiment.',
            'Five-run exact two-sided signflip minimum p=0.0625; no significance is not equivalence.',
            'Warm canonical-equivalent overlaps retained. Pooling does not prove strict canonical OOF.',
            'Global softplus weights are not sample-wise uncertainty routing.',
            'A branch ablation removes its fusion contribution, not that modality inside the retained joint module.'])
    atomic_json(dest/'audit.json',audit)
    lines=['# MGCA v10 unbounded joint tuning','',
        'Verified formal runs: %d / %d.'%(len(good),len(matrix)),'',
        'Selection: warm validation only. Below: five-run test mean +/- sample SD.','',
        '| Dataset | Protocol | Variant | n | MSE | RMSE | MAE | R2 | Pearson | Spearman |',
        '|---|---|---|---|---|---|---|---|---|---|']
    for r in groups:
        cells=[]
        for k in METRICS:
            m,s=r[k+'_mean'],r[k+'_sd']
            cells.append('NA' if m is None else ('%.4f +/- %.4f'%(m,s) if s is not None else '%.4f'%m))
        lines.append('| '+' | '.join([r['dataset'],r['protocol'],r['variant'],str(r['n'])]+cells)+' |')
    lines+=['','## Limitations','']+['- '+s for s in audit['caveats']]
    atomic_text(dest/'report.md','\n'.join(lines)+'\n')
    print('Summary:',dest,'complete=',audit['complete'],flush=True)
