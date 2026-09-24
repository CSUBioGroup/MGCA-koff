"""Complete-grid descriptive summaries; never pick a new best configuration."""
import csv
import html
import itertools
import math
from pathlib import Path
import random
import statistics as st
from common import METRICS, atomic_json, atomic_text, read_json, uid, verify_run, write_csv


def mean_sd(values):
    good = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return (st.mean(good) if good else None, st.stdev(good) if len(good)>1 else None, len(good))


def sign_flip(differences):
    n = len(differences)
    if not n or n > 20:
        return None
    observed = abs(sum(differences))
    # Exact two-sided test on mean paired difference, zero differences retained.
    extreme = sum(abs(sum(s*d for s,d in zip(signs,differences))) >= observed-1e-12
                  for signs in itertools.product((-1,1), repeat=n))
    return extreme/(2**n)


def holm(values):
    order = sorted(range(len(values)), key=lambda i:values[i])
    result = [None]*len(values); previous = 0.
    for rank,i in enumerate(order):
        previous = max(previous, min(1., (len(values)-rank)*values[i]))
        result[i] = previous
    return result


def bootstrap_mean(differences, seed):
    """Descriptive paired-run bootstrap, not an unseen-sample CI."""
    if len(differences)<2:
        return None,None
    rng = random.Random(seed)
    values = sorted(st.mean(rng.choices(differences,k=len(differences))) for _ in range(4000))
    return values[99], values[3899]


def format_num(x):
    return 'NA' if x is None else '%.5f'%x


def matrix_svg(path, family, rows, title):
    xkey,ykey = ('joint_gate_cap','drug_gate_cap') if family=='caps' else ('joint_gate_init','drug_gate_init')
    xs=sorted(set(r[xkey] for r in rows)); ys=sorted(set(r[ykey] for r in rows))
    width=210+160*len(xs); height=100+85*len(ys)
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">'%(width,height,width,height),
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="Arial,sans-serif" fill="#222"><text x="20" y="26" font-size="16">%s</text>'%html.escape(title),
           '<text x="20" y="49" font-size="12">Rows: drug; columns: joint. Validation MSE mean +/- SD; no test.</text>']
    for ix,x in enumerate(xs):
        parts.append('<text x="%d" y="78" text-anchor="middle">%g</text>'%(260+160*ix,x))
    for iy,y in enumerate(ys):
        parts.append('<text x="150" y="%d" text-anchor="end">%g</text>'%(128+85*iy,y))
        for ix,x in enumerate(xs):
            row=next((r for r in rows if r[xkey]==x and r[ykey]==y),None)
            if row is None: continue
            xx,yy=180+160*ix,92+85*iy
            fill='#e6e6e6' if row['reference'] else '#f8f8f8'
            parts.append('<rect x="%d" y="%d" width="150" height="74" fill="%s" stroke="#bbb"/>'%(xx,yy,fill))
            label=format_num(row['mse_mean'])+' +/- '+format_num(row['mse_sd'])
            parts.append('<text x="%d" y="%d" text-anchor="middle" font-size="12">%s</text>'%(xx+75,yy+27,label))
            parts.append('<text x="%d" y="%d" text-anchor="middle" font-size="11">n=%d/%d%s</text>'%(xx+75,yy+51,row['completed_runs'],row['expected_runs'],' reference' if row['reference'] else ''))
    parts.append('</g></svg>'); atomic_text(path,''.join(parts))


def summarize(plan):
    s=plan['scientific']; root=Path(s['output_root']); dest=root/'summary'; dest.mkdir(parents=True,exist_ok=True)
    per_run=[]; matrix=[]; lookup={}; trajectories=[]; events=[]
    for t in plan['tasks']:
        out=root/'runs'/t['relative_dir']
        state=dict(dataset=t['dataset'],protocol=t['protocol'],config_id=t['config']['config_id'],
                   run=t['run'],fold=t['fold'],seed=t['seed'],relative_dir=t['relative_dir'])
        if not (out/'verified.complete.json').exists():
            state.update(status='incomplete',reason='not sealed')
        else:
            try:
                m=verify_run(out,t['task_id'])
                state.update(status='complete',reason='')
                c=t['config']; diag=m['validation_branch_diagnostics']
                row={**state, **{k:c[k] for k in ('drug_gate_cap','joint_gate_cap','drug_gate_init','joint_gate_init','reference')},
                     'families':';'.join(c['families']), 'best_epoch':m['best_epoch'], 'epochs_ran':m['epochs_ran'],
                     'parameter_count':m['parameter_count'], **m['val_metrics'],
                     'drug_gate_final':diag['drug_global_gate'],'joint_gate_final':diag['joint_global_gate'],
                     'drug_gate_change':diag['drug_global_gate']-c['drug_gate_init'],
                     'joint_gate_change':diag['joint_global_gate']-c['joint_gate_init'],
                     'drug_cap_fraction':diag['drug_global_gate']/c['drug_gate_cap'],
                     'joint_cap_fraction':diag['joint_global_gate']/c['joint_gate_cap'],
                     'training_sec':m['timing']['training_duration_sec'],
                     'peak_allocated_mib':m['timing']['training_peak_gpu_memory_allocated_mb'],
                     'configured_concurrency':m['timing']['configured_concurrency']}
                per_run.append(row); lookup[(t['dataset'],t['protocol'],c['config_id'],t['run'])]=row
                with (out/'history.csv').open(encoding='utf-8',newline='') as f:
                    trajectories.extend({**state,**r} for r in csv.DictReader(f))
            except Exception as exc:
                state.update(status='invalid',reason=str(exc))
        matrix.append(state)
        for p in out.glob('attempt_*.json'):
            events.append({**state,**read_json(p)})
    grouped=[]
    for ds in s['settings']['datasets']:
        for protocol in s['settings']['protocols']:
            for c in s['configurations']:
                rs=[r for r in per_run if (r['dataset'],r['protocol'],r['config_id'])==(ds,protocol,c['config_id'])]
                g=dict(dataset=ds,protocol=protocol,**c,completed_runs=len(rs),expected_runs=s['settings']['n_runs'])
                g['families']=';'.join(g['families'])
                for k in METRICS+('drug_gate_final','joint_gate_final','drug_gate_change','joint_gate_change',
                                  'drug_cap_fraction','joint_cap_fraction','training_sec','peak_allocated_mib'):
                    g[k+'_mean'],g[k+'_sd'],g[k+'_n']=mean_sd([r[k] for r in rs])
                grouped.append(g)
    reference=next(c['config_id'] for c in s['configurations'] if c['reference'])
    paired=[]; stats=[]
    for g in grouped:
        if g['reference']: continue
        ds,protocol,cid=g['dataset'],g['protocol'],g['config_id']; differences=[]
        bases=[]
        for run in range(1,s['settings']['n_runs']+1):
            r=lookup.get((ds,protocol,cid,run)); b=lookup.get((ds,protocol,reference,run))
            if r is None or b is None: continue
            d=r['mse']-b['mse']; differences.append(d); bases.append(b['mse'])
            paired.append(dict(dataset=ds,protocol=protocol,config_id=cid,run=run,fold=r['fold'],seed=r['seed'],
                               reference_val_mse=b['mse'],candidate_val_mse=r['mse'],delta_mse=d,
                               relative_delta_percent=100*d/b['mse'] if b['mse'] else None))
        complete=len(differences)==s['settings']['n_runs']
        avg,sd,n=mean_sd(differences)
        ci=bootstrap_mean(differences,int(uid([ds,protocol,cid])[:8],16)) if complete else (None,None)
        stats.append(dict(dataset=ds,protocol=protocol,config_id=cid,paired_n=n,expected_n=s['settings']['n_runs'],
                          complete_pairs=complete,mean_delta_mse=avg,sd_delta_mse=sd,
                          relative_delta_percent=100*avg/st.mean(bases) if bases and st.mean(bases)>0 else None,
                          paired_run_bootstrap_low=ci[0],paired_run_bootstrap_high=ci[1],
                          sign_flip_p=sign_flip(differences) if complete else None,
                          holm_p=None,family='all_nonreference_configurations_x_datasets_x_protocols',
                          interpretation='descriptive validation sensitivity; not test significance/equivalence'))
    all_complete=all(r['status']=='complete' for r in matrix)
    # Do not shrink the correction family to the subset that happened to finish.
    if all_complete and all(r['sign_flip_p'] is not None for r in stats):
        for r,p in zip(stats,holm([r['sign_flip_p'] for r in stats])): r['holm_p']=p
    # Process overlap (not GPU-kernel overlap); report only bounded intervals.
    bounded=[e for e in events if 'ended_unix' in e]
    for e in bounded:
        a,b=e['started_unix'],e['ended_unix']
        overlaps=[(max(a,v['started_unix']),min(b,v['ended_unix'])) for v in bounded
                  if max(a,v['started_unix'])<min(b,v['ended_unix'])]
        e['observed_mean_process_concurrency']=sum(y-x for x,y in overlaps)/(b-a) if b>a else None
        points=sorted({a,b}|{v for p in overlaps for v in p})
        e['observed_max_process_concurrency']=max((sum(x<= (l+r)/2 <y for x,y in overlaps) for l,r in zip(points,points[1:])),default=1)
    write_csv(dest/'completion_matrix.csv',matrix); write_csv(dest/'validation_per_run.csv',per_run)
    write_csv(dest/'validation_mean_sd.csv',grouped); write_csv(dest/'paired_vs_reference.csv',paired)
    write_csv(dest/'paired_statistics.csv',stats); write_csv(dest/'gate_epoch_trajectories.csv',trajectories)
    write_csv(dest/'process_attempts.csv',[{k:v for k,v in e.items() if k not in ('command','environment','failure')} for e in events])
    atomic_json(dest/'audit.json',dict(plan_id=plan['plan_id'],expected=len(matrix),completed=len(per_run),
                complete=all_complete,test_accessed=False,holm_family_size=len(stats),
                notes=s['protocol_notes'],five_run_minimum_two_sided_sign_flip_p=.0625))
    sections=[]
    for ds in s['settings']['datasets']:
        for protocol in s['settings']['protocols']:
            rows=[g for g in grouped if g['dataset']==ds and g['protocol']==protocol]
            for family in ('caps','initialization'):
                name='%s_%s_%s.svg'%(ds,protocol,family)
                matrix_svg(dest/name,family,[r for r in rows if family in r['families'].split(';')],'%s / %s / %s'%(ds,protocol,family))
                sections.append('<img src="%s" alt="%s">'%(name,name))
    table=['<table><thead><tr><th>Dataset / protocol</th><th>Caps (D,J)</th><th>Initial (D,J)</th><th>n</th><th>Validation MSE</th><th>Final drug</th><th>Final joint</th></tr></thead><tbody>']
    for r in grouped:
        table.append('<tr%s><td>%s / %s</td><td>%g, %g</td><td>%g, %g</td><td>%d/%d</td><td>%s ± %s</td><td>%s</td><td>%s</td></tr>'%(
            ' class="reference"' if r['reference'] else '',r['dataset'],r['protocol'],r['drug_gate_cap'],r['joint_gate_cap'],
            r['drug_gate_init'],r['joint_gate_init'],r['completed_runs'],r['expected_runs'],format_num(r['mse_mean']),format_num(r['mse_sd']),format_num(r['drug_gate_final_mean']),format_num(r['joint_gate_final_mean'])))
    table.append('</tbody></table>')
    intro='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MGCA v10 sensitivity</title>
<style>body{font:15px Arial,sans-serif;max-width:1180px;margin:35px auto;line-height:1.6;color:#222;padding:20px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.reference{background:#e6e6e6}img{max-width:100%;display:block;margin:25px 0}aside{padding:15px;background:#f5f5f5}</style>
<h1>MGCA v10：上限与初始化敏感性</h1><aside>只使用 train/validation；灰色为原配置，不自动挑选新的最优参数。
改变上限也改变原归一化正则项的有效强度，因此不是纯粹的上限几何效应。
这里的 validation 同时用于早停，不能当作无偏 test 结果。已有 test 曾反馈架构开发，应披露为回顾性敏感性分析。
五次配对的双侧精确符号置换检验最小 p=0.0625；不显著不等于等效。
warm/drug 的 fold 共享部分训练数据，重复 seed 也不是独立新测试集；run bootstrap 区间仅作条件性描述。</aside>'''
    atomic_text(dest/'report.html',intro+'<p>完成 %d / %d；%s。</p>'%(len(per_run),len(matrix),'全部通过文件校验' if all_complete else '尚不完整，不作完整网格结论')+''.join(sections)+''.join(table)+'</html>')
    print('Summary:',dest/'report.html','complete=',all_complete,flush=True)
    return dict(completed=len(per_run),expected=len(matrix),complete=all_complete)
