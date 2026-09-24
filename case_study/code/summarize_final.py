"""Final evidence matrix: all five frozen checkpoints, both panels and both occlusions."""
import argparse,csv
from pathlib import Path
import common_v10 as c

def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();root=a.output_root
    records=c.checkpoint_records(root/'checkpoints')
    reference=None
    import torch
    for record in records:
        payload=c.torch_load(torch,record['path']);reference=c.validate_checkpoint(payload,record,reference)
    dest=root/'summary';dest.mkdir(exist_ok=True)
    rows=[];panels=[]
    for panel in ['four_target','k4dd']:
        folder=root/'predictions'/panel;m=c.read_json(folder/'manifest.json')
        if m['model_config']['model_variant']!=c.MODEL_VARIANT or not c.artifact_files_valid(folder,m):raise RuntimeError('Panel provenance/corruption')
        r=read_csv(folder/'metrics_by_target_and_seed.csv')
        rows.extend([{'panel':panel,**item} for item in r])
        unique=read_csv(folder/'unique_compound_five_seed_predictions.csv')
        panels.append({'panel':panel,'manifest_sha256':c.sha256_file(folder/'manifest.json'),'target_ids':sorted({x['uniprot_id'] for x in unique})})
    c.atomic_csv(dest/'all_target_metrics.csv',rows)
    occlusions=[]
    for target in ['factor_xa','dpp4']:
        folder=root/'occlusion'/target;m=c.read_json(folder/'factor_xa_stage2_manifest.json')
        if m['model_sha256']!=c.EXPECTED_MODEL_SHA256 or m['seeds']!=list(c.DEFAULT_SEEDS):raise RuntimeError('Old occlusion mixed in')
        audit=m['stage1_no_occlusion_prediction_audit']
        if not audit or not audit['passed']:raise RuntimeError('No-occlusion prediction equality audit failed')
        occlusions.append({'target':target,'manifest_sha256':c.sha256_file(folder/'factor_xa_stage2_manifest.json'),
            'baseline_max_difference':audit['maximum_absolute_difference'],'compound_count':m['compound_count']})
    structure=c.read_json(root/'structure/final_structure_audit.json')
    frozen=c.read_json(c.PROJECT_ROOT/'frozen/experiment_lock.json')
    report={'status':'complete','protocol':c.PROTOCOL,'checkpoint_count':len(records),'epochs':c.DEFAULT_EPOCHS,
        'seeds':list(c.DEFAULT_SEEDS),'model_sha256':c.EXPECTED_MODEL_SHA256,'config_id':c.EXPECTED_CONFIG_ID,
        'benchmark_plan_id':frozen['benchmark_plan_id'],'benchmark_ablations':'Frozen existing five runs; no extra training',
        'panels':panels,'occlusions':occlusions,'structure':structure,
        'checkpoint_selection_using_case_labels':False,'performance_gate':False,
        'illustrative_compounds':{'factor_xa':'factor_xa_05','dpp4':'dpp4_12'},
        'dpp4_illustration_origin':'Historically selected highest observed pKoff; fixed before this new run, not used to select checkpoints.',
        'boundaries':['Retrospective ranking, not prospective external validation.',
                      'All five full-data refits reported; no checkpoint selected by case scores.',
                      'Mean/SD over training seeds is not calibrated predictive uncertainty.',
                      'Exact-sequence and k-mer exposure checks are not proof of no homologs.',
                      'Docking geometry reused; new model-dependent occlusion/contact analysis recomputed.']}
    c.atomic_json(dest/'final_audit.json',report)
    lines=['# MGCA final: complete case-study report','',
        'Frozen unbounded MGCA; five full-2773 refits, 33 fixed epochs. No additional HPO or benchmark ablation training.','',
        '## Target-specific prediction metrics','',
        '| Panel | Target | Evaluation | N | MSE | MAE | Pearson | Spearman |',
        '|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append('| '+' | '.join(str(r.get(k,'')) for k in ['panel','target_name','evaluation','n','mse','mae','pearson','spearman'])+' |')
    lines+=['','## Interpretation boundaries','']+['- '+s for s in report['boundaries']]
    lines+=['','Structure renderer: '+structure['renderer'],
        'Full outputs: predictions/, occlusion/, figures/, structure/. Negative findings are retained.']
    c.atomic_text(dest/'final_report.md','\n'.join(lines)+'\n')
    c.atomic_text(dest/'.complete',c.sha256_file(dest/'final_audit.json')+'\n')
    print('COMPLETE: all model-dependent case-study stages verified',dest,flush=True)

if __name__=='__main__':main()
