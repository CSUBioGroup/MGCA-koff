"""Orchestration integration uses deterministic synthetic training results, not GPUs."""
import copy
import csv
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from common import atomic_json,atomic_text,read_json,split_paths,uid,write_csv
import workflow as wf


class PipelineTests(unittest.TestCase):
    def prepare(self,root,name='out'):
        env=dict(PROJECT_ROOT=str(root/'data'),OUTPUT_ROOT=str(root/name),
            CACHE_ROOT=str(root/name/'cache'),DATASETS='2773',ABLATION_DATASETS='2773',
            PROTOCOLS='warm drug_cold protein_cold',TUNING_TRIALS='12',TOP_K='2',DEVICE='cpu')
        with mock.patch.dict(os.environ,env): cfg=wf.settings()
        for protocol in ('warm','drug_cold','protein_cold'):
            for fold in ([1] if protocol=='protein_cold' else range(1,6)):
                for path in split_paths(cfg['project_root'],'2773',protocol,fold):
                    if not path.exists(): atomic_text(path,'FASTA,SMILES,pkoff\nAAAA,CC,0.5\nBBBB,CO,1.5\n')
        return wf.build_plan(cfg)

    def test_plan_no_test_access_and_mutation_guard(self):
        with tempfile.TemporaryDirectory() as td:
            plan=self.prepare(Path(td))
            self.assertTrue(all('test' not in Path(p).name for p in plan['scientific']['inputs']))
            cfg=plan['scientific']['settings']
            self.assertEqual(wf.build_plan(cfg),plan)
            cfg=copy.deepcopy(cfg); cfg['epochs']+=1
            with self.assertRaises(RuntimeError): wf.build_plan(cfg)

    def test_tpe_crash_resume_replay_and_frozen_formal_parameters(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); plan=self.prepare(root); cfg=plan['scientific']['settings']; calls=[]
            def results(p,ts,jobs):
                calls.extend(ts)
                return [dict(val_metrics={'mse':t['params']['lr']*100+t['params']['dropout']+
                     t['params']['drug_gate_init']*.1+t['params']['joint_gate_init']*.2+t['run']*.001}) for t in ts]
            counter=[0]
            def crash(p,ts,jobs):
                counter[0]+=1
                if counter[0]==4: raise RuntimeError('synthetic interrupted trial')
                return results(p,ts,jobs)
            with mock.patch.object(wf,'test_path',side_effect=AssertionError('test touched during tuning')):
                with mock.patch.object(wf,'launch_group',side_effect=crash):
                    with self.assertRaisesRegex(RuntimeError,'synthetic'): wf.tune(plan,'2773')
                pending=read_json(root/'out/tuning/2773/proposals/0003.json')
                with mock.patch.object(wf,'launch_group',side_effect=results): wf.tune(plan,'2773')
                self.assertEqual(pending,read_json(root/'out/tuning/2773/proposals/0003.json'))
                frozen=read_json(root/'out/frozen/2773/best_params.json')
                baseline=self.prepare(root,'baseline')
                with mock.patch.object(wf,'launch_group',side_effect=results): wf.tune(baseline,'2773')
            for n in range(12):
                a=read_json(root/('out/tuning/2773/proposals/%04d.json'%n))
                b=read_json(root/('baseline/tuning/2773/proposals/%04d.json'%n))
                self.assertEqual(a['params'],b['params'])
            self.assertEqual(frozen['selection']['params'],read_json(root/'baseline/frozen/2773/best_params.json')['selection']['params'])
            for protocol in cfg['protocols']:
                for fold in ([1] if protocol=='protein_cold' else range(1,6)):
                    atomic_text(wf.test_path(cfg,'2773',protocol,fold),'FASTA,SMILES,pkoff\nCCCC,CN,0.7\nDDDD,CF,1.7\n')
            tasks=wf.formal_tasks(plan)+wf.formal_tasks(plan,True)
            self.assertEqual(len(tasks),60)
            self.assertTrue(all(t['params']==frozen['selection']['params'] for t in tasks))
            self.assertEqual({t['seed'] for t in tasks if t['protocol']=='warm'},{42})
            self.assertEqual({t['seed'] for t in tasks if t['protocol']=='protein_cold'},{42,142,242,342,442})
            self.assertTrue(all(t['fold']==1 for t in tasks if t['protocol']=='protein_cold'))

    def test_statistics_and_completion_matrix(self):
        import summarize as sm
        self.assertEqual(sm.signflip([1,1,1,1,1]),.0625)
        self.assertEqual(sm.holm([.04,.01,.03]),[.06,.03,.06])
        with tempfile.TemporaryDirectory() as td:
            plan=self.prepare(Path(td)); cfg=plan['scientific']['settings']
            # No model files: summary must report incomplete, not silently succeed.
            sm.summarize(plan)
            audit=read_json(Path(cfg['output_root'])/'summary/audit.json')
            self.assertFalse(audit['complete']); self.assertEqual(audit['expected'],60)
            self.assertEqual(audit['verified'],0)

    def test_complete_summary_ensembles_and_paired_statistics(self):
        import summarize as sm
        with tempfile.TemporaryDirectory() as td:
            plan=self.prepare(Path(td)); root=Path(plan['scientific']['settings']['output_root'])
            tasks=[]; payloads={}
            for protocol in ('warm','drug_cold','protein_cold'):
                for vi,variant in enumerate(wf.VARIANTS):
                    for n in range(1,6):
                        out=root/'formal'/'2773'/protocol/variant/('run%d'%n)
                        t=dict(dataset='2773',protocol=protocol,variant=variant,run=n,
                            fold=1 if protocol=='protein_cold' else n,seed=42+100*(n-1) if protocol=='protein_cold' else 42,
                            output_dir=str(out),params={'lr':.001},input_sha256={'split':'fixture'})
                        tasks.append(t); rows=[]
                        for i,y in enumerate((0.,1.)):
                            delta=.1+.02*vi+.001*n
                            rows.append(dict(source_row=i,sample_id='%s_%s_%d'%(protocol,'shared' if protocol=='protein_cold' else n,i),
                                y_true=y,y_pred=y+delta,error=delta,abs_error=delta))
                        write_csv(out/'test_predictions.csv',rows);write_csv(out/'history.csv',[{'epoch':11,'drug_gate_mean':.1}])
                        payloads[str(out)]=dict(identity={'config_id':'fixture'},best_epoch=11,epochs_ran=12,
                            parameter_count=13392395,test_metrics=sm.metrics([r['y_true'] for r in rows],[r['y_pred'] for r in rows]),
                            timing={'training_duration_sec':1.},test_branch_diagnostics={'drug_global_gate':.1,'joint_global_gate':.03})
            atomic_json(root/'formal_manifest.json',dict(plan_id=plan['plan_id'],tasks=tasks))
            with mock.patch.object(sm,'verified',side_effect=lambda t:payloads[t['output_dir']]): sm.summarize(plan)
            self.assertTrue(read_json(root/'summary/audit.json')['complete'])
            stats=read_json(root/'summary/paired_statistics.json')
            self.assertEqual(len(stats),9)
            self.assertTrue(all(r['ablation_minus_full_mean']>0 for r in stats))
            with (root/'summary/test_mean_sd.csv').open(newline='') as f: groups=list(csv.DictReader(f))
            self.assertEqual(len(groups),12)
            with (root/'summary/pooled_ensemble_metrics.csv').open(newline='') as f: aggregates=list(csv.DictReader(f))
            self.assertEqual(sum(r['method']=='five_seed_prediction_ensemble' for r in aggregates),4)


if __name__=='__main__': unittest.main(verbosity=2)
