import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from common import (atomic_json, atomic_text, configurations, digest, file_lock, split_paths,
                    verify_files, verify_run, write_csv)
from workflow import make_plan, settings, build_args
from summarize import holm, sign_flip, summarize


class SensitivityTests(unittest.TestCase):
    def test_grid_deduplicates_reference(self):
        cs=configurations()
        self.assertEqual(len(cs),17)
        self.assertEqual(sum(c['reference'] for c in cs),1)
        self.assertEqual(sum('caps' in c['families'] for c in cs),9)
        self.assertEqual(sum('initialization' in c['families'] for c in cs),9)
        self.assertEqual(len(configurations(equal=True)),19)

    def test_grid_rejects_invalid_initialization(self):
        with self.assertRaises(ValueError): configurations(init_multipliers=(1,4))
        with self.assertRaises(ValueError): configurations(cap_multipliers=(.1,1))

    def test_exact_statistics(self):
        self.assertEqual(sign_flip([1]*5),.0625)
        self.assertEqual(sign_flip([1]*10),.001953125)
        self.assertEqual(sign_flip([0]*5),1)
        self.assertEqual(holm([.01,.04,.03]),[.03,.06,.06])

    def test_split_mapping_never_test(self):
        for ds in ('2773','KinetX'):
            for p in ('warm','drug_cold','protein_cold'):
                for f in range(1,6):
                    paths=split_paths(Path('/project'),ds,p,f)
                    self.assertEqual(len(paths),2)
                    self.assertFalse(any('test' in x.name for x in paths))

    def fixture(self, root):
        cfg=settings(); cfg.update(project_root=str(root/'data'), esm2_path=str(root/'esm'),
                                   datasets=['2773','KinetX'],protocols=['warm','drug_cold','protein_cold'],
                                   n_runs=5, seed_policy='benchmark',cap_multipliers=[.5,1,2],
                                   init_multipliers=[.5,1,2],equal_caps=False)
        for ds in cfg['datasets']:
            for p in cfg['protocols']:
                for f in range(1,6):
                    for path in split_paths(cfg['project_root'],ds,p,f):
                        atomic_text(path,'FASTA,SMILES,pkoff\nAAA,CC,1\nBBB,CO,2\n')
        return cfg,make_plan(cfg,root/'outputs',root/'cache')

    def test_plan_count_pairing_and_frozen_hyperparameters(self):
        with tempfile.TemporaryDirectory() as td:
            cfg,p=self.fixture(Path(td))
            self.assertEqual(len(p['tasks']),510)
            self.assertEqual(len(p['scientific']['inputs']),44)
            self.assertEqual(len({t['task_id'] for t in p['tasks']}),510)
            for t in p['tasks']:
                args=build_args(p,t,5)
                self.assertIn('--selection-only',args); self.assertNotIn('--test-csv',args)
                self.assertEqual(float(args[args.index('--dropout')+1]),.15 if t['dataset']=='2773' else .17)
                self.assertEqual(t['seed'],42+100*(t['run']-1) if t['protocol']=='protein_cold' else 42)
            t=p['tasks'][0]
            serial=build_args(p,t,1);parallel=build_args(p,t,5)
            at=serial.index('--concurrency')+1
            self.assertEqual(serial[:at]+serial[at+1:],parallel[:at]+parallel[at+1:])
            again=make_plan(cfg,Path(td)/'outputs',Path(td)/'cache')
            self.assertEqual(p,again)
            atomic_text(Path(t['train_csv']),'FASTA,SMILES,pkoff\nAAA,CC,9\nBBB,CO,2\n')
            changed=make_plan(cfg,Path(td)/'outputs',Path(td)/'cache')
            self.assertNotEqual(p['plan_id'],changed['plan_id'])

    def test_no_duplicate_benchmark_runs(self):
        with tempfile.TemporaryDirectory() as td:
            cfg,p=self.fixture(Path(td));cfg['n_runs']=10
            with self.assertRaises(ValueError):make_plan(cfg,Path(td)/'out',Path(td)/'cache')
            cfg['seed_policy']='paired_seeds'
            p=make_plan(cfg,Path(td)/'out',Path(td)/'cache')
            self.assertEqual(len(p['tasks']),1020)
            t=next(t for t in p['tasks'] if t['protocol']=='drug_cold' and t['run']==7)
            self.assertEqual((t['fold'],t['seed']),(2,642))

    def test_integrity_and_lock(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'data';atomic_text(p,'original')
            mapping={str(p):digest(p)};verify_files(mapping)
            atomic_text(p,'modified')
            with self.assertRaises(RuntimeError):verify_files(mapping)
            with file_lock(Path(td)/'.lock'):
                with self.assertRaises(RuntimeError):
                    with file_lock(Path(td)/'.lock'):pass

    def test_partial_summary_no_holm_or_claim_of_completion(self):
        with tempfile.TemporaryDirectory() as td:
            cfg,p=self.fixture(Path(td))
            result=summarize(p)
            self.assertEqual(result,dict(completed=0,expected=510,complete=False))
            dest=Path(td)/'outputs/summary'
            self.assertTrue((dest/'report.html').is_file())
            with (dest/'paired_statistics.csv').open(encoding='utf-8',newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),96)
            self.assertTrue(all(not r['holm_p'] for r in rows))
            import xml.etree.ElementTree as ET
            for path in dest.glob('*.svg'):ET.parse(path)

    def test_completed_summary_and_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            cfg,p=self.fixture(Path(td));cfg.update(datasets=['2773'],protocols=['warm'],n_runs=1)
            p=make_plan(cfg,Path(td)/'outputs',Path(td)/'cache')
            for t in p['tasks']:
                out=Path(td)/'outputs/runs'/t['relative_dir']
                c=t['config'];m=dict(selection_only=True,test_accessed=False,identity={'config_id':t['task_id']},
                    best_epoch=11,epochs_ran=11,parameter_count=13392395,
                    val_metrics=dict(mse=.3,rmse=.3**.5,mae=.4,r2=.2,pearson=None,spearman=None),
                    validation_branch_diagnostics=dict(drug_global_gate=c['drug_gate_init'],joint_global_gate=c['joint_gate_init']),
                    timing=dict(training_duration_sec=10,training_peak_gpu_memory_allocated_mb=None,configured_concurrency=5))
                atomic_json(out/'metrics.json',m)
                write_csv(out/'history.csv',[dict(epoch=11,drug_gate_mean=c['drug_gate_init'],joint_gate_mean=c['joint_gate_init'])])
                atomic_json(out/'verified.complete.json',dict(task_id=t['task_id'],files={n:digest(out/n) for n in ('metrics.json','history.csv')}))
            self.assertTrue(summarize(p)['complete'])
            t=p['tasks'][0];out=Path(td)/'outputs/runs'/t['relative_dir']
            atomic_text(out/'history.csv','corrupt')
            with self.assertRaises(RuntimeError):verify_run(out,t['task_id'])
            self.assertFalse(summarize(p)['complete'])


if __name__=='__main__':unittest.main(verbosity=2)
