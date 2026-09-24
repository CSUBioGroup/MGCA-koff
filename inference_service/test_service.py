"""Model-kernel CPU checks and HTTP contract tests; no learned refit is claimed."""
import ast
import asyncio
import math
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from contextlib import nullcontext
import core as c

def model_class():
    import torch
    ns=dict(torch=torch,nn=torch.nn,F=torch.nn.functional,math=math)
    tree=ast.parse((c.ROOT/'runtime/local/ESM_Morgan_Hybrid_Fusion.py').read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('MultiExpertEncoder','GatedExpertFusion')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<legacy classes>','exec'),ns)
    ns['legacy']=types.SimpleNamespace(**ns)
    tree=ast.parse(c.MODEL.read_text(encoding='utf-8-sig'))
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ABLATIONS' for t in n.targets):ns['ABLATIONS']=ast.literal_eval(n.value)
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef)],type_ignores=[]),'<model classes>','exec'),ns)
    return ns['FullRegressionTransformer']

class CoreTests(unittest.TestCase):
    def test_cohort(self):
        self.assertEqual(len(c.cohort()),5446)
        self.assertEqual(c.frozen()['refit_epochs'],55)
        self.assertEqual(c.SEED,43)
        self.assertEqual(c.frozen()['seeds'],[43])
        self.assertEqual(c.frozen()['params']['window_size'],8)
        self.assertTrue(all(r['selection_only'] and not r['test_accessed'] for r in c.read(c.ROOT/'frozen/epoch_selection.json')))

    def test_release(self):self.assertEqual(len(c.verify_release()),64)

    def test_real_model(self):
        import torch
        torch.set_num_threads(1);model=model_class()(**c.kwargs())
        self.assertEqual(sum(p.numel() for p in model.parameters()),13392395)
        p=torch.randn(2,4,2560);d=torch.randint(0,2,(2,4,2048)).float()
        model.eval();model.set_corrections_enabled(True);model.set_shrinkage_learnable(True)
        with torch.inference_mode():
            a,aux=model(p,d);b=torch.cat([model(p[i:i+1],d[i:i+1])[0] for i in range(2)])
        self.assertTrue(torch.isfinite(a).all());self.assertLess(float((a-b).abs().max()),1e-5)
        self.assertAlmostEqual(float(aux['drug_gate'][0]),.4,places=5);self.assertAlmostEqual(float(aux['joint_gate'][0]),.05,places=5)

    def test_checkpoint_guard(self):
        with self.assertRaises(RuntimeError):c.checkpoint_valid(dict(checkpoint_type='mgca_final_2773_full_refit_final_state'),43)

    def test_lru(self):
        from engine import LRU
        cache=LRU(2);cache.put('a',1);cache.put('b',2);cache.get('a');cache.put('c',3)
        self.assertIsNone(cache.get('b'));self.assertEqual(cache.get('a'),1)

    def test_complete_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c.atomic_text(p/'x','data')
            c.atomic_json(p/'manifest.json',dict(files={'x':c.sha(p/'x')}));c.atomic_text(p/'.complete',c.sha(p/'manifest.json'))
            self.assertTrue(c.completed(p));c.atomic_text(p/'x','changed')
            with self.assertRaises(RuntimeError):c.completed(p)

    def test_train_resume_contract(self):
        import torch
        import train
        class Tiny(torch.nn.Module):
            def __init__(self,**kw):
                super().__init__();self.linear=torch.nn.Linear(2,1)
                self.drug_shrinkage=torch.nn.Module();self.drug_shrinkage.logit=torch.nn.Parameter(torch.tensor(-1.0))
                self.joint_shrinkage=torch.nn.Module();self.joint_shrinkage.logit=torch.nn.Parameter(torch.tensor(-2.0))
            def set_corrections_enabled(self,x):self.corrections=x
            def set_shrinkage_learnable(self,x):self.learn=x
            def forward(self,p,d):
                a=torch.nn.functional.softplus(self.drug_shrinkage.logit if self.learn else self.drug_shrinkage.logit.detach())
                b=torch.nn.functional.softplus(self.joint_shrinkage.logit if self.learn else self.joint_shrinkage.logit.detach())
                y=self.linear(torch.stack([p[:,0,0],d[:,0,0]*(a+b) if self.corrections else d[:,0,0]*0],1))
                return y,{}
            def loss_components(self,y,aux):return {'total':self.linear.weight.sum()*0}
        module=types.SimpleNamespace(FullRegressionTransformer=Tiny)
        original_save=c.save
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'resumed';baseline=root/'baseline'
            for folder in (out,baseline):
                folder.mkdir();c.atomic_text(folder/'cohort.csv','synthetic fixture, not a result\n')
                c.save(folder/'features.pt',dict(protein=torch.ones(1,4,2560).expand(5446,-1,-1),
                       drug=torch.ones(1,4,2048).expand(5446,-1,-1),labels=torch.zeros(5446)))
                c.atomic_json(folder/'features_manifest.json',dict(files={n:c.sha(folder/n) for n in ('cohort.csv','features.pt')}))
            batch=[(torch.ones(2,4,2560),torch.ones(2,4,2048),torch.zeros(2))]
            def fail_after_saved_epoch2(path,value):
                original_save(path,value)
                if Path(path).name=='last_state.pt' and value.get('epoch')==2:raise RuntimeError('simulated interruption')
            def args(folder):return types.SimpleNamespace(seed=43,output_root=folder,device='cpu',concurrency=1)
            with patch.object(c,'lock',side_effect=lambda _:nullcontext()),patch.object(c,'module',return_value=module),\
                 patch.object(c,'checkpoint_valid'),patch.object(train,'DataLoader',return_value=batch):
                with patch.object(c,'save',side_effect=fail_after_saved_epoch2):
                    with self.assertRaisesRegex(RuntimeError,'simulated interruption'):train.run(args(out))
                train.run(args(out));train.run(args(baseline))
                # Complete rerun skips and preserved hashes are unchanged.
                checkpoint=out/'checkpoints/seed_43'/c.checkpoint_name(43);before=c.sha(checkpoint)
                train.run(args(out));self.assertEqual(c.sha(checkpoint),before)
            a=c.load(checkpoint);b=c.load(baseline/'checkpoints/seed_43'/c.checkpoint_name(43))
            for name,value in a['model_state_dict'].items():self.assertTrue(torch.equal(value,b['model_state_dict'][name]))
            self.assertEqual(a['epochs'],55)

class FakeEngine:
    """Only HTTP lifecycle mocked; numeric model is separately tested above."""
    def __init__(self,*args):self.startup_seconds=.1
    def close(self):pass
    def predict(self,pairs,allow):
        if any(r['smiles']=='BAD' for r in pairs):raise ValueError('Invalid SMILES')
        return dict(results=[dict(sample_id=r['sample_id'] or 'x',predicted_pkoff=float(i)) for i,r in enumerate(pairs)])

class APITests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        import api
        self.env=patch.dict(os.environ,{'ESM2_PATH':'fake','API_KEY':'test-secret'});self.env.start()
        self.mock=patch('engine.Engine',FakeEngine);self.mock.start()
        self.client=TestClient(api.app);self.client.__enter__();self.api=api
        self.headers={'X-API-Key':'test-secret'}
    def tearDown(self):self.client.__exit__(None,None,None);self.mock.stop();self.env.stop()
    def test_ready_and_auth(self):
        self.assertEqual(self.client.get('/readyz').status_code,401)
        response=self.client.get('/readyz',headers=self.headers).json()
        self.assertEqual(response['status'],'ready');self.assertEqual(response['models'],1);self.assertEqual(response['seed'],43)
    def test_screen_ranking(self):
        r=self.client.post('/screen',headers=self.headers,json=dict(fasta='ACDE',smiles=['CCO','CCN'],top_k=1))
        self.assertEqual(r.status_code,200);self.assertEqual(r.json()['results'][0]['sample_id'],'compound_000002')
        self.assertEqual(r.json()['total_screened'],2)
    def test_invalid(self):
        for data in (dict(fasta='ACDE',smiles=[]),dict(fasta='ACDE',smiles=['A'*4097]),dict(fasta='ACDE',smiles=['BAD'])):
            self.assertEqual(self.client.post('/screen',headers=self.headers,json=data).status_code,422)
    def test_body_limit(self):
        r=self.client.post('/screen',headers=self.headers,content=b' '*8388609)
        self.assertEqual(r.status_code,413)
    def test_busy(self):
        with patch.object(self.api.app.state.busy,'locked',return_value=True):
            r=self.client.post('/screen',headers=self.headers,json=dict(fasta='ACDE',smiles=['CCO']))
            self.assertEqual(r.status_code,429)

if __name__=='__main__':unittest.main(verbosity=2)
