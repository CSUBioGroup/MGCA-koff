"""Real PyTorch layers, AST-loaded to avoid ESM2 downloads/plotting imports."""
import ast
import math
from pathlib import Path
import types
import unittest
from common import HERE,RUNTIME


def extract(path,names,namespace):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    selected=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    if {n.name for n in selected}!=set(names): raise AssertionError('Missing definitions')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)


def model_namespace():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(1)
    legacy=dict(torch=torch,nn=nn,F=F,math=math)
    extract(HERE/'runtime/local/ESM_Morgan_Hybrid_Fusion.py',('MultiExpertEncoder','GatedExpertFusion'),legacy)
    ns=dict(torch=torch,nn=nn,F=F,math=math,legacy=types.SimpleNamespace(**legacy))
    tree=ast.parse((RUNTIME/'model.py').read_text(encoding='utf-8'))
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ABLATIONS' for t in n.targets): ns['ABLATIONS']=ast.literal_eval(n.value)
    extract(RUNTIME/'model.py',[n.name for n in tree.body if isinstance(n,ast.ClassDef)]+['parameter_count'],ns)
    return ns


class ModelContracts(unittest.TestCase):
    def test_positive_unbounded_initialization_and_gradients(self):
        ns=model_namespace(); torch=ns['torch']; Gate=ns['GlobalShrinkageGate']
        for initial in (1e-7,.015,.03,.05,.1,.2,1.,2.,100.):
            gate=Gate(initial); out,_=gate(2,torch.ones(2,1))
            self.assertTrue(torch.allclose(out,torch.full_like(out,initial),rtol=1e-5,atol=1e-8))
            out.sum().backward(); self.assertGreater(float(gate.logit.grad),0)
        for bad in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): Gate(bad)

    def test_all_initializations_and_branch_ablation_parameter_match(self):
        ns=model_namespace(); torch=ns['torch']; counts=set()
        for di,ji in ((d,j) for d in (.05,.1,.2) for j in (.015,.03,.06)):
            for variant in ('no','protein_anchor','drug_correction','joint_interaction'):
                torch.manual_seed(42)
                m=ns['FullRegressionTransformer'](drug_gate_init=di,joint_gate_init=ji,ablation=variant)
                counts.add(ns['parameter_count'](m)); m.eval()
                y,aux=m(torch.randn(2,4,2560),torch.randn(2,4,2048)); loss=m.loss_components(torch.tensor([.5,1.5]),aux)
                self.assertEqual(float(loss['gate_prior']),0.)
                self.assertAlmostEqual(float(aux['drug_gate'][0,0].detach()),di,places=6)
                self.assertAlmostEqual(float(aux['joint_gate'][0,0].detach()),ji,places=6)
                if variant=='drug_correction': self.assertEqual(float(aux['effective_drug_gate'].sum()),0.)
                if variant=='joint_interaction': self.assertEqual(float(aux['effective_joint_gate'].sum()),0.)
                (y.square().mean()+loss['total']).backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertEqual(counts,{13392395})

    def test_no_caps_or_prior_options_in_runtime(self):
        for name in ('model.py','trainer.py'):
            text=(RUNTIME/name).read_text(encoding='utf-8')
            self.assertNotIn('drug_gate_cap',text); self.assertNotIn('joint_gate_cap',text)
            self.assertNotIn('gate_prior_weight',text)


if __name__=='__main__': unittest.main(verbosity=2)
