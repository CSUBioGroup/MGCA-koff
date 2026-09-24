"""Optional real PyTorch test; AST extracts ORIGINAL legacy encoder classes only.

Does not fake encoder/model layers and does not require downloading ESM2 or importing
unused transformers/sklearn plotting dependencies. Not a full data/GPU integration test.
"""
import ast
from pathlib import Path
import types
import unittest
from common import HERE, RUNTIME, configurations


def extract(path, names, namespace):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    selected=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    if {n.name for n in selected}!=set(names):raise AssertionError('Missing original definitions')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)


class TorchContracts(unittest.TestCase):
    def test_all_settings_forward_loss_and_parameter_count(self):
        import math
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        torch.set_num_threads(1)
        legacy=dict(torch=torch,nn=nn,F=F,math=math)
        extract(HERE/'runtime/local/ESM_Morgan_Hybrid_Fusion.py',('MultiExpertEncoder','GatedExpertFusion'),legacy)
        ns=dict(torch=torch,nn=nn,F=F,math=math,legacy=types.SimpleNamespace(**legacy))
        tree=ast.parse((RUNTIME/'ESM_Morgan_Hybrid_Fusion_cgrs_v10.py').read_text(encoding='utf-8-sig'))
        names=[n.name for n in tree.body if isinstance(n,ast.ClassDef)]+['parameter_count']
        # Constructor validation uses the source module's ABLATIONS constant.
        for n in tree.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ABLATIONS' for t in n.targets):
                ns['ABLATIONS']=ast.literal_eval(n.value)
        extract(RUNTIME/'ESM_Morgan_Hybrid_Fusion_cgrs_v10.py',names,ns)
        counts=[]
        for c in configurations(equal=True):
            torch.manual_seed(42)
            model=ns['FullRegressionTransformer'](**{k:c[k] for k in ('drug_gate_cap','joint_gate_cap','drug_gate_init','joint_gate_init')})
            counts.append(ns['parameter_count'](model));model.eval()
            model.set_corrections_enabled(True);model.set_shrinkage_learnable(True)
            pred,aux=model(torch.randn(2,4,2560),torch.randn(2,4,2048))
            for name in ('drug','joint'):
                self.assertAlmostEqual(float(aux[name+'_gate'][0,0].detach()),c[name+'_gate_init'],places=6)
            components=model.loss_components(torch.tensor([.5,1.5]),aux)
            loss=(pred.reshape(-1)-torch.tensor([.5,1.5])).square().mean()+components['total']
            self.assertTrue(torch.isfinite(loss));loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(set(counts),{13392395})


if __name__=='__main__':unittest.main(verbosity=2)
