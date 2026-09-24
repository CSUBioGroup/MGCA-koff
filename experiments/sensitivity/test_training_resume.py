"""Synthetic-feature integration of the actual trainer, with crash injection.

No ESM2/download/GPU: input preparation and metrics/environment providers are test
fixtures. Actual v10 forward/loss/optimizer/checkpoint/RNG/early-stop/export code runs.
"""
import argparse
import ast
import contextlib
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time
import types
import unittest
import numpy as np
from common import HERE, RUNTIME, atomic_text, read_json
from test_model_contract import extract


class InterruptedAfterCheckpoint(Exception):pass


class ResumeIntegration(unittest.TestCase):
    def test_crash_at_early_stop_does_not_add_epoch_and_keeps_best(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
        torch.set_num_threads(1)
        utilspec=importlib.util.spec_from_file_location('sensitivity_test_utils',RUNTIME/'utils.py')
        utils=importlib.util.module_from_spec(utilspec);utilspec.loader.exec_module(utils)
        legacy=dict(torch=torch,nn=nn,F=F,math=math,Dataset=Dataset,csv=csv,List=list,Tuple=tuple)
        extract(HERE/'runtime/local/ESM_Morgan_Hybrid_Fusion.py',
                ('MultiExpertEncoder','GatedExpertFusion','ESM2MorganDataset','_pick_column','read_labeled_rows'),legacy)
        modelns=dict(torch=torch,nn=nn,F=F,math=math,legacy=types.SimpleNamespace(**legacy))
        model_path=RUNTIME/'ESM_Morgan_Hybrid_Fusion_cgrs_v10.py'
        tree=ast.parse(model_path.read_text(encoding='utf-8-sig'))
        for n in tree.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ABLATIONS' for t in n.targets):modelns['ABLATIONS']=ast.literal_eval(n.value)
        extract(model_path,[n.name for n in tree.body if isinstance(n,ast.ClassDef)]+['parameter_count'],modelns)
        def metrics(y,p):
            mse=float(np.mean((y-p)**2));mae=float(np.mean(np.abs(y-p)))
            return dict(mse=mse,rmse=mse**.5,mae=mae,r2=0.,pearson=float('nan'),spearman=float('nan'))
        mgca=types.SimpleNamespace(**{k:v for k,v in modelns.items() if not k.startswith('__')})
        mgca.__file__=str(model_path);mgca.MODEL_VARIANT='synthetic_fixture_actual_v10';mgca.MAX_GRAD_NORM=5.
        mgca.LEGACY_SCRIPT=HERE/'runtime/local/ESM_Morgan_Hybrid_Fusion.py'
        for version,dirname,filename in ((4,'','ESM_Morgan_Hybrid_Fusion_nonredundant.py'),(5,'v5_uar','ESM_Morgan_Hybrid_Fusion_uar_v5.py'),
            (6,'v6_crrf','ESM_Morgan_Hybrid_Fusion_crrf_v6.py'),(7,'v7_padg','ESM_Morgan_Hybrid_Fusion_padg_v7.py'),
            (8,'v8_pajg','ESM_Morgan_Hybrid_Fusion_pajg_v8.py'),(9,'v9_scrg','ESM_Morgan_Hybrid_Fusion_scrg_v9.py')):
            setattr(mgca,'SOURCE_V%d_SCRIPT'%version,HERE/'runtime/mgca_hyperparameter_tuning'/dirname/filename)
        mgca.compute_metrics=metrics;mgca.read_labeled_rows=legacy['read_labeled_rows']
        ns=dict(argparse=argparse,contextlib=contextlib,hashlib=hashlib,json=json,math=math,os=os,Path=Path,
                random=random,re=re,sys=sys,time=time,np=np,torch=torch,DataLoader=DataLoader,mgca=mgca,
                SCRIPT_DIR=RUNTIME,__file__=str(RUNTIME/'train_v10_cgrs.py'))
        for name in ('atomic_npz','atomic_json','atomic_text','atomic_torch_save','canonical_id','sha256_file','write_csv'):
            ns[name]=getattr(utils,name)
        ns['environment_snapshot']=lambda *a:{'synthetic_feature_test':True}
        source=ast.parse((RUNTIME/'train_v10_cgrs.py').read_text(encoding='utf-8-sig'))
        for node in source.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CGRS_ARCHIVE_KEYS' for t in node.targets):ns['CGRS_ARCHIVE_KEYS']=ast.literal_eval(node.value)
        extract(RUNTIME/'train_v10_cgrs.py',[n.name for n in source.body if isinstance(n,ast.FunctionDef)],ns)
        original_load=torch.load
        def load(*a,**kw):
            if 'weights_only' in inspect.signature(original_load).parameters:kw.setdefault('weights_only',False)
            return original_load(*a,**kw)
        oldargv=sys.argv[:]
        try:
            torch.load=load
            with tempfile.TemporaryDirectory() as td:
                root=Path(td)
                for part in ('train','val'):atomic_text(root/(part+'.csv'),'FASTA,SMILES,pkoff\nAAAA,CC,0.5\nBBBB,CO,1.5\n')
                esm=root/'esm';esm.mkdir()
                generator=torch.Generator().manual_seed(17)
                proteins=torch.randn(2,4,2560,generator=generator);drugs=torch.randn(2,4,2048,generator=generator)
                targets=torch.tensor([.5,1.5]);cache=root/'features.pt';torch.save(proteins,cache)
                dataset=legacy['ESM2MorganDataset'](drugs,proteins,targets)
                def prepare(args,device):
                    rows=[legacy['read_labeled_rows'](str(root/(p+'.csv'))) for p in ('train','val')]
                    return [dataset,dataset],rows,dict(esm2=cache,morgan=cache,esm2_hit_before=True,morgan_hit_before=True)
                ns['prepare_data']=prepare
                evaluate=ns['evaluate']
                def constant_eval(*a,**kw):
                    y,p,aux,dt=evaluate(*a,**kw)
                    return y,np.zeros_like(p),aux,dt
                ns['evaluate']=constant_eval
                def argv(out,resume):
                    return ['trainer','--train-csv',str(root/'train.csv'),'--val-csv',str(root/'val.csv'),'--selection-only',
                        '--output-dir',str(out),'--dataset','2773','--esm2-path',str(esm),'--device','cpu',
                        '--lr','0.00002','--weight-decay','0.0001','--batch-size','2','--dropout','0.15','--window-size','6',
                        '--epochs','14','--patience','1','--save-resume-state',resume]
                baseline=root/'baseline';sys.argv=argv(baseline,'false');ns['main']()
                save=ns['atomic_torch_save']
                for crash_epoch in (7,12):
                    resumed=root/('resumed%d'%crash_epoch);sys.argv=argv(resumed,'true')
                    def crash_save(torchmodule,path,payload):
                        # Persist the crash point only, avoiding redundant test-only I/O.
                        if Path(path).name=='last_state.pt':
                            if payload['epoch']!=crash_epoch:return
                            save(torchmodule,path,payload)
                            raise InterruptedAfterCheckpoint()
                        save(torchmodule,path,payload)
                    ns['atomic_torch_save']=crash_save
                    with self.assertRaises(InterruptedAfterCheckpoint):ns['main']()
                    self.assertTrue((resumed/'last_state.pt').exists())
                    ns['atomic_torch_save']=save;ns['main']()
                    a=read_json(baseline/'metrics.json');b=read_json(resumed/'metrics.json')
                    self.assertEqual((a['best_epoch'],a['epochs_ran']),(11,12))
                    self.assertEqual((b['best_epoch'],b['epochs_ran']),(11,12))
                    self.assertIsNone(b['val_metrics']['pearson'])
                    self.assertEqual((baseline/'validation_predictions.csv').read_bytes(),(resumed/'validation_predictions.csv').read_bytes())
                    astate=load(baseline/'best_model.pt',map_location='cpu')['model_state_dict']
                    bstate=load(resumed/'best_model.pt',map_location='cpu')['model_state_dict']
                    self.assertTrue(all(torch.equal(astate[k],bstate[k]) for k in astate))
                    self.assertFalse((resumed/'last_state.pt').exists())
        finally:
            torch.load=original_load;sys.argv=oldargv


if __name__=='__main__':unittest.main(verbosity=2)
