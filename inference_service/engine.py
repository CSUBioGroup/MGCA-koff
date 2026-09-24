"""Resident seed-43 MGCA and ESM2; bounded CPU LRU feature caches."""
from __future__ import annotations
from collections import OrderedDict
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
import core as c

class LRU:
    def __init__(self,limit):
        if limit<1:raise ValueError('Cache capacity must be positive')
        self.limit=limit;self.data=OrderedDict()
    def get(self,k):
        if k not in self.data:return None
        self.data.move_to_end(k);return self.data[k]
    def put(self,k,v):
        self.data[k]=v;self.data.move_to_end(k)
        while len(self.data)>self.limit:self.data.popitem(last=False)

class Engine:
    def __init__(self,bundle,esm,device):
        started=time.perf_counter();self.bundle=Path(bundle);self.device=c.configure_device(device)
        self.manifest=c.read(self.bundle/'bundle_manifest.json')
        c.verify_files(self.bundle,self.manifest['files'])
        if self.manifest['frozen']!=c.frozen():raise RuntimeError('Bundle config mismatch')
        c.verify_esm(esm);self.module=c.module();self.legacy=self.module.legacy
        self.batch=int(os.environ.get('INFER_BATCH_SIZE','64'))
        if not 1<=self.batch<=4096:raise ValueError('INFER_BATCH_SIZE must be 1..4096')
        self.pcache=LRU(int(os.environ.get('MAX_PROTEIN_CACHE','1024')))
        self.dcache=LRU(int(os.environ.get('MAX_DRUG_CACHE','10000')))
        seed=c.SEED
        payload=c.load(self.bundle/'checkpoints'/c.checkpoint_name(seed));c.checkpoint_valid(payload,seed,self.manifest['data_sha256'])
        self.model=self.module.FullRegressionTransformer(**c.kwargs()).to(self.device)
        self.model.load_state_dict(payload['model_state_dict'],strict=True);self.model.eval()
        self.model.set_corrections_enabled(True);self.model.set_shrinkage_learnable(True);del payload
        self.tokenizer=self.legacy.AutoTokenizer.from_pretrained(str(esm),local_files_only=True)
        self.esm=self.legacy.AutoModelForMaskedLM.from_pretrained(str(esm),local_files_only=True).to(self.device).eval()
        warm=c.load(self.bundle/'protein_warm_cache.pt')
        if warm['esm_identity']!=c.read(c.ROOT/'frozen/esm_identity.json') or warm['window_size']!=8 or warm['window_layout']!='even_span_v2':raise RuntimeError('Wrong warm-cache feature identity')
        if tuple(warm['features'].shape)!=(len(warm['sequences']),4,2560):raise RuntimeError('Invalid warm cache shape')
        if not torch.isfinite(warm['features']).all():raise RuntimeError('Invalid warm cache')
        for seq,feat in zip(warm['sequences'],warm['features']):self.pcache.put(seq,feat.float())
        # Genuine ESM forward on a short valid sequence warms kernels without
        # recomputing all training proteins. Inference head warms batch 1 and B.
        warm_seq='ACDEFGHIKLMNPQRSTVWY'*4
        feature=self.protein(warm_seq)
        drug=self.drug('CCO')
        with torch.inference_mode():
            for n in (1,self.batch):
                pp=feature.unsqueeze(0).expand(n,-1,-1).to(self.device);dd=drug.unsqueeze(0).expand(n,-1,-1).to(self.device)
                y,_=self.model(pp,dd)
                if not torch.isfinite(y).all():raise RuntimeError('Warmup produced non-finite prediction')
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        self.startup_seconds=time.perf_counter()-started
        print('READY: ESM2 + seed-43 KinetX checkpoint resident; warm protein cache=%d; startup=%.2fs'%(len(self.pcache.data),self.startup_seconds),flush=True)

    def protein(self,seq):
        value=self.pcache.get(seq)
        if value is not None:return value
        with torch.inference_mode():
            value=self.legacy.batch_extract_esm2([seq],self.tokenizer,self.esm,self.device,batch_size=1,window_size=8,window_layout='even_span_v2')[0].cpu().float()
        if value.shape!=(4,2560) or not torch.isfinite(value).all():raise RuntimeError('Invalid protein features')
        self.pcache.put(seq,value);return value

    def drug(self,smiles):
        value=self.dcache.get(smiles)
        if value is not None:return value
        mol=self.legacy.Chem.MolFromSmiles(smiles)
        if mol is None:raise ValueError('Invalid SMILES: '+smiles[:120])
        features=[]
        for radius in range(4):
            fp,valid=self.legacy.get_fingerprint(radius,[mol],device='cpu',fingerprint_type='morgan')
            if not valid.all():raise ValueError('Invalid Morgan channel')
            features.append(fp[0])
        value=torch.stack(features).float();self.dcache.put(smiles,value);return value

    def predict(self,pairs,allow_truncation=False):
        started=time.perf_counter();norm=[];seen=set()
        # Validate every input before expensive ESM work. Repeated pairs retained.
        for i,r in enumerate(pairs):
            seq=c.sequence(r['fasta']);smi=r['smiles'].strip();sid=r.get('sample_id') or 'sample_%06d'%(i+1)
            if sid in seen:raise ValueError('Duplicate sample_id: '+sid)
            seen.add(sid)
            if len(seq)>1022 and not allow_truncation:raise ValueError('Protein exceeds 1022 residues; explicitly allow_truncation to use frozen truncation policy')
            if not smi:raise ValueError('Empty SMILES')
            self.drug(smi)
            norm.append((sid,seq,smi))
        unique=list(dict.fromkeys(seq for _,seq,_ in norm))
        misses=sum(seq not in self.pcache.data for seq in unique)
        feature_started=time.perf_counter()
        # Keep current-request features even if larger than LRU capacity.
        features={seq:self.protein(seq) for seq in unique}
        feature_seconds=time.perf_counter()-feature_started;results=[];head_started=time.perf_counter()
        with torch.inference_mode():
            for start in range(0,len(norm),self.batch):
                chunk=norm[start:start+self.batch]
                p=torch.stack([features[seq] for _,seq,_ in chunk]).to(self.device)
                d=torch.stack([self.drug(sm) for _,_,sm in chunk]).to(self.device)
                predictions=self.model(p,d)[0].reshape(-1).cpu().double().numpy()
                if not np.isfinite(predictions).all():raise RuntimeError('Non-finite predictions')
                for (sid,seq,smi),value in zip(chunk,predictions):
                    value=float(value)
                    results.append(dict(sample_id=sid,smiles=smi,protein_sha256=c.key(seq),sequence_truncated=len(seq)>1022,
                                        seed=c.SEED,predicted_pkoff=value,
                                        predicted_koff_per_second=10.0**(-value) if -308<-value<308 else None))
        return dict(model='MGCA',dataset='KinetX_clean_5446_full_refit',epochs=c.frozen()['refit_epochs'],
                    config_id=c.frozen()['config_id'],seed=c.SEED,results=results,
                    timing=dict(total_seconds=time.perf_counter()-started,protein_encoding_seconds=feature_seconds,
                                head_seconds=time.perf_counter()-head_started,protein_cache_misses=misses),
                    uncertainty_note='A single checkpoint is used; no ensemble SD or calibrated uncertainty is reported.')

    def close(self):
        self.model=None;self.esm=None
        if self.device.type=='cuda':torch.cuda.empty_cache()
