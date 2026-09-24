"""Verify completed refits and atomically produce checkpoint + API ZIP."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import zipfile
import core as c

def build(out,esm,include_esm=False):
    out=Path(out);c.verify_release()
    fm=c.read(out/'features_manifest.json');c.verify_files(out,fm['files'])
    seed=c.SEED;folder=out/('checkpoints/seed_%d'%seed);m=c.completed(folder)
    if not m or m['identity']['release']!=c.sha(c.ROOT/'release_manifest.json'):raise RuntimeError('Missing/stale seed 43')
    name=c.checkpoint_name(seed);payload=c.load(folder/name)
    c.checkpoint_valid(payload,seed,fm['files']['cohort.csv']);del payload
    records=[(seed,folder/name)]
    identity=dict(release=c.sha(c.ROOT/'release_manifest.json'),checkpoints=[c.sha(p) for _,p in records],include_esm=include_esm)
    # Immutable keyed output path: code/weights changes never overwrite an old bundle.
    import json
    bundle_id=c.key(json.dumps(identity,sort_keys=True))[:16]
    parent=out/'deploy'/bundle_id;bundle=parent/'mgca_kinetx_api';parent.mkdir(parents=True,exist_ok=True)
    final=parent/'mgca_kinetx_api.zip';seal=parent/'zip_manifest.json'
    if final.exists():
        if not seal.exists() or c.read(seal)['sha256']!=c.sha(final):raise RuntimeError('Existing ZIP unverified/corrupt')
        print('Verified existing deployment ZIP:',final);return final
    bundle.mkdir(parents=True,exist_ok=True)
    def copy(source,name):
        target=bundle/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    names=['api.py','engine.py','core.py','start_service.sh','service_config.sh','requirements_service.txt','client_screen.py','verify_api.py',
           'frozen/refit_config.json','frozen/esm_identity.json','frozen/KinetX_best_params.json','frozen/epoch_selection.json',
           'runtime/mgca_hyperparameter_tuning/v10_unbounded/model.py','runtime/local/ESM_Morgan_Hybrid_Fusion.py','examples/screen.json']
    for name in names:copy(c.ROOT/name,name)
    copy(c.ROOT/'SERVICE_README.md','README.md')
    copy(out/'protein_warm_cache.pt','protein_warm_cache.pt')
    for seed,path in records:copy(path,'checkpoints/'+path.name)
    # Known-feature predictions from the newly trained checkpoint, for server smoke checking.
    import torch
    features=c.load(out/'features.pt');rows=c.cohort();module=c.module();values=[]
    for seed,path in records:
        model=module.FullRegressionTransformer(**c.kwargs());model.load_state_dict(c.load(path)['model_state_dict'],strict=True)
        model.eval();model.set_corrections_enabled(True);model.set_shrinkage_learnable(True)
        with torch.inference_mode():values.append(model(features['protein'][:2],features['drug'][:2])[0].reshape(-1).tolist())
        del model
    c.atomic_json(bundle/'examples/verification.json',dict(pairs=[dict(fasta=r['FASTA'],smiles=r['SMILES'],sample_id='verify_%d'%i) for i,r in enumerate(rows[:2])],allow_truncation=True,
                  expected_prediction=values[0]))
    if include_esm:
        c.verify_esm(esm)
        for name in c.read(c.ROOT/'frozen/esm_identity.json'):copy(Path(esm)/name,'esm2_t36/'+name)
    files={p.relative_to(bundle).as_posix():c.sha(p) for p in bundle.rglob('*') if p.is_file() and p.name not in ('bundle_manifest.json','service_config.sh')}
    c.atomic_json(bundle/'bundle_manifest.json',dict(bundle_id=bundle_id,frozen=c.frozen(),files=files,data_sha256=fm['files']['cohort.csv'],
                  seed=c.SEED,include_esm=include_esm,editable_paths='service_config.sh',source_release=identity['release']))
    temp=final.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,arcname='mgca_kinetx_api/'+p.relative_to(bundle).as_posix())
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP integrity failed')
    os.replace(temp,final);c.atomic_json(seal,dict(identity=identity,sha256=c.sha(final),bytes=final.stat().st_size))
    print('DEPLOYMENT ZIP READY:',final,flush=True);return final
