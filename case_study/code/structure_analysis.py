"""Reuse frozen docking geometry only; regenerate all model-dependent structure analysis."""
import argparse,csv,importlib.util,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import common_v10 as c

def load_plot():
    p=c.PROJECT_ROOT/'case_study/plot_factor_xa_stage4.py'
    spec=importlib.util.spec_from_file_location('final_structure_plot',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def fallback_3d(folder,geometry):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    receptor=next(folder.glob('*scored*.pdb'),None)
    if receptor is None:
        candidates=[p for p in folder.glob('*.pdb')]
        if len(candidates)!=1:raise RuntimeError('Cannot identify generated scored receptor')
        receptor=candidates[0]
    coords=[];scores=[]
    for line in receptor.read_text().splitlines():
        if line.startswith('ATOM') and line[12:16].strip()=='CA':
            coords.append([float(line[30:38]),float(line[38:46]),float(line[46:54])]);scores.append(float(line[60:66]))
    coords=np.asarray(coords)
    sdf=(geometry/'analysis/selected_factor_xa_05_pose.sdf').read_text().splitlines()
    n=int(sdf[3][:3]);lig=np.array([[float(line[i:i+10]) for i in [0,10,20]] for line in sdf[4:4+n]])
    fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(111,projection='3d')
    vmax=max(float(np.max(np.abs(scores))),1e-8)
    points=ax.scatter(*coords.T,c=scores,cmap='coolwarm',vmin=-vmax,vmax=vmax,s=14,alpha=.85)
    ax.scatter(*lig.T,color='#303030',s=28,label='Pinned docked ligand heavy atoms')
    ax.set(xlabel='x (A)',ylabel='y (A)',zlabel='z (A)',title='MGCA final: Factor Xa post-hoc occlusion\nC-alpha view; not native attention')
    ax.legend(loc='upper left',fontsize=8);fig.colorbar(points,ax=ax,shrink=.65,label='Scaled occlusion sensitivity (clipped to +/-99; not physical B-factor)')
    for ext in ['png','svg']:fig.savefig(folder/f'factor_xa_structure_fallback.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
    root=a.output_root;geom=c.PROJECT_ROOT/'geometry';stage2=root/'occlusion/factor_xa'
    manifest=c.read_json(stage2/'factor_xa_stage2_manifest.json')
    if manifest['model_sha256']!=c.EXPECTED_MODEL_SHA256 or manifest['epochs']!=c.DEFAULT_EPOCHS:
        raise RuntimeError('Structure analysis cannot consume old-model occlusion')
    selection=c.read_json(stage2/'dual_occlusion_selection.json')
    prep=c.read_json(geom/'prepared/preparation_manifest.json')
    compound=selection['representative_compound']
    if compound['compound_id']!='factor_xa_05' or compound['canonical_smiles']!=prep['source_stage2_smiles']:
        raise RuntimeError('Frozen docking ligand differs from current representative compound')
    if not c.read_json(geom/'redocking/redocking_gate.json')['passed']:
        raise RuntimeError('Pinned redocking qualification failed')
    module=load_plot()
    # Resolve current stage-2 directly, avoiding symlinks and any old analysis path.
    original_require=module.require_file
    def require(path):
        if path.name=='protein_window_importance_summary.csv':return original_require(stage2/path.name)
        return original_require(path)
    module.require_file=require
    dest=root/'structure';dest.mkdir(parents=True,exist_ok=True)
    renderer=os.environ.get('STRUCTURE_RENDERER','auto')
    if renderer not in ['auto','matplotlib','pymol']:raise ValueError('STRUCTURE_RENDERER: auto|matplotlib|pymol')
    python=os.environ.get('STRUCTURE_PYTHON_BIN',sys.executable)
    available=False
    if renderer!='matplotlib':
        try:available=subprocess.run([python,'-c','import pymol._cmd'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
        except FileNotFoundError:available=False
    if renderer=='pymol' and not available:raise RuntimeError('Requested PyMOL unavailable; configure STRUCTURE_PYTHON_BIN')
    sys.argv=['structure','--model','2773','--stage2-root',str(root/'occlusion'),'--stage3-root',str(geom),'--output-dir',str(dest)]
    if available:
        # Keep the established composite/reference figure pipeline, but permit
        # a separate PyMOL environment without changing the scientific runtime.
        module.locate_pymol=lambda ignored: python
        def render(command,script,output):
            subprocess.run([python,'-m','pymol','-cq',str(script)],check=True)
            if not output.is_file():raise RuntimeError('PyMOL did not create '+str(output))
        module.render_structure=render
    else:sys.argv+=['--skip-structure']
    module.main()
    if not available:
        fallback_3d(dest,geom)
        module.combine_panels(dest/'factor_xa_structure_fallback.png',
            dest/'factor_xa_sequence_occlusion_2773.png',dest/'factor_xa_fallback_composite',
            '2773','A  C-alpha fallback (not PyMOL rendering)')
    c.atomic_json(dest/'final_structure_audit.json',{
        'protocol':c.PROTOCOL,'current_occlusion_manifest_sha256':c.sha256_file(stage2/'factor_xa_stage2_manifest.json'),
        'geometry_origin':'Pinned historical docking/redocking, independent of MGCA parameters. Docking not rerun.',
        'old_model_occlusion_reused':False,'contact_and_residue_scoring_recomputed':True,
        'renderer':'pymol' if available else 'matplotlib_Calpha_fallback',
        'interpretation':'Perturbation sensitivity mapped onto reused geometry, not physical interaction proof',
        'files':{p.name:c.sha256_file(p) for p in dest.iterdir() if p.is_file() and p.name!='final_structure_audit.json'}})

if __name__=='__main__':main()
