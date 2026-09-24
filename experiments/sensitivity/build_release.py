"""Package this standalone supplement implementation, NEVER experiment outputs."""
from pathlib import Path
import zipfile
from common import HERE, atomic_json, digest


def main():
    provenance={}
    project=HERE.parents[1]
    for p in sorted((HERE/'runtime').rglob('*.py')):
        relative=p.relative_to(HERE/'runtime')
        original=project/relative
        provenance[relative.as_posix()]=dict(snapshot_sha256=digest(p),
            original_sha256=digest(original) if original.is_file() else None,
            changed_in_supplement=original.is_file() and digest(original)!=digest(p))
    atomic_json(HERE/'source_provenance.json',dict(files=provenance,
        modifications=['trainer: CPU RNG restoration; respect completed early stop on resume; null undefined correlations; cumulative epoch timing',
                       'legacy feature utility: atomic ESM cache rename only'],
        original_model_and_results_untouched=True))
    allowed={'.py','.sh','.md','.json'}
    files=[p for p in HERE.rglob('*') if p.is_file() and p.suffix in allowed
           and not any(part in {'outputs','__pycache__','cache','preview'} for part in p.relative_to(HERE).parts)
           and p.name not in {'release_manifest.json'}]
    atomic_json(HERE/'release_manifest.json',dict(files={p.relative_to(HERE).as_posix():digest(p) for p in sorted(files)}))
    files.append(HERE/'release_manifest.json')
    destination=HERE.parent/'v10_sensitivity_code.zip'
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an existing release: '+str(destination))
    with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,(Path(HERE.name)/p.relative_to(HERE)).as_posix())
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:raise RuntimeError('Corrupt release archive')
    print(destination,'bytes=',destination.stat().st_size)


if __name__=='__main__':main()
