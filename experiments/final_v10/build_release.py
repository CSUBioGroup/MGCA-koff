"""Build a code-only ZIP; never include experimental tensors or results."""
from pathlib import Path
import zipfile
from common import HERE,atomic_json,digest


def main():
    allowed={'.py','.sh','.md','.json'}
    files=[p for p in HERE.rglob('*') if p.is_file() and p.suffix in allowed
           and not any(part in {'outputs','__pycache__','cache','preview'} for part in p.relative_to(HERE).parts)
           and p.name!='release_manifest.json']
    atomic_json(HERE/'release_manifest.json',dict(files={p.relative_to(HERE).as_posix():digest(p) for p in sorted(files)},
        original_v10_sha256=digest(HERE/'sources/original_v10_model.py'),
        changes=['softplus positive unbounded scalars','no cap-dependent gate prior',
                 'scalar coordinates excluded from AdamW decay','two initial coefficients jointly tuned',
                 'standalone immutable/replayable workflow; legacy feature batch size configurable']))
    files.append(HERE/'release_manifest.json')
    path=HERE.parent/'v10_unbounded_jointtune_code.zip'
    if path.exists(): raise FileExistsError('Refusing to overwrite '+str(path))
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files): z.write(p,(Path(HERE.name)/p.relative_to(HERE)).as_posix())
    with zipfile.ZipFile(path) as z:
        if z.testzip(): raise RuntimeError('Corrupt archive')
    print(path,'bytes=',path.stat().st_size)


if __name__=='__main__': main()
