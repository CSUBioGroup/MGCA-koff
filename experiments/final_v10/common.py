"""Dependency-light planning, provenance and atomic artifact helpers (Python 3.8+)."""
import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / 'runtime/mgca_hyperparameter_tuning/v10_unbounded'
METRICS = ('mse', 'rmse', 'mae', 'r2', 'pearson', 'spearman')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def uid(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    stream = io.StringIO(newline='')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


@contextlib.contextmanager
def file_lock(path, blocking=False):
    """OS lock, released on process death. File existence alone is not a lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                f.write(b'0'); f.flush()
            while True:
                try:
                    f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if not blocking:
                        raise RuntimeError('Another process holds lock: ' + str(path))
                    time.sleep(0.25)
        else:
            import fcntl
            try:
                fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except OSError as exc:
                raise RuntimeError('Another process holds lock: ' + str(path)) from exc
        yield
    finally:
        f.close()


def split_paths(root, dataset, protocol, fold):
    """Intentionally returns train/val only. Never enumerate/read test files."""
    root = Path(root)
    if dataset == 'KinetX':
        if protocol == 'warm':
            base, names = root/'KinetX/random_split_mgca_input', ('train_run%d.csv'%fold, 'val_run%d.csv'%fold)
        elif protocol == 'drug_cold':
            base, names = root/('KinetX/drug_cold_start_canonical_5fold/fold%d'%fold), ('train.csv', 'val.csv')
        else:
            base, names = root/'KinetX/cold_start', ('train.csv', 'val.csv')
    else:
        if protocol == 'protein_cold':
            base, names = root/'2773/new_folds/target-cold', ('train.csv', 'val.csv')
        else:
            base = root/'2773/new_folds'/('warm' if protocol == 'warm' else 'drug-cold')
            names = ('train_run%d.csv'%fold, 'val_run%d.csv'%fold)
    return tuple(base / n for n in names)


def verify_files(mapping, root=None):
    for name, expected in mapping.items():
        p = Path(name) if root is None else Path(root)/name
        if not p.is_file() or digest(p) != expected:
            raise RuntimeError('Missing/changed artifact: ' + str(p))


def scientific_sources():
    paths = list(HERE.glob('*.py')) + list((HERE/'runtime').rglob('*.py')) + list((HERE/'sources').rglob('*.py'))
    # Unit tests/documentation are not runtime dependencies.
    return {p.relative_to(HERE).as_posix(): digest(p) for p in sorted(paths) if not p.name.startswith('test_')}
