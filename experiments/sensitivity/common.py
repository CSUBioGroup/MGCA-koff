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
RUNTIME = HERE / 'runtime/mgca_hyperparameter_tuning/v10_cgrs'
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


def configurations(cap_multipliers=(.5, 1., 2.), init_multipliers=(.5, 1., 2.), equal=False):
    values = {}
    def add(dc, jc, di, ji, family):
        key = tuple(round(float(x), 10) for x in (dc, jc, di, ji))
        dc, jc, di, ji = key
        if not all(math.isfinite(x) for x in key) or not (0 < di < dc <= 1 and 0 < ji < jc <= 1):
            raise ValueError('Invalid caps/initialization: ' + str(key))
        if key not in values:
            values[key] = dict(config_id=uid(key)[:12], drug_gate_cap=dc,
                               joint_gate_cap=jc, drug_gate_init=di, joint_gate_init=ji,
                               families=[], reference=key == (.3, .12, .1, .03))
        values[key]['families'].append(family)
    for d in cap_multipliers:
        for j in cap_multipliers:
            add(.3*d, .12*j, .1, .03, 'caps')
    for d in init_multipliers:
        for j in init_multipliers:
            add(.3, .12, .1*d, .03*j, 'initialization')
    if equal:
        for cap in (.15, .3):
            add(cap, cap, .1, .03, 'equal_caps')
    if sum(x['reference'] for x in values.values()) != 1:
        raise ValueError('Both grids must include the original reference.')
    return sorted(values.values(), key=lambda x: (not x['reference'], x['config_id']))


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
    paths = list(HERE.glob('*.py')) + list((HERE/'runtime').rglob('*.py')) + list((HERE/'frozen').rglob('*.json'))
    # Unit tests/documentation are not runtime dependencies.
    return {p.relative_to(HERE).as_posix(): digest(p) for p in sorted(paths) if not p.name.startswith('test_')}


def verify_run(out, task_id):
    out = Path(out)
    seal = read_json(out/'verified.complete.json')
    if seal['task_id'] != task_id:
        raise RuntimeError('Run identity mismatch: ' + str(out))
    verify_files(seal['files'], out)
    m = read_json(out/'metrics.json')
    if m['test_accessed'] is not False or m['selection_only'] is not True or m['identity']['config_id'] != task_id:
        raise RuntimeError('Invalid validation-only result: ' + str(out))
    return m
