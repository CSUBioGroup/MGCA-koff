"""Additive K4DD SMILES syntax fix; reuse verified refits, never launch training."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import os
import sys

import common_v10 as c
import workflow as w

HERE = Path(__file__).resolve().parent
BASE_RELEASE_SHA = '14f4eda6bf0ef058e8cf16b2d477cf9a0872c83f6c761b7fe81b63284bb2db93'
PATCH_ID = 'k4dd_bracket_hydrogen_charge_order_v1'
REPLACEMENTS = (('[N+H2]', '[NH2+]'), ('[N+H]', '[NH+]'))


def verify_package():
    w.verify_release()
    if c.sha256_file(HERE/'release_manifest.json') != BASE_RELEASE_SHA:
        raise RuntimeError('This additive patch requires the original frozen case release')
    manifest = c.read_json(HERE/'smiles_compat_release.json')
    if manifest['base_release_sha256'] != BASE_RELEASE_SHA:
        raise RuntimeError('Patch base identity mismatch')
    for name, digest in manifest['files'].items():
        if not (HERE/name).is_file() or c.sha256_file(HERE/name) != digest:
            raise RuntimeError('Compatibility patch changed: '+name)
    return c.sha256_file(HERE/'smiles_compat_release.json')


def normalized_rows(source):
    """Only reorder two explicit bracket tokens; no neutralization or row dropping."""
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import rdFingerprintGenerator
    with source.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    key = next(k for k in fields if k.lower() == 'smiles')
    generators = [rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=2048) for r in range(4)]
    changes = []
    normalized = []
    for index, row in enumerate(rows, 1):
        raw = row[key]
        value = raw
        counts = {old: value.count(old) for old, new in REPLACEMENTS}
        for old, new in REPLACEMENTS:
            value = value.replace(old, new)
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise RuntimeError('Unresolved invalid SMILES at CSV data row '+str(index))
        fingerprints = [g.GetFingerprint(mol) for g in generators]
        if any(fp.GetNumBits() != 2048 for fp in fingerprints):
            raise RuntimeError('Invalid Morgan shape at row '+str(index))
        if raw != value:
            # If a runtime accepts the nonstandard original, prove that parsing
            # the normalized spelling yields exactly the same canonical molecule
            # and every Morgan channel. Invalid originals cannot be compared.
            with rdBase.BlockLogs():
                original = Chem.MolFromSmiles(raw)
            equivalent = None
            if original is not None:
                equivalent = (Chem.MolToSmiles(original) == Chem.MolToSmiles(mol)
                              and all(DataStructs.TanimotoSimilarity(g.GetFingerprint(original), fp) == 1.
                                      for g, fp in zip(generators, fingerprints)))
                if not equivalent:
                    raise RuntimeError('Normalization changed a parseable molecular graph at row '+str(index))
            changes.append({'csv_data_row_1based': index, 'sample_id': f'case_{index:05d}',
                'target_name': row.get('target_name',''), 'raw_smiles': raw, 'normalized_smiles': value,
                'canonical_smiles': Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
                'n_NplusH': counts['[N+H]'], 'n_NplusH2': counts['[N+H2]'],
                'original_parseable': original is not None, 'equivalence_when_original_parseable': equivalent})
        item = dict(row);item[key] = value
        if any(item[k] != row[k] for k in fields if k != key):
            raise RuntimeError('Non-SMILES data changed')
        normalized.append(item)
    return normalized, changes, rdBase.rdkitVersion


def prepare_input(out, patch_hash):
    source = HERE/'inputs/k4dd.csv'
    rows, changes, version = normalized_rows(source)
    if len(rows) != 487 or len(changes) != 18:
        raise RuntimeError('Unexpected K4DD normalization coverage; expected 487 rows / 18 syntax repairs')
    folder = out/'compat_smiles_v1'
    normalized = folder/'k4dd.normalized.csv'
    audit_path = folder/'audit.json'
    identity = {'patch_id': PATCH_ID, 'patch_release_sha256': patch_hash,
                'base_release_sha256': BASE_RELEASE_SHA, 'source_sha256': c.sha256_file(source)}
    if audit_path.exists():
        audit = c.read_json(audit_path)
        if audit['identity'] != identity:
            raise RuntimeError('Compatibility audit belongs to different code/input')
        for name, digest in audit['files'].items():
            if not (folder/name).is_file() or c.sha256_file(folder/name) != digest:
                raise RuntimeError('Compatibility artifact is missing/corrupt: '+name)
        return normalized, audit_path
    folder.mkdir(parents=True, exist_ok=True)
    c.atomic_csv(normalized, rows)
    c.atomic_csv(folder/'smiles_changes.csv', changes)
    audit = {'identity': identity, 'source_csv': str(source.resolve()), 'row_count': len(rows),
        'modified_rows': len(changes), 'modified_rows_by_target': dict(Counter(r['target_name'] for r in changes)),
        'token_mapping': dict(REPLACEMENTS), 'remaining_invalid_rows': 0, 'rdkit_version': version,
        'all_four_morgan_channels_validated': True, 'original_csv_modified': False,
        'training_data_or_model_modified': False, 'rows_dropped': 0,
        'order_labels_sequences_other_columns_preserved': True,
        'scope': 'Explicit H-count/positive-charge token order only. No neutralization, tautomerization or label-driven repair.',
        'files': {p.name: c.sha256_file(p) for p in [normalized, folder/'smiles_changes.csv']}}
    c.atomic_json(audit_path, audit)
    print('K4DD syntax audit: 487 rows retained; 18 repaired; all four Morgan channels valid.', flush=True)
    return normalized, audit_path


def summary_addendum(out, audit_path):
    final = out/'summary/final_audit.json'
    payload = c.read_json(final)
    payload['case_input_compatibility'] = {
        'patch_id': PATCH_ID, 'audit_path': str(audit_path.resolve()),
        'audit_sha256': c.sha256_file(audit_path), 'modified_rows': 18,
        'rows_retained': 487, 'retrained_checkpoints': 0,
        'original_frozen_model_and_training_unchanged': True}
    c.atomic_json(final, payload)
    report = out/'summary/final_report.md'
    text = report.read_text(encoding='utf-8')
    text += ('\n## K4DD input syntax compatibility\n\n'
        'All 487 retained rows were processed. In 18 rows, `[N+H]` / `[N+H2]` were rewritten as '
        '`[NH+]` / `[NH2+]` for RDKit parsing. No labels, sequences, other columns, '
        'row order, model settings, training cohort or checkpoints were changed. '
        'The filtered case-panel CSV is preserved. See `compat_smiles_v1/audit.json` and '
        '`smiles_changes.csv` for per-row provenance.\n')
    c.atomic_text(report, text)
    c.atomic_text(out/'summary/.complete', c.sha256_file(final)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['all','audit','k4dd','factor_xa','dpp4','plots','structure','summary'], default='all')
    args = parser.parse_args()
    patch_hash = verify_package()
    out = Path(os.environ.get('OUTPUT_ROOT', HERE.parent/'case_study_outputs/mgca_final_unbounded_2773_v1')).resolve()
    esm = Path(os.environ.get('ESM2_PATH', HERE.parent.parent/'pretrained_model/esm2_t36')).resolve()
    device = os.environ.get('DEVICE','cuda:0')
    if not (out/'case_plan.json').is_file():
        raise RuntimeError('This is a resume-only patch; no existing case_plan.json')
    with w.lock_output(out/'.workflow.lock'):
        expected = {'protocol': c.PROTOCOL, 'package': BASE_RELEASE_SHA,
            'refit_config': c.sha256_file(c.FROZEN_PATH),
            'esm_expected': c.read_json(HERE/'frozen/esm_identity.json')}
        if c.read_json(out/'case_plan.json') != expected:
            raise RuntimeError('Existing results do not match the original frozen case plan')
        for stage in ['cohort','checkpoints','four_target']:
            if not w.stage_complete(out, stage):raise RuntimeError('Required completed stage missing: '+stage)
        records = c.checkpoint_records(out/'checkpoints')
        import torch
        data_hash = c.sha256_file(w.inputs(out)[0])
        for record in records:
            c.validate_checkpoint(c.torch_load(torch,record['path'],map_location='cpu'),record,data_hash)
        before = {r['seed']: r['sha256'] for r in records}
        normalized, audit_path = prepare_input(out, patch_hash)
        # Redirect only K4DD input paths, not model/training code or checkpoint identity.
        original_call = w.call
        def call(output, label, script, arguments, python=None):
            converted = [normalized if str(x)==str(HERE/'inputs/k4dd.csv') else x for x in arguments]
            return original_call(output,label,script,converted,python=python)
        w.call = call
        if w.stage_complete(out,'k4dd'):
            existing = c.read_json(out/'predictions/k4dd/manifest.json')
            if existing['case_csv_sha256'] != c.sha256_file(normalized):
                raise RuntimeError('A complete K4DD panel exists from different input; refusing to overwrite it')
        phases = ['k4dd','factor_xa','dpp4','plots','structure','summary'] if args.phase=='all' else [args.phase]
        if any(p in phases for p in ['k4dd','factor_xa','dpp4']):w.preflight(out,esm,device)
        print('Reusing all 5 verified checkpoints and the completed four-target panel. NO TRAINING.',flush=True)
        for phase in phases:
            if phase=='audit':continue
            if phase=='k4dd':w.panel(out,esm,device,'k4dd')
            elif phase in ['factor_xa','dpp4']:w.occlude(out,esm,device,phase)
            elif phase=='plots':w.plots(out)
            elif phase=='structure':w.structure(out)
            elif phase=='summary':
                w.summary(out)
                summary_addendum(out,audit_path)
        after = {r['seed']:r['sha256'] for r in c.checkpoint_records(out/'checkpoints')}
        if before != after:raise RuntimeError('Checkpoint hashes changed during case-only resume')
        print('Completed requested compatibility phase:',args.phase,'; checkpoint hashes unchanged.',flush=True)


if __name__=='__main__':main()
