#!/usr/bin/env python3
"""Factor Xa Stage-3 molecular docking and occlusion/contact comparison.

Protocol
--------
1. Download the 1NFU crystal structure and RRP chemical-component template.
2. Prepare a rigid Factor Xa receptor and the experimental RRP pose.
3. Redock RRP with ten Vina seeds.  The default gate requires >=5/10 rank-1
   poses to have a symmetry-corrected heavy-atom RMSD <=2.0 Angstrom.
4. Only after the gate passes, dock factor_xa_05 with the same ten seeds.
5. Cluster poses within 2 kcal/mol of the global best score at 2 Angstrom,
   selecting the cluster with the broadest seed support (then energy).
6. Calculate 4 Angstrom protein-ligand contacts and compare those contacts
   with the five per-checkpoint Stage-2 occlusion results for KinetX and 2773.

This program intentionally treats docking as a static pose plausibility check.
It does not claim to validate koff or residence time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


PDB_ID = "1NFU"
NATIVE_LIGAND = "RRP"
TARGET_COMPOUND_ID = "factor_xa_05"
TARGET_UNIPROT = "P00742"
DEFAULT_SEEDS = (101, 201, 301, 401, 501, 601, 701, 801, 901, 1001)
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
EXPECTED_HEAVY = {
    "ALA": {"N", "CA", "C", "O", "CB"},
    "ARG": {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"},
    "ASP": {"N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"},
    "CYS": {"N", "CA", "C", "O", "CB", "SG"},
    "GLN": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"},
    "GLU": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"},
    "GLY": {"N", "CA", "C", "O"},
    "HIS": {"N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": {"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"},
    "LEU": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"},
    "LYS": {"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"},
    "MET": {"N", "CA", "C", "O", "CB", "CG", "SD", "CE"},
    "PHE": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"N", "CA", "C", "O", "CB", "CG", "CD"},
    "SER": {"N", "CA", "C", "O", "CB", "OG"},
    "THR": {"N", "CA", "C", "O", "CB", "OG1", "CG2"},
    "TRP": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": {"N", "CA", "C", "O", "CB", "CG1", "CG2"},
}


@dataclass(frozen=True)
class ProteinAtom:
    element: str
    atom_name: str
    residue_name: str
    auth_chain: str
    auth_seq: str
    insertion_code: str
    label_seq: int
    uniprot_pos: Optional[int]
    x: float
    y: float
    z: float

    @property
    def residue_key(self) -> Tuple[str, str, str]:
        return self.auth_chain, self.auth_seq, self.insertion_code


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 100:
        return
    ensure_dir(destination.parent)
    request = urllib.request.Request(url, headers={"User-Agent": "MGCA-FactorXa-Stage3/1.0"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)


def run_command(command: Sequence[str], log_path: Path, cwd: Optional[Path] = None) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(str(item) for item in command) + "\n")
        log.flush()
        process = subprocess.run(
            [str(item) for item in command], cwd=str(cwd) if cwd else None,
            stdout=log, stderr=subprocess.STDOUT, text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed ({process.returncode}); inspect {log_path}: {' '.join(command)}")


def require_executables(names: Sequence[str]) -> Dict[str, str]:
    found: Dict[str, str] = {}
    missing: List[str] = []
    for name in names:
        resolved = shutil.which(name)
        if resolved:
            found[name] = resolved
        else:
            missing.append(name)
    if missing:
        raise RuntimeError("Missing required executables: " + ", ".join(missing))
    return found


def parse_seeds(value: str) -> Tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Seeds must be a non-empty comma-separated list of unique integers")
    return seeds


def normalize_cif_token(value: str) -> str:
    return "" if value in {".", "?"} else value


def cif_rows(block, tags: Sequence[str]) -> List[Tuple[str, ...]]:
    """Read same-loop mmCIF columns using Gemmi's stable find_values API."""
    columns = [list(block.find_values(tag)) for tag in tags]
    if any(not column for column in columns):
        missing = [tag for tag, column in zip(tags, columns) if not column]
        raise RuntimeError("Missing mmCIF tags: " + ", ".join(missing))
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise RuntimeError(f"mmCIF column length mismatch for {tags}: {sorted(lengths)}")
    return [tuple(str(value) for value in values) for values in zip(*columns)]


def parse_mmcif_protein_atoms(cif_path: Path) -> Tuple[List[ProteinAtom], List[Dict[str, object]]]:
    import gemmi

    document = gemmi.cif.read_file(str(cif_path))
    block = document.sole_block()
    ref_tags = [
        "_struct_ref_seq.pdbx_strand_id", "_struct_ref_seq.seq_align_beg",
        "_struct_ref_seq.seq_align_end", "_struct_ref_seq.pdbx_db_accession",
        "_struct_ref_seq.db_align_beg", "_struct_ref_seq.db_align_end",
    ]
    mappings: List[Dict[str, object]] = []
    for row in cif_rows(block, ref_tags):
        accession = str(row[3]).strip()
        if accession != TARGET_UNIPROT:
            continue
        chains = [part.strip() for part in str(row[0]).split(",")]
        for chain in chains:
            mappings.append({
                "auth_chain": chain,
                "seq_align_beg": int(str(row[1])),
                "seq_align_end": int(str(row[2])),
                "uniprot": accession,
                "db_align_beg": int(str(row[4])),
                "db_align_end": int(str(row[5])),
            })
    if not mappings:
        raise RuntimeError(f"No {TARGET_UNIPROT} mapping found in {cif_path}")

    atom_tags = [
        "_atom_site.group_PDB", "_atom_site.type_symbol", "_atom_site.label_atom_id",
        "_atom_site.label_alt_id", "_atom_site.label_comp_id", "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code", "_atom_site.Cartn_x", "_atom_site.Cartn_y",
        "_atom_site.Cartn_z", "_atom_site.occupancy", "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id", "_atom_site.auth_asym_id", "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    atoms: List[ProteinAtom] = []
    seen_alt: Set[Tuple[str, str, str, str]] = set()
    for row in cif_rows(block, atom_tags):
        if str(row[0]) != "ATOM" or str(row[15]) != "1":
            continue
        element = str(row[1]).strip().upper()
        if element == "H":
            continue
        alt = normalize_cif_token(str(row[3]).strip())
        if alt not in {"", "A"}:
            continue
        residue_name = str(row[12]).strip()
        if residue_name not in STANDARD_AA:
            continue
        chain = str(row[13]).strip()
        auth_seq = str(row[11]).strip()
        insertion = normalize_cif_token(str(row[6]).strip())
        atom_name = str(row[14]).strip()
        alt_key = (chain, auth_seq, insertion, atom_name)
        if alt_key in seen_alt:
            continue
        seen_alt.add(alt_key)
        label_seq_text = normalize_cif_token(str(row[5]).strip())
        if not label_seq_text:
            continue
        label_seq = int(label_seq_text)
        uniprot_pos: Optional[int] = None
        for mapping in mappings:
            if mapping["auth_chain"] == chain and int(mapping["seq_align_beg"]) <= label_seq <= int(mapping["seq_align_end"]):
                uniprot_pos = int(mapping["db_align_beg"]) + label_seq - int(mapping["seq_align_beg"])
                break
        atoms.append(ProteinAtom(
            element=element, atom_name=atom_name, residue_name=residue_name,
            auth_chain=chain, auth_seq=auth_seq, insertion_code=insertion,
            label_seq=label_seq, uniprot_pos=uniprot_pos,
            x=float(str(row[7])), y=float(str(row[8])), z=float(str(row[9])),
        ))
    if not atoms:
        raise RuntimeError("No protein atoms parsed from mmCIF")
    return atoms, mappings


def clean_receptor_pdb(source: Path, destination: Path) -> List[str]:
    """Retain only standard protein ATOM records, model 1, altloc blank/A."""
    source_lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    output: List[str] = []
    last_atom_chain: Optional[str] = None
    for line in source_lines:
        record = line[:6].strip()
        if record == "MODEL" and line[10:14].strip() not in {"", "1"}:
            break
        if record in {"HEADER", "TITLE", "COMPND", "SOURCE", "REMARK", "DBREF", "DBREF1", "DBREF2", "SEQADV", "SEQRES", "SSBOND"}:
            output.append(line)
        elif record == "ATOM":
            alt = line[16:17]
            residue = line[17:20].strip()
            if alt in {" ", "A"} and residue in STANDARD_AA:
                chain = line[21:22]
                if last_atom_chain is not None and chain != last_atom_chain:
                    output.append("TER")
                if alt == "A":
                    line = line[:16] + " " + line[17:]
                output.append(line)
                last_atom_chain = chain
    output.extend(["TER", "END"])
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")
    chains = sorted({line[21:22].strip() for line in output if line.startswith("ATOM")})
    if not chains:
        raise RuntimeError("Receptor cleaning produced no ATOM records")
    return chains


def extract_native_ligand_pdb(source: Path, destination: Path) -> Tuple[Set[int], Dict[str, str]]:
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    ligand_lines = [line for line in lines if line.startswith("HETATM") and line[17:20].strip() == NATIVE_LIGAND]
    if not ligand_lines:
        raise RuntimeError(f"Ligand {NATIVE_LIGAND} not found in {source}")
    instances = {(line[21:22].strip(), line[22:26].strip(), line[26:27].strip()) for line in ligand_lines}
    if len(instances) != 1:
        raise RuntimeError(f"Expected one {NATIVE_LIGAND} instance, found {sorted(instances)}")
    serials = {int(line[6:11]) for line in ligand_lines}
    conect: List[str] = []
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        numbers = []
        for start in range(6, len(line), 5):
            token = line[start:start + 5].strip()
            if token.isdigit():
                numbers.append(int(token))
        if numbers and numbers[0] in serials:
            filtered = [value for value in numbers if value in serials]
            if len(filtered) > 1:
                conect.append("CONECT" + "".join(f"{value:5d}" for value in filtered))
    destination.write_text("\n".join(ligand_lines + conect + ["END"]) + "\n", encoding="utf-8")
    chain, seq, ins = next(iter(instances))
    return serials, {"auth_chain": chain, "auth_seq": seq, "insertion_code": ins, "resname": NATIVE_LIGAND}


def remove_all_hydrogen_atoms(mol, sanitize: bool = True):
    """Remove every atomic-number-1 atom, including Hs RDKit keeps by policy.

    Chem.RemoveHs intentionally preserves certain explicit hydrogens (for
    example, some hydrogen atoms involved in valence/stereo bookkeeping).  For
    heavy-atom coordinate and RMSD audits we need an unconditional definition.
    """
    from rdkit import Chem

    editable = Chem.RWMol(mol)
    hydrogen_indices = [atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 1]
    for atom_index in sorted(hydrogen_indices, reverse=True):
        editable.RemoveAtom(atom_index)
    result = editable.GetMol()
    if sanitize:
        Chem.SanitizeMol(result)
    return result


def make_native_sdf(native_pdb: Path, ideal_sdf: Path, output_sdf: Path):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # The extracted PDB block already contains complete CONECT records.  Do not
    # ask RDKit to add distance-guessed bonds: in this compact aromatic ligand,
    # proximity bonding can create extra edges and make the CCD graph unmatched.
    pdb_mol = Chem.MolFromPDBFile(str(native_pdb), sanitize=False, removeHs=False, proximityBonding=False)
    template = Chem.SDMolSupplier(str(ideal_sdf), removeHs=False)[0]
    if pdb_mol is None or template is None:
        raise RuntimeError("RDKit failed to read experimental RRP PDB or ideal RRP SDF")
    pdb_mol = remove_all_hydrogen_atoms(pdb_mol, sanitize=False)
    template = remove_all_hydrogen_atoms(template, sanitize=True)
    coordinate_mapping_method = "AssignBondOrdersFromTemplate_with_PDB_CONECT"
    try:
        assigned = AllChem.AssignBondOrdersFromTemplate(template, pdb_mol)
        Chem.SanitizeMol(assigned)
    except Exception as exc:
        # RCSB chemical-component SDF atoms are emitted in CCD atom-table order,
        # and the ligand HETATM records use the same component order.  This
        # audited fallback transfers only experimental coordinates onto the CCD
        # chemical graph; it never guesses bonds from distances.
        if template.GetNumAtoms() != pdb_mol.GetNumAtoms():
            raise RuntimeError(
                f"RRP CCD/PDB heavy-atom count mismatch: {template.GetNumAtoms()} vs {pdb_mol.GetNumAtoms()}"
            ) from exc
        template_elements = [atom.GetAtomicNum() for atom in template.GetAtoms()]
        pdb_elements = [atom.GetAtomicNum() for atom in pdb_mol.GetAtoms()]
        if template_elements != pdb_elements:
            raise RuntimeError(
                "RRP CCD/PDB atom-order element audit failed; refusing index-based coordinate transfer. "
                f"CCD={template_elements}, PDB={pdb_elements}"
            ) from exc
        assigned = Chem.Mol(template)
        assigned.RemoveAllConformers()
        experimental = Chem.Conformer(assigned.GetNumAtoms())
        pdb_conf = pdb_mol.GetConformer()
        for atom_index in range(assigned.GetNumAtoms()):
            point = pdb_conf.GetAtomPosition(atom_index)
            experimental.SetAtomPosition(atom_index, point)
        experimental.Set3D(True)
        assigned.AddConformer(experimental, assignId=True)
        Chem.SanitizeMol(assigned)
        coordinate_mapping_method = "RCSB_CCD_atom_order_element_audited_coordinate_transfer"
    # RRP contains a benzamidine.  At physiological pH it should be treated as
    # the +1 amidinium protomer, while its experimental heavy-atom coordinates
    # remain unchanged.  Some CCD releases encode this group as neutral.
    if Chem.GetFormalCharge(assigned) == 0:
        amidine_query = Chem.MolFromSmarts("[N;H1]=[C]([N])[*]")
        amidine_matches = assigned.GetSubstructMatches(amidine_query)
        if len(amidine_matches) != 1:
            raise RuntimeError(f"Expected one neutral amidine in RRP; found {len(amidine_matches)}")
        editable = Chem.RWMol(assigned)
        imine_n = editable.GetAtomWithIdx(amidine_matches[0][0])
        imine_n.SetFormalCharge(+1)
        imine_n.SetNumExplicitHs(2)
        imine_n.SetNoImplicit(True)
        assigned = editable.GetMol()
        Chem.SanitizeMol(assigned)
    if Chem.GetFormalCharge(assigned) != 1:
        raise RuntimeError(f"Expected RRP amidinium formal charge +1, obtained {Chem.GetFormalCharge(assigned)}")
    for index, atom in enumerate(assigned.GetAtoms()):
        atom.SetAtomMapNum(index + 1)
    assigned.SetProp("_Name", "1NFU_RRP_experimental")
    assigned.SetProp("native_coordinate_mapping_method", coordinate_mapping_method)
    with_h = Chem.AddHs(assigned, addCoords=True)
    writer = Chem.SDWriter(str(output_sdf))
    writer.write(with_h)
    writer.close()
    return assigned


def build_target_ligand(source_smiles: str, output_sdf: Path, random_seed: int = 20260811):
    """Create the +1 guanidinium protomer while preserving Stage-2 heavy atom order."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    neutral = Chem.MolFromSmiles(source_smiles)
    if neutral is None:
        raise RuntimeError(f"Invalid Stage-2 target SMILES: {source_smiles}")
    target = Chem.RWMol(neutral)
    matches = neutral.GetSubstructMatches(Chem.MolFromSmarts("[N;H1]=[C]([N])[N]"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one neutral guanidine group; found {len(matches)}")
    imine_n = target.GetAtomWithIdx(matches[0][0])
    imine_n.SetFormalCharge(+1)
    imine_n.SetNumExplicitHs(2)
    imine_n.SetNoImplicit(True)
    charged = target.GetMol()
    Chem.SanitizeMol(charged)
    if Chem.GetFormalCharge(charged) != 1:
        raise RuntimeError(f"Expected target formal charge +1, obtained {Chem.GetFormalCharge(charged)}")
    if neutral.GetNumAtoms() != charged.GetNumAtoms():
        raise RuntimeError("Protonation changed heavy-atom count and invalidated Stage-2 atom mapping")
    for index, atom in enumerate(charged.GetAtoms()):
        atom.SetAtomMapNum(index + 1)
        atom.SetIntProp("stage2_atom_index", index)
    with_h = Chem.AddHs(charged)
    status = AllChem.EmbedMolecule(with_h, randomSeed=random_seed, useRandomCoords=True)
    if status != 0:
        raise RuntimeError("RDKit ETKDG embedding failed for factor_xa_05")
    if AllChem.MMFFHasAllMoleculeParams(with_h):
        AllChem.MMFFOptimizeMolecule(with_h, maxIters=1000)
        forcefield = "MMFF94"
    else:
        AllChem.UFFOptimizeMolecule(with_h, maxIters=1000)
        forcefield = "UFF"
    with_h.SetProp("_Name", TARGET_COMPOUND_ID + "_guanidinium_pH7p4")
    with_h.SetProp("source_stage2_smiles", source_smiles)
    with_h.SetProp("docking_smiles", Chem.MolToSmiles(charged, isomericSmiles=True))
    with_h.SetProp("formal_charge", "+1")
    with_h.SetProp("initial_3d_forcefield", forcefield)
    writer = Chem.SDWriter(str(output_sdf))
    writer.write(with_h)
    writer.close()
    return neutral, charged, forcefield


def ligand_heavy_coordinates(mol) -> Tuple[List[int], List[Tuple[float, float, float]]]:
    indices: List[int] = []
    coordinates: List[Tuple[float, float, float]] = []
    conformer = mol.GetConformer()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        point = conformer.GetAtomPosition(atom.GetIdx())
        indices.append(atom.GetIdx())
        coordinates.append((float(point.x), float(point.y), float(point.z)))
    return indices, coordinates


def box_from_ligand(mol, padding: float = 5.0, minimum: float = 22.0, maximum: float = 30.0) -> Dict[str, List[float]]:
    _, coords = ligand_heavy_coordinates(mol)
    center = [sum(point[axis] for point in coords) / len(coords) for axis in range(3)]
    spans = [max(point[axis] for point in coords) - min(point[axis] for point in coords) for axis in range(3)]
    size = [max(minimum, min(maximum, span + 2.0 * padding)) for span in spans]
    return {"center": center, "size": size, "native_heavy_atom_span": spans}


def squared_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((float(a[i]) - float(b[i])) ** 2 for i in range(3))


def audit_binding_site_completeness(protein_atoms: Sequence[ProteinAtom], native_mol, radius: float = 6.0) -> List[Dict[str, object]]:
    _, ligand_coords = ligand_heavy_coordinates(native_mol)
    by_residue: Dict[Tuple[str, str, str], List[ProteinAtom]] = defaultdict(list)
    for atom in protein_atoms:
        if min(squared_distance((atom.x, atom.y, atom.z), coord) for coord in ligand_coords) <= radius ** 2:
            by_residue[atom.residue_key].append(atom)
    rows: List[Dict[str, object]] = []
    incomplete: List[str] = []
    for key, atoms in sorted(by_residue.items()):
        residue_name = atoms[0].residue_name
        observed = {atom.atom_name for atom in atoms}
        # Fetch all atoms of the same residue, not just those within radius.
        observed = {atom.atom_name for atom in protein_atoms if atom.residue_key == key}
        missing = sorted(EXPECTED_HEAVY.get(residue_name, set()) - observed)
        row = {
            "auth_chain": key[0], "auth_seq": key[1], "insertion_code": key[2],
            "residue_name": residue_name, "uniprot_pos": atoms[0].uniprot_pos,
            "observed_heavy_atoms": len(observed), "missing_heavy_atoms": ";".join(missing),
            "complete": not missing,
        }
        rows.append(row)
        if missing:
            incomplete.append(f"{key[0]}:{residue_name}{key[1]}{key[2]} missing {','.join(missing)}")
    if incomplete:
        raise RuntimeError("Incomplete protein residues within 6 A of native RRP: " + "; ".join(incomplete))
    return rows


def stage2_target_smiles(stage2_root: Path) -> str:
    smiles_by_model: Dict[str, str] = {}
    for model in ("KinetX", "2773"):
        catalog = stage2_root / model / "factor_xa_compound_catalog.csv"
        if not catalog.exists():
            raise FileNotFoundError(f"Missing Stage-2 catalog: {catalog}")
        matches = [row for row in read_rows(catalog) if row["compound_id"] == TARGET_COMPOUND_ID]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {TARGET_COMPOUND_ID} row in {catalog}, found {len(matches)}")
        smiles_by_model[model] = matches[0]["canonical_smiles"]
    if len(set(smiles_by_model.values())) != 1:
        raise RuntimeError(f"KinetX and 2773 Stage-2 target SMILES differ: {smiles_by_model}")
    return next(iter(smiles_by_model.values()))


def prepare(args, paths: Mapping[str, Path]) -> Dict[str, object]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, rdFMCS

    inputs = ensure_dir(paths["inputs"])
    prepared = ensure_dir(paths["prepared"])
    logs = ensure_dir(paths["logs"])
    pdb_path = inputs / f"{PDB_ID}.pdb"
    cif_path = inputs / f"{PDB_ID}.cif"
    ideal_sdf = inputs / f"{NATIVE_LIGAND}_ideal.sdf"
    download(f"https://files.rcsb.org/download/{PDB_ID}.pdb", pdb_path)
    download(f"https://files.rcsb.org/download/{PDB_ID}.cif", cif_path)
    download(f"https://files.rcsb.org/ligands/download/{NATIVE_LIGAND}_ideal.sdf", ideal_sdf)

    receptor_clean = prepared / f"{PDB_ID}_receptor_clean.pdb"
    native_pdb = prepared / f"{PDB_ID}_{NATIVE_LIGAND}_experimental.pdb"
    native_sdf = prepared / f"{PDB_ID}_{NATIVE_LIGAND}_experimental.sdf"
    target_sdf = prepared / f"{TARGET_COMPOUND_ID}_guanidinium.sdf"
    chains = clean_receptor_pdb(pdb_path, receptor_clean)
    _, native_instance = extract_native_ligand_pdb(pdb_path, native_pdb)
    native_heavy = make_native_sdf(native_pdb, ideal_sdf, native_sdf)
    protein_atoms, mappings = parse_mmcif_protein_atoms(cif_path)
    completeness = audit_binding_site_completeness(protein_atoms, native_heavy)
    write_rows(prepared / "binding_site_completeness_audit.csv", completeness)
    write_rows(prepared / "pdb_uniprot_mapping.csv", mappings)

    source_smiles = stage2_target_smiles(args.stage2_root)
    neutral, target_heavy, forcefield = build_target_ligand(source_smiles, target_sdf)
    box = box_from_ligand(native_heavy, padding=args.box_padding, minimum=args.box_min, maximum=args.box_max)

    native_fp = AllChem.GetMorganFingerprintAsBitVect(native_heavy, 2, nBits=2048)
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_heavy, 2, nBits=2048)
    tanimoto = float(DataStructs.TanimotoSimilarity(native_fp, target_fp))
    mcs = rdFMCS.FindMCS([native_heavy, target_heavy], timeout=60, ringMatchesRingOnly=True, completeRingsOnly=True)

    executables = require_executables(["mk_prepare_ligand.py", "mk_prepare_receptor.py", "mk_export.py"])
    receptor_prefix = prepared / f"{PDB_ID}_receptor"
    receptor_log = logs / "receptor_preparation.log"
    receptor_command = [
        executables["mk_prepare_receptor.py"], "--read_pdb", str(receptor_clean),
        "-o", str(receptor_prefix), "-p", "-j", "-v",
        "--box_center", *[f"{value:.4f}" for value in box["center"]],
        "--box_size", *[f"{value:.4f}" for value in box["size"]], "-a",
    ]
    run_command(receptor_command, receptor_log, cwd=prepared)
    receptor_pdbqt = receptor_prefix.with_suffix(".pdbqt")
    if not receptor_pdbqt.exists():
        alternatives = sorted(prepared.glob(f"{receptor_prefix.name}*rigid*.pdbqt"))
        if alternatives:
            receptor_pdbqt = alternatives[0]
        else:
            raise RuntimeError(f"Meeko did not create receptor PDBQT under {prepared}")

    native_pdbqt = prepared / f"{PDB_ID}_{NATIVE_LIGAND}.pdbqt"
    target_pdbqt = prepared / f"{TARGET_COMPOUND_ID}.pdbqt"
    run_command([
        executables["mk_prepare_ligand.py"], "-i", str(native_sdf), "-o", str(native_pdbqt),
        "--charge_model", "gasteiger", "--add_index_map",
    ], logs / "native_ligand_preparation.log", cwd=prepared)
    run_command([
        executables["mk_prepare_ligand.py"], "-i", str(target_sdf), "-o", str(target_pdbqt),
        "--charge_model", "gasteiger", "--add_index_map",
    ], logs / "target_ligand_preparation.log", cwd=prepared)

    payload: Dict[str, object] = {
        "protocol": "factor_xa_stage3_prepare_v1",
        "created_at_utc": now_utc(), "pdb_id": PDB_ID,
        "native_ligand": NATIVE_LIGAND, "target_compound_id": TARGET_COMPOUND_ID,
        "target_uniprot": TARGET_UNIPROT, "protein_chains": chains,
        "native_ligand_instance": native_instance,
        "native_coordinate_mapping_method": native_heavy.GetProp("native_coordinate_mapping_method"),
        "source_stage2_smiles": source_smiles,
        "docking_smiles": Chem.MolToSmiles(target_heavy, isomericSmiles=True),
        "target_formal_charge": Chem.GetFormalCharge(target_heavy),
        "target_initial_3d_forcefield": forcefield,
        "native_target_morgan_tanimoto": tanimoto,
        "native_target_mcs_heavy_atoms": int(mcs.numAtoms),
        "box": box, "binding_site_completeness_passed": True,
        "pdb_uniprot_mapping": mappings,
        "files": {
            "pdb": str(pdb_path), "cif": str(cif_path), "ideal_sdf": str(ideal_sdf),
            "receptor_clean": str(receptor_clean), "receptor_pdbqt": str(receptor_pdbqt),
            "native_sdf": str(native_sdf), "native_pdbqt": str(native_pdbqt),
            "target_sdf": str(target_sdf), "target_pdbqt": str(target_pdbqt),
        },
        "sha256": {str(path): sha256_file(path) for path in [pdb_path, cif_path, ideal_sdf, receptor_clean, native_sdf, target_sdf, receptor_pdbqt, native_pdbqt, target_pdbqt]},
    }
    write_json(prepared / "preparation_manifest.json", payload)
    return payload


def vina_one_seed(
    receptor_pdbqt: Path, ligand_pdbqt: Path, box: Mapping[str, Sequence[float]],
    seed: int, output_pdbqt: Path, energies_csv: Path, cpu: int,
    exhaustiveness: int, n_poses: int, energy_range: float,
) -> None:
    from vina import Vina

    docking = Vina(sf_name="vina", cpu=cpu, seed=seed, verbosity=1)
    docking.set_receptor(str(receptor_pdbqt))
    docking.set_ligand_from_file(str(ligand_pdbqt))
    docking.compute_vina_maps(center=list(box["center"]), box_size=list(box["size"]))
    docking.dock(exhaustiveness=exhaustiveness, n_poses=n_poses, min_rmsd=1.0)
    energies = docking.energies(n_poses=n_poses, energy_range=energy_range)
    docking.write_poses(str(output_pdbqt), n_poses=n_poses, energy_range=energy_range, overwrite=True)
    columns = ["affinity", "intermolecular", "intramolecular", "torsional", "unbound"]
    rows = []
    for rank, values in enumerate(energies, start=1):
        row: Dict[str, object] = {"seed": seed, "pose_rank": rank}
        for index, value in enumerate(values):
            row[columns[index] if index < len(columns) else f"energy_{index}"] = float(value)
        rows.append(row)
    write_rows(energies_csv, rows)


def export_poses(pdbqt: Path, sdf: Path, log_path: Path) -> None:
    executable = require_executables(["mk_export.py"])["mk_export.py"]
    run_command([executable, str(pdbqt), "-s", str(sdf)], log_path, cwd=sdf.parent)
    if not sdf.exists():
        raise RuntimeError(f"Pose export did not create {sdf}")


def read_sdf_molecules(path: Path):
    from rdkit import Chem

    molecules = [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol is not None]
    if not molecules:
        raise RuntimeError(f"No valid molecules in {path}")
    return molecules


def remove_hydrogens(mol):
    return remove_all_hydrogen_atoms(mol, sanitize=True)


def symmetry_rmsd_fixed_frame(reference, probe, max_matches: int = 100000) -> float:
    """Graph-symmetry RMSD without rotational/translational superposition."""
    import numpy as np

    ref = remove_hydrogens(reference)
    fit = remove_hydrogens(probe)
    if ref.GetNumAtoms() != fit.GetNumAtoms():
        raise RuntimeError(f"RMSD atom count mismatch: {ref.GetNumAtoms()} vs {fit.GetNumAtoms()}")
    matches = fit.GetSubstructMatches(ref, uniquify=False, maxMatches=max_matches)
    if not matches:
        matches = (tuple(range(ref.GetNumAtoms())),)
        if [a.GetAtomicNum() for a in ref.GetAtoms()] != [a.GetAtomicNum() for a in fit.GetAtoms()]:
            raise RuntimeError("No graph match between RMSD reference and probe")
    ref_conf = ref.GetConformer()
    fit_conf = fit.GetConformer()
    ref_xyz = np.array([[ref_conf.GetAtomPosition(i).x, ref_conf.GetAtomPosition(i).y, ref_conf.GetAtomPosition(i).z] for i in range(ref.GetNumAtoms())])
    best = float("inf")
    for match in matches:
        fit_xyz = np.array([[fit_conf.GetAtomPosition(match[i]).x, fit_conf.GetAtomPosition(match[i]).y, fit_conf.GetAtomPosition(match[i]).z] for i in range(ref.GetNumAtoms())])
        value = float(np.sqrt(np.mean(np.sum((ref_xyz - fit_xyz) ** 2, axis=1))))
        best = min(best, value)
    return best


def run_seed_set(args, preparation: Mapping[str, object], ligand_key: str, output_dir: Path) -> List[Dict[str, object]]:
    import vina

    files = preparation["files"]
    receptor = Path(files["receptor_pdbqt"])
    ligand = Path(files[ligand_key])
    ensure_dir(output_dir)
    aggregate: List[Dict[str, object]] = []
    for seed in args.seeds:
        prefix = output_dir / f"seed_{seed}"
        pdbqt = prefix.with_suffix(".pdbqt")
        sdf = prefix.with_suffix(".sdf")
        energies = output_dir / f"seed_{seed}_energies.csv"
        run_manifest = output_dir / f"seed_{seed}_run.json"
        configuration = {
            "receptor_sha256": sha256_file(receptor), "ligand_sha256": sha256_file(ligand),
            "box_center": list(preparation["box"]["center"]), "box_size": list(preparation["box"]["size"]),
            "seed": seed, "cpu": args.cpu, "exhaustiveness": args.exhaustiveness,
            "n_poses": args.n_poses, "energy_range": args.energy_range,
            "vina_version": vina.__version__,
        }
        reusable = False
        if pdbqt.exists() and sdf.exists() and energies.exists() and run_manifest.exists() and not args.recompute:
            prior = read_json(run_manifest)
            reusable = prior.get("configuration") == configuration
            if not reusable:
                raise RuntimeError(
                    f"Existing seed outputs use a different configuration: {run_manifest}. "
                    "Use --recompute only after confirming that overwriting this Stage-3 run is intended."
                )
        if not reusable:
            vina_one_seed(
                receptor, ligand, preparation["box"], seed, pdbqt, energies,
                args.cpu, args.exhaustiveness, args.n_poses, args.energy_range,
            )
            export_poses(pdbqt, sdf, output_dir / "pose_export.log")
            write_json(run_manifest, {
                "protocol": "vina_seed_run_v1", "completed_at_utc": now_utc(),
                "configuration": configuration,
                "outputs": {"pdbqt": str(pdbqt), "sdf": str(sdf), "energies": str(energies)},
                "output_sha256": {"pdbqt": sha256_file(pdbqt), "sdf": sha256_file(sdf), "energies": sha256_file(energies)},
            })
        molecules = read_sdf_molecules(sdf)
        energy_rows = read_rows(energies)
        if len(molecules) != len(energy_rows):
            raise RuntimeError(f"Pose/energy count mismatch for seed {seed}: {len(molecules)} vs {len(energy_rows)}")
        for index, row in enumerate(energy_rows):
            aggregate.append({
                "seed": seed, "pose_rank": int(row["pose_rank"]),
                "affinity": float(row["affinity"]), "sdf": str(sdf),
                "molecule_index": index,
            })
    return aggregate


def redock(args, paths: Mapping[str, Path], preparation: Mapping[str, object]) -> Dict[str, object]:
    redock_dir = ensure_dir(paths["redocking"])
    pose_rows = run_seed_set(args, preparation, "native_pdbqt", redock_dir)
    native = read_sdf_molecules(Path(preparation["files"]["native_sdf"]))[0]
    by_sdf: Dict[str, List[object]] = {}
    metrics: List[Dict[str, object]] = []
    for row in pose_rows:
        if row["sdf"] not in by_sdf:
            by_sdf[row["sdf"]] = read_sdf_molecules(Path(row["sdf"]))
        pose = by_sdf[row["sdf"]][int(row["molecule_index"])]
        rmsd = symmetry_rmsd_fixed_frame(native, pose)
        metrics.append({**row, "symmetry_corrected_heavy_atom_rmsd": rmsd, "passes_2A": rmsd <= args.redock_rmsd})
    write_rows(redock_dir / "redocking_pose_metrics.csv", metrics)
    top1 = [row for row in metrics if int(row["pose_rank"]) == 1]
    passing = sum(bool(row["passes_2A"]) for row in top1)
    passed = passing >= args.redock_min_seed_passes
    summary = {
        "protocol": "native_ligand_redocking_gate_v1", "created_at_utc": now_utc(),
        "pdb_id": PDB_ID, "native_ligand": NATIVE_LIGAND,
        "seeds": list(args.seeds), "rank1_rmsd_threshold_A": args.redock_rmsd,
        "required_passing_seeds": args.redock_min_seed_passes,
        "observed_passing_seeds": passing, "passed": passed,
        "rank1_by_seed": [
            {key: row[key] for key in ["seed", "affinity", "symmetry_corrected_heavy_atom_rmsd", "passes_2A"]}
            for row in top1
        ],
    }
    write_json(redock_dir / "redocking_gate.json", summary)
    if not passed and not args.allow_redock_failure:
        raise RuntimeError(
            f"Native redocking gate failed: {passing}/{len(args.seeds)} rank-1 poses <= {args.redock_rmsd} A; "
            f"required {args.redock_min_seed_passes}. Formal docking was not run."
        )
    return summary


def cluster_formal_poses(args, pose_rows: List[Dict[str, object]], output_dir: Path) -> Tuple[List[Dict[str, object]], Dict[str, object], object]:
    all_molecules: Dict[str, List[object]] = {}
    for row in pose_rows:
        if row["sdf"] not in all_molecules:
            all_molecules[row["sdf"]] = read_sdf_molecules(Path(row["sdf"]))
        row["molecule"] = all_molecules[row["sdf"]][int(row["molecule_index"])]
    global_best = min(float(row["affinity"]) for row in pose_rows)
    eligible = [row for row in pose_rows if float(row["affinity"]) <= global_best + args.cluster_energy_window]
    eligible.sort(key=lambda row: (float(row["affinity"]), int(row["seed"]), int(row["pose_rank"])))
    clusters: List[Dict[str, object]] = []
    for row in eligible:
        assigned: Optional[Dict[str, object]] = None
        for cluster in clusters:
            rmsd = symmetry_rmsd_fixed_frame(cluster["representative"]["molecule"], row["molecule"])
            if rmsd <= args.cluster_rmsd:
                assigned = cluster
                break
        if assigned is None:
            assigned = {"cluster_id": len(clusters) + 1, "representative": row, "members": []}
            clusters.append(assigned)
        assigned["members"].append(row)
    cluster_rows: List[Dict[str, object]] = []
    for cluster in clusters:
        members = cluster["members"]
        seeds = sorted({int(row["seed"]) for row in members})
        affinities = [float(row["affinity"]) for row in members]
        cluster_rows.append({
            "cluster_id": cluster["cluster_id"], "n_poses": len(members),
            "n_unique_seeds": len(seeds), "seeds_json": json.dumps(seeds),
            "best_affinity": min(affinities), "mean_affinity": statistics.mean(affinities),
            "representative_seed": int(cluster["representative"]["seed"]),
            "representative_pose_rank": int(cluster["representative"]["pose_rank"]),
        })
    cluster_rows.sort(key=lambda row: (-int(row["n_unique_seeds"]), float(row["best_affinity"]), float(row["mean_affinity"]), int(row["cluster_id"])))
    selected_cluster_row = cluster_rows[0]
    selected_cluster = next(cluster for cluster in clusters if cluster["cluster_id"] == selected_cluster_row["cluster_id"])
    selected = min(selected_cluster["members"], key=lambda row: (float(row["affinity"]), int(row["seed"]), int(row["pose_rank"])))
    selection = {
        "selection_rule": (
            f"within {args.cluster_energy_window:g} kcal/mol of global best; greedy "
            f"{args.cluster_rmsd:g} A clusters; maximize unique seed support; then lowest energy"
        ),
        "global_best_affinity": global_best,
        "energy_window_kcal_mol": args.cluster_energy_window,
        "cluster_rmsd_A": args.cluster_rmsd,
        "eligible_pose_count": len(eligible),
        "selected_cluster": selected_cluster_row,
        "selected_pose": {key: selected[key] for key in ["seed", "pose_rank", "affinity", "sdf", "molecule_index"]},
        "seed_support_warning": int(selected_cluster_row["n_unique_seeds"]) < math.ceil(len(args.seeds) / 2),
    }
    clean_pose_rows = [{key: row[key] for key in ["seed", "pose_rank", "affinity", "sdf", "molecule_index"]} for row in pose_rows]
    write_rows(output_dir / "all_formal_pose_metrics.csv", clean_pose_rows)
    write_rows(output_dir / "pose_clusters.csv", cluster_rows)
    write_json(output_dir / "selected_pose_manifest.json", selection)
    return cluster_rows, selection, selected["molecule"]


def atom_original_index_map(pose, reference) -> Tuple[Dict[int, int], str]:
    """Return original Stage-2 heavy index -> pose heavy index."""
    pose_h = remove_hydrogens(pose)
    ref_h = remove_hydrogens(reference)
    mapped: Dict[int, int] = {}
    for pose_index, atom in enumerate(pose_h.GetAtoms()):
        atom_map = atom.GetAtomMapNum()
        if 1 <= atom_map <= ref_h.GetNumAtoms():
            mapped[atom_map - 1] = pose_index
    if len(mapped) == ref_h.GetNumAtoms() and len(set(mapped.values())) == ref_h.GetNumAtoms():
        return mapped, "atom_map_number"
    matches = pose_h.GetSubstructMatches(ref_h, uniquify=False, maxMatches=100000)
    if not matches:
        raise RuntimeError("Unable to map selected pose atoms back to Stage-2 atom indices")
    identity_best = max(matches, key=lambda match: sum(index == value for index, value in enumerate(match)))
    return {index: int(identity_best[index]) for index in range(len(identity_best))}, "graph_match_identity_preferred"


def contacts_for_pose(protein_atoms: Sequence[ProteinAtom], pose, reference, cutoff: float) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Set[Tuple[int, int]], str]:
    pose_h = remove_hydrogens(pose)
    ref_h = remove_hydrogens(reference)
    original_to_pose, mapping_method = atom_original_index_map(pose_h, ref_h)
    conf = pose_h.GetConformer()
    pose_coords = {
        original: (
            float(conf.GetAtomPosition(pose_index).x), float(conf.GetAtomPosition(pose_index).y), float(conf.GetAtomPosition(pose_index).z),
        )
        for original, pose_index in original_to_pose.items()
    }
    cutoff2 = cutoff ** 2
    residue_details: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    atom_details: Dict[int, Dict[str, object]] = {}
    pairs: Set[Tuple[int, int]] = set()
    for protein in protein_atoms:
        pcoord = (protein.x, protein.y, protein.z)
        for original_index, lcoord in pose_coords.items():
            d2 = squared_distance(pcoord, lcoord)
            if d2 > cutoff2:
                continue
            distance = math.sqrt(d2)
            if protein.uniprot_pos is not None:
                pairs.add((protein.uniprot_pos, original_index))
            rkey = protein.residue_key
            detail = residue_details.setdefault(rkey, {
                "auth_chain": protein.auth_chain, "auth_seq": protein.auth_seq,
                "insertion_code": protein.insertion_code, "residue_name": protein.residue_name,
                "residue_one_letter": AA1.get(protein.residue_name, "X"),
                "uniprot_pos": protein.uniprot_pos, "min_distance_A": float("inf"),
                "protein_atom_names": set(), "ligand_atom_indices": set(), "atom_pair_count": 0,
            })
            detail["min_distance_A"] = min(float(detail["min_distance_A"]), distance)
            detail["protein_atom_names"].add(protein.atom_name)
            detail["ligand_atom_indices"].add(original_index)
            detail["atom_pair_count"] = int(detail["atom_pair_count"]) + 1
            atom_detail = atom_details.setdefault(original_index, {
                "stage2_atom_index": original_index,
                "atom_symbol": ref_h.GetAtomWithIdx(original_index).GetSymbol(),
                "min_distance_A": float("inf"), "contact_uniprot_positions": set(),
                "contact_pdb_residues": set(), "atom_pair_count": 0,
            })
            atom_detail["min_distance_A"] = min(float(atom_detail["min_distance_A"]), distance)
            if protein.uniprot_pos is not None:
                atom_detail["contact_uniprot_positions"].add(protein.uniprot_pos)
            atom_detail["contact_pdb_residues"].add(f"{protein.auth_chain}:{protein.residue_name}{protein.auth_seq}{protein.insertion_code}")
            atom_detail["atom_pair_count"] = int(atom_detail["atom_pair_count"]) + 1
    residue_rows: List[Dict[str, object]] = []
    for detail in residue_details.values():
        detail = dict(detail)
        detail["protein_atom_names_json"] = json.dumps(sorted(detail.pop("protein_atom_names")))
        detail["ligand_atom_indices_json"] = json.dumps(sorted(detail.pop("ligand_atom_indices")))
        residue_rows.append(detail)
    residue_rows.sort(key=lambda row: (row["uniprot_pos"] is None, row["uniprot_pos"] or 10**9, row["auth_chain"], str(row["auth_seq"])))
    atom_rows: List[Dict[str, object]] = []
    for detail in atom_details.values():
        detail = dict(detail)
        detail["contact_uniprot_positions_json"] = json.dumps(sorted(detail.pop("contact_uniprot_positions")))
        detail["contact_pdb_residues_json"] = json.dumps(sorted(detail.pop("contact_pdb_residues")))
        atom_rows.append(detail)
    atom_rows.sort(key=lambda row: int(row["stage2_atom_index"]))
    return residue_rows, atom_rows, pairs, mapping_method


def ranking_metrics(scores: Mapping[object, float], positives: Set[object]) -> Dict[str, object]:
    universe = list(scores)
    positives = positives.intersection(universe)
    negatives = set(universe) - positives
    if not positives or not negatives:
        return {
            "universe_size": len(universe), "positive_count": len(positives), "top_k": len(positives),
            "topk_hits": None, "precision_at_k": None, "recall_at_k": None,
            "jaccard_at_k": None, "enrichment_at_k": None, "roc_auc": None, "average_precision": None,
        }
    ordered = sorted(universe, key=lambda item: (-float(scores[item]), str(item)))
    k = len(positives)
    selected = set(ordered[:k])
    hits = len(selected & positives)
    jaccard = hits / len(selected | positives) if selected | positives else 0.0
    baseline = len(positives) / len(universe)
    auc_pairs = []
    for pos in positives:
        for neg in negatives:
            auc_pairs.append(1.0 if scores[pos] > scores[neg] else 0.5 if scores[pos] == scores[neg] else 0.0)
    precisions = []
    true_count = 0
    for rank, item in enumerate(ordered, start=1):
        if item in positives:
            true_count += 1
            precisions.append(true_count / rank)
    return {
        "universe_size": len(universe), "positive_count": len(positives), "top_k": k,
        "topk_hits": hits, "precision_at_k": hits / k, "recall_at_k": hits / len(positives),
        "jaccard_at_k": jaccard, "enrichment_at_k": (hits / k) / baseline,
        "roc_auc": statistics.mean(auc_pairs), "average_precision": statistics.mean(precisions),
    }


def bool_csv(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def compare_one_model(
    model: str, stage2_root: Path, resolved_uniprot: Set[int], contact_residues: Set[int],
    ligand_atom_count: int, contact_atoms: Set[int], contact_pairs: Set[Tuple[int, int]],
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[int, float]]]:
    model_dir = stage2_root / model
    protein_rows = [
        row for row in read_rows(model_dir / "protein_window_importance_per_seed.csv")
        if row["compound_id"] == TARGET_COMPOUND_ID and bool_csv(row["is_primary_configuration"])
    ]
    ligand_rows = [
        row for row in read_rows(model_dir / "ligand_bit_importance_per_seed.csv")
        if row["compound_id"] == TARGET_COMPOUND_ID
    ]
    dual_rows = [
        row for row in read_rows(model_dir / "dual_occlusion_interactions_per_seed.csv")
        if row["compound_id"] == TARGET_COMPOUND_ID
    ]
    seeds = sorted({int(row["seed"]) for row in protein_rows})
    if seeds != sorted({int(row["seed"]) for row in ligand_rows}) or seeds != sorted({int(row["seed"]) for row in dual_rows}):
        raise RuntimeError(f"Stage-2 seed mismatch for {model}")
    output: List[Dict[str, object]] = []
    plot_scores: Dict[str, Dict[int, float]] = {}
    for seed in seeds:
        windows = [row for row in protein_rows if int(row["seed"]) == seed]
        per_residue_values: Dict[int, List[float]] = defaultdict(list)
        for row in windows:
            for residue in range(int(row["window_start"]), int(row["window_end"]) + 1):
                per_residue_values[residue].append(float(row["delta_pkoff"]))
        protein_scores = {residue: statistics.mean(per_residue_values.get(residue, [0.0])) for residue in resolved_uniprot}
        plot_scores[f"protein_seed_{seed}"] = protein_scores
        metrics = ranking_metrics(protein_scores, contact_residues)
        output.append({"training_dataset": model, "seed": seed, "feature_level": "protein_residue", **metrics})

        atom_values: Dict[int, List[float]] = defaultdict(list)
        for row in ligand_rows:
            if int(row["seed"]) != seed:
                continue
            for atom_index in json.loads(row["atom_indices_json"]):
                atom_values[int(atom_index)].append(float(row["delta_pkoff"]))
        ligand_scores = {atom: statistics.mean(atom_values.get(atom, [0.0])) for atom in range(ligand_atom_count)}
        plot_scores[f"ligand_seed_{seed}"] = ligand_scores
        metrics = ranking_metrics(ligand_scores, contact_atoms)
        output.append({"training_dataset": model, "seed": seed, "feature_level": "ligand_atom", **metrics})

        pair_scores: Dict[str, float] = {}
        pair_positive: Set[str] = set()
        for row_index, row in enumerate(row for row in dual_rows if int(row["seed"]) == seed):
            key = f"{row['protein_feature_key']}|{row['ligand_feature_key']}|{row_index}"
            pair_scores[key] = abs(float(row["interaction_pkoff"]))
            start, end = int(row["window_start"]), int(row["window_end"])
            atoms = {int(value) for value in json.loads(row["atom_indices_json"])}
            if any(start <= residue <= end and atom in atoms for residue, atom in contact_pairs):
                pair_positive.add(key)
        metrics = ranking_metrics(pair_scores, pair_positive)
        output.append({"training_dataset": model, "seed": seed, "feature_level": "dual_window_bit", **metrics})
    return output, plot_scores


def summarize_metrics(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["training_dataset"]), str(row["feature_level"]))].append(row)
    metrics = ["topk_hits", "precision_at_k", "recall_at_k", "jaccard_at_k", "enrichment_at_k", "roc_auc", "average_precision"]
    output: List[Dict[str, object]] = []
    for (model, level), group in sorted(grouped.items()):
        summary: Dict[str, object] = {
            "training_dataset": model, "feature_level": level, "n_seeds": len(group),
            "universe_size": group[0]["universe_size"], "positive_count": group[0]["positive_count"],
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group if row[metric] is not None]
            summary[f"mean_{metric}"] = statistics.mean(values) if values else None
            summary[f"sd_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
            summary[f"values_{metric}"] = "/".join(f"{value:.4f}" for value in values)
        output.append(summary)
    return output


def format_optional(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def create_plots(
    paths: Mapping[str, Path], redock_summary: Mapping[str, object], formal_rows: Sequence[Mapping[str, object]],
    contact_residues: Set[int], contact_atoms: Set[int], target_reference, model_plot_scores: Mapping[str, Mapping[str, Dict[int, float]]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from rdkit.Chem.Draw import rdMolDraw2D

    figures = ensure_dir(paths["figures"])
    top1 = redock_summary["rank1_by_seed"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar([str(row["seed"]) for row in top1], [float(row["symmetry_corrected_heavy_atom_rmsd"]) for row in top1])
    axes[0].axhline(2.0, color="red", linestyle="--", linewidth=1)
    axes[0].set(title="Native RRP redocking: rank-1 pose", xlabel="Vina seed", ylabel="Symmetry-corrected RMSD (A)")
    by_seed: Dict[int, List[float]] = defaultdict(list)
    for row in formal_rows:
        by_seed[int(row["seed"])].append(float(row["affinity"]))
    axes[1].boxplot([by_seed[seed] for seed in sorted(by_seed)], tick_labels=[str(seed) for seed in sorted(by_seed)])
    axes[1].set(title="factor_xa_05 docking scores", xlabel="Vina seed", ylabel="Vina affinity (kcal/mol)")
    fig.tight_layout()
    fig.savefig(figures / "redocking_and_formal_docking.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    for axis, model in zip(axes, ("KinetX", "2773")):
        scores_by_seed = [scores for key, scores in model_plot_scores[model].items() if key.startswith("protein_seed_")]
        residues = sorted(set().union(*(set(scores) for scores in scores_by_seed)))
        means = [statistics.mean(scores.get(residue, 0.0) for scores in scores_by_seed) for residue in residues]
        sds = [statistics.stdev([scores.get(residue, 0.0) for scores in scores_by_seed]) if len(scores_by_seed) > 1 else 0.0 for residue in residues]
        axis.plot(residues, means, linewidth=1.0, label=f"{model} mean occlusion score")
        axis.fill_between(residues, np.array(means) - np.array(sds), np.array(means) + np.array(sds), alpha=0.2)
        for residue in sorted(contact_residues):
            axis.axvline(residue, color="red", alpha=0.18, linewidth=0.8)
        axis.set_ylabel("Delta pKoff")
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("UniProt P00742 residue position (red: 4 A docking contact)")
    fig.tight_layout()
    fig.savefig(figures / "protein_occlusion_vs_docking_contacts.png", dpi=300)
    plt.close(fig)

    heavy = remove_hydrogens(target_reference)
    for model in ("KinetX", "2773"):
        ligand_seed_scores = [scores for key, scores in model_plot_scores[model].items() if key.startswith("ligand_seed_")]
        mean_score = {atom: statistics.mean(scores.get(atom, 0.0) for scores in ligand_seed_scores) for atom in range(heavy.GetNumAtoms())}
        top_atoms = set(sorted(mean_score, key=lambda atom: (-mean_score[atom], atom))[:max(1, len(contact_atoms))])
        drawer = rdMolDraw2D.MolDraw2DSVG(900, 500)
        options = drawer.drawOptions()
        options.addAtomIndices = True
        colors = {atom: (0.95, 0.30, 0.25) if atom in contact_atoms else (0.25, 0.55, 0.95) for atom in contact_atoms | top_atoms}
        radii = {atom: 0.42 for atom in colors}
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, heavy, highlightAtoms=sorted(colors), highlightAtomColors=colors, highlightAtomRadii=radii)
        drawer.FinishDrawing()
        (figures / f"{model}_ligand_atoms_contact_red_top_occlusion_blue.svg").write_text(drawer.GetDrawingText(), encoding="utf-8")


def write_complex_and_pymol(paths: Mapping[str, Path], pose) -> None:
    from rdkit import Chem

    selected_sdf = paths["analysis"] / "selected_factor_xa_05_pose.sdf"
    writer = Chem.SDWriter(str(selected_sdf))
    writer.write(pose)
    writer.close()
    pymol = paths["analysis"] / "view_selected_pose.pml"
    receptor = paths["prepared"] / f"{PDB_ID}_receptor_clean.pdb"
    pymol.write_text(textwrap.dedent(f"""\
        reinitialize
        load {receptor}, receptor
        load {selected_sdf}, ligand
        hide everything
        show cartoon, receptor
        show sticks, ligand
        select contact_residues, byres (receptor within 4.0 of ligand)
        show sticks, contact_residues
        color cyan, receptor
        color yellow, ligand
        color salmon, contact_residues
        set stick_radius, 0.18
        orient ligand
        zoom ligand, 12
        bg_color white
    """), encoding="utf-8")


def formal_docking_and_analysis(args, paths: Mapping[str, Path], preparation: Mapping[str, object], redock_summary: Mapping[str, object]) -> Dict[str, object]:
    formal_dir = ensure_dir(paths["docking"])
    analysis_dir = ensure_dir(paths["analysis"])
    pose_rows = run_seed_set(args, preparation, "target_pdbqt", formal_dir)
    cluster_rows, selection, selected_pose = cluster_formal_poses(args, pose_rows, formal_dir)
    write_complex_and_pymol(paths, selected_pose)

    protein_atoms, _ = parse_mmcif_protein_atoms(Path(preparation["files"]["cif"]))
    target_reference = read_sdf_molecules(Path(preparation["files"]["target_sdf"]))[0]
    residue_rows, atom_rows, contact_pairs, mapping_method = contacts_for_pose(
        protein_atoms, selected_pose, target_reference, args.contact_cutoff,
    )
    write_rows(analysis_dir / "selected_pose_protein_contacts.csv", residue_rows)
    write_rows(analysis_dir / "selected_pose_ligand_atom_contacts.csv", atom_rows)
    write_rows(analysis_dir / "selected_pose_residue_atom_contact_pairs.csv", [
        {"uniprot_pos": residue, "stage2_atom_index": atom} for residue, atom in sorted(contact_pairs)
    ])
    resolved_uniprot = {atom.uniprot_pos for atom in protein_atoms if atom.uniprot_pos is not None}
    contact_residues = {residue for residue, _ in contact_pairs}
    contact_atoms = {atom for _, atom in contact_pairs}
    ligand_atom_count = remove_hydrogens(target_reference).GetNumAtoms()
    all_metrics: List[Dict[str, object]] = []
    model_plot_scores: Dict[str, Dict[str, Dict[int, float]]] = {}
    for model in ("KinetX", "2773"):
        rows, scores = compare_one_model(
            model, args.stage2_root, resolved_uniprot, contact_residues,
            ligand_atom_count, contact_atoms, contact_pairs,
        )
        all_metrics.extend(rows)
        model_plot_scores[model] = scores
    summary_rows = summarize_metrics(all_metrics)
    write_rows(analysis_dir / "occlusion_contact_metrics_per_seed.csv", all_metrics)
    write_rows(analysis_dir / "occlusion_contact_metrics_summary.csv", summary_rows)
    create_plots(paths, redock_summary, pose_rows, contact_residues, contact_atoms, target_reference, model_plot_scores)

    report_lines = [
        "# Factor Xa Stage-3 docking summary", "",
        f"- Structure: {PDB_ID}; native ligand: {NATIVE_LIGAND}; target: {TARGET_COMPOUND_ID}.",
        f"- Native-redocking gate: **{'PASS' if redock_summary['passed'] else 'FAIL (override used)'}**, "
        f"{redock_summary['observed_passing_seeds']}/{len(args.seeds)} rank-1 poses at RMSD <= {args.redock_rmsd:.1f} A.",
        f"- Selected formal pose: seed {selection['selected_pose']['seed']}, rank {selection['selected_pose']['pose_rank']}, "
        f"Vina affinity {selection['selected_pose']['affinity']:.3f} kcal/mol.",
        f"- Selected cluster support: {selection['selected_cluster']['n_unique_seeds']}/{len(args.seeds)} seeds; "
        f"{selection['selected_cluster']['n_poses']} eligible poses.",
        f"- Contact cutoff: {args.contact_cutoff:.1f} A; protein contact residues: {len(contact_residues)}; "
        f"ligand contact atoms: {len(contact_atoms)}; residue-atom pairs: {len(contact_pairs)}.",
        f"- Pose-to-Stage-2 atom mapping: `{mapping_method}`.", "",
        "## Occlusion/contact comparison (five MGCA checkpoints remain separate)", "",
        "| Training data | Feature level | Positives | Mean top-K hits | Mean ROC-AUC | Mean AP |", 
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['training_dataset']} | {row['feature_level']} | {row['positive_count']} | "
            f"{format_optional(row['mean_topk_hits'], 2)} | {format_optional(row['mean_roc_auc'])} | "
            f"{format_optional(row['mean_average_precision'])} |"
        )
    report_lines.extend([
        "", "## Interpretation boundary", "",
        "Docking supplies a static, hypothesis-generating pose. Agreement with 4 A contacts supports spatial plausibility of "
        "the post-hoc occlusion signal, but neither the Vina score nor a pose validates koff/residence time. The Stage-2 "
        "analysis remains post-hoc occlusion sensitivity, not native residue-atom co-attention.", "",
    ])
    (analysis_dir / "factor_xa_stage3_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    payload = {
        "protocol": "factor_xa_stage3_docking_and_occlusion_contact_v1",
        "created_at_utc": now_utc(), "redocking_gate": redock_summary,
        "formal_pose_selection": selection, "contact_cutoff_A": args.contact_cutoff,
        "protein_contact_residue_count": len(contact_residues),
        "ligand_contact_atom_count": len(contact_atoms), "contact_pair_count": len(contact_pairs),
        "pose_atom_mapping_method": mapping_method,
        "interpretation_boundary": "Static docking plausibility and post-hoc occlusion/contact comparison; not koff validation and not native co-attention.",
    }
    write_json(analysis_dir / "analysis_manifest.json", payload)
    return payload


def environment_audit() -> Dict[str, object]:
    import gemmi
    import meeko
    import prolif
    import rdkit
    import spyrmsd
    import vina

    return {
        "python": sys.version, "python_executable": sys.executable,
        "platform": sys.platform, "vina": vina.__version__, "meeko": meeko.__version__,
        "rdkit": rdkit.__version__, "prolif": prolif.__version__,
        "gemmi": gemmi.__version__, "spyrmsd": getattr(spyrmsd, "__version__", "unknown"),
    }


def build_paths(args) -> Dict[str, Path]:
    return {
        "work": args.work_dir, "output": args.output_dir,
        "inputs": args.output_dir / "inputs", "prepared": args.output_dir / "prepared",
        "redocking": args.output_dir / "redocking", "docking": args.output_dir / "formal_docking",
        "analysis": args.output_dir / "analysis", "figures": args.output_dir / "figures",
        "logs": args.output_dir / "logs",
    }


def parse_args(argv: Optional[Sequence[str]] = None):
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--stage2-root", type=Path, default=repository_root / "results" / "case_study" / "factor_xa_stage2")
    parser.add_argument("--output-dir", type=Path, default=repository_root / "results" / "case_study" / "factor_xa_stage3_docking")
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--cpu", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--exhaustiveness", type=int, default=32)
    parser.add_argument("--n-poses", type=int, default=20)
    parser.add_argument("--energy-range", type=float, default=5.0)
    parser.add_argument("--box-padding", type=float, default=5.0)
    parser.add_argument("--box-min", type=float, default=22.0)
    parser.add_argument("--box-max", type=float, default=30.0)
    parser.add_argument("--redock-rmsd", type=float, default=2.0)
    parser.add_argument("--redock-min-seed-passes", type=int, default=5)
    parser.add_argument("--cluster-energy-window", type=float, default=2.0)
    parser.add_argument("--cluster-rmsd", type=float, default=2.0)
    parser.add_argument("--contact-cutoff", type=float, default=4.0)
    parser.add_argument("--allow-redock-failure", action="store_true", help="Diagnostic only; do not use for final case-study results")
    parser.add_argument("--recompute", action="store_true", help="Re-run completed Vina seeds")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if len(args.seeds) < args.redock_min_seed_passes:
        raise ValueError("redock-min-seed-passes cannot exceed number of seeds")
    paths = build_paths(args)
    for path in paths.values():
        if isinstance(path, Path):
            ensure_dir(path)
    audit = environment_audit()
    write_json(paths["output"] / "environment.json", audit)
    print(f"[1/4] Preparing {PDB_ID}, {NATIVE_LIGAND}, and {TARGET_COMPOUND_ID}...")
    preparation = prepare(args, paths)
    print("[2/4] Native-ligand redocking gate...")
    redock_summary = redock(args, paths, preparation)
    print("[3/4] Formal target docking, clustering, and contact analysis...")
    analysis = formal_docking_and_analysis(args, paths, preparation, redock_summary)
    final_manifest = {
        "protocol": "factor_xa_stage3_complete_v1", "completed_at_utc": now_utc(),
        "arguments": {key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value for key, value in vars(args).items()},
        "environment": audit, "preparation_manifest": str(paths["prepared"] / "preparation_manifest.json"),
        "redocking_manifest": str(paths["redocking"] / "redocking_gate.json"),
        "analysis_manifest": str(paths["analysis"] / "analysis_manifest.json"),
        "redocking_passed": bool(redock_summary["passed"]),
        "selected_pose": analysis["formal_pose_selection"]["selected_pose"],
    }
    write_json(paths["output"] / "stage3_manifest.json", final_manifest)
    print("[4/4] Completed.")
    print(f"Results: {paths['output']}")
    print(f"Summary: {paths['analysis'] / 'factor_xa_stage3_summary.md'}")


if __name__ == "__main__":
    main()
