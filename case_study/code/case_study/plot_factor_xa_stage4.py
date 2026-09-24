#!/usr/bin/env python3
"""Generate a HyperAttentionDTI-style Factor Xa structure/sequence figure.

The colour is a post-hoc occlusion sensitivity score (delta pKoff), not a
native attention weight.  The script consumes existing Stage-2 and Stage-3
outputs and does not retrain or run MGCA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image


TARGET_UNIPROT = "P00742"
TARGET_COMPOUND = "factor_xa_05"
PRIMARY_CONFIG = "w16_s4_sequence_mean"
CONTACT_CUTOFF_A = 4.0
SEQUENCE_LENGTH = 488
DISPLAY_START = 235
DISPLAY_END = 488
RESIDUES_PER_ROW = 50
STABLE_SIGN_THRESHOLD = 0.8


AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the Factor Xa Stage-4 occlusion/structure figure."
    )
    parser.add_argument("--model", default="2773", choices=("2773", "KinetX"))
    parser.add_argument(
        "--stage2-root",
        type=Path,
        default=Path("<BIO_PROJECT_ROOT>/case_study_outputs/factor_xa_stage2"),
    )
    parser.add_argument(
        "--stage3-root",
        type=Path,
        default=Path("<BIO_PROJECT_ROOT>/case_study_outputs/factor_xa_stage3_docking"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--compound-id", default=TARGET_COMPOUND)
    parser.add_argument("--config-key", default=PRIMARY_CONFIG)
    parser.add_argument("--contact-cutoff", type=float, default=CONTACT_CUTOFF_A)
    parser.add_argument("--pymol", default=None, help="PyMOL executable path")
    parser.add_argument(
        "--skip-structure",
        action="store_true",
        help="Generate data and the sequence panel only (for diagnostics).",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    return path


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def reconstruct_sequence(windows: pd.DataFrame, length: int) -> str:
    residues: List[Optional[str]] = [None] * length
    for row in windows.itertuples(index=False):
        start = int(row.window_start)
        end = int(row.window_end)
        fragment = str(row.window_sequence).strip()
        if len(fragment) != end - start + 1:
            raise ValueError(
                f"Window {start}-{end} has sequence length {len(fragment)}"
            )
        for offset, aa in enumerate(fragment):
            pos = start + offset
            previous = residues[pos - 1]
            if previous is not None and previous != aa:
                raise ValueError(
                    f"Conflicting amino acid at UniProt position {pos}: "
                    f"{previous} versus {aa}"
                )
            residues[pos - 1] = aa
    missing = [i + 1 for i, aa in enumerate(residues) if aa is None]
    if missing:
        raise ValueError(f"Primary windows do not cover sequence positions: {missing[:20]}")
    return "".join(str(aa) for aa in residues)


def residue_scores(
    summary_path: Path,
    compound_id: str,
    config_key: str,
    sequence_length: int,
) -> Tuple[str, List[int], List[Dict[str, object]]]:
    table = pd.read_csv(summary_path)
    required = {
        "compound_id", "config_key", "window_start", "window_end",
        "window_sequence", "delta_by_seed_json",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing Stage-2 columns: {sorted(missing)}")

    windows = table[
        (table["compound_id"].astype(str) == compound_id)
        & (table["config_key"].astype(str) == config_key)
    ].copy()
    if windows.empty:
        raise ValueError(
            f"No Stage-2 rows for compound={compound_id}, config={config_key}"
        )
    windows = windows.sort_values(["window_start", "window_end"])
    sequence = reconstruct_sequence(windows, sequence_length)

    parsed = [json.loads(value) for value in windows["delta_by_seed_json"]]
    seed_sets = [set(int(key) for key in item) for item in parsed]
    if len({tuple(sorted(values)) for values in seed_sets}) != 1:
        raise ValueError("Stage-2 windows do not contain a consistent seed set")
    seeds = sorted(seed_sets[0])

    values: Dict[int, List[List[float]]] = {
        seed: [[] for _ in range(sequence_length)] for seed in seeds
    }
    for row, seed_values in zip(windows.itertuples(index=False), parsed):
        start, end = int(row.window_start), int(row.window_end)
        for seed in seeds:
            delta = float(seed_values[str(seed)])
            for pos in range(start, end + 1):
                values[seed][pos - 1].append(delta)

    rows: List[Dict[str, object]] = []
    for index, aa in enumerate(sequence):
        pos = index + 1
        per_seed = []
        n_windows = None
        for seed in seeds:
            contributions = values[seed][index]
            if not contributions:
                raise ValueError(f"No occlusion window covers UniProt position {pos}")
            n_windows = len(contributions)
            per_seed.append(float(np.mean(contributions)))
        mean = float(np.mean(per_seed))
        sd = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
        positive_fraction = float(np.mean(np.asarray(per_seed) > 0.0))
        negative_fraction = float(np.mean(np.asarray(per_seed) < 0.0))
        sign_consistency = max(positive_fraction, negative_fraction)
        row: Dict[str, object] = {
            "uniprot": TARGET_UNIPROT,
            "uniprot_pos": pos,
            "amino_acid": aa,
            "n_covering_windows": int(n_windows or 0),
            "mean_delta_pkoff": mean,
            "sd_delta_pkoff": sd,
            "positive_seed_fraction": positive_fraction,
            "sign_consistency": sign_consistency,
            "stable_4_of_5": bool(sign_consistency >= STABLE_SIGN_THRESHOLD),
        }
        for seed, value in zip(seeds, per_seed):
            row[f"delta_seed_{seed}"] = value
        rows.append(row)
    return sequence, seeds, rows


def normalize_cif_token(value: str) -> str:
    value = value.strip()
    return "" if value in {".", "?"} else value


def cif_rows_from_path(cif_path: Path, tags: Sequence[str]) -> List[Tuple[str, ...]]:
    """Read selected mmCIF columns with gemmi, falling back to Biopython."""
    try:
        import gemmi

        block = gemmi.cif.read_file(str(cif_path)).sole_block()
        columns = [list(block.find_values(tag)) for tag in tags]
    except ImportError:
        try:
            from Bio.PDB.MMCIF2Dict import MMCIF2Dict
        except ImportError as exc:
            raise RuntimeError(
                "An mmCIF reader is required. Install `gemmi` (recommended) "
                "or `biopython`."
            ) from exc
        dictionary = MMCIF2Dict(str(cif_path))
        columns = []
        for tag in tags:
            value = dictionary.get(tag, [])
            columns.append(value if isinstance(value, list) else [value])
    if any(not column for column in columns):
        missing = [tag for tag, column in zip(tags, columns) if not column]
        raise RuntimeError("Missing mmCIF tags: " + ", ".join(missing))
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise RuntimeError(f"mmCIF column length mismatch: {sorted(lengths)}")
    return [tuple(str(value) for value in row) for row in zip(*columns)]


def parse_cif_protein(
    cif_path: Path,
) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, str, str], int]]:
    ref_tags = [
        "_struct_ref_seq.pdbx_strand_id", "_struct_ref_seq.seq_align_beg",
        "_struct_ref_seq.seq_align_end", "_struct_ref_seq.pdbx_db_accession",
        "_struct_ref_seq.db_align_beg", "_struct_ref_seq.db_align_end",
    ]
    mappings = []
    for row in cif_rows_from_path(cif_path, ref_tags):
        if row[3].strip() != TARGET_UNIPROT:
            continue
        for chain in row[0].split(","):
            mappings.append({
                "chain": chain.strip(),
                "seq_beg": int(row[1]),
                "seq_end": int(row[2]),
                "db_beg": int(row[4]),
                "db_end": int(row[5]),
            })
    if not mappings:
        raise RuntimeError(f"No {TARGET_UNIPROT} mapping in {cif_path}")

    atom_tags = [
        "_atom_site.group_PDB", "_atom_site.type_symbol", "_atom_site.label_atom_id",
        "_atom_site.label_alt_id", "_atom_site.label_comp_id", "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code", "_atom_site.Cartn_x", "_atom_site.Cartn_y",
        "_atom_site.Cartn_z", "_atom_site.auth_seq_id", "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id", "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    atoms: List[Dict[str, object]] = []
    residue_map: Dict[Tuple[str, str, str], int] = {}
    seen_atoms: Set[Tuple[str, str, str, str]] = set()
    for row in cif_rows_from_path(cif_path, atom_tags):
        if row[0] != "ATOM" or row[14] != "1":
            continue
        element = row[1].strip().upper()
        if element in {"H", "D"}:
            continue
        alt = normalize_cif_token(row[3])
        if alt not in {"", "A"}:
            continue
        label_seq_token = normalize_cif_token(row[5])
        if not label_seq_token:
            continue
        label_seq = int(label_seq_token)
        chain = row[12].strip()
        auth_seq = row[10].strip()
        insertion = normalize_cif_token(row[6])
        atom_name = row[13].strip()
        atom_key = (chain, auth_seq, insertion, atom_name)
        if atom_key in seen_atoms:
            continue
        seen_atoms.add(atom_key)
        uniprot_pos: Optional[int] = None
        for mapping in mappings:
            if (
                mapping["chain"] == chain
                and int(mapping["seq_beg"]) <= label_seq <= int(mapping["seq_end"])
            ):
                uniprot_pos = int(mapping["db_beg"]) + label_seq - int(mapping["seq_beg"])
                break
        if uniprot_pos is None:
            continue
        residue_key = (chain, auth_seq, insertion)
        previous = residue_map.get(residue_key)
        if previous is not None and previous != uniprot_pos:
            raise RuntimeError(f"Ambiguous residue mapping for {residue_key}")
        residue_map[residue_key] = uniprot_pos
        atoms.append({
            "element": element,
            "atom_name": atom_name,
            "residue_name": row[11].strip(),
            "auth_chain": chain,
            "auth_seq": auth_seq,
            "insertion_code": insertion,
            "uniprot_pos": uniprot_pos,
            "x": float(row[7]), "y": float(row[8]), "z": float(row[9]),
        })
    if not atoms:
        raise RuntimeError(f"No mapped protein atoms parsed from {cif_path}")
    return atoms, residue_map


def parse_pdb_heavy_atoms(path: Path) -> List[Tuple[str, float, float, float]]:
    atoms = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line[:6].strip() not in {"ATOM", "HETATM"}:
            continue
        element = line[76:78].strip().upper()
        if not element:
            element = "".join(char for char in line[12:16] if char.isalpha())[:1].upper()
        if element in {"H", "D"}:
            continue
        atoms.append((element, float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if not atoms:
        raise RuntimeError(f"No heavy atoms parsed from {path}")
    return atoms


def native_contacts(
    protein_atoms: Sequence[Mapping[str, object]],
    ligand_pdb: Path,
    cutoff: float,
) -> List[Dict[str, object]]:
    ligand = parse_pdb_heavy_atoms(ligand_pdb)
    ligand_xyz = np.asarray([[atom[1], atom[2], atom[3]] for atom in ligand], dtype=float)
    grouped: Dict[Tuple[str, str, str, str, int], List[Mapping[str, object]]] = defaultdict(list)
    for atom in protein_atoms:
        key = (
            str(atom["auth_chain"]), str(atom["auth_seq"]),
            str(atom["insertion_code"]), str(atom["residue_name"]),
            int(atom["uniprot_pos"]),
        )
        grouped[key].append(atom)

    result: List[Dict[str, object]] = []
    for key, atoms in grouped.items():
        xyz = np.asarray([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)
        distances = np.linalg.norm(xyz[:, None, :] - ligand_xyz[None, :, :], axis=2)
        minimum = float(np.min(distances))
        if minimum <= cutoff:
            result.append({
                "auth_chain": key[0], "auth_seq": key[1],
                "insertion_code": key[2], "residue_name": key[3],
                "residue_one_letter": AA1.get(key[3], "X"),
                "uniprot_pos": key[4], "min_distance_A": minimum,
                "contact_cutoff_A": cutoff,
            })
    result.sort(key=lambda row: (int(row["uniprot_pos"]), str(row["auth_chain"])))
    return result


def load_docked_contacts(path: Path) -> List[Dict[str, object]]:
    table = pd.read_csv(path, keep_default_na=False)
    required = {
        "auth_chain", "auth_seq", "insertion_code", "residue_name", "uniprot_pos"
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing docking contact columns: {sorted(missing)}")
    rows = []
    for row in table.to_dict(orient="records"):
        row["uniprot_pos"] = int(row["uniprot_pos"])
        row["auth_seq"] = str(row["auth_seq"])
        row["insertion_code"] = str(row["insertion_code"])
        rows.append(row)
    return rows


def scored_pdb(
    receptor_pdb: Path,
    destination: Path,
    residue_map: Mapping[Tuple[str, str, str], int],
    scores: Mapping[int, float],
    vmax: float,
) -> None:
    output = []
    for line in receptor_pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("ATOM") and len(line) >= 66:
            key = (line[21:22].strip(), line[22:26].strip(), line[26:27].strip())
            position = residue_map.get(key)
            score = scores.get(position, 0.0) if position is not None else 0.0
            scaled = float(np.clip(score / vmax * 99.0, -99.0, 99.0))
            line = line[:60] + f"{scaled:6.2f}" + line[66:]
        output.append(line)
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


def pymol_residue_expression(rows: Iterable[Mapping[str, object]]) -> str:
    groups: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        resi = str(row["auth_seq"]) + str(row.get("insertion_code", ""))
        groups[str(row["auth_chain"])].append(resi)
    parts = []
    for chain, residues in sorted(groups.items()):
        unique = sorted(set(residues), key=lambda value: (float(value.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ") or 0), value))
        parts.append(f"(chain {chain} and resi {'+'.join(unique)})")
    return " or ".join(parts) if parts else "none"


def write_pymol_script(
    path: Path,
    scored_receptor: Path,
    pose_sdf: Path,
    output_png: Path,
    native_rows: Sequence[Mapping[str, object]],
    docked_rows: Sequence[Mapping[str, object]],
    view_mode: str = "overview",
) -> None:
    if view_mode not in {"overview", "pocket"}:
        raise ValueError(f"Unsupported PyMOL view mode: {view_mode}")
    native_keys = {
        (str(r["auth_chain"]), str(r["auth_seq"]), str(r.get("insertion_code", "")))
        for r in native_rows
    }
    dock_keys = {
        (str(r["auth_chain"]), str(r["auth_seq"]), str(r.get("insertion_code", "")))
        for r in docked_rows
    }
    native_only = [r for r in native_rows if (
        str(r["auth_chain"]), str(r["auth_seq"]), str(r.get("insertion_code", ""))
    ) not in dock_keys]
    dock_only = [r for r in docked_rows if (
        str(r["auth_chain"]), str(r["auth_seq"]), str(r.get("insertion_code", ""))
    ) not in native_keys]
    overlap = [r for r in docked_rows if (
        str(r["auth_chain"]), str(r["auth_seq"]), str(r.get("insertion_code", ""))
    ) in native_keys]

    all_by_pdb = {
        (str(r["auth_chain"]), str(r["auth_seq"]), str(r.get("insertion_code", ""))): r
        for r in list(native_rows) + list(docked_rows)
    }
    preferred = [] if view_mode == "overview" else [
        ("A", "57", "", "His57 (H276)", (1.8, 1.5, 0.0)),
        ("A", "189", "", "Asp189 (D413)", (1.8, -1.5, 0.0)),
        ("A", "195", "", "Ser195 (S419)", (-1.8, -1.5, 0.0)),
    ]
    label_commands = []
    for chain, resi, ins, label, offset in preferred:
        row = all_by_pdb.get((chain, resi, ins))
        if row is None:
            continue
        selection = f"receptor and chain {chain} and resi {resi}{ins}"
        label_commands.append(
            f'label ({selection} and name CA), "{label}"'
        )
        label_commands.append(
            f"set label_position, [{offset[0]}, {offset[1]}, {offset[2]}], {selection}"
        )

    if view_mode == "pocket":
        view_commands = """select pocket_view, byres (receptor within 8.0 of factor_xa_05)
select view_focus, pocket_view or factor_xa_05
orient view_focus
center factor_xa_05
zoom view_focus, 3.5
turn x, -10
turn y, 20
turn z, 5"""
        cartoon_transparency = 0.18
        label_connector = 1
    else:
        view_commands = """orient receptor
zoom receptor, 3
turn x, -8
turn y, 18"""
        cartoon_transparency = 0.05
        label_connector = 0

    content = f"""reinitialize
load {scored_receptor.as_posix()}, receptor
load {pose_sdf.as_posix()}, factor_xa_05
hide everything, all
show cartoon, receptor
color gray80, receptor
spectrum b, blue_white_red, receptor and chain A, minimum=-99, maximum=99
color gray80, receptor and chain B
show sticks, factor_xa_05
color cyan, factor_xa_05
select native_only, receptor and ({pymol_residue_expression(native_only)})
select dock_only, receptor and ({pymol_residue_expression(dock_only)})
select contact_overlap, receptor and ({pymol_residue_expression(overlap)})
show sticks, (native_only or dock_only or contact_overlap) and not name N+C+O
color red, native_only
color orange, dock_only
color magenta, contact_overlap
set stick_radius, 0.13, native_only or dock_only or contact_overlap
set stick_radius, 0.22, factor_xa_05
set cartoon_transparency, {cartoon_transparency}
set cartoon_fancy_helices, 1
set depth_cue, 0
set antialias, 2
set ray_opaque_background, 0
set ray_trace_mode, 1
set label_color, black
set label_size, 18
set label_outline_color, white
set label_connector, {label_connector}
bg_color white
{view_commands}
{chr(10).join(label_commands)}
png {output_png.as_posix()}, 1800, 1600, dpi=300, ray=1
quit
"""
    path.write_text(content, encoding="utf-8")


def locate_pymol(explicit: Optional[str]) -> List[str]:
    if explicit:
        path = shutil.which(explicit) or explicit
        return [path]
    executable = shutil.which("pymol")
    if executable:
        return [executable]
    if importlib.util.find_spec("pymol") is not None:
        return [sys.executable, "-m", "pymol"]
    raise RuntimeError(
        "PyMOL was not found. Install the open-source build in this Python 3.10 "
        "environment with `pip install pymol-open-source`, or pass --pymol PATH."
    )


def render_structure(pymol_command: Sequence[str], pml: Path, png: Path) -> None:
    command = list(pymol_command) + ["-cq", str(pml)]
    completed = subprocess.run(command, text=True, capture_output=True)
    log = png.with_suffix(".pymol.log")
    log.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT\n" + completed.stdout
        + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0 or not png.is_file() or png.stat().st_size == 0:
        raise RuntimeError(f"PyMOL rendering failed; inspect {log}")


def write_reference_style_pymol_script(
    path: Path,
    receptor_pdb: Path,
    pose_sdf: Path,
    output_png: Path,
    high_occlusion_rows: Sequence[Mapping[str, object]],
) -> None:
    """Write a clean structure panel matching the cited paper's visual style."""
    high_expression = pymol_residue_expression(high_occlusion_rows)
    content = f"""reinitialize
load {receptor_pdb.as_posix()}, receptor
load {pose_sdf.as_posix()}, factor_xa_05
hide everything, all
show cartoon, receptor and chain A
color gray70, receptor
select high_occlusion, receptor and ({high_expression})
color red, high_occlusion
show sticks, factor_xa_05
color cyan, factor_xa_05
select labelled_residues, receptor and chain A and resi 189+195
show sticks, labelled_residues and not name N+C+O
color gray40, labelled_residues
set stick_radius, 0.14, labelled_residues
set stick_radius, 0.22, factor_xa_05
set cartoon_fancy_helices, 1
set cartoon_transparency, 0.0
set depth_cue, 0
set antialias, 2
set ray_opaque_background, 0
set ray_trace_mode, 0
set label_color, black
set label_outline_color, white
set label_size, 14
set label_connector, 0
bg_color white
orient receptor and chain A
zoom receptor and chain A, 3
turn x, -8
turn y, 18
label (receptor and chain A and resi 189 and name CA), "ASP189"
set label_position, [1.5, -0.8, 0.0], receptor and chain A and resi 189
label (receptor and chain A and resi 195 and name CA), "SER195"
set label_position, [-1.5, 1.0, 0.0], receptor and chain A and resi 195
png {output_png.as_posix()}, 1400, 1800, dpi=300, ray=1
quit
"""
    path.write_text(content, encoding="utf-8")


def robust_normalized_occlusion(
    residue_rows: Sequence[Mapping[str, object]],
    resolved_positions: Set[int],
) -> Tuple[Dict[int, float], float, float]:
    score_map = {
        int(row["uniprot_pos"]): float(row["mean_delta_pkoff"])
        for row in residue_rows
    }
    values = np.asarray([
        score_map[pos]
        for pos in range(DISPLAY_START, DISPLAY_END + 1)
        if pos in resolved_positions
    ], dtype=float)
    lower = float(np.quantile(values, 0.05))
    upper = float(np.quantile(values, 0.95))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower, upper = float(np.min(values)), float(np.max(values))
    if upper <= lower:
        upper = lower + 1.0
    normalized = {
        pos: float(np.clip((score - lower) / (upper - lower), 0.0, 1.0))
        for pos, score in score_map.items()
    }
    return normalized, lower, upper


def plot_reference_style_figure(
    structure_png: Path,
    sequence: str,
    normalized_scores: Mapping[int, float],
    resolved_positions: Set[int],
    experimental_contacts: Set[int],
    output_base: Path,
    model: str,
) -> None:
    """Create a paper-style structure plus number/sequence/heat-strip figure."""
    structure = Image.open(structure_png).convert("RGBA")
    cmap = plt.get_cmap("jet")
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    positions = list(range(DISPLAY_START, DISPLAY_END + 1))
    per_row = 50
    n_rows = math.ceil(len(positions) / per_row)

    fig = plt.figure(figsize=(15.2, 7.0))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.78, 1.85, 0.075], wspace=0.08)
    ax_structure = fig.add_subplot(grid[0, 0])
    ax_sequence = fig.add_subplot(grid[0, 1])
    cax = fig.add_subplot(grid[0, 2])

    ax_structure.imshow(structure)
    ax_structure.axis("off")
    ax_structure.text(
        0.01, 0.99, "A", transform=ax_structure.transAxes,
        ha="left", va="top", fontsize=25, fontweight="bold",
    )

    block_height = 3.15
    ax_sequence.set_xlim(-13.0, per_row + 0.7)
    ax_sequence.set_ylim(n_rows * block_height + 0.8, -0.7)
    ax_sequence.axis("off")

    for block_index in range(n_rows):
        start_index = block_index * per_row
        block_positions = positions[start_index:start_index + per_row]
        row_width = len(block_positions)
        y0 = block_index * block_height

        ax_sequence.add_patch(Rectangle(
            (0, y0), row_width, 0.58,
            facecolor="#f1f1e6", edgecolor="#333333", linewidth=0.8,
        ))
        ax_sequence.text(-1.0, y0 + 0.29, "Number", ha="right", va="center", fontsize=8.5)
        ax_sequence.text(-1.0, y0 + 1.12, "Sequence", ha="right", va="center", fontsize=8.5)
        ax_sequence.text(
            -1.0, y0 + 2.05, "Normalized\nocclusion", ha="right", va="center", fontsize=8.2,
        )

        for column, pos in enumerate(block_positions):
            if pos % 10 == 0 or pos == block_positions[-1]:
                ax_sequence.plot(
                    [column + 0.5, column + 0.5], [y0 + 0.05, y0 + 0.50],
                    color="#555555", linewidth=0.55,
                )
                ax_sequence.text(
                    column + 0.5, y0 + 0.22, str(pos),
                    ha="center", va="center", fontsize=6.6, color="#333333",
                )
            if pos in experimental_contacts:
                ax_sequence.scatter(
                    column + 0.5, y0 - 0.10, s=24, marker="o",
                    c="#e31a1c", edgecolors="none", zorder=5,
                )

            ax_sequence.text(
                column + 0.5, y0 + 1.12, sequence[pos - 1],
                ha="center", va="center", fontsize=7.0,
                family="DejaVu Sans Mono", color="#222222",
            )
            ax_sequence.plot(
                [column + 0.12, column + 0.88], [y0 + 1.48, y0 + 1.48],
                color="#333333", linewidth=0.35,
            )

            if pos in resolved_positions:
                face = cmap(norm(normalized_scores[pos]))
            else:
                face = (0.78, 0.78, 0.78, 1.0)
            ax_sequence.add_patch(Rectangle(
                (column, y0 + 1.67), 1.0, 0.66,
                facecolor=face, edgecolor="white", linewidth=0.12,
            ))

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, cax=cax, orientation="vertical")
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label(
        "Normalized mean occlusion sensitivity", fontsize=9, labelpad=8,
    )

    ax_sequence.legend(
        handles=[Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="#e31a1c",
            markersize=6, label="Experimental contact site (1NFU–RRP, ≤4 Å)",
        )],
        loc="lower left", bbox_to_anchor=(0.0, -0.035), frameon=False, fontsize=8.5,
    )
    fig.text(
        0.52, 0.018,
        f"{model} five-checkpoint mean; 0–1 robust normalization of ΔpKoff. "
        "Post-hoc occlusion sensitivity, not model-native attention.",
        ha="center", va="bottom", fontsize=8.5,
    )
    fig.subplots_adjust(left=0.015, right=0.965, top=0.985, bottom=0.08)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            output_base.with_suffix(f".{extension}"), dpi=300,
            bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)


def plot_sequence_panel(
    sequence: str,
    residue_rows: Sequence[Mapping[str, object]],
    resolved_positions: Set[int],
    native_positions: Set[int],
    docked_positions: Set[int],
    model: str,
    output_png: Path,
    vmax: float,
) -> None:
    score_by_pos = {int(row["uniprot_pos"]): float(row["mean_delta_pkoff"]) for row in residue_rows}
    stable_by_pos = {int(row["uniprot_pos"]): bool(row["stable_4_of_5"]) for row in residue_rows}
    positions = list(range(DISPLAY_START, DISPLAY_END + 1))
    n_rows = math.ceil(len(positions) / RESIDUES_PER_ROW)
    cmap = plt.get_cmap("RdBu_r")
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(13.2, 5.6))
    ax.set_xlim(-8, RESIDUES_PER_ROW + 1)
    ax.set_ylim(n_rows + 0.9, -0.8)
    ax.axis("off")
    ax.set_title(
        f"B  Factor Xa sequence: {model} five-checkpoint mean occlusion sensitivity",
        loc="left", fontsize=12, fontweight="bold", pad=10,
    )

    for index, pos in enumerate(positions):
        row_index, column = divmod(index, RESIDUES_PER_ROW)
        resolved = pos in resolved_positions
        face = cmap(norm(score_by_pos[pos])) if resolved else (0.84, 0.84, 0.84, 1.0)
        stable = stable_by_pos[pos] and resolved
        rectangle = Rectangle(
            (column, row_index), 1, 1,
            facecolor=face,
            edgecolor="white",
            linewidth=0.20,
        )
        ax.add_patch(rectangle)
        if stable:
            ax.plot(
                [column + 0.12, column + 0.88],
                [row_index + 0.94, row_index + 0.94],
                color="black", linewidth=0.85, solid_capstyle="butt", zorder=3,
            )
        luminance = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]
        text_color = "white" if resolved and luminance < 0.43 else "black"
        ax.text(
            column + 0.5, row_index + 0.53, sequence[pos - 1],
            ha="center", va="center", fontsize=6.2, color=text_color,
            family="DejaVu Sans Mono",
        )
        if pos in native_positions:
            ax.scatter(column + 0.23, row_index + 0.18, s=13, marker="o",
                       c="#d62728", edgecolors="white", linewidths=0.25, zorder=4)
        if pos in docked_positions:
            ax.scatter(column + 0.77, row_index + 0.82, s=18, marker="^",
                       c="#ff8c00", edgecolors="white", linewidths=0.25, zorder=4)

        if pos % 10 == 0:
            ax.text(column + 0.5, row_index - 0.08, str(pos), ha="center",
                    va="bottom", fontsize=6.5, color="#333333")

    for row_index in range(n_rows):
        start = DISPLAY_START + row_index * RESIDUES_PER_ROW
        end = min(start + RESIDUES_PER_ROW - 1, DISPLAY_END)
        ax.text(-0.7, row_index + 0.5, f"{start}–{end}", ha="right", va="center",
                fontsize=8.2, family="DejaVu Sans Mono")

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.22, 0.125, 0.64, 0.045])
    colorbar = fig.colorbar(scalar, cax=cax, orientation="horizontal")
    colorbar.set_label("Mean occlusion sensitivity, ΔpKoff (original − masked)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728",
               markeredgecolor="white", markersize=6, label="Experimental 1NFU–RRP contact (≤4 Å)"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#ff8c00",
               markeredgecolor="white", markersize=7, label="Docked factor_xa_05 contact (≤4 Å)"),
        Line2D([0], [0], marker="_", color="black", markersize=9,
               label="Occlusion sign consistent in ≥4/5 checkpoints"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#d6d6d6",
               markersize=7, label="Unresolved in 1NFU chain A"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               ncol=4, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.27)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def combine_panels(
    structure_png: Path,
    sequence_png: Path,
    output_base: Path,
    model: str,
    panel_a_title: str,
) -> None:
    structure = Image.open(structure_png).convert("RGBA")
    sequence = Image.open(sequence_png).convert("RGBA")
    fig = plt.figure(figsize=(16.0, 6.7))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.88, 2.10], wspace=0.02)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_a.imshow(structure)
    ax_a.axis("off")
    ax_a.set_title(panel_a_title, loc="left",
                   fontsize=12, fontweight="bold", pad=4)
    ax_b.imshow(sequence)
    ax_b.axis("off")
    ax_a.legend(handles=[
        Line2D([0], [0], color="#00bcd4", lw=4, label="Docked ligand"),
        Line2D([0], [0], color="#d62728", lw=4, label="Experimental-contact residue"),
        Line2D([0], [0], color="#ff8c00", lw=4, label="Dock-only contact residue"),
        Line2D([0], [0], color="#cc33cc", lw=4, label="Contact in both sets"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8)
    fig.suptitle(
        f"Factor Xa post-hoc occlusion interpretation ({model} model)",
        fontsize=14, fontweight="bold", y=0.99,
    )
    fig.text(
        0.01, 0.012,
        "Colour encodes post-hoc occlusion sensitivity, not model-native attention. "
        "Red contacts come from the experimental 1NFU–RRP complex; orange contacts come from the docked candidate pose.",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.01, right=0.995, top=0.94, bottom=0.07)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_base.with_suffix(f".{extension}"), dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    stage2_dir = args.stage2_root / args.model
    if args.output_dir is None:
        args.output_dir = args.stage3_root.parent / f"factor_xa_stage4_visualization_{args.model}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = require_file(stage2_dir / "protein_window_importance_summary.csv")
    cif_path = require_file(args.stage3_root / "inputs" / "1NFU.cif")
    receptor_pdb = require_file(args.stage3_root / "prepared" / "1NFU_receptor_clean.pdb")
    native_ligand_pdb = require_file(args.stage3_root / "prepared" / "1NFU_RRP_experimental.pdb")
    docked_contact_csv = require_file(
        args.stage3_root / "analysis" / "selected_pose_protein_contacts.csv"
    )
    pose_sdf = require_file(
        args.stage3_root / "analysis" / "selected_factor_xa_05_pose.sdf"
    )

    print(f"[1/9] Computing residue-level scores from {args.model} Stage-2 windows...")
    sequence, seeds, residue_rows = residue_scores(
        summary_csv, args.compound_id, args.config_key, SEQUENCE_LENGTH
    )
    residue_csv = args.output_dir / f"factor_xa_residue_occlusion_{args.model}.csv"
    write_csv(residue_csv, residue_rows)

    print("[2/9] Mapping 1NFU residues to UniProt and calculating native contacts...")
    protein_atoms, residue_map = parse_cif_protein(cif_path)
    # The main panel represents the heavy-chain catalytic domain (chain A).
    # Do not let the overlapping light-chain UniProt range make missing chain-A
    # residues appear resolved.
    resolved_positions = {
        int(atom["uniprot_pos"])
        for atom in protein_atoms
        if str(atom["auth_chain"]) == "A"
    }
    native_rows = native_contacts(protein_atoms, native_ligand_pdb, args.contact_cutoff)
    native_csv = args.output_dir / "1NFU_RRP_experimental_contacts_4A.csv"
    write_csv(native_csv, native_rows)
    docked_rows = load_docked_contacts(docked_contact_csv)

    for source_name, contact_rows in (("experimental", native_rows), ("docked", docked_rows)):
        for contact in contact_rows:
            pos = int(contact["uniprot_pos"])
            expected = sequence[pos - 1]
            observed = str(contact.get("residue_one_letter", AA1.get(str(contact["residue_name"]), "X")))
            if observed != "X" and observed != expected:
                raise RuntimeError(
                    f"{source_name} contact mapping mismatch at UniProt {pos}: "
                    f"structure={observed}, sequence={expected}"
                )

    display_values = np.asarray([
        float(row["mean_delta_pkoff"]) for row in residue_rows
        if DISPLAY_START <= int(row["uniprot_pos"]) <= DISPLAY_END
        and int(row["uniprot_pos"]) in resolved_positions
    ])
    vmax = float(np.quantile(np.abs(display_values), 0.95))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(np.abs(display_values))) or 1.0

    normalized_scores, normalization_lower, normalization_upper = robust_normalized_occlusion(
        residue_rows, resolved_positions
    )
    highlight_candidates = [
        pos for pos in range(DISPLAY_START, DISPLAY_END + 1)
        if pos in resolved_positions
    ]
    n_highlight = max(1, math.ceil(0.10 * len(highlight_candidates)))
    high_positions = set(sorted(
        highlight_candidates,
        key=lambda pos: (-normalized_scores[pos], pos),
    )[:n_highlight])
    high_occlusion_rows = [
        {
            "auth_chain": chain,
            "auth_seq": auth_seq,
            "insertion_code": insertion,
            "uniprot_pos": pos,
        }
        for (chain, auth_seq, insertion), pos in residue_map.items()
        if chain == "A" and pos in high_positions
    ]

    print("[3/9] Writing a score-annotated receptor and PyMOL scripts...")
    score_map = {
        int(row["uniprot_pos"]): float(row["mean_delta_pkoff"])
        for row in residue_rows
    }
    scored_receptor = args.output_dir / f"1NFU_receptor_occlusion_{args.model}.pdb"
    scored_pdb(receptor_pdb, scored_receptor, residue_map, score_map, vmax)
    structure_png = args.output_dir / f"factor_xa_structure_{args.model}.png"
    pml = args.output_dir / f"factor_xa_structure_{args.model}.pml"
    write_pymol_script(
        pml, scored_receptor, pose_sdf, structure_png, native_rows, docked_rows,
        view_mode="overview",
    )
    pocket_png = args.output_dir / f"factor_xa_pocket_zoom_{args.model}.png"
    pocket_pml = args.output_dir / f"factor_xa_pocket_zoom_{args.model}.pml"
    write_pymol_script(
        pocket_pml, scored_receptor, pose_sdf, pocket_png, native_rows, docked_rows,
        view_mode="pocket",
    )
    reference_structure_png = args.output_dir / f"factor_xa_reference_structure_{args.model}.png"
    reference_pml = args.output_dir / f"factor_xa_reference_structure_{args.model}.pml"
    write_reference_style_pymol_script(
        reference_pml, scored_receptor, pose_sdf, reference_structure_png,
        high_occlusion_rows,
    )

    print("[4/9] Rendering the sequence panel...")
    sequence_png = args.output_dir / f"factor_xa_sequence_occlusion_{args.model}.png"
    plot_sequence_panel(
        sequence, residue_rows, resolved_positions,
        {int(r["uniprot_pos"]) for r in native_rows},
        {int(r["uniprot_pos"]) for r in docked_rows},
        args.model, sequence_png, vmax,
    )

    composite_outputs: List[str] = []
    overview_outputs: List[str] = []
    reference_outputs: List[str] = []
    if args.skip_structure:
        print("[5/9] PyMOL rendering skipped by request.")
    else:
        pymol_command = locate_pymol(args.pymol)
        print("[5/9] Rendering the overview structure with PyMOL...")
        render_structure(pymol_command, pml, structure_png)
        print("[6/9] Rendering the binding-pocket close-up with PyMOL...")
        render_structure(pymol_command, pocket_pml, pocket_png)
        print("[7/9] Rendering the reference-style structure with PyMOL...")
        render_structure(pymol_command, reference_pml, reference_structure_png)
        print("[8/9] Combining overview and pocket panels...")
        overview_base = args.output_dir / f"factor_xa_occlusion_structure_{args.model}"
        combine_panels(
            structure_png, sequence_png, overview_base, args.model,
            "A  Factor Xa structural overview",
        )
        pocket_base = args.output_dir / f"factor_xa_occlusion_pocket_{args.model}"
        combine_panels(
            pocket_png, sequence_png, pocket_base, args.model,
            "A  Factor Xa binding-pocket close-up",
        )
        overview_outputs = [
            str(overview_base.with_suffix(f".{ext}")) for ext in ("png", "pdf", "svg")
        ]
        composite_outputs = [
            str(pocket_base.with_suffix(f".{ext}")) for ext in ("png", "pdf", "svg")
        ]
        print("[9/9] Creating the reference-style structure/sequence figure...")
        reference_base = args.output_dir / f"factor_xa_reference_style_{args.model}"
        plot_reference_style_figure(
            reference_structure_png, sequence, normalized_scores, resolved_positions,
            {int(row["uniprot_pos"]) for row in native_rows},
            reference_base, args.model,
        )
        reference_outputs = [
            str(reference_base.with_suffix(f".{ext}")) for ext in ("png", "pdf", "svg")
        ]

    manifest = {
        "protocol": "factor_xa_stage4_hyperattention_style_posthoc_occlusion_v2",
        "created_at_utc": now_utc(),
        "model": args.model,
        "compound_id": args.compound_id,
        "uniprot": TARGET_UNIPROT,
        "sequence_length": len(sequence),
        "display_range_uniprot": [DISPLAY_START, DISPLAY_END],
        "occlusion_config": args.config_key,
        "seeds": seeds,
        "contact_cutoff_A": args.contact_cutoff,
        "colour_scale": {
            "quantity": "mean delta pKoff (original minus masked)",
            "normalization": "diverging, centred at zero",
            "symmetric_limit": vmax,
            "limit_rule": "95th percentile of absolute resolved-residue scores in displayed range",
        },
        "reference_style_normalization": {
            "quantity": "five-checkpoint mean delta pKoff",
            "method": "clip((score - q05) / (q95 - q05), 0, 1)",
            "q05": normalization_lower,
            "q95": normalization_upper,
            "structure_highlight_rule": "top 10% of resolved displayed residues",
            "n_highlighted_uniprot_positions": len(high_positions),
        },
        "interpretation_boundary": (
            "Post-hoc occlusion sensitivity; not model-native attention or residue-atom co-attention."
        ),
        "contacts": {
            "experimental": "1NFU-RRP heavy-atom distance <= cutoff",
            "candidate": "selected factor_xa_05 docked-pose heavy-atom distance <= cutoff",
            "n_experimental_residues": len(native_rows),
            "n_candidate_residues": len(docked_rows),
        },
        "inputs": {
            str(path): sha256(path) for path in (
                summary_csv, cif_path, receptor_pdb, native_ligand_pdb,
                docked_contact_csv, pose_sdf,
            )
        },
        "outputs": {
            "residue_scores": str(residue_csv),
            "experimental_contacts": str(native_csv),
            "scored_receptor": str(scored_receptor),
            "overview_pymol_script": str(pml),
            "pocket_pymol_script": str(pocket_pml),
            "reference_style_pymol_script": str(reference_pml),
            "sequence_panel": str(sequence_png),
            "overview_structure_panel": str(structure_png) if structure_png.exists() else None,
            "pocket_structure_panel": str(pocket_png) if pocket_png.exists() else None,
            "reference_style_structure_panel": (
                str(reference_structure_png) if reference_structure_png.exists() else None
            ),
            "overview_composite": overview_outputs,
            "main_pocket_composite": composite_outputs,
            "reference_style_composite": reference_outputs,
        },
    }
    manifest_path = args.output_dir / "stage4_visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Completed: {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
