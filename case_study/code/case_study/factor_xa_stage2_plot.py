#!/usr/bin/env python3
"""Create publication-oriented plots for one Factor Xa Stage-2 model set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import infer_mgca_case_ensemble as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--structure-dir", type=Path)
    parser.add_argument("--top-features", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def protein_plot(args, selection, dataset: str) -> Path:
    representative = selection["representative_compound"]
    config = selection["protein_configuration"]
    rows = [
        row
        for row in read_csv(args.stage2_dir / "protein_window_importance_summary.csv")
        if row["compound_id"] == representative["compound_id"]
        and row["config_key"] == config
    ]
    sequence_length = max(int(row["window_end"]) for row in rows)
    residue_values = [[] for _ in range(sequence_length)]
    for row in rows:
        value = float(row["mean_delta_pkoff"])
        for index in range(int(row["window_start"]) - 1, int(row["window_end"])):
            residue_values[index].append(value)
    residue_score = np.asarray(
        [np.mean(values) if values else np.nan for values in residue_values], dtype=float
    )
    positions = np.arange(1, sequence_length + 1)
    figure, axis = plt.subplots(figsize=(12, 3.8))
    axis.plot(positions, residue_score, color="#1f77b4", linewidth=1.4)
    axis.fill_between(
        positions,
        0,
        residue_score,
        where=residue_score >= 0,
        color="#d62728",
        alpha=0.22,
        interpolate=True,
        label="positive contribution",
    )
    axis.axhline(0, color="black", linewidth=0.7)
    if args.structure_dir is not None:
        contact_path = args.structure_dir / "experimental_4A_contacts.csv"
        if contact_path.is_file():
            contacts = sorted(
                {int(row["uniprot_position"]) for row in read_csv(contact_path)}
            )
            for index, position in enumerate(contacts):
                axis.axvline(
                    position,
                    color="#2ca02c",
                    alpha=0.40,
                    linewidth=0.8,
                    label="4 Å PDB contact" if index == 0 else None,
                )
    axis.set_xlabel("Factor Xa residue position (UniProt P00742)")
    axis.set_ylabel(r"Mean occlusion contribution $\Delta pK_{off}$")
    axis.set_title(f"{dataset}: protein-window post-hoc importance")
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    path = args.output_dir / f"{dataset}_factor_xa_protein_importance.png"
    figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def ligand_plot(args, selection, dataset: str, legacy) -> Path:
    from rdkit.Chem.Draw import rdMolDraw2D

    representative = selection["representative_compound"]
    rows = [
        row
        for row in read_csv(args.stage2_dir / "ligand_bit_importance_summary.csv")
        if row["compound_id"] == representative["compound_id"]
    ]
    rows.sort(key=lambda row: -float(row["mean_delta_pkoff"]))
    selected = rows[: args.top_features]
    molecule = legacy.Chem.MolFromSmiles(representative["canonical_smiles"])
    if molecule is None:
        raise RuntimeError("Cannot parse representative compound")
    atom_scores = np.zeros(molecule.GetNumAtoms(), dtype=float)
    for row in selected:
        contribution = max(0.0, float(row["mean_delta_pkoff"]))
        atoms = [int(value) for value in json.loads(row["atom_indices_json"])]
        if atoms:
            for atom in atoms:
                atom_scores[atom] += contribution / len(atoms)
    maximum = float(atom_scores.max())
    normalized = atom_scores / maximum if maximum > 0 else atom_scores
    highlighted = [int(index) for index, value in enumerate(normalized) if value > 0]
    colors = {
        index: (1.0, 1.0 - 0.75 * float(normalized[index]), 0.25, 0.65)
        for index in highlighted
    }
    radii = {index: 0.25 + 0.25 * float(normalized[index]) for index in highlighted}
    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 700)
    options = drawer.drawOptions()
    options.addAtomIndices = True
    options.legendFontSize = 24
    drawer.DrawMolecule(
        molecule,
        legend=f"{dataset}: top-{args.top_features} Morgan-feature atom environments",
        highlightAtoms=highlighted,
        highlightAtomColors=colors,
        highlightAtomRadii=radii,
    )
    drawer.FinishDrawing()
    path = args.output_dir / f"{dataset}_factor_xa_ligand_fragments.png"
    path.write_bytes(drawer.GetDrawingText())
    return path


def dual_plot(args, selection, dataset: str) -> Path:
    representative = selection["representative_compound"]
    rows = [
        row
        for row in read_csv(args.stage2_dir / "dual_occlusion_interactions_summary.csv")
        if row["compound_id"] == representative["compound_id"]
    ]
    protein_keys = [row["feature_key"] for row in selection["protein_features"]]
    ligand_keys = [row["feature_key"] for row in selection["ligand_features"]]
    lookup = {
        (row["protein_feature_key"], row["ligand_feature_key"]): float(
            row["mean_interaction_pkoff"]
        )
        for row in rows
    }
    matrix = np.asarray(
        [[lookup.get((protein, ligand), np.nan) for ligand in ligand_keys] for protein in protein_keys],
        dtype=float,
    )
    limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    limit = limit if limit > 0 else 1.0
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(ligand_keys)))
    axis.set_xticklabels(ligand_keys, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(protein_keys)))
    axis.set_yticklabels(
        [value.split(":", 1)[-1] for value in protein_keys], fontsize=8
    )
    axis.set_xlabel("Morgan feature (radius channel, bit)")
    axis.set_ylabel("Protein window (residue range)")
    axis.set_title(f"{dataset}: second-order dual-occlusion interaction")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(r"Interaction $I= y_p+y_l-y_0-y_{pl}$")
    figure.tight_layout()
    path = args.output_dir / f"{dataset}_factor_xa_dual_interaction_heatmap.png"
    figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads(
        (args.stage2_dir / "dual_occlusion_selection.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (args.stage2_dir / "factor_xa_stage2_manifest.json").read_text(encoding="utf-8")
    )
    dataset = manifest["training_dataset"]
    corrected = base.load_corrected_module()
    paths = [
        protein_plot(args, selection, dataset),
        ligand_plot(args, selection, dataset, corrected.legacy),
        dual_plot(args, selection, dataset),
    ]
    print("Generated figures:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
