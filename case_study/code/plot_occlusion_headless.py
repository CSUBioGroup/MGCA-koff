#!/usr/bin/env python3
"""Case figures with an optional RDKit SVG and an explicit headless fallback."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common_v10 as common


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="Factor Xa")
    parser.add_argument("--top-features", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def protein_plot(args, selection, dataset):
    representative = selection["representative_compound"]
    config = selection["protein_configuration"]
    rows = [
        row for row in read_csv(args.stage2_dir / "protein_window_importance_summary.csv")
        if row["compound_id"] == representative["compound_id"] and row["config_key"] == config
    ]
    sequence_length = max(int(row["window_end"]) for row in rows)
    residue_values = [[] for _ in range(sequence_length)]
    for row in rows:
        for index in range(int(row["window_start"]) - 1, int(row["window_end"])):
            residue_values[index].append(float(row["mean_delta_pkoff"]))
    scores = np.asarray([np.mean(values) if values else np.nan for values in residue_values])
    positions = np.arange(1, sequence_length + 1)
    figure, axis = plt.subplots(figsize=(12, 3.8))
    axis.plot(positions, scores, color="black", linewidth=1.2)
    axis.fill_between(positions, 0, scores, where=scores >= 0, color="#b2182b", alpha=0.28)
    axis.fill_between(positions, 0, scores, where=scores < 0, color="#2166ac", alpha=0.22)
    axis.axhline(0, color="grey", linewidth=0.7)
    axis.set_xlabel("Protein residue position")
    axis.set_ylabel(r"Mean occlusion contribution $\Delta pK_{off}$")
    axis.set_title(f"MGCA final ({dataset}) — {args.label} protein-window sensitivity")
    figure.tight_layout()
    path = args.output_dir / "protein_window_importance.png"
    figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def ligand_bar_plot(args, selection, dataset):
    representative = selection["representative_compound"]
    rows = [
        row for row in read_csv(args.stage2_dir / "ligand_bit_importance_summary.csv")
        if row["compound_id"] == representative["compound_id"]
    ]
    rows.sort(key=lambda row: -float(row["mean_delta_pkoff"]))
    rows = rows[: args.top_features]
    labels = [row["feature_key"] for row in rows][::-1]
    values = [float(row["mean_delta_pkoff"]) for row in rows][::-1]
    errors = [float(row["sd_delta_pkoff"]) for row in rows][::-1]
    figure, axis = plt.subplots(figsize=(8.5, max(4.5, len(rows) * 0.45)))
    colors = ["#b2182b" if value >= 0 else "#2166ac" for value in values]
    axis.barh(range(len(rows)), values, xerr=errors, color=colors, alpha=0.82, capsize=2)
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels(labels)
    axis.axvline(0, color="black", linewidth=0.7)
    axis.set_xlabel(r"Mean bit-deletion contribution $\Delta pK_{off}$")
    axis.set_title(f"MGCA final ({dataset}) — {args.label} top Morgan features")
    figure.tight_layout()
    path = args.output_dir / "ligand_feature_importance_bar.png"
    figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def dual_plot(args, selection, dataset):
    representative = selection["representative_compound"]
    rows = [
        row for row in read_csv(args.stage2_dir / "dual_occlusion_interactions_summary.csv")
        if row["compound_id"] == representative["compound_id"]
    ]
    protein_keys = [row["feature_key"] for row in selection["protein_features"]]
    ligand_keys = [row["feature_key"] for row in selection["ligand_features"]]
    lookup = {
        (row["protein_feature_key"], row["ligand_feature_key"]): float(row["mean_interaction_pkoff"])
        for row in rows
    }
    matrix = np.asarray(
        [[lookup.get((protein, ligand), np.nan) for ligand in ligand_keys] for protein in protein_keys]
    )
    limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    if limit == 0:
        limit = 1.0
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(ligand_keys)))
    axis.set_xticklabels(ligand_keys, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(protein_keys)))
    axis.set_yticklabels([value.split(":", 1)[-1] for value in protein_keys], fontsize=8)
    axis.set_xlabel("Morgan feature")
    axis.set_ylabel("Protein window")
    axis.set_title(f"MGCA final ({dataset}) — {args.label} dual-occlusion interaction")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(r"Interaction $I=y_p+y_l-y_0-y_{pl}$")
    figure.tight_layout()
    path = args.output_dir / "dual_occlusion_interaction_heatmap.png"
    figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def ligand_structure(args, selection):
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except (ImportError, OSError) as exc:
        return None, str(exc)
    representative = selection['representative_compound']
    mol = Chem.MolFromSmiles(representative['canonical_smiles'])
    if mol is None:raise RuntimeError('Invalid representative ligand')
    rows = [r for r in read_csv(args.stage2_dir/'ligand_bit_importance_summary.csv')
            if r['compound_id']==representative['compound_id']]
    rows.sort(key=lambda r:-float(r['mean_delta_pkoff']))
    scores = np.zeros(mol.GetNumAtoms())
    for row in rows[:args.top_features]:
        atoms = json.loads(row['atom_indices_json'])
        for atom in atoms:
            scores[int(atom)] += max(0., float(row['mean_delta_pkoff'])) / max(1,len(atoms))
    if scores.max()>0:scores/=scores.max()
    atoms=[int(i) for i in np.flatnonzero(scores>0)]
    colors={i:(1., 1.-.75*float(scores[i]), .25) for i in atoms}
    drawer=rdMolDraw2D.MolDraw2DSVG(1000,700)
    drawer.drawOptions().addAtomIndices=True
    drawer.DrawMolecule(mol,legend='MGCA final: positive top-feature environments (post-hoc)',
                        highlightAtoms=atoms,highlightAtomColors=colors)
    drawer.FinishDrawing()
    path=args.output_dir/'ligand_feature_environments.svg'
    common.atomic_text(path,drawer.GetDrawingText())
    return path,None


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = common.read_json(args.stage2_dir / "dual_occlusion_selection.json")
    manifest = common.read_json(args.stage2_dir / "factor_xa_stage2_manifest.json")
    paths = [
        protein_plot(args, selection, manifest["training_dataset"]),
        ligand_bar_plot(args, selection, manifest["training_dataset"]),
        dual_plot(args, selection, manifest["training_dataset"]),
    ]
    ligand_svg, draw_error = ligand_structure(args, selection)
    if ligand_svg is not None:paths.append(ligand_svg)
    common.atomic_json(
        args.output_dir / "headless_plot_manifest.json",
        {
            "rendering": "matplotlib Agg with RDKit SVG" if ligand_svg else "matplotlib Agg; ligand feature bar chart fallback",
            "rdkit_draw_unavailable_reason": draw_error,
            "label": args.label,
            "source_manifest_sha256": common.sha256_file(args.stage2_dir / "factor_xa_stage2_manifest.json"),
            "outputs": [
                {"path": str(path.resolve()), "sha256": common.sha256_file(path)} for path in paths
            ],
        },
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
