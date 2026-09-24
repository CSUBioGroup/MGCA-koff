# Baseline implementation inventory

This directory contains every comparison-method implementation retained in the
original `ONLINE` archive. File contents were copied without algorithmic
changes.

| Directory | Methods | Role in the paper |
|---|---|---|
| `standard/` | DeepDTA, GraphDTA, AttentionDTA | aligned affinity-model baselines |
| `kinetic/bicoa_net/` | BiCoA-Net | published kinetic baseline snapshot |
| `kinetic/moe_baseline/` | MoE-Kinetic | published kinetic baseline snapshot |
| `cross_dataset_tuning/` | BiCoA-Net on 2773; MoE-Kinetic on KinetX | transfer tuning and final-evaluation runners |
| `mgca_legacy_variants/` | historical MGCA-Morgan and MGCA-FCFP workflow | legacy/internal comparison only |

The authoritative final MGCA implementation is **not** located here. It is
`../model/model.py`, with the complete final workflow under
`../experiments/final_v10/`.

## Standard baselines

The `standard/common/` code provides shared preprocessing, training, metric
calculation and Optuna selection. Method-specific networks and launchers are:

- `standard/deepdta/model.py` and `tune.sh`;
- `standard/graphdta/model.py` and `tune.sh`;
- `standard/attentiondta/model.py` and `tune.sh`.

Run all aligned standard baselines from `standard/` with:

```bash
bash tune_all.sh
```

## Kinetic baselines

`kinetic/` preserves the source snapshots used as fixed/original-dataset
implementations. `cross_dataset_tuning/` contains the aligned tuning runners
used when transferring each kinetic baseline to the other dataset. See its
README and `run_all.sh`.

## Historical MGCA variants

`mgca_legacy_variants/` is retained because the result archive contains
MGCA-Morgan and MGCA-FCFP development comparisons. These files predate the
final positive sequential-residual architecture and must not be used as the
definition of the manuscript's final MGCA model.

Hyperparameter trials and selected configurations for all baselines are under
`../results/baseline_tuning_evidence/`; paper-facing run-level and aggregate
tables are under `../results/paper_tables_and_figures/tables/`.
