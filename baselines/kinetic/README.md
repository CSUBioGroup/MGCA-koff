# Fixed-configuration table baselines

These source snapshots document the published/default configurations used on
each method's origin dataset.

- `moe_baseline/`: local MoE model and holdout training source.
- `bicoa_net/`: local BiCoA-Net random/cold training and inference source.

They are not invoked by `run_all_warm_mse_tuning.sh`. Cross-dataset adapters are
under `ONLINE/cross_dataset_tuning/`: MoE is tuned only on KinetX, and BiCoA-Net
is tuned only on 2773. Both adapters consume aligned warm train/validation
CSVs and reject test input during selection.
