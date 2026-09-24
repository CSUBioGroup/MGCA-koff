# Cross-dataset kinetic-baseline tuning

MoE-Kinetic retains its original/default configuration on the 2,773-pair
dataset and is tuned on KinetX. BiCoA-Net retains its original/default
configuration on KinetX and is tuned on the 2,773-pair dataset.

Both tuners use 30 warm-run-1 Optuna trials followed by five-candidate,
five-run validation. Selection minimizes mean warm-validation MSE and does not
read test files.

From the repository root:

```bash
bash cross_dataset_tuning/run_moe_kinetx_tuning.sh
bash cross_dataset_tuning/run_bicoa_2773_tuning.sh
```

The frozen selected configurations are archived under `stage_data/moe/KinetX`
and `stage_data/bicoa/2773`. To rerun all final protocols with those configs:

```bash
bash cross_dataset_tuning/run_best_params_all.sh
```

MoE requires the original pretrained Mol2Vec assets. Set `MOE_SOURCE_ROOT` to
a directory containing `necessary_files/model_300dim.pkl`,
`necessary_files/kinetics_mol2vec_resseq_bimodal_regression_embed.pt`, and
`necessary_files/res_list3.txt`. These large upstream weights are not bundled.

