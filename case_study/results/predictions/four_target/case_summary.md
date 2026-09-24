# four_target: frozen final MGCA trained on full 2773

Case rows: 62; unique compounds: 43; targets: 4.
Five checkpoints are ranked independently; ensemble rows are additional descriptive summaries.
Train is audited from the actual full-refit CSV using exact/subsequence or guarded 5-mer matching.

| Target | Evaluation | N | RMSE | MAE | Spearman | C-index |
|---|---:|---:|---:|---:|---:|---:|
| Dipeptidyl peptidase 4 | ensemble | 15 | 1.1389 | 1.0154 | 0.08928571428571427 | 0.5523809523809524 |
| NK1R | ensemble | 7 | 1.3737 | 1.2493 | 0.5 | 0.6666666666666666 |
| V2R | ensemble | 9 | 1.0025 | 0.9783 | 0.5666666666666667 | 0.6944444444444444 |
| sAC | ensemble | 12 | 0.6377 | 0.5733 | 0.17353775698917726 | 0.5948275862068966 |
