# k4dd: frozen final MGCA trained on full 2773

Case rows: 487; unique compounds: 426; targets: 6.
Five checkpoints are ranked independently; ensemble rows are additional descriptive summaries.
Train is audited from the actual full-refit CSV using exact/subsequence or guarded 5-mer matching.

| Target | Evaluation | N | RMSE | MAE | Spearman | C-index |
|---|---:|---:|---:|---:|---:|---:|
| A1 receptor | ensemble | 13 | 0.7375 | 0.6830 | 0.5171944370078857 | 0.6753246753246753 |
| A2A receptor | ensemble | 40 | 0.7397 | 0.4478 | 0.697560975609756 | 0.7653846153846153 |
| AURKA | ensemble | 130 | 0.4310 | 0.2774 | 0.5889197558223112 | 0.7252338690333413 |
| Factor Xa | ensemble | 7 | 2.1282 | 2.0054 | 0.32142857142857145 | 0.6190476190476191 |
| HSP90alpha | ensemble | 155 | 0.4535 | 0.2846 | 0.9238984453239846 | 0.9070596126435818 |
| IGF-1R | ensemble | 81 | 0.9562 | 0.8727 | 0.45042405125601515 | 0.665891472868217 |
