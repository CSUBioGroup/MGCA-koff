# Controlled aligned MGCA-Morgan versus MoE timing

All runs used the same aligned KinetX split files and were launched serially (concurrency=1). GPU work was synchronized at the timing boundaries. The primary endpoint includes all training epochs, per-epoch validation, early stopping, and best-state copying; feature precomputation, setup, final evaluation, and artifact writes are excluded.

## Controlled speed ratios

| scope | mgca_training_sec_mean | mgca_training_sec_std | moe_training_sec_mean | moe_training_sec_std | ratio_of_mean_training_time_moe_over_mgca | mgca_sec_per_epoch_mean | moe_sec_per_epoch_mean | per_epoch_ratio_of_means_moe_over_mgca | subprocess_ratio_of_means_moe_over_mgca |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| warm | 93.0733 | 24.2116 | 746.515 | 94.294 | 8.02072 | 1.55899 | 17.6577 | 11.3264 | 7.30292 |
| drug_cold | 50.56 | 11.0137 | 479.229 | 153.113 | 9.47843 | 1.54477 | 17.4826 | 11.3173 | 8.04814 |
| protein_cold | 36.838 | 2.81544 | 282.512 | 74.6958 | 7.66902 | 1.57368 | 17.0014 | 10.8036 | 6.33366 |

A ratio greater than 1 means MGCA-Morgan is faster. The total-training ratio reflects each model's selected stopping schedule. The per-epoch ratio compares synchronized mean epoch time. Batch size and actual epochs are reported rather than forced to match.

The subprocess ratio is a secondary sensitivity analysis, not the primary speed claim: the MGCA timed process performs final train/validation/test evaluation, whereas the existing MoE process performs final validation/test evaluation. The synchronized training-loop and per-epoch endpoints have the aligned boundary.

MoE traverses every data-loader batch but masks rows that its legacy feature conversion marks invalid. Coverage is retained in the per-run table and should be disclosed when interpreting model efficiency.

## Full mean and SD table

| study | model | dataset | configuration | split | runs | batch_size | parameter_count | trainable_parameter_count | epochs_ran_mean | epochs_ran_std | best_epoch_mean | best_epoch_std | training_duration_sec_mean | training_duration_sec_std | mean_epoch_duration_sec_mean | mean_epoch_duration_sec_std | setup_duration_sec_mean | setup_duration_sec_std | final_evaluation_duration_sec_mean | final_evaluation_duration_sec_std | artifact_write_duration_sec_mean | artifact_write_duration_sec_std | run_to_metrics_duration_sec_mean | run_to_metrics_duration_sec_std | subprocess_wall_clock_duration_sec_mean | subprocess_wall_clock_duration_sec_std | training_peak_gpu_memory_mb_mean | training_peak_gpu_memory_mb_std | test_mse_mean | test_mse_std | test_rmse_mean | test_rmse_std | test_mae_mean | test_mae_std | test_r2_mean | test_r2_std | test_pearson_mean | test_pearson_std | test_spearman_mean | test_spearman_std | test_ci_mean | test_ci_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mgca_kinetx_tuned | MGCA-Morgan | KinetX | tuned | warm | 5 | 64 | 18912528 | 18912528 | 59.8 | 15.8493 | 44.8 | 15.8493 | 93.0733 | 24.2116 | 1.55899 | 0.0678321 | 8.17418 | 0.468301 | 0.23827 | 0.00823482 | 0.0139526 | 0.000687946 | 101.715 | 24.0558 | 104.029 | 24.2453 | 701.503 | 0 | 0.437399 | 0.0550394 | 0.660291 | 0.0420477 | 0.424321 | 0.0207738 | 0.693219 | 0.0275729 | 0.835878 | 0.0163748 | 0.836706 | 0.0149866 | 0.831661 | 0.00795543 |
| mgca_kinetx_tuned | MGCA-Morgan | KinetX | tuned | drug_cold | 5 | 64 | 18912528 | 18912528 | 32.8 | 7.25948 | 17.8 | 7.25948 | 50.56 | 11.0137 | 1.54477 | 0.15823 | 8.14458 | 0.076507 | 0.22974 | 0.00772401 | 0.0145349 | 0.00111688 | 59.1627 | 11.01 | 61.1939 | 11.051 | 701.503 | 0 | 0.592104 | 0.0538426 | 0.768865 | 0.0344686 | 0.532963 | 0.0285753 | 0.587471 | 0.0311607 | 0.777623 | 0.0170827 | 0.766533 | 0.0205505 | 0.791162 | 0.00971174 |
| mgca_kinetx_tuned | MGCA-Morgan | KinetX | tuned | protein_cold | 5 | 64 | 18912528 | 18912528 | 23.4 | 0.547723 | 8.4 | 0.547723 | 36.838 | 2.81544 | 1.57368 | 0.104727 | 7.50639 | 0.303755 | 0.225855 | 0.00198109 | 0.0140436 | 0.000719557 | 44.8021 | 2.86436 | 46.8961 | 2.8748 | 701.503 | 0 | 0.809511 | 0.0332279 | 0.899577 | 0.018475 | 0.676432 | 0.0131534 | 0.369366 | 0.0258856 | 0.631453 | 0.0274433 | 0.590645 | 0.0307599 | 0.715152 | 0.0113439 |
| moe_kinetx_tuned | MoE | KinetX | tuned | warm | 5 | 64 | 264034 | 264034 | 42.4 | 6.1887 | 32.8 | 6.83374 | 746.515 | 94.294 | 17.6577 | 0.539459 | 3.68689 | 0.276583 | 6.12908 | 0.145536 | 0.082562 | 0.0430697 | 756.414 | 94.563 | 759.716 | 94.5656 | 23.4824 | 0 | 0.532064 | 0.0927083 | 0.72727 | 0.0626773 | 0.534183 | 0.0311175 | 0.650029 | 0.0440673 | 0.809724 | 0.0244933 | 0.824614 | 0.0234457 | 0.818451 | 0.0116844 |
| moe_kinetx_tuned | MoE | KinetX | tuned | drug_cold | 5 | 64 | 264034 | 264034 | 27.4 | 8.82043 | 17.4 | 8.82043 | 479.229 | 153.113 | 17.4826 | 0.462353 | 3.57305 | 0.0689306 | 6.24978 | 0.358768 | 0.0527657 | 0.0287816 | 489.105 | 153.307 | 492.497 | 153.547 | 23.4824 | 0 | 0.815329 | 0.334139 | 0.889973 | 0.17058 | 0.632662 | 0.0730117 | 0.487893 | 0.117437 | 0.712734 | 0.0897568 | 0.75232 | 0.0364348 | 0.780398 | 0.0197105 |
| moe_kinetx_tuned | MoE | KinetX | tuned | protein_cold | 5 | 64 | 264034 | 264034 | 16.6 | 4.27785 | 6.6 | 4.27785 | 282.512 | 74.6958 | 17.0014 | 0.306136 | 3.43447 | 0.188879 | 7.93863 | 0.100481 | 0.0300697 | 0.00422871 | 293.915 | 74.7027 | 297.024 | 74.728 | 23.4824 | 0 | 1.42535 | 0.0839417 | 1.19347 | 0.0349313 | 1.00006 | 0.0478653 | -0.222699 | 0.0720071 | 0.400089 | 0.0440082 | 0.333185 | 0.0642929 | 0.613659 | 0.0226163 |

Feature-cache construction remains a separate preprocessing cost and is not included in the primary training endpoint.
