# Missing formal training-time benchmark

Primary endpoint: synchronized wall-clock time for the complete training loop, including per-epoch validation and early stopping, but excluding feature precomputation, setup, final test evaluation, and file writes.

All 45 runs were launched sequentially with scheduler concurrency fixed to 1.

## Mean +/- SD by split

| study | split | model | dataset | configuration | runs | batch_size | epochs_ran_mean | epochs_ran_std | best_epoch_mean | training_duration_sec_mean | training_duration_sec_std | training_duration_min_mean | mean_epoch_duration_sec_mean | subprocess_wall_clock_sec_mean | training_peak_gpu_memory_mb_mean | test_mse_mean | test_mse_std | test_rmse_mean | test_rmse_std | test_mae_mean | test_mae_std | test_r2_mean | test_r2_std | test_pearson_mean | test_pearson_std | test_spearman_mean | test_spearman_std | test_ci_mean | test_ci_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bicoa_kinetx_default | warm | BiCoA-Net | KinetX | published/default | 5 | 64 | 168 | 37.1954 | 118 | 5673.86 | 1306.52 | 94.5644 | 33.7281 | 5691.03 | 17004.3 | 0.431959 | 0.0455617 | 0.656512 | 0.0344938 | 0.440258 | 0.0164794 | 0.696526 | 0.0253522 | 0.835276 | 0.0149832 | 0.830866 | 0.0137174 | 0.826943 | 0.00738349 |
| bicoa_kinetx_default | drug_cold | BiCoA-Net | KinetX | published/default | 5 | 64 | 140.6 | 25.3338 | 90.6 | 4685.92 | 827.896 | 78.0986 | 33.3396 | 4703.1 | 17004.4 | 0.574775 | 0.0611622 | 0.757295 | 0.0399782 | 0.533945 | 0.0265766 | 0.598854 | 0.0463359 | 0.778566 | 0.0273035 | 0.764292 | 0.0155604 | 0.789177 | 0.00802512 |
| bicoa_kinetx_default | protein_cold | BiCoA-Net | KinetX | published/default | 5 | 64 | 149.8 | 28.5167 | 99.8 | 5209.9 | 1009.99 | 86.8317 | 34.7719 | 5227.72 | 17004.4 | 1.04118 | 0.0684686 | 1.01994 | 0.0338359 | 0.807005 | 0.0290452 | 0.188885 | 0.0533391 | 0.495652 | 0.0397276 | 0.435759 | 0.0421901 | 0.652863 | 0.0157901 |
| bicoa_2773_tuned | warm | BiCoA-Net | 2773 | tuned | 5 | 8 | 115 | 35.4612 | 65 | 5064.58 | 1490.16 | 84.4096 | 44.1838 | 5079.08 | 17004.6 | 0.290691 | 0.0367348 | 0.538306 | 0.0338797 | 0.376861 | 0.0120741 | 0.692384 | 0.0421631 | 0.833357 | 0.0243504 | 0.814361 | 0.0210953 | 0.818937 | 0.00932799 |
| bicoa_2773_tuned | drug_cold | BiCoA-Net | 2773 | tuned | 5 | 8 | 86 | 29.3172 | 36 | 3891.81 | 1380.22 | 64.8635 | 45.1265 | 3906.9 | 17004.5 | 0.391523 | 0.05728 | 0.624394 | 0.0454879 | 0.467939 | 0.0380384 | 0.585291 | 0.0596769 | 0.76936 | 0.0359164 | 0.723891 | 0.0399781 | 0.77128 | 0.0190595 |
| bicoa_2773_tuned | protein_cold | BiCoA-Net | 2773 | tuned | 5 | 8 | 70.8 | 7.62889 | 20.8 | 3030.08 | 338.046 | 50.5013 | 42.7833 | 3044.56 | 17004.1 | 1.89729 | 0.300352 | 1.37415 | 0.106045 | 1.16239 | 0.090214 | -0.61747 | 0.256055 | 0.404677 | 0.151136 | 0.443133 | 0.120318 | 0.65339 | 0.0436184 |
| moe_kinetx_tuned | warm | MoE | KinetX | tuned | 5 | 64 | 42.4 | 6.1887 | 32.8 | 746.515 | 94.294 | 12.4419 | 17.6577 | 759.716 | 23.4824 | 0.532064 | 0.0927083 | 0.72727 | 0.0626773 | 0.534183 | 0.0311175 | 0.650029 | 0.0440673 | 0.809724 | 0.0244933 | 0.824614 | 0.0234457 | 0.818451 | 0.0116844 |
| moe_kinetx_tuned | drug_cold | MoE | KinetX | tuned | 5 | 64 | 27.4 | 8.82043 | 17.4 | 479.229 | 153.113 | 7.98716 | 17.4826 | 492.497 | 23.4824 | 0.815329 | 0.334139 | 0.889973 | 0.17058 | 0.632662 | 0.0730117 | 0.487893 | 0.117437 | 0.712734 | 0.0897568 | 0.75232 | 0.0364348 | 0.780398 | 0.0197105 |
| moe_kinetx_tuned | protein_cold | MoE | KinetX | tuned | 5 | 64 | 16.6 | 4.27785 | 6.6 | 282.512 | 74.6958 | 4.70853 | 17.0014 | 297.024 | 23.4824 | 1.42535 | 0.0839417 | 1.19347 | 0.0349313 | 1.00006 | 0.0478653 | -0.222699 | 0.0720071 | 0.400089 | 0.0440082 | 0.333185 | 0.0642929 | 0.613659 | 0.0226163 |

## Mean +/- SD over all 15 runs

| study | model | dataset | configuration | runs | batch_size | epochs_ran_mean | epochs_ran_std | best_epoch_mean | training_duration_sec_mean | training_duration_sec_std | training_duration_min_mean | mean_epoch_duration_sec_mean | subprocess_wall_clock_sec_mean | training_peak_gpu_memory_mb_mean | test_mse_mean | test_mse_std | test_rmse_mean | test_rmse_std | test_mae_mean | test_mae_std | test_r2_mean | test_r2_std | test_pearson_mean | test_pearson_std | test_spearman_mean | test_spearman_std | test_ci_mean | test_ci_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bicoa_kinetx_default | BiCoA-Net | KinetX | published/default | 15 | 64 | 152.8 | 30.8202 | 102.8 | 5189.89 | 1072.15 | 86.4982 | 33.9465 | 5207.28 | 17004.4 | 0.68264 | 0.274794 | 0.811248 | 0.162075 | 0.593736 | 0.162647 | 0.494755 | 0.231156 | 0.703165 | 0.156112 | 0.676972 | 0.180535 | 0.756328 | 0.0780689 |
| bicoa_2773_tuned | BiCoA-Net | 2773 | tuned | 15 | 8 | 90.6 | 31.3319 | 40.6 | 3995.49 | 1398.67 | 66.5915 | 44.0312 | 4010.18 | 17004.4 | 0.859834 | 0.778144 | 0.845617 | 0.393836 | 0.669064 | 0.366933 | 0.220068 | 0.630947 | 0.669131 | 0.212747 | 0.660462 | 0.177438 | 0.747869 | 0.0765489 |
| moe_kinetx_tuned | MoE | KinetX | tuned | 15 | 64 | 28.8 | 12.5823 | 18.9333 | 502.752 | 222.655 | 8.3792 | 17.3806 | 516.412 | 23.4824 | 0.924248 | 0.430368 | 0.936905 | 0.223106 | 0.722302 | 0.213346 | 0.305074 | 0.399864 | 0.640849 | 0.189101 | 0.636707 | 0.228043 | 0.737503 | 0.0936541 |

## Descriptive BiCoA/KinetX versus MoE/KinetX ratios

| scope | bicoa_kinetx_mean_sec | moe_kinetx_mean_sec | bicoa_over_moe_time_ratio | interpretation |
| --- | --- | --- | --- | --- |
| warm | 5673.86 | 746.515 | 7.60047 | descriptive; model configurations have different batch sizes and stopping schedules |
| drug_cold | 4685.92 | 479.229 | 9.77802 | descriptive; model configurations have different batch sizes and stopping schedules |
| protein_cold | 5209.9 | 282.512 | 18.4414 | descriptive; model configurations have different batch sizes and stopping schedules |
| all_15_runs | 5189.89 | 502.752 | 10.323 | descriptive; averages three evaluation protocols |

## Reporting note

Different batch sizes should not be forced to match when reporting each model under its selected or published configuration. Always show batch size, epochs actually run, early-stopping rule, hardware, and timing definition beside the time result. A controlled throughput ablation with a shared batch size is a separate experiment.
