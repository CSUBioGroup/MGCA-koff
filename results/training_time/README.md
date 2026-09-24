# Controlled training-time results

This directory archives the controlled timing evidence reported in the revised
manuscript. The benchmark was executed serially (`concurrency=1`) on one NVIDIA
RTX 4090. CUDA work was synchronized at the primary timing boundaries.

The primary endpoint, `training_duration_sec`, includes all training epochs,
per-epoch validation, early stopping, and best-state maintenance. It excludes
feature precomputation, cache loading, model setup, final evaluation, and
artifact writing. Consequently, the reported values are training-loop times,
not end-to-end preprocessing-plus-training times.

Key files:

- `training_time_per_run.csv`: BiCoA-Net/KinetX, BiCoA-Net/2773, and
  MoE-Kinetic/KinetX per-run timing records;
- `mgca_aligned_training_time_per_run.csv`: aligned MGCA-Morgan/KinetX timing;
- `mgca_moe_aligned_training_time_summary_by_split.csv`: five-run mean and
  sample standard deviation by protocol;
- `mgca_vs_moe_aligned_speed_ratios.csv`: primary total-training and per-epoch
  ratios;
- `benchmark_environment.json` and
  `mgca_aligned_benchmark_environment.json`: hardware/software audit records;
- `benchmark_manifest.json` and `mgca_aligned_benchmark_manifest.json`: exact
  run order, split/seed design, and timing endpoint;
- `runs/<study>/<protocol>/run<index>/`: per-run metrics and subprocess timing
  records without the large prediction/history artifacts.

Warm-start uses five aligned pre-generated split files with seed 42 for each
split. Drug cold-start uses five canonical compound-disjoint folds with seed
42. Protein cold-start uses one fixed sequence-clustered split and seeds 42,
142, 242, 342, and 442.

MoE-Kinetic traverses every data-loader batch but masks samples rejected by its
legacy feature conversion. Its usable-feature coverage must be disclosed when
interpreting either predictive performance or timing. BiCoA-Net/KinetX uses
the published/default configuration; the remaining timed configurations are
the frozen validation-selected settings documented in the release tables.

Reproduction scripts are available under `ONLINE/training_time_benchmark`.
