# Missing formal training-time benchmark

This package fills three missing timing blocks without overwriting the existing
formal benchmark outputs:

- BiCoA-Net / KinetX published-default configuration: 15 runs;
- BiCoA-Net / 2773 tuned configuration: 15 runs;
- MoE / KinetX tuned configuration: 15 runs.

Each block contains five warm, five drug-cold, and five protein-cold results.
Execution is strictly serial: concurrency is hard-coded to 1.

For every model, warm-start uses the five aligned files `train/val/test_run1..5`
with seed 42. These are five pre-generated random split runs, not a conventional
five-fold partition: their test sets overlap. Drug-cold uses the five canonical
folds with seed 42.
Protein-cold uses one fixed split and five seeds (42, 142, 242, 342, 442).
The older BiCoA/KinetX result that repeated warm run1 over five seeds is not
reused here because it is not aligned with the final five-run comparison.

## One-click launch on the server

```bash
cd /path/to/MGCA-koff
bash training_time_benchmark/run_missing_training_times_serial.sh
```

Run only the path/configuration preflight:

```bash
PREFLIGHT_ONLY=1 bash training_time_benchmark/run_missing_training_times_serial.sh
```

Resume after interruption by running the same command. Verified complete runs
are skipped. An incomplete output is deliberately not overwritten; inspect it,
then opt in to rerunning that target:

```bash
EXISTING_RESULT_ACTION=overwrite \
  bash training_time_benchmark/run_missing_training_times_serial.sh
```

Optional environment variables include `PYTHON_BIN`, `DEVICE`, `NUM_WORKERS`,
`BICOA_FEATURE_CACHE`, and `TRAINING_TIME_OUTPUT_ROOT`. There is intentionally
no parallelism setting.

If the existing MGCA formal per-run table is available online, pass it to also
produce a descriptive MGCA-versus-MoE speed comparison:

```bash
MGCA_REFERENCE_CSV=/absolute/path/final_per_run_metrics.csv \
  bash training_time_benchmark/run_missing_training_times_serial.sh
```

## Timing definitions

The primary value is `training_duration_sec`: synchronized wall-clock time from
the start of epoch 1 through the last epoch, including validation, early
stopping, EMA/best-state maintenance where applicable. It excludes feature
precomputation, data/cache setup, model initialization, final validation/test
evaluation, and output-file writing.

The scripts also record complete subprocess time, final evaluation time, setup
time, epochs actually run, mean epoch time, and peak allocated GPU memory.
BiCoA embedding/descriptor cache time is written separately to
`feature_precompute_timing.json` and is never mixed into training time.

## Output files

- `benchmark_environment.json`: hardware and software environment;
- `benchmark_manifest.json`: exact 45-run order and design;
- `feature_precompute_times.csv`: per-CSV BiCoA preprocessing time;
- `training_time_per_run.csv`: all 45 raw timing and performance rows;
- `training_time_summary_by_split.csv`: mean and SD for each five-run block;
- `training_time_summary_overall.csv`: descriptive mean and SD across 15 runs;
- `bicoa_vs_moe_time_ratios.csv`: descriptive KinetX model-time ratios;
- `performance_sanity_check.csv`: rerun performance values for comparison with the formal table;
- `mgca_vs_moe_descriptive.csv`: optional historical comparison;
- `training_time_report.md`: ready-to-review narrative table.

Batch sizes remain those of each published/selected configuration. They should
be reported alongside timing, rather than forcibly made equal. A common-batch
throughput experiment would be a separate controlled ablation.

## Controlled MGCA-Morgan add-on

After the 45-run aligned benchmark above has completed, run the missing
MGCA-Morgan/KinetX block with:

```bash
cd /path/to/MGCA-koff
bash training_time_benchmark/run_mgca_aligned_training_times_serial.sh
```

This adds `mgca_kinetx_tuned/{warm,drug_cold,protein_cold}/run1..5` to the
existing `training_time_benchmark_outputs_aligned` directory. It does not
rerun or overwrite MoE. The scheduler is hard-coded to concurrency 1 and uses
the same split/seed design as the existing aligned MoE block. CUDA work is
synchronized at all primary timing boundaries.

The launcher first constructs or validates the 11 unique ESM caches in a
separate preflight process. Cache construction is therefore excluded from both
the primary training time and the per-run subprocess wall clock. Each timed
run fails rather than silently extracting ESM2 features when its production
cache is missing.

Run path/configuration checks only:

```bash
PREFLIGHT_ONLY=1 \
  bash training_time_benchmark/run_mgca_aligned_training_times_serial.sh
```

Resume an interrupted run with the same one-click command. Complete runs are
validated and skipped. To replace an incomplete target after inspecting its
log:

```bash
EXISTING_RESULT_ACTION=overwrite \
  bash training_time_benchmark/run_mgca_aligned_training_times_serial.sh
```

The final controlled tables are:

- `mgca_aligned_training_time_per_run.csv`;
- `mgca_moe_aligned_training_time_summary_by_split.csv`;
- `mgca_vs_moe_aligned_paired_ratios.csv`;
- `mgca_vs_moe_aligned_speed_ratios.csv`;
- `mgca_vs_moe_aligned_training_time_report.md`.
