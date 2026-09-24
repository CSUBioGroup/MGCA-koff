# Release notes: v1.0.0

Even-span source update (2026-09-22): the executable MGCA model now uses
`even_span_v2` by default. Four ESM2 window starts are distributed across the
valid layer range with deterministic integer rounding. On ESM2-t36, the
selected KinetX width eight therefore uses `1-8, 10-17, 20-27, 29-36`.
`legacy_anchors_v1` remains an explicit compatibility option for archived
records; legacy KinetX feature caches and checkpoints are not compatible with
the updated layout.

This is the curated reproducibility archive for the final MGCA model used in the revised manuscript.

Included:

- final model, training, tuning, ablation, timing, and inference source code;
- KinetX and Zhao dataset experiment data and split definitions;
- frozen best configurations and hyperparameter-search evidence;
- per-run formal benchmark/ablation metrics, histories, validation/test predictions, and aggregate reports;
- compact fusion-sensitivity results;
- baseline implementations and retained tuning evidence;
- final five-refit Zhao dataset case-study code, inputs, predictions, occlusion/structure analyses, figures, and provenance manifests.

Intentionally omitted:

- neural-network checkpoint binaries;
- ESM2 pretrained weights;
- transient caches, locks, logs, diagnostic arrays, and training-set predictions;
- duplicate nested archives;
- superseded case-study outputs and obsolete paper-table copies.

Scientific identity:

- final case configuration ID: `3154ac214828eafe59655160112af7c34eaf8be4cf739ed0ed21f2eb460d2b30`;
- final case seeds: 42, 142, 242, 342, 442;
- fixed full-cohort refit duration: 33 epochs;
- license: Apache-2.0.
