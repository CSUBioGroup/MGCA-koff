# Implementation validation

Date: 2026-09-14.

- Twelve unit/contract tests passed, including HTTP authentication, readiness, body-size limit, schema rejection, busy response, ranking, immutable completion seals, LRU eviction and checkpoint-type rejection.
- Exact frozen neural-network classes executed on synthetic 4x2560 protein and 4x2048 Morgan features. Outputs were finite, batch-size 1 versus 2 agreed within 1e-5, parameter count was 13,392,395, and initial drug/joint coefficients were 0.4/0.05.
- A synthetic seed-43 training test simulated interruption after epoch 2, resumed through epoch 55, and produced parameter-identical state to uninterrupted training. A completed run was then verified and skipped without modifying its checkpoint hash.
- The complete clean KinetX cohort builder returned 5,446 rows. Frozen epoch records are selection-only and show no test access.
- Eleven Python files passed Python 3.8 grammar parsing. Four shell files passed `bash -n`.
- FastAPI tests used pinned FastAPI 0.115.6, Uvicorn 0.30.6 and Pydantic 2.10.6 in an isolated local test directory.

Not performed locally: full ESM2 extraction, real KinetX training, creation of a learned seed-43 checkpoint, deployment ZIP generation, or live GPU numerical reproduction. The delivered code ZIP contains no new checkpoint. Those steps run online with `bash run_train_and_package.sh`; the workflow verifies and packages the resulting checkpoint only after successful reload and finite-prediction checks.
