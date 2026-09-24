#!/usr/bin/env bash
set -euo pipefail

# Default: main-text figure from the target-unseen 2773 model.
# Optional supplement: MODEL=KinetX bash run_factor_xa_stage4.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MODEL="${MODEL:-2773}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE2_ROOT="${STAGE2_ROOT:-${PROJECT_ROOT}/results/case_study/factor_xa_stage2}"
STAGE3_ROOT="${STAGE3_ROOT:-${PROJECT_ROOT}/results/case_study/factor_xa_stage3_docking}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/case_study/factor_xa_stage4_visualization_${MODEL}}"

echo "Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable, sys.version.split()[0])')"
echo "Model: ${MODEL}"
echo "Stage-2 input: ${STAGE2_ROOT}/${MODEL}"
echo "Stage-3 input: ${STAGE3_ROOT}"
echo "Stage-4 output: ${OUTPUT_DIR}"

"${PYTHON_BIN}" - <<'PY'
import importlib.util
missing = [name for name in ("numpy", "pandas", "matplotlib", "PIL")
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))
if importlib.util.find_spec("gemmi") is None and importlib.util.find_spec("Bio") is None:
    raise SystemExit("Missing mmCIF reader: install gemmi (recommended) or biopython")
PY

PYMOL_ARGS=()
if [[ -n "${PYMOL_BIN:-}" ]]; then
  PYMOL_ARGS=(--pymol "${PYMOL_BIN}")
elif ! command -v pymol >/dev/null 2>&1; then
  if ! "${PYTHON_BIN}" -c 'import pymol' >/dev/null 2>&1; then
    echo "PyMOL is missing. Install once in the current Python 3.10 environment:"
    echo "  pip install pymol-open-source"
    exit 2
  fi
fi

# Some CUDA/Miniconda base images contain an old /opt/conda/lib/libstdc++.so.6,
# while the PyMOL wheel requires GLIBCXX_3.4.30. Prefer the newer Ubuntu system
# runtime for this process when it is available; do not replace either file.
if ! "${PYTHON_BIN}" -c 'import pymol._cmd' >/dev/null 2>&1; then
  SYSTEM_LIBSTDCPP="${SYSTEM_LIBSTDCPP:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
  if [[ -r "${SYSTEM_LIBSTDCPP}" ]] && grep -a -q 'GLIBCXX_3.4.30' "${SYSTEM_LIBSTDCPP}"; then
    export LD_PRELOAD="${SYSTEM_LIBSTDCPP}${LD_PRELOAD:+:${LD_PRELOAD}}"
    echo "PyMOL compatibility: preloading ${SYSTEM_LIBSTDCPP}"
  else
    echo "PyMOL cannot load because GLIBCXX_3.4.30 is unavailable."
    echo "Current Conda runtime: /opt/conda/lib/libstdc++.so.6"
    echo "System runtime checked: ${SYSTEM_LIBSTDCPP}"
    echo "If conda is available, update only its C++ runtime with:"
    echo "  /opt/conda/bin/conda install -y -c conda-forge 'libstdcxx-ng>=12'"
    exit 3
  fi
fi

"${PYTHON_BIN}" -c 'import pymol._cmd; print("PyMOL binary import: OK")'

"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_factor_xa_stage4.py" \
  --model "${MODEL}" \
  --stage2-root "${STAGE2_ROOT}" \
  --stage3-root "${STAGE3_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  "${PYMOL_ARGS[@]}"

echo "Done. Reference-style main figure: ${OUTPUT_DIR}/factor_xa_reference_style_${MODEL}.png"
echo "Pocket figure: ${OUTPUT_DIR}/factor_xa_occlusion_pocket_${MODEL}.png"
echo "Overview figure: ${OUTPUT_DIR}/factor_xa_occlusion_structure_${MODEL}.png"
