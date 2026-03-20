#!/bin/bash
# ============================================================================
#  Qorgau Benchmark — 5 Methods Comparison
#  Run: bash experiments/run_qorgau.sh
# ============================================================================

set -euo pipefail

echo "============================================"
echo " Qorgau Benchmark: 5 Methods × Russian Safety"
echo "============================================"

# ---- Config ----------------------------------------------------------------
LANG="ru"            # ru | kz | cs
SAMPLE=15            # max prompts per risk_area (~6 areas = ~90 total)
OUTPUT_DIR="experiments/qorgau_results"
# To run only specific methods:
# METHODS="--methods baseline prompt_eng"
METHODS=""
# For quick test:
# SAMPLE=3
# METHODS="--methods baseline safety_finetune"
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "${SCRIPT_DIR}/venv/bin/activate"

export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "[GPU]"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi

echo ""
echo "[START] Running benchmark..."
echo ""

cd "${SCRIPT_DIR}"

python experiments/benchmark_qorgau.py \
    --lang    ${LANG}       \
    --sample  ${SAMPLE}     \
    --output_dir ${OUTPUT_DIR} \
    ${METHODS}

echo ""
echo "============================================"
echo " DONE. Results in: ${OUTPUT_DIR}/"
echo " Plots:            ${OUTPUT_DIR}/plots/"
echo "============================================"
