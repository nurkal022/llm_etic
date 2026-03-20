#!/bin/bash
# ============================================================================
#  QA Fine-tune: Qwen3.5-4B as Sociology Consultant
#  Dataset: Aegis safe pairs only (~4682 examples)
#  Run: bash fine_tune/run_qa.sh
# ============================================================================

set -euo pipefail

echo "============================================"
echo " Qwen3.5-4B  x  Sociology QA Fine-tuning"
echo " Dataset: Aegis safe pairs only"
echo "============================================"

# ---- Config (edit if needed) -----------------------------------------------
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
LR=2e-4
LORA_R=16
MAX_LENGTH=2048
EVAL_SAMPLES=200
OUTPUT_DIR="outputs_qa"
EXTRA_ARGS=""
# For quick test: EXTRA_ARGS="--max_steps 30 --eval_samples 20"
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "${SCRIPT_DIR}/venv/bin/activate"

export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "[GPU]"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "[WARNING] nvidia-smi not found"
fi

echo ""
echo "[START] QA Training..."
echo ""

cd "${SCRIPT_DIR}"

python fine_tune/train_qa.py \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --lora_r ${LORA_R} \
    --max_length ${MAX_LENGTH} \
    --eval_samples ${EVAL_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    ${EXTRA_ARGS}

echo ""
echo "============================================"
echo " DONE. Results in: ${OUTPUT_DIR}/"
echo " LoRA adapter: ${OUTPUT_DIR}/qwen_sociology_qa_lora/"
echo "============================================"
