#!/bin/bash
# ============================================================================
#  Fine-tune Qwen3.5-4B on Aegis Safety Dataset
#  Run: bash fine_tune/run.sh
# ============================================================================

set -euo pipefail

echo "============================================"
echo " Qwen3.5-4B  x  Aegis Safety Fine-tuning"
echo "============================================"

# ---- Config (edit these) ---------------------------------------------------
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
LR=2e-4
LORA_R=16
MAX_LENGTH=2048
EVAL_SAMPLES=500
SAVE_STEPS=100
OUTPUT_DIR="outputs"
# Use --load_in_4bit if GPU has < 16GB VRAM
EXTRA_ARGS=""
# EXTRA_ARGS="--load_in_4bit"
# For quick test: EXTRA_ARGS="--max_steps 30 --eval_samples 50"
# ----------------------------------------------------------------------------

# Activate venv
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "${SCRIPT_DIR}/venv/bin/activate"

# Set HuggingFace cache to home directory
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

# Check GPU
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "[GPU]"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "[WARNING] nvidia-smi not found — make sure CUDA is available"
fi

# Install deps if needed
if ! python -c "import unsloth" 2>/dev/null; then
    echo ""
    echo "[INSTALL] Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "[START] Training..."
echo ""

python fine_tune/train.py \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --lora_r ${LORA_R} \
    --max_length ${MAX_LENGTH} \
    --eval_samples ${EVAL_SAMPLES} \
    --save_steps ${SAVE_STEPS} \
    --output_dir ${OUTPUT_DIR} \
    ${EXTRA_ARGS}

echo ""
echo "============================================"
echo " DONE. Results in: ${OUTPUT_DIR}/"
echo "============================================"
