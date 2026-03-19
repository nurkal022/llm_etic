#!/bin/bash
# ============================================================================
#  Run all 4 experiment methods + evaluation
#  Usage: bash experiments/run_all.sh
# ============================================================================

set -euo pipefail

echo "============================================"
echo "  Running all experiment methods"
echo "============================================"

echo ""
echo "[1/5] Baseline..."
python experiments/methods/baseline.py

echo ""
echo "[2/5] Prompt Engineering..."
python experiments/methods/prompt_eng.py

echo ""
echo "[3/5] RAG..."
python experiments/methods/rag.py

echo ""
echo "[4/5] Fine-tune..."
python experiments/methods/finetune.py

echo ""
echo "[5/5] Evaluation & comparison..."
python experiments/evaluate.py

echo ""
echo "============================================"
echo "  DONE. Results in experiments/results/"
echo "============================================"
