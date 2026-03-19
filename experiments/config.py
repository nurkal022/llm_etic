"""
Shared configuration for all experiments.
"""

# Base model used across all methods
MODEL_NAME = "unsloth/Qwen3.5-4B"

# Fine-tuned LoRA adapter path (after running fine_tune/train.py)
FINETUNED_LORA_PATH = "outputs/qwen_aegis_safety_lora"

# Generation params — same for all methods for fair comparison
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}

# Paths
SCENARIOS_PATH = "experiments/scenarios.json"
RESULTS_DIR = "experiments/results"
KNOWLEDGE_BASE_DIR = "experiments/knowledge_base"

# Evaluation — GPT-4o as judge (or manual)
JUDGE_MODEL = "gpt-4o"

# Categories in test scenarios
UNSAFE_CATEGORIES = [
    "data_falsification",
    "discrimination",
    "privacy",
    "manipulation",
    "harmful_research",
    "bias_methodology",
    "political_manipulation",
]

SAFE_CATEGORIES = [
    "safe_methodology",
    "safe_ethics",
    "safe_analysis",
]
