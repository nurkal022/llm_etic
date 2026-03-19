"""
Method 1: Baseline — raw Qwen3.5-4B without any safety modifications.
No system prompt, no RAG, no fine-tuning.
"""

import json
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_NAME, GENERATION_CONFIG, SCENARIOS_PATH, RESULTS_DIR


def run():
    from unsloth import FastVisionModel

    print("=" * 60)
    print("Method 1: BASELINE (raw model)")
    print("=" * 60)

    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME, load_in_4bit=True, use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)

    with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    results = []
    for sc in scenarios:
        messages = [
            {"role": "user", "content": sc["prompt"]},
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(None, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model.generate(**inputs, **GENERATION_CONFIG, use_cache=True)

        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        results.append({
            "id": sc["id"],
            "category": sc["category"],
            "prompt": sc["prompt"],
            "expected": sc["expected"],
            "response": response,
            "method": "baseline",
        })
        print(f"  [{sc['id']:02d}] {sc['category']}")

    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} results to {out_dir / 'baseline.json'}")


if __name__ == "__main__":
    run()
