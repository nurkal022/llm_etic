"""
Method 1: Baseline
Raw Qwen3.5-4B with a minimal "sociology consultant" identity.
No safety instructions, no ethics knowledge, no fine-tuning.
This is the control group.
"""

import json
import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MODEL_NAME, GENERATION_CONFIG, SCENARIOS_PATH, RESULTS_DIR, SOCIOLOGY_SYSTEM
)


def run():
    from unsloth import FastVisionModel

    print("=" * 60)
    print("Method 1: BASELINE (raw model, no ethics)")
    print("=" * 60)

    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)

    with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    results = []
    total_time = 0.0

    for i, sc in enumerate(scenarios):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SOCIOLOGY_SYSTEM}]},
            {"role": "user",   "content": [{"type": "text", "text": sc["prompt"]}]},
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(None, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

        t0 = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, **GENERATION_CONFIG, use_cache=True)
        elapsed = time.time() - t0
        total_time += elapsed

        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        results.append({
            "id":       sc["id"],
            "category": sc["category"],
            "prompt":   sc["prompt"],
            "expected": sc["expected"],
            "response": response,
            "method":   "baseline",
            "latency_s": round(elapsed, 2),
        })
        print(f"  [{i+1:02d}/{len(scenarios)}] {sc['category']:<25} | {elapsed:.1f}s | {len(response)} chars")

    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    avg_lat = total_time / len(results) if results else 0
    print(f"\nDone. Saved {len(results)} results to {out_path}")
    print(f"Total time: {total_time:.1f}s | Avg latency: {avg_lat:.2f}s/sample")


if __name__ == "__main__":
    run()
