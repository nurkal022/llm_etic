"""
QA Fine-tuning: Qwen3.5-4B as Sociology Consultant
Dataset: NVIDIA Aegis 2.0 — safe examples only (prompt_label==safe, response_label==safe)
Task: conversational QA with sociology system prompt

Outputs:
  outputs_qa/
  ├── checkpoints/
  ├── logs/training_log.json
  ├── plots/           — training loss, data stats
  ├── qwen_sociology_qa_lora/  — final LoRA adapter
  └── summary.json

Usage:
  python fine_tune/train_qa.py
  python fine_tune/train_qa.py --epochs 2 --eval_samples 200
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
_BG      = "#0D1117"
_CARD    = "#161B22"
_GRID    = "#21262D"
_TEXT    = "#E6EDF3"
_DIM     = "#8B949E"
_ACCENT  = "#58A6FF"
_GREEN   = "#3FB950"
_RED     = "#F85149"
_ORANGE  = "#D29922"
_PURPLE  = "#BC8CFF"

plt.rcParams.update({
    "figure.facecolor": _BG, "axes.facecolor": _CARD,
    "axes.edgecolor": _GRID, "axes.labelcolor": _TEXT,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": _GRID, "grid.alpha": 0.6,
    "text.color": _TEXT, "xtick.color": _DIM, "ytick.color": _DIM,
    "legend.facecolor": _CARD, "legend.edgecolor": _GRID,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.facecolor": _BG,
    "savefig.bbox": "tight", "font.family": "sans-serif", "font.size": 11,
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Ты — цифровой консультант по социологии. Ты помогаешь исследователям "
    "проектировать опросы, анализировать социальные данные, интерпретировать "
    "результаты исследований и применять корректную методологию. "
    "Отвечай профессионально и по существу."
)


def parse_args():
    p = argparse.ArgumentParser(description="QA fine-tune Qwen3.5-4B on Aegis safe pairs")
    p.add_argument("--epochs",       type=int,   default=1)
    p.add_argument("--max_steps",    type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=2)
    p.add_argument("--grad_accum",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--max_length",   type=int,   default=2048)
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--eval_samples", type=int,   default=200)
    p.add_argument("--output_dir",   type=str,   default="outputs_qa")
    return p.parse_args()


def setup_dirs(base: str) -> dict:
    dirs = {
        "base":        Path(base),
        "checkpoints": Path(base) / "checkpoints",
        "logs":        Path(base) / "logs",
        "plots":       Path(base) / "plots",
        "model":       Path(base) / "qwen_sociology_qa_lora",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_safe_pairs():
    """Load Aegis dataset and keep only safe prompt+response pairs."""
    print("Loading Aegis dataset...")
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")

    train_safe, test_safe = [], []
    for split_name, out_list in [("train", train_safe), ("test", test_safe)]:
        for sample in ds[split_name]:
            if (
                sample["prompt_label"] == "safe"
                and sample["response_label"] == "safe"
                and sample["prompt"]
                and sample["response"]
                and sample["prompt"] != "REDACTED"
            ):
                out_list.append(sample)

    print(f"  Train safe pairs: {len(train_safe)}")
    print(f"  Test  safe pairs: {len(test_safe)}")
    return train_safe, test_safe


def to_conversation(sample: dict) -> dict:
    """Convert Aegis safe pair to chat format with sociology system prompt."""
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": sample["prompt"]}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["response"]}],
            },
        ]
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_data_stats(train_data: list, test_data: list, dirs: dict):
    prompt_lens  = [len(s["prompt"])   for s in train_data]
    response_lens = [len(s["response"]) for s in train_data]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Aegis Safe Pairs — Dataset Statistics", fontsize=15, fontweight="bold")

    # 1. Train/Test split
    ax = axes[0]
    bars = ax.bar(["Train", "Test"], [len(train_data), len(test_data)],
                  color=[_ACCENT, _GREEN], alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, [len(train_data), len(test_data)]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                str(v), ha="center", fontweight="bold", fontsize=13, color=_TEXT)
    ax.set_title("Safe Pairs Split")
    ax.set_ylabel("Count")

    # 2. Prompt length distribution
    ax = axes[1]
    ax.hist(prompt_lens, bins=40, color=_ACCENT, alpha=0.75, edgecolor=_GRID)
    ax.axvline(np.mean(prompt_lens), color=_ORANGE, linewidth=2,
               label=f"Mean: {np.mean(prompt_lens):.0f}")
    ax.set_title("Prompt Length Distribution")
    ax.set_xlabel("Characters")
    ax.legend()

    # 3. Response length distribution
    ax = axes[2]
    ax.hist(response_lens, bins=40, color=_GREEN, alpha=0.75, edgecolor=_GRID)
    ax.axvline(np.mean(response_lens), color=_ORANGE, linewidth=2,
               label=f"Mean: {np.mean(response_lens):.0f}")
    ax.set_title("Response Length Distribution")
    ax.set_xlabel("Characters")
    ax.legend()

    fig.tight_layout()
    fig.savefig(dirs["plots"] / "01_data_stats.png")
    plt.close(fig)
    print("  Saved: 01_data_stats.png")


def plot_training_loss(log: list, dirs: dict):
    steps  = [e["step"]        for e in log if "loss" in e]
    losses = [e["loss"]        for e in log if "loss" in e]
    lrs    = [e.get("lr", 0)   for e in log if "loss" in e]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("QA Fine-tuning — Training Dynamics", fontsize=15, fontweight="bold")

    # Loss
    ax1.plot(steps, losses, color=_ACCENT, linewidth=1.2, alpha=0.5, label="Loss")
    if len(losses) > 20:
        w = max(5, len(losses) // 30)
        smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
        sx = steps[w - 1:][:len(smooth)]
        ax1.plot(sx, smooth, color=_GREEN, linewidth=2.5, label=f"Smoothed (w={w})")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # LR
    ax2.plot(steps, lrs, color=_ORANGE, linewidth=1.5)
    ax2.set_ylabel("Learning Rate")
    ax2.set_xlabel("Step")

    fig.tight_layout()
    fig.savefig(dirs["plots"] / "02_training_loss.png")
    plt.close(fig)
    print("  Saved: 02_training_loss.png")


def plot_response_quality(eval_results: list, dirs: dict):
    """Compare predicted vs reference response lengths."""
    ref_lens  = [r["ref_len"]  for r in eval_results]
    pred_lens = [r["pred_len"] for r in eval_results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("QA Evaluation — Response Quality", fontsize=15, fontweight="bold")

    # Scatter: predicted vs reference length
    ax = axes[0]
    ax.scatter(ref_lens, pred_lens, color=_ACCENT, alpha=0.5, s=20, edgecolors="none")
    lim = max(max(ref_lens), max(pred_lens)) * 1.05
    ax.plot([0, lim], [0, lim], color=_GRID, linewidth=1.5, linestyle="--", label="y=x (perfect)")
    ax.set_xlabel("Reference Length (chars)")
    ax.set_ylabel("Generated Length (chars)")
    ax.set_title("Response Length: Generated vs Reference")
    ax.legend()

    # Distribution comparison
    ax = axes[1]
    ax.hist(ref_lens,  bins=30, color=_GREEN,  alpha=0.6, label="Reference",  edgecolor=_GRID)
    ax.hist(pred_lens, bins=30, color=_ACCENT, alpha=0.6, label="Generated", edgecolor=_GRID)
    ax.set_xlabel("Response Length (chars)")
    ax.set_ylabel("Count")
    ax.set_title("Response Length Distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(dirs["plots"] / "03_response_quality.png")
    plt.close(fig)
    print("  Saved: 03_response_quality.png")


def plot_dashboard(log: list, eval_results: list, train_size: int, test_size: int, dirs: dict):
    steps  = [e["step"] for e in log if "loss" in e]
    losses = [e["loss"] for e in log if "loss" in e]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("QA Fine-tuning Dashboard — Qwen3.5-4B Sociology Consultant", fontsize=16, fontweight="bold")
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    # 1. Loss curve
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(steps, losses, color=_ACCENT, linewidth=1, alpha=0.4)
    if len(losses) > 20:
        w = max(5, len(losses) // 30)
        smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
        ax.plot(steps[w-1:][:len(smooth)], smooth, color=_GREEN, linewidth=2.5, label="Smoothed loss")
    ax.set_title("Training Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()

    # 2. Dataset sizes
    ax = fig.add_subplot(gs[0, 2])
    bars = ax.bar(["Train", "Test"], [train_size, test_size],
                  color=[_ACCENT, _GREEN], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, [train_size, test_size]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(v), ha="center", fontweight="bold", color=_TEXT)
    ax.set_title("Dataset Split (safe pairs)")

    # 3. Response length scatter
    ax = fig.add_subplot(gs[1, :2])
    ref_lens  = [r["ref_len"]  for r in eval_results]
    pred_lens = [r["pred_len"] for r in eval_results]
    ax.scatter(ref_lens, pred_lens, color=_PURPLE, alpha=0.4, s=15)
    lim = max(max(ref_lens), max(pred_lens)) * 1.05
    ax.plot([0, lim], [0, lim], color=_GRID, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Reference Length")
    ax.set_ylabel("Generated Length")
    ax.set_title("Response Length Correlation")

    # 4. Key stats
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    final_loss = losses[-1] if losses else 0
    avg_gen = np.mean(pred_lens) if pred_lens else 0
    avg_ref = np.mean(ref_lens) if ref_lens else 0
    stats = [
        ("Final Loss",     f"{final_loss:.4f}"),
        ("Train samples",  f"{train_size:,}"),
        ("Eval samples",   f"{len(eval_results):,}"),
        ("Avg gen length", f"{avg_gen:.0f} chars"),
        ("Avg ref length", f"{avg_ref:.0f} chars"),
        ("Steps",          f"{steps[-1] if steps else 0:,}"),
    ]
    y = 0.95
    ax.text(0.5, 1.0, "Summary", ha="center", va="top", fontsize=13,
            fontweight="bold", color=_TEXT, transform=ax.transAxes)
    for label, val in stats:
        ax.text(0.1, y, label, fontsize=11, color=_DIM, transform=ax.transAxes)
        ax.text(0.9, y, val,   fontsize=11, color=_ACCENT, fontweight="bold",
                ha="right", transform=ax.transAxes)
        y -= 0.14
    ax.set_facecolor(_CARD)

    fig.savefig(dirs["plots"] / "04_dashboard.png")
    plt.close(fig)
    print("  Saved: 04_dashboard.png")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, test_data: list, num_samples: int, dirs: dict) -> list:
    from unsloth import FastVisionModel

    FastVisionModel.for_inference(model)
    num_eval = min(num_samples, len(test_data))
    results = []

    print(f"\nEvaluating on {num_eval} samples...")
    for i in tqdm(range(num_eval), desc="Eval"):
        sample = test_data[i]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",   "content": [{"type": "text", "text": sample["prompt"]}]},
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(None, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256, use_cache=True,
                                    temperature=0.3, min_p=0.1)

        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({
            "prompt":   sample["prompt"],
            "reference": sample["response"],
            "generated": generated,
            "ref_len":  len(sample["response"]),
            "pred_len": len(generated),
        })

    # Save predictions
    with open(dirs["logs"] / "eval_predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    avg_ref  = np.mean([r["ref_len"]  for r in results])
    avg_pred = np.mean([r["pred_len"] for r in results])
    print(f"  Avg reference length: {avg_ref:.0f} chars")
    print(f"  Avg generated length: {avg_pred:.0f} chars")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    dirs = setup_dirs(args.output_dir)

    print("=" * 60)
    print("QA Fine-tuning: Qwen3.5-4B → Sociology Consultant")
    print("Dataset: Aegis safe pairs only")
    print("=" * 60)

    # --- Data ---
    train_data, test_data = load_safe_pairs()
    train_convs = [to_conversation(s) for s in train_data]
    test_convs  = [to_conversation(s) for s in test_data]

    plot_data_stats(train_data, test_data, dirs)

    # --- Model ---
    from unsloth import FastVisionModel
    from trl import SFTTrainer, SFTConfig

    print("\nLoading model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen3.5-4B",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # --- Dataset for trainer ---
    from datasets import Dataset as HFDataset

    def format_for_trainer(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    hf_train = HFDataset.from_list(train_convs).map(
        format_for_trainer, remove_columns=["messages"]
    )

    # --- Trainer ---
    total_steps = args.max_steps or (
        (len(train_convs) // (args.batch_size * args.grad_accum)) * args.epochs
    )

    sft_config = SFTConfig(
        output_dir=str(dirs["checkpoints"]),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps or -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_train,
        args=sft_config,
    )

    # --- Train ---
    print(f"\nTraining for {args.epochs} epoch(s), ~{total_steps} steps...")
    t0 = time.time()
    trainer_output = trainer.train()
    train_time = time.time() - t0
    print(f"Training done in {train_time/60:.1f} min")

    # Collect log
    training_log = [
        {"step": e["step"], "loss": e["loss"], "lr": e.get("learning_rate", 0)}
        for e in trainer.state.log_history
        if "loss" in e
    ]
    with open(dirs["logs"] / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    plot_training_loss(training_log, dirs)

    # --- Evaluate ---
    eval_results = evaluate(model, tokenizer, test_data, args.eval_samples, dirs)
    plot_response_quality(eval_results, dirs)
    plot_dashboard(training_log, eval_results, len(train_data), len(test_data), dirs)

    # --- Save model ---
    print("\nSaving LoRA adapter...")
    model.save_pretrained(str(dirs["model"]))
    tokenizer.save_pretrained(str(dirs["model"]))

    # --- Summary ---
    summary = {
        "model":           "unsloth/Qwen3.5-4B",
        "dataset":         "nvidia/Aegis-AI-Content-Safety-Dataset-2.0 (safe only)",
        "train_samples":   len(train_data),
        "test_samples":    len(test_data),
        "epochs":          args.epochs,
        "total_steps":     training_log[-1]["step"] if training_log else 0,
        "final_loss":      training_log[-1]["loss"] if training_log else None,
        "train_time_min":  round(train_time / 60, 1),
        "lora_path":       str(dirs["model"]),
        "avg_gen_length":  round(np.mean([r["pred_len"] for r in eval_results]), 0),
        "avg_ref_length":  round(np.mean([r["ref_len"]  for r in eval_results]), 0),
    }
    with open(dirs["base"] / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  LoRA saved : {dirs['model']}")
    print(f"  Plots      : {dirs['plots']}")
    print(f"  Train loss : {summary['final_loss']:.4f}")
    print(f"  Time       : {summary['train_time_min']} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
