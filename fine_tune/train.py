"""
Fine-tuning Qwen3.5 (4B) for Content Safety Classification
using NVIDIA Aegis AI Content Safety Dataset 2.0

Saves all training artifacts:
  outputs/
  ├── checkpoints/          — model checkpoints every N steps
  ├── logs/                 — training logs (JSON)
  ├── plots/                — training loss, confusion matrices, bar charts, etc.
  ├── metrics/              — evaluation metrics (JSON + classification reports)
  ├── qwen_aegis_safety_lora/  — final LoRA adapters
  └── training_args.json    — full training config

Usage:
  pip install unsloth transformers==5.3.0 trl==0.22.2 datasets scikit-learn matplotlib
  python train.py [--max_steps 30] [--epochs 1] [--eval_samples 200] [--batch_size 2]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import numpy as np
import torch
from datasets import load_dataset
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Global plot style — dark, modern, publication-grade
# ---------------------------------------------------------------------------
_BG       = "#0D1117"
_BG_CARD  = "#161B22"
_GRID     = "#21262D"
_TEXT     = "#E6EDF3"
_TEXT_DIM = "#8B949E"
_ACCENT   = "#58A6FF"
_GREEN    = "#3FB950"
_RED      = "#F85149"
_ORANGE   = "#D29922"
_PURPLE   = "#BC8CFF"
_CYAN     = "#39D2C0"
_PINK     = "#F778BA"

_PALETTE = [_ACCENT, _GREEN, _ORANGE, _RED, _PURPLE, _CYAN, _PINK,
            "#FFA657", "#79C0FF", "#7EE787", "#FF7B72", "#D2A8FF"]

plt.rcParams.update({
    "figure.facecolor":   _BG,
    "axes.facecolor":     _BG_CARD,
    "axes.edgecolor":     _GRID,
    "axes.labelcolor":    _TEXT,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "axes.grid":          True,
    "grid.color":         _GRID,
    "grid.alpha":         0.6,
    "grid.linewidth":     0.5,
    "text.color":         _TEXT,
    "xtick.color":        _TEXT_DIM,
    "ytick.color":        _TEXT_DIM,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.facecolor":   _BG_CARD,
    "legend.edgecolor":   _GRID,
    "legend.fontsize":    10,
    "figure.dpi":         150,
    "savefig.dpi":        150,
    "savefig.facecolor":  _BG,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.3,
    "font.family":        "sans-serif",
    "font.size":          11,
})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen3.5-4B on Aegis Safety")
    p.add_argument("--max_steps", type=int, default=None, help="Max training steps (overrides epochs)")
    p.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    p.add_argument("--eval_samples", type=int, default=200, help="Number of test samples for evaluation")
    p.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size")
    p.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    p.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    p.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    p.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

def setup_dirs(base: str) -> dict:
    dirs = {
        "base": Path(base),
        "checkpoints": Path(base) / "checkpoints",
        "logs": Path(base) / "logs",
        "plots": Path(base) / "plots",
        "metrics": Path(base) / "metrics",
        "model": Path(base) / "qwen_aegis_safety_lora",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# System prompt for safety classification
# ---------------------------------------------------------------------------

INSTRUCTION = (
    "You are a content safety classifier. Analyze the provided content and "
    "determine whether it is safe or unsafe. If the content is unsafe, identify "
    "the violated safety categories.\n\n"
    "Safety categories: Hate/Identity Hate, Sexual, Suicide and Self Harm, "
    "Violence, Guns/Illegal Weapons, Threat, PII/Privacy, Sexual (minor), "
    "Criminal Planning/Confessions, Harassment, Controlled/Regulated Substances, "
    "Profanity\n\n"
    "Classify the safety of the following content:"
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_user_message(prompt_text: str, response_text: str) -> str:
    if response_text:
        return (
            f"{INSTRUCTION}\n\n"
            f"[User Prompt]:\n{prompt_text}\n\n"
            f"[Assistant Response]:\n{response_text}"
        )
    return f"{INSTRUCTION}\n\n[User Prompt]:\n{prompt_text}"


def convert_to_conversation(sample: dict) -> dict:
    prompt_text = sample["prompt"] or ""
    response_text = sample["response"] or ""
    user_msg = build_user_message(prompt_text, response_text)

    parts = [f"Prompt label: {sample['prompt_label']}"]
    if sample["response_label"]:
        parts.append(f"Response label: {sample['response_label']}")
    categories = sample["violated_categories"] if sample["violated_categories"] else "None"
    parts.append(f"Violated categories: {categories}")
    assistant_msg = "\n".join(parts)

    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_msg}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]},
        ]
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, test_data, num_samples: int, dirs: dict):
    from unsloth import FastVisionModel

    FastVisionModel.for_inference(model)
    num_eval = min(num_samples, len(test_data))

    prompt_preds, prompt_trues = [], []
    response_preds, response_trues = [], []
    category_preds, category_trues = [], []
    raw_predictions = []

    print(f"\nEvaluating on {num_eval} samples...")
    for i in tqdm(range(num_eval), desc="Eval"):
        sample = test_data[i]
        prompt_text = sample["prompt"] or ""
        response_text = sample["response"] or ""

        if prompt_text == "REDACTED":
            continue

        user_msg = build_user_message(prompt_text, response_text)
        messages = [{"role": "user", "content": [{"type": "text", "text": user_msg}]}]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(None, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=128, use_cache=True, temperature=0.1, min_p=0.1)

        decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        decoded_lower = decoded.lower()

        # Parse prompt label
        if "prompt label: unsafe" in decoded_lower:
            pred_prompt = "unsafe"
        elif "prompt label: safe" in decoded_lower:
            pred_prompt = "safe"
        else:
            pred_prompt = None

        if pred_prompt and sample["prompt_label"]:
            prompt_preds.append(pred_prompt)
            prompt_trues.append(sample["prompt_label"])

        # Parse response label
        if sample["response_label"]:
            if "response label: unsafe" in decoded_lower:
                pred_response = "unsafe"
            elif "response label: safe" in decoded_lower:
                pred_response = "safe"
            else:
                pred_response = None
            if pred_response:
                response_preds.append(pred_response)
                response_trues.append(sample["response_label"])

        # Parse categories
        true_cats = sample["violated_categories"] if sample["violated_categories"] else "None"
        category_trues.append(true_cats)
        if "violated categories:" in decoded_lower:
            pred_cats = decoded[decoded_lower.index("violated categories:") + len("violated categories:"):].strip()
            pred_cats = pred_cats.split("\n")[0].strip()
        else:
            pred_cats = "None"
        category_preds.append(pred_cats)

        raw_predictions.append({
            "index": i,
            "prompt_true": sample["prompt_label"],
            "prompt_pred": pred_prompt,
            "response_true": sample["response_label"],
            "response_pred": pred_response if sample["response_label"] else None,
            "categories_true": true_cats,
            "categories_pred": pred_cats,
            "raw_output": decoded,
        })

    results = {
        "prompt_preds": prompt_preds,
        "prompt_trues": prompt_trues,
        "response_preds": response_preds,
        "response_trues": response_trues,
        "category_preds": category_preds,
        "category_trues": category_trues,
    }

    # Save raw predictions
    with open(dirs["metrics"] / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(raw_predictions, f, ensure_ascii=False, indent=2)

    # Compute and save metrics
    metrics = {"num_eval_samples": num_eval}

    if prompt_trues:
        p_acc = accuracy_score(prompt_trues, prompt_preds)
        p_f1_macro = f1_score(prompt_trues, prompt_preds, labels=["safe", "unsafe"], average="macro")
        p_f1_weighted = f1_score(prompt_trues, prompt_preds, labels=["safe", "unsafe"], average="weighted")
        metrics["prompt"] = {
            "accuracy": round(p_acc, 4),
            "f1_macro": round(p_f1_macro, 4),
            "f1_weighted": round(p_f1_weighted, 4),
            "n_samples": len(prompt_trues),
        }
        report = classification_report(prompt_trues, prompt_preds, labels=["safe", "unsafe"], digits=4, output_dict=True)
        metrics["prompt"]["classification_report"] = report

        print(f"\n{'='*50}")
        print(f"PROMPT LABEL — Accuracy: {p_acc:.3f} | F1 macro: {p_f1_macro:.3f}")
        print(f"{'='*50}")
        print(classification_report(prompt_trues, prompt_preds, labels=["safe", "unsafe"], digits=3))

    if response_trues:
        r_acc = accuracy_score(response_trues, response_preds)
        r_f1_macro = f1_score(response_trues, response_preds, labels=["safe", "unsafe"], average="macro")
        r_f1_weighted = f1_score(response_trues, response_preds, labels=["safe", "unsafe"], average="weighted")
        metrics["response"] = {
            "accuracy": round(r_acc, 4),
            "f1_macro": round(r_f1_macro, 4),
            "f1_weighted": round(r_f1_weighted, 4),
            "n_samples": len(response_trues),
        }
        report = classification_report(response_trues, response_preds, labels=["safe", "unsafe"], digits=4, output_dict=True)
        metrics["response"]["classification_report"] = report

        print(f"\n{'='*50}")
        print(f"RESPONSE LABEL — Accuracy: {r_acc:.3f} | F1 macro: {r_f1_macro:.3f}")
        print(f"{'='*50}")
        print(classification_report(response_trues, response_preds, labels=["safe", "unsafe"], digits=3))

    with open(dirs["metrics"] / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Metrics saved to {dirs['metrics']}")
    return results


# ---------------------------------------------------------------------------
# Helper — exponential moving average for smoothing curves
# ---------------------------------------------------------------------------

def _ema(values, alpha=0.3):
    s = []
    for i, v in enumerate(values):
        if i == 0:
            s.append(v)
        else:
            s.append(alpha * v + (1 - alpha) * s[-1])
    return s


def _extract_categories(cat_list):
    cats = []
    for c in cat_list:
        if c and c != "None":
            cats.extend([x.strip() for x in c.split(",") if x.strip()])
    return Counter(cats)


# ===========================================================================
#  PLOT 1 — Training Loss (with EMA smooth + annotations)
# ===========================================================================

def save_training_loss_plot(trainer, dirs: dict):
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    steps  = [e["step"] for e in trainer.state.log_history if "loss" in e]
    if not losses:
        return

    smooth = _ema(losses, alpha=0.25)

    fig, ax = plt.subplots(figsize=(10, 5))

    # raw loss — thin, low alpha
    ax.plot(steps, losses, linewidth=1, alpha=0.35, color=_ACCENT, label="Raw loss")
    # smoothed
    ax.plot(steps, smooth, linewidth=2.5, color=_ACCENT, label="Smoothed (EMA)")
    # glow fill
    ax.fill_between(steps, smooth, alpha=0.08, color=_ACCENT)

    # start / end annotations
    ax.annotate(f"{losses[0]:.3f}", xy=(steps[0], losses[0]),
                xytext=(15, 15), textcoords="offset points",
                fontsize=9, color=_ORANGE,
                arrowprops=dict(arrowstyle="->", color=_ORANGE, lw=1.2))
    ax.annotate(f"{losses[-1]:.3f}", xy=(steps[-1], losses[-1]),
                xytext=(-50, 15), textcoords="offset points",
                fontsize=9, color=_GREEN,
                arrowprops=dict(arrowstyle="->", color=_GREEN, lw=1.2))

    # min loss marker
    min_idx = int(np.argmin(losses))
    ax.scatter([steps[min_idx]], [losses[min_idx]], s=60, color=_GREEN,
               zorder=5, edgecolors="white", linewidths=0.8)
    ax.annotate(f"min {losses[min_idx]:.3f}", xy=(steps[min_idx], losses[min_idx]),
                xytext=(10, -20), textcoords="offset points",
                fontsize=8, color=_GREEN)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.grid(axis="y")

    fig.savefig(dirs["plots"] / "01_training_loss.png")
    plt.close(fig)

    drop = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f} (−{drop:.1f}%)")


# ===========================================================================
#  PLOT 2 — Learning-rate schedule
# ===========================================================================

def save_lr_schedule_plot(trainer, dirs: dict):
    lrs = [e["learning_rate"] for e in trainer.state.log_history if "learning_rate" in e]
    steps = [e["step"] for e in trainer.state.log_history if "learning_rate" in e]
    if not lrs:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, lrs, linewidth=2, color=_PURPLE)
    ax.fill_between(steps, lrs, alpha=0.10, color=_PURPLE)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))
    ax.grid(axis="y")

    fig.savefig(dirs["plots"] / "02_lr_schedule.png")
    plt.close(fig)


# ===========================================================================
#  PLOT 3 — Confusion matrices (heatmap style, counts + percentages)
# ===========================================================================

def save_confusion_matrices(results: dict, dirs: dict):
    labels = ["safe", "unsafe"]
    has_resp = bool(results["response_trues"])
    ncols = 2 if has_resp else 1

    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    cmap_blue = LinearSegmentedColormap.from_list("cb", [_BG_CARD, _ACCENT])
    cmap_org  = LinearSegmentedColormap.from_list("co", [_BG_CARD, _ORANGE])

    def _draw_cm(ax, y_true, y_pred, title, cmap):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_pct = cm.astype(float) / cm.sum() * 100
        ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100, aspect="equal")
        for i in range(2):
            for j in range(2):
                txt = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"
                ax.text(j, i, txt, ha="center", va="center", fontsize=13,
                        fontweight="bold", color="white",
                        path_effects=[pe.withStroke(linewidth=2, foreground=_BG)])
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title(title, pad=12)
        # disable grid for imshow
        ax.grid(False)

    _draw_cm(axes[0], results["prompt_trues"], results["prompt_preds"],
             "Prompt Safety — Confusion Matrix", cmap_blue)

    if has_resp:
        _draw_cm(axes[1], results["response_trues"], results["response_preds"],
                 "Response Safety — Confusion Matrix", cmap_org)

    fig.tight_layout(pad=2.0)
    fig.savefig(dirs["plots"] / "03_confusion_matrices.png")
    plt.close(fig)


# ===========================================================================
#  PLOT 4 — Precision / Recall / F1 (glass-bar style)
# ===========================================================================

def save_precision_recall_f1(results: dict, dirs: dict):
    labels = ["safe", "unsafe"]
    has_resp = bool(results["response_trues"])
    ncols = 2 if has_resp else 1

    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    metric_colors = [_GREEN, _ACCENT, _ORANGE]
    metric_names  = ["Precision", "Recall", "F1-Score"]

    def _draw_bars(ax, y_true, y_pred, title):
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None)
        x = np.arange(len(labels))
        w = 0.22

        for k, (vals, col, name) in enumerate(zip([prec, rec, f1], metric_colors, metric_names)):
            bars = ax.bar(x + (k - 1) * w, vals, w, label=name, color=col, alpha=0.85,
                          edgecolor="white", linewidth=0.4, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color=col)

        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score")
        ax.set_title(title, pad=10)
        ax.legend(loc="upper center", ncol=3, framealpha=0.7, fontsize=9)
        ax.grid(axis="y")

    _draw_bars(axes[0], results["prompt_trues"], results["prompt_preds"],
               "Prompt Safety — Classification Metrics")
    if has_resp:
        _draw_bars(axes[1], results["response_trues"], results["response_preds"],
                   "Response Safety — Classification Metrics")

    fig.tight_layout(pad=2.0)
    fig.savefig(dirs["plots"] / "04_precision_recall_f1.png")
    plt.close(fig)


# ===========================================================================
#  PLOT 5 — Violated categories (horizontal lollipop chart)
# ===========================================================================

def save_category_distribution(results: dict, dirs: dict):
    true_counts = _extract_categories(results["category_trues"])
    pred_counts = _extract_categories(results["category_preds"])
    all_cats = sorted(
        set(list(true_counts.keys()) + list(pred_counts.keys())),
        key=lambda c: true_counts.get(c, 0), reverse=True,
    )
    if not all_cats:
        return

    fig, ax = plt.subplots(figsize=(11, max(5, len(all_cats) * 0.55)))
    y = np.arange(len(all_cats))
    true_v = np.array([true_counts.get(c, 0) for c in all_cats])
    pred_v = np.array([pred_counts.get(c, 0) for c in all_cats])

    # lollipop lines
    for i in range(len(all_cats)):
        ax.plot([true_v[i], pred_v[i]], [y[i], y[i]], color=_GRID, linewidth=1.5, zorder=1)

    ax.scatter(true_v, y, s=90, color=_ACCENT, zorder=3, label="Ground Truth", edgecolors="white", linewidths=0.6)
    ax.scatter(pred_v, y, s=90, color=_ORANGE, zorder=3, label="Predicted", edgecolors="white", linewidths=0.6)

    # value labels
    for i in range(len(all_cats)):
        if true_v[i] > 0:
            ax.text(true_v[i] + 0.4, y[i] + 0.15, str(true_v[i]), fontsize=8, color=_ACCENT, va="center")
        if pred_v[i] > 0:
            ax.text(pred_v[i] + 0.4, y[i] - 0.15, str(pred_v[i]), fontsize=8, color=_ORANGE, va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(all_cats, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Violated Safety Categories — True vs Predicted")
    ax.legend(loc="lower right", framealpha=0.8)
    ax.grid(axis="x")

    fig.tight_layout()
    fig.savefig(dirs["plots"] / "05_category_distribution.png")
    plt.close(fig)


# ===========================================================================
#  PLOT 6 — Label distribution (donut charts)
# ===========================================================================

def save_label_distribution_pie(results: dict, dirs: dict):
    colors = [_GREEN, _RED]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    def _donut(ax, counts_dict, title, n):
        vals = [counts_dict.get("safe", 0), counts_dict.get("unsafe", 0)]
        wedges, texts, autotexts = ax.pie(
            vals, labels=["Safe", "Unsafe"], colors=colors,
            autopct="%1.1f%%", startangle=90, pctdistance=0.78,
            textprops={"fontsize": 12, "color": _TEXT, "fontweight": "bold"},
            wedgeprops={"width": 0.45, "edgecolor": _BG, "linewidth": 2},
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_color("white")
            at.set_path_effects([pe.withStroke(linewidth=2, foreground=_BG)])
        ax.text(0, 0, f"n={n}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=_TEXT_DIM)
        ax.set_title(title, pad=14)

    _donut(axes[0], Counter(results["prompt_trues"]),
           "Ground Truth Labels", len(results["prompt_trues"]))
    _donut(axes[1], Counter(results["prompt_preds"]),
           "Predicted Labels", len(results["prompt_preds"]))

    fig.suptitle("Prompt Safety — Label Distribution", fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout(pad=2.0)
    fig.savefig(dirs["plots"] / "06_label_distribution.png")
    plt.close(fig)


# ===========================================================================
#  PLOT 7 — Per-category F1 heatmap
# ===========================================================================

def save_category_f1_heatmap(results: dict, dirs: dict):
    true_cats_flat = _extract_categories(results["category_trues"])
    pred_cats_flat = _extract_categories(results["category_preds"])
    all_cats = sorted(set(list(true_cats_flat.keys()) + list(pred_cats_flat.keys())),
                      key=lambda c: true_cats_flat.get(c, 0), reverse=True)
    if len(all_cats) < 2:
        return

    # Per-sample binary vectors for each category
    n = len(results["category_trues"])
    cat_true_bin = {c: np.zeros(n) for c in all_cats}
    cat_pred_bin = {c: np.zeros(n) for c in all_cats}

    for i, (t, p) in enumerate(zip(results["category_trues"], results["category_preds"])):
        for c in (x.strip() for x in t.split(",") if x.strip() and x.strip() != "None"):
            if c in cat_true_bin:
                cat_true_bin[c][i] = 1
        for c in (x.strip() for x in p.split(",") if x.strip() and x.strip() != "None"):
            if c in cat_pred_bin:
                cat_pred_bin[c][i] = 1

    precs, recs, f1s = [], [], []
    for c in all_cats:
        tp = np.sum((cat_true_bin[c] == 1) & (cat_pred_bin[c] == 1))
        fp = np.sum((cat_true_bin[c] == 0) & (cat_pred_bin[c] == 1))
        fn = np.sum((cat_true_bin[c] == 1) & (cat_pred_bin[c] == 0))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        precs.append(p); recs.append(r); f1s.append(f)

    data = np.array([precs, recs, f1s])  # 3 × n_cats
    cmap_heat = LinearSegmentedColormap.from_list("heat", [_BG_CARD, _CYAN, _GREEN])

    fig, ax = plt.subplots(figsize=(max(8, len(all_cats) * 0.9), 4))
    im = ax.imshow(data, cmap=cmap_heat, aspect="auto", vmin=0, vmax=1)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Precision", "Recall", "F1"], fontsize=11)
    ax.set_xticks(np.arange(len(all_cats)))
    ax.set_xticklabels(all_cats, rotation=45, ha="right", fontsize=9)
    ax.set_title("Per-Category Safety Metrics", pad=14)
    ax.grid(False)

    for i in range(3):
        for j in range(len(all_cats)):
            val = data[i, j]
            txt_col = "white" if val > 0.5 else _TEXT_DIM
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=txt_col)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=_TEXT_DIM)
    cbar.outline.set_edgecolor(_GRID)

    fig.tight_layout()
    fig.savefig(dirs["plots"] / "07_category_f1_heatmap.png")
    plt.close(fig)


# ===========================================================================
#  PLOT 8 — Radar chart (overall metrics overview)
# ===========================================================================

def save_radar_chart(results: dict, dirs: dict):
    labels_cls = ["safe", "unsafe"]
    if not results["prompt_trues"]:
        return

    prec, rec, f1, _ = precision_recall_fscore_support(
        results["prompt_trues"], results["prompt_preds"],
        labels=labels_cls, average=None,
    )
    acc = accuracy_score(results["prompt_trues"], results["prompt_preds"])
    f1_macro = f1_score(results["prompt_trues"], results["prompt_preds"],
                        labels=labels_cls, average="macro")

    metric_names = ["Accuracy", "F1 Macro", "Prec (safe)", "Rec (safe)", "Prec (unsafe)", "Rec (unsafe)"]
    values = [acc, f1_macro, prec[0], rec[0], prec[1], rec[1]]
    n = len(metric_names)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_facecolor(_BG_CARD)
    fig.patch.set_facecolor(_BG)

    ax.plot(angles_closed, values_closed, linewidth=2, color=_ACCENT, zorder=3)
    ax.fill(angles_closed, values_closed, alpha=0.15, color=_ACCENT)
    ax.scatter(angles, values, s=50, color=_ACCENT, zorder=4, edgecolors="white", linewidths=0.6)

    # value labels
    for a, v in zip(angles, values):
        ax.text(a, v + 0.07, f"{v:.2f}", ha="center", va="center",
                fontsize=9, fontweight="bold", color=_ACCENT)

    ax.set_xticks(angles)
    ax.set_xticklabels(metric_names, fontsize=10, color=_TEXT)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color=_TEXT_DIM)
    ax.spines["polar"].set_color(_GRID)
    ax.grid(color=_GRID, linewidth=0.5)
    ax.set_title("Model Performance Overview", y=1.08, fontsize=14, fontweight="bold", color=_TEXT)

    fig.tight_layout()
    fig.savefig(dirs["plots"] / "08_radar_chart.png")
    plt.close(fig)


# ===========================================================================
#  PLOT 9 — Error analysis (stacked bar: correct / wrong per class)
# ===========================================================================

def save_error_analysis(results: dict, dirs: dict):
    if not results["prompt_trues"]:
        return

    trues = np.array(results["prompt_trues"])
    preds = np.array(results["prompt_preds"])

    labels = ["safe", "unsafe"]
    correct = [np.sum((trues == l) & (preds == l)) for l in labels]
    wrong   = [np.sum((trues == l) & (preds != l)) for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    w = 0.45

    b1 = ax.bar(x, correct, w, label="Correct", color=_GREEN, alpha=0.85,
                edgecolor="white", linewidth=0.4, zorder=3)
    b2 = ax.bar(x, wrong, w, bottom=correct, label="Incorrect", color=_RED, alpha=0.85,
                edgecolor="white", linewidth=0.4, zorder=3)

    for i in range(len(labels)):
        total = correct[i] + wrong[i]
        if total > 0:
            pct = correct[i] / total * 100
            ax.text(x[i], total + 1, f"{pct:.0f}% correct",
                    ha="center", fontsize=10, fontweight="bold", color=_GREEN)

    # value labels
    for bar, val in zip(b1, correct):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    str(val), ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white")
    for bar, val, bot in zip(b2, wrong, correct):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                    str(val), ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Number of Samples")
    ax.set_title("Error Analysis — Correct vs Incorrect per Class")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.grid(axis="y")

    fig.tight_layout()
    fig.savefig(dirs["plots"] / "09_error_analysis.png")
    plt.close(fig)


# ===========================================================================
#  PLOT 10 — Dashboard summary (single figure with key numbers)
# ===========================================================================

def save_dashboard(trainer, results: dict, training_log: dict, dirs: dict):
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(_BG)

    # Title
    fig.text(0.5, 0.96, "Qwen3.5-4B  ×  Aegis Safety — Training Dashboard",
             ha="center", va="top", fontsize=20, fontweight="bold", color=_TEXT)

    # ---- Top-left: loss curve (small) ----
    ax1 = fig.add_axes([0.05, 0.52, 0.42, 0.38])
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    steps  = [e["step"] for e in trainer.state.log_history if "loss" in e]
    if losses:
        smooth = _ema(losses, 0.25)
        ax1.plot(steps, losses, linewidth=0.8, alpha=0.3, color=_ACCENT)
        ax1.plot(steps, smooth, linewidth=2, color=_ACCENT)
        ax1.fill_between(steps, smooth, alpha=0.08, color=_ACCENT)
    ax1.set_title("Training Loss", fontsize=12)
    ax1.set_xlabel("Step", fontsize=9)
    ax1.grid(axis="y")

    # ---- Top-right: confusion matrix (prompt) ----
    ax2 = fig.add_axes([0.55, 0.52, 0.38, 0.38])
    labels_cls = ["safe", "unsafe"]
    if results["prompt_trues"]:
        cm = confusion_matrix(results["prompt_trues"], results["prompt_preds"], labels=labels_cls)
        cm_pct = cm.astype(float) / cm.sum() * 100
        cmap_b = LinearSegmentedColormap.from_list("cb", [_BG_CARD, _ACCENT])
        ax2.imshow(cm_pct, cmap=cmap_b, vmin=0, vmax=100, aspect="equal")
        for i in range(2):
            for j in range(2):
                ax2.text(j, i, f"{cm[i,j]}\n{cm_pct[i,j]:.0f}%",
                         ha="center", va="center", fontsize=12, fontweight="bold",
                         color="white", path_effects=[pe.withStroke(linewidth=2, foreground=_BG)])
        ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
        ax2.set_xticklabels(labels_cls); ax2.set_yticklabels(labels_cls)
        ax2.set_xlabel("Predicted", fontsize=9); ax2.set_ylabel("Actual", fontsize=9)
    ax2.set_title("Prompt Confusion Matrix", fontsize=12)
    ax2.grid(False)

    # ---- Bottom: KPI cards ----
    kpi_y = 0.22
    card_h = 0.22
    card_w = 0.17
    gap = 0.03

    def _kpi_card(x, y, label, value, color):
        rect = plt.Rectangle((x, y), card_w, card_h, transform=fig.transFigure,
                              facecolor=_BG_CARD, edgecolor=color, linewidth=1.5,
                              clip_on=False, zorder=5)
        fig.patches.append(rect)
        fig.text(x + card_w / 2, y + card_h * 0.65, value,
                 ha="center", va="center", fontsize=22, fontweight="bold",
                 color=color, transform=fig.transFigure, zorder=6)
        fig.text(x + card_w / 2, y + card_h * 0.22, label,
                 ha="center", va="center", fontsize=9, color=_TEXT_DIM,
                 transform=fig.transFigure, zorder=6)

    acc_val = "—"
    f1_val  = "—"
    if results["prompt_trues"]:
        acc_val = f"{accuracy_score(results['prompt_trues'], results['prompt_preds']):.1%}"
        f1_val = f"{f1_score(results['prompt_trues'], results['prompt_preds'], labels=labels_cls, average='macro'):.1%}"

    loss_val = f"{losses[-1]:.3f}" if losses else "—"
    steps_val = str(training_log.get("global_step", "—"))
    mem_val = f"{training_log.get('peak_memory_gb', '—')} GB"

    cards = [
        ("Accuracy", acc_val, _GREEN),
        ("F1 Macro", f1_val, _ACCENT),
        ("Final Loss", loss_val, _ORANGE),
        ("Total Steps", steps_val, _PURPLE),
        ("Peak Memory", mem_val, _CYAN),
    ]

    start_x = 0.05
    for i, (lbl, val, col) in enumerate(cards):
        _kpi_card(start_x + i * (card_w + gap), kpi_y, lbl, val, col)

    # Footnote
    fig.text(0.5, 0.04,
             f"GPU: {training_log.get('gpu_name', '?')}  |  "
             f"Runtime: {training_log.get('runtime_seconds', 0):.0f}s  |  "
             f"Eval samples: {len(results['prompt_trues'])}",
             ha="center", fontsize=9, color=_TEXT_DIM)

    fig.savefig(dirs["plots"] / "10_dashboard.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    dirs = setup_dirs(args.output_dir)
    start_time = time.time()

    # Save config
    config = vars(args)
    config["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(dirs["base"] / "training_args.json", "w") as f:
        json.dump(config, f, indent=2)

    # -----------------------------------------------------------------------
    # 1. Load model
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("1. Loading model: Qwen3.5-4B")
    print("=" * 60)

    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen3.5-4B",
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    # -----------------------------------------------------------------------
    # 2. Add LoRA adapters
    # -----------------------------------------------------------------------
    print("\n2. Adding LoRA adapters (r={}, alpha={})".format(args.lora_r, args.lora_r))

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # -----------------------------------------------------------------------
    # 3. Load & prepare dataset
    # -----------------------------------------------------------------------
    print("\n3. Loading dataset: nvidia/Aegis-AI-Content-Safety-Dataset-2.0")

    dataset = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")
    print(f"   Train:      {len(dataset['train'])} samples")
    print(f"   Validation: {len(dataset['validation'])} samples")
    print(f"   Test:       {len(dataset['test'])} samples")

    # Dataset stats
    prompt_labels = Counter(dataset["train"]["prompt_label"])
    all_cats = []
    for cats in dataset["train"]["violated_categories"]:
        if cats:
            all_cats.extend([c.strip() for c in cats.split(",")])
    cat_counts = Counter(all_cats)

    dataset_stats = {
        "train_size": len(dataset["train"]),
        "val_size": len(dataset["validation"]),
        "test_size": len(dataset["test"]),
        "prompt_label_distribution": dict(prompt_labels),
        "top_violated_categories": dict(cat_counts.most_common(12)),
    }
    with open(dirs["logs"] / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, ensure_ascii=False, indent=2)

    print(f"   Prompt labels: {dict(prompt_labels)}")
    print(f"   Top categories: {dict(cat_counts.most_common(5))}")

    # Filter and convert
    train_filtered = dataset["train"].filter(
        lambda x: x["prompt"] is not None and x["prompt"] != "REDACTED"
    )
    n_removed = len(dataset["train"]) - len(train_filtered)
    print(f"   After filtering: {len(train_filtered)} samples (removed {n_removed} redacted)")

    converted_dataset = [convert_to_conversation(sample) for sample in train_filtered]
    print(f"   Converted {len(converted_dataset)} training samples")

    # -----------------------------------------------------------------------
    # 4. Train
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. Training")
    print("=" * 60)

    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    FastVisionModel.for_training(model)

    sft_args = dict(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=5,
        learning_rate=args.lr,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=str(dirs["checkpoints"]),
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_length,
        save_steps=args.save_steps,
        save_total_limit=3,
    )

    if args.max_steps is not None:
        sft_args["max_steps"] = args.max_steps
    else:
        sft_args["num_train_epochs"] = args.epochs

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset,
        args=SFTConfig(**sft_args),
    )

    # GPU info
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"   GPU: {gpu_stats.name} | Max memory: {max_memory} GB | Reserved: {start_gpu_memory} GB")

    # Train
    trainer_stats = trainer.train()

    # Memory stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_for_lora = round(used_memory - start_gpu_memory, 3)
    train_runtime = trainer_stats.metrics["train_runtime"]

    print(f"\n   Training complete!")
    print(f"   Runtime: {train_runtime:.0f}s ({train_runtime / 60:.1f} min)")
    print(f"   Peak memory: {used_memory} GB (training: {used_for_lora} GB)")

    # Save training logs
    training_log = {
        "log_history": trainer.state.log_history,
        "runtime_seconds": train_runtime,
        "peak_memory_gb": used_memory,
        "training_memory_gb": used_for_lora,
        "gpu_name": gpu_stats.name,
        "gpu_memory_gb": max_memory,
        "global_step": trainer.state.global_step,
        "total_steps": trainer.state.max_steps,
    }
    with open(dirs["logs"] / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # Save training plots
    print("\n   Saving training plots...")
    save_training_loss_plot(trainer, dirs)
    print("   Saved: 01_training_loss.png")
    save_lr_schedule_plot(trainer, dirs)
    print("   Saved: 02_lr_schedule.png")

    # -----------------------------------------------------------------------
    # 5. Evaluate
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("5. Evaluation")
    print("=" * 60)

    results = evaluate(model, tokenizer, dataset["test"], args.eval_samples, dirs)

    # -----------------------------------------------------------------------
    # 6. Save plots
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("6. Saving plots")
    print("=" * 60)

    if results["prompt_trues"]:
        save_confusion_matrices(results, dirs)
        print("   Saved: 03_confusion_matrices.png")

        save_precision_recall_f1(results, dirs)
        print("   Saved: 04_precision_recall_f1.png")

        save_category_distribution(results, dirs)
        print("   Saved: 05_category_distribution.png")

        save_label_distribution_pie(results, dirs)
        print("   Saved: 06_label_distribution.png")

        save_category_f1_heatmap(results, dirs)
        print("   Saved: 07_category_f1_heatmap.png")

        save_radar_chart(results, dirs)
        print("   Saved: 08_radar_chart.png")

        save_error_analysis(results, dirs)
        print("   Saved: 09_error_analysis.png")

        save_dashboard(trainer, results, training_log, dirs)
        print("   Saved: 10_dashboard.png")
    else:
        print("   No predictions to plot.")

    # -----------------------------------------------------------------------
    # 7. Save model
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("7. Saving LoRA adapters")
    print("=" * 60)

    model.save_pretrained(str(dirs["model"]))
    tokenizer.save_pretrained(str(dirs["model"]))
    print(f"   Saved to {dirs['model']}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_time
    summary = {
        "total_time_seconds": round(elapsed, 1),
        "training_time_seconds": round(train_runtime, 1),
        "total_steps": trainer.state.global_step,
        "final_loss": [e["loss"] for e in trainer.state.log_history if "loss" in e][-1]
        if any("loss" in e for e in trainer.state.log_history) else None,
        "gpu": gpu_stats.name,
        "peak_memory_gb": used_memory,
    }
    if results["prompt_trues"]:
        summary["prompt_accuracy"] = round(accuracy_score(results["prompt_trues"], results["prompt_preds"]), 4)
        summary["prompt_f1_macro"] = round(
            f1_score(results["prompt_trues"], results["prompt_preds"], labels=["safe", "unsafe"], average="macro"), 4
        )
    if results["response_trues"]:
        summary["response_accuracy"] = round(accuracy_score(results["response_trues"], results["response_preds"]), 4)

    with open(dirs["base"] / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"\nAll outputs saved to: {dirs['base']}/")
    print(f"  checkpoints/          — model checkpoints")
    print(f"  logs/                 — training logs + dataset stats")
    print(f"  plots/                — all visualizations (PNG)")
    print(f"  metrics/              — eval metrics + predictions (JSON)")
    print(f"  qwen_aegis_safety_lora/ — final LoRA adapters")
    print(f"  training_args.json    — run config")
    print(f"  summary.json          — final summary")


if __name__ == "__main__":
    main()
