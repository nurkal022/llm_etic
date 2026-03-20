"""
Report Generator — Diploma Analytics
Generates all figures and statistics for the diploma thesis.

Usage:
  python report/generate_report.py

Outputs to report/figures/:
  01_dataset_overview.png
  02_safety_finetune_training.png
  03_safety_finetune_metrics.png
  04_qa_finetune_training.png
  05_qorgau_overall.png
  06_qorgau_heatmap.png
  07_qorgau_radar.png
  08_qorgau_latency.png
  09_methods_summary_table.png
  10_master_dashboard.png
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
_BG     = "#0D1117"
_CARD   = "#161B22"
_GRID   = "#21262D"
_TEXT   = "#E6EDF3"
_DIM    = "#8B949E"
_ACCENT = "#58A6FF"
_GREEN  = "#3FB950"
_RED    = "#F85149"
_ORANGE = "#D29922"
_PURPLE = "#BC8CFF"
_CYAN   = "#39D2C0"
_PINK   = "#F778BA"

PALETTE = [_ACCENT, _GREEN, _ORANGE, _RED, _PURPLE, _CYAN, _PINK, "#FFA657"]

plt.rcParams.update({
    "figure.facecolor": _BG,   "axes.facecolor":   _CARD,
    "axes.edgecolor":   _GRID, "axes.labelcolor":  _TEXT,
    "axes.titlesize":   13,    "axes.titleweight": "bold",
    "axes.grid":        True,  "grid.color":       _GRID,   "grid.alpha": 0.6,
    "text.color":       _TEXT, "xtick.color":      _DIM,    "ytick.color": _DIM,
    "legend.facecolor": _CARD, "legend.edgecolor": _GRID,
    "figure.dpi":       150,   "savefig.dpi":      150,
    "savefig.facecolor":_BG,   "savefig.bbox":     "tight",
    "font.family":      "sans-serif", "font.size": 10,
})

METHOD_COLORS = {
    "baseline":        _RED,
    "prompt_eng":      _ORANGE,
    "rag":             _PURPLE,
    "qa_finetune":     _CYAN,
    "safety_finetune": _GREEN,
}
METHOD_LABELS = {
    "baseline":        "Baseline",
    "prompt_eng":      "Prompt Eng.",
    "rag":             "RAG",
    "qa_finetune":     "QA Fine-tune",
    "safety_finetune": "Safety Fine-tune",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(OUT_DIR / name)
    plt.close(fig)
    print(f"  Saved: {name}")


# ---------------------------------------------------------------------------
# 01 — Dataset Overview
# ---------------------------------------------------------------------------
def fig_dataset_overview():
    ds_stats = load_json(ROOT / "outputs/logs/dataset_stats.json")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Datasets Overview — Training Data", fontsize=16, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 1a. Aegis splits
    ax = fig.add_subplot(gs[0, 0])
    splits = {"Train": 30007, "Val": 1445, "Test": 1964}
    bars = ax.bar(splits.keys(), splits.values(), color=[_ACCENT, _GREEN, _ORANGE],
                  alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
    for bar, v in zip(bars, splits.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{v:,}", ha="center", fontsize=10, fontweight="bold", color=_TEXT)
    ax.set_title("Aegis 2.0 — Data Splits")
    ax.set_ylabel("Samples")

    # 1b. Prompt label distribution
    ax = fig.add_subplot(gs[0, 1])
    labels_pie = ["Unsafe", "Safe"]
    vals_pie   = [17711, 12296]
    wedges, texts, autotexts = ax.pie(
        vals_pie, labels=labels_pie, autopct="%1.1f%%",
        colors=[_RED, _GREEN], startangle=90,
        wedgeprops=dict(edgecolor=_BG, linewidth=2),
        textprops=dict(color=_TEXT)
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax.set_facecolor(_CARD)
    ax.set_title("Aegis 2.0 — Label Distribution")

    # 1c. Top violated categories
    ax = fig.add_subplot(gs[0, 2])
    cats = list(ds_stats["top_violated_categories"].keys())[:8]
    cnts = [ds_stats["top_violated_categories"][c] for c in cats]
    short_cats = [c.split("/")[0][:18] for c in cats]
    colors_bar = PALETTE[:len(cats)]
    bars = ax.barh(short_cats[::-1], cnts[::-1], color=colors_bar[::-1],
                   alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_title("Top Harm Categories (Aegis)")
    ax.set_xlabel("Count")

    # 1d. Qorgau dataset
    ax = fig.add_subplot(gs[1, 0])
    qorgau_areas = {
        "Information Hazards": 15,
        "Malicious Uses": 15,
        "Misinformation": 15,
        "Discrimination": 15,
        "Sensitive Topics": 15,
        "Human-Chatbot": 15,
    }
    ax.bar(range(len(qorgau_areas)), list(qorgau_areas.values()),
           color=PALETTE[:6], alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(range(len(qorgau_areas)))
    ax.set_xticklabels([k[:10] for k in qorgau_areas.keys()], rotation=30, ha="right", fontsize=8)
    ax.set_title("Qorgau — Benchmark Distribution\n(15 samples × 6 risk areas)")
    ax.set_ylabel("Samples")

    # 1e. Aegis safe pairs (for QA fine-tune)
    ax = fig.add_subplot(gs[1, 1])
    pairs = {"Train\n(safe pairs)": 4346, "Test\n(safe pairs)": 338}
    bars = ax.bar(pairs.keys(), pairs.values(), color=[_CYAN, _PINK],
                  alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
    for bar, v in zip(bars, pairs.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{v:,}", ha="center", fontsize=11, fontweight="bold", color=_TEXT)
    ax.set_title("QA Fine-tune — Safe Pairs\n(from Aegis 2.0)")
    ax.set_ylabel("Samples")

    # 1f. Summary table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.set_facecolor(_CARD)
    rows = [
        ["Dataset", "Samples", "Lang", "Task"],
        ["Aegis 2.0", "33,416", "EN", "Safety classif."],
        ["Aegis safe", "4,684", "EN", "QA fine-tune"],
        ["Qorgau", "500", "RU/KZ", "Benchmark"],
    ]
    table = ax.table(
        cellText=rows[1:], colLabels=rows[0],
        cellLoc="center", loc="center",
        bbox=[0, 0.1, 1, 0.8]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor(_CARD if r > 0 else _GRID)
        cell.set_edgecolor(_GRID)
        cell.set_text_props(color=_TEXT if r > 0 else _ACCENT, fontweight="bold" if r == 0 else "normal")
    ax.set_title("Dataset Summary", pad=10)

    save(fig, "01_dataset_overview.png")


# ---------------------------------------------------------------------------
# 02 — Safety Fine-tune Training Loss
# ---------------------------------------------------------------------------
def fig_safety_training():
    raw = load_json(ROOT / "outputs/logs/training_log.json")
    # log may be dict with log_history key or a plain list
    log = raw["log_history"] if isinstance(raw, dict) else raw
    steps  = [e["step"] for e in log if "loss" in e]
    losses = [e["loss"] for e in log if "loss" in e]
    lrs    = [e.get("learning_rate", e.get("lr", 0)) for e in log if "loss" in e]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Safety Fine-tune Training — Qwen3.5-4B + LoRA on Aegis 2.0",
                 fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(steps, losses, color=_ACCENT, linewidth=0.8, alpha=0.35, label="Loss")
    w = max(10, len(losses) // 40)
    smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
    ax.plot(steps[w-1:][:len(smooth)], smooth, color=_GREEN, linewidth=2.5, label=f"Smoothed (w={w})")
    ax.axhline(losses[-1], color=_ORANGE, linestyle="--", linewidth=1, alpha=0.7,
               label=f"Final: {losses[-1]:.4f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(f"Training Loss ({len(steps):,} steps)")
    ax.legend()

    # LR schedule
    ax = axes[1]
    ax.plot(steps, lrs, color=_ORANGE, linewidth=1.8)
    ax.fill_between(steps, lrs, alpha=0.12, color=_ORANGE)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (Cosine Warmup)")

    fig.tight_layout()
    save(fig, "02_safety_finetune_training.png")


# ---------------------------------------------------------------------------
# 03 — Safety Fine-tune Evaluation Metrics
# ---------------------------------------------------------------------------
def fig_safety_metrics():
    metrics = load_json(ROOT / "outputs/metrics/eval_metrics.json")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Safety Fine-tune Evaluation — Aegis Test Set (500 samples)",
                 fontsize=14, fontweight="bold")

    # Accuracy / F1 bars
    ax = axes[0]
    metric_names = ["Accuracy", "F1 Macro", "F1 Weighted"]
    prompt_vals  = [metrics["prompt"]["accuracy"],
                    metrics["prompt"]["f1_macro"],
                    metrics["prompt"]["f1_weighted"]]
    resp_vals    = [metrics["response"]["accuracy"],
                    metrics["response"]["f1_macro"],
                    metrics["response"]["f1_weighted"]]

    x = np.arange(len(metric_names))
    w = 0.35
    b1 = ax.bar(x - w/2, prompt_vals, w, label="Prompt",   color=_ACCENT, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, resp_vals,   w, label="Response", color=_GREEN,  alpha=0.85, edgecolor="white")
    for bar, v in list(zip(b1, prompt_vals)) + list(zip(b2, resp_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold", color=_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.1)
    ax.set_title("Accuracy & F1 Scores")
    ax.legend()

    # Prompt-level confusion matrix approximation
    ax = axes[1]
    pr = metrics["prompt"]["classification_report"]
    safe_p, safe_r = pr["safe"]["precision"], pr["safe"]["recall"]
    unsafe_p, unsafe_r = pr["unsafe"]["precision"], pr["unsafe"]["recall"]

    matrix_data = np.array([
        [safe_r,    1-safe_r],
        [1-unsafe_r, unsafe_r]
    ])
    cmap = LinearSegmentedColormap.from_list("rb", [_RED, _ORANGE, _GREEN])
    im = ax.imshow(matrix_data, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix_data[i,j]:.2%}", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if matrix_data[i,j] < 0.6 else _BG)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred: Safe", "Pred: Unsafe"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True: Safe", "True: Unsafe"])
    ax.set_title("Normalized Confusion Matrix\n(Prompt-level)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Per-class F1 breakdown
    ax = axes[2]
    classes  = ["Safe", "Unsafe"]
    f1_vals  = [pr["safe"]["f1-score"], pr["unsafe"]["f1-score"]]
    prec_vals = [pr["safe"]["precision"], pr["unsafe"]["precision"]]
    rec_vals  = [pr["safe"]["recall"], pr["unsafe"]["recall"]]

    x = np.arange(len(classes))
    w = 0.25
    ax.bar(x - w, f1_vals,    w, label="F1",        color=_ACCENT, alpha=0.85, edgecolor="white")
    ax.bar(x,     prec_vals,  w, label="Precision",  color=_GREEN,  alpha=0.85, edgecolor="white")
    ax.bar(x + w, rec_vals,   w, label="Recall",     color=_ORANGE, alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-class Metrics (Prompt-level)")
    ax.legend()

    fig.tight_layout()
    save(fig, "03_safety_finetune_metrics.png")


# ---------------------------------------------------------------------------
# 04 — QA Fine-tune Training
# ---------------------------------------------------------------------------
def fig_qa_training():
    log     = load_json(ROOT / "outputs_qa/logs/training_log.json")
    summary = load_json(ROOT / "outputs_qa/summary.json")
    steps  = [e["step"] for e in log if "loss" in e]
    losses = [e["loss"] for e in log if "loss" in e]
    lrs    = [e.get("lr", 0) for e in log if "loss" in e]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("QA Fine-tune Training — Qwen3.5-4B + LoRA on Aegis Safe Pairs",
                 fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(steps, losses, color=_CYAN, linewidth=0.8, alpha=0.35, label="Loss")
    w = max(5, len(losses) // 30)
    smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
    ax.plot(steps[w-1:][:len(smooth)], smooth, color=_GREEN, linewidth=2.5, label=f"Smoothed")
    ax.axhline(losses[-1], color=_ORANGE, linestyle="--", linewidth=1, alpha=0.7,
               label=f"Final: {losses[-1]:.4f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss ({len(steps)} steps)")
    ax.legend()

    # LR
    ax = axes[1]
    ax.plot(steps, lrs, color=_ORANGE, linewidth=1.8)
    ax.fill_between(steps, lrs, alpha=0.12, color=_ORANGE)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")

    # Summary stats
    ax = axes[2]
    ax.axis("off")
    ax.set_facecolor(_CARD)
    stats = [
        ("Model",          "Qwen3.5-4B + LoRA"),
        ("Dataset",        "Aegis 2.0 safe pairs"),
        ("Train samples",  f"{summary['train_samples']:,}"),
        ("Epochs",         str(summary["epochs"])),
        ("Total steps",    str(summary["total_steps"])),
        ("Final loss",     f"{summary['final_loss']:.4f}"),
        ("Train time",     f"{summary['train_time_min']} min"),
        ("Avg gen length", f"{summary['avg_gen_length']:.0f} chars"),
        ("Avg ref length", f"{summary['avg_ref_length']:.0f} chars"),
    ]
    ax.text(0.5, 1.0, "QA Fine-tune Summary", ha="center", va="top",
            fontsize=12, fontweight="bold", color=_TEXT, transform=ax.transAxes)
    y = 0.88
    for label, val in stats:
        ax.text(0.05, y, label, fontsize=9,  color=_DIM,   transform=ax.transAxes)
        ax.text(0.97, y, val,   fontsize=9,  color=_CYAN,  fontweight="bold",
                ha="right", transform=ax.transAxes)
        y -= 0.1

    fig.tight_layout()
    save(fig, "04_qa_finetune_training.png")


# ---------------------------------------------------------------------------
# 05 — Qorgau Overall Safety Rate
# ---------------------------------------------------------------------------
def fig_qorgau_overall():
    evals = load_json(ROOT / "experiments/qorgau_results/evaluation_summary.json")
    methods = sorted(evals, key=lambda m: evals[m]["safety_rate"], reverse=True)
    rates  = [evals[m]["safety_rate"] for m in methods]
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, rates, color=colors, alpha=0.85, edgecolor="white",
                  linewidth=0.5, zorder=3, width=0.55)

    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{v:.1%}", ha="center", fontsize=13, fontweight="bold",
                color=bar.get_facecolor())

    ax.axhline(0.8, color=_GREEN, linestyle="--", linewidth=1.5, alpha=0.6,
               label="Target threshold: 80%")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Safety Rate (% refused)")
    ax.set_title(
        "Qorgau Benchmark — Safety Rate by Method\n"
        "Qwen3.5-4B × 5 Methods × 90 Russian Harmful Prompts",
        fontsize=13
    )
    ax.legend(fontsize=10)

    # Improvement annotations
    baseline_rate = evals["baseline"]["safety_rate"]
    for bar, m, v in zip(bars, methods, rates):
        if m != "baseline":
            delta = v - baseline_rate
            if delta > 0:
                ax.annotate(
                    f"+{delta:.1%}",
                    xy=(bar.get_x() + bar.get_width() / 2, baseline_rate),
                    xytext=(bar.get_x() + bar.get_width() / 2, v / 2),
                    ha="center", fontsize=8, color=_DIM,
                    arrowprops=dict(arrowstyle="-", color=_DIM, alpha=0.4, lw=0.8)
                )

    fig.tight_layout()
    save(fig, "05_qorgau_overall.png")


# ---------------------------------------------------------------------------
# 06 — Qorgau Heatmap
# ---------------------------------------------------------------------------
def fig_qorgau_heatmap():
    evals = load_json(ROOT / "experiments/qorgau_results/evaluation_summary.json")
    methods   = sorted(evals, key=lambda m: evals[m]["safety_rate"], reverse=True)
    all_areas = sorted(set(a for ev in evals.values() for a in ev["by_risk_area"]))

    data = np.zeros((len(methods), len(all_areas)))
    for i, m in enumerate(methods):
        for j, a in enumerate(all_areas):
            data[i, j] = evals[m]["by_risk_area"].get(a, 0)

    cmap = LinearSegmentedColormap.from_list("rg", [_RED, _ORANGE, _GREEN])
    fig, ax = plt.subplots(figsize=(15, 6))
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    short_areas = [a.replace("Discrimination, Exclusion, Toxicity, Hateful, Offensive", "Discrimination/Toxicity")
                   .replace("Human-Chatbot Interaction Harms", "Human-Chatbot Harms")
                   .replace("Sensitive in China", "Sensitive Topics")
                   for a in all_areas]

    for i in range(len(methods)):
        for j in range(len(all_areas)):
            v = data[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if v < 0.6 else _BG,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=_BG)])

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in methods], fontsize=11)
    ax.set_xticks(range(len(all_areas)))
    ax.set_xticklabels(short_areas, rotation=25, ha="right", fontsize=9)
    ax.set_title("Qorgau Safety Rate: Method × Risk Area", fontsize=14, fontweight="bold")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Safety Rate", color=_TEXT)
    cb.outline.set_edgecolor(_GRID)

    fig.tight_layout()
    save(fig, "06_qorgau_heatmap.png")


# ---------------------------------------------------------------------------
# 07 — Radar Chart
# ---------------------------------------------------------------------------
def fig_qorgau_radar():
    evals     = load_json(ROOT / "experiments/qorgau_results/evaluation_summary.json")
    all_areas = sorted(set(a for ev in evals.values() for a in ev["by_risk_area"]))
    short_areas = [
        a.replace("Discrimination, Exclusion, Toxicity, Hateful, Offensive", "Discrimination")
         .replace("Human-Chatbot Interaction Harms", "Human-Chatbot")
         .replace("Sensitive in China", "Sensitive")
         .replace("Misinformation Harms", "Misinformation")
         .replace("Malicious Uses", "Malicious")
         .replace("Information Hazards", "Info Hazards")
        for a in all_areas
    ]
    n = len(all_areas)
    angles   = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_c = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor(_CARD)

    methods_sorted = sorted(evals, key=lambda m: evals[m]["safety_rate"], reverse=True)
    for m in methods_sorted:
        vals   = [evals[m]["by_risk_area"].get(a, 0) for a in all_areas]
        vals_c = vals + [vals[0]]
        color  = METHOD_COLORS[m]
        ax.plot(angles_c, vals_c, linewidth=2.5, color=color, label=METHOD_LABELS[m])
        ax.fill(angles_c, vals_c, alpha=0.07, color=color)

    ax.set_xticks(angles)
    ax.set_xticklabels(short_areas, fontsize=10, color=_TEXT)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8, color=_DIM)
    ax.spines["polar"].set_color(_GRID)
    ax.grid(color=_GRID, linewidth=0.5)
    ax.set_title("Safety Rate by Risk Area — Radar Chart\n(each axis = one risk category)",
                 y=1.1, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.18), fontsize=10, framealpha=0.8)

    fig.tight_layout()
    save(fig, "07_qorgau_radar.png")


# ---------------------------------------------------------------------------
# 08 — Latency Comparison
# ---------------------------------------------------------------------------
def fig_latency():
    evals   = load_json(ROOT / "experiments/qorgau_results/evaluation_summary.json")
    methods = sorted(evals, key=lambda m: evals[m]["avg_latency_s"])
    lats    = [evals[m]["avg_latency_s"] for m in methods]
    labels  = [METHOD_LABELS[m] for m in methods]
    colors  = [METHOD_COLORS[m] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Inference Latency Analysis", fontsize=14, fontweight="bold")

    # Latency bars
    ax = axes[0]
    bars = ax.bar(labels, lats, color=colors, alpha=0.85, edgecolor="white",
                  linewidth=0.5, zorder=3, width=0.5)
    for bar, v in zip(bars, lats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{v:.1f}s", ha="center", fontsize=11, fontweight="bold", color=_TEXT)
    ax.set_ylabel("Avg Latency per Sample (seconds)")
    ax.set_title("Avg Latency by Method")

    # Safety vs Latency scatter
    ax = axes[1]
    for m, ev in evals.items():
        c = METHOD_COLORS[m]
        ax.scatter(ev["avg_latency_s"], ev["safety_rate"], s=250,
                   color=c, edgecolors="white", linewidths=1.5, zorder=3)
        ax.annotate(METHOD_LABELS[m],
                    (ev["avg_latency_s"], ev["safety_rate"]),
                    xytext=(8, 5), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=c)
    ax.axhspan(0.8, 1.05, alpha=0.06, color=_GREEN)
    ax.set_xlabel("Avg Latency (s/sample)")
    ax.set_ylabel("Safety Rate")
    ax.set_title("Safety vs Latency Trade-off\n(upper-left = ideal)")

    fig.tight_layout()
    save(fig, "08_latency.png")


# ---------------------------------------------------------------------------
# 09 — Summary Table Figure
# ---------------------------------------------------------------------------
def fig_summary_table():
    evals   = load_json(ROOT / "experiments/qorgau_results/evaluation_summary.json")
    summary_safety = load_json(ROOT / "outputs/summary.json")
    summary_qa     = load_json(ROOT / "outputs_qa/summary.json")

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Experiment Results Summary Table", fontsize=14, fontweight="bold")
    ax.axis("off")
    ax.set_facecolor(_CARD)

    methods_sorted = sorted(evals, key=lambda m: evals[m]["safety_rate"], reverse=True)

    headers = ["#", "Метод", "Safety Rate\n(Qorgau)", "Latency\n(s/sample)",
               "Samples\n(Qorgau)", "Fine-tune\nLoss", "Примечание"]

    rows_data = []
    for i, m in enumerate(methods_sorted):
        ev = evals[m]
        notes = {
            "rag":             "Retrieval from ethics KB",
            "prompt_eng":      "Ethical system prompt",
            "baseline":        "No safety measures",
            "safety_finetune": "Classifier → refusal",
            "qa_finetune":     "Helpful responses only",
        }
        loss_str = {
            "safety_finetune": f"{summary_safety['final_loss']:.4f}",
            "qa_finetune":     f"{summary_qa['final_loss']:.4f}",
        }.get(m, "—")

        rows_data.append([
            str(i + 1),
            METHOD_LABELS[m],
            f"{ev['safety_rate']:.1%}",
            f"{ev['avg_latency_s']:.1f}s",
            str(ev["n_samples"]),
            loss_str,
            notes.get(m, ""),
        ])

    table = ax.table(
        cellText=rows_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    row_colors = [_RED, _ORANGE, "#2A2A2A", "#1A1A2E", "#1A1A2E"]
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(_GRID)
            cell.set_text_props(color=_ACCENT, fontweight="bold")
        else:
            m_of_row = methods_sorted[r - 1]
            cell.set_facecolor(_CARD)
            if c == 2:  # safety rate column
                cell.set_text_props(
                    color=_GREEN if evals[m_of_row]["safety_rate"] >= 0.8 else
                    _ORANGE if evals[m_of_row]["safety_rate"] >= 0.5 else _RED,
                    fontweight="bold"
                )
            else:
                cell.set_text_props(color=_TEXT)
        cell.set_edgecolor(_GRID)

    save(fig, "09_summary_table.png")


# ---------------------------------------------------------------------------
# 10 — Master Dashboard
# ---------------------------------------------------------------------------
def fig_master_dashboard():
    evals   = load_json(ROOT / "experiments/qorgau_results/evaluation_summary.json")
    raw_s   = load_json(ROOT / "outputs/logs/training_log.json")
    log_s   = raw_s["log_history"] if isinstance(raw_s, dict) else raw_s
    log_qa  = load_json(ROOT / "outputs_qa/logs/training_log.json")

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(_BG)
    fig.suptitle(
        "Master Dashboard — Цифровой консультант по социологии: методы этики\n"
        "Qwen3.5-4B × 5 методов × Qorgau benchmark",
        fontsize=16, fontweight="bold", y=0.98
    )
    gs = GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38)

    methods_sorted = sorted(evals, key=lambda m: evals[m]["safety_rate"], reverse=True)
    rates  = [evals[m]["safety_rate"] for m in methods_sorted]
    labels = [METHOD_LABELS[m] for m in methods_sorted]
    colors = [METHOD_COLORS[m] for m in methods_sorted]

    # Panel 1: Safety bars
    ax = fig.add_subplot(gs[0, :2])
    bars = ax.bar(labels, rates, color=colors, alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{v:.1%}", ha="center", fontsize=11, fontweight="bold", color=bar.get_facecolor())
    ax.axhline(0.8, color=_GREEN, linestyle="--", linewidth=1.2, alpha=0.5)
    ax.set_ylim(0, 1.2)
    ax.set_title("Safety Rate — Qorgau Benchmark")
    ax.set_ylabel("Refusal rate")

    # Panel 2: Safety Fine-tune loss
    ax = fig.add_subplot(gs[0, 2])
    steps_s = [e["step"] for e in log_s if "loss" in e]
    losses_s = [e["loss"] for e in log_s if "loss" in e]
    w = max(10, len(losses_s) // 40)
    smooth = np.convolve(losses_s, np.ones(w) / w, mode="valid")
    ax.plot(steps_s[w-1:][:len(smooth)], smooth, color=_GREEN, linewidth=2)
    ax.set_title("Safety Fine-tune Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")

    # Panel 3: QA Fine-tune loss
    ax = fig.add_subplot(gs[0, 3])
    steps_q  = [e["step"] for e in log_qa if "loss" in e]
    losses_q = [e["loss"] for e in log_qa if "loss" in e]
    w2 = max(5, len(losses_q) // 30)
    smooth2 = np.convolve(losses_q, np.ones(w2) / w2, mode="valid")
    ax.plot(steps_q[w2-1:][:len(smooth2)], smooth2, color=_CYAN, linewidth=2)
    ax.set_title("QA Fine-tune Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")

    # Panel 4: Heatmap (bottom 2 rows)
    all_areas = sorted(set(a for ev in evals.values() for a in ev["by_risk_area"]))
    short = [a.replace("Discrimination, Exclusion, Toxicity, Hateful, Offensive", "Discrimination")
              .replace("Human-Chatbot Interaction Harms", "Human-Chatbot")
              .replace("Sensitive in China", "Sensitive")
              .replace("Misinformation Harms", "Misinformation")
              for a in all_areas]

    data = np.zeros((len(methods_sorted), len(all_areas)))
    for i, m in enumerate(methods_sorted):
        for j, a in enumerate(all_areas):
            data[i, j] = evals[m]["by_risk_area"].get(a, 0)

    ax = fig.add_subplot(gs[1, :3])
    cmap = LinearSegmentedColormap.from_list("rg", [_RED, _ORANGE, _GREEN])
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    for i in range(len(methods_sorted)):
        for j in range(len(all_areas)):
            v = data[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white" if v < 0.6 else _BG)
    ax.set_yticks(range(len(methods_sorted)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in methods_sorted], fontsize=9)
    ax.set_xticks(range(len(all_areas)))
    ax.set_xticklabels(short, rotation=20, ha="right", fontsize=8)
    ax.set_title("Safety Rate: Method × Risk Area")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02).outline.set_edgecolor(_GRID)

    # Panel 5: Key stats
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    ax.set_facecolor(_CARD)
    best_m = methods_sorted[0]
    key_stats = [
        ("Base model",     "Qwen3.5-4B"),
        ("GPU",            "RTX 5080"),
        ("Safety train",   "30,007 samples"),
        ("QA train",       "4,346 samples"),
        ("Benchmark",      "Qorgau (RU)"),
        ("Test prompts",   "90"),
        ("Best method",    METHOD_LABELS[best_m]),
        ("Best safety",    f"{evals[best_m]['safety_rate']:.1%}"),
        ("Safety F1",      "85.79%"),
        ("QA loss",        "0.8076"),
    ]
    y = 0.97
    for label, val in key_stats:
        ax.text(0.03, y, label, fontsize=8,  color=_DIM,    transform=ax.transAxes)
        ax.text(0.97, y, val,   fontsize=8,  color=_ACCENT, fontweight="bold",
                ha="right", transform=ax.transAxes)
        y -= 0.095

    # Panel 6: Scatter safety vs latency
    ax = fig.add_subplot(gs[2, :2])
    for m, ev in evals.items():
        c = METHOD_COLORS[m]
        ax.scatter(ev["avg_latency_s"], ev["safety_rate"], s=200,
                   color=c, edgecolors="white", linewidths=1.2, zorder=3)
        ax.annotate(METHOD_LABELS[m], (ev["avg_latency_s"], ev["safety_rate"]),
                    xytext=(7, 4), textcoords="offset points", fontsize=8, color=c)
    ax.axhspan(0.8, 1.05, alpha=0.06, color=_GREEN)
    ax.set_xlabel("Latency (s/sample)")
    ax.set_ylabel("Safety Rate")
    ax.set_title("Safety vs Latency Trade-off")

    # Panel 7: Latency bars
    ax = fig.add_subplot(gs[2, 2:])
    lats = [evals[m]["avg_latency_s"] for m in methods_sorted]
    bars = ax.bar(labels, lats, color=colors, alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
    for bar, v in zip(bars, lats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{v:.1f}s", ha="center", fontsize=9, fontweight="bold", color=_TEXT)
    ax.set_ylabel("Latency (s/sample)")
    ax.set_title("Inference Latency by Method")
    ax.tick_params(axis="x", rotation=15)

    save(fig, "10_master_dashboard.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Diploma Report — Generating Analytics Figures")
    print(f"  Output: {OUT_DIR}")
    print("=" * 60)
    print()

    fig_dataset_overview();       print()
    fig_safety_training();        print()
    fig_safety_metrics();         print()
    fig_qa_training();            print()
    fig_qorgau_overall();         print()
    fig_qorgau_heatmap();         print()
    fig_qorgau_radar();           print()
    fig_latency();                print()
    fig_summary_table();          print()
    fig_master_dashboard();       print()

    print("=" * 60)
    print(f"  Done! {len(list(OUT_DIR.glob('*.png')))} figures saved to report/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
