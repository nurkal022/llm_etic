"""
Report Generator — Diploma Analytics (Kazakh, Light Theme)
Generates all figures for the diploma thesis.

Usage:
  python report/generate_report.py

Outputs to report/figures/:
  01_dataset_overview.png         — деректер жиынына шолу
  02_safety_finetune_training.png — safety fine-tune оқыту
  03_safety_finetune_metrics.png  — safety fine-tune метрикалары
  04_qa_finetune_training.png     — QA fine-tune оқыту
  05_qorgau_overall.png           — жалпы қауіпсіздік деңгейі
  06_qorgau_heatmap.png           — heatmap
  07_qorgau_radar.png             — radar chart
  08_latency.png                  — кідіріс
  09_summary_table.png            — жиынтық кесте
  10_master_dashboard.png         — мастер-дашборд
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
# Light theme
# ---------------------------------------------------------------------------
_BG     = "#FFFFFF"
_CARD   = "#F6F8FA"
_GRID   = "#D0D7DE"
_TEXT   = "#1F2328"
_DIM    = "#57606A"
_ACCENT = "#0969DA"
_GREEN  = "#1A7F37"
_RED    = "#CF222E"
_ORANGE = "#BC4C00"
_PURPLE = "#6639BA"
_CYAN   = "#0969DA"
_PINK   = "#BF4B8A"

PALETTE = [_ACCENT, _GREEN, _ORANGE, _RED, _PURPLE, _CYAN, _PINK, "#953800"]

plt.rcParams.update({
    "figure.facecolor":  _BG,
    "axes.facecolor":    _CARD,
    "axes.edgecolor":    _GRID,
    "axes.labelcolor":   _TEXT,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   _TEXT,
    "axes.grid":         True,
    "grid.color":        _GRID,
    "grid.alpha":        0.8,
    "text.color":        _TEXT,
    "xtick.color":       _DIM,
    "ytick.color":       _DIM,
    "legend.facecolor":  _BG,
    "legend.edgecolor":  _GRID,
    "legend.labelcolor": _TEXT,
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.facecolor": _BG,
    "savefig.bbox":      "tight",
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Kazakh labels
# ---------------------------------------------------------------------------
METHOD_COLORS = {
    "baseline":        _RED,
    "prompt_eng":      _ORANGE,
    "rag":             _PURPLE,
    "qa_finetune":     _CYAN,
    "safety_finetune": _GREEN,
}
METHOD_LABELS = {
    "baseline":        "Базалық",
    "prompt_eng":      "Нұсқаулар инж.",
    "rag":             "RAG",
    "qa_finetune":     "QA бейімдеу",
    "safety_finetune": "Safety бейімдеу",
}
AREA_KAZ = {
    "Information Hazards":                                     "Ақпараттық қауіптер",
    "Malicious Uses":                                          "Зиянды пайдалану",
    "Misinformation Harms":                                    "Жалған ақпарат",
    "Discrimination, Exclusion, Toxicity, Hateful, Offensive": "Кемсітушілік/Улы сөз",
    "Sensitive in China":                                      "Сезімтал тақырыптар",
    "Human-Chatbot Interaction Harms":                         "Адам-чатбот зияны",
}


def kaz_area(name: str) -> str:
    return AREA_KAZ.get(name, name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(OUT_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  Сақталды: {name}")


def bar_labels(ax, bars, fmt="{:.1%}", color=None, offset=0.012):
    for bar in bars:
        v = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + offset,
            fmt.format(v) if "%" in fmt else fmt.format(v),
            ha="center", fontsize=9, fontweight="bold",
            color=color or bar.get_facecolor(),
        )


# ---------------------------------------------------------------------------
# 01 — Деректер жиынына шолу
# ---------------------------------------------------------------------------
def fig_dataset_overview():
    ds_stats = load_json(ROOT / "outputs/logs/dataset_stats.json")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Деректер жиынына шолу — Оқыту деректері", fontsize=16, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    # 1a. Aegis бөліктері
    ax = fig.add_subplot(gs[0, 0])
    splits = {"Оқыту": 30007, "Валидация": 1445, "Сынау": 1964}
    colors = [_ACCENT, _GREEN, _ORANGE]
    bars = ax.bar(splits.keys(), splits.values(), color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5, zorder=3)
    for bar, v in zip(bars, splits.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{v:,}", ha="center", fontsize=10, fontweight="bold", color=_TEXT)
    ax.set_title("Aegis 2.0 — Деректер бөліктері")
    ax.set_ylabel("Үлгілер саны")

    # 1b. Белгілер үлестірімі (доnut)
    ax = fig.add_subplot(gs[0, 1])
    vals_pie = [17711, 12296]
    wedges, texts, autotexts = ax.pie(
        vals_pie, labels=["Қауіпті", "Қауіпсіз"],
        autopct="%1.1f%%", colors=[_RED, _GREEN],
        startangle=90, wedgeprops=dict(edgecolor=_BG, linewidth=2),
        textprops=dict(color=_TEXT),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax.set_facecolor(_CARD)
    ax.set_title("Aegis 2.0 — Белгілер үлестірімі")

    # 1c. Зиян категориялары
    ax = fig.add_subplot(gs[0, 2])
    cats = list(ds_stats["top_violated_categories"].keys())[:8]
    cnts = [ds_stats["top_violated_categories"][c] for c in cats]
    short = [c.split("/")[0][:18] for c in cats]
    ax.barh(short[::-1], cnts[::-1], color=PALETTE[:len(cats)][::-1],
            alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_title("Зиян категориялары (Aegis)")
    ax.set_xlabel("Үлгілер саны")

    # 1d. Qorgau бенчмаркі
    ax = fig.add_subplot(gs[1, 0])
    qorgau_areas = {
        "Ақп. қауіптер": 15,
        "Зиянды пайдал.": 15,
        "Жалған ақпарат": 15,
        "Кемсітушілік":   15,
        "Сезімтал тақ.":  15,
        "Адам-чатбот":    15,
    }
    ax.bar(range(len(qorgau_areas)), list(qorgau_areas.values()),
           color=PALETTE[:6], alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(range(len(qorgau_areas)))
    ax.set_xticklabels(list(qorgau_areas.keys()), rotation=30, ha="right", fontsize=8)
    ax.set_title("Qorgau бенчмаркі\n(15 үлгі × 6 тәуекел аймағы)")
    ax.set_ylabel("Үлгілер саны")

    # 1e. QA бейімдеу safe жұптары
    ax = fig.add_subplot(gs[1, 1])
    pairs = {"Оқыту\n(safe жұптары)": 4346, "Сынау\n(safe жұптары)": 338}
    bars = ax.bar(pairs.keys(), pairs.values(), color=[_CYAN, _PINK],
                  alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
    for bar, v in zip(bars, pairs.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{v:,}", ha="center", fontsize=11, fontweight="bold", color=_TEXT)
    ax.set_title("QA бейімдеу — Safe жұптары\n(Aegis 2.0 негізінде)")
    ax.set_ylabel("Үлгілер саны")

    # 1f. Жиынтық кесте
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    rows = [
        ["Деректер жиыны", "Үлгілер", "Тіл", "Міндет"],
        ["Aegis 2.0",      "33,416",  "АҒ",  "Қауіпсіздік жіктеуі"],
        ["Aegis safe",     "4,684",   "АҒ",  "QA бейімдеу"],
        ["Qorgau",         "500",     "ОР/ҚЗ","Бенчмарк"],
    ]
    table = ax.table(
        cellText=rows[1:], colLabels=rows[0],
        cellLoc="center", loc="center", bbox=[0, 0.1, 1, 0.8],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor(_CARD if r > 0 else _GRID)
        cell.set_edgecolor(_GRID)
        cell.set_text_props(
            color=_ACCENT if r == 0 else _TEXT,
            fontweight="bold" if r == 0 else "normal",
        )
    ax.set_title("Деректер жиыны қорытындысы", pad=10)

    save(fig, "01_dataset_overview.png")


# ---------------------------------------------------------------------------
# 02 — Safety Fine-tune оқыту динамикасы
# ---------------------------------------------------------------------------
def fig_safety_training():
    raw = load_json(ROOT / "outputs/logs/training_log.json")
    log = raw["log_history"] if isinstance(raw, dict) else raw
    steps  = [e["step"] for e in log if "loss" in e]
    losses = [e["loss"] for e in log if "loss" in e]
    lrs    = [e.get("learning_rate", e.get("lr", 0)) for e in log if "loss" in e]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Safety бейімдеу оқытуы — Qwen3.5-4B + LoRA (Aegis 2.0)",
        fontsize=14, fontweight="bold",
    )

    ax = axes[0]
    ax.plot(steps, losses, color=_GRID, linewidth=0.6, alpha=0.5, label="Шығын")
    w = max(10, len(losses) // 40)
    smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
    ax.plot(steps[w-1:][:len(smooth)], smooth, color=_GREEN, linewidth=2.5,
            label=f"Тегістелген (w={w})")
    ax.axhline(losses[-1], color=_ORANGE, linestyle="--", linewidth=1.2,
               label=f"Соңғы: {losses[-1]:.4f}")
    ax.set_xlabel("Қадам")
    ax.set_ylabel("Кросс-энтропия шығыны")
    ax.set_title(f"Оқыту шығыны ({len(steps):,} қадам)")
    ax.legend()

    ax = axes[1]
    ax.plot(steps, lrs, color=_ACCENT, linewidth=1.8)
    ax.fill_between(steps, lrs, alpha=0.15, color=_ACCENT)
    ax.set_xlabel("Қадам")
    ax.set_ylabel("Оқыту жылдамдығы")
    ax.set_title("Оқыту жылдамдығы кестесі (Cosine Warmup)")

    fig.tight_layout()
    save(fig, "02_safety_finetune_training.png")


# ---------------------------------------------------------------------------
# 03 — Safety Fine-tune метрикалары
# ---------------------------------------------------------------------------
def fig_safety_metrics():
    metrics = load_json(ROOT / "outputs/metrics/eval_metrics.json")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "Safety бейімдеу бағалауы — Aegis сынау жиыны (500 үлгі)",
        fontsize=14, fontweight="bold",
    )

    # Дәлдік / F1
    ax = axes[0]
    metric_names = ["Дәлдік", "F1 Macro", "F1 Weighted"]
    prompt_vals  = [metrics["prompt"]["accuracy"],
                    metrics["prompt"]["f1_macro"],
                    metrics["prompt"]["f1_weighted"]]
    resp_vals    = [metrics["response"]["accuracy"],
                    metrics["response"]["f1_macro"],
                    metrics["response"]["f1_weighted"]]
    x = np.arange(len(metric_names))
    w = 0.35
    b1 = ax.bar(x - w/2, prompt_vals, w, label="Сұраным",  color=_ACCENT, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, resp_vals,   w, label="Жауап",    color=_GREEN,  alpha=0.85, edgecolor="white")
    for bar, v in list(zip(b1, prompt_vals)) + list(zip(b2, resp_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold", color=_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.12)
    ax.set_title("Дәлдік және F1 нәтижелері")
    ax.legend()

    # Шатасу матрицасы
    ax = axes[1]
    pr = metrics["prompt"]["classification_report"]
    matrix_data = np.array([
        [pr["safe"]["recall"],    1 - pr["safe"]["recall"]],
        [1 - pr["unsafe"]["recall"], pr["unsafe"]["recall"]],
    ])
    cmap = LinearSegmentedColormap.from_list("wg", ["#FFF3CD", _GREEN])
    im = ax.imshow(matrix_data, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix_data[i,j]:.1%}", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if matrix_data[i,j] > 0.7 else _TEXT)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Болжам: Қауіпсіз", "Болжам: Қауіпті"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Шын: Қауіпсіз", "Шын: Қауіпті"])
    ax.set_title("Қалыпқа келтірілген шатасу матрицасы\n(Сұраным деңгейі)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Мән")

    # Класс бойынша метрикалар
    ax = axes[2]
    classes   = ["Қауіпсіз", "Қауіпті"]
    f1_vals   = [pr["safe"]["f1-score"],    pr["unsafe"]["f1-score"]]
    prec_vals = [pr["safe"]["precision"],   pr["unsafe"]["precision"]]
    rec_vals  = [pr["safe"]["recall"],      pr["unsafe"]["recall"]]
    x = np.arange(len(classes))
    w = 0.25
    ax.bar(x - w, f1_vals,    w, label="F1",        color=_ACCENT, alpha=0.85, edgecolor="white")
    ax.bar(x,     prec_vals,  w, label="Нақтылық",  color=_GREEN,  alpha=0.85, edgecolor="white")
    ax.bar(x + w, rec_vals,   w, label="Толықтық",  color=_ORANGE, alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.12)
    ax.set_title("Класс бойынша метрикалар (Сұраным)")
    ax.legend()

    fig.tight_layout()
    save(fig, "03_safety_finetune_metrics.png")


# ---------------------------------------------------------------------------
# 04 — QA Fine-tune оқытуы
# ---------------------------------------------------------------------------
def fig_qa_training():
    log     = load_json(ROOT / "outputs_qa/logs/training_log.json")
    summary = load_json(ROOT / "outputs_qa/summary.json")
    steps  = [e["step"] for e in log if "loss" in e]
    losses = [e["loss"] for e in log if "loss" in e]
    lrs    = [e.get("lr", e.get("learning_rate", 0)) for e in log if "loss" in e]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "QA бейімдеу оқытуы — Qwen3.5-4B + LoRA (Aegis safe жұптары)",
        fontsize=14, fontweight="bold",
    )

    ax = axes[0]
    ax.plot(steps, losses, color=_GRID, linewidth=0.6, alpha=0.5, label="Шығын")
    w = max(5, len(losses) // 30)
    smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
    ax.plot(steps[w-1:][:len(smooth)], smooth, color=_CYAN, linewidth=2.5,
            label="Тегістелген")
    ax.axhline(losses[-1], color=_ORANGE, linestyle="--", linewidth=1.2,
               label=f"Соңғы: {losses[-1]:.4f}")
    ax.set_xlabel("Қадам")
    ax.set_ylabel("Шығын")
    ax.set_title(f"Оқыту шығыны ({len(steps)} қадам)")
    ax.legend()

    ax = axes[1]
    ax.plot(steps, lrs, color=_ORANGE, linewidth=1.8)
    ax.fill_between(steps, lrs, alpha=0.15, color=_ORANGE)
    ax.set_xlabel("Қадам")
    ax.set_ylabel("Оқыту жылдамдығы")
    ax.set_title("Оқыту жылдамдығы кестесі")

    ax = axes[2]
    ax.axis("off")
    stats = [
        ("Модель",            "Qwen3.5-4B + LoRA"),
        ("Деректер жиыны",    "Aegis 2.0 (safe жұптары)"),
        ("Оқыту үлгілері",    f"{summary['train_samples']:,}"),
        ("Эпохалар",          str(summary["epochs"])),
        ("Жалпы қадамдар",    str(summary["total_steps"])),
        ("Соңғы шығын",       f"{summary['final_loss']:.4f}"),
        ("Оқыту уақыты",      f"{summary['train_time_min']} мин"),
        ("Орт. генер. ұзын.", f"{summary['avg_gen_length']:.0f} таңба"),
        ("Орт. эталон ұзын.", f"{summary['avg_ref_length']:.0f} таңба"),
    ]
    ax.text(0.5, 1.0, "QA бейімдеу қорытындысы", ha="center", va="top",
            fontsize=12, fontweight="bold", color=_TEXT, transform=ax.transAxes)
    y = 0.88
    for label, val in stats:
        ax.text(0.05, y, label, fontsize=9, color=_DIM,  transform=ax.transAxes)
        ax.text(0.97, y, val,   fontsize=9, color=_CYAN, fontweight="bold",
                ha="right",   transform=ax.transAxes)
        y -= 0.10

    fig.tight_layout()
    save(fig, "04_qa_finetune_training.png")


# ---------------------------------------------------------------------------
# 05 — Qorgau жалпы қауіпсіздік деңгейі
# ---------------------------------------------------------------------------
def fig_qorgau_overall():
    evals   = load_json(ROOT / "experiments/qorgau_results/evaluation_summary.json")
    methods = sorted(evals, key=lambda m: evals[m]["safety_rate"], reverse=True)
    rates   = [evals[m]["safety_rate"] for m in methods]
    labels  = [METHOD_LABELS[m] for m in methods]
    colors  = [METHOD_COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, rates, color=colors, alpha=0.85, edgecolor="white",
                  linewidth=0.5, zorder=3, width=0.55)

    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.013,
                f"{v:.1%}", ha="center", fontsize=13, fontweight="bold",
                color=bar.get_facecolor())

    ax.axhline(0.8, color=_GREEN, linestyle="--", linewidth=1.5, alpha=0.7,
               label="Мақсатты шек: 80%")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Қауіпсіздік деңгейі (бас тарту үлесі)")
    ax.set_title(
        "Qorgau бенчмаркі — Әдіс бойынша қауіпсіздік деңгейі\n"
        "Qwen3.5-4B × 5 әдіс × 90 орысша зиянды сұраным",
        fontsize=13,
    )
    ax.legend(fontsize=10)

    baseline_rate = evals["baseline"]["safety_rate"]
    for bar, m, v in zip(bars, methods, rates):
        if m != "baseline" and v > baseline_rate:
            ax.annotate(
                f"+{v - baseline_rate:.1%}",
                xy=(bar.get_x() + bar.get_width() / 2, baseline_rate + 0.01),
                ha="center", fontsize=8, color=_DIM,
            )

    fig.tight_layout()
    save(fig, "05_qorgau_overall.png")


# ---------------------------------------------------------------------------
# 06 — Qorgau Heatmap
# ---------------------------------------------------------------------------
def fig_qorgau_heatmap():
    evals     = load_json(ROOT / "experiments/qorgau_results/evaluation_summary.json")
    methods   = sorted(evals, key=lambda m: evals[m]["safety_rate"], reverse=True)
    all_areas = sorted(set(a for ev in evals.values() for a in ev["by_risk_area"]))

    data = np.zeros((len(methods), len(all_areas)))
    for i, m in enumerate(methods):
        for j, a in enumerate(all_areas):
            data[i, j] = evals[m]["by_risk_area"].get(a, 0)

    cmap = LinearSegmentedColormap.from_list("wg", ["#FFEEF0", "#FFF3CD", "#DCFFE4"])
    fig, ax = plt.subplots(figsize=(15, 6))
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    kaz_areas = [kaz_area(a) for a in all_areas]

    for i in range(len(methods)):
        for j in range(len(all_areas)):
            v = data[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color=_GREEN if v >= 0.8 else _RED if v < 0.4 else _ORANGE)

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in methods], fontsize=11, color=_TEXT)
    ax.set_xticks(range(len(all_areas)))
    ax.set_xticklabels(kaz_areas, rotation=25, ha="right", fontsize=9, color=_TEXT)
    ax.set_title("Қауіпсіздік деңгейі: Әдіс × Тәуекел аймағы", fontsize=14, fontweight="bold")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Қауіпсіздік деңгейі", color=_TEXT)

    fig.tight_layout()
    save(fig, "06_qorgau_heatmap.png")


# ---------------------------------------------------------------------------
# 07 — Radar Chart
# ---------------------------------------------------------------------------
def fig_qorgau_radar():
    evals     = load_json(ROOT / "experiments/qorgau_results/evaluation_summary.json")
    all_areas = sorted(set(a for ev in evals.values() for a in ev["by_risk_area"]))
    kaz_areas = [kaz_area(a) for a in all_areas]

    n        = len(all_areas)
    angles   = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_c = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor(_CARD)
    fig.patch.set_facecolor(_BG)

    methods_sorted = sorted(evals, key=lambda m: evals[m]["safety_rate"], reverse=True)
    for m in methods_sorted:
        vals   = [evals[m]["by_risk_area"].get(a, 0) for a in all_areas]
        vals_c = vals + [vals[0]]
        color  = METHOD_COLORS[m]
        ax.plot(angles_c, vals_c, linewidth=2.5, color=color, label=METHOD_LABELS[m])
        ax.fill(angles_c, vals_c, alpha=0.08, color=color)

    ax.set_xticks(angles)
    ax.set_xticklabels(kaz_areas, fontsize=9, color=_TEXT)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8, color=_DIM)
    ax.spines["polar"].set_color(_GRID)
    ax.grid(color=_GRID, linewidth=0.7)
    ax.set_title(
        "Тәуекел аймағы бойынша қауіпсіздік деңгейі — Радар диаграммасы\n"
        "(жоғары = қауіпсізірек)",
        y=1.1, fontsize=13, fontweight="bold", color=_TEXT,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.18), fontsize=10)

    fig.tight_layout()
    save(fig, "07_qorgau_radar.png")


# ---------------------------------------------------------------------------
# 08 — Кідіріс талдауы
# ---------------------------------------------------------------------------
def fig_latency():
    evals   = load_json(ROOT / "experiments/qorgau_results/evaluation_summary.json")
    methods = sorted(evals, key=lambda m: evals[m]["avg_latency_s"])
    lats    = [evals[m]["avg_latency_s"] for m in methods]
    labels  = [METHOD_LABELS[m] for m in methods]
    colors  = [METHOD_COLORS[m] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Шығару кідірісін талдау", fontsize=14, fontweight="bold")

    ax = axes[0]
    bars = ax.bar(labels, lats, color=colors, alpha=0.85, edgecolor="white",
                  linewidth=0.5, zorder=3, width=0.5)
    for bar, v in zip(bars, lats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{v:.1f}с", ha="center", fontsize=11, fontweight="bold", color=_TEXT)
    ax.set_ylabel("Орт. кідіріс (секунд/үлгі)")
    ax.set_title("Әдіс бойынша орта кідіріс")

    ax = axes[1]
    for m, ev in evals.items():
        c = METHOD_COLORS[m]
        ax.scatter(ev["avg_latency_s"], ev["safety_rate"], s=250,
                   color=c, edgecolors=_GRID, linewidths=1.5, zorder=3)
        ax.annotate(METHOD_LABELS[m],
                    (ev["avg_latency_s"], ev["safety_rate"]),
                    xytext=(8, 5), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=c)
    ax.axhspan(0.8, 1.05, alpha=0.06, color=_GREEN)
    ax.text(ax.get_xlim()[1] * 0.02 if ax.get_xlim()[1] > 0 else 0.5,
            0.82, "ЖОҒАРЫ ҚАУІПСІЗДІК АЙМАҒЫ",
            fontsize=9, color=_GREEN, alpha=0.7, fontweight="bold")
    ax.set_xlabel("Кідіріс (с/үлгі)")
    ax.set_ylabel("Қауіпсіздік деңгейі")
    ax.set_title("Қауіпсіздік пен кідіріс арасындағы байланыс\n(сол жоғары = идеал)")

    fig.tight_layout()
    save(fig, "08_latency.png")


# ---------------------------------------------------------------------------
# 09 — Жиынтық кесте
# ---------------------------------------------------------------------------
def fig_summary_table():
    evals          = load_json(ROOT / "experiments/qorgau_results/evaluation_summary.json")
    summary_safety = load_json(ROOT / "outputs/summary.json")
    summary_qa     = load_json(ROOT / "outputs_qa/summary.json")

    fig, ax = plt.subplots(figsize=(15, 5))
    fig.suptitle("Эксперимент нәтижелерінің жиынтық кестесі", fontsize=14, fontweight="bold")
    ax.axis("off")

    methods_sorted = sorted(evals, key=lambda m: evals[m]["safety_rate"], reverse=True)

    headers = ["№", "Әдіс", "Қауіпсіздік\nдеңгейі", "Кідіріс\n(с/үлгі)",
               "Үлгілер\n(Qorgau)", "FT шығыны", "Ескертпе"]

    notes = {
        "rag":             "Этика БЖ-дан іздеу",
        "prompt_eng":      "Этикалық нұсқаулар",
        "baseline":        "Қауіпсіздік шарасы жоқ",
        "safety_finetune": "Жіктеуіш → бас тарту",
        "qa_finetune":     "Тек пайдалы жауаптар",
    }
    rows_data = []
    for i, m in enumerate(methods_sorted):
        ev       = evals[m]
        loss_str = {
            "safety_finetune": f"{summary_safety['final_loss']:.4f}",
            "qa_finetune":     f"{summary_qa['final_loss']:.4f}",
        }.get(m, "—")
        rows_data.append([
            str(i + 1),
            METHOD_LABELS[m],
            f"{ev['safety_rate']:.1%}",
            f"{ev['avg_latency_s']:.1f}",
            str(ev["n_samples"]),
            loss_str,
            notes.get(m, ""),
        ])

    table = ax.table(
        cellText=rows_data, colLabels=headers,
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(_GRID)
        if r == 0:
            cell.set_facecolor(_ACCENT)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            m_row = methods_sorted[r - 1]
            cell.set_facecolor(_BG if r % 2 == 0 else _CARD)
            if c == 2:
                rate = evals[m_row]["safety_rate"]
                cell.set_text_props(
                    color=_GREEN if rate >= 0.8 else _ORANGE if rate >= 0.5 else _RED,
                    fontweight="bold",
                )
            else:
                cell.set_text_props(color=_TEXT)

    save(fig, "09_summary_table.png")


# ---------------------------------------------------------------------------
# 10 — Мастер-дашборд
# ---------------------------------------------------------------------------
def fig_master_dashboard():
    evals   = load_json(ROOT / "experiments/qorgau_results/evaluation_summary.json")
    raw_s   = load_json(ROOT / "outputs/logs/training_log.json")
    log_s   = raw_s["log_history"] if isinstance(raw_s, dict) else raw_s
    log_qa  = load_json(ROOT / "outputs_qa/logs/training_log.json")

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(_BG)
    fig.suptitle(
        "Мастер-дашборд — Социология кеңесшісі: этика әдістері\n"
        "Qwen3.5-4B × 5 әдіс × Qorgau бенчмаркі",
        fontsize=16, fontweight="bold", color=_TEXT, y=0.98,
    )
    gs = GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38)

    methods_sorted = sorted(evals, key=lambda m: evals[m]["safety_rate"], reverse=True)
    rates  = [evals[m]["safety_rate"]  for m in methods_sorted]
    lats   = [evals[m]["avg_latency_s"] for m in methods_sorted]
    labels = [METHOD_LABELS[m]         for m in methods_sorted]
    colors = [METHOD_COLORS[m]         for m in methods_sorted]

    # 1. Қауіпсіздік деңгейі
    ax = fig.add_subplot(gs[0, :2])
    bars = ax.bar(labels, rates, color=colors, alpha=0.85, edgecolor="white",
                  linewidth=0.4, zorder=3)
    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{v:.1%}", ha="center", fontsize=11, fontweight="bold",
                color=bar.get_facecolor())
    ax.axhline(0.8, color=_GREEN, linestyle="--", linewidth=1.2, alpha=0.6,
               label="Шек: 80%")
    ax.set_ylim(0, 1.22)
    ax.set_title("Qorgau бенчмаркі бойынша қауіпсіздік деңгейі")
    ax.set_ylabel("Бас тарту үлесі")
    ax.legend()

    # 2. Safety FT шығыны
    ax = fig.add_subplot(gs[0, 2])
    steps_s  = [e["step"] for e in log_s if "loss" in e]
    losses_s = [e["loss"] for e in log_s if "loss" in e]
    w = max(10, len(losses_s) // 40)
    smooth = np.convolve(losses_s, np.ones(w) / w, mode="valid")
    ax.plot(steps_s[w-1:][:len(smooth)], smooth, color=_GREEN, linewidth=2)
    ax.set_title("Safety бейімдеу шығыны")
    ax.set_xlabel("Қадам")
    ax.set_ylabel("Шығын")

    # 3. QA FT шығыны
    ax = fig.add_subplot(gs[0, 3])
    steps_q  = [e["step"] for e in log_qa if "loss" in e]
    losses_q = [e["loss"] for e in log_qa if "loss" in e]
    w2 = max(5, len(losses_q) // 30)
    smooth2 = np.convolve(losses_q, np.ones(w2) / w2, mode="valid")
    ax.plot(steps_q[w2-1:][:len(smooth2)], smooth2, color=_CYAN, linewidth=2)
    ax.set_title("QA бейімдеу шығыны")
    ax.set_xlabel("Қадам")
    ax.set_ylabel("Шығын")

    # 4. Heatmap
    all_areas = sorted(set(a for ev in evals.values() for a in ev["by_risk_area"]))
    kaz_areas = [kaz_area(a) for a in all_areas]
    data = np.zeros((len(methods_sorted), len(all_areas)))
    for i, m in enumerate(methods_sorted):
        for j, a in enumerate(all_areas):
            data[i, j] = evals[m]["by_risk_area"].get(a, 0)

    ax = fig.add_subplot(gs[1, :3])
    cmap = LinearSegmentedColormap.from_list("wg", ["#FFEEF0", "#FFF3CD", "#DCFFE4"])
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    for i in range(len(methods_sorted)):
        for j in range(len(all_areas)):
            v = data[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    fontweight="bold",
                    color=_GREEN if v >= 0.8 else _RED if v < 0.4 else _ORANGE)
    ax.set_yticks(range(len(methods_sorted)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in methods_sorted], fontsize=9)
    ax.set_xticks(range(len(all_areas)))
    ax.set_xticklabels(kaz_areas, rotation=20, ha="right", fontsize=8)
    ax.set_title("Қауіпсіздік деңгейі: Әдіс × Тәуекел аймағы")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

    # 5. Негізгі деректер
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    ax.set_facecolor(_CARD)
    best_m = methods_sorted[0]
    key_stats = [
        ("Базалық модель",    "Qwen3.5-4B"),
        ("GPU",               "RTX 5080"),
        ("Safety оқыту",      "30,007 үлгі"),
        ("QA оқыту",          "4,346 үлгі"),
        ("Бенчмарк",          "Qorgau (орысша)"),
        ("Сынау сұранымдары", "90"),
        ("Үздік әдіс",        METHOD_LABELS[best_m]),
        ("Үздік нәтиже",      f"{evals[best_m]['safety_rate']:.1%}"),
        ("Safety F1",         "85.79%"),
        ("QA шығыны",         "0.8076"),
    ]
    y = 0.97
    for label, val in key_stats:
        ax.text(0.03, y, label, fontsize=8, color=_DIM,    transform=ax.transAxes)
        ax.text(0.97, y, val,   fontsize=8, color=_ACCENT, fontweight="bold",
                ha="right",   transform=ax.transAxes)
        y -= 0.095

    # 6. Scatter
    ax = fig.add_subplot(gs[2, :2])
    for m, ev in evals.items():
        c = METHOD_COLORS[m]
        ax.scatter(ev["avg_latency_s"], ev["safety_rate"], s=200,
                   color=c, edgecolors=_GRID, linewidths=1.2, zorder=3)
        ax.annotate(METHOD_LABELS[m], (ev["avg_latency_s"], ev["safety_rate"]),
                    xytext=(7, 4), textcoords="offset points", fontsize=8, color=c)
    ax.axhspan(0.8, 1.05, alpha=0.06, color=_GREEN)
    ax.set_xlabel("Кідіріс (с/үлгі)")
    ax.set_ylabel("Қауіпсіздік деңгейі")
    ax.set_title("Қауіпсіздік пен кідіріс байланысы")

    # 7. Кідіріс
    ax = fig.add_subplot(gs[2, 2:])
    bars = ax.bar(labels, lats, color=colors, alpha=0.85, edgecolor="white",
                  linewidth=0.4, zorder=3)
    for bar, v in zip(bars, lats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{v:.1f}с", ha="center", fontsize=9, fontweight="bold", color=_TEXT)
    ax.set_ylabel("Кідіріс (с/үлгі)")
    ax.set_title("Шығару кідірісі (әдіс бойынша)")
    ax.tick_params(axis="x", rotation=15)

    save(fig, "10_master_dashboard.png")


# ---------------------------------------------------------------------------
# Басты функция
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Дипломдық жұмыс — Аналитикалық графиктер генерациясы")
    print(f"  Жазылатын жер: {OUT_DIR}")
    print("=" * 60)
    print()

    fig_dataset_overview();      print()
    fig_safety_training();       print()
    fig_safety_metrics();        print()
    fig_qa_training();           print()
    fig_qorgau_overall();        print()
    fig_qorgau_heatmap();        print()
    fig_qorgau_radar();          print()
    fig_latency();               print()
    fig_summary_table();         print()
    fig_master_dashboard();      print()

    print("=" * 60)
    print(f"  Дайын! {len(list(OUT_DIR.glob('*.png')))} график сақталды → report/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
