"""Figures for the report, at CVPR camera-ready quality.

House style enforced here, and why each rule exists:

  * Vector PDF output, never raster. A 200-dpi PNG is visibly soft in print.
  * Type 42 (TrueType) font embedding -- the IEEE PDF check rejects Type 3,
    which is matplotlib's PDF default.
  * Serif text at the sizes the document actually uses (8pt tick / 9pt label
    against 10pt body), so figure text matches body text rather than appearing
    at a random scale after \\includegraphics rescaling.
  * Exact column geometry: 3.25in single column, 6.875in double, which are the
    CVPR \\linewidth and \\textwidth. Figures are drawn at final size and
    included at [width=\\linewidth] with no rescaling.
  * No chartjunk: no top/right spines, hairline grid behind the data, direct
    value labels rather than a legend wherever it fits.
  * Okabe-Ito colours: safe for common colour-vision deficiencies and separable
    in greyscale print.

All figures are built from the three task result files only, i.e. from the
configurations the assignment specifies.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C

FIG_DIR = C.HW_ROOT / "report_cvpr" / "fig"

COL_SINGLE = 3.25   # CVPR \linewidth in inches
COL_DOUBLE = 6.875  # CVPR \textwidth in inches

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GREY = "#7F7F7F"
LIGHT = "#94C5E0"
TASK_COLOR = {"Task 1": BLUE, "Task 2": GREEN, "Task 3": ORANGE}


def _register_cjk() -> list:
    """Make a serif CJK face available to matplotlib.

    The report is in Chinese, so axis labels contain CJK. The Noto CJK faces
    ship as .otf, which matplotlib's default font scan skips, so we register the
    file explicitly. A serif face is chosen deliberately: it matches the CVPR
    body text, whereas a sans fallback would look pasted in.
    """
    import matplotlib.font_manager as fm
    from pathlib import Path

    found = []
    for p in (
        "/usr/share/fonts/google-noto-cjk/NotoSerifCJKsc-Regular.otf",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJKsc-Regular.otf",
    ):
        if Path(p).exists():
            fm.fontManager.addfont(p)
            found.append(fm.FontProperties(fname=p).get_name())
    return found


_CJK = _register_cjk()

matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": _CJK + ["Nimbus Roman No9 L", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.labelcolor": "black",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})

NAMES = [
    ("bow_binary",  "Task 1", "Binary BoW"),
    ("bow_freq",    "Task 1", "Word Frequency"),
    ("bow_tfidf",   "Task 1", "TF-IDF"),
    ("embed_glove", "Task 2", "GloVe (100d)"),
    ("embed_ag",    "Task 2", "W2V-AG (100d)"),
    ("embed_nyt",   "Task 2", "W2V-NYT (100d)"),
    ("bert",        "Task 3", "BERT (len 64)"),
]


def polish(ax, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    (ax.yaxis if grid_axis == "y" else ax.xaxis).grid(
        True, color="#DDDDDD", linewidth=0.5
    )


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):  # pdf for the paper, png for quick viewing
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=400)
    plt.close(fig)
    C.log(f"wrote fig/{name}.pdf")


def load_rows() -> list:
    t1 = json.loads((C.RESULTS_DIR / "task1_bow.json").read_text())["results"]
    t2 = json.loads((C.RESULTS_DIR / "task2_word2vec.json").read_text())["results"]
    t3 = json.loads((C.RESULTS_DIR / "task3_bert.json").read_text())
    got = {}
    for k, v in t1.items():
        got[f"bow_{k}"] = v["test"]
    for k, v in t2.items():
        got[f"embed_{k}"] = v["test"]
    got["bert"] = t3["test"]
    return [
        {"key": k, "task": t, "name": n, **got[k]} for k, t, n in NAMES
    ]


# --------------------------------------------------------------------------- #
def fig_main(rows: list) -> None:
    """Double-column: Accuracy and Macro-F1 for all seven configurations."""
    names = [r["name"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    f1 = [r["macro_f1"] for r in rows]
    colors = [TASK_COLOR[r["task"]] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.6))
    for ax, vals, title in zip(axes, (acc, f1), ("Accuracy", "Macro-F1")):
        y = np.arange(len(rows))
        ax.barh(y, vals, 0.62, color=colors, edgecolor="none")
        ax.set_yticks(y)
        ax.set_yticklabels(names if ax is axes[0] else [])
        ax.invert_yaxis()
        lo = min(min(acc), min(f1))
        ax.set_xlim(max(0.93, lo - 0.015), 1.0)
        ax.set_title(title, fontsize=9)
        for yi, v in zip(y, vals):
            ax.text(v + 0.0012, yi, f"{v:.4f}", va="center", fontsize=6.6)
        polish(ax, "x")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TASK_COLOR.values()]
    fig.legend(handles, TASK_COLOR.keys(), loc="lower center", ncol=3,
               frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "main_comparison")


def fig_per_class(rows: list) -> None:
    """Double-column: per-class F1, where all the Macro-F1 variance lives."""
    labels = ["business", "politics", "sports"]
    x = np.arange(len(rows))
    w = 0.26
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 2.35))
    for i, (lab, col) in enumerate(zip(labels, (BLUE, ORANGE, GREEN))):
        ax.bar(x + (i - 1) * w, [r["per_class_f1"][lab] for r in rows], w,
               label=lab, color=col, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"].replace(" (", "\n(") for r in rows], fontsize=6.8)
    ax.set_ylim(0.90, 1.005)
    ax.set_ylabel("F1")
    ax.legend(frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, -0.46), handlelength=1.1)
    polish(ax)
    fig.tight_layout()
    save(fig, "per_class_f1")


def fig_truncation(analysis: dict, rows: list) -> None:
    """Single-column: the equal-budget control explaining BERT's ranking."""
    ctl = analysis["truncation_control"]
    bow_full = next(r["macro_f1"] for r in rows if r["key"] == "bow_binary")
    glove_full = next(r["macro_f1"] for r in rows if r["key"] == "embed_glove")
    bert = next(r["macro_f1"] for r in rows if r["key"] == "bert")

    labels = ["Binary\nBoW", "GloVe\navg", "BERT"]
    full = [bow_full, glove_full, np.nan]
    trunc = [ctl["bow_binary_trunc64"]["macro_f1"],
             ctl["glove_trunc64"]["macro_f1"], bert]

    x = np.arange(3)
    w = 0.36
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.15))
    ax.bar(x - w / 2, full, w, color=LIGHT, edgecolor="none", label="全文")
    ax.bar(x + w / 2, trunc, w, color=ORANGE, edgecolor="none", label="前 64 token")
    for xi, v in zip(x - w / 2, full):
        if not np.isnan(v):
            ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=6.4)
    for xi, v in zip(x + w / 2, trunc):
        ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=6.4)
    # BERT has no full-document counterpart (64 is the mandated setting).
    ax.text(x[2] - w / 2, 0.913, "n/a", ha="center", fontsize=6.0, color=GREY)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.91, 1.035)
    ax.set_yticks([0.92, 0.94, 0.96, 0.98, 1.00])
    ax.set_ylabel("Macro-F1")
    ax.legend(frameon=False, loc="upper center", ncol=2, handlelength=1.1,
              fontsize=6.6, columnspacing=1.0, borderaxespad=0.1)
    polish(ax)
    fig.tight_layout()
    save(fig, "truncation_control")


def fig_bert_curve() -> None:
    """Single-column: BERT loss vs validation Macro-F1 across the 3 epochs."""
    bert = json.loads((C.RESULTS_DIR / "task3_bert.json").read_text())
    hist = bert["history"]
    ep = [h["epoch"] for h in hist]

    fig, ax1 = plt.subplots(figsize=(COL_SINGLE, 2.15))
    ax1.plot(ep, [h["train_loss"] for h in hist], "o-", color=ORANGE,
             linewidth=1.2, markersize=3.4, label="训练损失")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("训练损失", color=ORANGE)
    ax1.tick_params(axis="y", labelcolor=ORANGE)
    ax1.set_xticks(ep)
    polish(ax1)

    ax2 = ax1.twinx()
    ax2.plot(ep, [h["val_macro_f1"] for h in hist], "s--", color=BLUE,
             linewidth=1.2, markersize=3.4, label="验证 Macro-F1")
    ax2.set_ylabel("验证集 Macro-F1", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax2.spines[["top"]].set_visible(False)

    fig.tight_layout()
    save(fig, "bert_training")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    analysis = json.loads((C.RESULTS_DIR / "analysis.json").read_text())
    fig_main(rows)
    fig_per_class(rows)
    fig_truncation(analysis, rows)
    fig_bert_curve()


if __name__ == "__main__":
    main()
