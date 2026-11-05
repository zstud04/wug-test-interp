#!/usr/bin/env python3
"""
Plot the PCA movement figures — a matplotlib port of analysis/movement.R
(R isn't installed on this machine), matching its ggplot theme_classic /
Times styling, sizes and file layouts. Reads pca_data/ (CSVs from
movement_pca.py, copied there by export_pca_data.py). The kept rebuttal results are image-only, so the two panels
are Qwen3-VL-2B and Qwen3-VL-4B (Vision) instead of Language / Vision.

Also prints/saves the per-number movement summary (mean, 95% CI, one-sided
t-test vs 0) that movement.R printed.
"""
import argparse
import pathlib
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy import stats

SG, PL = "#a6611a", "#018571"
PT = 72.27 / 25.4  # ggplot2's .pt: mm -> points

# Base-14 Times, exactly what R's pdf() device writes (Times-Roman / -Italic /
# -BoldItalic, unembedded) — so the PDFs match the original figures.
plt.rcParams.update({
    "pdf.use14corefonts": True,
    "font.family": "Times",
    "axes.unicode_minus": False,
    "font.size": 16,
    "axes.linewidth": 0.5 * PT * 0.75,
    "xtick.major.width": 0.5 * PT * 0.75,
    "ytick.major.width": 0.5 * PT * 0.75,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.labelsize": 12.8,
    "ytick.labelsize": 12.8,
    "xtick.color": "black",
    "ytick.color": "black",
    "savefig.transparent": True,
})

XLIM, YLIM = (-0.5, 0.5), (-0.4, 0.4)
XTICKS = [-0.5, -0.25, 0.0, 0.25, 0.5]
YTICKS = [-0.4, -0.2, 0.0, 0.2, 0.4]


class ArrowHandler(HandlerBase):
    """Legend key drawn as a short arrow, like ggplot's geom_segment key."""

    def create_artists(self, legend, orig, x0, y0, width, height, fontsize, trans):
        return [FancyArrowPatch((x0, y0 + height / 2), (x0 + width, y0 + height / 2),
                                arrowstyle="-|>", mutation_scale=7, lw=1.2,
                                color=orig.get_color(), transform=trans)]


def tick_label(v, decimals):
    # En dash stands in for R's minus glyph (base-14 WinAnsi has no U+2212).
    return f"{v:.{decimals}f}".replace("-", "\u2013")


def load(res_dir):
    real = pd.read_csv(res_dir / "sg_pl_reduced.csv")
    wug = pd.read_csv(res_dir / "wug_wugs_reduced_image.csv")
    wide = wug.pivot_table(index=["type", "pair_id"], columns="stage", values=["x", "y"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    move = pd.read_csv(res_dir / "wug_wugs_movement_image.csv")
    return real, wide, move


def draw_panel(ax, real, wide, title, annotate, annot_size):
    # Real nouns: ggplot default point size (1.5 mm), shapes 16/17, alpha 0.08.
    for lab, col, mk, size in [("sg", SG, "o", 1.5), ("pl", PL, "^", 1.75)]:
        r = real[real.label == lab]
        ax.scatter(r.x, r.y, c=col, marker=mk, s=(size * PT) ** 2,
                   alpha=0.08, linewidths=0, zorder=1)
    # Novel nouns: init -> final arrows (linewidth 0.6, alpha 0.6, 0.1 cm heads).
    for typ, col in [("wug", SG), ("wugs", PL)]:
        for _, row in wide[wide.type == typ].iterrows():
            ax.add_patch(FancyArrowPatch(
                (row.x_init, row.y_init), (row.x_final, row.y_final),
                arrowstyle="-|>", mutation_scale=5, lw=0.9,
                color=col, alpha=0.6, shrinkA=0, shrinkB=0, zorder=2))
    if annotate:
        ax.add_patch(FancyArrowPatch((0.2, 0.2), (0.28, 0.3), arrowstyle="-|>",
                                     mutation_scale=5, lw=1.2, color="black",
                                     shrinkA=0, shrinkB=0, zorder=3))
        ax.text(0.25, 0.165, "initial", fontstyle="italic", fontsize=annot_size * PT,
                ha="center", va="center")
        ax.text(0.335, 0.3, "final", fontstyle="italic", fontsize=annot_size * PT,
                ha="center", va="center")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks(XTICKS, [tick_label(v, 2) for v in XTICKS])
    ax.set_yticks(YTICKS, [tick_label(v, 1) for v in YTICKS])
    ax.set_title(title, fontweight="bold", fontstyle="italic", fontsize=12.8, pad=4)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def legends(fig, x_real, x_novel, y):
    real_h = [Line2D([], [], color=SG, marker="o", ls="", markersize=6, label="sg"),
              Line2D([], [], color=PL, marker="^", ls="", markersize=7, label="pl")]
    novel_h = [Line2D([], [], color=SG, label="sg"), Line2D([], [], color=PL, label="pl")]
    kw = dict(frameon=False, ncol=2, prop={"style": "italic", "size": 14}, title_fontsize=16,
              handlelength=1.2, handletextpad=0.4, columnspacing=0.9, borderaxespad=0,
              borderpad=0, loc="upper center")
    l1 = fig.legend(handles=real_h, title="Real Noun Type", bbox_to_anchor=(x_real, y), **kw)
    fig.legend(handles=novel_h, title="Novel Noun Type", bbox_to_anchor=(x_novel, y),
               handler_map={h: ArrowHandler() for h in novel_h}, **kw)
    fig.add_artist(l1)


def make_figure(data, models, layout, path):
    """Fixed panel geometry (in inches) mirroring the original ggsave sizes."""
    if layout == "legendtop":
        W, H = 6.8, 4.16
        left, right, bottom, top, gap = 0.62, 0.25, 0.62, 1.12, 0.62
        pw, ph = (W - left - right - gap) / 2, H - bottom - top
        boxes = [(left, bottom, pw, ph), (left + pw + gap, bottom, pw, ph)]
        annot_size, leg_x = 3.5, (0.33, 0.67)
    else:
        W, H = 4.22, 7.95
        left, right, bottom, top, gap = 0.62, 0.15, 0.62, 1.15, 0.6
        pw, ph = W - left - right, (H - bottom - top - gap) / 2
        boxes = [(left, bottom + ph + gap, pw, ph), (left, bottom, pw, ph)]
        annot_size, leg_x = 4.0, (0.27, 0.74)
    fig = plt.figure(figsize=(W, H))
    for i, ((x, y, w, h), (k, _)) in enumerate(zip(boxes, models)):
        ax = fig.add_axes((x / W, y / H, w / W, h / H))
        real, wide, _ = data[k]
        draw_panel(ax, real, wide, f"Qwen3-VL-{k}", annotate=(i == 0), annot_size=annot_size)
    # Shared axis titles, as with facet_wrap.
    fig.text((left + (W - left - right) / 2) / W, 0.12 / H, "PC1", ha="center", va="bottom", fontsize=16)
    fig.text(0.08 / W, (bottom + (H - bottom - top) / 2) / H, "PC2", ha="left", va="center",
             rotation=90, fontsize=16)
    legends(fig, *leg_x, y=1 - 0.06 / H)
    fig.savefig(path)
    plt.close(fig)


def summarize(move, name):
    rows = []
    for num in ["sg", "pl"]:
        m = move[move.number == num].movement.values
        n, mean, se = len(m), m.mean(), m.std(ddof=1) / np.sqrt(len(m))
        h = stats.t.ppf(0.975, n - 1) * se
        rows.append({"model": name, "number": num, "n": n, "mean_movement": mean,
                     "ci95_low": mean - h, "ci95_high": mean + h,
                     "t": mean / se, "p_one_sided": stats.t.sf(mean / se, n - 1)})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="eacl-rebuttal/pca_data",
                   help="output of scripts/export_pca_data.py")
    p.add_argument("--fig_dir", default="eacl-rebuttal/figures")
    a = p.parse_args()
    res, figs = pathlib.Path(a.results_dir), pathlib.Path(a.fig_dir)
    figs.mkdir(parents=True, exist_ok=True)

    models = [("2B", "Qwen_Qwen3-VL-2B-Instruct"), ("4B", "Qwen_Qwen3-VL-4B-Instruct")]
    data = {k: load(res / d) for k, d in models}

    outputs = [figs / "2b-4b-movement-pca-legendtop.pdf", figs / "2b-4b-movement-pca-vertical.pdf"]
    make_figure(data, models, "legendtop", outputs[0])
    make_figure(data, models, "vertical", outputs[1])
    # PNG previews rendered from the PDFs themselves, so they show the real fonts.
    for pdf in outputs:
        subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile", str(pdf),
                        str(pdf.with_suffix(""))], check=True)

    summary = pd.DataFrame([r for k, _ in models for r in summarize(data[k][2], k)])
    summary.to_csv(res / "movement_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
