#!/usr/bin/env python3
r"""Regenerate the RSC table-of-contents graphical abstract.

The previous TOCScheme.pdf was Figure 1's entire six-panel workflow reused in an
8 cm box.  Its source (figs/TOCScheme.svg) is not editable vector text -- the
lettering lives inside five embedded raster panels -- and at the 0.44x scale the
8 cm include imposes on a 511 pt-wide canvas its glyph ink measures ~1.3 pt
median against the document's own 6.8 pt caption ink, i.e. roughly 2.5 pt
effective type.  It cannot be rescaled into legibility; six panels do not fit.

This is a purpose-built TOC entry instead: one arrow chain naming what IBMDP
consumes and produces, and one panel carrying the paper's headline result.  It
is drawn at exactly the width it is included at (8 cm), so \includegraphics
applies no scale and every font size below is a literal on-page point size.  The
smallest type is 6.0 pt.

The lower panel is the population median first-batch cost against terminal
uncertainty tolerance -- the same quantity as main Figure 3A.  It is read from
the RELEASED SURROGATE sweep so the graphic is reproducible from public files;
the surrogate reproduces the cohort's level-by-level medians exactly ($4,000 at
eps=0 and $800 at all ten looser levels), which are the values the abstract
quotes, so no proprietary number is introduced by drawing it this way.

Usage:
  python3 figures/toc_graphical_abstract.py \
      --results data/cns220_synthetic_ibmdp_results.csv \
      --out TOCScheme
"""
import argparse
import csv
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# TrueType (42), not matplotlib's default Type 3 -- Type 3 subsets propagate into
# the compiled PDF and are flagged by publisher preflight.
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["mathtext.default"] = "regular"
# Draw math in the same face as the body text.  Left at the default, mathtext
# falls back to DejaVu, so the italic "in vivo" and the epsilon would be set in a
# visibly different face from the Arial around them.
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "Arial"
rcParams["mathtext.it"] = "Arial:italic"
rcParams["mathtext.bf"] = "Arial:bold"
rcParams["axes.linewidth"] = 0.6
rcParams["xtick.major.size"] = 2.0
rcParams["ytick.major.size"] = 2.0
rcParams["xtick.major.width"] = 0.6
rcParams["ytick.major.width"] = 0.6

CM = 1 / 2.54
W_CM, H_CM = 8.0, 4.0            # the RSC TOC box; \includegraphics[width=8cm]

INK = "#1F2933"
ACCENT = "#8B1A1A"
PROXY = "#2E4057"


def level_medians(path):
    """Median first-batch cost per on-grid tolerance level.

    Grid membership is tested by formatting to one decimal rather than against a
    generated float range: the sweep writes back the threshold it was handed, so
    the CSV holds clean decimal strings, and an arange-based test misclassifies
    0.30000000000000004.
    """
    by = defaultdict(list)
    for r in csv.DictReader(open(path)):
        e = float(r["eps"])
        if float("%.1f" % e) == e:
            by[round(e, 1)].append(float(r["cost"]))
    xs = sorted(by)
    return xs, [statistics.median(by[e]) for e in xs], [len(by[e]) for e in xs]


def box(fig, x, y, w, h, text, fc, ec, fs=6.4, tc=INK):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
        transform=fig.transFigure, facecolor=fc, edgecolor=ec,
        linewidth=0.7, zorder=2))
    fig.text(x + w / 2, y + h / 2, text, transform=fig.transFigure,
             ha="center", va="center", fontsize=fs, color=tc,
             linespacing=1.25, zorder=3)


def arrow(fig, x0, x1, y):
    fig.patches.append(FancyArrowPatch(
        (x0, y), (x1, y), transform=fig.transFigure,
        arrowstyle="-|>,head_width=1.6,head_length=3.0",
        linewidth=0.9, color=INK, shrinkA=0, shrinkB=0, zorder=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    xs, meds, ns = level_medians(a.results)

    fig = plt.figure(figsize=(W_CM * CM, H_CM * CM))

    # ---- top row: what goes in, what comes out ------------------------------
    ytop, hbox = 0.795, 0.190
    box(fig, 0.020, ytop, 0.255, hbox,
        "Historical assay\noutcomes\n(no simulator)", "#EDF1F5", INK)
    arrow(fig, 0.283, 0.337, ytop + hbox / 2)
    box(fig, 0.345, ytop, 0.255, hbox,
        "Implicit Bayesian\nbelief over\nanalogs", "#DCE6EF", INK)
    arrow(fig, 0.608, 0.662, ytop + hbox / 2)
    box(fig, 0.670, ytop, 0.255, hbox,
        "Ensemble MCTS\nplans the next\nassay batch", "#F4E3E3", ACCENT,
        tc=ACCENT)

    # Closing the loop: the measured batch becomes new evidence.  Drawn as a
    # polyline plus one arrowhead rather than four FancyArrowPatches -- separate
    # patches gave visible joint gaps at these stroke widths.
    yloop, xloop = ytop - 0.055, 0.968
    fig.add_artist(Line2D([0.925, xloop, xloop, 0.1475],
                          [ytop + hbox / 2, ytop + hbox / 2, yloop, yloop],
                          transform=fig.transFigure, color=INK, linewidth=0.7,
                          solid_joinstyle="miter", zorder=2))
    fig.patches.append(FancyArrowPatch(
        (0.1475, yloop), (0.1475, ytop - 0.004), transform=fig.transFigure,
        arrowstyle="-|>,head_width=1.7,head_length=3.2",
        linewidth=0.7, color=INK, shrinkA=0, shrinkB=0, zorder=2))
    fig.text(0.56, yloop, "measured outcome updates the belief",
             ha="center", va="center", fontsize=6.0, color=INK, style="italic",
             zorder=3, bbox=dict(boxstyle="square,pad=0.18", facecolor="white",
                                 edgecolor="none"))

    # ---- lower panel: the headline result -----------------------------------
    ax = fig.add_axes((0.205, 0.205, 0.775, 0.435))
    # Points joined by straight segments, not a step: the sweep evaluates the
    # eleven grid levels only, and `where="post"` would assert that the $4,000
    # median holds across all of [0, 0.1).
    ax.plot(xs, meds, "-", color=PROXY, linewidth=1.0, zorder=3)
    ax.plot(xs[1:], meds[1:], "o", ms=2.6, color=PROXY, mec="white",
            mew=0.35, zorder=4)
    ax.plot([xs[0]], [meds[0]], "o", ms=4.4, color=ACCENT, mec="white",
            mew=0.5, zorder=5)

    # in vivo / in vitro italicised via mathtext, as RSC house style requires.
    ax.text(0.085, 4520, f"\\${meds[0]:,.0f}: $\\mathit{{in\\ vivo}}$ assay",
            fontsize=6.2, color=ACCENT, va="center", ha="left")
    ax.text(0.255, 1320,
            f"\\${meds[-1]:,.0f}: $\\mathit{{in\\ vitro}}$ proxies suffice",
            fontsize=6.2, color=PROXY, va="bottom", ha="left")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 5100)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 2000, 4000])
    ax.set_yticklabels(["0", "2,000", "4,000"])
    ax.tick_params(axis="both", labelsize=6.0, pad=1.5)
    ax.set_xlabel(r"terminal uncertainty tolerance $\epsilon$",
                  fontsize=6.5, labelpad=1.2)
    ax.set_ylabel("median first-batch\ncost (\\$)", fontsize=6.5,
                  labelpad=1.8, linespacing=1.15)
    ax.grid(axis="y", linestyle=":", linewidth=0.35, color="gray", alpha=0.55)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=600)
    print(f"wrote {a.out}.pdf / .png at {W_CM} x {H_CM} cm")
    print(f"  levels {xs}")
    print(f"  medians {meds}")
    print(f"  contributing compounds {ns}")
    print("  smallest type on the canvas: 6.0 pt (axis ticks, loop caption)")


if __name__ == "__main__":
    main()
