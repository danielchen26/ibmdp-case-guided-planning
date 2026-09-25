#!/usr/bin/env python3
r"""Regenerate ESI Figure S1: the ensemble vote histogram at one planning state.

The published version of this figure (figs/figure3_improved_v2.png, produced by
IBMDP_arxiv/figs/recreate_figure3_corrected.py) carried hand-entered bar heights.
Its own source comments record how they were obtained -- "Data from the original
figure (extracted from visual inspection)" giving 40/5/25/25/5, which sums to 100,
then halved and rounded to 20/3/12/12/3 and relabelled as "counts out of 50".  No
ensemble-result CSV on this machine contains that multiset (130 files with
Action_Set + Frequency columns were checked), so those bar heights were not
traceable to a run.

This script instead reads an ensemble_results_tau_*.csv directly and plots the
vote counts it contains.  The default state is compound 3 at epsilon = 0, whose
plurality winner is the same MLASP point that Figure 2 annotates and the main text
quotes (objective 36,378 at 27/50 votes), so the ESI figure and the main figure now
show the same event from two sides.

Action sets use the short codes defined in the Figure 2 caption.

The figure is emitted at exactly the width it occupies on the page
(0.8 * textwidth = 375.8 pt = 5.22 in), so no scaling happens at \includegraphics
time and the font sizes below are literal on-page points.

Usage:
  python3 esi_figS1_votes.py --csv <ensemble_results_tau_0.9.csv> [--eps 0.0]
                            [--kpuu 0.541] [--label "compound 3"] --out <basename>

The published figure is this repository's shipped copy of that run's ensemble output:
  --csv figures/data/fig2_panel3_ensemble_tau0.9.csv \
        --eps 0.0 --kpuu 0.541 --label "compound 3" --tau 0.9
which prints "winner $P_1\,k$ at 27/50, objective 36377.6" and second place
$P_2\,k$ at 11/50 across 7 distinct sets -- the numbers the ESI caption quotes
and the objective annotated in panel 3 of main Figure 2.  The CSV is named for the
panel it feeds rather than for the run directory it came from, because the planner's
own output directories are named after the compound's measured k_puu at full
precision and that is proprietary data (see the repository README's data policy);
the CSV contents are planner output only -- tolerance, action set, objective,
vote count -- and carry no measured compound property.

Two ensembles were run at this state and they are NOT interchangeable: the one
shipped here is the N_e=50 run the manuscript reports (27/50, objective 36377.6).
A second run of the same state exists in the authors' working tree and gives
29/50, objective 35622.7.  Figure S1, the panel-3 annotation of main Figure 2 and
the ESI caption all quote the shipped run.
"""
import argparse
import csv
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

# Sans-serif, matching the other two ESI figures.
# Emit TrueType (42) rather than matplotlib's default Type 3 (3) fonts.
# Type 3 subsets propagate into the compiled ESI PDF and are flagged by
# publisher preflight; the glyphs drawn are identical either way.
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["mathtext.default"] = "regular"
rcParams["axes.linewidth"] = 0.6
rcParams["xtick.major.size"] = 2.5
rcParams["ytick.major.size"] = 2.5
rcParams["xtick.major.width"] = 0.6
rcParams["ytick.major.width"] = 0.6
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"

# The figure is placed at 0.8\textwidth with \textwidth = 469.755 pt.
WIDTH_IN = 0.8 * 469.755 / 72.0

CODE: "dict[str, str]" = {
    "100nM_PgP": "P_1",
    "1uM_PgP": "P_2",
    "100nM_BCRP": "B_1",
    "1uM_BCRP": "B_2",
    "kpuu": "k",
    "mrt": "M",
}


ORDER = list(CODE)   # proxies first, then kpuu -- the order Figure 2's labels use


def short_code(action_set):
    """Render one CSV Action_Set field as the paper's short code.

    A first-batch action set is unordered (the assays are run together), and the
    CSV lists its members in whatever order the planner emitted them, so codes are
    sorted into a fixed assay order.  That order is the one Figure 2's own labels
    already happen to carry, so the same set reads the same way in both figures.
    """
    toks = [t.strip() for t in re.sub(r'[\[\]"]', "", action_set).split(",")]
    toks = [t for t in toks if t]
    if not toks:
        return r"$\emptyset$"
    toks.sort(key=lambda t: ORDER.index(t) if t in ORDER else len(ORDER))
    return "$" + r"\,".join(CODE.get(t, t) for t in toks) + "$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--kpuu", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--tau", default="0.9")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.csv))
            if abs(float(r["Threshold"]) - a.eps) < 1e-9]
    if not rows:
        raise SystemExit(f"no rows at Threshold={a.eps} in {a.csv}")
    rows.sort(key=lambda r: -float(r["Frequency"]))

    counts = [int(float(r["Frequency"])) for r in rows]
    labels = [short_code(r["Action_Set"]) for r in rows]
    sizes = [len([t for t in re.sub(r'[\[\]"]', "", r["Action_Set"]).split(",")
                  if t.strip()]) for r in rows]
    total = sum(counts)
    util_win = float(rows[0]["Average_Utility"])

    # Winner in red; the rest shaded by how many assays the set contains.
    shade = {1: "#2E4057", 2: "#5C7FA6", 3: "#8BA8C4"}
    colors = ["#8B1A1A"] + [shade.get(s, "#B8C5D6") for s in sizes[1:]]

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 0.62 * WIDTH_IN),
                           constrained_layout=True)
    xs = range(len(counts))
    ax.bar(xs, counts, color=colors, edgecolor="black", linewidth=0.5, width=0.68)

    for x, c in zip(xs, counts):
        ax.text(x, c + max(counts) * 0.022, str(c), ha="center", va="bottom",
                fontsize=7, fontweight="bold" if c == counts[0] else "normal")

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Action set (short codes as in Fig. 2)", fontsize=8)
    ax.set_ylabel(f"Ensemble vote count (of ${total}$)", fontsize=8)
    head = f"Ensemble vote distribution at $\\epsilon = {a.eps:g}$"
    if a.label:
        head += f", {a.label}"
        if a.kpuu:
            head += f" ($k_{{puu}} = {a.kpuu}$)"
    ax.set_title(head, fontsize=8, pad=6)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.set_ylim(0, max(counts) * 1.30)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="gray", alpha=0.6)
    ax.set_axisbelow(True)

    handles = [Patch(facecolor="#8B1A1A", edgecolor="black", linewidth=0.5,
                     label="MLASP (plurality winner)")]
    for n, name in ((1, "one assay"), (2, "two assays"), (3, "three assays")):
        if n in sizes[1:]:
            handles.append(Patch(facecolor=shade[n], edgecolor="black",
                                 linewidth=0.5, label=name))
    ax.legend(handles=handles, fontsize=6.5, loc="upper right", frameon=True,
              framealpha=1.0, edgecolor="gray", borderpad=0.4,
              handlelength=1.2, handleheight=0.8, labelspacing=0.3)

    ax.text(0.015, 0.965,
            f"$N_e = {total}$ ensemble runs\n"
            f"goal-likelihood floor $\\tau = {a.tau}$",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", linewidth=0.5))

    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=300)
    print(f"wrote {a.out}.pdf / .png   width={WIDTH_IN:.3f} in")
    print(f"  {len(counts)} distinct action sets, votes sum to {total}")
    print(f"  winner {labels[0]} at {counts[0]}/{total}, objective {util_win:.1f}")
    for l, c in zip(labels, counts):
        print(f"     {c:3d}  {l}")


if __name__ == "__main__":
    main()
