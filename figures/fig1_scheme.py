#!/usr/bin/env python3
r"""Regenerate main Figure 1, the IBMDP workflow schematic.

WHY THIS REPLACES figs/IBMDP_scheme_hires.png
---------------------------------------------
The published Figure 1 is a 1827 x 602 px flattened raster included at
0.9\linewidth = 422.78 pt, i.e. 0.2314 pt of page per pixel.  Measured on that
raster (row-band ink measurement, figfix/inkmeasure.py's method applied to the
PNG): the lettering runs 4.86 pt from ascender to descender and "Cost" measures
3.47 pt of capital ink, i.e. about 5 pt of effective type; the subscripts of
$s_0$, $a_n$, $s_{1n}$ and $\Delta S(a, D_s)$ measure 2.1-2.6 pt.  Body text in
this manuscript is 10 pt.  Two- to three-point subscripts do not survive print.

The size cannot be fixed by rescaling.  The figure is a 3.03:1 strip with six
stations in its upper row and four in its lower one, so at the page width the
per-station budget is ~70 pt and the type has to be ~5 pt to fit.  Widening the
include to the full \linewidth buys 11%.  Splitting the strip into two stacked
halves would buy 2.2x but drops the raster to 140 effective dpi.  And the art
cannot be re-exported larger: its only source, figs/TOCScheme.svg, has zero
<text> elements and holds every letter inside five embedded PNGs no wider than
715 px (see extract_art.py), so there is no vector text to re-set.

This script therefore re-lays-out the same workflow as four panels on a 2 x 2
clockwise cycle, which raises the per-station budget from ~70 pt to ~215 pt.  All
lettering is re-set as vector Arial; the four pieces of pure line art (molecule,
petri dish, mouse, monkey) are the author's own, reused verbatim from the SVG at
their published physical sizes, so their effective resolution is unchanged from
the figure this replaces (400-1000 dpi).  Only the data-table panel is redrawn
rather than reused, because its lettering is raster.

FONT SIZES ARE LITERAL
----------------------
The canvas is 469.755 x 322 pt and the figure is included at width=\linewidth =
469.755 pt, so \includegraphics applies scale 1.0 and every size below is an
on-page point size -- no GR-style inflation factor and no include reduction to
correct for (contrast replot_fig2.jl, where both apply).  Sizes: 8.0 pt panel
titles, 6.8 pt labels, 6.6 pt connective lines, 9.0 pt for any string carrying a
subscript.  matplotlib sets subscripts at 0.7x the base, so 9.0 pt is the
smallest base at which a subscript still clears 6 pt: 6.3 pt.  The smallest type
anywhere on the canvas is therefore 6.3 pt, against the published figure's ~2.5.

Panel (d) is schematic -- the fronts are drawn from fixed vertices, not from run
output.  The measured ensemble fronts are main Figure 2 and the caption says so.
No number in this figure is a result.

Usage:
  python3 figures/fig1_scheme.py --art figures/art --out IBMDP_scheme
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

# TrueType (42), not matplotlib's default Type 3 -- Type 3 subsets propagate into
# the compiled PDF and are flagged by publisher preflight.
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
# Draw math in the body face.  At the default, mathtext falls back to DejaVu and
# every subscripted symbol would be set in a visibly different face.
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "Arial"
rcParams["mathtext.it"] = "Arial:italic"
rcParams["mathtext.bf"] = "Arial:bold"

W_PT, H_PT = 469.755, 322.0          # canvas == \linewidth, so scale is exactly 1
MARG, GUTX, GUTY = 3.0, 34.0, 26.0   # gutters carry the four cycle arrows
PW = (W_PT - 2 * MARG - GUTX) / 2
PH = (H_PT - 2 * MARG - GUTY) / 2
PX = (MARG, MARG + PW + GUTX)                 # left edges of the two columns
PY = (MARG + PH + GUTY, MARG)                 # bottom edges: row 0 top, row 1 bottom

INK, ACCENT, PROXY = "#1F2933", "#8B1A1A", "#2E4057"
FRAME, FILL = "#B8C2CC", "#F7F9FB"
FS_TITLE, FS_LABEL, FS_CONN, FS_TIER = 8.0, 6.8, 6.6, 7.5
FS_MATH = 9.0    # only where a string carries a SUBSCRIPT: 0.7 x 9.0 = 6.3 pt
FS_SYM = 7.2     # math without a subscript, e.g. a bare epsilon, needs no headroom


def fx(p):
    return p / W_PT


def fy(p):
    return p / H_PT


# ---- primitives, all in page points ---------------------------------------------
def panel(fig, col, row, title):
    x, y = PX[col], PY[row]
    fig.patches.append(FancyBboxPatch(
        (fx(x), fy(y)), fx(PW), fy(PH),
        boxstyle="round,pad=0,rounding_size=0.008",
        transform=fig.transFigure, facecolor=FILL, edgecolor=FRAME,
        linewidth=0.6, zorder=1))
    fig.text(fx(x + 8), fy(y + PH - 11), title, ha="left", va="center",
             fontsize=FS_TITLE, color=INK, weight="bold", zorder=4)
    return x, y


def connective(x, y, s):
    """The one-line consequence printed along a panel's bottom edge."""
    fig.text(fx(x + PW / 2), fy(y + 8.5), s, ha="center", va="center",
             fontsize=FS_CONN, color=PROXY, style="italic", zorder=4)


def label(x, y, s, fs=FS_LABEL, ha="center", va="center", color=INK, **kw):
    return fig.text(fx(x), fy(y), s, ha=ha, va=va, fontsize=fs, color=color,
                    zorder=4, **kw)


def box(x, y, w, h, s, fs=FS_MATH, fc="white", ec=INK, tc=INK, lw=0.7):
    """Rounded box centred on (x, y), width w and height h in points."""
    fig.patches.append(FancyBboxPatch(
        (fx(x - w / 2), fy(y - h / 2)), fx(w), fy(h),
        boxstyle="round,pad=0,rounding_size=0.006",
        transform=fig.transFigure, facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=3))
    label(x, y, s, fs=fs, color=tc)


def arrow(x0, y0, x1, y1, lw=1.0, color=INK, hw=2.2, hl=4.0, z=2):
    fig.patches.append(FancyArrowPatch(
        (fx(x0), fy(y0)), (fx(x1), fy(y1)), transform=fig.transFigure,
        arrowstyle=f"-|>,head_width={hw},head_length={hl}",
        linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=z))


def seg(xs, ys, lw=0.6, color=INK, ls="-", z=2, alpha=1.0):
    fig.add_artist(Line2D([fx(v) for v in xs], [fy(v) for v in ys],
                          transform=fig.transFigure, color=color, linewidth=lw,
                          linestyle=ls, alpha=alpha, solid_joinstyle="miter",
                          zorder=z))


def dots(cx, cy, n=3, gap=4.0, r=0.6, color=PROXY):
    """A vertical ellipsis, drawn rather than typeset: Arial has no U+22EE."""
    for k in range(n):
        fig.patches.append(Ellipse((fx(cx), fy(cy - k * gap)),
                                   width=fx(2 * r), height=fy(2 * r),
                                   transform=fig.transFigure, facecolor=color,
                                   edgecolor="none", zorder=3))


def art(path, cx, cy, w):
    """Place a line-art PNG centred on (cx, cy) with width w points.

    The author's panels are black line art on an OPAQUE white ground, which would
    print as a white patch inside the panel's tinted fill.  They are therefore
    re-keyed to black-plus-alpha: alpha = 1 - max(R,G,B), so a black stroke stays
    opaque, the ground becomes transparent, and antialiased edge pixels keep their
    partial coverage instead of turning into a halo.

    interpolation="none" so the vector backend embeds the PNG unresampled: the
    printer scales it, and the native resolution the author drew at is preserved.
    """
    src = plt.imread(path)
    ih, iw = src.shape[:2]
    cov = 1.0 - src[..., :3].max(axis=-1)
    if src.shape[2] == 4:
        cov = cov * src[..., 3]
    img = np.dstack([np.zeros_like(cov)] * 3 + [cov])
    h = w * ih / iw
    ax = fig.add_axes((fx(cx - w / 2), fy(cy - h / 2), fx(w), fy(h)), zorder=3)
    ax.imshow(img, interpolation="none")
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    return h, 72.0 * iw / w          # (height in pt, effective dpi on the page)


# ---- the four panels -------------------------------------------------------------
def panel_a(artdir):
    x, y = panel(fig, 0, 0, "(a)  One candidate, a menu of assays")
    dpis = {}
    mh, dpis["molecule"] = art(os.path.join(artdir, "molecule.png"),
                               x + 48, y + PH / 2 + 8, 76)
    label(x + 48, y + PH / 2 + 8 - mh / 2 - 8, "drug candidate", color=PROXY)

    # The dollar tiers are ordinal, not the study's price schedule: the paper
    # prices in vitro proxies and the in vivo k_puu assay, and does not price a
    # primate assay at all.  The caption says the tiers are illustrative.
    rows = (("petri.png", 24, "$\\mathit{in\\ vitro}$ proxies", "\\$"),
            ("mouse.png", 30, "rodent $\\mathit{in\\ vivo}$", "\\$\\$\\$\\$"),
            ("monkey.png", 24, "primate", "\\$\\$\\$\\$\\$"))
    ytop = y + PH - 30
    for i, (fn, w, name, tier) in enumerate(rows):
        cy = ytop - i * 32
        _, dpis[fn[:-4]] = art(os.path.join(artdir, fn), x + 112, cy, w)
        label(x + 132, cy + 5, name, ha="left")
        label(x + 132, cy - 5, tier, fs=FS_TIER, ha="left", color=ACCENT)
    connective(x, y, "Which assay next — or stop?")
    return dpis


def panel_b():
    x, y = panel(fig, 1, 0, "(b)  Implicit belief over historical analogs")
    # the record table, redrawn as vector: the SVG's version is a raster whose
    # own lettering is what this figure exists to enlarge.
    cw = (26.0, 15.0, 74.0, 13.0)
    rh = (16.0, 13.0, 16.0, 13.0)
    x0, ytop = x + 10, y + PH - 34
    xs = [x0]
    for w in cw:
        xs.append(xs[-1] + w)
    ys = [ytop]
    for h in rh:
        ys.append(ys[-1] - h)
    for gx in xs:
        seg([gx, gx], [ys[0], ys[-1]], lw=0.5, color="#8A96A3", z=3)
    for gy in ys:
        seg([xs[0], xs[-1]], [gy, gy], lw=0.5, color="#8A96A3", z=3)
    mid = lambda a, b: (a + b) / 2
    label(mid(xs[2], xs[3]), mid(ys[0], ys[1]), "features", color=INK)
    label(mid(xs[1], xs[2]), mid(ys[0], ys[1]), "…", color=PROXY)
    label(mid(xs[3], xs[4]), mid(ys[0], ys[1]), "…", color=PROXY)
    label(mid(xs[0], xs[1]), mid(ys[1], ys[2]), "…", color=PROXY)
    label(mid(xs[0], xs[1]), mid(ys[2], ys[3]), "records", color=INK)
    label(mid(xs[2], xs[3]), mid(ys[2], ys[3]),
          "$\\Delta S(a, D_s)$", fs=FS_MATH, color=ACCENT)
    label(mid(xs[0], xs[1]), mid(ys[3], ys[4]), "…", color=PROXY)

    # The similarity kernel that weights those records.  It sits beside the table
    # rather than being joined to it by an arrow: at this width an arrow between
    # the two would be shorter than its own head.
    kx0, kw, kb, kh = x + 146, 58.0, ys[-1] + 6, 36.0
    pts = [(0.00, 0.03), (0.12, 0.08), (0.26, 0.28), (0.38, 0.66),
           (0.50, 1.00), (0.62, 0.66), (0.74, 0.28), (0.88, 0.08), (1.00, 0.03)]
    seg([kx0 + u * kw for u, _ in pts], [kb + v * kh for _, v in pts],
        lw=1.0, color=PROXY, z=3)
    seg([kx0, kx0 + kw], [kb, kb], lw=0.5, color="#8A96A3", z=3)
    label(kx0 + kw / 2, kb + kh + 7, "weight $w$", fs=FS_LABEL, color=PROXY)
    label(kx0 + kw / 2, kb - 8, "distance to\nthe candidate", color=PROXY,
          linespacing=1.25)
    label(x + PW / 2, y + 32,
          "every record reweighted at every step; no simulator", color=INK)
    connective(x, y, "A weighted analog supplies the next state.")


def panel_c():
    x, y = panel(fig, 1, 1, "(c)  Ensemble MCTS with progressive widening")
    cx = x + PW / 2
    yr, ya, ys_, yd, yt = y + 117, y + 94, y + 71, y + 50, y + 24
    box(cx, yr, 30, 13, "$s_0$")
    acts = ((-64, "$a_1$", INK), (-24, "$a_2$", INK), (56, "$a_n$", ACCENT))
    for dx, s, col in acts:
        seg([cx, cx + dx], [yr - 6.5, ya + 5.5], lw=0.6, color=col, z=2)
        # an Ellipse sized in both axes, not a Circle: the figure transform has a
        # different scale in x and y, so a Circle with one radius comes out oval.
        fig.patches.append(Ellipse((fx(cx + dx), fy(ya)),
                                   width=fx(8.8), height=fy(8.8),
                                   transform=fig.transFigure, facecolor="white",
                                   edgecolor=col, linewidth=0.7, zorder=3))
        label(cx + dx + 11, ya, s, fs=FS_MATH, color=col)
    label(cx + 20, ya, "…", color=PROXY)
    kids = ((-64, 30, "$s_{11}$", INK, "white"),
            (-24, 30, "$s_{12}$", INK, "white"),
            (56, 78, "$s_{1n}\\!=\\!s_0 \\oplus \\Delta S$", ACCENT, "#FBF0F0"))
    for dx, w, s, col, fc in kids:
        seg([cx + dx, cx + dx], [ya - 4.4, ys_ + 6.5], lw=0.6, color=col, z=2)
        box(cx + dx, ys_, w, 13, s, fs=FS_MATH, ec=col, tc=col, fc=fc)
    for dx in (-64, -24, 56):
        seg([cx + dx, cx + dx], [ys_ - 6.5, yd + 5], lw=0.5, color=PROXY, z=2,
            ls=":")
        dots(cx + dx, yd + 3)
    arrow(cx, yd - 9, cx, yt + 6.5, lw=0.8, hw=2.0, hl=3.4)
    box(cx, yt, 62, 13, "terminal state", fs=FS_LABEL)
    label(x + 12, yr, "independently\nseeded searches", ha="left", color=PROXY,
          linespacing=1.25)
    connective(x, y, "Each search returns one cost–uncertainty front.")


def panel_d():
    x, y = panel(fig, 0, 1, "(d)  Plurality vote, then act or stop")
    ax = fig.add_axes((fx(x + 18), fy(y + 32), fx(100), fy(84)), zorder=3)
    base = [(0.03, 0.97), (0.17, 0.74), (0.33, 0.56), (0.52, 0.41),
            (0.72, 0.29), (0.96, 0.22)]
    for sx, sy in ((-0.02, 0.05), (0.05, -0.04), (-0.05, -0.07),
                   (0.08, 0.07), (0.02, -0.10)):
        ax.plot([p[0] + sx for p in base], [p[1] + sy for p in base],
                "-", color=PROXY, lw=0.5, alpha=0.35, zorder=2)
    ax.plot([p[0] for p in base], [p[1] for p in base], "-.", color=PROXY,
            lw=1.2, zorder=4)
    ax.plot([p[0] for p in base], [p[1] for p in base], "o", ms=2.0,
            color=PROXY, mec="white", mew=0.3, zorder=5)
    eps = base[3][1]
    ax.axhline(eps, ls=":", lw=0.6, color=ACCENT, zorder=3)
    ax.plot([base[3][0]], [eps], "o", ms=4.0, color=ACCENT, mec="white",
            mew=0.5, zorder=6)
    ax.text(1.0, eps + 0.05, "$\\epsilon$", fontsize=FS_SYM, color=ACCENT,
            ha="right", va="bottom")
    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(0.05, 1.12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("batch cost", fontsize=FS_LABEL, color=INK, labelpad=1.5)
    ax.set_ylabel("terminal uncertainty", fontsize=FS_LABEL, color=INK,
                  labelpad=2.0)
    ax.patch.set_alpha(0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
        ax.spines[sp].set_color(INK)

    tx = x + 126
    label(tx, y + 114, "plurality vote across runs\ngives the MLASP (bold)",
          ha="left", color=PROXY, linespacing=1.3)
    label(tx, y + 88, "uncertainty ≤ $\\epsilon$:", ha="left", fs=FS_SYM,
          color=ACCENT)
    label(tx + 5, y + 77, "stop testing", ha="left")
    label(tx, y + 58, "otherwise:", ha="left", fs=FS_SYM, color=ACCENT)
    label(tx + 5, y + 47, "run the batch", ha="left")
    connective(x, y, "The measured outcome becomes new evidence.")


# ---- assembly --------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    fig = plt.figure(figsize=(W_PT / 72.0, H_PT / 72.0))
    dpis = panel_a(a.art)
    panel_b()
    panel_c()
    panel_d()

    # the cycle: (a) -> (b) across the top, down the right, (c) -> (d) back along
    # the bottom, up the left.  The closing arrow is the feedback loop.
    ymidT, ymidB = PY[0] + PH / 2, PY[1] + PH / 2
    arrow(PX[0] + PW + 4, ymidT, PX[1] - 4, ymidT, lw=1.1)
    arrow(PX[1] + PW / 2, PY[0] - 4, PX[1] + PW / 2, PY[1] + PH + 4, lw=1.1)
    arrow(PX[1] - 4, ymidB, PX[0] + PW + 4, ymidB, lw=1.1)
    arrow(PX[0] + PW / 2, PY[1] + PH + 4, PX[0] + PW / 2, PY[0] - 4, lw=1.1,
          color=ACCENT)

    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=600)
    print(f"wrote {a.out}.pdf / .png at {W_PT} x {H_PT} pt "
          f"(include at width=\\linewidth for scale 1.0)")
    print(f"  panels {PW:.1f} x {PH:.1f} pt each")
    for k, v in dpis.items():
        print(f"  art {k:9s} placed at {v:5.0f} effective dpi")
    print(f"  smallest type on the canvas: {FS_MATH * 0.7:.1f} pt "
          f"(mathtext subscripts at {FS_MATH:.1f} pt base)")
