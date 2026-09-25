#!/usr/bin/env julia
# ESI Figure S2 -- effect of the goal-likelihood floor tau on the MLASP, for one CNS compound.
#
# This REPLACES the previous figs/figure4_improved.png, which was not traceable to run
# output.  Its generator (IBMDP_arxiv/pub_figures/tau_plot.py) plotted hard-coded
# coordinates from IBMDP_arxiv/figs/tau_example/mlasp_paths.json, and that JSON's own
# provenance note (pareto_front_action_sets_summary.md) records that its numbers were
# "extracted directly from the ensemble Pareto front figures" -- i.e. read by eye off a
# raster image.  This figure is computed from that same run's ensemble CSVs instead, so
# every plotted coordinate is a recorded value.  Three defects in the old figure make the
# replacement necessary rather than cosmetic; all three are checkable against the two
# CSVs named on the command line.
#
#  1. Every transcribed cost is snapped to a round hundred and several are wrong by a
#     large margin.  JSON vs run, tau=0.6: 700/1600/2200/2800 against the recorded
#     800.78/1702.22/2202.20/2460.17; tau=0.9: 400 (six times) against 404.88-409.73,
#     900 against 946.71, 3500 against 3266.59, 3800 against 3480.85.  The largest error
#     is 340 cost units.
#  2. One plotted point is fabricated.  The run's loosest tau=0.6 record is the initial
#     state: Threshold 0.8451, Average_Utility 0.0, Frequency 50, Action_Set EMPTY -- a
#     plan that measures nothing.  The next record is Threshold 0.80, ["1uM_PgP"],
#     400.02.  The JSON has only nine tau=0.6 points: it deletes the eps=0.80 point and
#     moves its action set onto the eps=0.845 one, publishing {cost 0, uncertainty 0.85,
#     action_set ["1uM_PgP"]}.  The published figure therefore labels a $400 PgP assay
#     against a plan whose run record has no assay in it.
#  3. The old locus is not the plurality-vote path the SI says it is.  At tau=0.6
#     eps=0.5 it plots {100nM_PgP, 1uM_PgP} (11 of 50) where the modal plan is
#     {1uM_PgP} (19 of 50); at eps=0.4 it plots {1uM_PgP} (6 of 50) where the modal plan
#     is {100nM_PgP, 1uM_PgP} (16 of 50); at tau=0.9 eps=0.2 it plots {100nM_BCRP}
#     (2 of 50) where the modal plan is {kpuu} (3 of 50).  This script takes the modal
#     plan at every tolerance, which is what the SI defines.
#
# The CSV paths are supplied on the command line rather than written here, because the
# run directory name encodes a measured k_puu value.  The CSVs themselves carry planner
# outputs only, so every number this script prints is releasable.
#
# Two panels share the tolerance axis, which is the point of the layout: panel A is the
# quantity the planner minimizes and panel B is the quantity the chemist pays.  The old
# figure plotted only the objective and then spent four caption sentences insisting it was
# not a price; here the reader sees the two disagree.
#
# Usage, from the repository root:
#   julia --project=. figures/esi_figS2_tau.jl \
#          --tau06 figures/data/esi_figS2_ensemble_tau0.6.csv \
#          --tau09 figures/data/esi_figS2_ensemble_tau0.9.csv \
#          --out esi_figS2
#
# The inputs carry planner outputs (tolerance, action set, objective, frequency) only --
# no measured compound values -- so everything this script prints is releasable.

import Pkg
# the repository root, one level up: that is where Project.toml/Manifest.toml live.
Pkg.activate(get(ENV, "FIGS2_ENV", dirname(@__DIR__)); io = devnull)

using CSV, DataFrames, Statistics, Printf, Plots
Plots.gr()
default(fontfamily = "Computer Modern", framestyle = :box)

argval(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i + 1])

F06 = argval("--tau06", "")
F09 = argval("--tau09", "")
OUT = argval("--out", joinpath(@__DIR__, "esi_fig_s2"))
YMAX = parse(Float64, argval("--ymax", "5200"))   # the full CNS panel price, as in main Fig 2
YTOP = 5750.0                                     # both panels share this scale
PANEL = 5200                                      # $4,000 k_puu + three $400 in vitro proxies

# Assay prices are the main text's schedule: $400 per in vitro proxy, $4,000 for the
# in vivo kpuu assay.  MRT is a QSAR-derived covariate, not a purchasable assay.
PRICE = Dict("100nM_PgP" => 400, "1uM_PgP" => 400, "100nM_BCRP" => 400,
             "1uM_BCRP" => 400, "kpuu" => 4000, "mrt" => 0)
# Short codes are those of main Figure 2, so the two figures share one key.
CODE  = Dict("kpuu" => "k", "100nM_PgP" => "P_1", "1uM_PgP" => "P_2",
             "100nM_BCRP" => "B_1", "1uM_BCRP" => "B_2", "mrt" => "M")

toks(::Missing) = String[]
function toks(s::AbstractString)
    filter(!isempty, [strip(t) for t in split(replace(s, r"[\[\]\"]" => ""), ",")])
end
label_of(t) = isempty(t) ? "\$\\emptyset\$" : "\$" * join([get(CODE, x, x) for x in t], "\\,") * "\$"
price_of(t) = sum(Int[get(PRICE, x, 0) for x in t]; init = 0)

# The maximum-likelihood action-set path: at each tolerance, the plan the plurality of the
# ensemble returned.  N_e is read off the tightest level, where every member has a plan --
# the same convention as main Figure 2, and the reason a share can be well below 1.
function mlasp(file)
    df = DataFrame(CSV.File(file))
    Ne = sum(df.Frequency[df.Threshold .== minimum(df.Threshold)])
    out = combine(groupby(df, :Threshold)) do g
        j = argmax(g.Frequency)
        (obj = g.Average_Utility[j], set = g.Action_Set[j], freq = g.Frequency[j])
    end
    sort!(out, :Threshold, rev = true)
    out.t     = [toks(s) for s in out.set]
    out.price = [price_of(t) for t in out.t]
    out.share = out.freq ./ Ne
    out.code  = [label_of(t) for t in out.t]
    return out, Ne, nrow(df)
end

p6, Ne6, n6 = mlasp(F06)
p9, Ne9, n9 = mlasp(F09)

for (nm, p, Ne, n) in (("tau=0.6", p6, Ne6, n6), ("tau=0.9", p9, Ne9, n9))
    @printf("\n=== %s   N_e=%d   %d ensemble records, %d tolerance levels ===\n", nm, Ne, n, nrow(p))
    println("     eps   objective   freq  share    price  batch")
    for r in eachrow(p)
        @printf("  %6.4f %11.1f %6d %6.2f  \$%5d  %s\n",
                r.Threshold, r.obj, r.freq, r.share, r.price,
                isempty(r.t) ? "(none -- prior alone)" : join(r.t, ", "))
    end
end
# the headline comparison, recomputed rather than quoted
for e in (0.3, 0.4)
    a = p6[p6.Threshold .== e, :]; b = p9[p9.Threshold .== e, :]
    if nrow(a) == 1 && nrow(b) == 1
        @printf("\nat eps=%.1f:  tau=0.6 -> \$%d %s (share %.2f, obj %.1f);  tau=0.9 -> \$%d %s (share %.2f, obj %.1f)\n",
                e, a.price[1], join(a.t[1], "+"), a.share[1], a.obj[1],
                b.price[1], join(b.t[1], "+"), b.share[1], b.obj[1])
    end
end
@printf("\nmodal-plan support along the two loci: %.2f--%.2f\n",
        min(minimum(p6.share), minimum(p9.share)), max(maximum(p6.share), maximum(p9.share)))

# The caption's load-bearing demonstration that panel A's vertical axis is not a price: one
# batch, at one price, plotted at several different ordinates.  Recomputed, not quoted.
for (nm, p) in (("tau=0.6", p6), ("tau=0.9", p9))
    for b in unique(p.t)
        d = p[[t == b for t in p.t], :]
        nrow(d) < 2 && continue
        @printf("%s: batch %s (\$%d) appears at %d ordinates: %s   (eps %s)\n",
                nm, isempty(b) ? "(none)" : join(b, "+"), price_of(b), nrow(d),
                join([@sprintf("%.1f", v) for v in d.obj], ", "),
                join([@sprintf("%.2f", v) for v in d.Threshold], ", "))
    end
end

# ---- plotting ----------------------------------------------------------------------
# Two multipliers stand between a nominal size and the size that reaches the page:
#   * GR inflation, 1.551x, measured -- ten capital H's requested at nominal 10 carry
#     10.86 pt of capital ink at 1200 dpi (figfix/inkmeasure.py on ft_470x400.pdf),
#     which at the usual 0.70 cap-height ratio is 15.51 pt of effective body size;
#   * the \includegraphics reduction, 0.89 * 469.755 / 470 = 0.8895.
# On-page effective size is therefore nominal * 1.3796.  An earlier version of this note
# applied only the first factor and so read 6.2 pt off a nominal 4 that in fact reached
# the page at 5.5 pt, under the 6 pt publisher floor.  Plots.Font stores pointsize as an
# Int, so the sizes below step by 1.38 pt: 9.7 pt titles, 8.3 pt axis labels, 6.9 pt
# ticks, legends and in-panel labels.
# The two panels are stacked rather than side by side so that they literally share the
# tolerance axis: a vertical line through the figure meets the same eps in both, which is
# the comparison the figure exists to make.
const FS_TITLE = 7    # -> 9.7 pt
const FS_GUIDE = 6    # -> 8.3 pt
const FS_TICK  = 5    # -> 6.9 pt
const FS_ANNOT = 5    # -> 6.9 pt
C6, C9 = RGB(0.824, 0.412, 0.118), RGB(0.118, 0.227, 0.541)   # chocolate / navy, as before
SZ(sh) = 1.8 + 3.6 * sh          # marker area tracks the ensemble share backing that plan
XR, YR = 1.12, YTOP + 400.0      # axis spans, used to normalize label clearances

# Both loci's eps=0 plans lie far beyond panel A's range, and at the same tolerance, so a
# single clamp height would draw one marker on top of the other and lose one of the two
# values.  Off-range markers are therefore parked at two different heights and each is
# annotated with its own exact objective; the caption says the range is exceeded.
PARK = Dict(:t06 => 0.985, :t09 => 0.930)

function locus!(p, d, col, mk, lab, key; clampy = nothing)
    park(v) = clampy === nothing ? v : min(v, clampy * PARK[key])
    ys = park.(clampy === nothing ? Float64.(d.price) : Float64.(d.obj))
    plot!(p, d.Threshold, ys; line = (:dash, col, 1.4), label = lab)
    for (i, r) in enumerate(eachrow(d))
        raw = clampy === nothing ? Float64(r.price) : Float64(r.obj)
        off = clampy !== nothing && raw > clampy
        scatter!(p, [r.Threshold], [ys[i]];
                 marker = (off ? :utriangle : mk, SZ(r.share), col),
                 markerstrokecolor = :gray20, markerstrokewidth = 0.3,
                 alpha = 0.85, label = "")
    end
end

# Batch codes are placed by clearance search rather than at a fixed offset.  The two loci
# coincide at several tolerances and both spend long runs at $400, so every fixed offset
# drives some label either through a marker or along the horizontal segment joining two of
# them -- which is what struck "P_1P_2B_1" through its twin in the first draft.  The
# obstacle set is both polylines, sampled between vertices, plus the labels already placed.
function obstacles(ds)
    pts = Tuple{Float64,Float64}[]
    for d in ds
        xs, ys = Float64.(d.Threshold), Float64.(d.price)
        for i in eachindex(xs)
            push!(pts, (xs[i], ys[i]))
            i < length(xs) && for s in 0.1:0.1:0.9
                push!(pts, (xs[i] + s * (xs[i + 1] - xs[i]), ys[i] + s * (ys[i + 1] - ys[i])))
            end
        end
    end
    return pts
end
clearance(ax, ay, obs) =
    minimum(hypot((ax - q[1]) / (0.05 * XR), (ay - q[2]) / (0.085 * YR)) for q in obs)

CANDS = [(dx, dy) for dy in (330.0, -330.0, 700.0, -700.0, 1080.0, -1080.0)
                  for dx in (0.0, 0.055, -0.055)]

function place!(p, d, col, obs)
    for r in eachrow(d)
        x, y = Float64(r.Threshold), Float64(r.price)
        best, bestc = (x, y + 330.0), -Inf
        for (dx, dy) in CANDS
            ax, ay = x + dx, y + dy
            (ay < -330.0 || ay > 5050.0 || ax < -0.04 || ax > 1.04) && continue
            c = clearance(ax, ay, obs)
            if c > 1.0
                best, bestc = (ax, ay), Inf
                break
            elseif c > bestc
                best, bestc = (ax, ay), c
            end
        end
        # A displaced label is closer to a neighbouring tolerance than to its own, so a
        # code pushed sideways gets a hairline leader back to the marker it describes.
        if abs(best[1] - x) > 0.01
            plot!(p, [x, x + 0.72 * (best[1] - x)], [y, y + 0.72 * (best[2] - y)];
                  line = (col, 0.4), alpha = 0.5, label = "")
        end
        annotate!(p, best[1], best[2], text(r.code, FS_ANNOT, :center, col))
        push!(obs, best)
    end
end

# (A) what the planner minimizes.  Its tick labels are suppressed because panel B carries
# the shared axis directly beneath it.
pA = plot(; xlims = (-0.06, 1.06), ylims = (-400, YTOP),
    ylabel = "Ensemble mean objective",
    title = "(A) What the planner minimises", xticks = (0:0.2:1.0, fill("", 6)),
    yticks = ([0, 1000, 2000, 3000, 4000, 5000], ["0", "1000", "2000", "3000", "4000", "5000"]),
    grid = (:both, :gray, :dash, 0.3), legend = :topright,
    titlefontsize = FS_TITLE, guidefontsize = FS_GUIDE, tickfontsize = FS_TICK, legendfontsize = FS_ANNOT,
    foreground_color_legend = :gray70, background_color_legend = :white)
locus!(pA, p6, C6, :circle, "\$\\tau=0.6\$", :t06; clampy = YMAX)
locus!(pA, p9, C9, :rect,   "\$\\tau=0.9\$", :t09; clampy = YMAX)
# nothing is silently clamped: each off-range point keeps its exact value, printed beside
# its own parked marker.  The offset lands the string between the eps=0.2 and eps=0.4
# gridlines -- a vertical gridline running through 7 pt type reads as a strikethrough.
for (d, col, key) in ((p6, C6, :t06), (p9, C9, :t09))
    for r in eachrow(d[d.obj .> YMAX, :])
        annotate!(pA, r.Threshold + 0.225, YMAX * PARK[key],
                  text(@sprintf("%s = %.0f", r.code, r.obj), 4, :left, col))
    end
end

# (B) what the batch actually costs.  Panel B shares panel A's vertical scale, so the two
# can be read against each other, and carries the full-panel price as a reference line --
# the quantity the main text's savings are expressed against.
pB = plot(; xlims = (-0.06, 1.06), ylims = (-400, YTOP),
    xlabel = "Tolerance \$\\epsilon\$", ylabel = "Modal batch price (USD)",
    title = "(B) What the batch costs", xticks = 0:0.2:1.0,
    yticks = ([0, 1000, 2000, 3000, 4000, 5000], ["0", "1000", "2000", "3000", "4000", "5000"]),
    grid = (:both, :gray, :dash, 0.3), legend = false,
    titlefontsize = FS_TITLE, guidefontsize = FS_GUIDE, tickfontsize = FS_TICK, legendfontsize = FS_ANNOT)
hline!(pB, [PANEL]; line = (:dot, :gray45, 0.9), label = "")
# GR's text engine reads a bare "$" as opening math mode and has no escape for a literal
# dollar sign, so the unit is carried by the axis label and the number stands alone here.
annotate!(pB, 1.04, PANEL + 300, text("full panel: $PANEL", FS_ANNOT, :right, :gray35))
locus!(pB, p6, C6, :circle, "", :t06)
locus!(pB, p9, C9, :rect,   "", :t09)
# label the batch only where its composition changes along a locus
changes(d) = (m = [i == 1 || d.t[i] != d.t[i - 1] for i in 1:nrow(d)]; d[m, :])
obs = obstacles((p6, p9))
place!(pB, changes(p9), C9, obs)
place!(pB, changes(p6), C6, obs)

fig = plot(pA, pB; layout = (2, 1), size = (470, 400), dpi = 300,
           left_margin = 4Plots.mm, bottom_margin = 2Plots.mm,
           right_margin = 2Plots.mm, top_margin = 1Plots.mm)
savefig(fig, OUT * ".pdf")
savefig(fig, OUT * ".png")
println("\nsaved $(OUT).pdf and $(OUT).png")
