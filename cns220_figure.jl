#!/usr/bin/env julia
# CNS-220 population figure and the aggregate counts derived from a sweep.
#
# Fixed re-render of the published Figure 3.  Differences from the version in the
# release repository, all deliberate:
#   1. panel A has no legend at all: it used to sit on top of the eps=0 marker at
#      ($4000, 0), the panel's headline point, and the front is too narrow a strip
#      for a three-entry box to clear it anywhere.  The two reference costs are
#      named by rotated text beside their own lines, the median line in the caption;
#   2. panel A's two decorative corner labels ("cheap proxies", "measure k_puu") are
#      gone.  They were placed at the tolerance that produces the OPPOSITE strategy
#      -- "measure k_puu" sat at eps=0.95, where 98% of plans use cheap proxies --
#      so a reader taking their vertical position as meaningful read the trend
#      backwards.  What they conveyed is now carried by the two labelled cost lines;
#   3. panels B and C give their markers the series colour, so the five QSAR
#      categories are distinguishable (they all rendered black before);
#   4. panel D's y range leaves headroom, so the tallest bar and its count label
#      are inside the axes instead of running into the title;
#   5. panel D's "prevented/missed" label sits above the taller of the two bars
#      it annotates instead of on top of the green one;
#   6. the in-figure super-title, which restated the LaTeX caption, is gone;
#   7. the canvas is rendered at final print width and the nominal font sizes are
#      set from GR's measured type inflation AND the \includegraphics reduction, so
#      what reaches the page is 10.2 pt titles and 7.3 pt ticks rather than 2 pt
#      (downscaled 1500 px canvas), 12.3 pt (nominal sizes taken at face value,
#      which clipped the titles), or 5.8 pt ticks (inflation applied but the
#      include reduction overlooked) -- see the note above `default` below;
#   8. panels carry (A)--(D) letters matching the caption, and the axis label
#      names the quantity a design is actually filed at (the requested terminal-
#      uncertainty tolerance, not a measured uncertainty).
#
# Usage:
#   julia --project=. cns220_figure.jl --results results.csv --data cohort.csv
#                                            [--out cns220_population] [--no-plot]
#
# NOTE the cohort file carries measured kpuu values.  This script prints only
# aggregates (counts and medians), never an individual record.
using Pkg
Pkg.activate(get(ENV, "FIG3_ENV", joinpath(@__DIR__)); io = devnull)
using CSV, DataFrames, Statistics, Printf

function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing ? default : ARGS[i + 1]
end
resfile = argval("--results", nothing)
resfile === nothing && error("--results <sweep csv> is required")
datafile = argval("--data", joinpath(@__DIR__, "data", "cns220_synthetic.csv"))
outbase  = argval("--out", "cns220_population")
doplot   = !("--no-plot" in ARGS)

df = CSV.read(resfile, DataFrame)
mt = CSV.read(datafile, DataFrame)
ids = "compound_id" in names(mt) ? string.(mt.compound_id) :
      [@sprintf("SYN-%04d", i) for i in 1:nrow(mt)]
qsar = Dict(ids[i] => (mt[i, "1uM_PgP_qsar"], mt[i, "100_nM_Mouse_BCRP_qsar"], mt[i, "kpuu"])
            for i in 1:nrow(mt))

# The rule-based heuristic's hard-threshold quadrants, plus the MID band it handles
# only crudely (at least one QSAR in [2,4]).
function quad(cid)
    p, b, _ = qsar[cid]
    (p < 2 && b < 2) && return "Q1"
    (p < 2 && b > 4) && return "Q2"
    (p > 4 && b < 2) && return "Q3"
    (p > 4 && b > 4) && return "Q4"
    return "MID"
end
cats    = ["Q1", "Q2", "Q3", "Q4", "MID"]
epsvals = 0.0:0.1:1.0
# A row belongs to tolerance level e only if it was filed at that level.  Designs are
# filed at the tolerance they were solved at, so this is an exact match on a requested
# grid value; the only rows filed elsewhere are the empty plans of the already-terminal
# s0 exception, which carry that compound's realized prior uncertainty instead.
# Exact equality is the right test and not a floating-point hazard here: the sweep writes
# back the threshold value it was handed, the CSV therefore holds clean decimal strings,
# and Julia's 0.0:0.1:1.0 reproduces them exactly.  A tolerance-based test is actively
# wrong at e=0, where it absorbs a realized prior uncertainty of 2.1e-12 belonging to a
# compound that already contributes an exact 0.0 row, counting that compound twice.
onlev(x, e) = x == e
df.compound_id = string.(df.compound_id)
df.cat = [quad(c) for c in df.compound_id]

# ---- Panel A quantities: median first-batch cost and kpuu fraction per epsilon -------
med, q1, q3, pk, nlev = Float64[], Float64[], Float64[], Float64[], Int[]
for e in epsvals
    sub = df[onlev.(df.eps, e), :]
    costs = filter(!isnan, Float64.(collect(skipmissing(sub.cost))))
    push!(med, isempty(costs) ? NaN : median(costs))
    push!(q1,  isempty(costs) ? NaN : quantile(costs, 0.25))
    push!(q3,  isempty(costs) ? NaN : quantile(costs, 0.75))
    push!(pk,  isempty(sub.uses_kpuu) ? NaN : mean(sub.uses_kpuu .== true))
    push!(nlev, length(unique(sub.compound_id)))
end

println("\nper-epsilon: median first-batch cost, fraction measuring kpuu, contributing compounds")
for (i, e) in enumerate(epsvals)
    @printf("   eps=%-4.1f  median=\$%-8.0f  IQR=[\$%.0f, \$%.0f]  kpuu=%.1f%%  n=%d\n",
            e, med[i], q1[i], q3[i], 100 * pk[i], nlev[i])
end
tight = filter(!isnan, Float64.(collect(skipmissing(df[df.eps .<= 0.1, :cost]))))
loose = filter(!isnan, Float64.(collect(skipmissing(df[df.eps .>= 0.5, :cost]))))
@printf("\nband-pooled median first-batch cost: tight (eps<=0.1) \$%.0f (n=%d);  loose (eps>=0.5) \$%.0f (n=%d)\n",
        median(tight), length(tight), median(loose), length(loose))
@printf("fraction measuring kpuu directly: %.1f%% at eps=0  ->  %.1f%% pooled over eps>=0.5\n",
        100 * pk[1], 100 * mean(df[df.eps .>= 0.5, :uses_kpuu] .== true))
kl = [100 * pk[i] for i in eachindex(collect(epsvals)) if collect(epsvals)[i] >= 0.5]
@printf("per-level kpuu fraction over eps>=0.5: %.1f%%--%.1f%%\n", minimum(kl), maximum(kl))

# rows the per-level panels cannot use: no plan returned, or an empty plan filed off-grid
# at a realized prior uncertainty
offgrid = count(r -> !ismissing(r.eps) && all(!onlev(r.eps, e) for e in epsvals),
                eachrow(df[.!ismissing.(df.eps), :]))
noplan  = count(ismissing, df.eps) + count(isnan, Float64.(coalesce.(df.cost, NaN)))
@printf("rows excluded from the per-level panels: %d filed off-grid at a realized prior uncertainty, %d with no plan\n",
        offgrid, noplan)

# ---- Panel D / table: confirmation recommended among good-but-not-advanced ----------
# "at tight tolerance" = the plan at the tightest epsilon for which the compound has a
# solution: eps=0 where the sweep returns one there, eps=0.1 only otherwise.
promising(c) = qsar[c][1] < 2 && qsar[c][2] < 2
function uses_kpuu_tight(c)
    for e in (0.0, 0.1)
        r = df[(df.compound_id .== c) .& onlev.(df.eps, e), :]
        nrow(r) > 0 && return r.uses_kpuu[1] == true
    end
    return false
end
compids = unique(df.compound_id)
catofc  = Dict(c => quad(c) for c in compids)
println("\nper-category counts (N, good, confirmation recommended among good-but-not-advanced)")
percat = map(cats) do q
    inq  = [c for c in compids if catofc[c] == q]
    good = [c for c in inq if qsar[c][3] > 0.5]
    elig = [c for c in good if !promising(c)]
    (; q, n = length(inq), good = length(good), rec = count(uses_kpuu_tight, elig),
       elig = length(elig))
end
for r in percat
    @printf("   %-4s N=%-4d good=%-4d confirmation recommended %d/%d\n",
            r.q, r.n, r.good, r.rec, r.elig)
end
@printf("   %-4s N=%-4d good=%-4d confirmation recommended %d/%d   <-- the pooled count\n",
        "all", length(compids), count(c -> qsar[c][3] > 0.5, compids),
        sum(r.rec for r in percat), sum(r.elig for r in percat))

if !doplot
    println("\n(--no-plot: skipping the figure)")
    exit(0)
end

using Plots
Plots.gr()
theme(:default)
# Two multipliers separate a nominal size from the size that reaches the page, and
# both have to be applied:
#   * GR inflation, 1.551x, measured rather than assumed -- ten capital H's requested
#     at nominal 10 carry 10.86 pt of capital ink at 1200 dpi (figfix/inkmeasure.py
#     on figfix/ft_470x400.pdf), which at the usual 0.70 cap-height ratio is 15.51 pt
#     of effective body size.
#   * the \includegraphics reduction, 0.92 * 469.755 / 460 = 0.9395.
# On-page effective size is therefore nominal * 1.4571.  An earlier revision of this
# comment applied only the first factor and so overstated the result by 6.4%: the
# nominal 4 it recorded as "6.2 pt on the page" in fact reached the page at 5.8 pt,
# under the 6 pt publisher floor.  Plots.Font stores pointsize as an Int, so the
# available on-page sizes come in 1.46 pt steps; the sizes below give 10.2 pt titles,
# 8.7 pt axis labels and 7.3 pt ticks, legends and in-panel labels.
const FS_TITLE = 7    # -> 10.2 pt
const FS_GUIDE = 6    # ->  8.7 pt
const FS_TICK  = 5    # ->  7.3 pt
const FS_ANNOT = 5    # ->  7.3 pt
default(fontfamily = "Computer Modern", framestyle = :box,
        titlefontsize = FS_TITLE, guidefontsize = FS_GUIDE,
        tickfontsize = FS_TICK, legendfontsize = FS_ANNOT,
        foreground_color_legend = :gray70, background_color_legend = :white)

# The quadrant definitions live in the caption and on panel D's axis; the legends
# carry only the short names, which is what lets them fit at print size.
catcol = Dict("Q1" => :seagreen, "Q2" => :steelblue, "Q3" => :darkorange,
              "Q4" => :firebrick, "MID" => :gray45)
epsc = collect(epsvals)
EPSLAB = "Terminal uncertainty tolerance \$\\epsilon\$"
COSTX  = ([0, 1000, 2000, 3000, 4000, 5000], ["0", "1k", "2k", "3k", "4k", "5k"])

# ---- (A) population Pareto front ----------------------------------------------------
# The two reference costs are named by rotated text beside the lines they mark, not by
# a legend box: the front is a narrow strip (one point at the kpuu price at eps=0, then
# a vertical run at the two-proxy price), so any legend large enough to hold three
# entries lands on data.  The red dash-dotted median line is named in the caption.
pA = plot(med, epsc; line = (:dashdot, :red, 1.8),
    xlabel = "Median first-batch cost (USD)", ylabel = EPSLAB,
    title = "(A) Population Pareto front (\$N=$(length(compids))\$)",
    grid = (:both, :gray, :dash, 0.3), xticks = COSTX,
    label = "", ylims = (-0.05, 1.05), xlims = (0, 5900),
    legend = false, colorbar = true, right_margin = 4Plots.mm,
    colorbar_ticks = ([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"]))
# linecolor has to be named explicitly: with `color` bound to a gradient for marker_z
# the series has no scalar line colour, and the IQR whiskers are then drawn invisible.
scatter!(pA, med, epsc; marker_z = pk, color = :cividis, clims = (0, 1),
    xerror = (med .- q1, q3 .- med), markersize = 3.5,
    linecolor = :gray35, linewidth = 0.7,
    markerstrokecolor = :gray20, markerstrokewidth = 0.4, alpha = 0.95,
    colorbar = true, colorbar_title = "fraction using \$k_{puu}\$",
    colorbar_titlefontsize = FS_ANNOT, colorbar_tickfontsize = FS_TICK,
    colorbar_ticks = ([0, 0.5, 1], ["0", "0.5", "1"]), label = "")
vline!(pA, [5200]; line = (:dash, :gray, 1.0), label = "")
vline!(pA, [4000]; line = (:dot, :purple, 1.0), label = "")
# Each name sits beside its own line with ~350 USD of clearance.  The colour bar leaves
# this panel about 138 pt of plot width for a 5900 USD range, so a 6 pt rotated glyph
# column is ~300 USD wide -- less clearance than that and the text lands on the line.
# "full panel" goes to the RIGHT of its 5200 line, not the left: the x grid has a line at
# 5000, so the mirror position 4850 puts the text on the 5000 gridline and 150 USD from it
# against 350 from the line it names, i.e. it reads as labelling the gridline.  Nothing is
# drawn between 5200 and the 5900 axis limit, so 5480 is clear of both.
annotate!(pA, 3650, 0.55, text("\$k_{puu}\$ assay", FS_ANNOT, :center, :purple, rotation = 90))
annotate!(pA, 5480, 0.55, text("full panel", FS_ANNOT, :center, :gray35, rotation = 90))

# ---- (B) strategy shift by QSAR category -------------------------------------------
pB = plot(; xlabel = EPSLAB, ylabel = "% of compounds measuring \$k_{puu}\$",
    title = "(B) Strategy shift by QSAR category",
    grid = (:both, :gray, :dash, 0.3), legend = :topright, ylims = (-5, 112),
    legend_columns = 2)
for q in cats
    sub = df[df.cat .== q, :]
    pkq = [(u = sub[onlev.(sub.eps, e), :uses_kpuu];
            isempty(u) ? NaN : 100 * mean(u .== true)) for e in epsvals]
    plot!(pB, epsc, pkq; color = catcol[q], lw = 1.5, marker = :circle, ms = 2.6,
          markercolor = catcol[q], markerstrokecolor = catcol[q], label = q)
end

# ---- (C) median cost by QSAR category ----------------------------------------------
pC = plot(; xlabel = EPSLAB, ylabel = "Median IBMDP cost (USD)",
    title = "(C) Median cost by QSAR category",
    grid = (:both, :gray, :dash, 0.3), legend = (0.60, 0.74), yticks = COSTX,
    ylims = (0, 5700))
for q in cats
    sub = df[df.cat .== q, :]
    mc = [(cs = filter(!isnan, Float64.(collect(skipmissing(sub[onlev.(sub.eps, e), :cost]))));
           isempty(cs) ? NaN : median(cs)) for e in epsvals]
    plot!(pC, epsc, mc; color = catcol[q], lw = 1.5, marker = :circle, ms = 2.6,
          markercolor = catcol[q], markerstrokecolor = catcol[q], label = q)
end
hline!(pC, [5200]; line = (:dash, :red, 1.0), alpha = 0.6, label = "full panel")

# ---- (D) cohort composition and confirmations --------------------------------------
Ns        = [count(c -> catofc[c] == q, compids) for q in cats]
missed    = [[c for c in compids if catofc[c] == q && qsar[c][3] > 0.5 && !promising(c)] for q in cats]
n_missed  = [length(b) for b in missed]
prevented = [count(uses_kpuu_tight, b) for b in missed]
ytop      = maximum(Ns) * 1.45
xs = 1:length(cats)
# no y grid here: every bar carries its count as a printed label, so the horizontal
# lines add nothing and the two that fell at 20 and 40 were struck through the "17"
# and "34/36" labels.  The tick marks stay, so the scale is still readable.
pD = bar(xs .- 0.25, Ns; bar_width = 0.25, color = :gray80, label = "all compounds",
    xlabel = "QSAR category", ylabel = "Number of compounds",
    title = "(D) Cohort and confirmations",
    xticks = (xs, cats), grid = false, legend = :topleft,
    ylims = (0, ytop), xlims = (0.55, length(cats) + 0.68))
bar!(pD, xs, Float64.(n_missed); bar_width = 0.25, color = :seagreen,
     label = "good, QSAR would not advance")
bar!(pD, xs .+ 0.25, Float64.(prevented); bar_width = 0.25, color = :steelblue,
     label = "IBMDP recommends \$k_{puu}\$")
ypad = 0.02 * ytop
# The axis tick marks reach inside the box frame, and the ratio label of the last
# category is wide enough that its final digit lands on one.  Each label is pushed
# clear of the nearest tick row; only MID moves.
function clear_of_ticks(y, step = 20, pad = 4.5)
    r = mod(y, step)
    r < pad && return y + (pad - r)
    r > step - pad && return y + (step - r) + pad
    return y
end
for (i, q) in enumerate(cats)
    annotate!(pD, i - 0.25, Ns[i] + ypad, text(string(Ns[i]), FS_ANNOT, :center))
    # centred over the blue bar it annotates, clear of the green bar's outline below and
    # of the grey bar to its left.  This needs the widened x range set above: the label is
    # wider than its bar, and on the default limits its final digit at MID -- the last
    # category -- ran under the right-hand axis tick.
    prevented[i] > 0 && annotate!(pD, i + 0.25,
        clear_of_ticks(max(n_missed[i], prevented[i]) + 1.8 * ypad),
        text("$(prevented[i])/$(n_missed[i])", FS_ANNOT, :center, :steelblue))
end

fig = plot(pA, pB, pC, pD; layout = (2, 2), size = (460, 420), dpi = 300,
           left_margin = 2Plots.mm, bottom_margin = 2Plots.mm,
           top_margin = 1Plots.mm)
savefig(fig, outbase * ".pdf")
savefig(fig, outbase * ".png")
println("\nsaved $(outbase).pdf and $(outbase).png")
