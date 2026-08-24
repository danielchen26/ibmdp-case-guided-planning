#!/usr/bin/env julia
# CNS-220 population figure and the aggregate counts derived from a sweep.
#
# This is the aggregation step that turns the per-(compound, epsilon) rows emitted by
# cns220_worker_synth.jl into the paper's population-level results:
#   * the four-panel population figure (cost-vs-uncertainty Pareto front; fraction
#     measuring kpuu directly, by QSAR category; median cost by category; cohort
#     composition and confirmation-recommended counts), and
#   * the per-category table -- N, good, and "confirmation recommended" -- together
#     with the pooled confirmation count over the truly-good compounds that the QSAR
#     heuristic would not advance.
#
# Separating this from the worker means the expensive planning sweep is run once and
# the aggregation can be re-derived from its CSV in seconds.
#
# Usage:
#   julia --project=. cns220_figure.jl --results results.csv [--data data/cns220_synthetic.csv]
#                                      [--out cns220_population] [--no-plot]
#
#   --results  sweep output from cns220_worker_synth.jl (one row per compound x epsilon)
#   --data     the cohort the sweep was run on; supplies the QSAR predictors and kpuu
#              used to form the quadrants and the good/bad label.  Defaults to the
#              synthetic surrogate.  Point it at the real cohort (MSD publishes it as
#              CNS_example/data/multitier.csv in MSDLLCpapers/IBMDPDesigns.jl) to
#              recompute the paper's cohort numbers.
#   --out      output basename for the .png/.pdf (default cns220_population)
#   --no-plot  print the counts only; skips Plots.jl entirely
#
# NOTE the cohort file carries measured kpuu values.  This script prints only
# aggregates (counts and medians), never an individual record.
using Pkg
Pkg.activate(joinpath(@__DIR__))
using CSV, DataFrames, Statistics, Printf

function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing ? default : ARGS[i + 1]
end
resfile = argval("--results", nothing)
resfile === nothing && error("--results <sweep csv> is required (output of cns220_worker_synth.jl)")
datafile = argval("--data", joinpath(@__DIR__, "data", "cns220_synthetic.csv"))
outbase  = argval("--out", "cns220_population")
doplot   = !("--no-plot" in ARGS)

df = CSV.read(resfile, DataFrame)
mt = CSV.read(datafile, DataFrame)
# The cohort file may lack compound_id (the public multitier.csv does); the worker
# synthesizes ids from the row order, so reconstruct the same mapping here.
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
cats   = ["Q1", "Q2", "Q3", "Q4", "MID"]
epsvals = 0.0:0.1:1.0
df.compound_id = string.(df.compound_id)
df.cat = [quad(c) for c in df.compound_id]

# ---- Panel A quantities: median first-batch cost and kpuu fraction per epsilon -------
med, q1, q3, pk = Float64[], Float64[], Float64[], Float64[]
for e in epsvals
    sub = df[abs.(df.eps .- e) .< 1e-6, :]
    costs = collect(skipmissing(sub.cost))
    costs = filter(!isnan, Float64.(costs))
    push!(med, isempty(costs) ? NaN : median(costs))
    push!(q1, isempty(costs) ? NaN : quantile(costs, 0.25))
    push!(q3, isempty(costs) ? NaN : quantile(costs, 0.75))
    push!(pk, isempty(sub.uses_kpuu) ? NaN : mean(sub.uses_kpuu .== true))
end

println("\nper-epsilon: median first-batch cost and fraction measuring kpuu directly")
for (i, e) in enumerate(epsvals)
    @printf("   eps=%-4.1f  median=\$%-8.0f  IQR=[\$%.0f, \$%.0f]  kpuu=%.1f%%\n",
            e, med[i], q1[i], q3[i], 100 * pk[i])
end
tight = [c for r in eachrow(df) if r.eps <= 0.1 for c in (r.cost,) if !ismissing(c) && !isnan(c)]
loose = [c for r in eachrow(df) if r.eps >= 0.5 for c in (r.cost,) if !ismissing(c) && !isnan(c)]
@printf("\nband-pooled median first-batch cost: tight (eps<=0.1) \$%.0f (n=%d);  loose (eps>=0.5) \$%.0f (n=%d)\n",
        median(tight), length(tight), median(loose), length(loose))
@printf("fraction measuring kpuu directly: %.1f%% at eps=0  ->  %.1f%% pooled over eps>=0.5\n",
        100 * pk[1], 100 * mean(df[df.eps .>= 0.5, :uses_kpuu] .== true))

# ---- Panel D / table: confirmation recommended among good-but-not-advanced ----------
# "at tight tolerance" = the plan at the tightest epsilon for which the compound has a
# solution: eps=0 where the sweep returns one there, eps=0.1 only otherwise.
promising(c) = qsar[c][1] < 2 && qsar[c][2] < 2
function uses_kpuu_tight(c)
    for e in (0.0, 0.1)
        r = df[(df.compound_id .== c) .& (abs.(df.eps .- e) .< 1e-6), :]
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
    # truly good, but the heuristic would not advance them (not flagged promising)
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
default(fontfamily = "Computer Modern", framestyle = :box)
catlab = Dict("Q1" => "Q1: PgP<2, BCRP<2", "Q2" => "Q2: PgP<2, BCRP>4",
              "Q3" => "Q3: PgP>4, BCRP<2", "Q4" => "Q4: PgP>4, BCRP>4",
              "MID" => "MID: QSAR in [2,4]")
catcol = Dict("Q1" => :seagreen, "Q2" => :steelblue, "Q3" => :darkorange,
              "Q4" => :firebrick, "MID" => :gray45)
epsc = collect(epsvals)

pA = plot(med, epsc; line = (:dashdot, :red, 2.5),
    xlabel = "Cost (USD)", ylabel = "State Uncertainty tolerance ε",
    title = "Population Pareto front (N = $(length(compids)))",
    titlefontsize = 14, guidefontsize = 12, tickfontsize = 10, legendfontsize = 9,
    grid = (:both, :gray, :dash, 0.3), size = (720, 640),
    label = "median MLASP cost", ylims = (-0.05, 1.05), xlims = (0, 5600),
    left_margin = 5Plots.mm, right_margin = 20Plots.mm)
scatter!(pA, med, epsc; marker_z = pk, color = :cividis, clims = (0, 1),
    xerror = (med .- q1, q3 .- med), markersize = 9,
    markerstrokecolor = :white, markerstrokewidth = 0.4, alpha = 0.95,
    colorbar_title = "  fraction measuring k_puu directly", colorbar_titlefontsize = 9,
    label = "")
vline!(pA, [5200]; line = (:dash, :gray, 1.5), label = "full panel (5,200)")
vline!(pA, [4000]; line = (:dot, :purple, 1.5), label = "k_puu assay (4,000)")
annotate!(pA, 1050, 0.05, text("cheap proxies", 8, :left, :darkblue))
annotate!(pA, 4080, 0.95, text("measure k_puu", 8, :left, :darkblue))

pB = plot(; xlabel = "State Uncertainty tolerance ε",
    ylabel = "% compounds measuring k_puu directly",
    title = "Strategy shift by QSAR category",
    titlefontsize = 14, guidefontsize = 12, tickfontsize = 10, legendfontsize = 8,
    grid = (:both, :gray, :dash, 0.3), legend = :topright, left_margin = 5Plots.mm)
for q in cats
    sub = df[df.cat .== q, :]
    pkq = [(u = sub[abs.(sub.eps .- e) .< 1e-6, :uses_kpuu];
            isempty(u) ? NaN : 100 * mean(u .== true)) for e in epsvals]
    plot!(pB, epsc, pkq; color = catcol[q], lw = 2, marker = :circle, ms = 4, label = catlab[q])
end

pC = plot(; xlabel = "State Uncertainty tolerance ε", ylabel = "median IBMDP cost (USD)",
    title = "Cost vs tolerance by QSAR category",
    titlefontsize = 14, guidefontsize = 12, tickfontsize = 10, legendfontsize = 8,
    grid = (:both, :gray, :dash, 0.3), legend = :right, left_margin = 5Plots.mm)
for q in cats
    sub = df[df.cat .== q, :]
    mc = [(cs = filter(!isnan, Float64.(collect(skipmissing(sub[abs.(sub.eps .- e) .< 1e-6, :cost]))));
           isempty(cs) ? NaN : median(cs)) for e in epsvals]
    plot!(pC, epsc, mc; color = catcol[q], lw = 2, marker = :circle, ms = 4, label = catlab[q])
end
hline!(pC, [5200]; line = (:dash, :red, 1.2), alpha = 0.6, label = "full panel (5,200)")

Ns      = [count(c -> catofc[c] == q, compids) for q in cats]
missed  = [[c for c in compids if catofc[c] == q && qsar[c][3] > 0.5 && !promising(c)] for q in cats]
n_missed  = [length(b) for b in missed]
prevented = [count(uses_kpuu_tight, b) for b in missed]
xs = 1:length(cats)
pD = bar(xs .- 0.25, Ns; bar_width = 0.25, color = :gray80, label = "N compounds",
    xlabel = "QSAR category", ylabel = "# compounds",
    title = "Cohort composition + premature No-Go prevented",
    titlefontsize = 13, guidefontsize = 12, tickfontsize = 10, legendfontsize = 8,
    xticks = (xs, cats), grid = (:y, :gray, :dash, 0.3), legend = :topleft,
    left_margin = 5Plots.mm)
bar!(pD, xs, Float64.(n_missed); bar_width = 0.25, color = :seagreen,
     label = "truly good, QSAR would not advance")
bar!(pD, xs .+ 0.25, Float64.(prevented); bar_width = 0.25, color = :steelblue,
     label = "IBMDP recommends confirmatory k_puu\n(premature No-Go prevented)")
for (i, q) in enumerate(cats)
    annotate!(pD, i - 0.25, Ns[i] + 2, text(string(Ns[i]), 7, :center))
    prevented[i] > 0 && annotate!(pD, i + 0.25, prevented[i] + 2,
        text("$(prevented[i])/$(n_missed[i])", 7, :center, :steelblue))
end

fig = plot(pA, pB, pC, pD; layout = (2, 2), size = (1500, 1150),
    plot_title = "IBMDP on CNS brain-penetration cohort (N=$(length(compids))): cost-uncertainty Pareto behavior",
    plot_titlefontsize = 15)
savefig(fig, outbase * ".png")
savefig(fig, outbase * ".pdf")
println("\nsaved $(outbase).png and $(outbase).pdf")
