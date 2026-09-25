#!/usr/bin/env julia
# Standalone re-render of Figure 2 (four-panel ensemble Pareto fronts).
#
# Usage, with the ensemble output shipped in this repository:
#   julia --project=. figures/fig2_fronts.jl \
#       --s1 figures/data/fig2_panel1_ensemble_tau0.9.csv --k1 0.529 \
#       --s2 figures/data/fig2_panel2_ensemble_tau0.9.csv --k2 0.534 \
#       --s3 figures/data/fig2_panel3_ensemble_tau0.9.csv --k3 0.541 \
#       --s4 figures/data/fig2_panel4_ensemble_tau0.9.csv --k4 0.640 \
#       --out replotted.pdf
# The --k values are only panel-title labels and enter no computation; they are the
# four compounds' measured k_puu to three decimals, the form the manuscript itself
# prints on these panel titles (Table 1 rounds them to two).  The CSVs are
# named for the panel they feed rather than for the planner's own output directory,
# whose name encodes that k_puu at full precision (see the README's data policy);
# their contents are planner output only -- tolerance, action set, ensemble mean
# objective, vote count -- and carry no measured compound property.
#
# Reads the four stored ensemble_results_tau_0.9.csv files and emits one
# print-sized PDF. Differences from the previous version, all deliberate:
#   * axis labels name the quantities actually plotted (planner objective,
#     tolerance eps) instead of "Cost" and "State Uncertainty";
#   * all four panels share one horizontal and one vertical range;
#   * one shared colour key instead of none;
#   * points beyond the shared range are not silently clamped -- the
#     maximum-likelihood point among them is marked with its exact value and
#     the panel states how many others there are and their maximum;
#   * action sets are labelled with short codes, only where the modal set
#     changes, so the labels fit at print size;
#   * the nominal font sizes account for two multipliers at once, so that the
#     type reaching the page clears the 6 pt publisher floor:
#       - GR inflation, measured not assumed.  Ten capital H's requested at
#         nominal 10 on this canvas carry 10.86 pt of capital ink (1200 dpi
#         measurement, figfix/inkmeasure.py on figfix/ft_470x400.pdf); at the
#         usual 0.70 cap-height ratio that is 15.51 pt of effective body size,
#         i.e. GR renders 1.551x the number it is handed.
#       - the \includegraphics reduction, 0.85 * 469.755 / 470 = 0.8496.
#     On-page effective size is therefore nominal * 1.551 * 0.8496 = nominal *
#     1.318.  Plots.Font stores pointsize as an Int, so only whole nominal sizes
#     are available and the on-page sizes come in 1.32 pt steps: 9.2 pt titles,
#     7.9 pt axis labels, 6.6 pt ticks, in-panel labels and legend, against the
#     journal's 10 pt body text.  The previous sizes (4 pt ticks) reached the
#     page at 5.3 pt, below the 6 pt floor, which is what this corrects.

import Pkg
# the repository root, one level up: that is where Project.toml/Manifest.toml live.
Pkg.activate(get(ENV, "FIG2_ENV", dirname(@__DIR__)); io = devnull)

using CSV, DataFrames, Statistics, Printf, Plots
Plots.gr()
default(fontfamily = "Computer Modern", framestyle = :box)

# Nominal sizes handed to GR.  Multiply by 1.318 for the on-page effective size
# (see the header note); every entry below therefore lands above 6 pt.  These
# must be Int -- Plots.Font converts pointsize with Int64() and throws on a
# fractional value.
const FS_TITLE = 7    # -> 9.2 pt
const FS_GUIDE = 6    # -> 7.9 pt
const FS_TICK  = 5    # -> 6.6 pt
const FS_ANNOT = 5    # -> 6.6 pt

argval(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i + 1])

OUT  = argval("--out", joinpath(@__DIR__, "replotted.pdf"))
XMAX = parse(Float64, argval("--xmax", "5200"))

srcs = [argval("--s$i", "") for i in 1:4]
kpuu = [argval("--k$i", "") for i in 1:4]

# short codes; the key is given in the caption
CODE = Dict(
    "kpuu"       => "k",
    "100nM_PgP"  => "P_1",
    "1uM_PgP"    => "P_2",
    "100nM_BCRP" => "B_1",
    "1uM_BCRP"   => "B_2",
    "mrt"        => "M",
)

# an empty Action_Set field means the plan measures nothing and terminates
code(::Missing) = "\$\\emptyset\$"

function code(s::AbstractString)
    toks = filter(!isempty, [strip(t) for t in split(replace(s, r"[\[\]\"]" => ""), ",")])
    isempty(toks) ? "\$\\emptyset\$" : "\$" * join([get(CODE, t, t) for t in toks], "\\,") * "\$"
end

panels = Plots.Plot[]

for i in 1:4
    df = DataFrame(CSV.File(srcs[i]))
    # share of the whole ensemble, not of the members that reached the level:
    # sum(Frequency) within a level collapses to as few as one at loose
    # tolerance, which rendered a plan backed by a single run at the same
    # saturation as a unanimous one. N_e is read off the tightest level, where
    # every member has a plan.
    Ne = sum(df.Frequency[df.Threshold .== minimum(df.Threshold)])
    df.share = df.Frequency ./ Ne
    @printf("panel %d: N_e = %d, share range %.3f-%.3f\n",
            i, Ne, minimum(df.share), maximum(df.share))

    p = plot(; xlims = (-200, XMAX), ylims = (-0.04, 1.04),
             xlabel = i in (3, 4) ? "Ensemble mean planner objective" : "",
             ylabel = i in (1, 3) ? "Tolerance \$\\epsilon\$" : "",
             xticks = (0:1000:5000, ["0", "1000", "2000", "3000", "4000", "5000"]),
             yticks = 0:0.2:1.0,
             title  = @sprintf("Compound %d  (\$k_{puu}=%.3f\$)", i, parse(Float64, kpuu[i])),
             titlefontsize = FS_TITLE, guidefontsize = FS_GUIDE, tickfontsize = FS_TICK,
             legendfontsize = FS_ANNOT, legend = (i == 1 ? :top : false),
             foreground_color_legend = :gray70, background_color_legend = :white)

    # in-range ensemble members
    keep = df.Average_Utility .<= XMAX
    scatter!(p, df.Average_Utility[keep], df.Threshold[keep];
             zcolor = df.share[keep], color = :Blues, clims = (0, 1),
             marker = (:circle, 3.5), markerstrokewidth = 0.3,
             markerstrokecolor = :gray40, colorbar = false, label = "")

    # maximum-likelihood action-set path: the modal action set at each tolerance
    mlas = combine(groupby(df, :Threshold)) do g
        j = argmax(g.Frequency)
        (Average_Utility = g.Average_Utility[j], Action_Set = g.Action_Set[j])
    end
    sort!(mlas, :Threshold)
    mx, my, ms = mlas.Average_Utility, mlas.Threshold, mlas.Action_Set

    xclamp = min.(mx, XMAX * 0.98)
    plot!(p, xclamp, my; line = (:dashdot, :red, 1.6), label = "MLASP")

    offpath = mx .> XMAX
    if any(offpath)
        scatter!(p, xclamp[offpath], my[offpath]; marker = (:rtriangle, 6, :red),
                 markerstrokewidth = 0, label = "")
    end
    if i == 1   # the shared key lives in this panel's empty upper region
        scatter!(p, [NaN], [NaN]; marker = (:rtriangle, 6, :red),
                 markerstrokewidth = 0, label = "MLASP beyond range")
    end

    # every off-scale ensemble member, path or not, is accounted for in text
    nbeyond = count(.!keep)
    if nbeyond > 0
        annotate!(p, 0.97 * XMAX, 0.965,
                  text(@sprintf("%d of %d points beyond range", nbeyond, nrow(df)),
                       FS_ANNOT, :right, :gray30))
        annotate!(p, 0.97 * XMAX, 0.915,
                  text(@sprintf("(max %.0f)", maximum(df.Average_Utility)),
                       FS_ANNOT, :right, :gray30))
    end

    cands = [(0.035, 0.04), (0.035, -0.04), (-0.035, 0.04), (-0.035, -0.04),
             (0.035, 0.09), (-0.035, 0.09), (0.035, -0.09), (-0.035, -0.09)]
    placed = Tuple{Float64,Float64}[]

    # an off-scale MLASP point keeps its exact objective, printed in the empty
    # lower-left corner rather than crowded against the right-hand edge
    for (r, j) in enumerate(findall(offpath))
        ay = 0.05 + 0.06 * (r - 1)
        annotate!(p, 0.03 * XMAX, ay,
                  text(@sprintf("\$\\epsilon=%.1f\$: ", my[j]) * code(ms[j]) *
                       @sprintf(" = %.0f", mx[j]), FS_ANNOT, :left, :red))
        push!(placed, (0.22 * XMAX, ay))   # reserve the strip this text occupies
    end

    # label the modal action set where it changes along the path
    changes = [j == 1 || !isequal(ms[j], ms[j - 1]) for j in eachindex(ms)]
    for j in eachindex(ms)
        (changes[j] && !offpath[j]) || continue
        xc, yy = xclamp[j], my[j]
        for (dx, dy) in cands
            ax, ay = xc + dx * XMAX, yy + dy
            (ay < 0.0 || ay > 0.93) && continue
            (ax < 0.0 || ax > XMAX) && continue
            if all(q -> abs(ax - q[1]) > 0.17 * XMAX || abs(ay - q[2]) > 0.055, placed)
                annotate!(p, ax, ay, text(code(ms[j]), FS_ANNOT, dx > 0 ? :left : :right, :black))
                push!(placed, (ax, ay))
                break
            end
        end
    end
    push!(panels, p)
end

# one shared colour key, drawn as a fifth subplot
cbar = heatmap([0.0], collect(0:0.02:1), reshape(collect(0:0.02:1), :, 1);
    color = :Blues, clims = (0, 1), colorbar = false, framestyle = :box,
    xticks = false, xlims = (-0.5, 0.5), ymirror = true,
    yticks = (0:0.25:1, ["0", "0.25", "0.5", "0.75", "1"]), tickfontsize = FS_TICK,
    ylabel = "ensemble vote share", guidefontsize = FS_GUIDE)

lay = @layout [Plots.grid(2, 2) key{0.05w}]
fig = plot(panels..., cbar; layout = lay, size = (470, 380), dpi = 300,
           left_margin = 1Plots.mm, right_margin = 1Plots.mm,
           top_margin = 0Plots.mm, bottom_margin = 1Plots.mm)
savefig(fig, OUT)
println("wrote $OUT")
