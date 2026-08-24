#!/usr/bin/env julia
# Recompute Table 1 from route1_results/route1_summary.csv without re-running the
# benchmark.  route1_benchmark.jl already prints these figures at the end of its own
# run; this script exists so a reader can verify Table 1 from the shipped CSV alone.
#
#   julia --project=. benchmark/score_table1.jl [path/to/route1_summary.csv]
using CSV, DataFrames, Printf

path = length(ARGS) >= 1 ? ARGS[1] :
       joinpath(@__DIR__, "route1_results", "route1_summary.csv")
rows = CSV.read(path, DataFrame)
n = nrow(rows)
pct(c) = n == 0 ? 0.0 : 100c / n

@printf("trials: %d\n\n", n)
@printf("IBMDP Top-1 membership  : %d/%d = %.1f%%\n", sum(rows.T1_member), n, pct(sum(rows.T1_member)))
# T2 is CUMULATIVE: membership in the union of Top-1 and Top-2, not in Top-2 alone.
@printf("IBMDP Top-1 u Top-2     : %d/%d = %.1f%%  (cumulative, not a second independent measurement)\n",
        sum(rows.T2_member), n, pct(sum(rows.T2_member)))
@printf("VI-Sim  membership      : %d/%d = %.1f%%\n", sum(rows.sim_member), n, pct(sum(rows.sim_member)))

# Paired McNemar with continuity correction: IBMDP Top-1 vs VI-Sim, both scored by
# the same set-membership rule.
b = sum((rows.T1_member .== 1) .& (rows.sim_member .== 0))
c = sum((rows.T1_member .== 0) .& (rows.sim_member .== 1))
@printf("\nPaired McNemar (IBMDP Top-1 vs VI-Sim): b=%d, c=%d\n", b, c)
if b + c > 0
    @printf("  chi2_cc = %.2f  (1 df)\n", (abs(b - c) - 1)^2 / (b + c))
end

# VI-Theo's optimum must genuinely move across trials, else the comparison is vacuous.
counts = Dict{Int,Int}()
for f in rows.theo_feat
    counts[f] = get(counts, f, 0) + 1
end
print("\nVI-Theo optimal feature distribution: ")
println(join(["f$k=$(counts[k])" for k in sort(collect(keys(counts)); by = k -> -counts[k])], ", "))
