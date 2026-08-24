#!/usr/bin/env julia
"""
Route-1 benchmark: NON-DEGENERATE synthetic comparison of VI-Theo / VI-Sim / IBMDP.

Fixes the degeneracy of the original standalone benchmark, where every iteration reused
the SAME frozen dataset (seed=42) so VI-Theo's optimum never moved and IBMDP's Top-1 was
the constant batch {3,4}. Here, EACH iteration:
  1. draws a genuinely different generative config (per-iteration seed + random beta),
  2. feeds the SAME dataset to BOTH the VI side and the IBMDP/CEED ensemble side,
  3. records VI-Theo, VI-Sim (single actions) and IBMDP Top-1 / Top-2 (batches),
  4. is scored SYMMETRICALLY (set-membership of VI-Theo's optimum in each method's set).

We reuse the real src/ modules (no reimplementation): generate_historical_dataset,
simulate_experiments (VI-Theo/VI-Sim), and perform_ensemble_designs (IBMDP).
Run from repo root:  julia --project=. benchmark/route1_benchmark.jl [n_iterations] [N_ensemble]
"""

using Pkg
if isdir("src"); Pkg.activate("."); elseif isdir("../src"); cd(".."); Pkg.activate("."); else error("run from repo root"); end

using Random, Statistics, StatsBase, Distributions
using StatsBase: countmap
using DataFrames, CSV, Dates
using Plots
using MCTS
using MCTS: DPWSolver
using CEEDesigns
using CEEDesigns.GenerativeDesigns
using CEEDesigns.GenerativeDesigns: Exponential, Variance, Evidence, QuadraticDistance, DistanceBased
using CEEDesigns.ValueIteration
using CEEDesigns.CEEDUtilities

# ---------- knobs ----------
n_iterations = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
N_ENSEMBLE   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 30
n_historical = 200
n_features   = 6
feature_costs = [1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 10.0]   # incl. target cost
uncertainty_threshold = 0.1
results_dir = "benchmark/route1_results"
mkpath(results_dir)

@info "ROUTE-1 benchmark: $n_iterations iterations, ensemble N=$N_ENSEMBLE"

# helper: extract integer feature indices. Handles BOTH the CEED form
# ["feature_3","feature_4"] AND the VI form [5] / [3, 4] (bare integers).
function feat_nums(x::AbstractString)
    if occursin("feature_", x)
        return [parse(Int, m.captures[1]) for m in eachmatch(r"feature_(\d+)", x)]
    else
        return [parse(Int, m.match) for m in eachmatch(r"\d+", x)]
    end
end
feat_nums(x::AbstractVector) = reduce(vcat, [feat_nums(string(e)) for e in x]; init=Int[])
feat_nums(::Missing) = Int[]

rows = DataFrame(iter=Int[], beta=String[],
                 vi_theo=String[], vi_sim=String[],
                 ceed_top1=String[], ceed_top2=String[],
                 theo_feat=Int[],
                 T1_member=Int[], T2_member=Int[],
                 sim_member=Int[], sim_equal=Int[])

for i in 1:n_iterations
    println("\n" * "="^60); println("Route-1 iteration $i / $n_iterations"); println("="^60)
    iter_seed = 1000 + i

    # ---- per-iteration random beta (normalized), the dominant lever on VI-Theo's optimum ----
    Random.seed!(iter_seed + 7)
    raw = rand(n_features)
    beta = raw ./ sum(raw)

    # ---- generate ONE dataset for this iteration (shared by VI and CEED) ----
    hist, tcoef, fstats = generate_historical_dataset(n_historical, n_features, iter_seed, false;
                                                       custom_coeffs = beta)
    # write CSVs so the CEED config path loads exactly this dataset
    data_dir = joinpath(results_dir, "data_iter_$i")
    mkpath(data_dir)
    istate = generate_initial_state(; n_features=n_features, seed=iter_seed)
    save_synthetic_data(hist, tcoef, fstats, istate; data_dir=data_dir)

    shared = (historical_data = hist, target_coeffs = tcoef, feature_stats = fstats)

    # ---- VI side (both flavors) on the shared dataset ----
    _, policy_cmp = value_iteration_analysis(; seed=iter_seed, shared_data=shared)
    vi_sim  = policy_cmp.VI_sim[1]
    vi_theo = policy_cmp.VI_theo[1]
    theo_feats = feat_nums(vi_theo)
    theo_feat  = isempty(theo_feats) ? -1 : theo_feats[1]

    # ---- IBMDP / CEED ensemble on the SAME dataset ----
    ceed_top1 = ""; ceed_top2 = ""
    try
        config = setup_ceed_configuration(data_dir=data_dir, generate_if_missing=false)
        money_biased, _ = cost_bias_tuple()
        # ROUTE-1 FIX: UNIFORM experiment costs so the reachable-band batch is chosen on
        # informativeness, not on the cheapest-experiment tiebreak. Features 3..6 all cost 1.0.
        experiments_uniform = create_experiments_dict(; feature_range=3:6, base_cost=1.0, cost_increment=0.0)
        sampler_setup = setup_enhanced_sampler(config.data.historical_data;
                            target="target", lambda=0.5, conditional_range=Dict())
        taus = [0.9]
        ens = perform_ensemble_designs(experiments_uniform;
                sampler = sampler_setup.sampler,
                uncertainty = sampler_setup.uncertainty,
                thresholds = config.threshold_NO,
                evidence = config.ceed_initial_state,
                weights = sampler_setup.weights,
                data = config.data.historical_data,
                terminal_condition = (Dict{String,Vector{Float64}}(), 0.0),
                realized_uncertainty = true,
                solver = config.solver,
                repetitions = 0,
                mdp_options = (max_parallel = config.parallel_assays_NO, costs_tradeoff = money_biased),
                N = N_ENSEMBLE,
                thred_set = taus)
        df_09, _, _ = process_ensemble_results_for_belief(ens, 0.9; ensemble_folder = data_dir)
        tops = find_top_n_action_sets_with_utility(df_09, 3) |> x -> sort(x, :Threshold, rev=true)
        # ROUTE-1 FIX: read IBMDP's recommendation at a REACHABLE uncertainty threshold.
        # Designs whose terminal condition is unreachable hit the bigM penalty (|utility|~1e6),
        # collapsing to the cheapest batch as a cost tiebreak. Only reachable-band designs
        # (finite Q, |Average_Utility| < 1e5) reflect genuine informativeness-driven choices.
        reachable = tops[ (.!ismissing.(tops.Top_1_Average_Utility)) .& (abs.(coalesce.(tops.Top_1_Average_Utility, 1e9)) .< 1e5), :]
        sel = isempty(reachable) ? tops[tops.Threshold .== maximum(tops.Threshold), :] :
                                   reachable[reachable.Threshold .== minimum(reachable.Threshold), :]
        if nrow(sel) >= 1
            ceed_top1 = string(sel[1, :Top_1_Action_Set])
            if "Top_2_Action_Set" in names(sel); ceed_top2 = string(sel[1, :Top_2_Action_Set]); end
            println("  [reachable band] read at Threshold=$(sel[1,:Threshold]) utility=$(sel[1,:Top_1_Average_Utility])")
        end
    catch e
        @warn "CEED ensemble failed on iter $i: $e"
    end

    t1 = feat_nums(ceed_top1); t2 = feat_nums(ceed_top2); sm = feat_nums(vi_sim)
    T1_member  = (theo_feat in t1) ? 1 : 0
    T2_member  = (theo_feat in vcat(t1, t2)) ? 1 : 0     # Top-2 = union of top1 & top2 batches
    sim_member = (theo_feat in sm) ? 1 : 0               # symmetric membership for VI-Sim
    sim_equal  = (Set(sm) == Set(theo_feats)) ? 1 : 0    # original strict-equality score

    push!(rows, (i, string(round.(beta, digits=3)), string(vi_theo), string(vi_sim),
                 ceed_top1, ceed_top2, theo_feat, T1_member, T2_member, sim_member, sim_equal))
    println("  beta=$(round.(beta,digits=3))")
    println("  VI-Theo=$vi_theo  VI-Sim=$vi_sim  CEED Top1=$ceed_top1  Top2=$ceed_top2")
    println("  T1_member=$T1_member T2_member=$T2_member sim_member=$sim_member sim_equal=$sim_equal")
end

CSV.write(joinpath(results_dir, "route1_summary.csv"), rows)

n = nrow(rows)
println("\n" * "#"^60)
println("ROUTE-1 RESULTS  (n=$n iterations)")
println("#"^60)
println("VI-Theo optimum feature distribution: ", sort(collect(countmap(rows.theo_feat)); by=p->-p[2]))
println("# distinct CEED Top-1 batches        : ", length(unique(rows.ceed_top1)), "  (1 = constant/degenerate)")
println("CEED Top-1 batch distribution        : ", sort(collect(countmap(rows.ceed_top1)); by=p->-p[2]))
println()
pct(c) = n == 0 ? 0.0 : 100*c/n
println("IBMDP Top-1 membership  : $(sum(rows.T1_member))/$n = $(round(pct(sum(rows.T1_member)),digits=1))%")
println("IBMDP Top-2 membership  : $(sum(rows.T2_member))/$n = $(round(pct(sum(rows.T2_member)),digits=1))%")
println("VI-Sim  membership      : $(sum(rows.sim_member))/$n = $(round(pct(sum(rows.sim_member)),digits=1))%  (symmetric)")
println("VI-Sim  strict-equality : $(sum(rows.sim_equal))/$n = $(round(pct(sum(rows.sim_equal)),digits=1))%  (original rule)")

# paired McNemar: IBMDP Top-1 membership vs VI-Sim membership (symmetric, apples-to-apples)
b = sum((rows.T1_member .== 1) .& (rows.sim_member .== 0))
c = sum((rows.T1_member .== 0) .& (rows.sim_member .== 1))
println("\nPaired McNemar (IBMDP Top-1 vs VI-Sim, both membership): b=$b, c=$c")
if b + c > 0
    chi2 = (abs(b - c) - 1)^2 / (b + c)
    println("  chi2_cc = $(round(chi2, digits=3))  (approx p from chi2_1df)")
end
println("\nSaved: $(joinpath(results_dir, "route1_summary.csv"))")
