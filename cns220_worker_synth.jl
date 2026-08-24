#!/usr/bin/env julia
# CNS-220 population worker (PARETO version): processes a shard of compounds [lo..hi].
# For EACH compound, emits the FULL per-uncertainty-tolerance (epsilon) curve:
# one CSV row per (compound, epsilon-level), giving the IBMDP-recommended assay batch
# and its cost at that uncertainty tolerance. This captures the cost-vs-uncertainty
# Pareto front: tight epsilon -> measure kpuu directly ($4000); loose epsilon -> cheap
# proxies (PgP/BCRP) when they suffice.
# Settings: n_iter=2000 (~10x faster than 20000). NOTE: this is NOT MLASP-identical
#   to n_iter=20000. Measured on the synthetic surrogate (niter_equiv_test.jl, 7
#   compounds x 11 eps, 58 paired cells): exact MLASP agrees 42/58, cost 52/58,
#   kpuu-in-batch 57/58; the band-pooled median first-batch cost ($4000 tight /
#   $800 loose) is unchanged. Of the 16 disagreements, 10 are permutations among
#   equal-cost proxies and 5 differ by a single $400 proxy (4 of those 5 nested);
#   the 16th changes the decision (SYN-0100 at eps=0.1: two proxies, $800, vs
#   kpuu alone, $4000 -- the REDUCED budget is the one that declines kpuu).
#   Reproduce with `python3 niter_equiv_report.py`.
# Also: per-compound seed=1000+row (reproducible + parallel-safe), leave-one-out sampler,
#   save_plots=false. Streams CSV, flushed.
# Usage: julia --project=RIPK1_Enhanced cns220_worker.jl <lo> <hi> <out_csv> [n_iter] [ensemble]
#                                                       [--data path/to/cohort.csv]
# The cohort defaults to the synthetic surrogate shipped here.  Point --data at the
# real CNS cohort (MSD publishes it as CNS_example/data/multitier.csv in
# MSDLLCpapers/IBMDPDesigns.jl) to recompute the paper's Figure 3 numbers.  That
# file has no compound_id column; preprocess_ripk1_data then synthesizes
# placeholder ids, which affects nothing -- ids never enter the distance or the
# belief.  NOTE the output CSV carries a kpuu column, so when run on the real
# cohort the output contains real measured values: keep it out of any public repo.

using Pkg
Pkg.activate(joinpath(@__DIR__))
let _c = joinpath(@__DIR__, "src", "RIPK1.jl"), _d = joinpath(@__DIR__, "RIPK1.jl")
    include(isfile(_c) ? _c : _d)
end
using .RIPK1
using .RIPK1: RIPK1_CONFIG, perform_ensemble_designs, process_ensemble_results_enhanced,
              preprocess_ripk1_data, setup_ripk1_distances
using CEEDesigns.GenerativeDesigns: DistanceBased, Evidence, Variance, Exponential,
                                    find_top_n_action_sets_with_utility
using Random, DataFrames, CSV, Printf
using MCTS: DPWSolver

di       = findfirst(==("--data"), ARGS)
datafile = di === nothing ? joinpath(@__DIR__, "data", "cns220_synthetic.csv") : ARGS[di + 1]
pos      = di === nothing ? ARGS : ARGS[setdiff(1:length(ARGS), (di, di + 1))]

lo      = parse(Int, pos[1])
hi      = parse(Int, pos[2])
out_csv = pos[3]
n_iter  = length(pos) >= 4 ? parse(Int, pos[4]) : 2000
ens     = length(pos) >= 5 ? parse(Int, pos[5]) : 10
tau     = 0.9

@info "cohort" datafile
data_full = DataFrame(CSV.File(datafile))
data, compound_ids = preprocess_ripk1_data(data_full)
distances = setup_ripk1_distances()

assay_cost = Dict(k => v[1] for (k,v) in RIPK1_CONFIG.experiments)
function cost_of(s::AbstractString)
    tot = 0.0
    for k in keys(assay_cost); occursin(k, s) && (tot += assay_cost[k]); end
    tot
end
uses_kpuu(s::AbstractString) = occursin("kpuu", s)
heuristic_class(pgp, bcrp) = (pgp < 2 && bcrp < 2) ? "promising" :
                             (pgp > 4 || bcrp > 4) ? "non_promising" : "borderline"

open(out_csv, "w") do io
    println(io, "row,compound_id,kpuu,true_label,heuristic_class,eps,mlasp,cost,uses_kpuu,freq,seed,n_iter")
end
logrow(s) = (open(out_csv, "a") do io; println(io, s); flush(io); end)

for i in lo:hi
    i > nrow(data) && break
    row = data[i, :]; cid = row.compound_id; kpuu = row.kpuu
    label = kpuu > 0.5 ? 1 : 0
    hcls = heuristic_class(row["1uM_PgP_qsar"], row["100_nM_Mouse_BCRP_qsar"])

    pool = data[setdiff(1:nrow(data), i), :]
    pool_for_sampler = select(pool, Not(:compound_id))
    (; sampler, uncertainty, weights) = DistanceBased(
        pool_for_sampler; target = RIPK1_CONFIG.sampler_params.target, uncertainty = Variance(),
        similarity = Exponential(; λ = RIPK1_CONFIG.sampler_params.lambda), distance = distances)
    state_init = Evidence(
        "qsar_mrt" => row["qsar_mrt"], "1uM_PgP_qsar" => row["1uM_PgP_qsar"],
        "100_nM_Mouse_BCRP_qsar" => row["100_nM_Mouse_BCRP_qsar"])

    seed = 1000 + i
    Random.seed!(seed)
    solver = DPWSolver(; n_iterations = n_iter,
                         exploration_constant = RIPK1_CONFIG.solver_params.exploration_constant,
                         depth = RIPK1_CONFIG.solver_params.depth, tree_in_info = false, keep_tree = false)
    try
        ens_res = perform_ensemble_designs(
            RIPK1_CONFIG.experiments; sampler = sampler, uncertainty = uncertainty,
            thresholds = RIPK1_CONFIG.threshold_NO, evidence = state_init, weights = weights,
            data = pool_for_sampler,
            terminal_condition = (RIPK1_CONFIG.target_condition, RIPK1_CONFIG.conditional_weights_thred),
            realized_uncertainty = true, solver = solver, repetitions = 0,
            mdp_options = (conditional_constraints_enabled = true,
                           max_parallel = RIPK1_CONFIG.parallel_assays_NO,
                           costs_tradeoff = RIPK1_CONFIG.money_biased),
            N = ens, thred_set = [tau])
        df, _, _ = process_ensemble_results_enhanced(ens_res, tau; save_dir = "", save_plots = false)
        if nrow(df) > 0
            tops = find_top_n_action_sets_with_utility(df, 1)
            sort!(tops, :Threshold)
            for r in eachrow(tops)   # ONE ROW PER epsilon LEVEL -> full Pareto curve
                a = replace(string(r.Top_1_Action_Set), ","=>";")
                logrow("$i,$cid,$(round(kpuu,digits=4)),$label,$hcls,$(r.Threshold),$a,$(cost_of(a)),$(uses_kpuu(a)),$(r.Top_1_Frequency),$seed,$n_iter")
            end
        else
            logrow("$i,$cid,$(round(kpuu,digits=4)),$label,$hcls,NA,<none>,NaN,false,0,$seed,$n_iter")
        end
    catch e
        logrow("$i,$cid,$(round(kpuu,digits=4)),$label,$hcls,NA,ERROR:$(replace(string(e),","=>";")[1:min(40,end)]),NaN,false,0,$seed,$n_iter")
    end
    println("row $i / $hi done"); flush(stdout)
end
println("SHARD_DONE $lo-$hi"); flush(stdout)
