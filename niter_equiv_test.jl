#!/usr/bin/env julia
# Empirical test: does n_iter=2000 give the SAME MLASP as n_iter=20000?
# Uses the SYNTHETIC CNS surrogate (safe to report). Writes /tmp/niter_equiv.csv incrementally.
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

const OUT = get(ENV, "NITER_OUT", "/tmp/niter_equiv.csv")
open(OUT, "w") do io
    println(io, "row,compound_id,n_iter,ens,seed,eps,mlasp,cost,freq,elapsed_s")
end
logrow(s) = (open(OUT,"a") do io; println(io,s); flush(io); end)

data_full = DataFrame(CSV.File(joinpath(@__DIR__, "data", "cns220_synthetic.csv")))
data, compound_ids = preprocess_ripk1_data(data_full)
distances = setup_ripk1_distances()
assay_cost = Dict(k => v[1] for (k,v) in RIPK1_CONFIG.experiments)
cost_of(s) = (t=0.0; for k in keys(assay_cost); occursin(k,s) && (t+=assay_cost[k]); end; t)
tau = 0.9

# Test compounds: spread across the cohort
TEST_ROWS = [1, 25, 50, 75, 100, 150, 200]
CONFIGS = [(2000,10), (20000,10)]

# Optional sharding.  Each (compound, budget) job re-seeds with `1000+i` immediately
# before it runs and shares no state with any other job, so splitting the loop across
# processes reproduces the single-process results exactly.  Set NITER_ROWS and/or
# NITER_BUDGETS (comma-separated) plus a distinct NITER_OUT per shard, then
# concatenate the shard files.  Unset = run everything in one process.
parse_list(v) = [parse(Int, strip(x)) for x in split(v, ",") if !isempty(strip(x))]
haskey(ENV, "NITER_ROWS")    && (TEST_ROWS = parse_list(ENV["NITER_ROWS"]))
haskey(ENV, "NITER_BUDGETS") && (CONFIGS = [c for c in CONFIGS if c[1] in parse_list(ENV["NITER_BUDGETS"])])

for i in TEST_ROWS
    i > nrow(data) && continue
    row = data[i,:]; cid = row.compound_id
    pool = data[setdiff(1:nrow(data), i), :]
    pool_for_sampler = select(pool, Not(:compound_id))
    (; sampler, uncertainty, weights) = DistanceBased(
        pool_for_sampler; target = RIPK1_CONFIG.sampler_params.target, uncertainty = Variance(),
        similarity = Exponential(; λ = RIPK1_CONFIG.sampler_params.lambda), distance = distances)
    state_init = Evidence(
        "qsar_mrt" => row["qsar_mrt"], "1uM_PgP_qsar" => row["1uM_PgP_qsar"],
        "100_nM_Mouse_BCRP_qsar" => row["100_nM_Mouse_BCRP_qsar"])
    for (ni, ens) in CONFIGS
        seed = 1000 + i
        Random.seed!(seed)
        solver = DPWSolver(; n_iterations = ni,
                             exploration_constant = RIPK1_CONFIG.solver_params.exploration_constant,
                             depth = RIPK1_CONFIG.solver_params.depth,
                             tree_in_info=false, keep_tree=false)
        t0 = time()
        try
            ens_res = perform_ensemble_designs(
                RIPK1_CONFIG.experiments; sampler=sampler, uncertainty=uncertainty,
                thresholds=RIPK1_CONFIG.threshold_NO, evidence=state_init, weights=weights,
                data=pool_for_sampler,
                terminal_condition=(RIPK1_CONFIG.target_condition, RIPK1_CONFIG.conditional_weights_thred),
                realized_uncertainty=true, solver=solver, repetitions=0,
                mdp_options=(conditional_constraints_enabled=true,
                             max_parallel=RIPK1_CONFIG.parallel_assays_NO,
                             costs_tradeoff=RIPK1_CONFIG.money_biased),
                N=ens, thred_set=[tau])
            df,_,_ = process_ensemble_results_enhanced(ens_res, tau; save_dir="", save_plots=false)
            el = round(time()-t0, digits=1)
            if nrow(df) > 0
                tops = find_top_n_action_sets_with_utility(df,1); sort!(tops,:Threshold)
                for r in eachrow(tops)
                    a = replace(string(r.Top_1_Action_Set), ","=>";")
                    logrow("$i,$cid,$ni,$ens,$seed,$(r.Threshold),$a,$(cost_of(a)),$(r.Top_1_Frequency),$el")
                end
            else
                logrow("$i,$cid,$ni,$ens,$seed,NA,<none>,NaN,0,$el")
            end
        catch e
            logrow("$i,$cid,$ni,$ens,$seed,NA,ERRORED,NaN,0,-1"); println("ERR row=$i ni=$ni: ", e)
        end
        println("done row=$i n_iter=$ni ens=$ens"); flush(stdout)
    end
end
println("ALL_DONE"); flush(stdout)
