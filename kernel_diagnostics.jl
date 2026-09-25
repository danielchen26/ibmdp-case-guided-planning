#!/usr/bin/env julia
# Kernel-locality diagnostics for the similarity model: effective neighborhood size
# N_eff = (sum_i wtilde_i^2)^-1, the number of records carrying 95% of the weight mass,
# and the maximum single-record weight.  Evaluated under leave-one-out at each
# compound's initial QSAR-only evidence -- the same state the planner starts from.
#
# These are the figures quoted in the ESI's hyperparameter section.  Run at the
# coefficients actually used and, for contrast, at unit coefficients:
#
#   julia --project=. kernel_diagnostics.jl                 # lambda_w*lambda_k = 25 / 100
#   julia --project=. kernel_diagnostics.jl --unit          # lambda_w*lambda_k = 1
#   julia --project=. kernel_diagnostics.jl --gamma 0.5     # the synthetic-benchmark value
#   julia --project=. kernel_diagnostics.jl --ratio         # paired comparison across settings
#
# The second block additionally reports the noise-term prefactor 1 - 1/N_eff by depth and
# its spread across candidate batches, i.e. the diagnostics quoted where the ESI discusses
# the gap between the asymptotic consistency statement and fixed-bandwidth practice.
#
# Point it at a different cohort with --data path/to/file.csv (default: the surrogate).
using Pkg
Pkg.activate(joinpath(@__DIR__))
let _c = joinpath(@__DIR__, "src", "RIPK1.jl"), _d = joinpath(@__DIR__, "RIPK1.jl")
    include(isfile(_c) ? _c : _d)
end
using .RIPK1
using .RIPK1: RIPK1_CONFIG, preprocess_ripk1_data
using CEEDesigns.GenerativeDesigns: DistanceBased, Evidence, Variance, Exponential,
                                    QuadraticDistance
using DataFrames, CSV, Statistics, Printf

# The noise-term prefactor of the conditional-variance estimator, 1 - 1/N_eff.  The
# second block below reports how it varies with depth and across candidate batches;
# those are the figures quoted in the ESI's consistency discussion.
prefactor(wt) = 1 - sum(abs2, wt)

unit = "--unit" in ARGS
di = findfirst(==("--data"), ARGS)
datafile = di === nothing ? joinpath(@__DIR__, "data", "cns220_synthetic.csv") : ARGS[di + 1]
gi = findfirst(==("--gamma"), ARGS)
gamma = gi === nothing ? nothing : parse(Float64, ARGS[gi + 1])

# Effective per-feature coefficients.  Only the products lambda_w*lambda_k are
# identifiable, so "unit coefficients" means lambda_w*lambda_k = 1 throughout, and
# --gamma g sets that product to g uniformly (the synthetic benchmark runs at g = 0.5).
lw = RIPK1_CONFIG.sampler_params.lambda                       # 0.5
flat = unit ? 1.0 : gamma
lk_sil = flat === nothing ? RIPK1_CONFIG.sampler_params.in_silico_lambda : flat / lw  # 50
lk_phy = flat === nothing ? RIPK1_CONFIG.sampler_params.physical_lambda : flat / lw   # 200

in_silico = ["1uM_PgP_qsar", "100_nM_Mouse_BCRP_qsar", "qsar_mrt"]
physical = ["blood_frac_conc", "brain_conc", "brain_binding",
            "plasma_protein_binding", "kpuu", "100nM_PgP", "1uM_PgP", "100nM_BCRP"]
distances = Dict()
foreach(e -> push!(distances, e => QuadraticDistance(; λ = lk_sil)), in_silico)
foreach(e -> push!(distances, e => QuadraticDistance(; λ = lk_phy)), physical)

@printf("cohort: %s\n", datafile)
@printf("effective coefficients lambda_w*lambda_k: %.3g (in silico QSAR) / %.3g (physical)\n\n",
        lw * lk_sil, lw * lk_phy)

data_full = DataFrame(CSV.File(datafile))
data, _ = preprocess_ripk1_data(data_full)

neffs = Float64[]
n95s = Int[]
maxws = Float64[]
beliefs = Float64[]

for i in 1:nrow(data)
    row = data[i, :]
    pool = data[setdiff(1:nrow(data), i), :]                 # leave-one-out
    pool_for_sampler = select(pool, Not(:compound_id))
    (; weights) = DistanceBased(
        pool_for_sampler; target = RIPK1_CONFIG.sampler_params.target,
        uncertainty = Variance(), similarity = Exponential(; λ = lw),
        distance = distances)

    # Initial evidence: the three QSAR predictions available before any assay.
    ev = Evidence("qsar_mrt" => row["qsar_mrt"],
                  "1uM_PgP_qsar" => row["1uM_PgP_qsar"],
                  "100_nM_Mouse_BCRP_qsar" => row["100_nM_Mouse_BCRP_qsar"])
    w = weights(ev)
    wt = w ./ sum(w)                                          # normalized

    push!(neffs, 1 / sum(abs2, wt))
    push!(n95s, findfirst(>=(0.95), cumsum(sort(wt; rev = true))))
    push!(maxws, maximum(wt))
    push!(beliefs, sum(wt .* pool_for_sampler[!, RIPK1_CONFIG.sampler_params.target]))
end

@printf("records available under leave-one-out : %d\n", nrow(data) - 1)
@printf("median N_eff                          : %.1f\n", median(neffs))
@printf("median # records holding 95%% of mass  : %d\n", round(Int, median(n95s)))
@printf("median max single-record weight        : %.4f\n", median(maxws))
@printf("across-compound sd of belief mean of %s : %.3f\n",
        RIPK1_CONFIG.sampler_params.target, std(beliefs))
@printf("marginal cohort sd of %s               : %.3f\n",
        RIPK1_CONFIG.sampler_params.target, std(data[!, RIPK1_CONFIG.sampler_params.target]))

# ---------------------------------------------------------------------------------
# Prefactor diagnostics.  The estimator's noise term carries the state-dependent
# factor 1 - 1/N_eff(s).  Two things matter for how H(s) may be read: how it changes
# with depth (measuring proxies contracts the neighborhood mechanically), and how much
# it varies across the actions being ranked at a single state (if it varied little it
# would cancel in the comparison -- it does not).
qs = ["qsar_mrt", "1uM_PgP_qsar", "100_nM_Mouse_BCRP_qsar"]
proxies = ["1uM_PgP", "100nM_PgP", "100nM_BCRP"]
DEPTHS = [String[], proxies[1:1], proxies[1:2], proxies]
CANDIDATES = [[p] for p in proxies]                       # singletons
append!(CANDIDATES, [[proxies[1], proxies[2]], [proxies[1], proxies[3]],
                     [proxies[2], proxies[3]], proxies])  # pairs and the triple

byDepth = [Float64[] for _ in DEPTHS]          # prefactor at each depth
neffDepth = [Float64[] for _ in DEPTHS]
hOverPf = [Float64[] for _ in DEPTHS]          # H(s) divided by the prefactor
hRoot = Float64[]
pfRoot = Float64[]
spanRatio = Float64[]                          # within-compound max/min over candidates
spanMin = Float64[]
spanMax = Float64[]
argminMoves = Ref(0)                           # prefactor changes the best batch
pairFlips = Ref(0)
pairTotal = Ref(0)
degenerate = [0 for _ in DEPTHS]               # states whose weight collapses onto 1 record

for i in 1:nrow(data)
    row = data[i, :]
    pool_for_sampler = select(data[setdiff(1:nrow(data), i), :], Not(:compound_id))
    (; uncertainty, weights) = DistanceBased(
        pool_for_sampler; target = RIPK1_CONFIG.sampler_params.target,
        uncertainty = Variance(), similarity = Exponential(; λ = lw),
        distance = distances)

    # prefactor and uncertainty at an evidence state = QSAR predictions + `acquired`
    function at(acquired)
        ev = Evidence(vcat([q => row[q] for q in qs],
                           [a => row[a] for a in acquired])...)
        w = weights(ev)
        s = sum(w)
        s <= 0 && return (nothing, nothing)
        (prefactor(w ./ s), uncertainty(ev))
    end

    for (k, acq) in enumerate(DEPTHS)
        pf, h = at(acq)
        pf === nothing && continue
        # pf == 0 means all weight sits on a single record (N_eff = 1).  These states are
        # real and must be counted in the depth medians -- dropping them is what makes the
        # deeper-depth N_eff look larger than it is -- but H/pf is undefined for them.
        push!(byDepth[k], pf)
        push!(neffDepth[k], pf >= 1 ? Inf : 1 / (1 - pf))
        pf <= 1e-9 ? (degenerate[k] += 1) : push!(hOverPf[k], h / pf)
        if k == 1 && pf > 1e-9
            push!(hRoot, h)
            push!(pfRoot, pf)
        end
    end

    pfs = Float64[]
    hs = Float64[]
    ok = true
    for acq in CANDIDATES
        pf, h = at(acq)
        if pf === nothing || pf <= 1e-9
            ok = false
            break
        end
        push!(pfs, pf)
        push!(hs, h)
    end
    if ok
        push!(spanRatio, maximum(pfs) / minimum(pfs))
        push!(spanMin, minimum(pfs))
        push!(spanMax, maximum(pfs))
        # Does dividing out the prefactor change the ranking of candidate batches?
        argminMoves[] += argmin(hs) != argmin(hs ./ pfs)
        for a in 1:length(hs), b in (a + 1):length(hs)
            pairTotal[] += 1
            pairFlips[] += (hs[a] < hs[b]) != ((hs[a] / pfs[a]) < (hs[b] / pfs[b]))
        end
    end
end

fmt(v) = join([@sprintf("%.2f", median(x)) for x in v], ", ")
println()
@printf("prefactor 1-1/N_eff by depth (0..3 proxies measured) : %s\n", fmt(byDepth))
@printf("median N_eff by depth                                : %s\n", fmt(neffDepth))
@printf("  compounds with N_eff == 1 (all weight on 1 record) : %s\n",
        join(string.(degenerate), ", "))
@printf("H / prefactor by depth (flat => H tracks the factor) : %s\n", fmt(hOverPf))
@printf("cor(H, prefactor) at the QSAR-only root              : %.2f\n", cor(hRoot, pfRoot))
@printf("within-compound prefactor span over %d candidate batches:\n", length(CANDIDATES))
@printf("    median max/min ratio %.1f   median min %.2f   median max %.2f   share >2x %.0f%%\n",
        median(spanRatio), median(spanMin), median(spanMax), 100mean(spanRatio .> 2))
@printf("    dividing out the prefactor moves the best batch for %.0f%% of compounds\n",
        100argminMoves[] / length(spanRatio))
@printf("    and flips %.0f%% of pairwise batch orderings (%d of %d pairs)\n",
        100pairFlips[] / pairTotal[], pairFlips[], pairTotal[])

# ---------------------------------------------------------------------------------
# Paired coefficient comparison (--ratio).  The ESI quotes two different ratios between
# coefficient settings: the ratio of the per-cohort median N_eff, and the median of the
# per-compound ratios.  The second needs both settings evaluated on the *same* compound
# in one pass, which separate single-setting runs cannot supply, so it is computed here.
# This block ignores --unit/--gamma and always evaluates all three settings.
if "--ratio" in ARGS
    settings = [("study (25/100)", RIPK1_CONFIG.sampler_params.in_silico_lambda,
                                   RIPK1_CONFIG.sampler_params.physical_lambda),
                ("kappa = 0.5",    0.5 / lw, 0.5 / lw),
                ("kappa = 1",      1.0 / lw, 1.0 / lw)]
    dists = map(settings) do (_, ks, kp)
        d = Dict()
        foreach(e -> push!(d, e => QuadraticDistance(; λ = ks)), in_silico)
        foreach(e -> push!(d, e => QuadraticDistance(; λ = kp)), physical)
        d
    end
    paired = [Float64[] for _ in settings]
    for i in 1:nrow(data)
        row = data[i, :]
        pool = select(data[setdiff(1:nrow(data), i), :], Not(:compound_id))
        ev = Evidence([q => row[q] for q in qs]...)
        for (k, dist) in enumerate(dists)
            (; weights) = DistanceBased(
                pool; target = RIPK1_CONFIG.sampler_params.target,
                uncertainty = Variance(), similarity = Exponential(; λ = lw),
                distance = dist)
            w = weights(ev)
            push!(paired[k], 1 / sum(abs2, w ./ sum(w)))
        end
    end
    println()
    for (k, (name, _, _)) in enumerate(settings)
        @printf("median N_eff at %-15s : %8.2f\n", name, median(paired[k]))
    end
    for k in 2:length(settings)
        @printf("%s vs study: ratio of medians %.1fx, median of per-compound ratios %.1fx\n",
                settings[k][1], median(paired[k]) / median(paired[1]),
                median(paired[k] ./ paired[1]))
    end
end
