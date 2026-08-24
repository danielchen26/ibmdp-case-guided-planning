module GenerativeDesigns

using POMDPs
using POMDPTools

using Combinatorics
using DataFrames, ScientificTypes
using LinearAlgebra
using Statistics
using StatsBase: Weights, countmap, entropy, sample
using Random: default_rng, AbstractRNG
using MCTS

using ..CEEDesigns: front

export UncertaintyReductionMDP, DistanceBased
export QuadraticDistance, DiscreteDistance, SquaredMahalanobisDistance
export Exponential
export Variance, Entropy
export Evidence, State
export efficient_design, efficient_designs
export efficient_value
export conditional_efficient_design, conditional_efficient_designs, perform_ensemble_designs
export conditional_likelihood, analyze_ensemble_results_with_utility_and_frequency_to_df
export find_max_likelihood_action_sets_with_utility, find_top_n_action_sets_with_utility
export process_ensemble_results_for_belief, process_ensemble_results_enhanced
export plot_ensemble_histogram, plot_ensemble_pareto, plot_multiple_max_likelihood_action_sets, create_action_set_legend, create_multi_threshold_legend, validate_ensemble_frequencies

"""
Represent experimental evidence as an immutable dictionary.
"""
const Evidence = Base.ImmutableDict{String,Any}

function Base.merge(d::Evidence, dsrc::Dict)
    for (k, v) in dsrc
        # ! FIXED: Ensure all keys are strings to prevent numeric key issues
        string_key = string(k)
        d = Evidence(d, string_key, v)
    end

    return d
end

Evidence(p1::Pair, pairs::Pair...) = merge(Evidence(), Dict(p1, pairs...))

"""
Represent experimental state as a tuple of experimental costs and evidence.
"""
const State = NamedTuple{(:evidence, :costs),Tuple{Evidence,NTuple{2,Float64}}}

function Base.merge(state::State, evidence, costs)
    # ! FIXED: Ensure evidence is properly handled as a dictionary with string keys
    if evidence isa Dict
        # Convert any non-string keys to strings
        string_evidence = Dict{String, Any}(string(k) => v for (k, v) in evidence)
        return State((merge(state.evidence, string_evidence), state.costs .+ costs))
    else
        return State((merge(state.evidence, evidence), state.costs .+ costs))
    end
end

include("distancebased.jl")

"""
Represent action as a named tuple `(; costs=(monetary cost, time), features)`.
"""
const ActionCost = NamedTuple{(:costs, :features),<:Tuple{NTuple{2,Float64},Vector{String}}}

const const_bigM = 1_000_000

const default_solver = DPWSolver(; n_iterations = 100_000, tree_in_info = true)

# minimize the expected experimental cost while ensuring the uncertainty remains below a specified threshold.
include("UncertaintyReductionMDP.jl")

# maximize the value of the experimental evidence (such as clinical utility), adjusted for experimental costs.
include("EfficientValueMDP.jl")

# ! NEW: Include ensemble analysis utilities for conditional CEED
include("ensemble_analysis.jl")

# ! NEW: Include advanced workflows and specialized visualization
include("CEEDWorkflows.jl")
using .CEEDWorkflows: ensemble_Vis, CEED_single_state_init, CEED_multiple_state_init_run

# Export advanced workflow functions
export ensemble_Vis, CEED_single_state_init, CEED_multiple_state_init_run

end
