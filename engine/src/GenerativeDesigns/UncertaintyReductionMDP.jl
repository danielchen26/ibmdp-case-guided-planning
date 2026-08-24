"""
    UncertaintyReductionMDP(costs; sampler, uncertainty, threshold, evidence=Evidence(), <keyword arguments>)

Structure that parametrizes the experimental decision-making process. It is used in the object interface of POMDPs.

In this experimental setup, our objective is to minimize the expected experimental cost while ensuring the uncertainty remains below a specified threshold.

Internally, a state of the decision process is modeled as a tuple `(evidence::Evidence, [total accumulated monetary cost, total accumulated execution time])`.

# Arguments

  - `costs`: a dictionary containing pairs `experiment => cost`, where `cost` can either be a scalar cost (modelled as a monetary cost) or a tuple `(monetary cost, execution time)`.

# Keyword Arguments

  - `sampler`: a function of `(evidence, features, rng)`, in which `evidence` denotes the current experimental evidence, `features` represent the set of features we want to sample from, and `rng` is a random number generator; it returns a dictionary mapping the features to outcomes.
  - `uncertainty`: a function of `evidence`; it returns the measure of variance or uncertainty about the target variable, conditioned on the experimental evidence acquired so far.
  - `threshold`: a number representing the acceptable level of uncertainty about the target variable.
  - `evidence=Evidence()`: initial experimental evidence.
  - `costs_tradeoff`: tradeoff between monetary cost and execution time of an experimental designs, given as a tuple of floats.
  - `max_parallel`: maximum number of parallel experiments.
  - `discount`: this is the discounting factor utilized in reward computation.
  - `bigM`: it refers to the penalty that arises in a scenario where further experimental action is not an option, yet the uncertainty exceeds the allowable limit.
  - `max_experiments`: this denotes the maximum number of experiments that are permissible to be conducted.
"""
struct UncertaintyReductionMDP <: POMDPs.MDP{State,Vector{String}}
    # initial state
    initial_state::State
    # uncertainty threshold
    threshold::Float64

    # actions and costs
    costs::Dict{String,ActionCost}
    # monetary cost v. time tradeoff
    costs_tradeoff::NTuple{2,Float64}
    # maximum number of assays that can be run in parallel
    max_parallel::Int
    # discount
    discount::Float64
    # max experiments
    max_experiments::Int64
    # penalty if max number of experiments exceeded
    bigM::Int64

    ## sample readouts from the posterior
    sampler::Function
    ## measure of uncertainty about the ground truth
    uncertainty::Function

    ## ! compute current state weights
    weights::Function

    ## ! make MDP have access to historical data (external data)
    data::DataFrame

    ## ! make MDP have access to terminal_condition which specify the target constraints as `Dict` and desirability as probability threshold as a scaler
    terminal_condition::Tuple{Dict, Float64}

    ## ! NEW: Enable conditional constraint checking
    conditional_constraints_enabled::Bool
    
    ## ! NEW: Ensemble configuration
    ensemble_config::NamedTuple{(:enabled, :size, :thresholds),Tuple{Bool,Int,Vector{Float64}}}

    function UncertaintyReductionMDP(
        costs;
        sampler,
        uncertainty,
        threshold,
        evidence = Evidence(),
        costs_tradeoff = (1, 0),
        max_parallel::Int = 1,
        discount = 1.0,
        bigM = const_bigM,
        max_experiments = bigM,
        weights, # ! Added this line
        data, # ! Added this line
        terminal_condition = (Dict("target" => [0.0, 1.0]), 0.0),  # !Default target value for terminal_condition
        conditional_constraints_enabled::Bool = false,  # ! NEW: Default to false for backward compatibility
        ensemble_config = (enabled=false, size=30, thresholds=[0.9]),  # ! NEW: Ensemble configuration
    )
        state = State((evidence, Tuple(zeros(2))))

        # check if `sampler`, `uncertainty` are compatible
        @assert hasmethod(sampler, Tuple{Evidence,Vector{String},AbstractRNG}) """`sampler` must implement a method accepting `(evidence, readout features, rng)` as its arguments."""
        @assert hasmethod(uncertainty, Tuple{Evidence}) """`uncertainty` must implement a method accepting `evidence` as its argument."""

        # actions and their costs
        costs = Dict{String,ActionCost}(
            try
                if action isa Pair && action[2] isa Pair
                    string(action[1]) => (;
                        costs = Tuple(Float64[action[2][1]..., 0][1:2]),
                        features = convert(Vector{String}, action[2][2]),
                    )
                elseif action isa Pair
                    string(action[1]) => (;
                        costs = Tuple(Float64[action[2]..., 0][1:2]),
                        features = String[action[1]],
                    )
                else
                    error()
                end
            catch
                error("could not parse $action as an action")
            end for action in costs
        )

        return new(
            state,
            threshold,
            costs,
            costs_tradeoff,
            max_parallel,
            discount,
            max_experiments,
            bigM,
            sampler,
            uncertainty,
            weights, # !Added this line
            data, # !Added this line
            terminal_condition,  # !Pass the terminal_condition to the struct
            conditional_constraints_enabled,  # ! NEW
            ensemble_config  # ! NEW
        )
    end
end

"""
A penalized action that results in a terminal state, e.g., in situations where conducting additional experiments is not possible, but the level of uncertainty remains above an acceptable threshold.
"""
const eox = "EOX"

function POMDPs.actions(m::UncertaintyReductionMDP, state)
    all_actions = filter!(collect(keys(m.costs))) do a
        return !isempty(m.costs[a].features) &&
               !in(first(m.costs[a].features), keys(state.evidence))
    end

    if !isempty(all_actions) && (length(state.evidence) < m.max_experiments)
        collect(powerset(all_actions, 1, m.max_parallel))
    else
        [[eox]]
    end
end

# function target_likelihood(
#     evidence;
#     compute_weights,
#     hist_data,
#     target_condition = Dict("target" => [0., 1.0]),
# )

#     # compute the weights
#     current_weights = compute_weights(evidence) # current state's weights with respect to all historical data
#     # in this current_weights find the indices of the target where the condition is true
#     valid_indices = trues(nrow(hist_data))
#     for (colname, range) in target_condition
#         if colname in names(hist_data)
#             range_min = range[1]
#             range_max = range[2]
#             valid_indices .&=
#                 (hist_data[!, colname] .>= range_min) .&
#                 (hist_data[!, colname] .<= range_max)
#             # calculated the sum of  valid indices associated rows for the current_weights
#             target_conditioned_weights_percentage = sum(current_weights[valid_indices])
#             # @info "the target conditioned weights are $target_conditioned_weights_percentage"
#             # @info "The total weights are $(sum(current_weights))"
#             return target_conditioned_weights_percentage
#         else
#             error("Column $colname not found in data.")
#         end
#     end
# end



function POMDPs.isterminal(m::UncertaintyReductionMDP, state)
    # First, check if 'EOX' is a key in state.evidence to avoid KeyError
    if haskey(state.evidence, "EOX")
        return true
    end
    # Extract the target_condition and condition_weights_percentage_threshold from terminal_condition
    target_condition, condition_weights_percentage_threshold = m.terminal_condition

    # Initialize likelihood_condition as true to handle cases where target_condition might be empty
    likelihood_condition = true

    # Check if target_condition is not empty (i.e., the dictionary has at least one key-value pair)
    if !isempty(target_condition)
        condition_weights_percentage = conditional_likelihood(
            state.evidence;
            compute_weights = m.weights, # Assuming this line correctly adds weights
            hist_data = m.data, # Assuming this line correctly adds historical data
            target_condition = target_condition, # Use the extracted target_condition
        )
        likelihood_condition =
            condition_weights_percentage >= condition_weights_percentage_threshold
    end

    # Check if the uncertainty condition and conditional likelihood condition (if applicable) are met
    return (m.uncertainty(state.evidence) <= m.threshold) && likelihood_condition
end

POMDPs.discount(m::UncertaintyReductionMDP) = m.discount

POMDPs.initialstate(m::UncertaintyReductionMDP) = Deterministic(m.initial_state)





# =========  consider the following when minizing sum(L(s_t))
# function POMDPs.reward(m::UncertaintyReductionMDP, s, action, s_next)
#     base_reward = -sum(s_next.costs .* m.costs_tradeoff)
#     likelihood = target_likelihood(s_next.evidence, compute_weights=m.weights, hist_data=m.data, target_condition=m.terminal_condition[1])
#     likelihood_penalty = likelihood < m.terminal_condition[2] ? -m.bigM * (m.terminal_condition[2] - likelihood) : 0
#     return base_reward - likelihood_penalty
# end




function POMDPs.reward(m::UncertaintyReductionMDP, _, action, state)
    if action == [eox]
        -m.bigM
    else
        -sum(state.costs .* m.costs_tradeoff)
    end
end

"""
    efficient_design(costs; sampler, uncertainty, threshold, evidence=Evidence(), <keyword arguments>)

In the uncertainty reduction setup, minimize the expected experimental cost while ensuring the uncertainty remains below a specified threshold.

# Arguments

  - `costs`: a dictionary containing pairs `experiment => cost`, where `cost` can either be a scalar cost (modelled as a monetary cost) or a tuple `(monetary cost, execution time)`.

# Keyword Arguments

  - `sampler`: a function of `(evidence, features, rng)`, in which `evidence` denotes the current experimental evidence, `features` represent the set of features we want to sample from, and `rng` is a random number generator; it returns a dictionary mapping the features to outcomes.
  - `uncertainty`: a function of `evidence`; it returns the measure of variance or uncertainty about the target variable, conditioned on the experimental evidence acquired so far.
  - `threshold`: uncertainty threshold.
  - `evidence=Evidence()`: initial experimental evidence.
  - `solver=default_solver`: a POMDPs.jl compatible solver used to solve the decision process. The default solver is [`DPWSolver`](https://juliapomdp.github.io/MCTS.jl/dev/dpw/).
  - `repetitions=0`: number of runoffs used to estimate the expected experimental cost.
  - `mdp_options`: a `NamedTuple` of additional keyword arguments that will be passed to the constructor of [`UncertaintyReductionMDP`](@ref).
  - `realized_uncertainty=false`: whenever the initial state uncertainty is below the selected threshold, return the actual uncertainty of this state.

# Example

```julia
(; sampler, uncertainty, weights) = DistanceBased(
    data;
    target = "HeartDisease",
    uncertainty = Entropy(),
    similarity = Exponential(; λ = 5),
);
# initialize evidence
evidence = Evidence("Age" => 35, "Sex" => "M")
# set up solver (or use default)
solver = GenerativeDesigns.DPWSolver(; n_iterations = 60_000, tree_in_info = true)
designs = efficient_design(
    costs;
    experiments,
    sampler,
    uncertainty,
    threshold = 0.6,
    evidence,
    solver,            # planner
    mdp_options = (; max_parallel = 1),
    repetitions = 5,
)
```
"""
function efficient_design(
    costs;
    sampler,
    uncertainty,
    threshold,
    evidence = Evidence(),
    weights,  # ! include weights here to place weights of target experiemnt constraints in the POMDP.terminal condition
    data, # ! access to historical data
    terminal_condition = (Dict(), 0.0),  # Default terminal condition
    solver = default_solver,
    repetitions = 0,
    realized_uncertainty = false,
    mdp_options = (;),
)
    mdp = UncertaintyReductionMDP(
        costs;
        sampler,
        uncertainty,
        threshold,
        evidence,
        weights,  # !Pass weights to the MDP
        data, # !Pass data to the MDP
        terminal_condition = terminal_condition,  # Pass terminal_condition to the MDP
        mdp_options...,
    )
    if isterminal(mdp, mdp.initial_state)
        return (
            (0.0, if realized_uncertainty
                mdp.uncertainty(mdp.initial_state.evidence)
            else
                threshold
            end),
            (; monetary_cost = 0.0, time = 0.0),
        )
    else
        # planner
        planner = solve(solver, mdp)
        action, info = action_info(planner, mdp.initial_state)

        if repetitions > 0
            queue = [Sim(mdp, planner) for _ = 1:repetitions]

            stats = run_parallel(queue) do _, hist
                monetary_cost, time = hist[end][:s].costs
                return (;
                    monetary_cost,
                    time,
                    combined_cost = -discounted_reward(hist),
                    actions = hist[:a],
                )
            end

            if haskey(info, :tree)
                return (
                    (-info[:best_Q], threshold),
                    (;
                        planner,
                        arrangement = [action],
                        monetary_cost = mean(stats.monetary_cost),
                        time = mean(stats.time),
                        tree = info[:tree],
                        stats,
                    ),
                )
            else
                return (
                    (-info[:best_Q], threshold),
                    (;
                        planner,
                        arrangement = [action],
                        monetary_cost = mean(stats.monetary_cost),
                        time = mean(stats.time),
                        stats,
                    ),
                )
            end
        else
            if haskey(info, :tree)
                return (
                    (-info[:best_Q], threshold),
                    (; planner, arrangement = [action], tree = info[:tree]),
                )
            else
                return ((-info[:best_Q], threshold), (; planner, arrangement = [action]))
            end
        end
    end
end

"""
    efficient_designs(costs; sampler, uncertainty, thresholds, evidence=Evidence(), <keyword arguments>)

In the uncertainty reduction setup, minimize the expected experimental resource spend over a range of uncertainty thresholds, and return the set of Pareto-efficient designs in the dimension of cost and uncertainty threshold.

Internally, an instance of the `UncertaintyReductionMDP` structure is created for every selected uncertainty threshold and the corresponding runoffs are simulated.

# Arguments

  - `costs`: a dictionary containing pairs `experiment => cost`, where `cost` can either be a scalar cost (modelled as a monetary cost) or a tuple `(monetary cost, execution time)`.

# Keyword Arguments

  - `sampler`: a function of `(evidence, features, rng)`, in which `evidence` denotes the current experimental evidence, `features` represent the set of features we want to sample from, and `rng` is a random number generator; it returns a dictionary mapping the features to outcomes.
  - `uncertainty`: a function of `evidence`; it returns the measure of variance or uncertainty about the target variable, conditioned on the experimental evidence acquired so far.
  - `thresholds`: number of thresholds to consider uniformly in the range between 0 and 1, inclusive.
  - `evidence=Evidence()`: initial experimental evidence.
  - `solver=default_solver`: a POMDPs.jl compatible solver used to solve the decision process. The default solver is [`DPWSolver`](https://juliapomdp.github.io/MCTS.jl/dev/dpw/).
  - `repetitions=0`: number of runoffs used to estimate the expected experimental cost.
  - `mdp_options`: a `NamedTuple` of additional keyword arguments that will be passed to the constructor of [`UncertaintyReductionMDP`](@ref).
  - `realized_uncertainty=false`: whenever the initial state uncertainty is below the selected threshold, return the actual uncertainty of this state.

# Example

```julia
(; sampler, uncertainty, weights) = DistanceBased(
    data;
    target = "HeartDisease",
    uncertainty = Entropy(),
    similarity = Exponential(; λ = 5),
);
# initialize evidence
evidence = Evidence("Age" => 35, "Sex" => "M")
# set up solver (or use default)
solver = GenerativeDesigns.DPWSolver(; n_iterations = 60_000, tree_in_info = true)
designs = efficient_designs(
    costs;
    experiments,
    sampler,
    uncertainty,
    thresholds = 6,
    evidence,
    solver,            # planner
    mdp_options = (; max_parallel = 1),
    repetitions = 5,
)
```
"""
function efficient_designs(
    costs;
    sampler,
    uncertainty,
    thresholds,
    evidence = Evidence(),
    weights,  # !Include weights here
    data, # !Include historical data 
    terminal_condition = (Dict(), 0.0),  # Default terminal condition
    solver = default_solver,
    repetitions = 0,
    realized_uncertainty = false,
    mdp_options = (;),
)
    designs = []
    for threshold in range(0.0, 1.0, thresholds)
        @info "Current threshold level : $threshold"
        push!(
            designs,
            efficient_design(
                costs;
                sampler,
                uncertainty,
                threshold,
                evidence,
                weights, # !Include weights here
                data, # !Include historical data
                terminal_condition = terminal_condition,  # !Include terminal_condition here
                solver,
                repetitions,
                realized_uncertainty,
                mdp_options,
            ),
        )
    end
    ## rewrite 
    return front(x -> x[1], designs)
end

# ! ======= Enhanced likelihood function supporting multiple target conditions ========
function conditional_likelihood(
    evidence;
    compute_weights,
    hist_data,
    target_condition = Dict("target" => [0.0, 1.0]),
)
    # compute the weights
    current_weights = compute_weights(evidence) # current state's weights with respect to all historical data
    # Initialize valid_indices to true for all rows
    valid_indices = trues(nrow(hist_data))

    # Iterate through each condition in target_condition
    for (colname, range) in target_condition
        if colname in names(hist_data)
            range_min, range_max = range
            # Update valid_indices based on the current condition
            valid_indices .&=
                (hist_data[!, colname] .>= range_min) .&
                (hist_data[!, colname] .<= range_max)
        else
            error("Column $colname not found in data.")
        end
    end

    # Calculate the sum of valid indices associated rows for the current_weights
    conditioned_weights_percentage = sum(current_weights[valid_indices])
    # @info "the conditioned weights are $conditioned_weights_percentage"
    # @info "The total weights are $(sum(current_weights))"
    return conditioned_weights_percentage
end

# ! ======= Enhanced transition function with conditional constraint checking ========
function POMDPs.transition(m::UncertaintyReductionMDP, state, action_set)
    if action_set == [eox]
        Deterministic(merge(state, Dict(eox => -1), (0.0, 0.0)))
    else
        # costs
        cost_m, cost_t = 0.0, 0.0
        for experiment in action_set
            cost_m += m.costs[experiment].costs[1] # monetary cost
            cost_t = max(cost_t, m.costs[experiment].costs[2]) # time
        end

        # readout features 
        features = vcat(map(action -> m.costs[action].features, action_set)...)
        ImplicitDistribution() do rng
            # sample readouts from history
            observation, target_value, conditional_weights = m.sampler(state.evidence, features, rng) # target_value is just end point           
            # create new evidence, add new information
            new_state = merge(state, observation, (cost_m, cost_t))

            # ! NEW: Check conditional constraints if enabled
            if m.conditional_constraints_enabled
                # Check the probability weight for the target condition
                target_condition, condition_weights_percentage_threshold = m.terminal_condition
                condition_weights_percentage = conditional_likelihood(
                    new_state.evidence;
                    compute_weights = m.weights,
                    hist_data = m.data,
                    target_condition = target_condition,
                )

                # If the condition is met, return the new state; otherwise, return the current state
                if condition_weights_percentage >= condition_weights_percentage_threshold
                    return new_state
                else
                    return state
                end
            else
                # Standard transition without conditional constraints
                return new_state
            end
        end
    end
end

# ! ======= Ensemble Design Functions for Conditional CEED ========

"""
    perform_ensemble_designs(costs; sampler, uncertainty, thresholds, evidence, weights, data, terminal_condition, realized_uncertainty, solver, repetitions, mdp_options, thred_set, N)

Perform ensemble experimental designs with conditional CEED approach.

# Arguments
- `costs`: Dictionary of experimental costs
- `sampler`: Sampling function
- `uncertainty`: Uncertainty measurement function 
- `thresholds`: Uncertainty thresholds to evaluate
- `evidence`: Initial experimental evidence
- `weights`: Weight calculation function
- `data`: Historical data
- `terminal_condition`: Terminal condition (target_constraints, threshold)
- `realized_uncertainty`: Whether to use realized uncertainty
- `solver`: MDP solver to use
- `repetitions`: Number of repetitions per design
- `mdp_options`: Additional MDP options
- `thred_set`: Set of belief thresholds to evaluate (default [0.9])
- `N`: Number of ensemble runs per threshold (default 30)

# Returns
Dictionary mapping belief thresholds to ensemble results
"""
function perform_ensemble_designs(
    costs;
    sampler,
    uncertainty,
    thresholds,
    evidence = Evidence(),
    weights,
    data,
    terminal_condition = (Dict("target" => [0.0, 1.0]), 0.0),
    realized_uncertainty = false,
    solver = default_solver,
    repetitions = 0,
    mdp_options = (;),
    thred_set = [0.9],
    N = 30
)
    results = Dict()

    for threshold in thred_set
        ensemble_results = []
        for i in 1:N
            println("Running ensemble $i for belief threshold $threshold")
            design = efficient_designs(
                costs;
                sampler = sampler,
                uncertainty = uncertainty,
                thresholds = thresholds,
                evidence = evidence,
                weights = weights,
                data = data,
                terminal_condition = (terminal_condition[1], threshold),
                realized_uncertainty = realized_uncertainty,
                solver = solver,
                repetitions = repetitions,
                mdp_options = (conditional_constraints_enabled=true, mdp_options...)
            )
            push!(ensemble_results, design)
        end
        # Store results using a tuple (:belief, threshold) as the key
        key = (:belief => threshold)
        results[key] = ensemble_results
    end

    return results
end

"""
    conditional_efficient_design(costs; kwargs...)

Create an efficient design with conditional constraints enabled.

# Arguments
Same as `efficient_design` with additional conditional parameters

# Returns
Design result with conditional constraints applied
"""
function conditional_efficient_design(
    costs;
    sampler,
    uncertainty,
    threshold,
    evidence = Evidence(),
    weights,
    data,
    terminal_condition = (Dict("target" => [0.0, 1.0]), 0.8),
    solver = default_solver,
    repetitions = 0,
    realized_uncertainty = false,
    mdp_options = (;),
)
    return efficient_design(
        costs;
        sampler = sampler,
        uncertainty = uncertainty,
        threshold = threshold,
        evidence = evidence,
        weights = weights,
        data = data,
        terminal_condition = terminal_condition,
        solver = solver,
        repetitions = repetitions,
        realized_uncertainty = realized_uncertainty,
        mdp_options = (conditional_constraints_enabled=true, mdp_options...)
    )
end

"""
    conditional_efficient_designs(costs; kwargs...)

Create efficient designs with conditional constraints enabled.

# Arguments
Same as `efficient_designs` with additional conditional parameters

# Returns
Pareto-efficient designs with conditional constraints applied
"""
function conditional_efficient_designs(
    costs;
    sampler,
    uncertainty,
    thresholds,
    evidence = Evidence(),
    weights,
    data,
    terminal_condition = (Dict("target" => [0.0, 1.0]), 0.8),
    solver = default_solver,
    repetitions = 0,
    realized_uncertainty = false,
    mdp_options = (;),
)
    return efficient_designs(
        costs;
        sampler = sampler,
        uncertainty = uncertainty,
        thresholds = thresholds,
        evidence = evidence,
        weights = weights,
        data = data,
        terminal_condition = terminal_condition,
        solver = solver,
        repetitions = repetitions,
        realized_uncertainty = realized_uncertainty,
        mdp_options = (conditional_constraints_enabled=true, mdp_options...)
    )
end
