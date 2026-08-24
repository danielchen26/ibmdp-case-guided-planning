module ValueIteration

using DataFrames, Distributions, LinearAlgebra, Statistics, Distances, CSV, Random, StatsBase, Plots, Combinatorics
# StatsPlots removed due to dependency conflicts
using Plots.PlotMeasures



export CompoundExperimentMDP, CompoundState, generate_historical_dataset, simulate_experiments, 
       calculate_target_variance, optimize_policy, get_policy_action, analyze_optimization_results,
       plot_convergence_analysis, save_historical_data, ConvergenceMetrics, track_convergence,
       state_to_index, index_to_state, get_reachable_states, value_iteration_analysis,
       similarity_based_value_iteration, theoretical_value_iteration, setup_mdp_framework,
       run_mdp_analysis

"""
Structure to hold MDP parameters for a single compound experiment.
Fields:
- n_features: Number of features that can be measured
- feature_costs: Cost of measuring each feature
- uncertainty_threshold: Target threshold for variance reduction
- feature_distributions: Distributions for generating feature values
- target_coefficients: Coefficients for target value calculation
- policy: Vector storing the optimal action for each state
"""
mutable struct CompoundExperimentMDP
    n_features::Int64
    feature_costs::Vector{Float64}
    uncertainty_threshold::Float64
    feature_distributions::Vector{Distribution}
    target_coefficients::Vector{Float64}
    policy::Vector{Vector{Int}}
    
    function CompoundExperimentMDP(
        n_features::Int64,
        feature_costs::Vector{Float64},
        uncertainty_threshold::Float64
    )
        @assert length(feature_costs) == n_features+1 "Feature costs must match number of features plus target"
        feature_distributions = Vector{Distribution}(undef, n_features+1)
        for i in 1:n_features  # Regular features
            mean_val = 50.0 * (i / n_features)
            std_val = mean_val * 0.3
            feature_distributions[i] = truncated(Normal(mean_val, std_val), 0.0, mean_val * 2)
        end
        # Target distribution
        feature_distributions[end] = truncated(Normal(75.0, 25.0), 0.0, 150.0)
        
        target_coefficients = [0.3, 0.25, 0.2, 0.15, 0.07, 0.03]  # 6 coefficients for features
        policy = [Vector{Int}() for _ in 1:2^(n_features+1)]  # Include target in state space
        
        new(
            n_features+1,  # Total features including target
            feature_costs,
            uncertainty_threshold,
            feature_distributions,
            target_coefficients,
            policy
        )
    end
end

"""
Structure to represent the state of a compound's measurements.
Fields:
- measured: BitVector indicating which features are measured
- values: Vector of measured feature values (missing for unmeasured)
- n_features: Total number of possible features
"""
mutable struct CompoundState
    measured::BitVector
    values::Vector{Union{Missing, Float64}}
    n_features::Int

    function CompoundState(n_features::Int)
        new(falses(n_features), 
            Vector{Union{Missing, Float64}}(missing, n_features),
            n_features)
    end

    # Constructor for initial state from dictionary
    function CompoundState(initial_state::Dict{Int,Float64}, n_features::Int)
        measured = falses(n_features + 1)  # +1 for target
        values = Vector{Union{Missing, Float64}}(missing, n_features + 1)
        
        for (idx, val) in initial_state
            measured[idx] = true
            values[idx] = val
        end
        
        new(measured, values, n_features + 1)
    end

    # Constructor from measured flags and values
    function CompoundState(measured::BitVector, values::Vector{Union{Missing, Float64}}, n_features::Int)
        @assert length(measured) == n_features "Length of measured vector must match total number of features"
        @assert length(values) == n_features "Length of values vector must match total number of features"
        new(measured, values, n_features)
    end

    # Copy constructor
    function CompoundState(state::CompoundState)
        new(copy(state.measured),
            copy(state.values),
            state.n_features)
    end
end

# Helper functions for CompoundState
Base.copy(state::CompoundState) = CompoundState(copy(state.measured), copy(state.values), state.n_features)
Base.:(==)(a::CompoundState, b::CompoundState) = a.measured == b.measured && a.values == b.values
Base.hash(state::CompoundState) = hash(state.measured)

# Get measured features
measured_features(state::CompoundState) = findall(state.measured)
unmeasured_features(state::CompoundState) = findall(.!state.measured)

# Update state with new measurements
function update!(state::CompoundState, feature_idx::Int, value::Float64)
    state.measured[feature_idx] = true
    state.values[feature_idx] = value
end

function update!(state::CompoundState, features::Vector{Int}, values::Vector{Float64})
    for (idx, val) in zip(features, values)
        update!(state, idx, val)
    end
end

"""
Get all possible feature combinations as actions from current state.
Uses combinations from Combinatorics to generate all possible combinations of unmeasured features.
"""
function get_possible_actions(state::CompoundState)::Vector{Vector{Int}}
    unmeasured = unmeasured_features(state)
    actions = Vector{Vector{Int}}()
    
    for r in 1:length(unmeasured)
        for combo in combinations(unmeasured, r)
            push!(actions, collect(combo))
        end
    end
    
    return actions
end

"""
Structure to hold convergence metrics
"""
struct ConvergenceMetrics
    iteration::Int
    max_value_diff::Float64
    mean_value_diff::Float64
    n_actions_changed::Int
    value_range::Tuple{Float64,Float64}
    policy_entropy::Float64
    rewards_range::Tuple{Float64,Float64}
    q_values_range::Tuple{Float64,Float64}
    policy::Vector{Vector{Int}}  # Add policy snapshot
end

"""
Track convergence metrics during value iteration.
"""
function track_convergence(
    iteration::Int64,
    V::Vector{Float64},
    V_old::Vector{Float64},
    policy::Vector{Vector{Int}},
    old_policy::Vector{Vector{Int}},
    rewards::Dict{Int, Float64},
    q_values::Dict{Int, Float64}
)::ConvergenceMetrics
    # Value function differences
    value_diffs = V .- V_old
    max_diff = maximum(abs.(value_diffs))
    mean_diff = mean(abs.(value_diffs))
    
    # Count policy changes
    n_changes = sum(policy .!= old_policy)
    
    # Value function range
    value_range = (minimum(V), maximum(V))
    
    # Policy entropy (based on action set sizes)
    action_counts = [length(actions) for actions in policy]
    total_actions = sum(action_counts)
    if total_actions > 0
        probs = action_counts ./ total_actions
        policy_entropy = -sum(p * log(p) for p in probs if p > 0)
    else
        policy_entropy = 0.0
    end
    
    # Reward and Q-value ranges
    reward_values = collect(values(rewards))
    q_values_vec = collect(values(q_values))
    rewards_range = isempty(reward_values) ? (0.0, 0.0) : (minimum(reward_values), maximum(reward_values))
    q_values_range = isempty(q_values_vec) ? (0.0, 0.0) : (minimum(q_values_vec), maximum(q_values_vec))
    
    return ConvergenceMetrics(
        iteration,
        max_diff,
        mean_diff,
        n_changes,
        value_range,
        policy_entropy,
        rewards_range,
        q_values_range,
        deepcopy(policy)  # Store current policy
    )
end

"""
Helper function to convert state (any format) to index
"""
function state_to_index(state::Union{Vector{Bool}, BitVector, Dict{Int, Float64}}, n_features::Int64=length(state))::Int
    # If state is a dictionary, convert to boolean vector first
    if state isa Dict
        state_vec = falses(n_features)
        for (idx, _) in state
            state_vec[idx] = true
        end
        state = state_vec
    end
    
    # Convert boolean vector to index (using binary representation)
    idx = 0
    for (i, bit) in enumerate(state)
        if bit
            idx |= (1 << (i-1))  # First element is LSB (2^0)
        end
    end
    return idx + 1  # Add 1 because Julia is 1-based
end

"""
Helper function to convert index to state
"""
function index_to_state(idx::Int64, n_features::Int64)::Vector{Bool}
    idx -= 1  # Subtract 1 because Julia is 1-based
    state = zeros(Bool, n_features)
    for i in 1:n_features
        if (idx & (1 << (i-1))) != 0  # Match state_to_index convention
            state[i] = true
        end
    end
    return state
end

"""
Get all states reachable from an initial CompoundState.
"""
function get_reachable_states(initial_state::CompoundState)::Vector{BitVector}
    reachable = Vector{BitVector}()
    push!(reachable, copy(initial_state.measured))
    
    unmeasured = unmeasured_features(initial_state)
    for r in 1:length(unmeasured)
        for combo in combinations(unmeasured, r)
            new_state = copy(initial_state.measured)
            new_state[collect(combo)] .= true
            push!(reachable, new_state)
        end
    end
    
    return reachable
end

# Backward compatibility version
function get_reachable_states(initial_state::Dict{Int, Float64}, n_features::Int)::Vector{BitVector}
    compound_state = CompoundState(initial_state, n_features)
    return get_reachable_states(compound_state)
end

"""
Generate synthetic historical dataset for training.
"""
function generate_historical_dataset(
    n_compounds::Int64 = 200,
    n_features::Int64 = 6,
    seed::Int64 = 42,
    add_noise::Bool = false;
    use_random_coeffs::Bool = false,
    custom_coeffs::Union{Nothing, Vector{Float64}} = nothing
)::Tuple{DataFrame, Vector{Float64}, DataFrame}
    # Set seed for dataset generation
    Random.seed!(seed)
    
    # Build feature distributions. ROUTE-1 FIX: ALL features (including the last)
    # are genuine informative features drawn from the same truncated-normal family,
    # so the linear model g = beta . y is well-posed and every feature can carry signal.
    # (Previously feature_$n_features was overwritten with a zero-mean noise distribution,
    #  giving it near-zero variance and no real signal despite a nonzero beta coefficient.)
    feature_distributions = Vector{Distribution}(undef, n_features)
    for i in 1:n_features
        mean_val = 50.0 * (i / n_features)
        std_val = mean_val * 0.3
        feature_distributions[i] = truncated(Normal(mean_val, std_val), 0.0, mean_val * 2)
    end

    # Determine target coefficients
    if !isnothing(custom_coeffs)
        if length(custom_coeffs) != n_features
            error("Custom coefficients must have length $n_features")
        end
        target_coefficients = custom_coeffs / sum(custom_coeffs)
        println("Using custom coefficients:")
    elseif use_random_coeffs
        coeff_seed = Int(round(time() * 1000)) % 10000
        Random.seed!(coeff_seed)
        raw_coeffs = rand(n_features)
        target_coefficients = raw_coeffs / sum(raw_coeffs)
        println("Using random coefficients (seed: $coeff_seed):")
        Random.seed!(seed)  # Reset to original seed
    else
        target_coefficients = [0.3, 0.25, 0.2, 0.15, 0.07, 0.03]
        println("Using default coefficients:")
    end
    println(round.(target_coefficients, digits=4))

    # Generate feature data
    data = zeros(n_compounds, n_features + 1)  # +1 for target column
    for i in 1:n_compounds
        # Generate regular features
        for j in 1:n_features
            data[i,j] = rand(feature_distributions[j])
        end
        
        # Calculate target value with noise
        target_value = sum(target_coefficients .* data[i,1:n_features])
        if add_noise
            target_value += rand(feature_distributions[end])
        end
        data[i,n_features + 1] = target_value  # Store target in last column
    end
    
    # Create DataFrame with proper column names
    feature_cols = ["feature_$j" for j in 1:n_features]  # Regular features
    historical_data = DataFrame(data, [feature_cols; "target"])  # Add target column
    
    # Calculate feature statistics
    feature_stats = DataFrame(
        feature = 1:n_features,
        mean = [mean(historical_data[:,i]) for i in 1:n_features],
        std = [std(historical_data[:,i]) for i in 1:n_features],
        min = [minimum(historical_data[:,i]) for i in 1:n_features],
        max = [maximum(historical_data[:,i]) for i in 1:n_features]
    )
    
    return historical_data, target_coefficients, feature_stats
end

"""
Calculate similarities between a compound state and historical data.
"""
function calculate_similarities(
    state::CompoundState,
    historical_data::DataFrame,
    known_indices::Vector{Int};
    λ_distance::Float64=1.0,
    λ_similarity::Float64=0.5
)::Vector{Float64}
    if isempty(known_indices)
        return zeros(nrow(historical_data))
    end
    
    # Initialize array for storing distances for each feature
    array_distances = zeros((nrow(historical_data), length(known_indices)))
    
    # Calculate standardized quadratic distances for each known feature
    for (i, idx) in enumerate(known_indices)
        if !ismissing(state.values[idx])
            col_values = historical_data[:, idx]
            μ = mean(col_values)
            σ = std(col_values)
            
            # Using QuadraticDistance approach with λ scaling
            array_distances[:, i] .= λ_distance * (state.values[idx] .- col_values).^2 / var(col_values)
        end
    end
    
    # Sum distances across features
    distances_sum = vec(sum(array_distances; dims=2))
    
    # Convert distances to similarities using exponential with λ scaling
    similarities = exp.(-λ_similarity .* distances_sum)
    
    # Normalize similarities
    if sum(similarities) > 0
        similarities ./= sum(similarities)
    else
        similarities .= 1.0 / length(similarities)
    end
    
    return similarities
end

"""
Calculate variance of target value using both similarity-based and theoretical approaches.
"""
function calculate_target_variance(
    state::CompoundState,
    historical_data::DataFrame,
    target_coeffs::Vector{Float64},
    feature_stats::DataFrame;
    λ_distance::Float64=1.0,
    λ_similarity::Float64=0.5
)::Tuple{Float64, Float64}
    # Similarity-based variance
    known_indices = measured_features(state)
    
    if isempty(known_indices)
        sim_variance = 1.0  # Maximum uncertainty
    else
        similarities = calculate_similarities(
            state, 
            historical_data, 
            known_indices;
            λ_distance=λ_distance,
            λ_similarity=λ_similarity
        )
        weights = Weights(similarities)
        targets = historical_data[:, "target"]
        
        # Calculate initial variance with uniform weights
        initial_variance = var(targets, Weights(ones(length(targets))); corrected=false)
        
        # Calculate current variance and normalize
        current_variance = var(targets, weights; corrected=false)
        sim_variance = current_variance / initial_variance
    end
    
    # Theoretical variance (using feature coefficients)
    theo_variance = 0.0
    for i in 1:length(target_coeffs)
        if !state.measured[i]
            σᵢ = feature_stats[i, :std]
            βᵢ = target_coeffs[i]
            theo_variance += (βᵢ^2) * (σᵢ^2)
        end
    end
    
    return sim_variance, theo_variance
end

"""
Sample a feature value using similarity-based approach.
"""
function sample_feature_value(
    historical_data::DataFrame,
    state::CompoundState,
    feature_idx::Int;
    n_samples::Int=100,
    λ_distance::Float64=1.0,
    λ_similarity::Float64=0.5
)::Union{Float64, Missing}
    known_indices = measured_features(state)
    
    if isempty(known_indices)
        return missing
    end
    
    # Calculate weights using the same approach as distancebased.jl
    similarities = calculate_similarities(
        state, 
        historical_data, 
        known_indices;
        λ_distance=λ_distance,
        λ_similarity=λ_similarity
    )
    weights = Weights(similarities)
    
    # Filter out missing values and adjust weights
    feature_values = historical_data[:, feature_idx]
    valid_indices = .!ismissing.(feature_values)
    if !any(valid_indices)
        return missing
    end
    
    valid_values = feature_values[valid_indices]
    valid_weights = Weights(similarities[valid_indices])
    
    # Take multiple samples and average them for robustness
    total_value = 0.0
    for _ in 1:n_samples
        sampled_idx = sample(1:length(valid_values), valid_weights)
        total_value += valid_values[sampled_idx]
    end
    
    return total_value / n_samples
end

"""
Optimize policy using value iteration.
"""
function optimize_policy(
    mdp::CompoundExperimentMDP,
    initial_state::CompoundState,
    historical_data::DataFrame,
    target_coeffs::Vector{Float64},
    feature_stats::DataFrame;
    use_theoretical::Bool=false,
    γ::Float64=0.95,
    max_iterations::Int=1000,
    tolerance::Float64=1e-6,
    stable_count::Int64=10,
    verbose::Bool=true
)::Tuple{Vector{Vector{Int}}, Vector{Float64}, Bool, Vector{ConvergenceMetrics}}
    # Get reachable states using CompoundState
    reachable_states = get_reachable_states(initial_state)
    n_states = 2^mdp.n_features
    
    # Initialize value function and policy
    V = fill(0.1, n_states)
    policy = [Vector{Int}() for _ in 1:n_states]
    metrics = ConvergenceMetrics[]
    
    if verbose
        println("\n=== Value Iteration Setup ===")
        println("Total possible states: $n_states")
        println("Reachable states: $(length(reachable_states))")
        println("Initial measured features: $(measured_features(initial_state))")
        println("Initial values: $(initial_state.values)")
    end
    
    # Add counter for stable iterations
    stable_iterations = 0
    
    # Main value iteration loop
    for iter in 1:max_iterations
        V_old = copy(V)
        max_diff = 0.0
        n_actions_changed = 0
        
        all_rewards = Dict{Int, Float64}()
        all_q_values = Dict{Int, Float64}()
        
        # Only evaluate reachable states
        for bool_state in reachable_states
            state_idx = state_to_index(bool_state)
            old_action = policy[state_idx]
            
            if verbose && iter <= 3
                println("\nEvaluating state $(state_idx):")
                println("  Measured features: $(findall(bool_state))")
            end
            
            if all(bool_state)
                V[state_idx] = 0.0
                policy[state_idx] = Vector{Int}()
                continue
            end
            
            # Create CompoundState for current state
            current_state = CompoundState(mdp.n_features)
            current_state.measured = bool_state
            # Copy known values from initial state
            for i in measured_features(initial_state)
                if bool_state[i]
                    current_state.values[i] = initial_state.values[i]
                end
            end
            # Sample values for other measured features
            for i in measured_features(current_state)
                if !initial_state.measured[i]
                    current_state.values[i] = sample_feature_value(
                        historical_data,
                        current_state,
                        i
                    )
                end
            end
            
            # Calculate current state's variance
            sim_variance, theo_variance = calculate_target_variance(
                current_state,
                historical_data,
                target_coeffs,
                feature_stats
            )
            current_variance = use_theoretical ? theo_variance : sim_variance
            
            if verbose && iter <= 3
                println("  Current variance: $(round(current_variance; digits=4))")
            end
            
            # Get possible actions using CompoundState
            possible_actions = get_possible_actions(current_state)
            
            if verbose && iter <= 3
                println("  Possible actions: $possible_actions")
            end
            
            max_q = -Inf
            best_action = Vector{Int}()
            
            # Evaluate each action
            for action in possible_actions
                test_state = copy(current_state)
                total_cost = 0.0
                
                # Simulate measuring features in action
                for feat in action
                    sampled_value = sample_feature_value(
                        historical_data,
                        test_state,
                        feat
                    )
                    update!(test_state, feat, sampled_value)
                    total_cost += mdp.feature_costs[feat]
                end
                
                # Calculate new variance
                sim_var_new, theo_var_new = calculate_target_variance(
                    test_state,
                    historical_data,
                    target_coeffs,
                    feature_stats
                )
                new_variance = use_theoretical ? theo_var_new : sim_var_new
                
                # Calculate reward
                var_reduction = (current_variance - new_variance) / current_variance
                cost_penalty = total_cost / sum(mdp.feature_costs)
                r = log(var_reduction + 1) - log(cost_penalty)  # Changed to log ratio
                
                # Get next state index
                next_bool_state = copy(bool_state)
                next_bool_state[action] .= true
                next_s = state_to_index(next_bool_state)
                
                # Calculate Q-value with proper terminal state handling
                q = r + γ * (new_variance <= mdp.uncertainty_threshold ? 0.0 : V_old[next_s])
                
                if verbose && iter <= 3
                    println("\n  Evaluating action: $action")
                    println("    Variance reduction: $(round(var_reduction; digits=4))")
                    println("    Cost penalty: $(round(cost_penalty; digits=4))")
                    println("    Reward: $(round(r; digits=4))")
                    println("    Q-value: $(round(q; digits=4))")
                end
                
                # Track metrics
                state_action_key = state_idx * mdp.n_features + sum(2 .^(action .- 1))
                all_rewards[state_action_key] = r
                all_q_values[state_action_key] = q
                
                if q > max_q
                    max_q = q
                    best_action = action
                end
            end
            
            # Update policy and value function
            if max_q > -Inf
                V[state_idx] = max_q
                policy[state_idx] = best_action
                if best_action != old_action
                    n_actions_changed += 1
                end
            end
            
            max_diff = max(max_diff, abs(V[state_idx] - V_old[state_idx]))
            
            if verbose && iter <= 3
                println("\n  Selected action: $(best_action)")
                println("  New value: $(round(V[state_idx]; digits=4))")
            end
        end
        
        # Track convergence metrics with policy
        metric = track_convergence(
            iter, V, V_old, policy, copy(policy),
            all_rewards, all_q_values
        )
        push!(metrics, metric)  # Store metrics with policy snapshot
        
        if verbose
            if iter <= 3
                println("\nIteration $iter summary:")
                println("  Max value difference: $(round(max_diff; digits=6))")
                println("  Actions changed: $n_actions_changed")
            elseif iter % 50 == 0
                println("Iteration $iter - Max diff: $(round(max_diff; digits=6)), Changes: $n_actions_changed")
            end
        end
        
        # Update stable iterations counter
        if n_actions_changed == 0
            stable_iterations += 1
        else
            stable_iterations = 0
        end
        
        # Check convergence with new stability criterion
        if max_diff < tolerance || stable_iterations >= stable_count || all(isempty.(policy))
            if verbose
                println("\nConverged after $iter iterations")
                println("Final max difference: $(round(max_diff; digits=6))")
                println("Convergence reason: $(
                    all(isempty.(policy)) ? "Empty policy" : 
                    stable_iterations >= stable_count ? "Policy stable for $stable_count iterations" : 
                    "Value convergence"
                )")
            end
            return policy, V, true, metrics
        end
    end
    
    if verbose
        println("\nDid not converge after $max_iterations iterations")
    end
    return policy, V, false, metrics
end

"""
Simulate experiments using the optimized policy.
"""
function simulate_experiments(
    mdp::CompoundExperimentMDP,
    initial_state::CompoundState,
    historical_data::DataFrame,
    target_coeffs::Vector{Float64},
    feature_stats::DataFrame;
    use_theoretical::Bool=false,
    verbose::Bool=false,
    max_iterations::Int=1000
)::Tuple{BitVector, Float64, Bool, Vector{ConvergenceMetrics}}
    # Get optimal policy
    final_policy, V, converged, metrics = optimize_policy(
        mdp, initial_state, historical_data, target_coeffs, feature_stats;
        use_theoretical=use_theoretical,
        verbose=verbose,
        max_iterations=max_iterations
    )
    
    # Get state index for initial state
    state_idx = state_to_index(initial_state.measured)
    
    # Get recommended action from policy
    action = final_policy[state_idx]
    
    # Calculate total cost of action
    total_cost = isempty(action) ? 0.0 : sum(mdp.feature_costs[feat] for feat in action)
    
    # Update state with recommended action
    final_state = copy(initial_state.measured)
    if !isempty(action)
        final_state[action] .= true
    end
    
    if verbose
        println("\nFor initial state with F1,F2 measured:")
        println("Recommended action: ", isempty(action) ? "None" : "F$(join(action, ",F"))")
        println("Action cost: $total_cost")
    end
    
    return final_state, total_cost, converged, metrics
end

"""
Get the optimal action from policy for a given state
"""
function get_policy_action(policy::Vector{Vector{Int}}, state::CompoundState)
    # Convert CompoundState to BitVector for state_to_index
    state_idx = state_to_index(state.measured)
    return policy[state_idx]
end

"""
Analyze and compare optimization results
"""
function analyze_optimization_results(
    sim_results::Tuple,
    theo_results::Tuple,
    feature_costs::Vector{Float64},
    historical_data::DataFrame,
    feature_stats::DataFrame,
    initial_state::CompoundState
)
    final_state_sim, total_cost_sim, converged_sim, sim_metrics = sim_results
    final_state_theo, total_cost_theo, converged_theo, theo_metrics = theo_results
    
    # Get optimal actions from final policies for initial state
    sim_action = get_policy_action(sim_metrics[end].policy, initial_state)
    theo_action = get_policy_action(theo_metrics[end].policy, initial_state)
    
    # Print diagnostic information
    println("\nDiagnostic Information:")
    println("Initial state measured features: $(measured_features(initial_state))")
    println("Initial state index: $(state_to_index(initial_state.measured))")
    println("Similarity-based final policy length: $(length(sim_metrics[end].policy))")
    println("Theoretical final policy length: $(length(theo_metrics[end].policy))")
    println("Similarity-based next action: [F$(join(sim_action, ", F"))]")
    println("Theoretical next action: [F$(join(theo_action, ", F"))]")
    
    # Print results
    println("\nSimilarity-based approach:")
    println("  Planning phase:")
    println("    - Number of iterations: $(length(sim_metrics))")
    println("    - Final policy entropy: $(round(sim_metrics[end].policy_entropy; digits=4))")
    println("  Policy recommendation:")
    println("    - For initial state [F$(join(measured_features(initial_state), ",F"))]: Measure [F$(join(sim_action, ",F"))]")
    println("  Execution phase:")
    println("    - Total cost: $(round(total_cost_sim; digits=4))")
    println("    - Final measured features: $(findall(final_state_sim))")
    
    println("\nTheoretical approach:")
    println("  Planning phase:")
    println("    - Number of iterations: $(length(theo_metrics))")
    println("    - Final policy entropy: $(round(theo_metrics[end].policy_entropy; digits=4))")
    println("  Policy recommendation:")
    println("    - For initial state [F$(join(measured_features(initial_state), ",F"))]: Measure [F$(join(theo_action, ",F"))]")
    println("  Execution phase:")
    println("    - Total cost: $(round(total_cost_theo; digits=4))")
    println("    - Final measured features: $(findall(final_state_theo))")
    
    # Print cost comparison
    println("\nCost Comparison:")
    println("  Absolute difference: $(round(abs(total_cost_sim - total_cost_theo); digits=4))")
    println("  Relative difference: $(round(100 * abs(total_cost_sim - total_cost_theo) / total_cost_theo; digits=2))%")
    
    return sim_metrics, theo_metrics
end

"""
Plot feature selection comparison between similarity-based and theoretical approaches.
"""
function plot_feature_selection_strategy(
    sim_policy::Vector{Vector{Int}},
    theo_policy::Vector{Vector{Int}},
    n_features::Int64,
    initial_state::CompoundState
)
    p = plot(
        xlabel = "Approach",
        ylabel = "Feature",
        title = "Feature Selection Comparison",
        legend = :outerright,
        ylims = (0.5, n_features + 0.5),
        yticks = (1:n_features, ["F$i" for i in 1:n_features]),
        grid = true,
        gridstyle = :dash,
        gridalpha = 0.3,
        right_margin = 30mm
    )
    
    # Get single action for initial state from each policy
    sim_action = get_policy_action(sim_policy, initial_state)
    theo_action = get_policy_action(theo_policy, initial_state)
    
    # Plot similarity-based action
    if !isempty(sim_action)
        scatter!(fill(1, length(sim_action)), sim_action, 
                label="Similarity-based",
                marker=:circle,
                markersize=8,
                color=:blue)
    end
    
    # Plot theoretical action
    if !isempty(theo_action)
        scatter!(fill(2, length(theo_action)), theo_action,
                label="Theoretical",
                marker=:square,
                markersize=8,
                color=:red)
    end
    
    # Set x-axis ticks
    xticks!([1,2], ["Similarity", "Theoretical"])
    
    return p
end

"""
Plot convergence analysis comparing approaches
"""
function plot_convergence_analysis(
    sim_metrics::Vector{ConvergenceMetrics},
    theo_metrics::Vector{ConvergenceMetrics},
    final_state_sim::BitVector,
    final_state_theo::BitVector,
    n_features::Int64,
    initial_state::CompoundState
)
    # Value convergence plot
    p1 = plot(
        [m.iteration for m in sim_metrics],
        [m.max_value_diff for m in sim_metrics],
        label="Similarity-based",
        title="Value Function Convergence",
        xlabel="Iteration",
        ylabel="Max Value Difference",
        lw=2,
        legend=:outerright,
        grid=true,
        gridstyle=:dash,
        gridalpha=0.3,
        right_margin=30mm
    )
    plot!(
        [m.iteration for m in theo_metrics],
        [m.max_value_diff for m in theo_metrics],
        label="Theoretical",
        lw=2,
        linestyle=:dash
    )

    # Policy stability plot
    p2 = plot(
        [m.iteration for m in sim_metrics],
        [m.policy_entropy for m in sim_metrics],
        label="Similarity-based",
        title="Policy Stability",
        xlabel="Iteration",
        ylabel="Entropy",
        lw=2,
        legend=:outerright,
        grid=true,
        gridstyle=:dash,
        gridalpha=0.3,
        right_margin=30mm
    )
    plot!(
        [m.iteration for m in theo_metrics],
        [m.policy_entropy for m in theo_metrics],
        label="Theoretical",
        lw=2,
        linestyle=:dash
    )

    # Feature selection comparison plot
    p3 = plot_feature_selection_strategy(
        sim_metrics[end].policy,  # Use final policy
        theo_metrics[end].policy, # Use final policy
        n_features,
        initial_state
    )

    return p1, p2, p3
end

# Higher-level interface functions

"""
Run the complete value iteration analysis that matches the original CEED_simulation
"""
function value_iteration_analysis(; seed::Int=42,
                                    custom_coeffs::Union{Nothing,Vector{Float64}}=nothing,
                                    shared_data::Union{Nothing,NamedTuple}=nothing)
    # Set plotting defaults for publication-quality figures
    theme(:default)
    default(fontfamily="Computer Modern", framestyle=:box)

    # Define experimental parameters
    n_historical = 200  # Number of historical compounds
    n_features = 6      # Regular features
    feature_costs = [1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 10.0]  # Including target cost
    uncertainty_threshold = 0.1  # Target uncertainty level

    # Generate/obtain dataset.
    # ROUTE-1: if shared_data is provided, VI uses the SAME dataset as the CEED/IBMDP side
    # (unified, no one-iteration lag). Otherwise regenerate; per-iteration variation is driven
    # by `seed` and (optionally) explicit `custom_coeffs` so VI-Theo's optimum genuinely moves.
    if shared_data !== nothing
        historical_data = shared_data.historical_data
        target_coeffs   = shared_data.target_coeffs
        feature_stats   = shared_data.feature_stats
    else
        historical_data, target_coeffs, feature_stats = generate_historical_dataset(
            n_historical,
            n_features,  # Only regular features
            seed,
            false;  # Set add_noise to false
            use_random_coeffs = (custom_coeffs === nothing),
            custom_coeffs = custom_coeffs
        )
    end

    # Print coefficients used in analysis
    println("\nTarget coefficients used in analysis:")
    for (i, coeff) in enumerate(target_coeffs)
        println("  F$i: $(round(coeff; digits=4))")
    end

    # Display configuration
    println("Experiment Configuration:")
    println("  Number of compounds: $n_historical")
    println("  Regular features: $(n_features-1)")
    println("  Feature costs: $(feature_costs[1:end-1])")
    println("  Uncertainty threshold: $uncertainty_threshold")

    # Create MDP with experimental parameters
    mdp = CompoundExperimentMDP(
        n_features,  # Will add +1 internally for target
        feature_costs,
        uncertainty_threshold
    )

    # Initialize state with first two features measured
    Random.seed!(seed)
    initial_state = Dict{Int,Float64}()
    initial_state[1] = rand(Uniform(0, 100.0 * (1/n_features)))
    initial_state[2] = rand(Uniform(0, 100.0 * (2/n_features)))
    initial_compound_state = CompoundState(initial_state, n_features)  # n_features will be adjusted internally

    # Calculate initial uncertainties
    sim_variance, theo_variance = calculate_target_variance(
        initial_compound_state,
        historical_data,
        target_coeffs,
        feature_stats
    )

    # Run similarity-based approach
    println("\n=== Similarity-based Approach ===")
    final_state_sim, total_cost_sim, converged_sim, sim_metrics = simulate_experiments(
        mdp,
        initial_compound_state,
        historical_data,
        target_coeffs,
        feature_stats;
        use_theoretical=false,
        verbose=false,
        max_iterations=500
    )

    # Run theoretical approach
    println("\n=== Theoretical Approach ===")
    final_state_theo, total_cost_theo, converged_theo, theo_metrics = simulate_experiments(
        mdp,
        initial_compound_state,
        historical_data,
        target_coeffs,
        feature_stats;
        use_theoretical=true,
        verbose=false,
        max_iterations=500
    )

    # First run the optimization analysis
    sim_metrics, theo_metrics = analyze_optimization_results(
        (final_state_sim, total_cost_sim, converged_sim, sim_metrics),
        (final_state_theo, total_cost_theo, converged_theo, theo_metrics),
        feature_costs,
        historical_data,
        feature_stats,
        initial_compound_state
    )

    # Create all plots
    p1, p2, p3 = plot_convergence_analysis(
        sim_metrics,
        theo_metrics,
        final_state_sim,
        final_state_theo,
        n_features,
        initial_compound_state
    )

    # Combine plots
    p = plot(p1, p2, p3,
         layout=(3,1),
         size=(1000,1000),
         margin=10mm,
         right_margin=20mm,
         dpi=300)

    # Save the generated data to benchmark location
    data_dir = joinpath(@__DIR__, "..", "..", "benchmark", "data", "synthetic")
    if !isdir(data_dir)
        mkpath(data_dir)
    end
    
    # Save historical data
    CSV.write(joinpath(data_dir, "historical_data.csv"), historical_data)
    
    # Save target coefficients
    coeff_df = DataFrame(feature = 1:length(target_coeffs), coefficient = target_coeffs)
    CSV.write(joinpath(data_dir, "target_coefficients.csv"), coeff_df)
    
    # Save feature statistics  
    CSV.write(joinpath(data_dir, "feature_stats.csv"), feature_stats)
    
    # Save initial state
    initial_state_df = DataFrame(
        feature = measured_features(initial_compound_state),
        value = [initial_compound_state.values[f] for f in measured_features(initial_compound_state)]
    )
    CSV.write(joinpath(data_dir, "initial_state.csv"), initial_state_df)

    # Create comparison DataFrame
    println("\nValue Iteration Actions:")
    # Get initial state actions from both VI approaches
    sim_action = get_policy_action(sim_metrics[end].policy, initial_compound_state)
    theo_action = get_policy_action(theo_metrics[end].policy, initial_compound_state)

    println("Similarity-based VI: ", sim_action)
    println("Theoretical VI: ", theo_action)

    policy_comparison = DataFrame(
        Threshold = [0.1, 0.0],
        VI_sim = ["[$(join(sim_action, ","))]", "[$(join(sim_action, ","))]"],  # Format as strings
        VI_theo = ["[$(join(theo_action, ","))]", "[$(join(theo_action, ","))]"]  # Format as strings
    )

    # Display the comparison
    println("\nPolicy Comparison:")
    println(policy_comparison)

    return p, policy_comparison
end

# Convenience wrapper functions for backward compatibility
similarity_based_value_iteration() = value_iteration_analysis()
theoretical_value_iteration() = value_iteration_analysis()
setup_mdp_framework() = value_iteration_analysis()
run_mdp_analysis() = value_iteration_analysis()

end 