module CEEDUtilities

using DataFrames, CSV
using Statistics, StatsBase
using Random
using Dates
using Plots
using Distributions
using ..GenerativeDesigns: Evidence, QuadraticDistance, Exponential, Variance, DistanceBased
using MCTS

export gradient_bar, make_labels_modi, cost_bias_tuple
export load_synthetic_data, generate_synthetic_data, save_synthetic_data, generate_initial_state
export setup_ceed_configuration, setup_default_plotting_options
export create_experiments_dict, setup_enhanced_sampler, setup_solver
export find_project_data_directory

"""
    gradient_bar(values::Vector, colors=[:red, :yellow, :green])

Create a gradient color bar for visualization.

# Arguments
- `values::Vector`: Values to create gradient for
- `colors`: Color scheme for gradient (default: red-yellow-green)

# Returns
- Array of colors corresponding to the values
"""
function gradient_bar(values::Vector, colors=[:red, :yellow, :green])
    n = length(values)
    if n == 0
        return []
    end
    
    # Normalize values to [0, 1]
    min_val, max_val = extrema(values)
    if min_val == max_val
        return fill(colors[2], n)  # Use middle color if all values are the same
    end
    
    normalized = (values .- min_val) ./ (max_val - min_val)
    
    # Map normalized values to colors
    color_indices = round.(Int, normalized .* (length(colors) - 1)) .+ 1
    color_indices = clamp.(color_indices, 1, length(colors))
    
    return colors[color_indices]
end

"""
    make_labels_modi(designs; prefix="Design")

Create modified labels for design visualization.

# Arguments
- `designs`: Design results from CEED analysis
- `prefix::String`: Prefix for labels (default: "Design")

# Returns
- Vector of formatted label strings
"""
function make_labels_modi(designs; prefix="Design")
    labels = String[]
    
    for (i, design) in enumerate(designs)
        if haskey(design[2], :arrangement) && !isempty(design[2][:arrangement])
            action_str = join(design[2][:arrangement], ", ")
            label = "$prefix $i: [$action_str]"
        else
            label = "$prefix $i: [No actions]"
        end
        push!(labels, label)
    end
    
    return labels
end

"""
    cost_bias_tuple()

Return predefined cost bias tuples for different optimization strategies.

# Returns
- Tuple containing (money_biased, time_biased) parameter tuples
"""
function cost_bias_tuple()
    money_biased = (1, 0)  # Prioritize monetary cost, ignore time
    time_biased = (0, 1)   # Prioritize time, ignore monetary cost
    return money_biased, time_biased
end

"""
    find_project_data_directory(start_dir::String = @__DIR__)

Find the project data directory by searching up the directory tree.

# Arguments
- `start_dir::String`: Starting directory for search (default: current directory)

# Returns
- Path to the data directory containing synthetic data files
"""
function find_project_data_directory(start_dir::String = @__DIR__)
    current_dir = start_dir
    
    # Search up the directory tree for the project root
    while true
        # Look for benchmark/data/synthetic directory (primary location)
        potential_data_dir = joinpath(current_dir, "benchmark", "data", "synthetic")
        if isdir(potential_data_dir) && isfile(joinpath(potential_data_dir, "historical_data.csv"))
            return potential_data_dir
        end
        
        # Look for CEED_simulation/results/data directory (legacy external location)
        potential_data_dir = joinpath(current_dir, "CEED_simulation", "results", "data")
        if isdir(potential_data_dir) && isfile(joinpath(potential_data_dir, "historical_data.csv"))
            return potential_data_dir
        end
        
        # Move up one directory
        parent_dir = dirname(current_dir)
        if parent_dir == current_dir  # We've reached the root
            error("Could not find project data directory with synthetic data files")
        end
        current_dir = parent_dir
    end
end

"""
    load_synthetic_data(data_dir::String = find_project_data_directory())

Load all synthetic data files required for CEED analysis.

# Arguments
- `data_dir::String`: Path to data directory (default: auto-detected)

# Returns
- NamedTuple containing all loaded data: (historical_data, feature_stats, target_coeffs, initial_state_df)
"""
function load_synthetic_data(data_dir::String = find_project_data_directory())
    @info "Loading synthetic data from: $data_dir"
    
    # Load all required data files
    historical_data = CSV.read(joinpath(data_dir, "historical_data.csv"), DataFrame)
    feature_stats = CSV.read(joinpath(data_dir, "feature_stats.csv"), DataFrame)
    target_coeffs = CSV.read(joinpath(data_dir, "target_coefficients.csv"), DataFrame).coefficient
    initial_state_df = CSV.read(joinpath(data_dir, "initial_state.csv"), DataFrame)
    
    @info "Successfully loaded synthetic data:"
    @info "  - Historical data: $(nrow(historical_data)) rows × $(ncol(historical_data)) columns"
    @info "  - Feature stats: $(nrow(feature_stats)) rows × $(ncol(feature_stats)) columns" 
    @info "  - Target coefficients: $(length(target_coeffs)) coefficients"
    @info "  - Initial state: $(nrow(initial_state_df)) features"
    
    return (
        historical_data = historical_data,
        feature_stats = feature_stats, 
        target_coeffs = target_coeffs,
        initial_state_df = initial_state_df
    )
end

"""
    generate_synthetic_data(; n_compounds=200, n_features=6, seed=42, add_noise=false, use_random_coeffs=true)

Generate synthetic historical dataset for CEED analysis.

# Arguments
- `n_compounds::Int`: Number of historical compounds (default: 200)
- `n_features::Int`: Number of features (default: 6)
- `seed::Int`: Random seed for reproducibility (default: 42)
- `add_noise::Bool`: Whether to add noise to target values (default: false)
- `use_random_coeffs::Bool`: Use random coefficients instead of fixed ones (default: true)

# Returns
- Tuple of (historical_data, target_coefficients, feature_stats)
"""
function generate_synthetic_data(;
    n_compounds::Int = 200,
    n_features::Int = 6,
    seed::Int = 42,
    add_noise::Bool = false,
    use_random_coeffs::Bool = true
)
    # Set seed for reproducible data generation
    Random.seed!(seed)
    
    @info "Generating synthetic data with parameters:"
    @info "  - Compounds: $n_compounds"
    @info "  - Features: $n_features" 
    @info "  - Seed: $seed"
    @info "  - Add noise: $add_noise"
    @info "  - Random coefficients: $use_random_coeffs"
    
    # Create feature distributions
    feature_distributions = Vector{Distribution}(undef, n_features)
    for i in 1:n_features
        # Each feature has different mean and variance
        mean_val = 8.0 + (i * 8.0)  # Features range from ~8 to ~56
        std_val = mean_val * 0.3     # 30% coefficient of variation
        feature_distributions[i] = truncated(Normal(mean_val, std_val), 0.0, mean_val * 2)
    end
    
    # Generate target coefficients
    if use_random_coeffs
        # Generate random coefficients and normalize
        coeff_seed = Int(round(time() * 1000)) % 10000
        Random.seed!(coeff_seed)
        raw_coeffs = rand(n_features)
        target_coefficients = raw_coeffs / sum(raw_coeffs)
        @info "Using random coefficients (seed: $coeff_seed):"
        Random.seed!(seed)  # Reset to original seed
    else
        # Use fixed coefficients (decreasing importance)
        target_coefficients = [0.3, 0.25, 0.2, 0.15, 0.07, 0.03]
        @info "Using fixed coefficients:"
    end
    
    for (i, coeff) in enumerate(target_coefficients)
        @info "  Feature $i: $(round(coeff, digits=4))"
    end
    
    # Generate synthetic data
    data = zeros(n_compounds, n_features + 1)  # +1 for target
    
    for i in 1:n_compounds
        # Generate feature values
        for j in 1:n_features
            data[i, j] = rand(feature_distributions[j])
        end
        
        # Calculate target value as linear combination
        target_value = sum(target_coefficients .* data[i, 1:n_features])
        
        # Add noise if requested
        if add_noise
            noise = rand(Normal(0.0, target_value * 0.1))  # 10% noise
            target_value += noise
        end
        
        data[i, n_features + 1] = target_value
    end
    
    # Create DataFrame
    feature_cols = ["feature_$j" for j in 1:n_features]
    historical_data = DataFrame(data, [feature_cols; "target"])
    
    # Calculate feature statistics
    feature_stats = DataFrame(
        feature = 1:n_features,
        mean = [mean(historical_data[:, i]) for i in 1:n_features],
        std = [std(historical_data[:, i]) for i in 1:n_features],
        min = [minimum(historical_data[:, i]) for i in 1:n_features],
        max = [maximum(historical_data[:, i]) for i in 1:n_features]
    )
    
    @info "✅ Synthetic data generation complete!"
    @info "  - Historical data: $(nrow(historical_data)) compounds × $(ncol(historical_data)) columns"
    @info "  - Target range: [$(round(minimum(historical_data.target), digits=2)), $(round(maximum(historical_data.target), digits=2))]"
    
    return historical_data, target_coefficients, feature_stats
end

"""
    generate_initial_state(; n_features=6, seed=42)

Generate initial experimental state with features 1 and 2 measured.

# Arguments
- `n_features::Int`: Total number of features (default: 6)
- `seed::Int`: Random seed for reproducibility (default: 42)

# Returns
- DataFrame with initial state measurements
"""
function generate_initial_state(; n_features::Int = 6, seed::Int = 42)
    Random.seed!(seed)
    
    # Generate values for first two features
    feature_1_value = rand(Uniform(5.0, 15.0))   # Feature 1 range
    feature_2_value = rand(Uniform(13.0, 23.0))  # Feature 2 range
    
    initial_state_df = DataFrame(
        feature = [1, 2],
        value = [feature_1_value, feature_2_value]
    )
    
    @info "Generated initial state:"
    @info "  - Feature 1: $(round(feature_1_value, digits=2))"
    @info "  - Feature 2: $(round(feature_2_value, digits=2))"
    
    return initial_state_df
end

"""
    save_synthetic_data(historical_data, target_coefficients, feature_stats, initial_state_df; data_dir="benchmark/data/synthetic")

Save all synthetic data to CSV files in specified directory.

# Arguments
- `historical_data::DataFrame`: Historical experimental data
- `target_coefficients::Vector`: Feature coefficients for target calculation
- `feature_stats::DataFrame`: Feature statistics
- `initial_state_df::DataFrame`: Initial experimental state
- `data_dir::String`: Directory to save files (default: "benchmark/data/synthetic")

# Returns
- Path to the data directory where files were saved
"""
function save_synthetic_data(
    historical_data::DataFrame, 
    target_coefficients::Vector, 
    feature_stats::DataFrame, 
    initial_state_df::DataFrame; 
    data_dir::String = "benchmark/data/synthetic"
)
    # Create directory if it doesn't exist
    if !isdir(data_dir)
        mkpath(data_dir)
    end
    
    @info "Saving synthetic data to: $data_dir"
    
    # Save historical data
    CSV.write(joinpath(data_dir, "historical_data.csv"), historical_data)
    @info "✓ Saved historical_data.csv ($(nrow(historical_data)) compounds)"
    
    # Save target coefficients
    coeff_df = DataFrame(
        feature = 1:length(target_coefficients),
        coefficient = target_coefficients
    )
    CSV.write(joinpath(data_dir, "target_coefficients.csv"), coeff_df)
    @info "✓ Saved target_coefficients.csv ($(length(target_coefficients)) coefficients)"
    
    # Save feature statistics  
    CSV.write(joinpath(data_dir, "feature_stats.csv"), feature_stats)
    @info "✓ Saved feature_stats.csv ($(nrow(feature_stats)) features)"
    
    # Save initial state
    CSV.write(joinpath(data_dir, "initial_state.csv"), initial_state_df)
    @info "✓ Saved initial_state.csv ($(nrow(initial_state_df)) measured features)"
    
    @info "✅ All synthetic data files saved successfully!"
    return data_dir
end

"""
    create_experiments_dict(; feature_range=3:6, base_cost=1.5, cost_increment=0.3, time_cost=1)

Create experiments dictionary with cost structure.

# Arguments
- `feature_range`: Range of features to include (default: 3:6)
- `base_cost::Float64`: Base monetary cost (default: 1.5)
- `cost_increment::Float64`: Cost increment per feature (default: 0.3)
- `time_cost::Float64`: Time cost for all features (default: 1)

# Returns
- Dictionary mapping feature names to [monetary_cost, time_cost] vectors
"""
function create_experiments_dict(; feature_range=3:6, base_cost=1.5, cost_increment=0.3, time_cost=1)
    experiments = Dict{String, Vector{Float64}}()
    
    for i in feature_range
        feature_name = "feature_$i"
        monetary_cost = base_cost + (i - first(feature_range)) * cost_increment
        experiments[feature_name] = [monetary_cost, time_cost]
    end
    
    @info "Created experiments dictionary with $(length(experiments)) features:"
    for (feature, cost) in experiments
        @info "  $feature: [monetary: $(cost[1]), time: $(cost[2])]"
    end
    
    return experiments
end

"""
    setup_enhanced_sampler(historical_data::DataFrame; target="target", lambda=0.5, conditional_range=Dict("target" => [0.5, 1.0]))

Set up enhanced distance-based sampler with conditional support.

# Arguments
- `historical_data::DataFrame`: Historical data for sampling
- `target::String`: Target column name (default: "target")
- `lambda::Float64`: Exponential similarity parameter (default: 0.5)
- `conditional_range::Dict`: Conditional range constraints (default: target in [0.5, 1.0])

# Returns
- Named tuple containing sampler, uncertainty, and weights functions
"""
function setup_enhanced_sampler(historical_data::DataFrame; target="target", lambda=0.5, conditional_range=Dict("target" => [0.5, 1.0]))
    @info "Setting up enhanced distance-based sampler with conditional support"
    
    # Define features and target
    features = ["feature_$i" for i in 1:6]
    target_vec = [target]
    
    # Set up distance functions
    distances = Dict()
    foreach(e -> push!(distances, e => QuadraticDistance(; λ = 1)), features)
    foreach(e -> push!(distances, e => QuadraticDistance(; λ = 1)), target_vec)
    
    # Setup enhanced distance-based sampler with conditional support
    sampler_setup = DistanceBased(
        historical_data;
        target = target,
        uncertainty = Variance(),
        similarity = Exponential(; λ = lambda),
        distance = distances,
        conditional_range = conditional_range
    )
    
    @info "Enhanced sampler configured with:"
    @info "  - Target: $target"
    @info "  - Lambda: $lambda"
    @info "  - Conditional range: $conditional_range"
    
    return sampler_setup
end

"""
    setup_solver(; n_iterations=1000, depth=6, exploration_constant=20.0, tree_info=true, keep_tree=true)

Set up MCTS solver with specified parameters.

# Arguments
- `n_iterations::Int`: Number of MCTS iterations (default: 1000)
- `depth::Int`: Search depth (default: 6)
- `exploration_constant::Float64`: Exploration parameter (default: 20.0)
- `tree_info::Bool`: Whether to include tree information (default: true)
- `keep_tree::Bool`: Whether to keep the search tree (default: true)

# Returns
- Configured DPWSolver instance
"""
function setup_solver(; n_iterations=1000, depth=6, exploration_constant=20.0, tree_info=true, keep_tree=true)
    solver = DPWSolver(
        n_iterations = n_iterations,
        depth = depth,
        exploration_constant = exploration_constant,
        tree_in_info = tree_info,
        keep_tree = keep_tree
    )
    
    @info "MCTS solver configured with:"
    @info "  - Iterations: $n_iterations"
    @info "  - Depth: $depth"
    @info "  - Exploration constant: $exploration_constant"
    
    return solver
end

"""
    setup_default_plotting_options(; money_biased=(1,0))

Set up default plotting and saving options for CEED analysis.

# Arguments
- `money_biased::Tuple`: Cost bias parameters (default: (1,0))

# Returns
- Named tuple containing (plot_opt, save_opt)
"""
function setup_default_plotting_options(; money_biased=(1,0))
    plot_opt = Dict(
        :title => "Enhanced CEED Analysis",
        :xlabel => "Total Cost",
        :ylabel => "Uncertainty (%)",
        :legendtitle => "Cost_tradeoff = $money_biased",
        :ylim => [0, 1],
        :legend => :outerright,        
        :size => (1000, 600),         
        :right_margin => 30Plots.mm,   
        :margin => 10Plots.mm         
    )
    
    save_opt = Dict(
        :savefig => true,
        :perturbed => false,
        :folder_name => "enhanced_CEED",
        :file_extension => ".png",
        :cost_bias => "money",
        :parallel_assays_NO => 2,
        :interventions_str => "",
        :event_date_cut_off => Dates.now()
    )
    
    return (plot_opt = plot_opt, save_opt = save_opt)
end

"""
    setup_ceed_configuration(data_dir::String = find_project_data_directory())

Set up complete CEED configuration including data loading, sampler, solver, and options.

# Arguments
- `data_dir::String`: Path to data directory (default: auto-detected)

# Returns
- Named tuple containing all configuration components
"""
function setup_ceed_configuration(; data_dir::Union{String, Nothing} = nothing, generate_if_missing::Bool = true)
    @info "="^60
    @info "SETTING UP COMPLETE CEED CONFIGURATION"
    @info "="^60
    
    # Try to find and load existing data, or generate if missing
    if data_dir === nothing
        try
            data_dir = find_project_data_directory()
            @info "Found existing data directory: $data_dir"
        catch e
            if generate_if_missing
                @info "No existing data found. Generating new synthetic data..."
                data_dir = "benchmark/data/synthetic"
                historical_data, target_coefficients, feature_stats = generate_synthetic_data()
                initial_state_df = generate_initial_state()
                save_synthetic_data(historical_data, target_coefficients, feature_stats, initial_state_df; data_dir=data_dir)
                @info "✅ Synthetic data generated and saved to: $data_dir"
            else
                rethrow(e)
            end
        end
    end
    
    # Load data
    data = load_synthetic_data(data_dir)
    
    # Create initial state
    ceed_initial_state = Evidence(
        "feature_1" => data.initial_state_df[data.initial_state_df.feature .== 1, :value][1],
        "feature_2" => data.initial_state_df[data.initial_state_df.feature .== 2, :value][1]
    )
    
    # Create experiments
    experiments = create_experiments_dict()
    
    # Setup sampler
    sampler_setup = setup_enhanced_sampler(data.historical_data)
    
    # Setup solver
    solver = setup_solver()
    
    # Setup plotting options
    money_biased, time_biased = cost_bias_tuple()
    plot_save_opts = setup_default_plotting_options(; money_biased = money_biased)
    
    @info "✅ CEED configuration setup complete!"
    @info "Initial features measured:"
    for (feature, value) in pairs(ceed_initial_state)
        @info "  $feature: $(round(value; digits=4))"
    end
    
    return (
        data = data,
        ceed_initial_state = ceed_initial_state,
        experiments = experiments,
        sampler_setup = sampler_setup,
        solver = solver,
        money_biased = money_biased,
        time_biased = time_biased,
        plot_opt = plot_save_opts.plot_opt,
        save_opt = plot_save_opts.save_opt,
        # Analysis parameters
        threshold_NO = 11,
        parallel_assays_NO = 2,
        target_condition = Dict("target" => [0.5, 1.0]),
        conditional_weights_thred = 0.8
    )
end

end # module CEEDUtilities 