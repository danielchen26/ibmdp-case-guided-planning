module RIPK1

# Package management and setup
using Pkg

# Package management handled at project level - commented out for execution within CEEDesigns project
# let
#     # Get the path to CEEDesigns (parent directory of RIPK1_Enhanced)
#     ceed_path = abspath(joinpath(@__DIR__, ".."))
#     
#     # Check if we're in a project environment
#     if !isnothing(Pkg.project().path)
#         # Check if CEEDesigns is already in the project
#         deps = Pkg.dependencies()
#         ceed_in_project = any(dep -> dep.second.name == "CEEDesigns", deps)
#         
#         if !ceed_in_project
#             println("📦 Adding CEEDesigns from local path: $ceed_path")
#             Pkg.develop(path=ceed_path)
#             Pkg.instantiate()
#         end
#     end
# end

using DataFrames, CSV
using Plots
using Statistics

# Set publication-quality plotting backend
Plots.gr()  # Use GR backend for consistency
ENV["GKSwstype"] = "100"  # Non-interactive backend to avoid GUI issues
using CEEDesigns.GenerativeDesigns: Evidence, QuadraticDistance, DistanceBased, Exponential, Variance
using CEEDesigns.GenerativeDesigns: perform_ensemble_designs, process_ensemble_results_enhanced
using CEEDesigns.GenerativeDesigns: find_max_likelihood_action_sets_with_utility
using CEEDesigns.GenerativeDesigns: plot_multiple_max_likelihood_action_sets, plot_ensemble_pareto
using CEEDesigns.GenerativeDesigns: validate_ensemble_frequencies
using MCTS
using Random

# Set random seed for reproducibility
Random.seed!(42)

# ============= EXPORTS =============
export RIPK1_CONFIG, AnalysisSetup
export setup_ripk1_analysis, run_single_molecule, run_all_molecules
export create_comparison_plots, generate_final_report
export select_representative_rows, preprocess_ripk1_data, setup_ripk1_distances

# ============= CONFIGURATION =============
"""
    RIPK1_CONFIG

Central configuration for RIPK1 analysis parameters.
"""
const RIPK1_CONFIG = (
    # Experiment costs: [monetary_cost, time_cost]
    experiments = Dict(
        "100nM_PgP" => [400, 7],
        "100nM_BCRP" => [400, 7],
        "1uM_PgP" => [400, 7],
        "kpuu" => [4000, 21],
    ),
    
    # Analysis parameters
    money_biased = (1.0, 0.0),           # Prioritize monetary cost
    parallel_assays_NO = 3,               # Max parallel assays
    threshold_NO = 11,                    # Number of threshold levels
    
    # Terminal conditions for conditional CEED
    target_condition = Dict("kpuu" => [0.5, 1.0]),
    conditional_weights_thred = 0.8,
    
    # Ensemble parameters
    belief_thresholds = [0.9],            # τ values to analyze
    
    # Solver parameters
    solver_params = (
        n_iterations = 20_000,
        exploration_constant = 5.0,
        depth = 11,
        tree_in_info = true,
        keep_tree = true
    ),
    
    # Distance-based sampler parameters
    sampler_params = (
        target = "kpuu",
        lambda = 0.5,
        in_silico_lambda = 50,
        physical_lambda = 200
    ),
    
    # Scenario names
    scenario_names = [
        "Scenario1_Low_kpuu_High_PgP_High_BCRP",
        "Scenario2_Low_kpuu_High_PgP_Low_BCRP",
        "Scenario3_High_kpuu_Low_PgP_Low_BCRP",
        "Scenario4_Low_kpuu_Low_PgP_High_BCRP"
    ],
    
    # Output configuration
    results_dir_name = "representative_molecules_fixed",
    comparison_plot_name = "compare_plots_enforced.png"
)

# ============= ANALYSIS SETUP STRUCTURE =============
"""
    AnalysisSetup

Container for all analysis components after initialization.
"""
struct AnalysisSetup
    data::DataFrame
    data_for_sampler::DataFrame
    compound_ids::Vector
    sampler::Function
    uncertainty::Function
    weights::Function
    solver::DPWSolver
    state_init_list::Vector
    selected_data::DataFrame
    results_base::String
end

# ============= SETUP FUNCTIONS =============
"""
    setup_ripk1_analysis(data_path::String; results_base::String="")

Initialize all components needed for RIPK1 analysis.

# Arguments
- `data_path::String`: Path to the RIPK1 data file
- `results_base::String`: Base directory for results (optional)

# Returns
- `AnalysisSetup`: Struct containing all initialized components
"""
function setup_ripk1_analysis(data_path::String; results_base::String="")
    println("\n⚙️ Setting up RIPK1 analysis...")
    
    # Load and preprocess data
    data_full = DataFrame(CSV.File(data_path))
    data, compound_ids = preprocess_ripk1_data(data_full)
    println("  ✓ Loaded $(nrow(data)) compounds with $(ncol(data)) features")
    
    # Setup distances
    distances = setup_ripk1_distances()
    
    # Prepare data for sampler
    data_for_sampler = select(data, Not(:compound_id))
    
    # Create sampler
    println("  Setting up sampler...")
    (; sampler, uncertainty, weights) = DistanceBased(
        data_for_sampler;
        target = RIPK1_CONFIG.sampler_params.target,
        uncertainty = Variance(),
        similarity = Exponential(; λ = RIPK1_CONFIG.sampler_params.lambda),
        distance = distances,
    )
    
    # Setup solver
    println("  Setting up solver...")
    solver = DPWSolver(;
        n_iterations = RIPK1_CONFIG.solver_params.n_iterations,
        exploration_constant = RIPK1_CONFIG.solver_params.exploration_constant,
        depth = RIPK1_CONFIG.solver_params.depth,
        tree_in_info = RIPK1_CONFIG.solver_params.tree_in_info,
        keep_tree = RIPK1_CONFIG.solver_params.keep_tree
    )
    
    # Select representative compounds
    println("  Selecting representative compounds...")
    state_init_list, selected_data = select_representative_rows(data)
    println("  ✓ Found $(length(state_init_list)) representative compounds")
    
    if size(selected_data, 1) > 0
        println("  📊 kpuu values: ", round.(selected_data.kpuu, digits=3))
    end
    
    # Setup results directory
    if isempty(results_base)
        results_base = joinpath(dirname(data_path), "..", "results", RIPK1_CONFIG.results_dir_name)
    end
    mkpath(results_base)
    
    println("  ✓ Results will be saved to: $results_base")
    
    return AnalysisSetup(
        data, data_for_sampler, compound_ids,
        sampler, uncertainty, weights, solver,
        state_init_list, selected_data, results_base
    )
end

# ============= ANALYSIS FUNCTIONS =============
"""
    run_single_molecule(setup::AnalysisSetup, scenario_idx::Int, scenario_name::String; 
                        ensemble_size::Int=10, save_plots::Bool=true)

Run analysis for a single molecule/scenario.

# Arguments
- `setup::AnalysisSetup`: Analysis setup from setup_ripk1_analysis
- `scenario_idx::Int`: Index of the scenario (1-based)
- `scenario_name::String`: Name of the scenario
- `ensemble_size::Int`: Number of ensemble runs
- `save_plots::Bool`: Whether to save plots to disk

# Returns
- `scenario_results::Dict`: Results for each belief threshold
"""
function run_single_molecule(setup::AnalysisSetup, scenario_idx::Int, scenario_name::String; 
                            ensemble_size::Int=10, save_plots::Bool=true, 
                            tau_values::Vector{Float64}=RIPK1_CONFIG.belief_thresholds)
    
    println("\n📁 Processing $scenario_name")
    
    # Get initial state for this scenario
    if scenario_idx > length(setup.state_init_list)
        println("  ⚠️ No compound data for scenario $scenario_idx")
        return Dict()
    end
    
    state_init = setup.state_init_list[scenario_idx]
    
    # Get kpuu value
    kpuu_value = scenario_idx <= size(setup.selected_data, 1) ? 
                 setup.selected_data.kpuu[scenario_idx] : missing
    println("  kpuu = $(ismissing(kpuu_value) ? "N/A" : round(kpuu_value, digits=3))")
    
    # Show initial state
    println("  Initial state:")
    for (key, val) in pairs(state_init)
        println("    $key = $(round(val[1], digits=3))")
    end
    println("  Initial uncertainty: $(round(setup.uncertainty(state_init), digits=3))")
    
    # Create scenario directory
    scenario_dir = joinpath(setup.results_base, scenario_name)
    mkpath(scenario_dir)
    
    # Create and save metadata about this scenario
    metadata_df = DataFrame(
        scenario = [scenario_name],
        kpuu = [ismissing(kpuu_value) ? NaN : kpuu_value],
        initial_uncertainty = [round(setup.uncertainty(state_init), digits=3)],
        ensemble_size = [ensemble_size],
        parallel_assays = [RIPK1_CONFIG.parallel_assays_NO],
        threshold_levels = [RIPK1_CONFIG.threshold_NO],
        tau_values_str = [join(tau_values, ", ")]
    )
    
    # Add initial state columns
    for (key, val) in pairs(state_init)
        metadata_df[!, Symbol("initial_" * key)] = [round(val[1], digits=3)]
    end
    
    # Save metadata
    CSV.write(joinpath(scenario_dir, "scenario_metadata.csv"), metadata_df)
    println("  📋 Saved scenario metadata")
    
    # Store results for all tau values
    scenario_results = Dict()
    
    for tau in tau_values
        println("  Processing τ = $tau")
        
        # Run ensemble analysis
        ensemble_results = perform_ensemble_designs(
            RIPK1_CONFIG.experiments;
            sampler = setup.sampler,
            uncertainty = setup.uncertainty,
            thresholds = RIPK1_CONFIG.threshold_NO,
            evidence = state_init,
            weights = setup.weights,
            data = setup.data_for_sampler,
            terminal_condition = (RIPK1_CONFIG.target_condition, RIPK1_CONFIG.conditional_weights_thred),
            realized_uncertainty = true,
            solver = setup.solver,
            repetitions = 0,
            mdp_options = (
                conditional_constraints_enabled = true,
                max_parallel = RIPK1_CONFIG.parallel_assays_NO,
                costs_tradeoff = RIPK1_CONFIG.money_biased
            ),
            N = ensemble_size,
            thred_set = [tau]
        )
        
        # Process results
        try
            df, plt_hist, ensemble_front = process_ensemble_results_enhanced(
                ensemble_results, tau;
                save_dir = scenario_dir,
                save_plots = save_plots
            )
            
            # Save the DataFrame as CSV for further analysis
            csv_filename = joinpath(scenario_dir, "ensemble_results_tau_$(tau).csv")
            CSV.write(csv_filename, df)
            println("    📄 Saved CSV: ensemble_results_tau_$(tau).csv")
            
            # Validate ensemble frequencies
            validation_stats = validate_ensemble_frequencies(df)
            println("    📊 Ensemble validation:")
            println("       - Success rate: $(validation_stats.success_rate)% ($(validation_stats.avg_ensemble)/$(validation_stats.max_ensemble) avg)")
            println("       - Range: $(validation_stats.min_ensemble)-$(validation_stats.max_ensemble) successful runs per threshold")
            
            # Action set legend plots removed per user request
            
            scenario_results[tau] = (
                df = df, 
                plt_hist = plt_hist, 
                ensemble_front = ensemble_front,
                validation = validation_stats
            )
            println("    ✅ Generated $(nrow(df)) action sets from $(validation_stats.max_ensemble) ensemble runs")
            
        catch e
            println("    ⚠️ Error processing results: $e")
        end
    end
    
    # Create combined plots if multiple tau values
    if save_plots && length(tau_values) > 1 && length(scenario_results) > 0
        create_scenario_combined_plots(scenario_results, scenario_dir)
    end
    
    println("  ✅ Scenario complete!")
    return scenario_results
end

"""
    run_all_molecules(setup::AnalysisSetup; ensemble_size::Int=10, save_plots::Bool=true)

Run analysis for all selected molecules/scenarios.

# Arguments
- `setup::AnalysisSetup`: Analysis setup from setup_ripk1_analysis
- `ensemble_size::Int`: Number of ensemble runs (default: 10)
- `save_plots::Bool`: Whether to save plots

# Returns
- `all_results::Dict`: Results for all scenarios
"""
function run_all_molecules(setup::AnalysisSetup; ensemble_size::Int=10, save_plots::Bool=true, 
                          tau_values::Vector{Float64}=RIPK1_CONFIG.belief_thresholds)
    println("\n" * "="^80)
    println("🚀 Starting RIPK1 Analysis")
    println("  Ensemble size: $ensemble_size")
    println("="^80)
    all_results = Dict()
    
    for (idx, scenario_name) in enumerate(RIPK1_CONFIG.scenario_names)
        # Run single molecule analysis
        scenario_results = run_single_molecule(
            setup, idx, scenario_name;
            ensemble_size = ensemble_size,
            save_plots = save_plots,
            tau_values = tau_values
        )
        
        # Store results
        if !isempty(scenario_results)
            all_results[scenario_name] = scenario_results
        end
        
    end
    
    return all_results
end

"""
    create_scenario_combined_plots(scenario_results::Dict, scenario_dir::String)

Create combined plots for a single scenario with multiple tau values.
"""
function create_scenario_combined_plots(scenario_results::Dict, scenario_dir::String)
    # Extract tau values from the results keys
    taus = sort(collect(keys(scenario_results)))
    
    # Combined histogram
    try
        hist_plots = [scenario_results[tau].plt_hist for tau in taus if haskey(scenario_results, tau)]
        if length(hist_plots) > 0
            combined_hist = plot(hist_plots..., 
                               layout=(1, length(hist_plots)), 
                               size=(2800, 1200),  # Larger size for publication
                               dpi=300,  # High DPI for publication quality
                               plot_title="Belief thresholds: " * join(taus, ", "))
            savefig(combined_hist, joinpath(scenario_dir, "hist_ensemble_belief_Money.png"))
        end
    catch e
        println("    ⚠️ Error creating combined histogram: $e")
    end
    
    # Combined Pareto fronts
    try
        pareto_plots = [scenario_results[tau].ensemble_front for tau in taus if haskey(scenario_results, tau)]
        if length(pareto_plots) > 0
            combined_pareto = plot(pareto_plots..., 
                                 layout=(1, length(pareto_plots)), 
                                 size=(3000, 800),  # Wider for publication
                                 dpi=300,  # High DPI
                                 plot_title="Belief thresholds: " * join(taus, ", "))
            savefig(combined_pareto, joinpath(scenario_dir, "pareto_ensemble_belief_Money.png"))
        end
    catch e
        println("    ⚠️ Error creating combined Pareto: $e")
    end
    
    # Multi-tau comparison
    try
        dfs_list = [scenario_results[tau].df for tau in taus if haskey(scenario_results, tau)]
        if length(dfs_list) > 0
            plt_taus = plot_multiple_max_likelihood_action_sets(dfs_list, collect(taus))
            savefig(plt_taus, joinpath(scenario_dir, "taus_pareto.png"))
        end
    catch e
        println("    ⚠️ Error creating taus_pareto: $e")
    end
end

"""
    create_comparison_plots(all_results::Dict, setup::AnalysisSetup)

Create comparison plots across all scenarios.

# Arguments
- `all_results::Dict`: Results from all scenarios
- `setup::AnalysisSetup`: Analysis setup containing selected_data and results_base

# Returns
- `success::Bool`: Whether the plots were created successfully
"""
function create_comparison_plots(all_results::Dict, setup::AnalysisSetup)
    println("\n📊 Creating final comparison plot...")
    
    try
        plots = []
        # Extract first tau from results
        first_scenario = first(values(all_results))
        tau = isempty(first_scenario) ? 0.9 : first(sort(collect(keys(first_scenario))))
        
        for (i, scenario_name) in enumerate(RIPK1_CONFIG.scenario_names[1:min(4, length(RIPK1_CONFIG.scenario_names))])
            if haskey(all_results, scenario_name) && 
               haskey(all_results[scenario_name], tau) &&
               haskey(all_results[scenario_name][tau], :df)
                
                df = all_results[scenario_name][tau].df
                kpuu_value = i <= size(setup.selected_data, 1) ? setup.selected_data.kpuu[i] : missing
                
                # Use the enhanced plot_ensemble_pareto function with global normalization
                # Global normalization ensures fair comparison across different thresholds
                p = plot_ensemble_pareto(df, tau; show_annotations=true, normalization=:global)
                
                # Update title to match Python style
                plot!(p, title = "Scenario $i, kpuu = $(ismissing(kpuu_value) ? "N/A" : round(kpuu_value, digits=3))",
                      titlefontsize = 11)
                
                # Set x-axis limits based on scenario
                if i == 3  # Scenario 3 with potential outliers
                    xlims!(p, (0, 40000))
                    # Check for outliers
                    outliers = filter(row -> row.Average_Utility > 40000, df)
                    if nrow(outliers) > 0
                        annotate!(p, 0.02, 0.02, 
                                 text("$(nrow(outliers)) points\nbeyond range", 6, :left, :red),
                                 subplot=1)
                    end
                else
                    xlims!(p, (0, 5000))
                end
                
                # Set y-axis limits based on scenario
                if i in [1, 4]  # Scenarios 1 and 4
                    ylims!(p, (-0.01, 0.31))
                else  # Scenarios 2 and 3
                    ylims!(p, (-0.01, 1.01))
                end
                
                # Only show y-axis label for left column
                if i in [1, 3]
                    ylabel!(p, "State Uncertainty")
                else
                    ylabel!(p, "")
                end
                
                # Show legend only on first subplot
                if i == 1
                    plot!(p, legend = :topleft, legendfontsize = 8)
                else
                    plot!(p, legend = false)
                end
                
                push!(plots, p)
            end
        end
        
        if length(plots) > 0
            # Create 2x2 grid with main title
            compare_plots = plot(plots..., 
                               layout = (2, 2), 
                               size = (1000, 800),  # Match Python figure size (10x8 inches at 100 DPI)
                               dpi = 300,  # High DPI for publication
                               margin = 5Plots.mm,  # Add margins for better layout
                               plot_title = "Pareto Fronts (2×2 Grid)",  # Main title
                               plot_titlefontsize = 14)
            
            savefig(compare_plots, joinpath(setup.results_base, RIPK1_CONFIG.comparison_plot_name))
            println("  ✅ Comparison plot saved to: $(joinpath(setup.results_base, RIPK1_CONFIG.comparison_plot_name))")
            return true
        else
            println("  ⚠️ No plots to compare")
            return false
        end
    catch e
        println("  ⚠️ Error creating comparison plot: $e")
        return false
    end
end

"""
    generate_final_report(all_results::Dict, setup::AnalysisSetup; ensemble_size::Int=10)

Generate final analysis report.
"""
function generate_final_report(all_results::Dict, setup::AnalysisSetup; ensemble_size::Int=10)
    println("\n" * "="^80)
    println("🎉 RIPK1 Analysis Complete!")
    println("="^80)
    println("✅ Results Summary:")
    println("  - Ensemble size: $ensemble_size")
    println("  - Processed scenarios: $(length(all_results))")
    
    if size(setup.selected_data, 1) > 0
        println("  - kpuu values: ", round.(setup.selected_data.kpuu[1:min(4, size(setup.selected_data,1))], digits=3))
    end
    
    # Create summary DataFrame
    summary_rows = []
    for (scenario_name, scenario_results) in all_results
        for (tau, results) in scenario_results
            if haskey(results, :df)
                df = results.df
                
                # Find the most likely action set
                max_freq_idx = argmax(df.Frequency)
                mlasp = df[max_freq_idx, :]
                
                # Get kpuu value if available
                scenario_idx = findfirst(x -> x == scenario_name, RIPK1_CONFIG.scenario_names)
                kpuu_value = scenario_idx !== nothing && scenario_idx <= size(setup.selected_data, 1) ? 
                            setup.selected_data.kpuu[scenario_idx] : missing
                
                push!(summary_rows, (
                    scenario = scenario_name,
                    tau = tau,
                    kpuu = ismissing(kpuu_value) ? NaN : kpuu_value,
                    n_unique_action_sets = length(unique(df.Action_Set)),
                    total_action_sets = nrow(df),
                    most_likely_action_set = mlasp.Action_Set,
                    mlasp_frequency = mlasp.Frequency,
                    mlasp_utility = round(mlasp.Average_Utility, digits=2),
                    mlasp_uncertainty = round(mlasp.Threshold, digits=3),
                    min_utility = round(minimum(df.Average_Utility), digits=2),
                    max_utility = round(maximum(df.Average_Utility), digits=2)
                ))
            end
        end
    end
    
    if length(summary_rows) > 0
        summary_df = DataFrame(summary_rows)
        CSV.write(joinpath(setup.results_base, "analysis_summary.csv"), summary_df)
        println("  📊 Saved analysis summary to: analysis_summary.csv")
    end
    
    # Extract actual tau values used from the results
    actual_taus = Float64[]
    for (scenario_name, scenario_results) in all_results
        append!(actual_taus, collect(keys(scenario_results)))
    end
    unique_taus = sort(unique(actual_taus))
    
    println("  - Results saved to: $(setup.results_base)")
    println("\n🔧 Configuration:")
    println("  - Belief thresholds (τ): $(unique_taus)")
    println("  - Max parallel assays: $(RIPK1_CONFIG.parallel_assays_NO)")
    println("  - Terminal condition: kpuu ∈ $(RIPK1_CONFIG.target_condition["kpuu"])")
    println("="^80)
end

# ============= RIPK1-SPECIFIC DATA FUNCTIONS =============
"""
    select_representative_rows(data::DataFrame; selected_condition::Symbol = :false_negative, num_instances::Int = 20)

Select representative compounds for RIPK1 scenarios based on PgP/BCRP/kpuu characteristics.
"""
function select_representative_rows(data::DataFrame; selected_condition::Symbol = :false_negative, num_instances::Int = 20)
    # RIPK1-specific conditions
    false_negative_conditions = [
        (data[!, "1uM_PgP_qsar"] .< 2) .& (data[!, "100_nM_Mouse_BCRP_qsar"] .< 2) .& (data[!, "kpuu"] .> 0.5),
        (data[!, "1uM_PgP_qsar"] .< 2) .& (data[!, "100_nM_Mouse_BCRP_qsar"] .> 4) .& (data[!, "kpuu"] .> 0.5),
        (data[!, "1uM_PgP_qsar"] .> 4) .& (data[!, "100_nM_Mouse_BCRP_qsar"] .< 2) .& (data[!, "kpuu"] .> 0.5),
        (data[!, "1uM_PgP_qsar"] .> 4) .& (data[!, "100_nM_Mouse_BCRP_qsar"] .> 4) .& (data[!, "kpuu"] .> 0.5)
    ]
    
    conditions = false_negative_conditions  # Using false_negative for RIPK1 scenarios
    
    selected_columns = ["1uM_PgP_qsar", "100_nM_Mouse_BCRP_qsar", "qsar_mrt", "kpuu", "100nM_PgP", "1uM_PgP", "100nM_BCRP"]
    selected_rows = DataFrame()
    state_init_list = []
    
    for (i, condition) in enumerate(conditions)
        filtered_data = data[condition, :]
        if nrow(filtered_data) == 0
            println("    Warning: No rows satisfy condition $i")
        else
            sorted_data = sort(filtered_data, :kpuu)
            representative_rows = first(sorted_data, min(num_instances, nrow(sorted_data)))[!, selected_columns]
            selected_rows = vcat(selected_rows, representative_rows)
            
            # Create state initialization for the first row of each condition
            if nrow(representative_rows) > 0
                row = first(eachrow(representative_rows))
                state_init = Evidence(
                    "qsar_mrt" => row["qsar_mrt"],
                    "1uM_PgP_qsar" => row["1uM_PgP_qsar"],
                    "100_nM_Mouse_BCRP_qsar" => row["100_nM_Mouse_BCRP_qsar"]
                )
                push!(state_init_list, state_init)
            end
        end
    end
    
    return state_init_list, selected_rows
end

"""
    preprocess_ripk1_data(data::DataFrame)

Preprocess RIPK1 data by handling compound IDs and removing unnecessary columns.
"""
function preprocess_ripk1_data(data::DataFrame)
    # Store compound_id separately if it exists
    if "compound_id" in names(data)
        compound_ids = data.compound_id
        data = select(data, Not(:compound_id))
    else
        compound_ids = ["L-" * lpad(i, 9, "0") * "-000X" for i in 1:nrow(data)]
    end
    
    # Remove mrt columns temporarily (as in original RIPK1 analysis)
    if "cassette_mrt" in names(data) && "full_pk_mrt" in names(data)
        select!(data, Not([:cassette_mrt, :full_pk_mrt]))
    end
    
    # Add compound tracking back
    data.compound_id = compound_ids
    
    return data, compound_ids
end

"""
    setup_ripk1_distances()

Setup distance configurations specific to RIPK1 features.
"""
function setup_ripk1_distances()
    # RIPK1-specific feature categorization
    in_silico = ["1uM_PgP_qsar", "100_nM_Mouse_BCRP_qsar", "qsar_mrt"]
    physical = [
        "blood_frac_conc", "brain_conc", "brain_binding", 
        "plasma_protein_binding", "kpuu", "100nM_PgP", 
        "1uM_PgP", "100nM_BCRP"
    ]
    
    distances = Dict()
    
    # Different lambda values for in silico vs physical measurements
    foreach(e -> push!(distances, e => QuadraticDistance(; λ = RIPK1_CONFIG.sampler_params.in_silico_lambda)), in_silico)
    foreach(e -> push!(distances, e => QuadraticDistance(; λ = RIPK1_CONFIG.sampler_params.physical_lambda)), physical)
    
    return distances
end

end # module RIPK1