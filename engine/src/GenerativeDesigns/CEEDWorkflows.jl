module CEEDWorkflows

using DataFrames, CSV
using Statistics, StatsBase
using Random, Dates
using Plots
using Colors, PlotUtils
using PrettyTables
using ..GenerativeDesigns: Evidence, perform_ensemble_designs, process_ensemble_results_for_belief
using ..GenerativeDesigns: find_top_n_action_sets_with_utility
using MCTS

export ensemble_Vis, CEED_single_state_init, CEED_multiple_state_init_run
export plot_action_sets_vs_average_utility, plot_histograms_ensemble_results
export save_ensemble_results, load_ensemble_results

"""
    ensemble_Vis(df, style::Symbol; max_likelihood_action_sets=nothing, kwargs...)

Advanced ensemble visualization with multiple styles and sophisticated formatting.

# Arguments
- `df::DataFrame`: Ensemble results with columns: Threshold, Action_Set, Frequency, Average_Utility
- `style::Symbol`: Plotting style (:hist for histogram, :scatter for scatter plot)
- `max_likelihood_action_sets::DataFrame`: Optional maximum likelihood action sets for overlay
- `kwargs...`: Additional plotting options

# Returns
- Vector of plots (for :hist) or single plot (for :scatter)

# Styles
- `:hist`: Creates histogram plots for each threshold with gradient colors and alpha blending
- `:scatter`: Creates scatter plot with connected maximum likelihood paths

# Features
- Gradient color schemes based on action complexity
- Alpha blending based on likelihood values
- Interactive legends with action set labels
- Automatic annotation for maximum likelihood paths
- Professional formatting with proper margins and labels
"""
function ensemble_Vis(df::DataFrame, style::Symbol; max_likelihood_action_sets=nothing, kwargs...)
    # Helper function to count actions based on commas
    action_count_sorter(action) = count(c -> c == ',', action) + 1
    
    # Extract unique actions and sort them by complexity
    CAT = unique(df.Action_Set)
    sorted_CAT = sort(CAT, by=action_count_sorter)
    
    # Generate sophisticated gradient color scheme
    gradient_colors = cgrad(:viridis, length(sorted_CAT), categorical=true)
    color_map = Dict(sorted_CAT[i] => gradient_colors[(i - 1) % length(gradient_colors) + 1] 
                     for i in 1:length(sorted_CAT))
    
    # Define shape mapping for scatter plots
    shapes = [:circle, :square, :diamond, :cross, :star5, :hexagon, :octagon, :utriangle, :dtriangle]
    threshold_shapes = unique(df.Threshold)
    shape_map = Dict(threshold_shapes[i] => shapes[(i - 1) % length(shapes) + 1] 
                     for i in 1:length(threshold_shapes))
    
    if style == :hist
        plots = []
        
        for threshold in sort(unique(df.Threshold))
            df_threshold = filter(row -> row.Threshold == threshold, df)
            
            # Calculate normalized likelihoods for alpha blending
            likelihoods = [sum(filter(row -> row.Action_Set == action, df_threshold).Frequency) / 
                          sum(df_threshold.Frequency) for action in sorted_CAT]
            max_likelihood = maximum(likelihoods)
            
            # Create sophisticated histogram
            p = plot(size=(1200, 1000), 
                    title="State Uncertainty Level: $threshold", 
                    legendtitle="Action Sets",
                    bottom_margin=20Plots.mm, 
                    left_margin=40Plots.mm; kwargs...)
            
            for (i, action) in enumerate(sorted_CAT)
                likelihood = likelihoods[i]
                # Dynamic alpha based on likelihood (0.4 to 1.0 range)
                alpha_value = 0.4 + 0.6 * (likelihood / max_likelihood)
                
                action_color = color_map[action]
                action_color_with_alpha = RGBA(action_color.r, action_color.g, action_color.b, alpha_value)
                
                bar!(p, [i], [likelihood], label=nothing, color=action_color_with_alpha)
            end
            
            # Enhanced legend with invisible scatter points
            for (i, action) in enumerate(sorted_CAT)
                scatter!(p, [NaN], [NaN], 
                        label=action, 
                        color=gradient_colors[(i - 1) % length(gradient_colors) + 1],
                        shape=:circle, markersize=8, legend=:outerright)
            end
            
            # Professional formatting
            ylabel!(p, "Likelihood")
            labels_with_utility = ["$(action)\n(Avg: $(round(mean(df_threshold[df_threshold.Action_Set .== action, :Average_Utility]), digits=2)))" 
                                  for action in sorted_CAT]
            xticks!(p, 1:length(sorted_CAT), labels_with_utility, rotation=45)
            
            push!(plots, p)
        end
        return plots
        
    elseif style == :scatter
        p = scatter(size=(1200, 1000), legend=:outerbottom,
                   bottom_margin=20Plots.mm, left_margin=40Plots.mm; kwargs...)
        
        # Plot data points
        for row in eachrow(df)
            scatter!(p, [row.Average_Utility], [row.Threshold],
                    label=false, color=color_map[row.Action_Set],
                    shape=shape_map[row.Threshold], markersize=8)
        end
        
        # Enhanced legend
        for (i, action) in enumerate(sorted_CAT)
            scatter!(p, [NaN], [NaN], label=action,
                    color=gradient_colors[(i - 1) % length(gradient_colors) + 1],
                    shape=:circle, markersize=8)
        end
        
        # Plot maximum likelihood paths with enhanced styling
        if max_likelihood_action_sets !== nothing
            sorted_max = sort(max_likelihood_action_sets, :Threshold)
            thresholds = sorted_max.Threshold
            utilities = sorted_max.Average_Utility
            
            plot!(p, utilities, thresholds, 
                 line=(:dash, :red, 3), 
                 marker=(:circle, 8, :red, stroke(2, :black)),
                 label="Maximum Likelihood Path")
            
            # Enhanced annotations
            for row in eachrow(sorted_max)
                y_offset = (maximum(df.Threshold) - minimum(df.Threshold)) * 0.03
                annotate!(p, row.Average_Utility, row.Threshold + y_offset,
                         text(row.Action_Set, :center, 8, color_map[row.Action_Set]))
            end
        end
        
        xlabel!(p, "Average Utility")
        ylabel!(p, "State Uncertainty")
        return p
    else
        error("Unsupported style: $style. Use :hist or :scatter")
    end
end

"""
    CEED_single_state_init(state_init, target_value; N_ensemble=50, taus=collect(0.5:0.3:1), costs_tradeoff=(1,0))

Complete CEED workflow for single initial state with ensemble analysis across multiple belief thresholds.

# Arguments
- `state_init`: Initial evidence state
- `target_value::Float64`: Target value for analysis
- `N_ensemble::Int`: Number of ensemble runs (default: 50)
- `taus::Vector`: Belief thresholds to analyze (default: 0.5:0.3:1)
- `costs_tradeoff::Tuple`: Cost trade-off weights (default: money-biased (1,0))

# Returns
- Tuple containing: (dfs, plts_hist, ensembles_Pareto_fronts, combined_hist_plot, combined_pareto_plot, tau_comparison_plot)

# Features
- Automated ensemble analysis across multiple belief thresholds
- Organized result saving with structured folder hierarchy
- Comprehensive visualization generation
- Error handling and logging
- Automatic top action set analysis and reporting
"""
function CEED_single_state_init(state_init, target_value; 
                                N_ensemble=50, 
                                taus=collect(0.5:0.3:1), 
                                costs_tradeoff=(1,0),
                                experiments=nothing,
                                sampler_setup=nothing,
                                solver=nothing,
                                data=nothing)
    
    @info "🚀 Starting CEED single state initialization for target_value: $target_value"
    
    # Validation
    if any(isnothing, [experiments, sampler_setup, solver, data])
        error("Missing required parameters: experiments, sampler_setup, solver, or data")
    end
    
    # Target condition setup
    target_condition = Dict("target" => [0.5, 1.0])
    conditional_weights_thred = 0.8
    
    # Run ensemble designs
    @info "🔄 Running ensemble designs with N=$N_ensemble across $(length(taus)) belief thresholds"
    ensemble_results = @time perform_ensemble_designs(
        experiments;
        sampler = sampler_setup.sampler,
        uncertainty = sampler_setup.uncertainty,
        thresholds = 11,
        evidence = state_init,
        weights = sampler_setup.weights,
        data = data.historical_data,
        terminal_condition = (target_condition, conditional_weights_thred),
        realized_uncertainty = true,
        solver = solver,
        repetitions = 0,
        mdp_options = (max_parallel = 2, costs_tradeoff = costs_tradeoff),
        N = N_ensemble,
        thred_set = taus
    )
    
    # Create organized folder structure
    state_init_str = join(["$(k)_$(round(v, digits=3))" for (k, v) in pairs(state_init)], "_")
    costs_tradeoff_str = costs_tradeoff == (1,0) ? "Money" : costs_tradeoff == (0,1) ? "Time" : "Balanced"
    
    ensemble_folder = joinpath("results", "workflows", 
                              "$(costs_tradeoff_str)_prioritized_target_$(target_value)_N$(N_ensemble)_$(state_init_str)")
    mkpath(ensemble_folder)
    @info "📁 Created results folder: $ensemble_folder"
    
    # Process results for each belief threshold
    dfs = Dict()
    plts_hist = Dict()
    ensembles_Pareto_fronts = Dict()
    
    @info "📊 Processing ensemble results for $(length(taus)) belief thresholds"
    for tau in taus
        try
            df, plt_hist, ensemble_Pareto_front = process_ensemble_results_for_belief(
                ensemble_results, tau; 
                ensemble_folder = ensemble_folder,
                suppress_hist_plot = false
            )
            
            dfs[tau] = df
            plts_hist[tau] = plt_hist
            ensembles_Pareto_fronts[tau] = ensemble_Pareto_front
            
            # Save detailed results
            CSV.write(joinpath(ensemble_folder, "ensemble_results_tau_$(tau).csv"), df)
            
            # Generate top action sets report
            top_actions = find_top_n_action_sets_with_utility(df, 3)
            @info "📈 Top 3 action sets for τ=$tau:"
            pretty_table(top_actions; nosubheader=true)
            
        catch e
            @error "❌ Error processing tau=$tau: $e"
        end
    end
    
    # Generate combined visualizations
    @info "🎨 Generating combined visualizations"
    
    # Combined histograms
    combined_hist = plot([plts_hist[tau] for tau in taus]...,
                        layout=(1, length(taus)),
                        size=(2800, 800),
                        title=reshape(["τ = $tau" for tau in taus], 1, :))
    savefig(combined_hist, joinpath(ensemble_folder, "combined_histograms_$(costs_tradeoff_str).png"))
    
    # Combined Pareto fronts  
    combined_pareto = plot([ensembles_Pareto_fronts[tau] for tau in taus]...,
                          layout=(1, length(taus)),
                          size=(3000, 800),
                          title=reshape(["τ = $tau" for tau in taus], 1, :))
    savefig(combined_pareto, joinpath(ensemble_folder, "combined_pareto_$(costs_tradeoff_str).png"))
    
    # Cross-threshold comparison
    tau_comparison = plot_tau_comparison([dfs[tau] for tau in taus], taus)
    savefig(tau_comparison, joinpath(ensemble_folder, "tau_comparison.png"))
    
    @info "✅ CEED single state analysis completed successfully!"
    @info "📁 Results saved in: $ensemble_folder"
    
    return dfs, plts_hist, ensembles_Pareto_fronts, combined_hist, combined_pareto, tau_comparison
end

"""
    CEED_multiple_state_init_run(initial_states, target_values; kwargs...)

Batch processing workflow for multiple initial states with comprehensive result management.

# Arguments
- `initial_states::Vector`: Vector of initial evidence states
- `target_values::Vector`: Corresponding target values for each state
- `kwargs...`: Additional parameters passed to CEED_single_state_init

# Returns
- Dictionaries containing results for each target value

# Features
- Batch processing with automatic skipping of existing results
- Comprehensive error handling and recovery
- Organized result storage and indexing
- Progress tracking and logging
"""
function CEED_multiple_state_init_run(initial_states::Vector, 
                                     target_values::Vector;
                                     N_ensemble=30,
                                     taus=[0.8, 0.9],
                                     costs_tradeoff=(1,0),
                                     skip_existing=true,
                                     kwargs...)
    
    @info "🔄 Starting multiple state CEED analysis for $(length(initial_states)) compounds"
    
    # Initialize result storage
    results_dict = Dict()
    
    for (i, (state_init, target_value)) in enumerate(zip(initial_states, target_values))
        target = round(target_value, digits=3)
        @info "📋 Processing compound $i/$(length(initial_states)): target=$target"
        
        # Check for existing results
        state_init_str = join(["$(k)_$(round(v, digits=3))" for (k, v) in pairs(state_init)], "_")
        costs_tradeoff_str = costs_tradeoff == (1,0) ? "Money" : costs_tradeoff == (0,1) ? "Time" : "Balanced"
        ensemble_folder = joinpath("results", "workflows",
                                  "$(costs_tradeoff_str)_prioritized_target_$(target)_N$(N_ensemble)_$(state_init_str)")
        
        if skip_existing && isdir(ensemble_folder)
            @info "⏭️  Skipping existing results for target=$target"
            continue
        end
        
        try
            # Run single state analysis
            results = CEED_single_state_init(state_init, target;
                                           N_ensemble=N_ensemble,
                                           taus=taus,
                                           costs_tradeoff=costs_tradeoff,
                                           kwargs...)
            
            results_dict[target] = results
            @info "✅ Completed analysis for target=$target"
            
        catch e
            @error "❌ Error processing target=$target: $e"
            continue
        end
    end
    
    @info "🎉 Multiple state analysis completed! Processed $(length(results_dict)) compounds"
    return results_dict
end

"""
    plot_tau_comparison(dfs::Vector{DataFrame}, taus::Vector)

Create comparison plot across different belief thresholds.
"""
function plot_tau_comparison(dfs::Vector{DataFrame}, taus::Vector)
    p = plot(size=(1000, 600), title="Cross-Threshold Action Set Comparison")
    
    colors = cgrad(:plasma, length(taus))
    
    for (i, (df, tau)) in enumerate(zip(dfs, taus))
        # Plot average utilities for each threshold
        utilities = [mean(group.Average_Utility) for group in groupby(df, :Threshold)]
        thresholds = [group.Threshold[1] for group in groupby(df, :Threshold)]
        
        plot!(p, utilities, thresholds, 
             label="τ = $tau", 
             color=colors[i],
             linewidth=2,
             marker=:circle,
             markersize=4)
    end
    
    xlabel!(p, "Average Utility")
    ylabel!(p, "State Uncertainty")
    return p
end

"""
    save_ensemble_results(file_path::String, ensemble_results)

Save ensemble results to file with compression.
"""
function save_ensemble_results(file_path::String, ensemble_results)
    # Implementation for saving results (serialization)
    @info "💾 Saving ensemble results to: $file_path"
    # Add actual saving logic here
end

"""
    load_ensemble_results(file_path::String)

Load ensemble results from file.
"""
function load_ensemble_results(file_path::String)
    # Implementation for loading results
    @info "📂 Loading ensemble results from: $file_path"
    # Add actual loading logic here
end

end # module CEEDWorkflows 