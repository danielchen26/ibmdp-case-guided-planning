module CEEDesigns

using DataFrames, Plots
export front, plot_front, plot_dict, save_figure
export make_labels, plot_evals
# export remove_outliers!, load_outerjoin_assays
# export Smiles  # Commented out as Smiles module is not loaded for RIPK1 analysis

# make Pareto fronts
include("fronts.jl")

# experimental designs
include("StaticDesigns/StaticDesigns.jl")
include("GenerativeDesigns/GenerativeDesigns.jl")

# ! NEW: Enhanced modules for comprehensive CEED analysis
include("ValueIteration/ValueIteration.jl")
include("CEEDUtilities/CEEDUtilities.jl")

## utility modules
# NOTE: The following modules are commented out for RIPK1 analysis
# as they have additional dependencies not needed for this analysis
# include("modules/StAR.jl")
# include("modules/DBConnector/DBConnector.jl")
# include("modules/Smiles.jl")
# using .Smiles
# include("modules/Imputation/Imputation.jl")
# include("modules/Discretizers.jl")

# ! NEW: Enhanced exports for value iteration and CEED utilities
export value_iteration_analysis, similarity_based_value_iteration, theoretical_value_iteration
export setup_mdp_framework, run_mdp_analysis
export gradient_bar, make_labels_modi, cost_bias_tuple
export load_synthetic_data, setup_ceed_configuration, setup_default_plotting_options
export create_experiments_dict, setup_enhanced_sampler, setup_solver
export find_project_data_directory

# ! NEW: Advanced workflow and visualization functions
export ensemble_Vis, CEED_single_state_init, CEED_multiple_state_init_run

end
