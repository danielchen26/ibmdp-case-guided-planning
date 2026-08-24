# using MLJ, DataFrames, Statistics
# using RDatasets
# using MLJLinearModels # Import the MLJLinearModels package

# # Load linear regressor from MLJModels
# @load LinearRegressor pkg=MLJLinearModels

# function featureSelection(dataset::DataFrame, target::Symbol)
#     # Split the data into features (X) and target (y)
#     X = select(dataset, Not(Symbol(target)))
#     y = dataset[:, Symbol(target)]

#     # Standardize the features
#     X = DataFrames.transform(X, names(X) .=> (col -> (col .- mean(col)) / std(col)) .=> names(X))

#     # Generate initial model with arbitrary regularization strength
#     lr_model = LassoRegression(0.1)



#     # Use cross validation to tune hyperparameters and find best lambda value
#     self_tuning_lr_model = TunedModel(model=lr_model, 
#                                       tuning=Grid(resolution=100), 
#                                       resampling=CV(), 
#                                       measure=rms,
#                                       acceleration=CPUThreads(),
#                                       range=[range(lr_model, :lambda, lower=0.001, upper=3.0)])

#     # Train the model
#     mach = machine(self_tuning_lr_model, X, y)
#     fit!(mach)

#     # Get the optimal lambda value from the trained machine
#     optimal_lambda = fitted_params(mach).best_model.lambda

#     # Refit the model with optimal lambda
#     lr_model = LinearRegressor(lambda=optimal_lambda, solver=:LassoReg) # Use a symbol for the solver argument
#     mach = machine(lr_model, X, y)
#     fit!(mach)
    
#     # Getting feature weights / coefficients
#     coefs = fitted_params(mach).coefs

#     # Check which features have non-zero coefficients - represent selected features
#     selected_features = coefs .!= 0

#     return selected_features
# end

# # Load data and perform feature selection
# boston = dataset("MASS", "Boston")
# selected_features = featureSelection(boston, :Crim)


## ========
using DataFrames
using MLJ
using MLJLinearModels
using MLJTuning
using Plots



function lasso_feature_selection(
    data::DataFrame,
    target_col::Union{String, Symbol};
    excluded_col::Union{Nothing,String,Array{String,1}} = nothing,
    kwargs...
)
    # Prepare the data
    y = data[:, target_col]
    if excluded_col == nothing
        X = data[:, Not(target_col)] |> DataFrame
    else
        X = data[:, Not([target_col, excluded_col])] |> DataFrame
    end

    # Specify data types
    sch = schema(data[:, Not(target_col)])
    types = Dict(zip(sch.names, sch.scitypes))

    # Coerce data
    X = coerce(X, types)
    y = coerce(y, Continuous)


    # Load and instantiate the Lasso regression model
    @load LassoRegressor pkg = MLJLinearModels verbosity = 0 # The verbosity=0 argument is for silent fitting
    model = LassoRegressor()

    # Define the tuning strategy
    tuning = Grid(; goal = 100)

    # Define the range of hyperparameters to search over
    param_range = [range(model, :lambda; lower = 1e-9, upper = 1.0, scale = :log)]

    # Define the performance metric to optimize
    measure = rms

    # Wrap the model in a tuning strategy
    tuned_model =
        TunedModel(; model = model, tuning = tuning, range = param_range, measure = measure)

    # Create a machine with the wrapped model
    mach = machine(tuned_model, X, y)

    # Tune the model
    fit!(mach, verbosity = 0) 

    # Get the best model
    best_model = fitted_params(mach).best_model
    best_model_performances_plt = plot(
        mach
    )
    # Get the most informative features
    coefs = fitted_params(mach).best_fitted_params.coefs

    # Separate the feature names and coefficients
    feature_names = [p.first for p in coefs]
    coefficients = [p.second for p in coefs]

    # Filter out coefficients close to zero and their corresponding features
    informative_indices = findall(x -> abs(x) > 1e-6, coefficients)
    informative_features = feature_names[informative_indices]
    absolute_importance = coefficients[informative_indices]

    # Calculate total importance
    total_importance = sum(abs.(absolute_importance))

    # Calculate relative importance
    relative_importance = abs.(absolute_importance) / total_importance

    return Dict(
        "features" => informative_features,
        "absolute_importance" => absolute_importance,
        "relative_importance" => relative_importance,
    ),
    best_model_performances_plt
end





function plot_feature_importance(importance_data::Dict)
    # Extract data
    features = importance_data["features"]
    absolute_importance = importance_data["absolute_importance"]
    relative_importance = importance_data["relative_importance"]
    
    features = string.(features)
    # color_gradient = cgrad(:viridis) # Create a color gradient from the :viridis palette
    color_gradient = cgrad(:viridis; scale = x -> exp(x))
    colors = get(color_gradient, relative_importance)
    # Map the relative importance values to colors

    p1 = bar(
        features,
        absolute_importance;
        legend = false,
        title = "Absolute Importance",
        color = colors,
        colorbar = true,
    )
    ylabel!("Coefficient Value")

    p2 = bar(
        features,
        relative_importance;
        legend = false,
        title = "Relative Importance",
        color = colors,
        colorbar = true
    )
    ylabel!("Normalized Coefficient Value")
    p = plot(p1, p2; layout = (1, 2), size = (900, 400)) # Create the plot object
    plot!(p; colorbar = true) # Add the colorbar attribute

    return p
end

# # Example usage:
# using RDatasets
# mtcars = dataset("datasets", "mtcars")
# # Assuming `importance_data` contains the feature importance information
# importance_data = lasso_feature_importance(mtcars, :MPG, excluded_col = [:Model, :Gear, :Carb])
# # Plot the feature importance
# plot_feature_importance(importance_data)
