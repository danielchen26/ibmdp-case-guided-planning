"""
    front(v)
    front(f, v; atol1=0, atol2=0)

Construct a Pareto front of `v`. Elements of `v` will be masked by `f` in the computation.

The first and second (objective) coordinates have to differ by at least `atol1`, `atol2`, respectively, relatively to the latest point on the front.

# Examples

```jldocstest
v = [(1,2), (2,3), (2,1)]
front(v)

# output
[(1, 2), (2, 1)]
```

```jldoctest
v = [(1, (1, 2)), (2, (2, 3)), (3, (2, 1))]
front(x -> x[2], v)

# output

[(1, (1, 2)), (3, (2, 1))]
```

```jldoctest
v = [(1, 2), (2, 1.99), (3, 1)]
front(v; atol2 = 0.2)

# output

[(1, 2), (3, 1)]
```
"""
function front end

function front(v::T; atol1::Float64 = 0.0, atol2::Float64 = atol1) where {T<:AbstractVector}
    return front(identity, v; atol1, atol2)
end

function front(
    f::F,
    v::T;
    atol1::Float64 = 0.0,
    atol2::Float64 = atol1,
) where {F<:Function,T<:AbstractVector}
    # dict sort
    v_sorted = sort(
        v;
        lt = (x, y) ->
            (f(x)[1] < f(y)[1] || (f(x)[1] == f(y)[1] && f(x)[2] <= f(y)[2])),
    )

    # check if the second coordinate drops below the second coordinate of the last non-dominated point
    ix_front = 1
    ix_current = 2
    while ix_current <= length(v_sorted)
        if (f(v_sorted[ix_current])[2] < f(v_sorted[ix_front])[2] - atol2) &&
           (f(v_sorted[ix_current])[1] > f(v_sorted[ix_front])[1] + atol1)
            ix_front = ix_current
            ix_current += 1
        else
            deleteat!(v_sorted, ix_current)
        end
    end

    return v_sorted
end

"""
    plot_front(designs; grad=cgrad(:Paired_12), xlabel, ylabel, labels=get_labels(designs))

Render scatter plot of efficient designs, as returned from `efficient_designs`.

You may optionally specify a color gradient, to draw the colors from.

# Examples

```julia
designs = efficient_designs(experiment, state)
plot_front(designs)
plot_front(designs; grad = cgrad(:Paired_12))
```
"""
function plot_front(
    designs;
    grad = cgrad(:Paired_12),
    xlabel = "combined cost",
    ylabel = "information measure",
    labels = make_labels(designs),
    annotation = true,
    kwargs...,
)
    xs = map(x -> x[1][1], designs)
    ys = map(x -> x[1][2], designs)

    # Calculate offset for annotations if enabled
    offset = annotation ? 0.04 * (maximum(ys) - minimum(ys)) : 0

    p = scatter(
        [xs[1]],
        [ys[1]];
        xlabel,
        ylabel,
        label = labels[1],
        c = grad[1],
        mscolor = nothing,
        fontsize = 16,
        kwargs...,
    )
    # Add annotation if enabled
    if annotation
        Plots.annotate!(xs[1], ys[1] + offset, text(labels[1], 10; color = grad[1]))
    end
    for i = 2:length(designs)
        if xs[i] < Inf
            scatter!(
                p,
                [xs[i]],
                [ys[i]];
                label = labels[i],
                c = grad[i],
                mscolor = nothing,
                kwargs...,
            )
            # Add annotation if enabled
            if annotation
                Plots.annotate!(xs[i], ys[i] + offset, text(labels[i], 8; color = grad[i]))
            end
        end
    end


    # Dynamically adjust plot limits to ensure annotations are not cut off, if annotation is enabled
    if annotation
        xmin, xmax = minimum(xs), maximum(xs)
        ymin, ymax = minimum(ys), maximum(ys)
        plot!(
            p;
            xlims = (xmin - 0.1 * (xmax - xmin), xmax + 0.5 * (xmax - xmin)),
            ylims = (ymin - offset, ymax + 2 * offset),
            legend = :outerbottom,
        )
    end

    # Add a line plot on top of the scatter plot
    # plot!(p, xs, ys; line = (:dashdot, 2), c = :green, label = "Pareto front") # optional to plot the line
    if kwargs[:savefig]
        save_figure(
            p, 
            kwargs[:parallel_assays_NO], 
            kwargs[:interventions_str], 
            kwargs[:event_date_cut_off]; 
            is_perturbed = kwargs[:perturbed],
            cost_bias = kwargs[:cost_bias],
            kwargs...)
    end
    return p
end


"""
    add_pareto_front_reference!(designs, plt; kwargs...)

Add a reference Pareto front line plot to the given plot `plt` using
the Pareto efficient `designs`.

# Arguments

  - `designs`: The Pareto efficient designs
  - `plt`: The plot to add the reference line to
  - `kwargs`: Additional keyword arguments to pass to `plot!`

# Returns

The updated plot `plt` with the added reference Pareto front line.
"""
function add_pareto_front_reference!(designs, plt; kwargs...)
    x_ref = map(x -> x[1][1], designs)
    y_ref = map(x -> x[1][2], designs)
    return plot!(
        plt,
        x_ref,
        y_ref;
        c = :green,
        line = (:dashdotdot, 2),
        label = "Pareto front reference",
        kwargs...,
    )
end


"""
    plot_dict(target_belief::Dict{Float64,Float64})

Plot a dictionary of target beliefs. The function takes a dictionary where keys are target values
and values are probabilities. It sorts the data by target values for better visualization and plots
a bar chart with target values on the x-axis and probabilities on the y-axis. The plot does not include a legend.

# Arguments

  - `target_belief::Dict{Float64,Float64}`: a dictionary where keys are target values and values are probabilities.

# Returns

  - `p`: a bar plot object (spectral plot).
"""
function plot_dict(target_belief::Dict{Float64,Float64}; kwargs...)
    # Extract keys (target values) and values (probabilities) from the dict
    target_values = collect(keys(target_belief))
    probabilities = collect(values(target_belief))

    # Sort the data by target_value for better visualization
    idx = sortperm(target_values)
    target_values = target_values[idx]
    probabilities = probabilities[idx]

    # Create and return the plot
    p = bar(
        target_values,
        probabilities;
        xlabel = "Target Values",
        ylabel = "Probability",
        title = "Distribution of Probabilities",
        legend = false,
    )
    if haskey(kwargs, :savefig) && kwargs[:savefig]
        save_figure(
            p,
            kwargs[:parallel_assays_NO],
            kwargs[:interventions_str],
            kwargs[:event_date_cut_off];
            is_perturbed = kwargs[:perturbed],
            file_extension = kwargs[:file_extension],
            cost_bias = kwargs[:cost_bias],
            kwargs...
        )
    end
    return p
end


function save_figure(
    front_plt,
    parallel_assays_NO,
    interventions_str,
    event_date_cut_off;
    is_perturbed = false,
    file_extension = ".png",
    cost_bias = "money",
    kwargs...,
)
    try
        # Check if :folder_name exists in kwargs
        folder_name = get(kwargs, :folder_name, "")

        # Create the folder path
        folder_path = "figs" * (folder_name != "" ? "/" * folder_name : "")
        
        # Check if the directory exists; if not, create it
        if !isdir(folder_path)
            mkdir(folder_path)
        end
        # Base file name
        base_name =
            folder_path * (folder_path != "" ? "/" : "") * "front" *
            string(parallel_assays_NO) *
            "_"* cost_bias * "_" *
            interventions_str *
            "_intervention_" *
            string(event_date_cut_off)

        # Append "perturbed" if is_perturbed is true    
        base_name = is_perturbed ? base_name * "_perturbed" : base_name

        # Add file extension and save the figure
        savefig(front_plt, base_name * file_extension)

        # println("Figure saved successfully.")
        @info "Figure saved successfully at $(base_name * file_extension)"
    catch e
        println("An error occurred: ", e)
    end
end



# Merck colors
const MRK_COLOR_TEAL = colorant"rgb(0,133,124)"
const MRK_COLOR_LIGHT_TEAL = colorant"rgb(110,206,178)"

"""
    make_labels(designs)

Make labels used plotting of experimental designs.
"""
function make_labels(designs)
    return map(designs) do x
        if !haskey(x[2], :arrangement) || isempty(x[2].arrangement)
            "∅"
        else
            labels =
                ["$i: " * join(group, ", ") for (i, group) in enumerate(x[2].arrangement)]

            join(labels, "; ")
        end
    end
end

"""
    plot_evals(eval; ylabel="information measure")

Create a stick plot that visualizes the performance measures evaluated for subsets of experiments.
"""
function plot_evals(evals; ylabel = "information measure", kwargs...)
    xs = sort!(collect(keys(evals)); by = x -> length(x))
    ys = map(xs) do x
        return evals[x] isa Number ? evals[x] : evals[x].loss
    end
    xformatter = i -> isempty(xs[Int(i)]) ? "∅" : join(xs[Int(i)], ", ")

    return sticks(
        1:length(evals),
        ys;
        ticks = 1:length(evals),
        xformatter,
        guidefontsize = 8,
        tickfontsize = 8,
        ylabel,
        c = MRK_COLOR_TEAL,
        label = nothing,
        xrotation = 30,
    )
end
