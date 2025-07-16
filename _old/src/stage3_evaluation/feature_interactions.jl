module FeatureInteractions

using Statistics
using Random
using LinearAlgebra
using SparseArrays
using DataFrames
using MLJ
using MLJBase
using ProgressMeter
using Base.Threads
using StatsBase

export InteractionCalculator, InteractionResult, SparseInteractionMatrix
export calculate_h_statistic, calculate_mutual_information
export calculate_partial_dependence_interaction, calculate_performance_degradation
export get_significant_interactions, export_interaction_heatmap
export combine_interaction_methods

"""
Result structure for feature interaction analysis
"""
struct InteractionResult
    feature_indices::Tuple{Int, Int}
    feature_names::Union{Nothing, Tuple{String, String}}
    interaction_strength::Float64
    confidence_interval::Union{Nothing, Tuple{Float64, Float64}}
    method::Symbol  # :h_statistic, :mutual_info, :partial_dependence, :performance_degradation
    metadata::Dict{Symbol, Any}
end

"""
Sparse matrix for storing interaction scores efficiently
"""
struct SparseInteractionMatrix
    n_features::Int
    interactions::SparseMatrixCSC{Float64, Int}
    feature_names::Union{Nothing, Vector{String}}
    method::Symbol
    threshold::Float64
    
    function SparseInteractionMatrix(n_features::Int; 
                                   feature_names::Union{Nothing, Vector{String}}=nothing,
                                   method::Symbol=:combined,
                                   threshold::Float64=0.01)
        # Initialize sparse matrix (upper triangular only)
        interactions = spzeros(n_features, n_features)
        new(n_features, interactions, feature_names, method, threshold)
    end
end

"""
Main calculator for feature interactions
"""
mutable struct InteractionCalculator
    method::Symbol  # :h_statistic, :mutual_info, :partial_dependence, :performance_degradation, :all
    n_samples::Union{Nothing, Int}  # For sampling large datasets
    categorical_features::Union{Nothing, Vector{Int}}  # Indices of categorical features
    n_jobs::Int
    random_state::Union{Nothing, Int}
    min_samples_leaf::Int  # For partial dependence
    
    function InteractionCalculator(; method::Symbol=:all,
                                  n_samples::Union{Nothing, Int}=nothing,
                                  categorical_features::Union{Nothing, Vector{Int}}=nothing,
                                  n_jobs::Int=1,
                                  random_state::Union{Nothing, Int}=nothing,
                                  min_samples_leaf::Int=10)
        new(method, n_samples, categorical_features, n_jobs, random_state, min_samples_leaf)
    end
end

"""
Calculate H-statistic for pairwise feature interactions
H-statistic measures the fraction of variance explained by the interaction
"""
function calculate_h_statistic(calc::InteractionCalculator, model, machine,
                             X::AbstractMatrix, y::AbstractVector,
                             feature_i::Int, feature_j::Int;
                             n_permutations::Int=10)
    
    n_samples = size(X, 1)
    rng = isnothing(calc.random_state) ? Random.GLOBAL_RNG : MersenneTwister(calc.random_state)
    
    # Sample data if needed
    if !isnothing(calc.n_samples) && calc.n_samples < n_samples
        sample_idx = randperm(rng, n_samples)[1:calc.n_samples]
        X_sample = X[sample_idx, :]
    else
        X_sample = X
    end
    
    # Get predictions for original data
    y_pred_original = predict(machine, MLJ.table(X_sample))
    
    # Calculate variance of predictions
    var_original = var(y_pred_original)
    
    # Calculate H-statistic through permutations
    h_values = zeros(n_permutations)
    
    for perm in 1:n_permutations
        # Create copies for permutation
        X_perm_i = copy(X_sample)
        X_perm_j = copy(X_sample)
        X_perm_both = copy(X_sample)
        
        # Permute features independently
        perm_idx = randperm(rng, size(X_sample, 1))
        X_perm_i[:, feature_i] = X_sample[perm_idx, feature_i]
        X_perm_j[:, feature_j] = X_sample[perm_idx, feature_j]
        X_perm_both[:, feature_i] = X_sample[perm_idx, feature_i]
        X_perm_both[:, feature_j] = X_sample[perm_idx, feature_j]
        
        # Get predictions
        y_pred_i = predict(machine, MLJ.table(X_perm_i))
        y_pred_j = predict(machine, MLJ.table(X_perm_j))
        y_pred_both = predict(machine, MLJ.table(X_perm_both))
        
        # Calculate partial dependence functions
        pd_i = mean(y_pred_original) - mean(y_pred_i)
        pd_j = mean(y_pred_original) - mean(y_pred_j)
        pd_ij = mean(y_pred_original) - mean(y_pred_both)
        
        # H-statistic numerator: interaction effect
        interaction_effect = pd_ij - pd_i - pd_j
        
        # Store normalized H-statistic
        h_values[perm] = abs(interaction_effect) / (sqrt(var_original) + 1e-10)
    end
    
    # Average H-statistic across permutations
    h_statistic = mean(h_values)
    h_std = std(h_values)
    
    # Confidence interval
    ci_lower = h_statistic - 1.96 * h_std / sqrt(n_permutations)
    ci_upper = h_statistic + 1.96 * h_std / sqrt(n_permutations)
    
    return InteractionResult(
        (feature_i, feature_j),
        nothing,
        h_statistic,
        (max(0.0, ci_lower), ci_upper),
        :h_statistic,
        Dict(:n_permutations => n_permutations,
             :variance_original => var_original)
    )
end

"""
Calculate mutual information between feature pairs
"""
function calculate_mutual_information(calc::InteractionCalculator,
                                    X::AbstractMatrix, 
                                    feature_i::Int, feature_j::Int;
                                    n_bins::Int=10)
    
    x_i = X[:, feature_i]
    x_j = X[:, feature_j]
    
    # Handle categorical features
    is_cat_i = !isnothing(calc.categorical_features) && feature_i in calc.categorical_features
    is_cat_j = !isnothing(calc.categorical_features) && feature_j in calc.categorical_features
    
    if is_cat_i && is_cat_j
        # Both categorical
        mi = mutual_information_categorical(x_i, x_j)
    elseif is_cat_i || is_cat_j
        # One categorical, one continuous
        mi = mutual_information_mixed(x_i, x_j, is_cat_i, n_bins)
    else
        # Both continuous
        mi = mutual_information_continuous(x_i, x_j, n_bins)
    end
    
    # Normalize by entropy
    h_i = entropy_discrete(discretize(x_i, n_bins))
    h_j = entropy_discrete(discretize(x_j, n_bins))
    normalized_mi = 2 * mi / (h_i + h_j + 1e-10)
    
    return InteractionResult(
        (feature_i, feature_j),
        nothing,
        normalized_mi,
        nothing,  # No CI for mutual information
        :mutual_info,
        Dict(:n_bins => n_bins,
             :mi_raw => mi,
             :entropy_i => h_i,
             :entropy_j => h_j)
    )
end

"""
Calculate mutual information for continuous features
"""
function mutual_information_continuous(x::AbstractVector, y::AbstractVector, n_bins::Int)
    # Discretize continuous variables
    x_discrete = discretize(x, n_bins)
    y_discrete = discretize(y, n_bins)
    
    return mutual_information_discrete(x_discrete, y_discrete)
end

"""
Calculate mutual information for categorical features
"""
function mutual_information_categorical(x::AbstractVector, y::AbstractVector)
    return mutual_information_discrete(x, y)
end

"""
Calculate mutual information for mixed features
"""
function mutual_information_mixed(x::AbstractVector, y::AbstractVector, 
                                x_is_categorical::Bool, n_bins::Int)
    if x_is_categorical
        x_discrete = x
        y_discrete = discretize(y, n_bins)
    else
        x_discrete = discretize(x, n_bins)
        y_discrete = y
    end
    
    return mutual_information_discrete(x_discrete, y_discrete)
end

"""
Calculate mutual information for discrete variables
"""
function mutual_information_discrete(x::AbstractVector, y::AbstractVector)
    n = length(x)
    
    # Joint probability
    joint_counts = countmap(collect(zip(x, y)))
    joint_prob = Dict(k => v/n for (k, v) in joint_counts)
    
    # Marginal probabilities
    x_counts = countmap(x)
    y_counts = countmap(y)
    x_prob = Dict(k => v/n for (k, v) in x_counts)
    y_prob = Dict(k => v/n for (k, v) in y_counts)
    
    # Calculate MI
    mi = 0.0
    for ((xi, yi), p_xy) in joint_prob
        p_x = x_prob[xi]
        p_y = y_prob[yi]
        if p_xy > 0
            mi += p_xy * log(p_xy / (p_x * p_y))
        end
    end
    
    return mi
end

"""
Calculate entropy of discrete variable
"""
function entropy_discrete(x::AbstractVector)
    n = length(x)
    counts = countmap(x)
    
    entropy = 0.0
    for (_, count) in counts
        if count > 0
            p = count / n
            entropy -= p * log(p)
        end
    end
    
    return entropy
end

"""
Discretize continuous variable into bins
"""
function discretize(x::AbstractVector, n_bins::Int)
    # Use quantile-based binning
    edges = quantile(x, range(0, 1, length=n_bins+1))
    edges[1] -= eps()  # Ensure minimum value is included
    edges[end] += eps()  # Ensure maximum value is included
    
    # Remove duplicate edges
    unique!(edges)
    
    # Assign to bins
    discrete_x = zeros(Int, length(x))
    for (i, xi) in enumerate(x)
        for (j, edge) in enumerate(edges[2:end])
            if xi <= edge
                discrete_x[i] = j
                break
            end
        end
    end
    
    return discrete_x
end

"""
Calculate interaction strength using partial dependence plots
"""
function calculate_partial_dependence_interaction(calc::InteractionCalculator,
                                                model, machine,
                                                X::AbstractMatrix,
                                                feature_i::Int, feature_j::Int;
                                                grid_size::Int=20)
    
    n_samples = size(X, 1)
    
    # Create grid for features i and j
    x_i_values = get_feature_grid(X[:, feature_i], grid_size, 
                                 feature_i in something(calc.categorical_features, Int[]))
    x_j_values = get_feature_grid(X[:, feature_j], grid_size,
                                 feature_j in something(calc.categorical_features, Int[]))
    
    # Calculate 2D partial dependence
    pd_matrix = zeros(length(x_i_values), length(x_j_values))
    
    for (idx_i, val_i) in enumerate(x_i_values)
        for (idx_j, val_j) in enumerate(x_j_values)
            # Create modified dataset
            X_modified = copy(X)
            X_modified[:, feature_i] .= val_i
            X_modified[:, feature_j] .= val_j
            
            # Get predictions
            y_pred = predict(machine, MLJ.table(X_modified))
            pd_matrix[idx_i, idx_j] = mean(y_pred)
        end
    end
    
    # Calculate marginal effects
    pd_i = vec(mean(pd_matrix, dims=2))
    pd_j = vec(mean(pd_matrix, dims=1))
    
    # Calculate interaction strength as deviation from additivity
    interaction_matrix = zeros(size(pd_matrix))
    for i in 1:length(x_i_values)
        for j in 1:length(x_j_values)
            # Expected under no interaction
            expected = pd_i[i] + pd_j[j] - mean(pd_matrix)
            # Actual
            actual = pd_matrix[i, j]
            # Interaction effect
            interaction_matrix[i, j] = actual - expected
        end
    end
    
    # Summarize interaction strength
    interaction_strength = sqrt(mean(interaction_matrix.^2))
    
    return InteractionResult(
        (feature_i, feature_j),
        nothing,
        interaction_strength,
        nothing,
        :partial_dependence,
        Dict(:grid_size => grid_size,
             :pd_matrix => pd_matrix,
             :interaction_matrix => interaction_matrix)
    )
end

"""
Get grid values for a feature
"""
function get_feature_grid(x::AbstractVector, grid_size::Int, is_categorical::Bool)
    if is_categorical
        return unique(x)
    else
        return quantile(x, range(0, 1, length=grid_size))
    end
end

"""
Calculate interaction using model performance degradation
"""
function calculate_performance_degradation(calc::InteractionCalculator,
                                         model, machine,
                                         X::AbstractMatrix, y::AbstractVector,
                                         feature_i::Int, feature_j::Int;
                                         metric::Function=accuracy,
                                         n_shuffles::Int=10)
    
    rng = isnothing(calc.random_state) ? Random.GLOBAL_RNG : MersenneTwister(calc.random_state)
    
    # Baseline performance
    y_pred_baseline = predict(machine, MLJ.table(X))
    baseline_score = metric(y, y_pred_baseline)
    
    # Performance when shuffling individual features
    scores_i = zeros(n_shuffles)
    scores_j = zeros(n_shuffles)
    scores_both = zeros(n_shuffles)
    scores_joint = zeros(n_shuffles)
    
    for shuffle_idx in 1:n_shuffles
        # Shuffle feature i
        X_shuffle_i = copy(X)
        X_shuffle_i[:, feature_i] = X_shuffle_i[randperm(rng, size(X, 1)), feature_i]
        y_pred_i = predict(machine, MLJ.table(X_shuffle_i))
        scores_i[shuffle_idx] = metric(y, y_pred_i)
        
        # Shuffle feature j
        X_shuffle_j = copy(X)
        X_shuffle_j[:, feature_j] = X_shuffle_j[randperm(rng, size(X, 1)), feature_j]
        y_pred_j = predict(machine, MLJ.table(X_shuffle_j))
        scores_j[shuffle_idx] = metric(y, y_pred_j)
        
        # Shuffle both independently
        X_shuffle_both = copy(X)
        X_shuffle_both[:, feature_i] = X_shuffle_both[randperm(rng, size(X, 1)), feature_i]
        X_shuffle_both[:, feature_j] = X_shuffle_both[randperm(rng, size(X, 1)), feature_j]
        y_pred_both = predict(machine, MLJ.table(X_shuffle_both))
        scores_both[shuffle_idx] = metric(y, y_pred_both)
        
        # Shuffle both jointly (same permutation)
        perm = randperm(rng, size(X, 1))
        X_shuffle_joint = copy(X)
        X_shuffle_joint[:, feature_i] = X_shuffle_joint[perm, feature_i]
        X_shuffle_joint[:, feature_j] = X_shuffle_joint[perm, feature_j]
        y_pred_joint = predict(machine, MLJ.table(X_shuffle_joint))
        scores_joint[shuffle_idx] = metric(y, y_pred_joint)
    end
    
    # Calculate degradations
    degradation_i = baseline_score - mean(scores_i)
    degradation_j = baseline_score - mean(scores_j)
    degradation_both = baseline_score - mean(scores_both)
    degradation_joint = baseline_score - mean(scores_joint)
    
    # Interaction strength: difference between independent and joint shuffling
    interaction_strength = abs(degradation_both - degradation_joint)
    
    # Confidence interval
    interaction_values = abs.((baseline_score .- scores_both) - (baseline_score .- scores_joint))
    ci_lower = quantile(interaction_values, 0.025)
    ci_upper = quantile(interaction_values, 0.975)
    
    return InteractionResult(
        (feature_i, feature_j),
        nothing,
        interaction_strength,
        (ci_lower, ci_upper),
        :performance_degradation,
        Dict(:baseline_score => baseline_score,
             :degradation_i => degradation_i,
             :degradation_j => degradation_j,
             :degradation_both => degradation_both,
             :degradation_joint => degradation_joint,
             :n_shuffles => n_shuffles)
    )
end

"""
Calculate all pairwise interactions and store in sparse matrix
"""
function calculate_all_interactions(calc::InteractionCalculator,
                                  model, machine,
                                  X::AbstractMatrix, y::AbstractVector;
                                  feature_names::Union{Nothing, Vector{String}}=nothing,
                                  threshold::Float64=0.01,
                                  show_progress::Bool=true)
    
    n_features = size(X, 2)
    n_pairs = n_features * (n_features - 1) ÷ 2
    
    # Initialize sparse matrix
    sparse_matrix = SparseInteractionMatrix(n_features, 
                                          feature_names=feature_names,
                                          method=calc.method,
                                          threshold=threshold)
    
    # Progress bar
    p = show_progress ? Progress(n_pairs, desc="Calculating interactions: ") : nothing
    
    # Calculate interactions
    pair_idx = 0
    for i in 1:n_features
        for j in (i+1):n_features
            pair_idx += 1
            
            # Calculate interaction based on method
            if calc.method == :h_statistic
                result = calculate_h_statistic(calc, model, machine, X, y, i, j)
            elseif calc.method == :mutual_info
                result = calculate_mutual_information(calc, X, i, j)
            elseif calc.method == :partial_dependence
                result = calculate_partial_dependence_interaction(calc, model, machine, X, i, j)
            elseif calc.method == :performance_degradation
                result = calculate_performance_degradation(calc, model, machine, X, y, i, j)
            else  # :all - combine methods
                results = [
                    calculate_h_statistic(calc, model, machine, X, y, i, j),
                    calculate_mutual_information(calc, X, i, j),
                    calculate_partial_dependence_interaction(calc, model, machine, X, i, j),
                    calculate_performance_degradation(calc, model, machine, X, y, i, j)
                ]
                # Average normalized scores
                scores = [r.interaction_strength for r in results]
                result = InteractionResult(
                    (i, j),
                    nothing,
                    mean(scores),
                    nothing,
                    :combined,
                    Dict(:individual_scores => scores)
                )
            end
            
            # Store if above threshold
            if result.interaction_strength >= threshold
                sparse_matrix.interactions[i, j] = result.interaction_strength
            end
            
            !isnothing(p) && ProgressMeter.next!(p)
        end
    end
    
    !isnothing(p) && ProgressMeter.finish!(p)
    
    return sparse_matrix
end

"""
Get significant interactions from sparse matrix
"""
function get_significant_interactions(matrix::SparseInteractionMatrix; 
                                    top_k::Union{Nothing, Int}=nothing,
                                    min_strength::Float64=matrix.threshold)
    
    # Extract non-zero interactions
    I, J, V = findnz(matrix.interactions)
    
    # Filter by minimum strength
    mask = V .>= min_strength
    I, J, V = I[mask], J[mask], V[mask]
    
    # Create result array
    interactions = InteractionResult[]
    for idx in 1:length(I)
        i, j, strength = I[idx], J[idx], V[idx]
        
        feature_names = isnothing(matrix.feature_names) ? nothing : 
                       (matrix.feature_names[i], matrix.feature_names[j])
        
        push!(interactions, InteractionResult(
            (i, j),
            feature_names,
            strength,
            nothing,
            matrix.method,
            Dict()
        ))
    end
    
    # Sort by strength
    sort!(interactions, by=x->x.interaction_strength, rev=true)
    
    # Return top k if specified
    if !isnothing(top_k) && top_k < length(interactions)
        return interactions[1:top_k]
    else
        return interactions
    end
end

"""
Combine interaction results from multiple methods
"""
function combine_interaction_methods(results::Vector{InteractionResult};
                                   weights::Union{Nothing, Vector{Float64}}=nothing)
    
    if isempty(results)
        error("No results to combine")
    end
    
    # Group by feature pair
    grouped = Dict{Tuple{Int, Int}, Vector{InteractionResult}}()
    for result in results
        key = result.feature_indices
        if !haskey(grouped, key)
            grouped[key] = InteractionResult[]
        end
        push!(grouped[key], result)
    end
    
    # Combine for each pair
    combined_results = InteractionResult[]
    
    for (pair, pair_results) in grouped
        # Extract scores by method
        method_scores = Dict{Symbol, Float64}()
        for r in pair_results
            method_scores[r.method] = r.interaction_strength
        end
        
        # Calculate weighted average
        if isnothing(weights)
            combined_score = mean(values(method_scores))
        else
            # Assume weights in order: h_statistic, mutual_info, partial_dependence, performance_degradation
            methods = [:h_statistic, :mutual_info, :partial_dependence, :performance_degradation]
            combined_score = 0.0
            total_weight = 0.0
            
            for (i, method) in enumerate(methods)
                if haskey(method_scores, method) && i <= length(weights)
                    combined_score += weights[i] * method_scores[method]
                    total_weight += weights[i]
                end
            end
            
            combined_score /= total_weight
        end
        
        push!(combined_results, InteractionResult(
            pair,
            pair_results[1].feature_names,
            combined_score,
            nothing,
            :combined,
            Dict(:method_scores => method_scores)
        ))
    end
    
    return combined_results
end

"""
Export interaction matrix as heatmap data
"""
function export_interaction_heatmap(matrix::SparseInteractionMatrix, 
                                  filename::String;
                                  symmetric::Bool=true)
    
    # Convert to dense matrix for export
    dense_matrix = Matrix(matrix.interactions)
    
    if symmetric
        # Make symmetric for visualization
        dense_matrix = dense_matrix + dense_matrix'
        # Fix diagonal
        for i in 1:matrix.n_features
            dense_matrix[i, i] = 0.0
        end
    end
    
    # Create DataFrame
    df = DataFrame(dense_matrix, :auto)
    
    if !isnothing(matrix.feature_names)
        # Add feature names as column names
        rename!(df, [Symbol(name) for name in matrix.feature_names])
        # Add row names
        insertcols!(df, 1, :Feature => matrix.feature_names)
    else
        # Generic names
        rename!(df, [Symbol("F$i") for i in 1:matrix.n_features])
        insertcols!(df, 1, :Feature => ["F$i" for i in 1:matrix.n_features])
    end
    
    # Save to CSV
    CSV.write(filename, df)
    
    return df
end

"""
Parallel calculation of interactions
"""
function calculate_all_interactions_parallel(calc::InteractionCalculator,
                                           model, machine,
                                           X::AbstractMatrix, y::AbstractVector;
                                           feature_names::Union{Nothing, Vector{String}}=nothing,
                                           threshold::Float64=0.01)
    
    n_features = size(X, 2)
    
    # Create pairs to process
    pairs = Tuple{Int, Int}[]
    for i in 1:n_features
        for j in (i+1):n_features
            push!(pairs, (i, j))
        end
    end
    
    # Process in parallel
    n_pairs = length(pairs)
    results = Vector{Union{Nothing, InteractionResult}}(nothing, n_pairs)
    
    @threads for idx in 1:n_pairs
        i, j = pairs[idx]
        
        try
            if calc.method == :h_statistic
                results[idx] = calculate_h_statistic(calc, model, machine, X, y, i, j)
            elseif calc.method == :mutual_info
                results[idx] = calculate_mutual_information(calc, X, i, j)
            elseif calc.method == :partial_dependence
                results[idx] = calculate_partial_dependence_interaction(calc, model, machine, X, i, j)
            elseif calc.method == :performance_degradation
                results[idx] = calculate_performance_degradation(calc, model, machine, X, y, i, j)
            end
        catch e
            @warn "Error calculating interaction for features $i and $j: $e"
            results[idx] = nothing
        end
    end
    
    # Build sparse matrix from results
    sparse_matrix = SparseInteractionMatrix(n_features,
                                          feature_names=feature_names,
                                          method=calc.method,
                                          threshold=threshold)
    
    for (idx, result) in enumerate(results)
        if !isnothing(result) && result.interaction_strength >= threshold
            i, j = result.feature_indices
            sparse_matrix.interactions[i, j] = result.interaction_strength
        end
    end
    
    return sparse_matrix
end

end # module