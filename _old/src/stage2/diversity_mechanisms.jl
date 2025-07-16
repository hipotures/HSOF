"""
Diversity Mechanisms Implementation for MCTS Ensemble Forest
Builds diversity-promoting mechanisms ensuring trees explore different parts of feature space
through random feature subsampling, exploration constant variation, initial state randomization,
and feature masking diversity.

This module ensures ensemble diversity by preventing trees from selecting identical feature
subsets while maintaining exploration efficiency and convergence properties.
"""

module DiversityMechanisms

using Random
using Statistics
using Dates
using Printf

# Import ensemble forest for tree management
include("ensemble_forest.jl")
using .EnsembleForest

"""
Diversity strategy enumeration for different diversity approaches
"""
@enum DiversityStrategy begin
    RANDOM_SUBSAMPLING = 1     # Random feature subsampling per tree
    EXPLORATION_VARIATION = 2   # Variation in exploration constants
    INITIAL_RANDOMIZATION = 3   # Different starting feature selections
    MASKING_DIVERSITY = 4       # Feature masking to prevent overlap
    COMBINED_DIVERSITY = 5      # Combination of all strategies
end

"""
Configuration for diversity mechanisms
"""
struct DiversityConfig
    # Feature subsampling
    feature_subsample_rate::Float32      # Fraction of features available per tree (default 0.8)
    min_features_per_tree::Int           # Minimum features each tree must have access to
    enable_dynamic_subsampling::Bool     # Allow dynamic adjustment of subsampling rate
    
    # Exploration variation
    exploration_constant_min::Float32    # Minimum exploration constant (default 0.5)
    exploration_constant_max::Float32    # Maximum exploration constant (default 2.0)
    exploration_strategy::String         # "uniform", "normal", "adaptive"
    
    # Initial state randomization
    enable_initial_randomization::Bool   # Enable different starting feature selections
    initial_selection_rate::Float32     # Fraction of features to select initially
    randomization_seed_offset::Int      # Offset for tree-specific random seeds
    
    # Feature masking diversity
    enable_feature_masking::Bool         # Enable feature masking to reduce overlap
    max_overlap_threshold::Float32      # Maximum allowed feature overlap between trees
    overlap_penalty_weight::Float32     # Weight for overlap penalty in selection
    
    # Diversity metrics and monitoring
    target_diversity_score::Float32     # Target diversity score (0.0-1.0)
    diversity_check_frequency::Int      # How often to check diversity (iterations)
    enable_diversity_monitoring::Bool   # Enable real-time diversity tracking
    
    # Adaptive mechanisms
    enable_adaptive_diversity::Bool     # Enable adaptive diversity adjustment
    diversity_adaptation_rate::Float32  # Rate of diversity adaptation
    convergence_diversity_threshold::Float32  # Diversity threshold for convergence
end

"""
Create default diversity configuration
"""
function create_diversity_config(;
    feature_subsample_rate::Float32 = 0.8f0,
    min_features_per_tree::Int = 100,
    enable_dynamic_subsampling::Bool = true,
    exploration_constant_min::Float32 = 0.5f0,
    exploration_constant_max::Float32 = 2.0f0,
    exploration_strategy::String = "uniform",
    enable_initial_randomization::Bool = true,
    initial_selection_rate::Float32 = 0.1f0,
    randomization_seed_offset::Int = 1000,
    enable_feature_masking::Bool = true,
    max_overlap_threshold::Float32 = 0.7f0,
    overlap_penalty_weight::Float32 = 0.1f0,
    target_diversity_score::Float32 = 0.6f0,
    diversity_check_frequency::Int = 100,
    enable_diversity_monitoring::Bool = true,
    enable_adaptive_diversity::Bool = true,
    diversity_adaptation_rate::Float32 = 0.01f0,
    convergence_diversity_threshold::Float32 = 0.3f0
)
    return DiversityConfig(
        feature_subsample_rate, min_features_per_tree, enable_dynamic_subsampling,
        exploration_constant_min, exploration_constant_max, exploration_strategy,
        enable_initial_randomization, initial_selection_rate, randomization_seed_offset,
        enable_feature_masking, max_overlap_threshold, overlap_penalty_weight,
        target_diversity_score, diversity_check_frequency, enable_diversity_monitoring,
        enable_adaptive_diversity, diversity_adaptation_rate, convergence_diversity_threshold
    )
end

"""
Diversity metrics tracking structure
"""
mutable struct DiversityMetrics
    # Feature overlap metrics
    pairwise_overlap_matrix::Matrix{Float32}  # Overlap between each pair of trees
    average_overlap::Float32                  # Average pairwise overlap
    max_overlap::Float32                      # Maximum pairwise overlap
    min_overlap::Float32                      # Minimum pairwise overlap
    
    # Feature selection diversity
    feature_selection_entropy::Float32       # Entropy of feature selection across trees
    unique_feature_combinations::Int         # Number of unique feature combinations
    feature_usage_distribution::Vector{Int}  # How many trees use each feature
    
    # Exploration diversity
    exploration_constant_variance::Float32   # Variance in exploration constants
    score_diversity::Float32                 # Diversity in tree scores
    
    # Overall diversity score
    overall_diversity_score::Float32         # Combined diversity metric (0.0-1.0)
    diversity_trend::Vector{Float32}         # Historical diversity scores
    
    # Monitoring metadata
    last_update::DateTime
    update_count::Int
    computation_time::Float64
end

"""
Initialize diversity metrics
"""
function initialize_diversity_metrics(n_trees::Int, n_features::Int)
    return DiversityMetrics(
        zeros(Float32, n_trees, n_trees),  # pairwise_overlap_matrix
        0.0f0,  # average_overlap
        0.0f0,  # max_overlap
        0.0f0,  # min_overlap
        0.0f0,  # feature_selection_entropy
        0,      # unique_feature_combinations
        zeros(Int, n_features),  # feature_usage_distribution
        0.0f0,  # exploration_constant_variance
        0.0f0,  # score_diversity
        0.0f0,  # overall_diversity_score
        Float32[],  # diversity_trend
        now(),  # last_update
        0,      # update_count
        0.0     # computation_time
    )
end

"""
Diversity manager for coordinating diversity mechanisms across ensemble
"""
mutable struct DiversityManager
    # Configuration
    config::DiversityConfig
    
    # Tree management
    n_trees::Int
    n_features::Int
    tree_feature_masks::Dict{Int, Vector{Bool}}      # Feature availability per tree
    tree_exploration_constants::Dict{Int, Float32}   # Exploration constant per tree
    tree_random_seeds::Dict{Int, Int}                # Random seed per tree
    
    # Diversity tracking
    diversity_metrics::DiversityMetrics
    diversity_history::Vector{Tuple{DateTime, Float32}}  # Time series of diversity
    
    # Adaptive mechanisms
    current_subsample_rate::Float32
    adaptation_iterations::Int
    last_diversity_check::DateTime
    
    # Performance monitoring
    computation_times::Vector{Float64}
    last_computation_time::Float64
    
    # Status
    manager_state::String  # "initialized", "active", "converged", "error"
    error_log::Vector{String}
end

"""
Initialize diversity manager
"""
function initialize_diversity_manager(config::DiversityConfig, n_trees::Int, n_features::Int = 500)
    # Initialize tree-specific configurations
    tree_feature_masks = Dict{Int, Vector{Bool}}()
    tree_exploration_constants = Dict{Int, Float32}()
    tree_random_seeds = Dict{Int, Int}()
    
    # Create diversity metrics
    metrics = initialize_diversity_metrics(n_trees, n_features)
    
    # Create manager
    manager = DiversityManager(
        config,
        n_trees,
        n_features,
        tree_feature_masks,
        tree_exploration_constants,
        tree_random_seeds,
        metrics,
        Tuple{DateTime, Float32}[],
        config.feature_subsample_rate,
        0,
        now(),
        Float64[],
        0.0,
        "initialized",
        String[]
    )
    
    # Apply diversity mechanisms
    apply_diversity_mechanisms!(manager)
    
    manager.manager_state = "active"
    
    return manager
end

"""
Apply all diversity mechanisms to the ensemble
"""
function apply_diversity_mechanisms!(manager::DiversityManager)
    try
        start_time = time()
        
        # Apply feature subsampling
        apply_feature_subsampling!(manager)
        
        # Apply exploration constant variation
        apply_exploration_variation!(manager)
        
        # Apply initial state randomization
        if manager.config.enable_initial_randomization
            apply_initial_randomization!(manager)
        end
        
        # Apply feature masking diversity
        if manager.config.enable_feature_masking
            apply_feature_masking_diversity!(manager)
        end
        
        # Record computation time
        computation_time = time() - start_time
        push!(manager.computation_times, computation_time)
        manager.last_computation_time = computation_time
        
        # Limit history size
        if length(manager.computation_times) > 1000
            deleteat!(manager.computation_times, 1)
        end
        
    catch e
        error_msg = "Failed to apply diversity mechanisms: $e"
        push!(manager.error_log, error_msg)
        manager.manager_state = "error"
        @error error_msg
        rethrow(e)
    end
end

"""
Apply random feature subsampling to trees
"""
function apply_feature_subsampling!(manager::DiversityManager)
    n_features_per_tree = max(
        manager.config.min_features_per_tree,
        round(Int, manager.n_features * manager.current_subsample_rate)
    )
    
    for tree_id in 1:manager.n_trees
        # Create tree-specific random number generator
        tree_seed = manager.config.randomization_seed_offset + tree_id
        tree_rng = MersenneTwister(tree_seed)
        
        # Store seed for reproducibility
        manager.tree_random_seeds[tree_id] = tree_seed
        
        # Randomly select features for this tree
        available_features = randperm(tree_rng, manager.n_features)[1:n_features_per_tree]
        
        # Create feature mask
        feature_mask = fill(false, manager.n_features)
        feature_mask[available_features] .= true
        
        manager.tree_feature_masks[tree_id] = feature_mask
    end
    
    @info "Applied feature subsampling: $(n_features_per_tree)/$(manager.n_features) features per tree"
end

"""
Apply exploration constant variation across trees
"""
function apply_exploration_variation!(manager::DiversityManager)
    min_c = manager.config.exploration_constant_min
    max_c = manager.config.exploration_constant_max
    
    for tree_id in 1:manager.n_trees
        if manager.config.exploration_strategy == "uniform"
            # Uniform distribution
            tree_seed = manager.config.randomization_seed_offset + tree_id
            tree_rng = MersenneTwister(tree_seed)
            exploration_constant = min_c + (max_c - min_c) * rand(tree_rng)
            
        elseif manager.config.exploration_strategy == "normal"
            # Normal distribution centered at midpoint
            tree_seed = manager.config.randomization_seed_offset + tree_id
            tree_rng = MersenneTwister(tree_seed)
            center = (min_c + max_c) / 2
            std_dev = (max_c - min_c) / 6  # 99.7% within range
            exploration_constant = clamp(center + std_dev * randn(tree_rng), min_c, max_c)
            
        elseif manager.config.exploration_strategy == "adaptive"
            # Adaptive based on tree position in ensemble
            progress = (tree_id - 1) / (manager.n_trees - 1)
            exploration_constant = min_c + (max_c - min_c) * progress
            
        else
            # Default to uniform
            exploration_constant = min_c + (max_c - min_c) * ((tree_id - 1) / (manager.n_trees - 1))
        end
        
        manager.tree_exploration_constants[tree_id] = Float32(exploration_constant)
    end
    
    constants = collect(values(manager.tree_exploration_constants))
    @info "Applied exploration variation: range [$(minimum(constants):.3f), $(maximum(constants):.3f)]"
end

"""
Apply initial state randomization
"""
function apply_initial_randomization!(manager::DiversityManager)
    n_initial_features = round(Int, manager.n_features * manager.config.initial_selection_rate)
    
    for tree_id in 1:manager.n_trees
        tree_seed = manager.config.randomization_seed_offset + tree_id + 10000
        tree_rng = MersenneTwister(tree_seed)
        
        # Only select from available features for this tree
        available_features = findall(manager.tree_feature_masks[tree_id])
        
        if length(available_features) >= n_initial_features
            # Randomly select initial features
            initial_features = sample(tree_rng, available_features, n_initial_features, replace=false)
            
            # This would be applied to the actual tree during initialization
            # For now, we just store the information
            if !haskey(manager.tree_feature_masks, Symbol("initial_$tree_id"))
                manager.tree_feature_masks[Symbol("initial_$tree_id")] = fill(false, manager.n_features)
                manager.tree_feature_masks[Symbol("initial_$tree_id")][initial_features] .= true
            end
        end
    end
    
    @info "Applied initial randomization: $(n_initial_features) features selected initially per tree"
end

"""
Apply feature masking diversity to reduce overlap
"""
function apply_feature_masking_diversity!(manager::DiversityManager)
    # Calculate current overlap
    calculate_diversity_metrics!(manager)
    
    if manager.diversity_metrics.average_overlap <= manager.config.max_overlap_threshold
        return  # Already diverse enough
    end
    
    # Apply masking to reduce overlap
    for tree_id in 1:manager.n_trees
        current_mask = manager.tree_feature_masks[tree_id]
        
        # Find features used by other trees
        other_usage = zeros(Int, manager.n_features)
        for other_id in 1:manager.n_trees
            if other_id != tree_id
                other_mask = manager.tree_feature_masks[other_id]
                other_usage .+= Int.(other_mask)
            end
        end
        
        # Prefer features with lower usage by other trees
        available_indices = findall(current_mask)
        usage_weights = 1.0 ./ (1.0 .+ other_usage[available_indices])
        
        # Resample features with bias toward less used ones
        tree_seed = manager.config.randomization_seed_offset + tree_id + 20000
        tree_rng = MersenneTwister(tree_seed)
        
        n_features_to_keep = sum(current_mask)
        if length(available_indices) > n_features_to_keep
            # Weighted sampling without replacement
            selected_indices = weighted_sample_without_replacement(
                tree_rng, available_indices, usage_weights, n_features_to_keep
            )
            
            # Update mask
            new_mask = fill(false, manager.n_features)
            new_mask[selected_indices] .= true
            manager.tree_feature_masks[tree_id] = new_mask
        end
    end
    
    # Recalculate metrics after masking
    calculate_diversity_metrics!(manager)
    
    @info "Applied feature masking: average overlap reduced to $(manager.diversity_metrics.average_overlap:.3f)"
end

"""
Weighted sampling without replacement
"""
function weighted_sample_without_replacement(rng::AbstractRNG, items::Vector{Int}, 
                                           weights::Vector{Float64}, n::Int)
    if length(items) <= n
        return items
    end
    
    # Normalize weights
    weights = weights ./ sum(weights)
    
    selected = Int[]
    remaining_items = copy(items)
    remaining_weights = copy(weights)
    
    for _ in 1:n
        # Sample one item
        idx = sample(rng, 1:length(remaining_items), Weights(remaining_weights))
        push!(selected, remaining_items[idx])
        
        # Remove selected item
        deleteat!(remaining_items, idx)
        deleteat!(remaining_weights, idx)
        
        # Renormalize weights
        if !isempty(remaining_weights)
            remaining_weights ./= sum(remaining_weights)
        end
    end
    
    return selected
end

"""
Calculate comprehensive diversity metrics
"""
function calculate_diversity_metrics!(manager::DiversityManager)
    start_time = time()
    
    # Calculate pairwise overlap matrix
    for i in 1:manager.n_trees
        for j in 1:manager.n_trees
            if i == j
                manager.diversity_metrics.pairwise_overlap_matrix[i, j] = 1.0f0
            else
                mask_i = manager.tree_feature_masks[i]
                mask_j = manager.tree_feature_masks[j]
                
                intersection = sum(mask_i .& mask_j)
                union_size = sum(mask_i .| mask_j)
                
                # Jaccard similarity
                overlap = union_size > 0 ? Float32(intersection / union_size) : 0.0f0
                manager.diversity_metrics.pairwise_overlap_matrix[i, j] = overlap
            end
        end
    end
    
    # Calculate summary statistics
    upper_triangle = [manager.diversity_metrics.pairwise_overlap_matrix[i, j] 
                     for i in 1:manager.n_trees for j in (i+1):manager.n_trees]
    
    if !isempty(upper_triangle)
        manager.diversity_metrics.average_overlap = mean(upper_triangle)
        manager.diversity_metrics.max_overlap = maximum(upper_triangle)
        manager.diversity_metrics.min_overlap = minimum(upper_triangle)
    end
    
    # Calculate feature selection entropy
    calculate_feature_selection_entropy!(manager)
    
    # Calculate exploration constant diversity
    if !isempty(manager.tree_exploration_constants)
        constants = collect(values(manager.tree_exploration_constants))
        manager.diversity_metrics.exploration_constant_variance = var(constants)
    end
    
    # Calculate overall diversity score
    calculate_overall_diversity_score!(manager)
    
    # Update metadata
    manager.diversity_metrics.last_update = now()
    manager.diversity_metrics.update_count += 1
    manager.diversity_metrics.computation_time = time() - start_time
    
    # Store in history
    push!(manager.diversity_history, (now(), manager.diversity_metrics.overall_diversity_score))
    
    # Limit history size
    if length(manager.diversity_history) > 10000
        deleteat!(manager.diversity_history, 1)
    end
    
    # Update trend
    push!(manager.diversity_metrics.diversity_trend, manager.diversity_metrics.overall_diversity_score)
    if length(manager.diversity_metrics.diversity_trend) > 100
        deleteat!(manager.diversity_metrics.diversity_trend, 1)
    end
end

"""
Calculate feature selection entropy
"""
function calculate_feature_selection_entropy!(manager::DiversityManager)
    # Count feature usage across trees
    manager.diversity_metrics.feature_usage_distribution .= 0
    
    for tree_id in 1:manager.n_trees
        mask = manager.tree_feature_masks[tree_id]
        for (i, selected) in enumerate(mask)
            if selected
                manager.diversity_metrics.feature_usage_distribution[i] += 1
            end
        end
    end
    
    # Calculate entropy
    total_selections = sum(manager.diversity_metrics.feature_usage_distribution)
    if total_selections > 0
        probabilities = manager.diversity_metrics.feature_usage_distribution ./ total_selections
        probabilities = probabilities[probabilities .> 0]  # Remove zeros
        
        if !isempty(probabilities)
            manager.diversity_metrics.feature_selection_entropy = Float32(-sum(probabilities .* log2.(probabilities)))
        end
    end
    
    # Count unique feature combinations
    unique_combinations = Set{Vector{Bool}}()
    for tree_id in 1:manager.n_trees
        push!(unique_combinations, manager.tree_feature_masks[tree_id])
    end
    manager.diversity_metrics.unique_feature_combinations = length(unique_combinations)
end

"""
Calculate overall diversity score
"""
function calculate_overall_diversity_score!(manager::DiversityManager)
    # Combine different diversity measures into overall score
    
    # Overlap diversity (lower overlap = higher diversity)
    overlap_diversity = 1.0f0 - manager.diversity_metrics.average_overlap
    
    # Feature entropy diversity (normalized by maximum possible entropy)
    max_entropy = log2(Float32(manager.n_features))
    entropy_diversity = manager.diversity_metrics.feature_selection_entropy / max_entropy
    
    # Exploration constant diversity (normalized by variance range)
    const_range = (manager.config.exploration_constant_max - manager.config.exploration_constant_min)^2 / 12  # Variance of uniform distribution
    exploration_diversity = min(1.0f0, manager.diversity_metrics.exploration_constant_variance / const_range)
    
    # Unique combinations diversity
    max_combinations = manager.n_trees
    combination_diversity = Float32(manager.diversity_metrics.unique_feature_combinations) / max_combinations
    
    # Weighted combination
    weights = [0.4f0, 0.3f0, 0.2f0, 0.1f0]  # Overlap, entropy, exploration, combinations
    diversities = [overlap_diversity, entropy_diversity, exploration_diversity, combination_diversity]
    
    manager.diversity_metrics.overall_diversity_score = sum(weights .* diversities)
end

"""
Update diversity mechanisms based on current state
"""
function update_diversity_mechanisms!(manager::DiversityManager, forest_manager::ForestManager)
    if !manager.config.enable_adaptive_diversity
        return
    end
    
    # Check if it's time for diversity update
    time_since_check = (now() - manager.last_diversity_check).value / 1000.0  # seconds
    if time_since_check < manager.config.diversity_check_frequency
        return
    end
    
    # Calculate current diversity
    calculate_diversity_metrics!(manager)
    current_diversity = manager.diversity_metrics.overall_diversity_score
    
    # Adapt diversity mechanisms based on current state
    if current_diversity < manager.config.target_diversity_score
        # Increase diversity
        increase_diversity!(manager)
    elseif current_diversity > manager.config.target_diversity_score * 1.2
        # Slightly reduce diversity if too high
        decrease_diversity!(manager)
    end
    
    # Update timestamp
    manager.last_diversity_check = now()
    manager.adaptation_iterations += 1
    
    @info "Diversity update: score=$(current_diversity:.3f), target=$(manager.config.target_diversity_score:.3f)"
end

"""
Increase ensemble diversity
"""
function increase_diversity!(manager::DiversityManager)
    # Reduce feature subsampling rate slightly
    if manager.config.enable_dynamic_subsampling
        old_rate = manager.current_subsample_rate
        manager.current_subsample_rate = max(
            0.5f0,  # Minimum rate
            manager.current_subsample_rate - manager.config.diversity_adaptation_rate
        )
        
        if manager.current_subsample_rate != old_rate
            apply_feature_subsampling!(manager)
            @info "Reduced subsample rate from $(old_rate:.3f) to $(manager.current_subsample_rate:.3f)"
        end
    end
    
    # Re-apply feature masking with stricter overlap threshold
    if manager.config.enable_feature_masking
        original_threshold = manager.config.max_overlap_threshold
        # Temporarily reduce threshold for more aggressive diversity
        temp_config = DiversityConfig(
            manager.config.feature_subsample_rate,
            manager.config.min_features_per_tree,
            manager.config.enable_dynamic_subsampling,
            manager.config.exploration_constant_min,
            manager.config.exploration_constant_max,
            manager.config.exploration_strategy,
            manager.config.enable_initial_randomization,
            manager.config.initial_selection_rate,
            manager.config.randomization_seed_offset,
            manager.config.enable_feature_masking,
            max(0.3f0, original_threshold - 0.1f0),  # Stricter threshold
            manager.config.overlap_penalty_weight,
            manager.config.target_diversity_score,
            manager.config.diversity_check_frequency,
            manager.config.enable_diversity_monitoring,
            manager.config.enable_adaptive_diversity,
            manager.config.diversity_adaptation_rate,
            manager.config.convergence_diversity_threshold
        )
        
        old_config = manager.config
        manager.config = temp_config
        apply_feature_masking_diversity!(manager)
        manager.config = old_config  # Restore original config
    end
end

"""
Decrease ensemble diversity (allow more convergence)
"""
function decrease_diversity!(manager::DiversityManager)
    # Increase feature subsampling rate slightly
    if manager.config.enable_dynamic_subsampling
        old_rate = manager.current_subsample_rate
        manager.current_subsample_rate = min(
            1.0f0,  # Maximum rate
            manager.current_subsample_rate + manager.config.diversity_adaptation_rate
        )
        
        if manager.current_subsample_rate != old_rate
            apply_feature_subsampling!(manager)
            @info "Increased subsample rate from $(old_rate:.3f) to $(manager.current_subsample_rate:.3f)"
        end
    end
end

"""
Get diversity status and statistics
"""
function get_diversity_status(manager::DiversityManager)
    return Dict{String, Any}(
        "manager_state" => manager.manager_state,
        "n_trees" => manager.n_trees,
        "n_features" => manager.n_features,
        "current_subsample_rate" => manager.current_subsample_rate,
        "adaptation_iterations" => manager.adaptation_iterations,
        "overall_diversity_score" => manager.diversity_metrics.overall_diversity_score,
        "average_overlap" => manager.diversity_metrics.average_overlap,
        "max_overlap" => manager.diversity_metrics.max_overlap,
        "min_overlap" => manager.diversity_metrics.min_overlap,
        "feature_selection_entropy" => manager.diversity_metrics.feature_selection_entropy,
        "unique_combinations" => manager.diversity_metrics.unique_feature_combinations,
        "exploration_constant_variance" => manager.diversity_metrics.exploration_constant_variance,
        "last_computation_time" => manager.last_computation_time,
        "error_count" => length(manager.error_log),
        "last_update" => manager.diversity_metrics.last_update
    )
end

"""
Generate comprehensive diversity report
"""
function generate_diversity_report(manager::DiversityManager)
    status = get_diversity_status(manager)
    
    report = String[]
    
    push!(report, "=== Ensemble Diversity Report ===")
    push!(report, "Generated: $(now())")
    push!(report, "Manager State: $(status["manager_state"])")
    push!(report, "")
    
    # Configuration summary
    push!(report, "Configuration:")
    push!(report, "  Feature Subsample Rate: $(manager.config.feature_subsample_rate)")
    push!(report, "  Current Subsample Rate: $(status["current_subsample_rate"])")
    push!(report, "  Exploration Range: [$(manager.config.exploration_constant_min), $(manager.config.exploration_constant_max)]")
    push!(report, "  Max Overlap Threshold: $(manager.config.max_overlap_threshold)")
    push!(report, "")
    
    # Diversity metrics
    push!(report, "Diversity Metrics:")
    push!(report, "  Overall Score: $(round(status["overall_diversity_score"], digits=3))")
    push!(report, "  Average Overlap: $(round(status["average_overlap"], digits=3))")
    push!(report, "  Overlap Range: [$(round(status["min_overlap"], digits=3)), $(round(status["max_overlap"], digits=3))]")
    push!(report, "  Feature Entropy: $(round(status["feature_selection_entropy"], digits=3))")
    push!(report, "  Unique Combinations: $(status["unique_combinations"])/$(status["n_trees"])")
    push!(report, "  Exploration Variance: $(round(status["exploration_constant_variance"], digits=4))")
    push!(report, "")
    
    # Feature usage distribution
    feature_usage = manager.diversity_metrics.feature_usage_distribution
    if !isempty(feature_usage)
        push!(report, "Feature Usage Statistics:")
        push!(report, "  Most Used Feature: $(maximum(feature_usage)) trees")
        push!(report, "  Least Used Feature: $(minimum(feature_usage)) trees")
        push!(report, "  Average Usage: $(round(mean(feature_usage), digits=1)) trees")
        push!(report, "  Usage Std Dev: $(round(std(feature_usage), digits=1))")
        push!(report, "")
    end
    
    # Performance
    push!(report, "Performance:")
    push!(report, "  Last Computation: $(round(status["last_computation_time"]*1000, digits=1))ms")
    push!(report, "  Adaptation Iterations: $(status["adaptation_iterations"])")
    push!(report, "  Update Count: $(manager.diversity_metrics.update_count)")
    
    if status["error_count"] > 0
        push!(report, "")
        push!(report, "Recent Errors:")
        for error in manager.error_log[max(1, length(manager.error_log)-2):end]
            push!(report, "  - $error")
        end
    end
    
    push!(report, "")
    push!(report, "=== End Diversity Report ===")
    
    return join(report, "\n")
end

"""
Apply diversity configuration to forest manager
"""
function apply_diversity_to_forest!(manager::DiversityManager, forest_manager::ForestManager)
    # Apply exploration constants to trees
    for (tree_id, exploration_constant) in manager.tree_exploration_constants
        if haskey(forest_manager.trees, tree_id)
            tree = forest_manager.trees[tree_id]
            tree.config["exploration_constant"] = exploration_constant
        end
    end
    
    # Apply feature masks (this would integrate with actual MCTS implementation)
    for (tree_id, feature_mask) in manager.tree_feature_masks
        if haskey(forest_manager.trees, tree_id)
            tree = forest_manager.trees[tree_id]
            tree.metadata["available_features"] = feature_mask
            tree.metadata["n_available_features"] = sum(feature_mask)
        end
    end
    
    # Apply random seeds
    for (tree_id, seed) in manager.tree_random_seeds
        if haskey(forest_manager.trees, tree_id)
            tree = forest_manager.trees[tree_id]
            tree.config["random_seed"] = seed
        end
    end
    
    @info "Applied diversity configuration to $(length(forest_manager.trees)) trees"
end

# Export main types and functions
export DiversityStrategy, DiversityConfig, DiversityMetrics, DiversityManager
export RANDOM_SUBSAMPLING, EXPLORATION_VARIATION, INITIAL_RANDOMIZATION, MASKING_DIVERSITY, COMBINED_DIVERSITY
export create_diversity_config, initialize_diversity_manager
export apply_diversity_mechanisms!, calculate_diversity_metrics!, update_diversity_mechanisms!
export get_diversity_status, generate_diversity_report, apply_diversity_to_forest!

end # module DiversityMechanisms