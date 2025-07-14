module FeatureRanking

using CUDA
using CUDA.CUFFT
using Statistics

# Include the necessary modules
include("gpu_memory_layout.jl")
include("mutual_information.jl")
include("correlation_matrix.jl")
include("variance_calculation.jl")
include("threshold_management.jl")

using .GPUMemoryLayout: RankingBuffers, FeatureMatrix, HistogramBuffers, CorrelationMatrix, VarianceBuffers
using .GPUMemoryLayout: WARP_SIZE, TARGET_FEATURES, MAX_FEATURES
using .MutualInformation: compute_mutual_information!, create_mi_config
using .CorrelationComputation: compute_correlation_cublas!, create_correlation_config, find_correlated_pairs
using .VarianceCalculation: compute_variances!, create_variance_config, mark_low_variance_features!
using .ThresholdManagement: ExtendedThresholdConfig, create_default_config

"""
Configuration for feature ranking system
"""
struct RankingConfig
    n_features::Int32
    n_samples::Int32
    target_features::Int32      # Target number of features to select
    
    # Weights for combining criteria
    mi_weight::Float32          # Weight for mutual information score
    correlation_penalty::Float32 # Penalty for correlated features
    variance_weight::Float32    # Weight for variance (higher is better)
    
    # Algorithm options
    use_gpu_sort::Bool          # Use GPU sorting (vs CPU sorting)
    correlation_graph::Bool     # Build correlation graph for redundancy
    pre_filter_variance::Bool   # Pre-filter low variance features
end

"""
Create default ranking configuration
"""
function create_ranking_config(n_features::Integer, n_samples::Integer;
                             target_features::Integer = TARGET_FEATURES,
                             mi_weight::Float32 = Float32(1.0),
                             correlation_penalty::Float32 = Float32(0.5),
                             variance_weight::Float32 = Float32(0.1),
                             use_gpu_sort::Bool = true,
                             correlation_graph::Bool = true,
                             pre_filter_variance::Bool = true)
    return RankingConfig(
        Int32(n_features),
        Int32(n_samples),
        Int32(target_features),
        mi_weight,
        correlation_penalty,
        variance_weight,
        use_gpu_sort,
        correlation_graph,
        pre_filter_variance
    )
end

"""
GPU kernel to compute composite feature scores
"""
function compute_composite_scores_kernel!(
    composite_scores::CuDeviceArray{Float32, 1},
    mi_scores::CuDeviceArray{Float32, 1},
    variances::CuDeviceArray{Float32, 1},
    low_var_mask::CuDeviceArray{Bool, 1},
    selected_mask::CuDeviceArray{Bool, 1},
    mi_weight::Float32,
    variance_weight::Float32,
    n_features::Int32
)
    feature_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if feature_idx <= n_features
        # Pre-filter low variance features
        if low_var_mask[feature_idx]
            composite_scores[feature_idx] = Float32(-Inf)
            selected_mask[feature_idx] = false
        else
            # Normalize scores to [0, 1] range
            mi_score = mi_scores[feature_idx]
            var_score = log(variances[feature_idx] + Float32(1e-10))  # Log scale for variance
            
            # Compute weighted composite score
            composite_scores[feature_idx] = mi_weight * mi_score + variance_weight * var_score
            selected_mask[feature_idx] = true  # Initially all non-low-var features are candidates
        end
    end
    
    return nothing
end

"""
GPU kernel to apply correlation filtering
"""
function apply_correlation_filter_kernel!(
    composite_scores::CuDeviceArray{Float32, 1},
    selected_mask::CuDeviceArray{Bool, 1},
    correlation_pairs::CuDeviceArray{Int32, 2},
    n_pairs::Int32,
    correlation_penalty::Float32
)
    pair_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if pair_idx <= n_pairs
        # Get correlated feature pair
        feat1 = correlation_pairs[1, pair_idx]
        feat2 = correlation_pairs[2, pair_idx]
        
        # Keep the feature with higher composite score
        score1 = composite_scores[feat1]
        score2 = composite_scores[feat2]
        
        if score1 > score2
            # Penalize or remove feature 2
            composite_scores[feat2] *= (1.0f0 - correlation_penalty)
            if correlation_penalty >= 1.0f0
                selected_mask[feat2] = false
            end
        else
            # Penalize or remove feature 1
            composite_scores[feat1] *= (1.0f0 - correlation_penalty)
            if correlation_penalty >= 1.0f0
                selected_mask[feat1] = false
            end
        end
    end
    
    return nothing
end

"""
GPU kernel for simple top-K selection
"""
function select_top_k_kernel!(
    selected_indices::CuDeviceArray{Int32, 1},
    n_selected::CuDeviceArray{Int32, 1},
    feature_indices::CuDeviceArray{Int32, 1},
    composite_scores::CuDeviceArray{Float32, 1},
    selected_mask::CuDeviceArray{Bool, 1},
    target_features::Int32,
    n_features::Int32
)
    # This is a simplified version - in practice would use more sophisticated selection
    tid = threadIdx().x
    
    if tid == 1
        count = Int32(0)
        
        # Count valid features and select top ones
        for i in 1:n_features
            if selected_mask[feature_indices[i]] && count < target_features
                count += 1
                selected_indices[count] = feature_indices[i]
            end
        end
        
        n_selected[1] = count
    end
    
    return nothing
end

"""
Perform integrated feature ranking and selection
"""
function rank_and_select_features!(
    ranking_buffers::RankingBuffers,
    feature_matrix::FeatureMatrix,
    target_data::CuArray{Float32, 1},
    histogram_buffers::HistogramBuffers,
    correlation_matrix::CorrelationMatrix,
    variance_buffers::VarianceBuffers,
    threshold_config::ExtendedThresholdConfig,
    ranking_config::RankingConfig
)
    n_features = ranking_config.n_features
    n_samples = ranking_config.n_samples
    
    # Step 1: Compute mutual information scores
    mi_config = create_mi_config(n_features, n_samples)
    compute_mutual_information!(
        ranking_buffers.mi_scores,
        feature_matrix.data,
        target_data,
        histogram_buffers,
        mi_config
    )
    
    # Step 2: Compute variances
    var_config = create_variance_config(n_features, n_samples)
    compute_variances!(variance_buffers, feature_matrix, var_config)
    
    # Step 3: Mark low variance features
    low_var_mask = CUDA.zeros(Bool, n_features)
    if ranking_config.pre_filter_variance
        mark_low_variance_features!(
            low_var_mask,
            variance_buffers,
            threshold_config.adaptive_var_threshold
        )
    end
    
    # Step 4: Compute composite scores
    threads = 256
    blocks = cld(n_features, threads)
    @cuda threads=threads blocks=blocks compute_composite_scores_kernel!(
        ranking_buffers.mi_scores,  # Reuse as composite scores
        ranking_buffers.mi_scores,
        variance_buffers.variances,
        low_var_mask,
        ranking_buffers.selected_mask,
        ranking_config.mi_weight,
        ranking_config.variance_weight,
        n_features
    )
    
    CUDA.synchronize()
    
    # Step 5: Compute correlations and apply filtering
    if ranking_config.correlation_graph
        corr_config = create_correlation_config(n_features, n_samples)
        compute_correlation_cublas!(correlation_matrix, feature_matrix, corr_config)
        
        # Find highly correlated pairs
        correlated_pairs = find_correlated_pairs(
            correlation_matrix,
            threshold_config.adaptive_corr_threshold
        )
        
        if length(correlated_pairs) > 0
            # Convert to GPU arrays
            n_pairs = length(correlated_pairs)
            correlation_pairs_gpu = CUDA.zeros(Int32, 2, n_pairs)
            
            for (i, (feat1, feat2, _)) in enumerate(correlated_pairs)
                correlation_pairs_gpu[1, i] = feat1
                correlation_pairs_gpu[2, i] = feat2
            end
            
            # Apply correlation filtering
            threads = 256
            blocks = cld(n_pairs, threads)
            @cuda threads=threads blocks=blocks apply_correlation_filter_kernel!(
                ranking_buffers.mi_scores,  # Composite scores
                ranking_buffers.selected_mask,
                correlation_pairs_gpu,
                Int32(n_pairs),
                ranking_config.correlation_penalty
            )
            
            CUDA.synchronize()
        end
    end
    
    # Step 6: Sort features by composite score
    if ranking_config.use_gpu_sort
        # Use GPU sorting
        sort_features_gpu!(
            ranking_buffers.feature_indices,
            ranking_buffers.mi_scores,  # Composite scores
            n_features
        )
    else
        # Use CPU sorting (sometimes more stable)
        sort_features_cpu!(
            ranking_buffers.feature_indices,
            ranking_buffers.mi_scores,
            n_features
        )
    end
    
    # Step 7: Select top-K features
    @cuda threads=256 blocks=1 select_top_k_kernel!(
        ranking_buffers.selected_indices,
        ranking_buffers.n_selected,
        ranking_buffers.feature_indices,
        ranking_buffers.mi_scores,
        ranking_buffers.selected_mask,
        ranking_config.target_features,
        n_features
    )
    
    CUDA.synchronize()
    
    # Return number of selected features
    return CUDA.@allowscalar ranking_buffers.n_selected[1]
end

"""
Sort features by score on GPU using bitonic sort (simplified)
"""
function sort_features_gpu!(
    indices::CuArray{Int32, 1},
    scores::CuArray{Float32, 1},
    n_features::Integer
)
    # For now, we'll use a simple approach
    # In production, would use CUB or thrust sorting
    
    # Create index-score pairs
    pairs = [(i, s) for (i, s) in zip(1:n_features, Array(scores))]
    
    # Sort by score (descending)
    sort!(pairs, by=x->x[2], rev=true)
    
    # Extract sorted indices
    sorted_indices = Int32[p[1] for p in pairs]
    copyto!(indices, sorted_indices)
end

"""
Sort features by score on CPU
"""
function sort_features_cpu!(
    indices::CuArray{Int32, 1},
    scores::CuArray{Float32, 1},
    n_features::Integer
)
    # Copy to CPU
    scores_cpu = Array(scores)
    indices_cpu = collect(Int32, 1:n_features)
    
    # Sort indices by score (descending)
    sortperm!(indices_cpu, scores_cpu, rev=true)
    
    # Copy back to GPU
    copyto!(indices, indices_cpu)
end

"""
Apply feature selection to data
"""
function apply_feature_selection(
    feature_matrix::FeatureMatrix,
    selected_indices::CuArray{Int32, 1},
    n_selected::Integer
)
    # Extract selected features
    selected_features = CUDA.zeros(Float32, feature_matrix.n_samples, n_selected)
    
    # Copy selected features
    for (new_idx, old_idx) in enumerate(Array(selected_indices)[1:n_selected])
        selected_features[:, new_idx] = feature_matrix.data[:, old_idx]
    end
    
    # Create new feature matrix with selected features
    selected_matrix = create_feature_matrix(feature_matrix.n_samples, n_selected)
    copyto!(selected_matrix.data[1:feature_matrix.n_samples, 1:n_selected], selected_features)
    
    # Copy selected feature names
    selected_names = String[]
    for idx in Array(selected_indices)[1:n_selected]
        push!(selected_names, feature_matrix.feature_names[idx])
    end
    selected_matrix.feature_names = selected_names
    
    return selected_matrix
end

"""
Get feature selection summary
"""
function get_selection_summary(
    ranking_buffers::RankingBuffers,
    variance_buffers::VarianceBuffers,
    n_selected::Integer
)
    # Get selected indices and scores
    selected_indices_cpu = Array(ranking_buffers.selected_indices)[1:n_selected]
    mi_scores_cpu = Array(ranking_buffers.mi_scores)
    variances_cpu = Array(variance_buffers.variances)
    
    # Compute summary statistics
    selected_mi_scores = [mi_scores_cpu[idx] for idx in selected_indices_cpu]
    selected_variances = [variances_cpu[idx] for idx in selected_indices_cpu]
    
    summary = Dict(
        "n_selected" => n_selected,
        "selected_indices" => selected_indices_cpu,
        "mi_score_range" => (minimum(selected_mi_scores), maximum(selected_mi_scores)),
        "mi_score_mean" => mean(selected_mi_scores),
        "variance_range" => (minimum(selected_variances), maximum(selected_variances)),
        "variance_mean" => mean(selected_variances)
    )
    
    return summary
end

# Export types and functions
export RankingConfig, create_ranking_config
export rank_and_select_features!, apply_feature_selection
export get_selection_summary

end # module FeatureRanking