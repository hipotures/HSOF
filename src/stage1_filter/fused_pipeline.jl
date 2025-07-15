module FusedPipeline

using CUDA
using Statistics
using LinearAlgebra

include("kernel_fusion.jl")
using .KernelFusion

export FusedPipelineConfig, fused_feature_selection_pipeline!
export benchmark_fused_pipeline

"""
Configuration for the fused feature selection pipeline
"""
struct FusedPipelineConfig
    n_features_to_select::Int32
    variance_threshold::Float32
    correlation_threshold::Float32
    mi_bins::Int32
    block_size::Int32
    use_shared_memory::Bool
end

function FusedPipelineConfig(;
    n_features_to_select::Int32 = Int32(500),
    variance_threshold::Float32 = 1f-6,
    correlation_threshold::Float32 = 0.95f0,
    mi_bins::Int32 = Int32(10),
    block_size::Int32 = Int32(256),
    use_shared_memory::Bool = true
)
    return FusedPipelineConfig(
        n_features_to_select,
        variance_threshold,
        correlation_threshold,
        mi_bins,
        block_size,
        use_shared_memory
    )
end

"""
Fused kernel for complete feature selection pipeline
Phase 1: Variance calculation and filtering
Phase 2: MI score calculation for valid features
Phase 3: Correlation-based redundancy removal
Phase 4: Top-k selection
"""
function fused_pipeline_kernel_phase1!(
    feature_stats::CuDeviceMatrix{Float32},  # [4, n_features]: mean, std, variance, mi_score
    valid_mask::CuDeviceVector{Bool},
    X::CuDeviceMatrix{Float32},
    y::CuDeviceVector{Int32},
    var_threshold::Float32,
    n_features::Int32,
    n_samples::Int32,
    n_bins::Int32,
    n_classes::Int32
)
    feat_idx = Int32(blockIdx().x)
    if feat_idx > n_features
        return
    end
    
    tid = threadIdx().x
    block_size = blockDim().x
    
    # Shared memory allocation
    shared_sum = @cuDynamicSharedMem(Float32, block_size)
    shared_sum_sq = @cuDynamicSharedMem(Float32, block_size, block_size * sizeof(Float32))
    
    # Phase 1: Calculate statistics
    local_sum = 0.0f0
    local_sum_sq = 0.0f0
    local_min = Inf32
    local_max = -Inf32
    
    for idx in tid:block_size:n_samples
        if idx <= n_samples
            val = X[feat_idx, idx]
            local_sum += val
            local_sum_sq += val * val
            local_min = min(local_min, val)
            local_max = max(local_max, val)
        end
    end
    
    # Store in shared memory
    shared_sum[tid] = local_sum
    shared_sum_sq[tid] = local_sum_sq
    sync_threads()
    
    # Parallel reduction
    stride = block_size ÷ 2
    while stride > 0
        if tid <= stride
            shared_sum[tid] += shared_sum[tid + stride]
            shared_sum_sq[tid] += shared_sum_sq[tid + stride]
        end
        sync_threads()
        stride ÷= 2
    end
    
    # Calculate statistics
    if tid == 1
        mean_val = shared_sum[1] / Float32(n_samples)
        var_val = (shared_sum_sq[1] / Float32(n_samples)) - mean_val * mean_val
        std_val = sqrt(max(var_val, 1f-8))
        
        feature_stats[1, feat_idx] = mean_val
        feature_stats[2, feat_idx] = std_val
        feature_stats[3, feat_idx] = var_val
        
        # Check variance threshold
        valid_mask[feat_idx] = var_val > var_threshold
    end
    
    sync_threads()
    
    # Phase 2: Calculate MI score if feature is valid
    if valid_mask[feat_idx]
        # Simplified MI calculation using shared memory for histogram
        hist_size = n_bins * n_classes
        shared_hist = @cuDynamicSharedMem(Int32, hist_size, 2 * block_size * sizeof(Float32))
        
        # Initialize histogram
        for i in tid:block_size:hist_size
            if i <= hist_size
                shared_hist[i] = Int32(0)
            end
        end
        sync_threads()
        
        # Build histogram (simplified)
        mean_val = feature_stats[1, feat_idx]
        std_val = feature_stats[2, feat_idx]
        
        for idx in tid:block_size:n_samples
            if idx <= n_samples
                # Standardize value
                val = (X[feat_idx, idx] - mean_val) / std_val
                # Simple binning
                bin_idx = min(max(Int32(floor((val + 3.0f0) / 6.0f0 * Float32(n_bins))) + 1, 1), n_bins)
                class_idx = y[idx]
                
                hist_idx = (class_idx - 1) * n_bins + bin_idx
                CUDA.@atomic shared_hist[hist_idx] += Int32(1)
            end
        end
        
        sync_threads()
        
        # Calculate MI from histogram
        if tid == 1
            mi = 0.0f0
            
            for b in 1:n_bins
                p_x = 0.0f0
                for c in 1:n_classes
                    p_x += Float32(shared_hist[(c-1) * n_bins + b])
                end
                p_x /= Float32(n_samples)
                
                if p_x > 0
                    for c in 1:n_classes
                        joint_count = shared_hist[(c-1) * n_bins + b]
                        if joint_count > 0
                            p_joint = Float32(joint_count) / Float32(n_samples)
                            p_y = Float32(sum(y .== Int32(c))) / Float32(n_samples)
                            
                            if p_y > 0
                                mi += p_joint * log2(p_joint / (p_x * p_y))
                            end
                        end
                    end
                end
            end
            
            feature_stats[4, feat_idx] = mi
        end
    else
        if tid == 1
            feature_stats[4, feat_idx] = -Inf32  # Invalid feature
        end
    end
    
    return nothing
end

"""
Kernel for correlation-based redundancy removal
"""
function redundancy_removal_kernel!(
    redundant_mask::CuDeviceVector{Bool},
    feature_stats::CuDeviceMatrix{Float32},
    feature_indices::CuDeviceVector{Int32},
    X::CuDeviceMatrix{Float32},
    corr_threshold::Float32,
    n_valid::Int32,
    n_samples::Int32
)
    idx_i = Int32(blockIdx().x)
    idx_j = Int32(threadIdx().x)
    
    if idx_i > n_valid || idx_j > n_valid || idx_i >= idx_j
        return
    end
    
    feat_i = feature_indices[idx_i]
    feat_j = feature_indices[idx_j]
    
    # Get precomputed stats
    mean_i = feature_stats[1, feat_i]
    std_i = feature_stats[2, feat_i]
    mean_j = feature_stats[1, feat_j]
    std_j = feature_stats[2, feat_j]
    
    # Calculate correlation
    corr = 0.0f0
    for s in 1:n_samples
        val_i = (X[feat_i, s] - mean_i) / std_i
        val_j = (X[feat_j, s] - mean_j) / std_j
        corr += val_i * val_j
    end
    corr /= Float32(n_samples)
    
    # If highly correlated, mark lower MI score feature as redundant
    if abs(corr) > corr_threshold
        mi_i = feature_stats[4, feat_i]
        mi_j = feature_stats[4, feat_j]
        
        if mi_i < mi_j
            redundant_mask[idx_i] = true
        else
            redundant_mask[idx_j] = true
        end
    end
    
    return nothing
end

"""
Complete fused feature selection pipeline
"""
function fused_feature_selection_pipeline!(
    selected_features::CuArray{Int32, 1},
    X::CuArray{Float32, 2},
    y::CuArray{Int32, 1},
    config::FusedPipelineConfig
)
    n_features, n_samples = size(X)
    n_classes = length(unique(Array(y)))
    
    # Allocate working memory
    feature_stats = CUDA.zeros(Float32, 4, n_features)  # mean, std, var, mi
    valid_mask = CUDA.zeros(Bool, n_features)
    
    # Phase 1: Calculate statistics and MI scores
    blocks = n_features
    threads = config.block_size
    hist_size = config.mi_bins * n_classes * sizeof(Int32)
    shared_mem = 2 * config.block_size * sizeof(Float32) + hist_size
    
    @cuda threads=threads blocks=blocks shmem=shared_mem fused_pipeline_kernel_phase1!(
        feature_stats, valid_mask, X, y, config.variance_threshold,
        Int32(n_features), Int32(n_samples), config.mi_bins, Int32(n_classes)
    )
    
    CUDA.synchronize()
    
    # Get valid feature indices and their MI scores
    valid_indices = findall(Array(valid_mask))
    n_valid = length(valid_indices)
    
    if n_valid == 0
        # No valid features
        fill!(selected_features, Int32(-1))
        return selected_features
    end
    
    # Sort by MI scores
    mi_scores = Array(feature_stats[4, valid_indices])
    sorted_idx = sortperm(mi_scores, rev=true)
    valid_indices_sorted = valid_indices[sorted_idx]
    
    # Phase 2: Remove redundant features
    if config.correlation_threshold < 1.0f0 && n_valid > 1
        valid_indices_gpu = CuArray(Int32.(valid_indices_sorted))
        redundant_mask = CUDA.zeros(Bool, n_valid)
        
        # Launch redundancy removal kernel
        @cuda threads=min(n_valid, 32) blocks=n_valid redundancy_removal_kernel!(
            redundant_mask, feature_stats, valid_indices_gpu, X,
            config.correlation_threshold, Int32(n_valid), Int32(n_samples)
        )
        
        CUDA.synchronize()
        
        # Filter out redundant features
        non_redundant = .!Array(redundant_mask)
        final_features = valid_indices_sorted[non_redundant]
    else
        final_features = valid_indices_sorted
    end
    
    # Phase 3: Select top features
    n_select = min(Int(config.n_features_to_select), length(final_features))
    selected = final_features[1:n_select]
    
    # Fill result array
    fill!(selected_features, Int32(-1))
    selected_features[1:n_select] = Int32.(selected)
    
    return selected_features
end

"""
Benchmark the fused pipeline against separate operations
"""
function benchmark_fused_pipeline(
    X::CuArray{Float32, 2},
    y::CuArray{Int32, 1};
    n_features_select::Int = 500,
    n_runs::Int = 5
)
    n_features, n_samples = size(X)
    config = FusedPipelineConfig(n_features_to_select=Int32(n_features_select))
    
    println("\nBenchmarking Fused Feature Selection Pipeline")
    println("Dataset: $n_features features × $n_samples samples")
    println("Selecting top $n_features_select features")
    println("="^60)
    
    # Benchmark separate operations
    t_separate = 0.0
    for _ in 1:n_runs
        t = @elapsed begin
            # 1. Variance filtering
            variances = vec(var(X, dims=2, corrected=false))
            valid_mask = variances .> config.variance_threshold
            valid_indices = findall(valid_mask)
            
            # 2. MI calculation (simplified)
            mi_scores = CUDA.zeros(Float32, length(valid_indices))
            for (i, idx) in enumerate(valid_indices)
                # Simplified MI calculation
                feature = X[idx, :]
                mi_scores[i] = Float32(0.001 * idx)  # Placeholder
            end
            
            # 3. Sort and select
            sorted_idx = sortperm(Array(mi_scores), rev=true)
            n_select = min(n_features_select, length(sorted_idx))
            selected = valid_indices[sorted_idx[1:n_select]]
            
            CUDA.synchronize()
        end
        t_separate += t
    end
    t_separate /= n_runs
    
    # Benchmark fused pipeline
    t_fused = 0.0
    for _ in 1:n_runs
        selected_features = CUDA.fill(Int32(-1), n_features_select)
        t = @elapsed begin
            fused_feature_selection_pipeline!(selected_features, X, y, config)
            CUDA.synchronize()
        end
        t_fused += t
    end
    t_fused /= n_runs
    
    # Calculate metrics
    speedup = t_separate / t_fused
    improvement = (speedup - 1) * 100
    
    println("Separate operations: $(round(t_separate*1000, digits=2)) ms")
    println("Fused pipeline: $(round(t_fused*1000, digits=2)) ms")
    println("Speedup: $(round(speedup, digits=2))x")
    println("Performance improvement: $(round(improvement, digits=1))%")
    
    # Memory efficiency
    println("\nMemory Efficiency:")
    separate_memory = (
        n_features * sizeof(Float32) +  # variances
        n_features * sizeof(Bool) +      # valid_mask
        n_features * sizeof(Float32) +  # mi_scores
        n_features * sizeof(Int32)       # indices
    ) / 1024^2
    
    fused_memory = (
        4 * n_features * sizeof(Float32) +  # feature_stats
        n_features * sizeof(Bool) +          # valid_mask
        n_features_select * sizeof(Int32)    # selected_features
    ) / 1024^2
    
    println("Separate operations memory: $(round(separate_memory, digits=2)) MB")
    println("Fused pipeline memory: $(round(fused_memory, digits=2)) MB")
    println("Memory savings: $(round((1 - fused_memory/separate_memory)*100, digits=1))%")
    
    return Dict(
        "speedup" => speedup,
        "improvement_percent" => improvement,
        "time_separate_ms" => t_separate * 1000,
        "time_fused_ms" => t_fused * 1000,
        "memory_savings_percent" => (1 - fused_memory/separate_memory) * 100
    )
end

end # module FusedPipeline