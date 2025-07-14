module CorrelationComputation

using CUDA
using CUDA.CUBLAS
using Statistics
using LinearAlgebra

# Include the GPUMemoryLayout module
include("gpu_memory_layout.jl")
using .GPUMemoryLayout: CorrelationMatrix, FeatureMatrix, WARP_SIZE, TILE_SIZE, get_upper_triangle_index

"""
Configuration for correlation computation
"""
struct CorrelationConfig
    n_features::Int32
    n_samples::Int32
    use_cublas::Bool
    batch_size::Int32  # For processing large datasets
    epsilon::Float32   # Small value for numerical stability
end

"""
Create default correlation configuration
"""
function create_correlation_config(n_features::Integer, n_samples::Integer;
                                  use_cublas::Bool = true,
                                  batch_size::Integer = 100000,
                                  epsilon::Float32 = Float32(1e-8))
    return CorrelationConfig(
        Int32(n_features),
        Int32(n_samples),
        use_cublas,
        Int32(batch_size),
        epsilon
    )
end

"""
GPU kernel for computing feature means
"""
function compute_means_kernel!(
    means::CuDeviceArray{Float32, 1},
    feature_data::CuDeviceArray{Float32, 2},
    n_samples::Int32,
    n_features::Int32
)
    feature_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if feature_idx <= n_features
        sum = Float32(0)
        
        # Compute sum for this feature
        for sample_idx in 1:n_samples
            sum += feature_data[sample_idx, feature_idx]
        end
        
        # Store mean
        means[feature_idx] = sum / Float32(n_samples)
    end
    
    return nothing
end

"""
GPU kernel for computing feature standard deviations
"""
function compute_stds_kernel!(
    stds::CuDeviceArray{Float32, 1},
    feature_data::CuDeviceArray{Float32, 2},
    means::CuDeviceArray{Float32, 1},
    n_samples::Int32,
    n_features::Int32,
    epsilon::Float32
)
    feature_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if feature_idx <= n_features
        mean = means[feature_idx]
        sum_sq = Float32(0)
        
        # Compute sum of squared differences
        for sample_idx in 1:n_samples
            diff = feature_data[sample_idx, feature_idx] - mean
            sum_sq += diff * diff
        end
        
        # Compute standard deviation with epsilon for stability
        variance = sum_sq / Float32(n_samples - 1)
        stds[feature_idx] = sqrt(variance + epsilon)
    end
    
    return nothing
end

"""
GPU kernel for standardizing features (zero mean, unit variance)
"""
function standardize_features_kernel!(
    standardized_data::CuDeviceArray{Float32, 2},
    feature_data::CuDeviceArray{Float32, 2},
    means::CuDeviceArray{Float32, 1},
    stds::CuDeviceArray{Float32, 1},
    n_samples::Int32,
    n_features::Int32
)
    sample_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    feature_idx = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    
    if sample_idx <= n_samples && feature_idx <= n_features
        mean = means[feature_idx]
        std = stds[feature_idx]
        
        # Standardize: (x - mean) / std
        if std > Float32(0)
            standardized_data[sample_idx, feature_idx] = 
                (feature_data[sample_idx, feature_idx] - mean) / std
        else
            # Constant feature - set to zero
            standardized_data[sample_idx, feature_idx] = Float32(0)
        end
    end
    
    return nothing
end

"""
GPU kernel to extract upper triangle from correlation matrix
"""
function extract_upper_triangle_kernel!(
    upper_triangle::CuDeviceArray{Float32, 1},
    full_matrix::CuDeviceArray{Float32, 2},
    n_features::Int32
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    # Total number of elements in upper triangle
    n_elements = div(n_features * (n_features - 1), 2)
    
    if tid <= n_elements
        # Convert linear index to (i, j) pair
        # Using quadratic formula to find row index
        i = Int32(floor((-1 + sqrt(1 + 8 * tid)) / 2))
        j = tid - div(i * (i + 1), 2) + i + 1
        
        # Ensure valid indices
        if i < n_features && j < n_features && i < j
            # Get value from full matrix
            upper_triangle[tid] = full_matrix[i, j]
        end
    end
    
    return nothing
end

"""
Compute correlation matrix using cuBLAS
"""
function compute_correlation_cublas!(
    corr_matrix::CorrelationMatrix,
    feature_matrix::FeatureMatrix,
    config::CorrelationConfig
)
    n_features = config.n_features
    n_samples = config.n_samples
    
    # Step 1: Compute means
    threads = 256
    blocks = cld(n_features, threads)
    @cuda threads=threads blocks=blocks compute_means_kernel!(
        corr_matrix.feature_means,
        feature_matrix.data,
        n_samples,
        n_features
    )
    
    # Step 2: Compute standard deviations
    @cuda threads=threads blocks=blocks compute_stds_kernel!(
        corr_matrix.feature_stds,
        feature_matrix.data,
        corr_matrix.feature_means,
        n_samples,
        n_features,
        config.epsilon
    )
    
    # Step 3: Standardize features
    threads_x = 32
    threads_y = 8
    blocks_x = cld(n_samples, threads_x)
    blocks_y = cld(n_features, threads_y)
    @cuda threads=(threads_x, threads_y) blocks=(blocks_x, blocks_y) standardize_features_kernel!(
        corr_matrix.standardized_data,
        feature_matrix.data,
        corr_matrix.feature_means,
        corr_matrix.feature_stds,
        n_samples,
        n_features
    )
    
    CUDA.synchronize()
    
    # Step 4: Compute correlation matrix using cuBLAS
    # Correlation = (1/n) * X^T * X where X is standardized
    
    # Create temporary full matrix for cuBLAS computation
    full_corr_matrix = CUDA.zeros(Float32, n_features, n_features)
    
    # Use cuBLAS for matrix multiplication
    # C = alpha * A^T * A + beta * C
    # where A = standardized_data, C = full_corr_matrix
    alpha = Float32(1.0 / (n_samples - 1))
    beta = Float32(0.0)
    
    CUBLAS.gemm!('T', 'N', alpha, 
                 corr_matrix.standardized_data[1:n_samples, 1:n_features],
                 corr_matrix.standardized_data[1:n_samples, 1:n_features],
                 beta, full_corr_matrix)
    
    # Step 5: Extract upper triangle
    threads = 256
    n_upper_elements = div(n_features * (n_features - 1), 2)
    blocks = cld(n_upper_elements, threads)
    @cuda threads=threads blocks=blocks extract_upper_triangle_kernel!(
        corr_matrix.data,
        full_corr_matrix,
        n_features
    )
    
    CUDA.synchronize()
end

"""
GPU kernel for direct correlation computation (alternative to cuBLAS)
"""
function compute_correlation_direct_kernel!(
    correlations::CuDeviceArray{Float32, 1},  # Upper triangle only
    standardized_data::CuDeviceArray{Float32, 2},
    n_samples::Int32,
    n_features::Int32
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    # Total number of elements in upper triangle
    n_elements = div(n_features * (n_features - 1), 2)
    
    if tid <= n_elements
        # Convert linear index to (i, j) pair
        i = Int32(floor((-1 + sqrt(1 + 8 * tid)) / 2))
        j = tid - div(i * (i + 1), 2) + i + 1
        
        if i < n_features && j < n_features && i < j
            # Compute correlation between features i and j
            corr = Float32(0)
            
            for sample_idx in 1:n_samples
                corr += standardized_data[sample_idx, i] * standardized_data[sample_idx, j]
            end
            
            correlations[tid] = corr / Float32(n_samples - 1)
        end
    end
    
    return nothing
end

"""
Compute correlation matrix without cuBLAS (for comparison/debugging)
"""
function compute_correlation_direct!(
    corr_matrix::CorrelationMatrix,
    feature_matrix::FeatureMatrix,
    config::CorrelationConfig
)
    n_features = config.n_features
    n_samples = config.n_samples
    
    # Steps 1-3: Same as cuBLAS version (compute means, stds, standardize)
    threads = 256
    blocks = cld(n_features, threads)
    
    @cuda threads=threads blocks=blocks compute_means_kernel!(
        corr_matrix.feature_means,
        feature_matrix.data,
        n_samples,
        n_features
    )
    
    @cuda threads=threads blocks=blocks compute_stds_kernel!(
        corr_matrix.feature_stds,
        feature_matrix.data,
        corr_matrix.feature_means,
        n_samples,
        n_features,
        config.epsilon
    )
    
    threads_x = 32
    threads_y = 8
    blocks_x = cld(n_samples, threads_x)
    blocks_y = cld(n_features, threads_y)
    @cuda threads=(threads_x, threads_y) blocks=(blocks_x, blocks_y) standardize_features_kernel!(
        corr_matrix.standardized_data,
        feature_matrix.data,
        corr_matrix.feature_means,
        corr_matrix.feature_stds,
        n_samples,
        n_features
    )
    
    # Step 4: Direct correlation computation
    threads = 256
    n_upper_elements = div(n_features * (n_features - 1), 2)
    blocks = cld(n_upper_elements, threads)
    @cuda threads=threads blocks=blocks compute_correlation_direct_kernel!(
        corr_matrix.data,
        corr_matrix.standardized_data,
        n_samples,
        n_features
    )
    
    CUDA.synchronize()
end

"""
Get correlation value from upper triangle storage
"""
function get_correlation(corr_matrix::CorrelationMatrix, i::Integer, j::Integer)
    if i == j
        return Float32(1.0)  # Diagonal elements are always 1
    end
    
    # Ensure i < j for upper triangle
    if i > j
        i, j = j, i
    end
    
    idx = get_upper_triangle_index(Int32(i), Int32(j), corr_matrix.n_features)
    return CUDA.@allowscalar corr_matrix.data[idx + 1]  # Julia is 1-indexed
end

"""
Find highly correlated feature pairs
"""
function find_correlated_pairs(
    corr_matrix::CorrelationMatrix,
    threshold::Float32 = Float32(0.95)
)
    n_features = corr_matrix.n_features
    n_elements = length(corr_matrix.data)
    
    # Copy to CPU for easier processing
    corr_data_cpu = Array(corr_matrix.data)
    
    correlated_pairs = Vector{Tuple{Int32, Int32, Float32}}()
    
    for idx in 1:n_elements
        if abs(corr_data_cpu[idx]) >= threshold
            # Convert linear index back to (i, j)
            i = Int32(floor((-1 + sqrt(1 + 8 * idx)) / 2))
            j = idx - div(i * (i + 1), 2) + i + 1
            
            push!(correlated_pairs, (i, j, corr_data_cpu[idx]))
        end
    end
    
    return correlated_pairs
end

"""
Batch processing for large datasets
"""
function compute_correlation_batched!(
    corr_matrix::CorrelationMatrix,
    feature_matrix::FeatureMatrix,
    config::CorrelationConfig
)
    n_features = config.n_features
    n_samples = config.n_samples
    batch_size = config.batch_size
    
    if n_samples <= batch_size
        # Process in one batch
        if config.use_cublas
            compute_correlation_cublas!(corr_matrix, feature_matrix, config)
        else
            compute_correlation_direct!(corr_matrix, feature_matrix, config)
        end
    else
        # TODO: Implement streaming correlation computation
        # For now, use single batch computation
        @warn "Batch processing not yet implemented, processing all samples at once"
        if config.use_cublas
            compute_correlation_cublas!(corr_matrix, feature_matrix, config)
        else
            compute_correlation_direct!(corr_matrix, feature_matrix, config)
        end
    end
end

# Export functions
export CorrelationConfig, create_correlation_config
export compute_correlation_cublas!, compute_correlation_direct!, compute_correlation_batched!
export get_correlation, find_correlated_pairs

end # module CorrelationComputation