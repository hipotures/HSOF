module GPUMemoryLayout

using CUDA
using Printf

# Constants for memory alignment and optimization
const WARP_SIZE = Int32(32)
const ALIGNMENT_BYTES = Int32(128)  # 128-byte alignment for coalesced access
const TILE_SIZE = Int32(32)  # 32x32 tile size for shared memory
const MAX_FEATURES = Int32(5000)
const TARGET_FEATURES = Int32(500)
const HISTOGRAM_BINS = Int32(256)
const MAX_SAMPLES_PER_BATCH = Int32(100000)  # Process in 100K batches

"""
Feature data storage format
"""
@enum DataLayout begin
    COLUMN_MAJOR = 0  # Features as columns for coalesced access
    ROW_MAJOR = 1     # Samples as rows (not recommended for GPU)
end

"""
Feature data type
"""
@enum FeatureType begin
    NUMERIC = 0
    CATEGORICAL = 1
    BINARY = 2
end

"""
GPU memory allocation for feature matrix
Column-major format: features are stored contiguously
"""
struct FeatureMatrix
    # Main data storage
    data::CuArray{Float32, 2}           # [n_samples × n_features] in column-major
    n_samples::Int32
    n_features::Int32
    
    # Metadata
    feature_types::CuArray{Int32, 1}    # Feature type for each column
    feature_names::Vector{String}       # Keep on CPU
    
    # Padding info
    padded_samples::Int32               # Samples padded to multiple of WARP_SIZE
    padded_features::Int32              # Features padded for alignment
end

"""
Histogram storage for mutual information calculation
"""
struct HistogramBuffers
    # Feature histograms
    feature_hist::CuArray{Int32, 2}     # [HISTOGRAM_BINS × n_features]
    target_hist::CuArray{Int32, 1}      # [HISTOGRAM_BINS]
    joint_hist::CuArray{Int32, 3}       # [HISTOGRAM_BINS × HISTOGRAM_BINS × n_features]
    
    # Bin edges
    bin_edges::CuArray{Float32, 2}      # [HISTOGRAM_BINS+1 × n_features]
    target_bin_edges::CuArray{Float32, 1}  # [HISTOGRAM_BINS+1]
    
    # Probabilities (computed from histograms)
    feature_probs::CuArray{Float32, 2}  # [HISTOGRAM_BINS × n_features]
    target_probs::CuArray{Float32, 1}   # [HISTOGRAM_BINS]
    joint_probs::CuArray{Float32, 3}    # [HISTOGRAM_BINS × HISTOGRAM_BINS × n_features]
end

"""
Correlation matrix storage (symmetric, upper triangle only)
"""
struct CorrelationMatrix
    # Upper triangle storage
    data::CuArray{Float32, 1}           # Packed upper triangle
    n_features::Int32
    
    # Statistics for standardization
    feature_means::CuArray{Float32, 1}  # [n_features]
    feature_stds::CuArray{Float32, 1}   # [n_features]
    
    # Temporary buffers for cuBLAS
    standardized_data::CuArray{Float32, 2}  # [n_samples × n_features]
end

"""
Variance calculation buffers
"""
struct VarianceBuffers
    variances::CuArray{Float32, 1}      # [n_features]
    
    # Welford's algorithm intermediates
    means::CuArray{Float32, 1}          # [n_features]
    m2_values::CuArray{Float32, 1}      # Sum of squared differences
    counts::CuArray{Int32, 1}           # Sample counts per feature
    
    # Shared memory workspace per block
    block_sums::CuArray{Float32, 2}     # [n_blocks × n_features]
    block_counts::CuArray{Int32, 2}     # [n_blocks × n_features]
end

"""
Threshold configuration stored in constant memory
"""
struct ThresholdConfig
    mi_threshold::Float32               # Mutual information threshold (default 0.01)
    correlation_threshold::Float32      # Correlation threshold (default 0.95)
    variance_threshold::Float32         # Variance threshold (default 1e-6)
    target_features::Int32              # Target number of features (500)
end

"""
Feature ranking and selection buffers
"""
struct RankingBuffers
    # Scores and indices
    mi_scores::CuArray{Float32, 1}      # [n_features]
    feature_indices::CuArray{Int32, 1}  # [n_features] - for sorting
    selected_mask::CuArray{Bool, 1}     # [n_features] - selection flags
    
    # Correlation graph for redundancy removal
    correlation_pairs::CuArray{Int32, 2} # [2 × max_pairs] - highly correlated pairs
    n_correlation_pairs::CuArray{Int32, 1}  # Number of correlation pairs
    
    # Final selected features
    selected_indices::CuArray{Int32, 1}  # [TARGET_FEATURES]
    n_selected::CuArray{Int32, 1}       # Actual number selected
end

"""
Shared memory configuration for kernels
"""
struct SharedMemoryConfig
    # Tile dimensions
    tile_rows::Int32
    tile_cols::Int32
    
    # Histogram bins in shared memory
    shared_bins::Int32
    
    # Reduction workspace
    reduction_size::Int32
    
    # Total shared memory required
    total_bytes::Int32
end

"""
Calculate padding for memory alignment
"""
function calculate_padding(size::Int32, alignment::Int32)
    remainder = size % alignment
    return remainder == 0 ? size : size + (alignment - remainder)
end

"""
Create feature matrix with optimal memory layout
"""
function create_feature_matrix(n_samples::Integer, n_features::Integer, 
                             feature_types::Vector{FeatureType} = fill(NUMERIC, n_features))
    # Convert to Int32
    n_samples = Int32(n_samples)
    n_features = Int32(n_features)
    
    # Calculate padded dimensions
    padded_samples = calculate_padding(n_samples, WARP_SIZE)
    padded_features = calculate_padding(n_features, Int32(4))  # Align to float4
    
    # Allocate GPU memory
    data = CUDA.zeros(Float32, padded_samples, padded_features)
    feature_types_gpu = CuArray(Int32.(feature_types))
    feature_names = ["feature_$i" for i in 1:n_features]
    
    return FeatureMatrix(
        data,
        n_samples,
        n_features,
        feature_types_gpu,
        feature_names,
        padded_samples,
        padded_features
    )
end

"""
Create histogram buffers for MI calculation
"""
function create_histogram_buffers(n_features::Int)
    feature_hist = CUDA.zeros(Int32, HISTOGRAM_BINS, n_features)
    target_hist = CUDA.zeros(Int32, HISTOGRAM_BINS)
    joint_hist = CUDA.zeros(Int32, HISTOGRAM_BINS, HISTOGRAM_BINS, n_features)
    
    bin_edges = CUDA.zeros(Float32, HISTOGRAM_BINS + 1, n_features)
    target_bin_edges = CUDA.zeros(Float32, HISTOGRAM_BINS + 1)
    
    feature_probs = CUDA.zeros(Float32, HISTOGRAM_BINS, n_features)
    target_probs = CUDA.zeros(Float32, HISTOGRAM_BINS)
    joint_probs = CUDA.zeros(Float32, HISTOGRAM_BINS, HISTOGRAM_BINS, n_features)
    
    return HistogramBuffers(
        feature_hist, target_hist, joint_hist,
        bin_edges, target_bin_edges,
        feature_probs, target_probs, joint_probs
    )
end

"""
Create correlation matrix storage
"""
function create_correlation_matrix(n_features::Int, n_samples::Int)
    # Upper triangle size: n*(n-1)/2
    upper_triangle_size = div(n_features * (n_features - 1), 2)
    data = CUDA.zeros(Float32, upper_triangle_size)
    
    feature_means = CUDA.zeros(Float32, n_features)
    feature_stds = CUDA.ones(Float32, n_features)
    
    # Padded dimensions for cuBLAS efficiency
    padded_samples = calculate_padding(Int32(n_samples), Int32(16))
    padded_features = calculate_padding(Int32(n_features), Int32(4))
    standardized_data = CUDA.zeros(Float32, padded_samples, padded_features)
    
    return CorrelationMatrix(
        data,
        Int32(n_features),
        feature_means,
        feature_stds,
        standardized_data
    )
end

"""
Create variance calculation buffers
"""
function create_variance_buffers(n_features::Int, n_blocks::Int = 256)
    variances = CUDA.zeros(Float32, n_features)
    means = CUDA.zeros(Float32, n_features)
    m2_values = CUDA.zeros(Float32, n_features)
    counts = CUDA.zeros(Int32, n_features)
    
    block_sums = CUDA.zeros(Float32, n_blocks, n_features)
    block_counts = CUDA.zeros(Int32, n_blocks, n_features)
    
    return VarianceBuffers(
        variances, means, m2_values, counts,
        block_sums, block_counts
    )
end

"""
Create ranking buffers for feature selection
"""
function create_ranking_buffers(n_features::Int)
    mi_scores = CUDA.zeros(Float32, n_features)
    feature_indices = CuArray(collect(Int32, 1:n_features))
    selected_mask = CUDA.zeros(Bool, n_features)
    
    # Maximum possible correlation pairs
    max_pairs = div(n_features * (n_features - 1), 2)
    correlation_pairs = CUDA.zeros(Int32, 2, max_pairs)
    n_correlation_pairs = CUDA.zeros(Int32, 1)
    
    selected_indices = CUDA.zeros(Int32, TARGET_FEATURES)
    n_selected = CUDA.zeros(Int32, 1)
    
    return RankingBuffers(
        mi_scores, feature_indices, selected_mask,
        correlation_pairs, n_correlation_pairs,
        selected_indices, n_selected
    )
end

"""
Calculate shared memory requirements for kernels
"""
function calculate_shared_memory_config(tile_size::Int32 = TILE_SIZE)
    tile_rows = tile_size
    tile_cols = tile_size
    
    # Histogram bins that fit in shared memory
    shared_bins = min(HISTOGRAM_BINS, Int32(64))  # Partial histogram in shared
    
    # Reduction workspace (for variance/mean calculations)
    reduction_size = tile_size * 2  # Double buffer for reduction
    
    # Calculate total bytes
    tile_bytes = tile_rows * tile_cols * sizeof(Float32)
    hist_bytes = shared_bins * sizeof(Int32)
    reduction_bytes = reduction_size * sizeof(Float32)
    
    total_bytes = tile_bytes + hist_bytes + reduction_bytes
    
    return SharedMemoryConfig(
        tile_rows, tile_cols,
        shared_bins, reduction_size,
        total_bytes
    )
end

"""
Get upper triangle index for correlation matrix
"""
function get_upper_triangle_index(i::Int32, j::Int32, n::Int32)
    # Ensure i < j
    if i > j
        i, j = j, i
    end
    # Calculate index in packed upper triangle
    # Index = i*n - i*(i+1)/2 + j - i - 1
    return i * n - div(i * (i + 1), 2) + j - i - 1
end

"""
Memory usage summary
"""
function print_memory_usage(n_samples::Int, n_features::Int)
    println("GPU Memory Usage Summary")
    println("========================")
    println("Dataset: $n_samples samples × $n_features features")
    println()
    
    # Calculate theoretical memory usage without actual allocation
    try
        # Feature matrix bytes
        padded_samples = calculate_padding(Int32(n_samples), WARP_SIZE)
        padded_features = calculate_padding(Int32(n_features), Int32(4))
        fm_bytes = padded_samples * padded_features * sizeof(Float32) + n_features * sizeof(Int32)
        
        # Histogram buffers bytes
        hb_bytes = HISTOGRAM_BINS * n_features * sizeof(Int32) +  # feature_hist
                   HISTOGRAM_BINS * sizeof(Int32) +  # target_hist
                   HISTOGRAM_BINS * HISTOGRAM_BINS * n_features * sizeof(Int32) +  # joint_hist
                   (HISTOGRAM_BINS + 1) * n_features * sizeof(Float32) +  # bin_edges
                   (HISTOGRAM_BINS + 1) * sizeof(Float32) +  # target_bin_edges
                   HISTOGRAM_BINS * n_features * sizeof(Float32) +  # feature_probs
                   HISTOGRAM_BINS * sizeof(Float32) +  # target_probs
                   HISTOGRAM_BINS * HISTOGRAM_BINS * n_features * sizeof(Float32)  # joint_probs
        
        # Correlation matrix bytes
        upper_triangle_size = div(n_features * (n_features - 1), 2)
        padded_samples_corr = calculate_padding(Int32(n_samples), Int32(16))
        padded_features_corr = calculate_padding(Int32(n_features), Int32(4))
        cm_bytes = upper_triangle_size * sizeof(Float32) +  # data
                   n_features * sizeof(Float32) +  # feature_means
                   n_features * sizeof(Float32) +  # feature_stds
                   padded_samples_corr * padded_features_corr * sizeof(Float32)  # standardized_data
        
        # Variance buffers bytes
        n_blocks = 256
        vb_bytes = n_features * sizeof(Float32) +  # variances
                   n_features * sizeof(Float32) +  # means
                   n_features * sizeof(Float32) +  # m2_values
                   n_features * sizeof(Int32) +    # counts
                   n_blocks * n_features * sizeof(Float32) +  # block_sums
                   n_blocks * n_features * sizeof(Int32)      # block_counts
        
        # Ranking buffers bytes
        max_pairs = div(n_features * (n_features - 1), 2)
        rb_bytes = n_features * sizeof(Float32) +  # mi_scores
                   n_features * sizeof(Int32) +    # feature_indices
                   n_features * sizeof(Bool) +     # selected_mask
                   2 * max_pairs * sizeof(Int32) + # correlation_pairs
                   sizeof(Int32) +                 # n_correlation_pairs
                   TARGET_FEATURES * sizeof(Int32) + # selected_indices
                   sizeof(Int32)                   # n_selected
        
        total_bytes = fm_bytes + hb_bytes + cm_bytes + vb_bytes + rb_bytes
        total_gb = total_bytes / (1024^3)
        
        @printf("Feature Matrix:      %.2f MB\n", fm_bytes / (1024^2))
        @printf("Histogram Buffers:   %.2f MB\n", hb_bytes / (1024^2))
        @printf("Correlation Matrix:  %.2f MB\n", cm_bytes / (1024^2))
        @printf("Variance Buffers:    %.2f MB\n", vb_bytes / (1024^2))
        @printf("Ranking Buffers:     %.2f MB\n", rb_bytes / (1024^2))
        println("------------------------")
        @printf("Total GPU Memory:    %.2f GB\n", total_gb)
        
        # Check against typical GPU memory
        if total_gb > 24.0
            println("\n⚠️  Warning: Memory usage exceeds typical 24GB GPU memory!")
            println("   Consider processing in smaller batches.")
        elseif total_gb > 8.0
            println("\n⚠️  Warning: Memory usage exceeds 8GB budget for RTX 4090!")
        else
            println("\n✓ Memory usage within 8GB budget")
        end
    catch e
        println("Error calculating memory usage: ", e)
    end
end

# Export types and functions
export DataLayout, FeatureType, FeatureMatrix, HistogramBuffers
export CorrelationMatrix, VarianceBuffers, ThresholdConfig, RankingBuffers
export SharedMemoryConfig
export COLUMN_MAJOR, ROW_MAJOR, NUMERIC, CATEGORICAL, BINARY
export WARP_SIZE, ALIGNMENT_BYTES, TILE_SIZE, MAX_FEATURES, TARGET_FEATURES
export HISTOGRAM_BINS, MAX_SAMPLES_PER_BATCH
export calculate_padding, create_feature_matrix, create_histogram_buffers
export create_correlation_matrix, create_variance_buffers, create_ranking_buffers
export calculate_shared_memory_config, get_upper_triangle_index, print_memory_usage

end # module GPUMemoryLayout