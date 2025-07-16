module MutualInformation

using CUDA
using Statistics

# Include the GPUMemoryLayout module here
include("gpu_memory_layout.jl")
using .GPUMemoryLayout: HistogramBuffers, HISTOGRAM_BINS, WARP_SIZE, MAX_FEATURES

"""
MI calculation configuration
"""
struct MIConfig
    n_features::Int32
    n_samples::Int32
    n_bins::Int32               # Number of histogram bins (256)
    min_samples_per_bin::Int32  # Minimum samples for valid bin
    epsilon::Float32            # Small value to avoid log(0)
    use_shared_memory::Bool     # Use shared memory optimization
end

"""
Create default MI configuration
"""
function create_mi_config(n_features::Integer, n_samples::Integer;
                         n_bins::Integer = HISTOGRAM_BINS,
                         min_samples_per_bin::Integer = 3,
                         epsilon::Float32 = Float32(1e-10),
                         use_shared_memory::Bool = true)
    return MIConfig(
        Int32(n_features),
        Int32(n_samples),
        Int32(n_bins),
        Int32(min_samples_per_bin),
        epsilon,
        use_shared_memory
    )
end

"""
GPU kernel for computing histogram bin edges based on feature ranges
"""
function compute_bin_edges_kernel!(
    bin_edges::CuDeviceArray{Float32, 2},      # [n_bins+1 × n_features]
    feature_data::CuDeviceArray{Float32, 2},   # [n_samples × n_features]
    feature_mins::CuDeviceArray{Float32, 1},   # [n_features]
    feature_maxs::CuDeviceArray{Float32, 1},   # [n_features]
    n_samples::Int32,
    n_features::Int32,
    n_bins::Int32
)
    feature_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if feature_idx <= n_features
        # Find min and max for this feature
        min_val = Inf32
        max_val = -Inf32
        
        for i in 1:n_samples
            val = feature_data[i, feature_idx]
            min_val = min(min_val, val)
            max_val = max(max_val, val)
        end
        
        # Store min/max
        feature_mins[feature_idx] = min_val
        feature_maxs[feature_idx] = max_val
        
        # Compute bin edges with small margin to ensure all values fit
        range = max_val - min_val
        margin = range * 0.001f0  # 0.1% margin
        
        for bin in 0:n_bins
            t = Float32(bin) / Float32(n_bins)
            bin_edges[bin + 1, feature_idx] = min_val - margin + t * (range + 2 * margin)
        end
    end
    
    return nothing
end

"""
GPU kernel for histogram computation using atomic operations
"""
function compute_histogram_kernel!(
    histogram::CuDeviceArray{Int32, 2},        # [n_bins × n_features]
    feature_data::CuDeviceArray{Float32, 2},   # [n_samples × n_features]
    bin_edges::CuDeviceArray{Float32, 2},      # [n_bins+1 × n_features]
    n_samples::Int32,
    n_features::Int32,
    n_bins::Int32
)
    tid = threadIdx().x
    bid = blockIdx().x
    
    # Each block processes one feature
    if bid <= n_features
        feature_idx = bid
        
        # Process samples in grid-stride loop
        sample_idx = tid
        while sample_idx <= n_samples
            value = feature_data[sample_idx, feature_idx]
            
            # Binary search for bin
            bin_idx = Int32(1)
            for b in 1:n_bins
                if value <= bin_edges[b + 1, feature_idx]
                    bin_idx = b
                    break
                end
            end
            
            # Clamp to valid range (1-indexed)
            bin_idx = min(max(bin_idx, Int32(1)), n_bins)
            
            # Atomic increment - convert to linear index
            linear_idx = (feature_idx - 1) * n_bins + bin_idx
            @inbounds CUDA.atomic_add!(pointer(histogram, linear_idx), Int32(1))
            
            sample_idx += blockDim().x
        end
    end
    
    return nothing
end

"""
GPU kernel for joint histogram computation
"""
function compute_joint_histogram_kernel!(
    joint_hist::CuDeviceArray{Int32, 3},       # [n_bins × n_bins × n_features]
    feature_data::CuDeviceArray{Float32, 2},   # [n_samples × n_features]
    target_data::CuDeviceArray{Float32, 1},    # [n_samples]
    feature_bin_edges::CuDeviceArray{Float32, 2},  # [n_bins+1 × n_features]
    target_bin_edges::CuDeviceArray{Float32, 1},   # [n_bins+1]
    n_samples::Int32,
    n_features::Int32,
    n_bins::Int32
)
    tid = threadIdx().x
    bid = blockIdx().x
    
    # Each block processes one feature
    if bid <= n_features
        feature_idx = bid
        
        # Process samples in grid-stride loop
        sample_idx = tid
        while sample_idx <= n_samples
            feature_val = feature_data[sample_idx, feature_idx]
            target_val = target_data[sample_idx]
            
            # Find feature bin
            feature_bin = Int32(1)
            for b in 1:n_bins
                if feature_val <= feature_bin_edges[b + 1, feature_idx]
                    feature_bin = b
                    break
                end
            end
            
            # Find target bin
            target_bin = Int32(1)
            for b in 1:n_bins
                if target_val <= target_bin_edges[b + 1]
                    target_bin = b
                    break
                end
            end
            
            # Clamp to valid range
            feature_bin = min(max(feature_bin, Int32(1)), n_bins)
            target_bin = min(max(target_bin, Int32(1)), n_bins)
            
            # Atomic increment joint histogram - convert to linear index
            linear_idx = (feature_idx - 1) * n_bins * n_bins + (target_bin - 1) * n_bins + feature_bin
            @inbounds CUDA.atomic_add!(pointer(joint_hist, linear_idx), Int32(1))
            
            sample_idx += blockDim().x
        end
    end
    
    return nothing
end

"""
GPU kernel to convert histograms to probabilities
"""
function histogram_to_probability_kernel!(
    probabilities::CuDeviceArray{Float32, 2},  # [n_bins × n_features]
    histogram::CuDeviceArray{Int32, 2},        # [n_bins × n_features]
    n_samples::Int32,
    n_features::Int32,
    n_bins::Int32,
    epsilon::Float32
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= n_bins * n_features
        bin_idx = (tid - 1) % n_bins + 1
        feature_idx = (tid - 1) ÷ n_bins + 1
        
        if feature_idx <= n_features
            count = Float32(histogram[bin_idx, feature_idx])
            probabilities[bin_idx, feature_idx] = (count + epsilon) / (Float32(n_samples) + epsilon * n_bins)
        end
    end
    
    return nothing
end

"""
GPU kernel to compute entropy from probabilities
"""
function compute_entropy_kernel!(
    entropy::CuDeviceArray{Float32, 1},        # [n_features]
    probabilities::CuDeviceArray{Float32, 2},  # [n_bins × n_features]
    n_features::Int32,
    n_bins::Int32
)
    feature_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if feature_idx <= n_features
        H = Float32(0)
        
        for bin in 1:n_bins
            p = probabilities[bin, feature_idx]
            if p > 0
                H -= p * log2(p)
            end
        end
        
        entropy[feature_idx] = H
    end
    
    return nothing
end

"""
GPU kernel to compute mutual information scores
"""
function compute_mi_scores_kernel!(
    mi_scores::CuDeviceArray{Float32, 1},      # [n_features]
    feature_entropy::CuDeviceArray{Float32, 1}, # [n_features]
    target_entropy::Float32,
    joint_entropy::CuDeviceArray{Float32, 1},  # [n_features]
    n_features::Int32
)
    feature_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if feature_idx <= n_features
        # MI(X,Y) = H(X) + H(Y) - H(X,Y)
        mi_scores[feature_idx] = feature_entropy[feature_idx] + target_entropy - joint_entropy[feature_idx]
    end
    
    return nothing
end

"""
GPU kernel to compute joint entropy from joint probabilities
"""
function compute_joint_entropy_kernel!(
    joint_entropy::CuDeviceArray{Float32, 1},  # [n_features]
    joint_probs::CuDeviceArray{Float32, 3},    # [n_bins × n_bins × n_features]
    n_features::Int32,
    n_bins::Int32
)
    feature_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if feature_idx <= n_features
        H_joint = Float32(0)
        
        for target_bin in 1:n_bins
            for feature_bin in 1:n_bins
                p = joint_probs[feature_bin, target_bin, feature_idx]
                if p > 0
                    H_joint -= p * log2(p)
                end
            end
        end
        
        joint_entropy[feature_idx] = H_joint
    end
    
    return nothing
end

"""
Compute mutual information for all features
"""
function compute_mutual_information!(
    mi_scores::CuArray{Float32, 1},
    feature_data::CuArray{Float32, 2},
    target_data::CuArray{Float32, 1},
    histogram_buffers::HistogramBuffers,
    config::MIConfig
)
    n_features = config.n_features
    n_samples = config.n_samples
    n_bins = config.n_bins
    
    # Clear histograms
    fill!(histogram_buffers.feature_hist, Int32(0))
    fill!(histogram_buffers.target_hist, Int32(0))
    fill!(histogram_buffers.joint_hist, Int32(0))
    
    # Temporary arrays for min/max
    feature_mins = CUDA.zeros(Float32, n_features)
    feature_maxs = CUDA.zeros(Float32, n_features)
    target_min = CUDA.zeros(Float32, 1)
    target_max = CUDA.zeros(Float32, 1)
    
    # Step 1: Compute bin edges
    threads = 256
    blocks = cld(n_features, threads)
    @cuda threads=threads blocks=blocks compute_bin_edges_kernel!(
        histogram_buffers.bin_edges,
        feature_data,
        feature_mins,
        feature_maxs,
        n_samples,
        n_features,
        n_bins
    )
    
    # Compute target bin edges (on CPU for simplicity)
    target_cpu = Array(target_data)
    target_min_val = minimum(target_cpu)
    target_max_val = maximum(target_cpu)
    target_range = target_max_val - target_min_val
    target_margin = target_range * 0.001f0
    
    target_edges_cpu = zeros(Float32, n_bins + 1)
    for i in 0:n_bins
        t = Float32(i) / Float32(n_bins)
        target_edges_cpu[i + 1] = target_min_val - target_margin + t * (target_range + 2 * target_margin)
    end
    copyto!(histogram_buffers.target_bin_edges, target_edges_cpu)
    
    # Step 2: Compute feature histograms
    threads = 256
    blocks = n_features  # One block per feature
    @cuda threads=threads blocks=blocks compute_histogram_kernel!(
        histogram_buffers.feature_hist,
        feature_data,
        histogram_buffers.bin_edges,
        n_samples,
        n_features,
        n_bins
    )
    
    # Step 3: Compute target histogram
    # Simple CPU version for target (could optimize)
    target_hist_cpu = zeros(Int32, n_bins)
    for i in 1:n_samples
        val = target_cpu[i]
        bin_idx = 1
        for b in 1:n_bins
            if val <= target_edges_cpu[b + 1]
                bin_idx = b
                break
            end
        end
        bin_idx = min(max(bin_idx, 1), n_bins)
        target_hist_cpu[bin_idx] += 1
    end
    copyto!(histogram_buffers.target_hist, target_hist_cpu)
    
    # Step 4: Compute joint histograms
    @cuda threads=threads blocks=blocks compute_joint_histogram_kernel!(
        histogram_buffers.joint_hist,
        feature_data,
        target_data,
        histogram_buffers.bin_edges,
        histogram_buffers.target_bin_edges,
        n_samples,
        n_features,
        n_bins
    )
    
    CUDA.synchronize()
    
    # Step 5: Convert histograms to probabilities
    threads = 256
    blocks = cld(n_bins * n_features, threads)
    @cuda threads=threads blocks=blocks histogram_to_probability_kernel!(
        histogram_buffers.feature_probs,
        histogram_buffers.feature_hist,
        n_samples,
        n_features,
        n_bins,
        config.epsilon
    )
    
    # Convert target histogram to probabilities
    target_probs_cpu = (target_hist_cpu .+ config.epsilon) ./ (n_samples + config.epsilon * n_bins)
    copyto!(histogram_buffers.target_probs, Float32.(target_probs_cpu))
    
    # Convert joint histograms to probabilities
    threads = 256
    total_elements = n_bins * n_bins * n_features
    blocks = cld(total_elements, threads)
    
    # Flatten and convert joint histogram
    function joint_histogram_to_probability_kernel!(
        joint_probs::CuDeviceArray{Float32, 3},
        joint_hist::CuDeviceArray{Int32, 3},
        n_samples::Int32,
        n_bins::Int32,
        n_features::Int32,
        epsilon::Float32
    )
        tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        
        if tid <= n_bins * n_bins * n_features
            idx = tid - 1
            feature_idx = idx ÷ (n_bins * n_bins) + 1
            remainder = idx % (n_bins * n_bins)
            target_bin = remainder ÷ n_bins + 1
            feature_bin = remainder % n_bins + 1
            
            if feature_idx <= n_features
                count = Float32(joint_hist[feature_bin, target_bin, feature_idx])
                joint_probs[feature_bin, target_bin, feature_idx] = (count + epsilon) / (Float32(n_samples) + epsilon * n_bins * n_bins)
            end
        end
        
        return nothing
    end
    
    @cuda threads=threads blocks=blocks joint_histogram_to_probability_kernel!(
        histogram_buffers.joint_probs,
        histogram_buffers.joint_hist,
        n_samples,
        n_bins,
        n_features,
        config.epsilon
    )
    
    # Step 6: Compute entropies
    feature_entropy = CUDA.zeros(Float32, n_features)
    threads = 256
    blocks = cld(n_features, threads)
    @cuda threads=threads blocks=blocks compute_entropy_kernel!(
        feature_entropy,
        histogram_buffers.feature_probs,
        n_features,
        n_bins
    )
    
    # Compute target entropy
    target_entropy = Float32(0)
    for i in 1:n_bins
        p = target_probs_cpu[i]
        if p > 0
            target_entropy -= p * log2(p)
        end
    end
    
    # Compute joint entropies
    joint_entropy = CUDA.zeros(Float32, n_features)
    @cuda threads=threads blocks=blocks compute_joint_entropy_kernel!(
        joint_entropy,
        histogram_buffers.joint_probs,
        n_features,
        n_bins
    )
    
    # Step 7: Compute MI scores
    @cuda threads=threads blocks=blocks compute_mi_scores_kernel!(
        mi_scores,
        feature_entropy,
        target_entropy,
        joint_entropy,
        n_features
    )
    
    CUDA.synchronize()
end

"""
Compute MI scores with optimized shared memory version
"""
function compute_mutual_information_shared!(
    mi_scores::CuArray{Float32, 1},
    feature_data::CuArray{Float32, 2},
    target_data::CuArray{Float32, 1},
    histogram_buffers::HistogramBuffers,
    config::MIConfig
)
    # For now, use the standard version
    # Shared memory optimization would require more complex kernel design
    compute_mutual_information!(mi_scores, feature_data, target_data, histogram_buffers, config)
end

# Export types and functions
export MIConfig, create_mi_config
export compute_mutual_information!, compute_mutual_information_shared!

end # module MutualInformation