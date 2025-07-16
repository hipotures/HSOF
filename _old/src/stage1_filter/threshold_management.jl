module ThresholdManagement

using CUDA
using Statistics
using JSON3

# Import necessary types from GPUMemoryLayout
include("gpu_memory_layout.jl")
using .GPUMemoryLayout: ThresholdConfig, MAX_FEATURES, TARGET_FEATURES

"""
Extended threshold configuration with validation and adaptation
"""
mutable struct ExtendedThresholdConfig
    base_config::ThresholdConfig
    
    # Adaptive thresholds
    adaptive_mi_threshold::Float32
    adaptive_corr_threshold::Float32
    adaptive_var_threshold::Float32
    
    # Threshold ranges
    mi_threshold_range::Tuple{Float32, Float32}
    corr_threshold_range::Tuple{Float32, Float32}
    var_threshold_range::Tuple{Float32, Float32}
    
    # Adaptation parameters
    adaptation_rate::Float32
    min_features_buffer::Int32  # Extra features to select before filtering
    max_iterations::Int32       # Max iterations for adaptive adjustment
    
    # Validation flags
    validate_feature_count::Bool
    strict_mode::Bool          # Fail if exact count not achieved
end

"""
Threshold adjustment statistics
"""
struct ThresholdStats
    mi_values::CuArray{Float32, 1}
    correlations::CuArray{Float32, 1}
    variances::CuArray{Float32, 1}
    
    # Summary statistics
    mi_percentiles::Vector{Float32}
    corr_percentiles::Vector{Float32}
    var_percentiles::Vector{Float32}
end

"""
Create default extended threshold configuration
"""
function create_default_config()
    base = ThresholdConfig(
        Float32(0.01),    # MI threshold
        Float32(0.95),    # Correlation threshold
        Float32(1e-6),    # Variance threshold
        Int32(TARGET_FEATURES)
    )
    
    return ExtendedThresholdConfig(
        base,
        base.mi_threshold,      # Start with base values
        base.correlation_threshold,
        base.variance_threshold,
        (Float32(0.0), Float32(1.0)),     # MI range
        (Float32(0.5), Float32(0.999)),   # Correlation range
        (Float32(1e-10), Float32(0.1)),   # Variance range
        Float32(0.1),          # Adaptation rate
        Int32(50),             # Buffer of 50 extra features
        Int32(20),             # Max 20 iterations
        true,                  # Validate feature count
        false                  # Not strict by default
    )
end

"""
Create configuration from JSON file
"""
function load_config(filepath::String)
    if !isfile(filepath)
        @warn "Configuration file not found, using defaults" filepath
        return create_default_config()
    end
    
    json_data = JSON3.read(read(filepath, String))
    
    base = ThresholdConfig(
        Float32(get(json_data, "mi_threshold", 0.01)),
        Float32(get(json_data, "correlation_threshold", 0.95)),
        Float32(get(json_data, "variance_threshold", 1e-6)),
        Int32(get(json_data, "target_features", TARGET_FEATURES))
    )
    
    config = ExtendedThresholdConfig(
        base,
        base.mi_threshold,
        base.correlation_threshold,
        base.variance_threshold,
        (Float32(get(json_data, "mi_min", 0.0)), 
         Float32(get(json_data, "mi_max", 1.0))),
        (Float32(get(json_data, "corr_min", 0.5)), 
         Float32(get(json_data, "corr_max", 0.999))),
        (Float32(get(json_data, "var_min", 1e-10)), 
         Float32(get(json_data, "var_max", 0.1))),
        Float32(get(json_data, "adaptation_rate", 0.1)),
        Int32(get(json_data, "min_features_buffer", 50)),
        Int32(get(json_data, "max_iterations", 20)),
        get(json_data, "validate_feature_count", true),
        get(json_data, "strict_mode", false)
    )
    
    return config
end

"""
Save configuration to JSON file
"""
function save_config(config::ExtendedThresholdConfig, filepath::String)
    data = Dict(
        "mi_threshold" => config.base_config.mi_threshold,
        "correlation_threshold" => config.base_config.correlation_threshold,
        "variance_threshold" => config.base_config.variance_threshold,
        "target_features" => config.base_config.target_features,
        "mi_min" => config.mi_threshold_range[1],
        "mi_max" => config.mi_threshold_range[2],
        "corr_min" => config.corr_threshold_range[1],
        "corr_max" => config.corr_threshold_range[2],
        "var_min" => config.var_threshold_range[1],
        "var_max" => config.var_threshold_range[2],
        "adaptation_rate" => config.adaptation_rate,
        "min_features_buffer" => config.min_features_buffer,
        "max_iterations" => config.max_iterations,
        "validate_feature_count" => config.validate_feature_count,
        "strict_mode" => config.strict_mode
    )
    
    open(filepath, "w") do io
        JSON3.pretty(io, data)
    end
end

"""
Validate threshold configuration
"""
function validate_config!(config::ExtendedThresholdConfig)
    # Ensure thresholds are within ranges
    config.adaptive_mi_threshold = clamp(
        config.adaptive_mi_threshold,
        config.mi_threshold_range[1],
        config.mi_threshold_range[2]
    )
    
    config.adaptive_corr_threshold = clamp(
        config.adaptive_corr_threshold,
        config.corr_threshold_range[1],
        config.corr_threshold_range[2]
    )
    
    config.adaptive_var_threshold = clamp(
        config.adaptive_var_threshold,
        config.var_threshold_range[1],
        config.var_threshold_range[2]
    )
    
    # Validate target features
    if config.base_config.target_features > MAX_FEATURES
        error("Target features $(config.base_config.target_features) exceeds maximum $(MAX_FEATURES)")
    end
    
    if config.base_config.target_features < 1
        error("Target features must be at least 1")
    end
    
    # Validate adaptation parameters
    if config.adaptation_rate <= 0 || config.adaptation_rate > 1
        error("Adaptation rate must be in (0, 1]")
    end
    
    if config.max_iterations < 1
        error("Max iterations must be at least 1")
    end
    
    return true
end

"""
Calculate threshold statistics from data
"""
function calculate_threshold_stats(
    mi_scores::CuArray{Float32, 1},
    correlations::CuArray{Float32, 1},
    variances::CuArray{Float32, 1}
)
    # Calculate percentiles on GPU
    percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    # Copy to CPU for percentile calculation
    mi_cpu = Array(mi_scores)
    corr_cpu = Array(correlations)
    var_cpu = Array(variances)
    
    mi_percentiles = Float32.(quantile(mi_cpu, percentiles))
    corr_percentiles = Float32.(quantile(corr_cpu, percentiles))
    var_percentiles = Float32.(quantile(var_cpu, percentiles))
    
    return ThresholdStats(
        mi_scores, correlations, variances,
        mi_percentiles, corr_percentiles, var_percentiles
    )
end

"""
Adjust thresholds based on feature count
"""
function adjust_thresholds_for_count!(
    config::ExtendedThresholdConfig,
    current_count::Int32,
    stats::ThresholdStats,
    iteration::Int32 = Int32(1)
)
    if iteration > config.max_iterations
        if config.strict_mode
            error("Failed to achieve target feature count after $(config.max_iterations) iterations")
        else
            @warn "Max iterations reached, using current thresholds" current_count
            return false
        end
    end
    
    target = config.base_config.target_features
    
    # If we have the exact count, we're done
    if current_count == target
        return true
    end
    
    # Calculate adjustment factor
    ratio = Float32(target) / Float32(max(1, current_count))
    adjustment = config.adaptation_rate * (ratio - 1.0f0)
    
    if current_count < target
        # Too few features - relax thresholds
        # Lower MI threshold to include more features
        percentile_idx = min(9, max(1, round(Int, 9 * (1.0 - ratio))))
        config.adaptive_mi_threshold = stats.mi_percentiles[percentile_idx]
        
        # Increase correlation threshold to keep more correlated features
        config.adaptive_corr_threshold = min(
            config.corr_threshold_range[2],
            config.adaptive_corr_threshold * (1.0f0 + abs(adjustment))
        )
        
        # Lower variance threshold to include lower variance features
        config.adaptive_var_threshold = max(
            config.var_threshold_range[1],
            config.adaptive_var_threshold * (1.0f0 - abs(adjustment))
        )
    else
        # Too many features - tighten thresholds
        # Increase MI threshold to be more selective
        percentile_idx = min(9, max(1, round(Int, 9 * ratio)))
        config.adaptive_mi_threshold = stats.mi_percentiles[percentile_idx]
        
        # Decrease correlation threshold to remove more correlated features
        config.adaptive_corr_threshold = max(
            config.corr_threshold_range[1],
            config.adaptive_corr_threshold * (1.0f0 - abs(adjustment))
        )
        
        # Increase variance threshold to remove low variance features
        config.adaptive_var_threshold = min(
            config.var_threshold_range[2],
            config.adaptive_var_threshold * (1.0f0 + abs(adjustment))
        )
    end
    
    # Validate adjusted thresholds
    validate_config!(config)
    
    return false  # Need another iteration
end

"""
GPU kernel to count features passing thresholds
"""
function count_passing_features_kernel!(
    count::CuDeviceArray{Int32, 1},
    mi_scores::CuDeviceArray{Float32, 1},
    variances::CuDeviceArray{Float32, 1},
    mi_threshold::Float32,
    var_threshold::Float32,
    n_features::Int32
)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= n_features
        if mi_scores[tid] >= mi_threshold && variances[tid] >= var_threshold
            CUDA.atomic_add!(pointer(count), Int32(1))
        end
    end
    
    return nothing
end

"""
Count features that would pass current thresholds
"""
function count_passing_features(
    mi_scores::CuArray{Float32, 1},
    variances::CuArray{Float32, 1},
    config::ExtendedThresholdConfig
)
    n_features = length(mi_scores)
    count = CUDA.zeros(Int32, 1)
    
    threads = 256
    blocks = cld(n_features, threads)
    
    @cuda threads=threads blocks=blocks count_passing_features_kernel!(
        count, mi_scores, variances,
        config.adaptive_mi_threshold,
        config.adaptive_var_threshold,
        Int32(n_features)
    )
    
    return CUDA.@allowscalar count[1]
end

"""
Update configuration at runtime
"""
function update_runtime_config!(
    config::ExtendedThresholdConfig;
    mi_threshold::Union{Float32, Nothing} = nothing,
    correlation_threshold::Union{Float32, Nothing} = nothing,
    variance_threshold::Union{Float32, Nothing} = nothing,
    target_features::Union{Int32, Nothing} = nothing,
    adaptation_rate::Union{Float32, Nothing} = nothing,
    strict_mode::Union{Bool, Nothing} = nothing
)
    # Update base configuration
    if !isnothing(mi_threshold)
        config.base_config = ThresholdConfig(
            mi_threshold,
            config.base_config.correlation_threshold,
            config.base_config.variance_threshold,
            config.base_config.target_features
        )
        config.adaptive_mi_threshold = mi_threshold
    end
    
    if !isnothing(correlation_threshold)
        config.base_config = ThresholdConfig(
            config.base_config.mi_threshold,
            correlation_threshold,
            config.base_config.variance_threshold,
            config.base_config.target_features
        )
        config.adaptive_corr_threshold = correlation_threshold
    end
    
    if !isnothing(variance_threshold)
        config.base_config = ThresholdConfig(
            config.base_config.mi_threshold,
            config.base_config.correlation_threshold,
            variance_threshold,
            config.base_config.target_features
        )
        config.adaptive_var_threshold = variance_threshold
    end
    
    if !isnothing(target_features)
        config.base_config = ThresholdConfig(
            config.base_config.mi_threshold,
            config.base_config.correlation_threshold,
            config.base_config.variance_threshold,
            target_features
        )
    end
    
    # Update adaptation parameters
    if !isnothing(adaptation_rate)
        config.adaptation_rate = adaptation_rate
    end
    
    if !isnothing(strict_mode)
        config.strict_mode = strict_mode
    end
    
    # Validate the updated configuration
    validate_config!(config)
end

"""
Create threshold configuration for constant memory
"""
function create_constant_memory_config(config::ExtendedThresholdConfig)
    # For CUDA constant memory, we need a simple struct
    # This would be defined in the kernel file
    return (
        config.adaptive_mi_threshold,
        config.adaptive_corr_threshold,
        config.adaptive_var_threshold,
        config.base_config.target_features
    )
end

"""
Log threshold configuration
"""
function log_config(config::ExtendedThresholdConfig, prefix::String = "")
    println(prefix * "Threshold Configuration:")
    println(prefix * "  MI Threshold: $(config.adaptive_mi_threshold) (base: $(config.base_config.mi_threshold))")
    println(prefix * "  Correlation Threshold: $(config.adaptive_corr_threshold) (base: $(config.base_config.correlation_threshold))")
    println(prefix * "  Variance Threshold: $(config.adaptive_var_threshold) (base: $(config.base_config.variance_threshold))")
    println(prefix * "  Target Features: $(config.base_config.target_features)")
    println(prefix * "  Adaptation Rate: $(config.adaptation_rate)")
    println(prefix * "  Strict Mode: $(config.strict_mode)")
end

# Export types and functions
export ExtendedThresholdConfig, ThresholdStats
export create_default_config, load_config, save_config
export validate_config!, calculate_threshold_stats
export adjust_thresholds_for_count!, count_passing_features
export update_runtime_config!, create_constant_memory_config
export log_config

end # module ThresholdManagement