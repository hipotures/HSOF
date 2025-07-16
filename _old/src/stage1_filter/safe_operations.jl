module SafeOperations

using CUDA
using Statistics

# Include error handling
include("error_handling.jl")
using .ErrorHandling

# Include existing modules
include("variance_calculation.jl")
include("correlation_matrix.jl") 
include("mutual_information.jl")

using .VarianceCalculation
using .CorrelationComputation
using .MutualInformation

export safe_variance_calculation, safe_correlation_matrix, safe_mutual_information
export SafeComputeConfig, create_safe_config

"""
Configuration for safe GPU operations
"""
struct SafeComputeConfig
    recovery::ErrorRecovery
    logger::ErrorLogger
    enable_validation::Bool
    validation_sample_size::Int
    numerical_tolerance::Float32
end

"""
Create default safe computation configuration
"""
function create_safe_config(;
    log_dir::String = ".taskmaster/logs",
    enable_validation::Bool = true,
    validation_sample_size::Int = 100
)
    recovery = create_error_recovery()
    logger = create_error_logger(log_dir=log_dir, detail_level=:normal)
    
    return SafeComputeConfig(
        recovery,
        logger,
        enable_validation,
        validation_sample_size,
        Float32(1e-6)
    )
end

"""
Safe variance calculation with error handling
"""
function safe_variance_calculation(
    X::CuArray{Float32, 2},
    config::SafeComputeConfig = create_safe_config()
)
    n_features, n_samples = size(X)
    context = Dict{String, Any}(
        "operation" => "variance_calculation",
        "n_features" => n_features,
        "n_samples" => n_samples,
        "batch_size" => n_features
    )
    
    result = with_gpu_recovery(config.recovery, config.logger, context=context) do
        # Check if we should use CPU fallback
        if get(context, "use_cpu", false)
            @info "Using CPU fallback for variance calculation"
            X_cpu = Array(X)
            variances_cpu = cpu_fallback_variance(X_cpu)
            return CuArray(variances_cpu)
        end
        
        # Allocate output with memory check
        variances = ErrorHandling.@memory_check begin
            CUDA.zeros(Float32, n_features)
        end (n_features * sizeof(Float32))
        
        # Validate input data if enabled
        if config.enable_validation && n_samples >= config.validation_sample_size
            sample_data = Array(X[:, 1:config.validation_sample_size])
            if !all(isfinite.(sample_data))
                throw(NumericalError(
                    "Non-finite input values detected",
                    "variance_calculation",
                    Dict("sample_size" => config.validation_sample_size)
                ))
            end
        end
        
        # Perform GPU calculation
        batch_size = get(context, "batch_size", n_features)
        if batch_size < n_features
            # Process in smaller batches if needed
            for start_idx in 1:batch_size:n_features
                end_idx = min(start_idx + batch_size - 1, n_features)
                batch_features = end_idx - start_idx + 1
                
                X_batch = @view X[start_idx:end_idx, :]
                variances_batch = @view variances[start_idx:end_idx]
                
                # Create temporary config for batch
                batch_config = VarianceConfig(
                    Int32(batch_features),
                    Int32(n_samples),
                    128,  # threads_per_block
                    true, # use_shared_memory
                    Float32(1e-8)  # epsilon
                )
                
                compute_variance_gpu!(variances_batch, X_batch, batch_config)
            end
        else
            # Process all at once
            var_config = create_variance_config(n_features, n_samples)
            compute_variance_gpu!(variances, X, var_config)
        end
        
        CUDA.synchronize()
        
        # Validate output
        with_numerical_checks(() -> variances, [:finite, :variance], logger=config.logger)
        
        return variances
    end
    
    return result
end

"""
Safe correlation matrix calculation with error handling
"""
function safe_correlation_matrix(
    X::CuArray{Float32, 2},
    config::SafeComputeConfig = create_safe_config()
)
    n_features, n_samples = size(X)
    context = Dict{String, Any}(
        "operation" => "correlation_matrix",
        "n_features" => n_features,
        "n_samples" => n_samples,
        "matrix_size" => n_features * n_features
    )
    
    result = with_gpu_recovery(config.recovery, config.logger, context=context) do
        # Check if we should use CPU fallback
        if get(context, "use_cpu", false)
            @info "Using CPU fallback for correlation calculation"
            X_cpu = Array(X)
            corr_cpu = cpu_fallback_correlation(X_cpu)
            return CuArray(corr_cpu)
        end
        
        # Check memory requirements
        required_memory = n_features * n_features * sizeof(Float32) * 3  # For correlation + intermediates
        mem_info = CUDA.memory_status()
        
        if mem_info.free < required_memory * 1.2  # 20% buffer
            # Try to free memory
            CUDA.reclaim()
            mem_info = CUDA.memory_status()
            
            if mem_info.free < required_memory
                throw(MemoryError(
                    "Insufficient memory for correlation matrix",
                    required_memory,
                    mem_info.free,
                    CUDA.device().handle
                ))
            end
        end
        
        # Validate input if enabled
        if config.enable_validation
            sample_size = min(config.validation_sample_size, n_samples)
            sample_data = Array(X[:, 1:sample_size])
            if !all(isfinite.(sample_data))
                throw(NumericalError(
                    "Non-finite input values detected",
                    "correlation_matrix",
                    Dict("sample_size" => sample_size)
                ))
            end
        end
        
        # Create correlation config
        corr_config = create_correlation_config(n_features, n_samples, use_cublas=true)
        
        # Allocate correlation matrix with memory check
        correlation_matrix = ErrorHandling.@memory_check begin
            CUDA.zeros(Float32, n_features, n_features)
        end (n_features * n_features * sizeof(Float32))
        
        # Compute correlation
        compute_correlation_matrix!(correlation_matrix, X, corr_config)
        CUDA.synchronize()
        
        # Validate output
        if config.enable_validation
            # Check diagonal elements
            diag_elements = [correlation_matrix[i, i] for i in 1:min(10, n_features)]
            diag_array = CuArray(diag_elements)
            
            # Diagonal should be close to 1.0
            if any(abs.(diag_elements .- 1.0f0) .> config.numerical_tolerance * 100)
                throw(NumericalError(
                    "Invalid correlation matrix diagonal",
                    "correlation_computation",
                    Dict(
                        "diagonal_samples" => diag_elements,
                        "expected" => 1.0
                    )
                ))
            end
        end
        
        return correlation_matrix
    end
    
    return result
end

"""
Safe mutual information calculation with error handling
"""
function safe_mutual_information(
    X::CuArray{Float32, 2},
    y::CuArray{Int32, 1},
    config::SafeComputeConfig = create_safe_config();
    n_bins::Int = 10,
    n_classes::Int = 3
)
    n_features, n_samples = size(X)
    
    # Validate input dimensions
    if length(y) != n_samples
        throw(ArgumentError("Label array length ($(length(y))) must match number of samples ($n_samples)"))
    end
    
    context = Dict{String, Any}(
        "operation" => "mutual_information",
        "n_features" => n_features,
        "n_samples" => n_samples,
        "n_bins" => n_bins,
        "n_classes" => n_classes
    )
    
    result = with_gpu_recovery(config.recovery, config.logger, context=context) do
        # Check if we should use CPU fallback
        if get(context, "use_cpu", false)
            @info "Using CPU fallback for mutual information calculation"
            X_cpu = Array(X)
            y_cpu = Array(y)
            mi_cpu = cpu_fallback_mutual_information(X_cpu, y_cpu, n_bins=n_bins)
            return CuArray(mi_cpu)
        end
        
        # Validate labels
        y_array = Array(y[1:min(100, end)])
        if any(y_array .< 1) || any(y_array .> n_classes)
            throw(ArgumentError("Labels must be in range 1:$n_classes"))
        end
        
        # Create MI config
        mi_config = create_mi_config(
            n_features, n_samples,
            n_bins=n_bins,
            n_classes=n_classes
        )
        
        # Allocate output with memory check
        mi_scores = ErrorHandling.@memory_check begin
            CUDA.zeros(Float32, n_features)
        end (n_features * sizeof(Float32))
        
        # Compute mutual information
        compute_mutual_information!(mi_scores, X, y, mi_config)
        CUDA.synchronize()
        
        # Validate output
        with_numerical_checks(() -> mi_scores, [:finite, :positive], logger=config.logger)
        
        # MI scores should be non-negative
        min_mi = minimum(Array(mi_scores))
        if min_mi < -config.numerical_tolerance
            throw(NumericalError(
                "Negative mutual information scores",
                "mutual_information",
                Dict(
                    "min_score" => min_mi,
                    "negative_count" => sum(Array(mi_scores) .< 0)
                )
            ))
        end
        
        return mi_scores
    end
    
    return result
end

"""
Safe batch processing wrapper
"""
function safe_batch_process(
    f::Function,
    data::CuArray,
    batch_size::Int,
    config::SafeComputeConfig
)
    n_elements = size(data, 2)
    n_batches = cld(n_elements, batch_size)
    
    results = []
    
    for batch_idx in 1:n_batches
        start_idx = (batch_idx - 1) * batch_size + 1
        end_idx = min(batch_idx * batch_size, n_elements)
        
        batch_data = @view data[:, start_idx:end_idx]
        
        context = Dict{String, Any}(
            "batch_index" => batch_idx,
            "batch_size" => end_idx - start_idx + 1,
            "total_batches" => n_batches
        )
        
        batch_result = with_gpu_recovery(
            config.recovery, 
            config.logger, 
            context=context
        ) do
            f(batch_data)
        end
        
        push!(results, batch_result)
    end
    
    return results
end

"""
Validate GPU computation results
"""
function validate_gpu_results(
    gpu_result::CuArray,
    cpu_reference::Array;
    tolerance::Float32 = 1e-4f0,
    sample_size::Int = 100
)
    gpu_array = Array(gpu_result)
    
    # Sample comparison if arrays are large
    n_elements = length(gpu_array)
    if n_elements > sample_size
        indices = rand(1:n_elements, sample_size)
        gpu_sample = gpu_array[indices]
        cpu_sample = cpu_reference[indices]
    else
        gpu_sample = gpu_array
        cpu_sample = cpu_reference
    end
    
    # Compute differences
    abs_diff = abs.(gpu_sample .- cpu_sample)
    rel_diff = abs_diff ./ (abs.(cpu_sample) .+ 1e-8)
    
    max_abs_diff = maximum(abs_diff)
    max_rel_diff = maximum(rel_diff)
    
    validation_passed = max_abs_diff < tolerance || max_rel_diff < tolerance
    
    return (
        passed = validation_passed,
        max_abs_diff = max_abs_diff,
        max_rel_diff = max_rel_diff,
        sample_size = length(gpu_sample)
    )
end

end # module