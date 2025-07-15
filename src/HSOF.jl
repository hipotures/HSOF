module HSOF

# Standard library imports
using Dates
using LinearAlgebra
using Logging
using Printf
using Random
using Statistics
using Test

# External dependencies
using CUDA
using DataFrames
using Flux
using MLJ
using SQLite
using TOML

# Re-export commonly used functions
export initialize_project, validate_environment

# Include submodules
include("gpu/GPU.jl")
include("database/Database.jl")
include("ui/UI.jl")

# Using and re-exporting submodules
using .GPU
using .Database
using .UI

# Export main components
export GPU, Database, UI
export initialize_project, validate_environment, generate_sample_data
# GPU exports
export MCTSGPUEngine, select_features

# Module initialization
function __init__()
    # Check for CUDA availability on module load
    if CUDA.functional()
        @info "CUDA is functional. Found $(length(CUDA.devices())) GPU(s)"
    else
        @warn "CUDA is not functional. GPU acceleration will not be available."
    end
end

"""
    initialize_project()

Initialize the HSOF project environment and validate system requirements.
"""
function initialize_project()
    @info "Initializing HSOF project..."
    
    # Validate Julia version
    if VERSION < v"1.9"
        error("Julia 1.9+ is required. Current version: $VERSION")
    end
    
    # Check CUDA
    if !CUDA.functional()
        @warn "CUDA is not functional. Please check your CUDA installation."
        return false
    end
    
    # Check for dual GPUs
    devices = CUDA.devices()
    if length(devices) < 2
        @warn "Found only $(length(devices)) GPU(s). This project is optimized for 2 GPUs."
    end
    
    @info "Project initialized successfully!"
    return true
end

"""
    validate_environment()

Run comprehensive environment validation checks.
"""
function validate_environment()
    @info "Running environment validation..."
    
    results = Dict{String, Bool}()
    
    # Julia version
    results["Julia 1.9+"] = VERSION >= v"1.9"
    
    # CUDA functionality
    results["CUDA functional"] = CUDA.functional()
    
    # GPU count
    if CUDA.functional()
        gpu_count = length(CUDA.devices())
        results["GPU count ≥ 2"] = gpu_count >= 2
        
        # Check each GPU
        for (i, dev) in enumerate(CUDA.devices())
            CUDA.device!(dev)
            results["GPU $i ($(CUDA.name(dev)))"] = true
            
            # Check compute capability
            cc = CUDA.capability(dev)
            results["GPU $i compute capability ≥ 8.9"] = cc.major >= 8 && cc.minor >= 9
            
            # Check memory
            mem_gb = CUDA.totalmem(dev) / 1024^3
            results["GPU $i memory ≥ 20GB"] = mem_gb >= 20
        end
    end
    
    # Package availability
    for pkg in ["CUDA", "Flux", "MLJ", "SQLite", "DataFrames"]
        try
            eval(Meta.parse("using $pkg"))
            results["Package $pkg"] = true
        catch
            results["Package $pkg"] = false
        end
    end
    
    # Print results
    println("\nEnvironment Validation Results:")
    println("=" ^ 50)
    for (check, passed) in sort(collect(results))
        status = passed ? "✓" : "✗"
        color = passed ? :green : :red
        printstyled(@sprintf("%-40s %s\n", check, status); color=color)
    end
    
    all_passed = all(values(results))
    if all_passed
        printstyled("\nAll checks passed! ✓\n"; color=:green, bold=true)
    else
        printstyled("\nSome checks failed! ✗\n"; color=:red, bold=true)
    end
    
    return all_passed
end

"""
    generate_sample_data(; n_samples::Int=1000, n_features::Int=100, n_informative::Int=20)

Generate sample data for testing and development.
"""
function generate_sample_data(; n_samples::Int=1000, n_features::Int=100, n_informative::Int=20)
    # Generate informative features
    X_informative = randn(n_samples, n_informative)
    
    # Generate noise features
    X_noise = randn(n_samples, n_features - n_informative)
    
    # Combine
    X = hcat(X_informative, X_noise)
    
    # Generate target based on informative features
    true_weights = randn(n_informative)
    y_continuous = X_informative * true_weights + 0.1 * randn(n_samples)
    y = Int.(y_continuous .> median(y_continuous))
    
    return X, y
end

end # module