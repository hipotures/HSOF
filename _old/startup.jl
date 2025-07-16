# HSOF Project Startup Script
# This file is loaded automatically when starting Julia with --project=.

println("Loading HSOF project environment...")

# Activate the project environment
using Pkg
Pkg.activate(".")

# Load commonly used packages
using CUDA
using DataFrames
using Flux
using MLJ
using SQLite

# Set up logging
using Logging
global_logger(ConsoleLogger(stderr, Logging.Info))

# Configure CUDA settings
if CUDA.functional()
    # Set memory pool growth
    CUDA.memory_pool_grow!(2^30)  # Allow 1GB growth increments
    
    # Enable fast math
    CUDA.math_mode!(CUDA.FAST_MATH)
    
    println("CUDA initialized with $(length(CUDA.devices())) GPU(s)")
else
    @warn "CUDA is not functional"
end

# Load the main module
using HSOF

println("HSOF environment loaded successfully!")
println("Run `HSOF.validate_environment()` to check system requirements.")