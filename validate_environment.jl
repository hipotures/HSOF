#!/usr/bin/env julia

# HSOF Environment Validation Script
# Checks system requirements and reports capabilities

using Pkg
using InteractiveUtils

println("="^80)
println("HSOF Environment Validation")
println("="^80)

# Track validation results
validation_results = Dict{String, Bool}()
validation_details = Dict{String, String}()

# Helper function for section headers
function section_header(title)
    println("\n" * "─"^60)
    println("📋 $title")
    println("─"^60)
end

# Check Julia version
section_header("Julia Environment")

julia_version = VERSION
julia_required = v"1.9.0"
julia_ok = julia_version >= julia_required

println("Julia Version: $julia_version $(julia_ok ? "✓" : "✗")")
validation_results["julia_version"] = julia_ok
if !julia_ok
    validation_details["julia_version"] = "Requires Julia $julia_required or higher"
end

# Check system information
println("\nSystem Information:")
println("  OS: $(Sys.KERNEL) $(Sys.MACHINE)")
println("  CPU: $(Sys.CPU_NAME) ($(Sys.CPU_THREADS) threads)")
println("  Memory: $(round(Sys.total_memory() / 2^30, digits=1)) GB")

# Check package dependencies
section_header("Package Dependencies")

required_packages = [
    "CUDA", "BenchmarkTools", "DataFrames", "Documenter",
    "Flux", "JSON", "MLJ", "Test", "TOML"
]

# Activate project
try
    Pkg.activate(dirname(@__FILE__))
    println("✓ Project environment activated")
catch e
    println("✗ Failed to activate project: $e")
    validation_results["project_activation"] = false
end

# Check each package
missing_packages = String[]
for pkg in required_packages
    if haskey(Pkg.project().dependencies, pkg)
        print("  $pkg: ✓")
        
        # Get version if available
        try
            deps = Pkg.dependencies()
            for (uuid, dep) in deps
                if dep.name == pkg
                    println(" (v$(dep.version))")
                    break
                end
            end
        catch
            println()
        end
    else
        println("  $pkg: ✗ Missing")
        push!(missing_packages, pkg)
    end
end

validation_results["packages"] = isempty(missing_packages)
if !isempty(missing_packages)
    validation_details["packages"] = "Missing: " * join(missing_packages, ", ")
end

# Check CUDA environment
section_header("CUDA Environment")

cuda_functional = false
gpu_count = 0
gpu_memory_total = 0.0

try
    using CUDA
    
    cuda_functional = CUDA.functional()
    println("CUDA Functional: $(cuda_functional ? "✓" : "✗")")
    
    if cuda_functional
        # CUDA version
        cuda_version = CUDA.runtime_version()
        cuda_required = v"11.8"
        cuda_ok = cuda_version >= cuda_required
        
        println("CUDA Version: $cuda_version $(cuda_ok ? "✓" : "✗")")
        validation_results["cuda_version"] = cuda_ok
        
        # GPU information
        gpu_count = length(CUDA.devices())
        println("\nGPU Devices: $gpu_count")
        
        for (i, dev) in enumerate(CUDA.devices())
            CUDA.device!(dev)
            gpu_name = CUDA.name(dev)
            gpu_memory = CUDA.totalmem(dev) / 2^30
            gpu_memory_total += gpu_memory
            cc = CUDA.capability(dev)
            
            println("  GPU $i: $gpu_name")
            println("    Compute Capability: $(cc.major).$(cc.minor)")
            println("    Memory: $(round(gpu_memory, digits=1)) GB")
            
            # Check compute capability
            cc_ok = cc.major >= 8 || (cc.major == 7 && cc.minor >= 5)
            validation_results["gpu$(i)_compute_capability"] = cc_ok
            
            # Check memory
            mem_ok = gpu_memory >= 12.0
            validation_results["gpu$(i)_memory"] = mem_ok
        end
        
        # Check peer access for multi-GPU
        if gpu_count > 1
            println("\nPeer Access Matrix:")
            for i in 0:gpu_count-1
                for j in 0:gpu_count-1
                    if i != j
                        CUDA.device!(i)
                        can_access = CUDA.can_access_peer(CUDA.CuDevice(j))
                        print("  GPU$i → GPU$j: $(can_access ? "✓" : "✗")")
                        println()
                    end
                end
            end
        end
        
        # Run simple CUDA test
        println("\nCUDA Test:")
        try
            CUDA.device!(0)
            a = CUDA.rand(1000)
            b = CUDA.rand(1000)
            c = a .+ b
            CUDA.synchronize()
            println("  Basic operations: ✓")
            validation_results["cuda_operations"] = true
        catch e
            println("  Basic operations: ✗ ($e)")
            validation_results["cuda_operations"] = false
        end
        
    else
        validation_results["cuda_functional"] = false
        validation_details["cuda_functional"] = "CUDA is not functional"
    end
    
catch e
    println("✗ CUDA.jl not available: $e")
    validation_results["cuda_available"] = false
end

# Check GPU requirements
section_header("Hardware Requirements")

# Minimum requirements
min_gpu_count = 1
recommended_gpu_count = 2
min_gpu_memory = 12.0
recommended_gpu_memory = 20.0
min_system_memory = 32.0
recommended_system_memory = 64.0

# Check GPU count
gpu_count_ok = gpu_count >= min_gpu_count
gpu_count_recommended = gpu_count >= recommended_gpu_count
println("GPU Count: $gpu_count $(gpu_count_ok ? "✓" : "✗") $(gpu_count_recommended ? "(Recommended ✓)" : "(Recommended: $recommended_gpu_count)")")
validation_results["gpu_count"] = gpu_count_ok

# Check GPU memory
avg_gpu_memory = gpu_count > 0 ? gpu_memory_total / gpu_count : 0.0
gpu_memory_ok = avg_gpu_memory >= min_gpu_memory
gpu_memory_recommended = avg_gpu_memory >= recommended_gpu_memory
println("GPU Memory (avg): $(round(avg_gpu_memory, digits=1)) GB $(gpu_memory_ok ? "✓" : "✗") $(gpu_memory_recommended ? "(Recommended ✓)" : "(Recommended: $recommended_gpu_memory GB)")")
validation_results["gpu_memory"] = gpu_memory_ok

# Check system memory
system_memory = Sys.total_memory() / 2^30
system_memory_ok = system_memory >= min_system_memory
system_memory_recommended = system_memory >= recommended_system_memory
println("System Memory: $(round(system_memory, digits=1)) GB $(system_memory_ok ? "✓" : "✗") $(system_memory_recommended ? "(Recommended ✓)" : "(Recommended: $recommended_system_memory GB)")")
validation_results["system_memory"] = system_memory_ok

# Check configuration files
section_header("Configuration Files")

config_files = [
    "configs/gpu_config.toml",
    "configs/algorithm_config.toml",
    "configs/data_config.toml",
]

all_configs_present = true
for config in config_files
    exists = isfile(config)
    println("  $config: $(exists ? "✓" : "✗")")
    if !exists
        all_configs_present = false
        
        # Check for example file
        example = config * ".example"
        if isfile(example)
            println("    → Example available: $example")
        end
    end
end
validation_results["config_files"] = all_configs_present

# Performance recommendations
section_header("Performance Recommendations")

if cuda_functional && gpu_count > 0
    # Check for persistence mode
    try
        persistence_output = read(`nvidia-smi -q -d PERSISTENCE_MODE`, String)
        persistence_on = occursin("Enabled", persistence_output)
        println("GPU Persistence Mode: $(persistence_on ? "✓" : "✗ (run: sudo nvidia-smi -pm 1)")")
    catch
        println("GPU Persistence Mode: Unable to check")
    end
    
    # Check for compute mode
    try
        compute_output = read(`nvidia-smi -q -d COMPUTE_MODE`, String)
        exclusive_mode = occursin("Exclusive", compute_output)
        println("GPU Compute Mode: $(exclusive_mode ? "Exclusive ✓" : "Default (consider exclusive mode)")")
    catch
        println("GPU Compute Mode: Unable to check")
    end
end

# Environment variables
println("\nEnvironment Variables:")
important_vars = [
    "JULIA_NUM_THREADS",
    "JULIA_CUDA_MEMORY_POOL",
    "JULIA_CUDA_SOFT_MEMORY_LIMIT",
    "CUDA_HOME",
]

for var in important_vars
    val = get(ENV, var, nothing)
    if val !== nothing
        println("  $var = $val")
    else
        println("  $var = <not set>")
    end
end

# Summary
section_header("Validation Summary")

all_passed = all(values(validation_results))
critical_passed = get(validation_results, "julia_version", false) && 
                 get(validation_results, "packages", false) &&
                 get(validation_results, "cuda_functional", true)

println("\nValidation Results:")
println("  Total Checks: $(length(validation_results))")
println("  Passed: $(count(values(validation_results)))")
println("  Failed: $(count(!v for v in values(validation_results)))")

if all_passed
    println("\n✅ All validation checks passed!")
    println("   HSOF is ready to use.")
elseif critical_passed
    println("\n⚠️  Some optional checks failed, but HSOF can still run.")
    println("   See details above for recommendations.")
else
    println("\n❌ Critical validation checks failed.")
    println("   HSOF may not function correctly.")
end

# Show failed checks
if !all_passed
    println("\nFailed Checks:")
    for (check, passed) in validation_results
        if !passed
            detail = get(validation_details, check, "")
            println("  - $check" * (isempty(detail) ? "" : ": $detail"))
        end
    end
end

# Quick start
if critical_passed
    println("\n🚀 Quick Start:")
    println("   julia> using HSOF")
    println("   julia> X, y = HSOF.generate_sample_data()")
    println("   julia> results = HSOF.select_features(X, y)")
end

println("\n" * "="^80)

# Return validation status
exit(all_passed ? 0 : 1)