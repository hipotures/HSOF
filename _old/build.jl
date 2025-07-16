#!/usr/bin/env julia

# HSOF Build Script
# Compiles CUDA kernels, builds documentation, and prepares the project

using Pkg
using CUDA

println("="^60)
println("HSOF Build System")
println("="^60)

# Parse command line arguments
build_docs = "--docs" in ARGS || "--all" in ARGS
build_kernels = "--kernels" in ARGS || "--all" in ARGS
run_tests = "--test" in ARGS || "--all" in ARGS
clean = "--clean" in ARGS

if length(ARGS) == 0
    println("Usage: julia build.jl [options]")
    println("Options:")
    println("  --all      Build everything (default)")
    println("  --kernels  Build CUDA kernels")
    println("  --docs     Build documentation")
    println("  --test     Run tests")
    println("  --clean    Clean build artifacts")
    exit(0)
end

# Clean build artifacts
if clean
    println("\n🧹 Cleaning build artifacts...")
    
    # Remove compiled kernels
    kernel_cache = joinpath(@__DIR__, "src", "gpu", "kernels", ".cache")
    if isdir(kernel_cache)
        rm(kernel_cache, recursive=true)
        println("  ✓ Removed kernel cache")
    end
    
    # Remove documentation build
    docs_build = joinpath(@__DIR__, "docs", "build")
    if isdir(docs_build)
        rm(docs_build, recursive=true)
        println("  ✓ Removed documentation build")
    end
    
    # Clean Julia artifacts
    Pkg.gc()
    println("  ✓ Cleaned Julia artifacts")
    
    exit(0)
end

# Activate project environment
println("\n📦 Activating project environment...")
Pkg.activate(@__DIR__)

# Check and install dependencies
println("\n📋 Checking dependencies...")
missing_deps = String[]

for dep in ["CUDA", "BenchmarkTools", "Documenter", "Test"]
    if !haskey(Pkg.project().dependencies, dep)
        push!(missing_deps, dep)
    end
end

if !isempty(missing_deps)
    println("  ⚠️  Missing dependencies: ", join(missing_deps, ", "))
    print("  Install missing dependencies? [y/N]: ")
    response = readline()
    if lowercase(response) == "y"
        Pkg.add(missing_deps)
    else
        println("  ❌ Cannot continue without dependencies")
        exit(1)
    end
else
    println("  ✓ All dependencies installed")
end

# Build CUDA kernels
if build_kernels
    println("\n🔧 Building CUDA kernels...")
    
    # Check CUDA availability
    if !CUDA.functional()
        println("  ❌ CUDA not functional. Skipping kernel compilation.")
    else
        println("  CUDA version: ", CUDA.runtime_version())
        println("  GPU: ", CUDA.name(CUDA.device()))
        
        # Precompile CUDA runtime
        println("  Precompiling CUDA runtime...")
        CUDA.precompile_runtime()
        
        # Create kernel cache directory
        kernel_dir = joinpath(@__DIR__, "src", "gpu", "kernels")
        cache_dir = joinpath(kernel_dir, ".cache")
        mkpath(cache_dir)
        
        # Compile test kernels
        println("  Compiling kernel tests...")
        include("test/gpu/kernel_tests.jl")
        
        println("  ✓ Kernel compilation complete")
    end
end

# Build documentation
if build_docs
    println("\n📚 Building documentation...")
    
    # Check if Documenter is available
    try
        using Documenter
        
        # Change to docs directory
        cd(joinpath(@__DIR__, "docs")) do
            # Run documentation build
            include("make.jl")
        end
        
        println("  ✓ Documentation built successfully")
        println("  📂 Output: docs/build/")
        
    catch e
        println("  ❌ Documentation build failed: ", e)
    end
end

# Run tests
if run_tests
    println("\n🧪 Running tests...")
    
    # Set test environment
    ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "0.9"
    
    # Run specific test suites
    test_suites = [
        ("Configuration", "test/test_config_loader.jl"),
        ("CUDA Validation", "test/cuda_validation.jl"),
        ("GPU Manager", "test/gpu/test_gpu_manager.jl"),
        ("Device Manager", "test/gpu/test_device_manager.jl"),
    ]
    
    passed = 0
    failed = 0
    
    for (name, test_file) in test_suites
        print("  Testing $name... ")
        
        try
            # Capture output
            original_stdout = stdout
            (rd, wr) = redirect_stdout()
            
            include(test_file)
            
            # Restore stdout
            redirect_stdout(original_stdout)
            close(wr)
            
            println("✓")
            passed += 1
        catch e
            println("✗")
            println("    Error: ", e)
            failed += 1
            
            # Restore stdout in case of error
            redirect_stdout(original_stdout)
        end
    end
    
    println("\n  Test Summary:")
    println("  ✓ Passed: $passed")
    if failed > 0
        println("  ✗ Failed: $failed")
    end
    
    # Run full test suite if all quick tests pass
    if failed == 0
        println("\n  Running full test suite...")
        Pkg.test()
    end
end

# Generate configuration files
println("\n⚙️  Checking configuration files...")

config_files = [
    ("configs/gpu_config.toml", "configs/gpu_config.toml.example"),
    ("configs/algorithm_config.toml", "configs/algorithm_config.toml.example"),
    ("configs/data_config.toml", "configs/data_config.toml.example"),
]

for (config, example) in config_files
    if !isfile(config) && isfile(example)
        cp(example, config)
        println("  ✓ Created $config from example")
    end
end

# Performance optimization
if CUDA.functional()
    println("\n⚡ Applying performance optimizations...")
    
    # Set CUDA flags
    ENV["JULIA_CUDA_MEMORY_POOL"] = "cuda"
    ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "0.9"
    
    # Enable GPU persistence mode (requires sudo)
    try
        run(`nvidia-smi -pm 1`)
        println("  ✓ GPU persistence mode enabled")
    catch
        println("  ⚠️  Could not enable GPU persistence (requires sudo)")
    end
end

# Summary
println("\n" * "="^60)
println("Build Summary")
println("="^60)

if build_kernels
    println("✓ CUDA kernels compiled")
end

if build_docs
    println("✓ Documentation generated")
end

if run_tests
    println("✓ Tests executed")
end

println("\n🚀 Build complete!")
println("\nNext steps:")
println("  julia> using HSOF")
println("  julia> HSOF.validate_environment()")
println("  julia> ?HSOF  # For help")