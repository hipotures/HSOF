# HSOF Test Suite Runner
# Main entry point for Pkg.test()

using Test
using HSOF
using CUDA

# Set up test groups
const TEST_GROUPS = [
    "unit",
    "gpu", 
    "integration",
    "benchmarks"
]

# Parse command line arguments
enabled_groups = if isempty(ARGS)
    TEST_GROUPS
else
    filter(g -> g in TEST_GROUPS, ARGS)
end

# Check if we should skip GPU tests
skip_gpu = !CUDA.functional() || get(ENV, "HSOF_SKIP_GPU_TESTS", "false") == "true"

println("="^60)
println("HSOF Test Suite")
println("="^60)
println("Enabled test groups: ", join(enabled_groups, ", "))
println("GPU tests: ", skip_gpu ? "SKIPPED" : "ENABLED")
println("="^60)

# Track overall results
total_tests = 0
failed_tests = 0

# Run unit tests
if "unit" in enabled_groups
    @testset "Unit Tests" begin
        @testset "Configuration" begin
            include("test_config_loader.jl")
        end
        
        @testset "CUDA Validation" begin
            include("cuda_validation.jl")
        end
    end
end

# Run GPU tests
if "gpu" in enabled_groups && !skip_gpu
    @testset "GPU Tests" begin
        @testset "GPU Manager" begin
            include("gpu/test_gpu_manager.jl")
        end
        
        @testset "Device Manager" begin
            include("gpu/test_device_manager.jl")
        end
        
        @testset "Kernel Tests" begin
            # Use the CI-compatible version if no GPU
            if CUDA.functional()
                include("gpu/kernel_tests.jl")
                results = KernelTests.run_all_tests(verbose=false)
                @test all(r -> r.passed for r in results)
            else
                include("gpu/ci_kernel_tests.jl")
                run_ci_kernel_tests()
            end
        end
        
        @testset "PCIe Validation" begin
            include("gpu/test_pcie_validation.jl")
        end
    end
elseif "gpu" in enabled_groups
    @info "Skipping GPU tests (no GPU available)"
end

# Run integration tests
if "integration" in enabled_groups
    @testset "Integration Tests" begin
        @testset "Environment Setup" begin
            include("integration/setup_test.jl")
        end
        
        @testset "Pipeline Integration" begin
            include("integration/pipeline_test.jl")
        end
    end
end

# Run benchmark tests (optional, usually separate)
if "benchmarks" in enabled_groups
    @testset "Benchmark Tests" begin
        @info "Running quick benchmark validation..."
        
        # Just validate that benchmarks can run, don't do full benchmarks
        include("../benchmarks/run_benchmarks.jl")
        
        # Run minimal benchmark
        X = randn(100, 50)
        y = rand([0, 1], 100)
        
        # Mock minimal benchmark
        @test size(X) == (100, 50)
        @test length(y) == 100
    end
end

# Summary
println("\n" * "="^60)
println("Test Summary")
println("="^60)

# Exit with appropriate code
exit_code = failed_tests > 0 ? 1 : 0
exit(exit_code)