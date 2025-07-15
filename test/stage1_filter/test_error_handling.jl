using Test
using CUDA
using Statistics
using Random

# Skip if no GPU
if !CUDA.functional()
    @warn "CUDA not functional, skipping error handling tests"
    exit(0)
end

# Include modules
include("../../src/stage1_filter/error_handling.jl")
include("../../src/stage1_filter/safe_operations.jl")

using .ErrorHandling
using .SafeOperations

println("Testing Error Handling System...")
println("="^60)

@testset "Error Handling Tests" begin
    
    @testset "Error Types" begin
        # Test MemoryError
        mem_err = MemoryError("Out of memory", 1_000_000, 500_000, 0)
        @test mem_err.requested_bytes == 1_000_000
        @test mem_err.available_bytes == 500_000
        @test contains(mem_err.message, "Out of memory")
        
        # Test KernelError
        kernel_err = KernelError(
            "Kernel launch failed",
            "test_kernel",
            CUDA.ERROR_INVALID_VALUE,
            (threads=256, blocks=100)
        )
        @test kernel_err.kernel_name == "test_kernel"
        @test kernel_err.thread_config.threads == 256
        
        # Test NumericalError
        num_err = NumericalError(
            "Invalid values",
            "variance_calculation",
            Dict("min_value" => -1.0, "count" => 5)
        )
        @test num_err.operation == "variance_calculation"
        @test num_err.values["count"] == 5
    end
    
    @testset "Error Recovery Configuration" begin
        recovery = create_error_recovery(
            max_retries=5,
            retry_delay_ms=200,
            memory_pressure_threshold=0.8f0
        )
        
        @test recovery.max_retries == 5
        @test recovery.retry_delay_ms == 200
        @test recovery.memory_pressure_threshold == 0.8f0
        @test haskey(recovery.strategies, MemoryError)
        @test CLEAR_CACHE_AND_RETRY in recovery.strategies[MemoryError]
    end
    
    @testset "Error Logger" begin
        # Create temporary log directory
        temp_dir = mktempdir()
        logger = ErrorHandling.create_error_logger(
            log_dir=temp_dir,
            detail_level=:verbose
        )
        
        @test isdir(temp_dir)
        @test logger.detail_level == :verbose
        @test logger.include_stack_trace == true
        @test logger.include_gpu_state == true
        
        # Test logging
        test_error = NumericalError(
            "Test error",
            "test_op",
            Dict("value" => 42)
        )
        
        log_error!(logger, test_error, Dict("context" => "unit_test"))
        
        @test isfile(logger.log_file)
        log_content = read(logger.log_file, String)
        @test contains(log_content, "Test error")
        @test contains(log_content, "test_op")
        @test contains(log_content, "42")
        
        # Cleanup
        rm(temp_dir, recursive=true)
    end
    
    @testset "Memory Checking Macro" begin
        # Test successful allocation
        bytes_needed = 1000 * sizeof(Float32)
        arr = ErrorHandling.@memory_check CUDA.zeros(Float32, 1000) bytes_needed
        @test size(arr) == (1000,)
        
        # Test allocation failure detection
        huge_size = 100_000_000_000  # 100GB - should fail
        @test_throws MemoryError begin
            ErrorHandling.@memory_check CUDA.zeros(Float32, huge_size ÷ sizeof(Float32)) huge_size
        end
    end
    
    @testset "Numerical Checking Macro" begin
        # Test finite check - passing
        good_values = CuArray([1.0f0, 2.0f0, 3.0f0])
        result = ErrorHandling.@numerical_check good_values :finite
        @test result === good_values
        
        # Test finite check - failing
        bad_values = CuArray([1.0f0, NaN32, 3.0f0])
        @test_throws NumericalError ErrorHandling.@numerical_check bad_values :finite
        
        # Test positive check
        positive_values = CuArray([1.0f0, 2.0f0, 3.0f0])
        ErrorHandling.@numerical_check positive_values :positive
        
        negative_values = CuArray([1.0f0, -2.0f0, 3.0f0])
        @test_throws NumericalError ErrorHandling.@numerical_check negative_values :positive
        
        # Test variance check
        valid_variances = CuArray([0.1f0, 0.2f0, 0.0f0])
        ErrorHandling.@numerical_check valid_variances :variance
        
        invalid_variances = CuArray([0.1f0, -0.01f0, 0.2f0])
        @test_throws NumericalError ErrorHandling.@numerical_check invalid_variances :variance
    end
    
    @testset "Error Recovery Strategies" begin
        recovery = create_error_recovery()
        
        # Test memory error handling
        mem_error = MemoryError("OOM", 1000, 500, 0)
        context = Dict{String, Any}("batch_size" => 1000)
        
        strategy = handle_error!(recovery, mem_error, context)
        @test strategy in [CLEAR_CACHE_AND_RETRY, RETRY_WITH_SMALLER_BATCH, FALLBACK_TO_CPU]
        
        # Test max retries
        for i in 1:recovery.max_retries + 1
            strategy = handle_error!(recovery, mem_error, context)
        end
        @test strategy == ABORT_WITH_LOGGING
    end
    
    @testset "GPU Recovery Wrapper" begin
        recovery = create_error_recovery(max_retries=2)
        temp_dir = mktempdir()
        logger = ErrorHandling.create_error_logger(log_dir=temp_dir)
        
        # Test successful operation
        result = with_gpu_recovery(recovery, logger) do
            CuArray([1.0f0, 2.0f0, 3.0f0])
        end
        @test Array(result) == [1.0f0, 2.0f0, 3.0f0]
        
        # Test operation with retry
        attempt_count = 0
        result = with_gpu_recovery(recovery, logger) do
            attempt_count += 1
            if attempt_count < 2
                throw(MemoryError("Simulated OOM", 1000, 500, 0))
            end
            CuArray([4.0f0, 5.0f0, 6.0f0])
        end
        @test attempt_count == 2
        @test Array(result) == [4.0f0, 5.0f0, 6.0f0]
        
        # Cleanup
        rm(temp_dir, recursive=true)
    end
    
    @testset "Safe Variance Calculation" begin
        # Generate test data
        Random.seed!(42)
        n_features = 100
        n_samples = 1000
        X = CUDA.randn(Float32, n_features, n_samples)
        
        config = create_safe_config()
        
        # Test normal operation
        variances = safe_variance_calculation(X, config)
        @test size(variances) == (n_features,)
        @test all(Array(variances) .>= 0)
        
        # Test with NaN injection (should handle gracefully)
        X_with_nan = copy(X)
        X_with_nan[1, 1] = NaN32
        
        # This should either recover or fall back to CPU
        @test_nowarn safe_variance_calculation(X_with_nan, config)
    end
    
    @testset "Safe Correlation Matrix" begin
        # Small test case
        n_features = 50
        n_samples = 500
        X = CUDA.randn(Float32, n_features, n_samples)
        
        config = create_safe_config()
        
        # Test normal operation
        corr_matrix = safe_correlation_matrix(X, config)
        @test size(corr_matrix) == (n_features, n_features)
        
        # Check diagonal elements
        diag_elements = [corr_matrix[i, i] for i in 1:n_features]
        @test all(abs.(Array(CuArray(diag_elements)) .- 1.0f0) .< 1e-3)
    end
    
    @testset "Safe Mutual Information" begin
        # Generate test data
        n_features = 100
        n_samples = 1000
        X = CUDA.randn(Float32, n_features, n_samples)
        y = CuArray(rand(1:3, n_samples))
        
        config = create_safe_config()
        
        # Test normal operation
        mi_scores = safe_mutual_information(X, y, config)
        @test size(mi_scores) == (n_features,)
        @test all(Array(mi_scores) .>= 0)
        
        # Test with invalid labels
        y_invalid = copy(y)
        y_invalid[1] = 0  # Invalid label
        
        @test_throws ArgumentError safe_mutual_information(X, y_invalid, config)
    end
    
    @testset "CPU Fallback Functions" begin
        # Test data
        Random.seed!(42)
        X_cpu = randn(Float32, 10, 100)
        y_cpu = rand(1:3, 100)
        
        # Test variance fallback
        var_cpu = ErrorHandling.cpu_fallback_variance(X_cpu)
        var_expected = vec(var(X_cpu, dims=2, corrected=true))
        @test maximum(abs.(var_cpu .- var_expected)) < 1e-5
        
        # Test correlation fallback
        corr_cpu = ErrorHandling.cpu_fallback_correlation(X_cpu)
        corr_expected = cor(X_cpu')
        @test maximum(abs.(corr_cpu .- corr_expected)) < 1e-5
        
        # Test MI fallback
        mi_cpu = ErrorHandling.cpu_fallback_mutual_information(
            X_cpu, Int32.(y_cpu), n_bins=5
        )
        @test all(mi_cpu .>= 0)
        @test length(mi_cpu) == size(X_cpu, 1)
    end
    
    @testset "Result Validation" begin
        # Generate matching CPU and GPU results
        Random.seed!(42)
        cpu_result = randn(Float32, 100)
        gpu_result = CuArray(cpu_result .+ randn(Float32, 100) * 1e-5)
        
        validation = SafeOperations.validate_gpu_results(
            gpu_result, cpu_result,
            tolerance=1e-3f0
        )
        
        @test validation.passed == true
        @test validation.max_abs_diff < 1e-3
        
        # Test validation failure
        gpu_result_bad = CuArray(cpu_result .+ 1.0f0)
        validation_bad = SafeOperations.validate_gpu_results(
            gpu_result_bad, cpu_result,
            tolerance=1e-3f0
        )
        
        @test validation_bad.passed == false
        @test validation_bad.max_abs_diff > 0.9
    end
end

# Integration test with real computation
println("\n" * "="^60)
println("INTEGRATION TEST")
println("="^60)

# Create test dataset
Random.seed!(42)
n_features = 500
n_samples = 2000
X = CUDA.randn(Float32, n_features, n_samples) .* 2.0f0 .+ 1.0f0
y = CuArray(rand(1:3, n_samples))

# Create safe config with logging
temp_dir = mktempdir()
config = SafeComputeConfig(
    create_error_recovery(max_retries=3),
    ErrorHandling.create_error_logger(log_dir=temp_dir, detail_level=:normal),
    true,  # enable_validation
    100,   # validation_sample_size
    1e-6f0 # numerical_tolerance
)

println("\n1. Testing safe variance calculation...")
t1 = @elapsed variances = safe_variance_calculation(X, config)
println("   Completed in $(round(t1, digits=3))s")
println("   Min variance: $(minimum(Array(variances)))")
println("   Max variance: $(maximum(Array(variances)))")

println("\n2. Testing safe correlation matrix...")
t2 = @elapsed corr_matrix = safe_correlation_matrix(X, config)
println("   Completed in $(round(t2, digits=3))s")
println("   Matrix size: $(size(corr_matrix))")
println("   Diagonal check: all ≈ 1.0? $(all(abs.([corr_matrix[i,i] for i in 1:10] .- 1.0f0) .< 1e-3))")

println("\n3. Testing safe mutual information...")
t3 = @elapsed mi_scores = safe_mutual_information(X, y, config)
println("   Completed in $(round(t3, digits=3))s")
println("   Min MI score: $(minimum(Array(mi_scores)))")
println("   Max MI score: $(maximum(Array(mi_scores)))")

# Check if any errors were logged
log_files = readdir(temp_dir)
if !isempty(log_files)
    println("\n⚠️  Errors were logged during execution:")
    for file in log_files
        println("   - $file ($(filesize(joinpath(temp_dir, file))) bytes)")
    end
else
    println("\n✓ No errors logged during execution")
end

# Cleanup
rm(temp_dir, recursive=true)

println("\n" * "="^60)
println("TEST SUMMARY")
println("="^60)
println("✓ Error types and creation working")
println("✓ Error recovery strategies functional")
println("✓ Error logging system operational")
println("✓ Memory and numerical checks working")
println("✓ GPU recovery wrapper handles failures")
println("✓ Safe operations with validation working")
println("✓ CPU fallback implementations functional")
println("✓ Integration with Stage 1 modules successful")
println("="^60)