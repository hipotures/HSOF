using Test
using CUDA
using PyCall
using Statistics
using LinearAlgebra
using Random

# Set Python path
ENV["PYTHON"] = ""  # Use Julia's conda Python

# Import sklearn validation module
include("../../src/stage1_filter/sklearn_validation.jl")
using .SklearnValidation

println("Testing Stage 1 Sklearn Validation...")

# Skip tests if no GPU available
if !CUDA.functional()
    @warn "CUDA not functional, skipping GPU validation tests"
    exit(0)
end

# Test 1: Small dataset validation
println("\n=== Test 1: Small Dataset ===")
Random.seed!(42)
X_small = randn(Float32, 100, 20)
y_small = Int32.(rand(0:1, 100))

config = create_validation_config(verbose=true)
results_small = run_validation_suite(X_small, y_small, config)

@test results_small.all_passed

# Test 2: Medium dataset with correlated features
println("\n=== Test 2: Correlated Features ===")
X_corr = randn(Float32, 500, 50)
# Add correlated features
for i in 1:10
    X_corr[:, i+10] = X_corr[:, i] + 0.1f0 * randn(Float32, 500)
end
y_corr = Int32.(X_corr[:, 1] .> 0)

results_corr = run_validation_suite(X_corr, y_corr, config)
@test results_corr.all_passed

# Test 3: Performance test
println("\n=== Test 3: Performance Test ===")
X_perf = randn(Float32, 10000, 1000)
y_perf = Int32.(rand(0:2, 10000))

# Time limit test
perf_config = create_validation_config(
    time_limit = 5.0f0,  # 5 second limit for this size
    verbose = true
)
results_perf = run_validation_suite(X_perf, y_perf, perf_config)

println("\nGPU time: $(round(results_perf.time_gpu, digits=2))s")
println("Speedup: $(round(results_perf.speedup, digits=1))x")
@test results_perf.time_gpu < 5.0

# Test 4: Tolerance tests
println("\n=== Test 4: Tolerance Tests ===")

# Test mutual information tolerance
mi_result = validate_mutual_information(X_small, y_small, config)
@test mi_result.passed
@test mi_result.max_error < config.mi_tolerance

# Test correlation tolerance
corr_result = validate_correlation_matrix(X_small, config)
@test corr_result.passed
@test corr_result.max_error < config.corr_tolerance

# Test variance tolerance
var_result = validate_variance_calculation(X_small, config)
@test var_result.passed
@test var_result.max_error < config.var_tolerance

# Test 5: Feature selection agreement
println("\n=== Test 5: Feature Selection Agreement ===")
selection_result = validate_feature_selection(X_corr, y_corr, 10, config)
@test selection_result.agreement >= 0.8
println("Feature selection agreement: $(round(selection_result.agreement * 100, digits=1))%")

# Summary
println("\n" * "="^60)
println("SKLEARN VALIDATION TEST SUMMARY")
println("="^60)
println("✓ All validation tests passed!")
println("✓ GPU implementation matches sklearn within tolerances")
println("✓ Performance targets met")
println("="^60)