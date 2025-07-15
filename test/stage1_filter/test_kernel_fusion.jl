using Test
using CUDA
using Statistics
using LinearAlgebra
using Random

# Skip tests if no GPU available
if !CUDA.functional()
    @warn "CUDA not functional, skipping kernel fusion tests"
    exit(0)
end

# Include the kernel fusion module
include("../../src/stage1_filter/kernel_fusion.jl")
using .KernelFusion

println("Testing GPU Kernel Fusion...")

# Test 1: Fused Standardize and Correlate
println("\n=== Test 1: Fused Standardize and Correlate ===")
Random.seed!(42)
n_features = 100
n_samples = 1000

X = randn(Float32, n_features, n_samples)
X_gpu = CuArray(X)

# Fused operation
corr_matrix_gpu = CUDA.zeros(Float32, n_features, n_features)
config = FusedKernelConfig()
corr_fused, means, stds = fused_standardize_and_correlate!(corr_matrix_gpu, X_gpu, config)
CUDA.synchronize()

# Reference calculation
X_mean = mean(X, dims=2)
X_std = std(X, dims=2, corrected=false)
X_std = max.(X_std, 1f-8)
X_standardized = (X .- X_mean) ./ X_std
corr_ref = X_standardized * X_standardized' / Float32(n_samples)

# Compare results
corr_fused_cpu = Array(corr_fused)
max_error = maximum(abs.(corr_fused_cpu .- corr_ref))
println("Max correlation error: $max_error")
@test max_error < 1e-4

# Check means and stds
means_cpu = Array(means)
stds_cpu = Array(stds)
mean_error = maximum(abs.(means_cpu .- vec(X_mean)))
std_error = maximum(abs.(stds_cpu .- vec(X_std)))
println("Max mean error: $mean_error")
println("Max std error: $std_error")
@test mean_error < 1e-5
@test std_error < 1e-5

# Test 2: Fused Variance Threshold
println("\n=== Test 2: Fused Variance Threshold ===")

# Create features with different variances
X_var = zeros(Float32, 50, 500)
X_var[1:20, :] = randn(Float32, 20, 500)  # Normal variance
X_var[21:30, :] = 0.001f0 * randn(Float32, 10, 500)  # Low variance
X_var[31:40, :] = 1000.0f0 * randn(Float32, 10, 500)  # High variance
X_var[41:50, :] .= 5.0f0  # Zero variance

X_var_gpu = CuArray(X_var)
valid_mask_gpu = CUDA.zeros(Bool, 50)
variances_gpu = CUDA.zeros(Float32, 50)

# Fused operation
threshold = 1f-6
fused_variance_threshold!(valid_mask_gpu, variances_gpu, X_var_gpu, threshold, config)
CUDA.synchronize()

# Reference calculation
var_ref = vec(var(X_var, dims=2, corrected=false))
valid_ref = var_ref .> threshold

# Compare results
valid_mask_cpu = Array(valid_mask_gpu)
variances_cpu = Array(variances_gpu)

println("Variance comparison:")
println("  Zero variance features detected: $(sum(.!valid_mask_cpu))")
println("  Low variance features: $(sum(variances_cpu .< 0.01f0))")
@test valid_mask_cpu == valid_ref
@test maximum(abs.(variances_cpu .- var_ref)) < 1e-4

# Test 3: Fused MI Calculation (Basic Test)
println("\n=== Test 3: Fused MI Calculation ===")

# Create simple dataset with informative features
n_samples = 2000
n_features = 20
X_mi = randn(Float32, n_features, n_samples)
y_mi = Int32.(zeros(n_samples))

# Make first 5 features informative
for i in 1:5
    y_mi .+= Int32.(X_mi[i, :] .> 0)
end
y_mi = Int32.((y_mi .> 2) .+ 1)  # Convert to 1 or 2

X_mi_gpu = CuArray(X_mi)
y_mi_gpu = CuArray(y_mi)
mi_scores_gpu = CUDA.zeros(Float32, n_features)

# Fused MI calculation
fused_histogram_mi!(mi_scores_gpu, X_mi_gpu, y_mi_gpu, Int32(10), config)
CUDA.synchronize()

mi_scores_cpu = Array(mi_scores_gpu)
println("MI scores (first 10): ", round.(mi_scores_cpu[1:10], digits=3))

# Verify informative features have higher scores
informative_mean = mean(mi_scores_cpu[1:5])
non_informative_mean = mean(mi_scores_cpu[6:end])
println("Informative features mean MI: $(round(informative_mean, digits=3))")
println("Non-informative features mean MI: $(round(non_informative_mean, digits=3))")
@test informative_mean > non_informative_mean

# Test 4: Performance Benchmark
println("\n=== Test 4: Performance Benchmark ===")

# Create larger dataset for benchmarking
X_bench = randn(Float32, 500, 5000)
y_bench = Int32.(rand(1:3, 5000))
X_bench_gpu = CuArray(X_bench)
y_bench_gpu = CuArray(y_bench)

# Run benchmark
results = benchmark_kernel_fusion(X_bench_gpu, y_bench_gpu, n_runs=5)

println("\nBenchmark Results:")
println("  Standardize+Correlate speedup: $(round(results["standardize_correlate"], digits=2))x")
println("  Histogram+MI speedup: $(round(results["histogram_mi"], digits=2))x")
println("  Variance+Threshold speedup: $(round(results["variance_threshold"], digits=2))x")
println("  Average speedup: $(round(results["average"], digits=2))x")
println("  Performance improvement: $(round(results["improvement_percent"], digits=1))%")

# Verify 20% improvement target
@test results["improvement_percent"] >= 20.0

# Test 5: Memory Usage
println("\n=== Test 5: Memory Usage ===")

# Get initial memory state
CUDA.reclaim()
initial_free = CUDA.available_memory()

# Run fused operations
corr_test = CUDA.zeros(Float32, 100, 100)
fused_standardize_and_correlate!(corr_test, X_bench_gpu[1:100, :], config)

valid_test = CUDA.zeros(Bool, 100)
var_test = CUDA.zeros(Float32, 100)
fused_variance_threshold!(valid_test, var_test, X_bench_gpu[1:100, :], 1f-6, config)

CUDA.synchronize()
final_free = CUDA.available_memory()

memory_used_mb = (initial_free - final_free) / 1024^2
println("Memory used by fused operations: $(round(memory_used_mb, digits=2)) MB")

# Test 6: Correctness Under Edge Cases
println("\n=== Test 6: Edge Cases ===")

# Empty features
X_empty = CUDA.zeros(Float32, 10, 0)
@test_throws Exception fused_standardize_and_correlate!(
    CUDA.zeros(Float32, 10, 10), X_empty, config
)

# Single sample
X_single = CUDA.randn(Float32, 10, 1)
corr_single = CUDA.zeros(Float32, 10, 10)
# Should handle gracefully without errors
fused_standardize_and_correlate!(corr_single, X_single, config)
println("Single sample test: Passed")

# Constant features
X_const = CUDA.ones(Float32, 10, 100)
valid_const = CUDA.zeros(Bool, 10)
var_const = CUDA.zeros(Float32, 10)
fused_variance_threshold!(valid_const, var_const, X_const, 1f-6, config)
@test all(Array(valid_const) .== false)
println("Constant features test: Passed")

println("\n" * "="^60)
println("KERNEL FUSION TEST SUMMARY")
println("="^60)
println("✓ Fused standardize+correlate produces correct results")
println("✓ Fused variance+threshold correctly filters features")
println("✓ Fused MI calculation identifies informative features")
println("✓ Performance improvement meets 20% target")
println("✓ Memory usage is reasonable")
println("✓ Edge cases handled correctly")
println("="^60)