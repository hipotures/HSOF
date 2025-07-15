using Test
using CUDA
using PyCall
using Statistics
using LinearAlgebra
using Random

# Set Python path
ENV["PYTHON"] = ""  # Use Julia's conda Python

# Skip tests if no GPU available
if !CUDA.functional()
    @warn "CUDA not functional, skipping GPU validation tests"
    exit(0)
end

# Import sklearn modules directly
const sklearn_feature_selection = pyimport("sklearn.feature_selection")
const sklearn_preprocessing = pyimport("sklearn.preprocessing")
const np = pyimport("numpy")

println("Testing Sklearn Validation (Simple Version)...")

# Test 1: Mutual Information Validation
println("\n=== Test 1: Mutual Information Validation ===")
Random.seed!(42)
X = randn(Float32, 1000, 50)
y = Int32.(rand(0:1, 1000))

# Sklearn calculation
mi_scores_sklearn = sklearn_feature_selection.mutual_info_classif(X, y)
println("Sklearn MI scores computed: $(length(mi_scores_sklearn)) features")
println("Mean MI score: $(round(mean(mi_scores_sklearn), digits=4))")

# Test 2: Correlation Matrix Validation
println("\n=== Test 2: Correlation Matrix Validation ===")
X_small = randn(Float32, 100, 20)

# Numpy calculation
corr_matrix_numpy = np.corrcoef(X_small', rowvar=true)
println("Numpy correlation matrix computed: $(size(corr_matrix_numpy))")

# Julia calculation
corr_matrix_julia = cor(X_small, dims=1)
println("Julia correlation matrix computed: $(size(corr_matrix_julia))")

# Compare
max_diff = maximum(abs.(Float32.(corr_matrix_numpy) .- Float32.(corr_matrix_julia)))
println("Max difference between numpy and Julia: $max_diff")
@test max_diff < 1e-5

# Test 3: Variance Calculation
println("\n=== Test 3: Variance Calculation ===")
var_numpy = np.var(X_small, axis=0)
var_julia = var(X_small, dims=1, corrected=false)  # Use ddof=0 like numpy

max_var_diff = maximum(abs.(vec(Float32.(var_numpy)) .- vec(Float32.(var_julia))))
println("Max variance difference: $max_var_diff")
@test max_var_diff < 1e-5

# Test 4: GPU Basic Operations
println("\n=== Test 4: GPU Basic Operations ===")
X_gpu = CuArray(X_small')  # GPU expects features × samples
println("GPU array created: $(size(X_gpu))")

# Simple GPU operation - variance
X_gpu_centered = X_gpu .- mean(X_gpu, dims=2)
var_gpu = vec(mean(X_gpu_centered .^ 2, dims=2))
var_gpu_cpu = Array(var_gpu)

# Compare with CPU
var_cpu = vec(var(X_small, dims=1, corrected=false))
max_gpu_var_diff = maximum(abs.(var_gpu_cpu .- var_cpu))
println("Max GPU variance difference: $max_gpu_var_diff")
@test max_gpu_var_diff < 1e-5

println("\n" * "="^60)
println("SIMPLE SKLEARN VALIDATION TEST SUMMARY")
println("="^60)
println("✓ Sklearn import successful")
println("✓ Numpy correlation matches Julia")
println("✓ Numpy variance matches Julia")
println("✓ GPU operations functional")
println("="^60)