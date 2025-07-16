using CUDA
using Statistics
using LinearAlgebra
using Random

# Test simple correlation calculation
Random.seed!(42)
n_features = 5
n_samples = 100

# Generate test data
test_data = randn(Float32, n_features, n_samples)

# CPU calculation
expected_corr = cor(test_data')
println("Expected correlation (first 3x3):")
display(expected_corr[1:3, 1:3])

# GPU calculation mimicking batch processing
sum_x = sum(test_data, dims=2)
sum_xx = test_data * test_data'
n = n_samples

# Convert to means
means = sum_x ./ n
means_outer = means * means'

# Compute covariance
cov_matrix = (sum_xx ./ n) .- means_outer

# Get standard deviations
std_devs = sqrt.(diag(cov_matrix))
std_outer = std_devs * std_devs'

# Compute correlation
computed_corr = cov_matrix ./ std_outer

println("\nComputed correlation (first 3x3):")
display(computed_corr[1:3, 1:3])

println("\nMax difference: ", maximum(abs.(computed_corr .- expected_corr)))