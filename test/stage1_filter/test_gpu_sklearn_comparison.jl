using Test
using CUDA
using PyCall
using Statistics
using LinearAlgebra
using Random
using StatsBase

# Set Python path
ENV["PYTHON"] = ""  # Use Julia's conda Python

# Skip tests if no GPU available
if !CUDA.functional()
    @warn "CUDA not functional, skipping GPU validation tests"
    exit(0)
end

# Import sklearn modules
const sklearn_feature_selection = pyimport("sklearn.feature_selection")
const sklearn_preprocessing = pyimport("sklearn.preprocessing")
const np = pyimport("numpy")

println("Testing GPU vs Sklearn Feature Selection...")

# GPU implementation of mutual information (simplified version)
function compute_mutual_information_gpu(X_gpu::CuArray{Float32,2}, y_gpu::CuArray{Int32,1})
    n_features, n_samples = size(X_gpu)
    mi_scores = CUDA.zeros(Float32, n_features)
    
    # Get unique classes
    y_cpu = Array(y_gpu)
    classes = unique(y_cpu)
    n_classes = length(classes)
    
    # Compute class probabilities
    class_probs = Float32[]
    for c in classes
        push!(class_probs, sum(y_cpu .== c) / n_samples)
    end
    
    # For each feature, compute MI
    for i in 1:n_features
        feature = X_gpu[i, :]
        
        # Discretize feature into bins
        n_bins = 10
        feature_cpu = Array(feature)
        edges = range(minimum(feature_cpu), maximum(feature_cpu), length=n_bins+1)
        bins = searchsortedlast.(Ref(edges), feature_cpu)
        
        # Compute joint probability
        mi = 0.0f0
        for b in 1:n_bins
            for (ci, c) in enumerate(classes)
                # Joint probability P(X=b, Y=c)
                joint_count = sum((bins .== b) .& (y_cpu .== c))
                if joint_count > 0
                    p_joint = joint_count / n_samples
                    # Marginal probability P(X=b)
                    p_x = sum(bins .== b) / n_samples
                    # MI contribution
                    mi += p_joint * log2(p_joint / (p_x * class_probs[ci]))
                end
            end
        end
        
        mi_scores[i] = mi
    end
    
    return mi_scores
end

# GPU implementation of correlation matrix
function compute_correlation_matrix_gpu(X_gpu::CuArray{Float32,2})
    n_features, n_samples = size(X_gpu)
    
    # Standardize features
    X_mean = mean(X_gpu, dims=2)
    X_std = std(X_gpu, dims=2, corrected=false)
    X_std = max.(X_std, 1f-8)  # Avoid division by zero
    X_standardized = (X_gpu .- X_mean) ./ X_std
    
    # Compute correlation matrix
    corr_matrix = (X_standardized * X_standardized') / Float32(n_samples)
    
    return corr_matrix
end

# Test 1: Mutual Information Comparison
println("\n=== Test 1: Mutual Information Comparison ===")
Random.seed!(42)
n_samples = 5000
n_features = 100
n_informative = 20

# Create dataset with informative features
X = randn(Float32, n_samples, n_features)
# Make first n_informative features informative
y = Int32.(zeros(n_samples))
for i in 1:n_informative
    y .+= Int32.(X[:, i] .> 0)
end
y = Int32.(y .> (n_informative ÷ 2))

# GPU calculation
X_gpu = CuArray(X')  # GPU expects features × samples
y_gpu = CuArray(y)
t_gpu = @elapsed mi_scores_gpu = compute_mutual_information_gpu(X_gpu, y_gpu)
CUDA.synchronize()
mi_scores_gpu_cpu = Array(mi_scores_gpu)

# Sklearn calculation
t_sklearn = @elapsed mi_scores_sklearn = sklearn_feature_selection.mutual_info_classif(X, y)

println("GPU time: $(round(t_gpu*1000, digits=2))ms")
println("Sklearn time: $(round(t_sklearn*1000, digits=2))ms")
println("Speedup: $(round(t_sklearn/t_gpu, digits=1))x")

# Compare top features
top_k = 20
top_gpu = partialsortperm(mi_scores_gpu_cpu, 1:top_k, rev=true)
top_sklearn = partialsortperm(mi_scores_sklearn, 1:top_k, rev=true)
agreement = length(intersect(top_gpu, top_sklearn)) / top_k

println("Top $top_k features agreement: $(round(agreement*100, digits=1))%")
println("Informative features found by GPU: $(length(intersect(top_gpu, 1:n_informative)))/$n_informative")
println("Informative features found by sklearn: $(length(intersect(top_sklearn, 1:n_informative)))/$n_informative")

@test agreement >= 0.7  # At least 70% agreement on top features

# Test 2: Correlation Matrix Comparison
println("\n=== Test 2: Correlation Matrix Comparison ===")
X_small = randn(Float32, 1000, 50)
# Add some correlated features
for i in 1:10
    X_small[:, i+10] = X_small[:, i] + 0.1f0 * randn(Float32, 1000)
end

# GPU calculation
X_gpu_small = CuArray(X_small')
t_gpu_corr = @elapsed corr_matrix_gpu = compute_correlation_matrix_gpu(X_gpu_small)
CUDA.synchronize()
corr_matrix_gpu_cpu = Array(corr_matrix_gpu)

# Numpy calculation
t_numpy_corr = @elapsed corr_matrix_numpy = np.corrcoef(X_small', rowvar=true)
corr_matrix_numpy = Float32.(corr_matrix_numpy)

println("GPU time: $(round(t_gpu_corr*1000, digits=2))ms")
println("Numpy time: $(round(t_numpy_corr*1000, digits=2))ms")
println("Speedup: $(round(t_numpy_corr/t_gpu_corr, digits=1))x")

# Check accuracy
max_error = maximum(abs.(corr_matrix_gpu_cpu .- corr_matrix_numpy))
mean_error = mean(abs.(corr_matrix_gpu_cpu .- corr_matrix_numpy))
println("Max absolute error: $max_error")
println("Mean absolute error: $mean_error")

@test max_error < 1e-4

# Test 3: Feature Selection Pipeline
println("\n=== Test 3: End-to-End Feature Selection ===")
n_features_to_select = 50

# GPU feature selection (simplified)
function select_features_gpu(X_gpu, y_gpu, k)
    # Compute MI scores
    mi_scores = compute_mutual_information_gpu(X_gpu, y_gpu)
    
    # Compute variance
    var_scores = vec(var(X_gpu, dims=2, corrected=false))
    
    # Filter low variance features
    var_threshold = 1e-6
    valid_mask = var_scores .> var_threshold
    valid_indices = findall(valid_mask)
    
    # Select top k by MI score
    mi_scores_valid = mi_scores[valid_indices]
    if length(valid_indices) > k
        top_k_idx = partialsortperm(mi_scores_valid, 1:k, rev=true)
        selected = valid_indices[top_k_idx]
    else
        selected = valid_indices
    end
    
    return selected
end

# GPU selection
t_gpu_pipeline = @elapsed selected_gpu = select_features_gpu(X_gpu, y_gpu, n_features_to_select)
CUDA.synchronize()

# Sklearn selection
t_sklearn_pipeline = @elapsed begin
    # Variance threshold
    var_selector = sklearn_feature_selection.VarianceThreshold(threshold=1e-6)
    var_mask = var_selector.fit(X).get_support()
    
    # MI scores
    mi_scores_sk = sklearn_feature_selection.mutual_info_classif(X, y)
    
    # Apply variance filter and select top k
    valid_features = findall(var_mask)
    mi_scores_filtered = mi_scores_sk[valid_features]
    
    if length(valid_features) > n_features_to_select
        top_k_indices = partialsortperm(mi_scores_filtered, 1:n_features_to_select, rev=true)
        selected_sklearn = valid_features[top_k_indices]
    else
        selected_sklearn = valid_features
    end
end

println("GPU pipeline time: $(round(t_gpu_pipeline*1000, digits=2))ms")
println("Sklearn pipeline time: $(round(t_sklearn_pipeline*1000, digits=2))ms")
println("Speedup: $(round(t_sklearn_pipeline/t_gpu_pipeline, digits=1))x")

# Check agreement
selected_gpu_cpu = Array(selected_gpu)
pipeline_agreement = length(intersect(selected_gpu_cpu, selected_sklearn)) / n_features_to_select
println("Selected features agreement: $(round(pipeline_agreement*100, digits=1))%")

@test pipeline_agreement >= 0.6  # At least 60% agreement

# Test 4: Large Scale Performance
println("\n=== Test 4: Large Scale Performance ===")
X_large = randn(Float32, 10000, 1000)
y_large = Int32.(rand(0:2, 10000))

X_gpu_large = CuArray(X_large')
y_gpu_large = CuArray(y_large)

# Time GPU on large dataset
t_gpu_large = @elapsed begin
    mi_scores_large = compute_mutual_information_gpu(X_gpu_large, y_gpu_large)
    CUDA.synchronize()
end

println("GPU time for 10K×1K dataset: $(round(t_gpu_large, digits=2))s")
@test t_gpu_large < 5.0  # Should complete within 5 seconds

println("\n" * "="^60)
println("GPU VS SKLEARN VALIDATION SUMMARY")
println("="^60)
println("✓ Mutual information implementation validated")
println("✓ Correlation matrix computation accurate")
println("✓ Feature selection pipeline agreement acceptable")
println("✓ Large scale performance meets targets")
println("="^60)