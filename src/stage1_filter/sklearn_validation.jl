module SklearnValidation

using CUDA
using PyCall
using Statistics
using LinearAlgebra
using DataFrames
using CSV
using Random
using Test
using ProgressMeter

# Import our GPU modules
include("gpu_memory_layout.jl")
include("mutual_information.jl")
include("correlation_matrix.jl")
include("variance_calculation.jl")
include("feature_ranking.jl")

using .GPUMemoryLayout
using .MutualInformation
using .CorrelationMatrix
using .VarianceCalculation
using .FeatureRanking

# Import sklearn modules
const sklearn_feature_selection = PyNULL()
const sklearn_preprocessing = PyNULL()
const np = PyNULL()

function __init__()
    copy!(sklearn_feature_selection, pyimport("sklearn.feature_selection"))
    copy!(sklearn_preprocessing, pyimport("sklearn.preprocessing"))
    copy!(np, pyimport("numpy"))
end

"""
Configuration for validation tolerances
"""
struct ValidationConfig
    mi_tolerance::Float32        # Mutual information tolerance (relative)
    corr_tolerance::Float32      # Correlation tolerance (absolute)
    var_tolerance::Float32       # Variance tolerance (relative)
    time_limit::Float32          # Time limit in seconds
    verbose::Bool
end

"""
Create default validation configuration
"""
function create_validation_config(;
    mi_tolerance::Float32 = 0.01f0,      # 1% relative tolerance
    corr_tolerance::Float32 = 1f-5,      # Absolute tolerance
    var_tolerance::Float32 = 1f-6,      # Relative tolerance
    time_limit::Float32 = 30.0f0,       # 30 second target
    verbose::Bool = true
)
    return ValidationConfig(
        mi_tolerance,
        corr_tolerance,
        var_tolerance,
        time_limit,
        verbose
    )
end

"""
Validation results structure
"""
struct ValidationResults
    mi_passed::Bool
    mi_max_error::Float32
    mi_mean_error::Float32
    
    corr_passed::Bool
    corr_max_error::Float32
    corr_mean_error::Float32
    
    var_passed::Bool
    var_max_error::Float32
    var_mean_error::Float32
    
    selection_agreement::Float32
    time_gpu::Float32
    time_sklearn::Float32
    speedup::Float32
    
    all_passed::Bool
end

"""
Validate mutual information calculation against sklearn
"""
function validate_mutual_information(
    X::Matrix{Float32},
    y::Vector{Int32},
    config::ValidationConfig
)
    n_samples, n_features = size(X)
    
    if config.verbose
        println("\n=== Mutual Information Validation ===")
    end
    
    # GPU calculation
    gpu_config = MutualInformationConfig()
    X_gpu = CuArray(X')  # GPU expects features × samples
    y_gpu = CuArray(y)
    
    t_gpu = @elapsed begin
        mi_scores_gpu = compute_mutual_information(X_gpu, y_gpu, gpu_config)
    end
    CUDA.synchronize()
    
    mi_scores_gpu_cpu = Array(mi_scores_gpu)
    
    # Sklearn calculation
    t_sklearn = @elapsed begin
        mi_scores_sklearn = sklearn_feature_selection.mutual_info_classif(X, y)
    end
    
    # Convert to Float32 for comparison
    mi_scores_sklearn = Float32.(mi_scores_sklearn)
    
    # Calculate errors
    rel_errors = abs.(mi_scores_gpu_cpu .- mi_scores_sklearn) ./ (mi_scores_sklearn .+ 1f-8)
    max_error = maximum(rel_errors)
    mean_error = mean(rel_errors)
    
    passed = max_error <= config.mi_tolerance
    
    if config.verbose
        println("  GPU time: $(round(t_gpu*1000, digits=2))ms")
        println("  Sklearn time: $(round(t_sklearn*1000, digits=2))ms")
        println("  Speedup: $(round(t_sklearn/t_gpu, digits=1))x")
        println("  Max relative error: $(round(max_error*100, digits=3))%")
        println("  Mean relative error: $(round(mean_error*100, digits=3))%")
        println("  Passed: $passed")
    end
    
    return (
        passed = passed,
        max_error = max_error,
        mean_error = mean_error,
        time_gpu = t_gpu,
        time_sklearn = t_sklearn,
        gpu_scores = mi_scores_gpu_cpu,
        sklearn_scores = mi_scores_sklearn
    )
end

"""
Validate correlation matrix calculation against numpy
"""
function validate_correlation_matrix(
    X::Matrix{Float32},
    config::ValidationConfig
)
    n_samples, n_features = size(X)
    
    if config.verbose
        println("\n=== Correlation Matrix Validation ===")
    end
    
    # GPU calculation
    corr_config = CorrelationConfig()
    X_gpu = CuArray(X')  # GPU expects features × samples
    
    t_gpu = @elapsed begin
        corr_matrix_gpu = compute_correlation_matrix(X_gpu, corr_config)
    end
    CUDA.synchronize()
    
    corr_matrix_gpu_cpu = Array(corr_matrix_gpu)
    
    # Numpy calculation
    t_numpy = @elapsed begin
        corr_matrix_numpy = np.corrcoef(X', rowvar=true)
    end
    
    # Convert to Float32 for comparison
    corr_matrix_numpy = Float32.(corr_matrix_numpy)
    
    # Calculate errors
    abs_errors = abs.(corr_matrix_gpu_cpu .- corr_matrix_numpy)
    max_error = maximum(abs_errors)
    mean_error = mean(abs_errors)
    
    passed = max_error <= config.corr_tolerance
    
    if config.verbose
        println("  GPU time: $(round(t_gpu*1000, digits=2))ms")
        println("  Numpy time: $(round(t_numpy*1000, digits=2))ms")
        println("  Speedup: $(round(t_numpy/t_gpu, digits=1))x")
        println("  Max absolute error: $max_error")
        println("  Mean absolute error: $mean_error")
        println("  Passed: $passed")
    end
    
    return (
        passed = passed,
        max_error = max_error,
        mean_error = mean_error,
        time_gpu = t_gpu,
        time_numpy = t_numpy,
        gpu_matrix = corr_matrix_gpu_cpu,
        numpy_matrix = corr_matrix_numpy
    )
end

"""
Validate variance calculation
"""
function validate_variance_calculation(
    X::Matrix{Float32},
    config::ValidationConfig
)
    n_samples, n_features = size(X)
    
    if config.verbose
        println("\n=== Variance Calculation Validation ===")
    end
    
    # GPU calculation
    X_gpu = CuArray(X')  # GPU expects features × samples
    
    t_gpu = @elapsed begin
        variances_gpu = compute_variance(X_gpu)
    end
    CUDA.synchronize()
    
    variances_gpu_cpu = Array(variances_gpu)
    
    # CPU calculation
    t_cpu = @elapsed begin
        variances_cpu = vec(var(X, dims=1))
    end
    
    # Calculate errors
    rel_errors = abs.(variances_gpu_cpu .- variances_cpu) ./ (variances_cpu .+ 1f-8)
    max_error = maximum(rel_errors)
    mean_error = mean(rel_errors)
    
    passed = max_error <= config.var_tolerance
    
    if config.verbose
        println("  GPU time: $(round(t_gpu*1000, digits=2))ms")
        println("  CPU time: $(round(t_cpu*1000, digits=2))ms")
        println("  Speedup: $(round(t_cpu/t_gpu, digits=1))x")
        println("  Max relative error: $max_error")
        println("  Mean relative error: $mean_error")
        println("  Passed: $passed")
    end
    
    return (
        passed = passed,
        max_error = max_error,
        mean_error = mean_error,
        time_gpu = t_gpu,
        time_cpu = t_cpu,
        gpu_variances = variances_gpu_cpu,
        cpu_variances = variances_cpu
    )
end

"""
Validate end-to-end feature selection
"""
function validate_feature_selection(
    X::Matrix{Float32},
    y::Vector{Int32},
    n_features_to_select::Int,
    config::ValidationConfig
)
    n_samples, n_features = size(X)
    
    if config.verbose
        println("\n=== End-to-End Feature Selection Validation ===")
    end
    
    # GPU feature selection
    t_gpu_total = @elapsed begin
        # Prepare GPU data
        X_gpu = CuArray(X')
        y_gpu = CuArray(y)
        
        # Run complete pipeline
        ranking_config = RankingConfig(
            n_features_to_select = n_features_to_select,
            mi_threshold = 0.0f0,
            correlation_threshold = 0.95f0,
            variance_threshold = 1f-6
        )
        
        selected_features_gpu = select_features(X_gpu, y_gpu, ranking_config)
    end
    CUDA.synchronize()
    
    selected_gpu = sort(Array(selected_features_gpu))
    
    # Sklearn feature selection
    t_sklearn_total = @elapsed begin
        # Mutual information scores
        mi_scores = sklearn_feature_selection.mutual_info_classif(X, y)
        
        # Variance filter
        var_selector = sklearn_feature_selection.VarianceThreshold(threshold=1e-6)
        var_mask = var_selector.fit(X).get_support()
        
        # Apply variance filter
        valid_features = findall(var_mask)
        mi_scores_filtered = mi_scores[valid_features]
        
        # Select top k features
        if length(valid_features) > n_features_to_select
            top_k_indices = partialsortperm(mi_scores_filtered, 1:n_features_to_select, rev=true)
            selected_sklearn = sort(valid_features[top_k_indices])
        else
            selected_sklearn = sort(valid_features)
        end
    end
    
    # Calculate agreement
    intersection = length(intersect(selected_gpu, selected_sklearn))
    agreement = intersection / n_features_to_select
    
    time_within_limit = t_gpu_total <= config.time_limit
    
    if config.verbose
        println("  GPU total time: $(round(t_gpu_total, digits=2))s")
        println("  Sklearn total time: $(round(t_sklearn_total, digits=2))s")
        println("  Speedup: $(round(t_sklearn_total/t_gpu_total, digits=1))x")
        println("  Selected agreement: $(round(agreement*100, digits=1))%")
        println("  Time within limit: $time_within_limit")
        
        if agreement < 0.8
            println("\n  GPU selected: $selected_gpu")
            println("  Sklearn selected: $selected_sklearn")
            println("  Intersection: $(intersect(selected_gpu, selected_sklearn))")
        end
    end
    
    return (
        agreement = agreement,
        time_gpu = t_gpu_total,
        time_sklearn = t_sklearn_total,
        time_within_limit = time_within_limit,
        gpu_features = selected_gpu,
        sklearn_features = selected_sklearn
    )
end

"""
Run comprehensive validation suite
"""
function run_validation_suite(
    X::Matrix{Float32},
    y::Vector{Int32},
    config::ValidationConfig = create_validation_config()
)
    n_samples, n_features = size(X)
    n_features_to_select = min(500, div(n_features, 10))
    
    println("\n" * "="^60)
    println("Running Sklearn Validation Suite")
    println("Dataset: $n_samples samples × $n_features features")
    println("Target features: $n_features_to_select")
    println("="^60)
    
    # Run individual validations
    mi_results = validate_mutual_information(X, y, config)
    corr_results = validate_correlation_matrix(X, config)
    var_results = validate_variance_calculation(X, config)
    selection_results = validate_feature_selection(X, y, n_features_to_select, config)
    
    # Aggregate results
    all_passed = mi_results.passed && 
                 corr_results.passed && 
                 var_results.passed && 
                 selection_results.time_within_limit &&
                 selection_results.agreement >= 0.8
    
    results = ValidationResults(
        mi_results.passed,
        mi_results.max_error,
        mi_results.mean_error,
        corr_results.passed,
        corr_results.max_error,
        corr_results.mean_error,
        var_results.passed,
        var_results.max_error,
        var_results.mean_error,
        selection_results.agreement,
        selection_results.time_gpu,
        selection_results.time_sklearn,
        selection_results.time_sklearn / selection_results.time_gpu,
        all_passed
    )
    
    # Summary
    println("\n" * "="^60)
    println("VALIDATION SUMMARY")
    println("="^60)
    println("Mutual Information: $(mi_results.passed ? "✓ PASSED" : "✗ FAILED")")
    println("Correlation Matrix: $(corr_results.passed ? "✓ PASSED" : "✗ FAILED")")
    println("Variance Calculation: $(var_results.passed ? "✓ PASSED" : "✗ FAILED")")
    println("Feature Selection Agreement: $(round(selection_results.agreement*100, digits=1))%")
    println("Time Constraint (<$(config.time_limit)s): $(selection_results.time_within_limit ? "✓ PASSED" : "✗ FAILED")")
    println("\nOverall: $(all_passed ? "✓ ALL TESTS PASSED" : "✗ SOME TESTS FAILED")")
    println("="^60)
    
    return results
end

"""
Create standard test datasets
"""
function create_test_datasets()
    datasets = Dict{String, Tuple{Matrix{Float32}, Vector{Int32}}}()
    
    # Small dataset for quick testing
    Random.seed!(42)
    X_small = randn(Float32, 1000, 100)
    y_small = Int32.(rand(0:1, 1000))
    datasets["small"] = (X_small, y_small)
    
    # Medium dataset
    X_medium = randn(Float32, 10000, 1000)
    y_medium = Int32.(rand(0:2, 10000))
    datasets["medium"] = (X_medium, y_medium)
    
    # Large dataset (closer to production)
    X_large = randn(Float32, 50000, 5000)
    y_large = Int32.(rand(0:4, 50000))
    datasets["large"] = (X_large, y_large)
    
    # Dataset with correlated features
    X_corr = randn(Float32, 5000, 500)
    # Add some highly correlated features
    for i in 1:50
        X_corr[:, i+50] = X_corr[:, i] + 0.1f0 * randn(Float32, 5000)
    end
    y_corr = Int32.(X_corr[:, 1] .> 0)
    datasets["correlated"] = (X_corr, y_corr)
    
    return datasets
end

"""
Run performance regression tests
"""
function run_performance_regression_tests(config::ValidationConfig = create_validation_config())
    datasets = create_test_datasets()
    
    println("\n" * "="^60)
    println("PERFORMANCE REGRESSION TESTS")
    println("="^60)
    
    results = Dict{String, NamedTuple}()
    
    for (name, (X, y)) in datasets
        println("\nTesting dataset: $name")
        println("-"^40)
        
        # Run validation
        result = run_validation_suite(X, y, config)
        
        # Store results
        results[name] = (
            passed = result.all_passed,
            gpu_time = result.time_gpu,
            speedup = result.speedup,
            agreement = result.selection_agreement
        )
        
        # Check performance regression
        if name == "large" && result.time_gpu > 30.0
            @warn "Performance regression detected! Large dataset took $(round(result.time_gpu, digits=2))s (target: <30s)"
        end
    end
    
    return results
end

# Export functions
export ValidationConfig, create_validation_config
export ValidationResults
export validate_mutual_information, validate_correlation_matrix
export validate_variance_calculation, validate_feature_selection
export run_validation_suite, create_test_datasets
export run_performance_regression_tests

end # module SklearnValidation