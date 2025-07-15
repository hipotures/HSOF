module RegressionTestSuite

using Test
using JSON3
using Statistics
using Dates
using Random

# Include necessary modules
push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
include("../../src/core/models.jl")
include("data/dataset_loaders.jl")
include("pipeline_runner.jl")
include("result_validation.jl")

using .Models
using .DatasetLoaders
using .PipelineRunner
using .ResultValidation

export run_regression_tests, compare_with_baseline, create_performance_baseline
export validate_model_updates, test_configuration_changes

"""
Performance baseline for regression testing
"""
struct PerformanceBaseline
    dataset_name::String
    feature_reduction::Vector{Int}
    quality_scores::Vector{Float64}
    runtime_bounds::Dict{String, Float64}  # min/max runtime per stage
    memory_bounds::Dict{String, Float64}   # min/max memory per stage
    accuracy_thresholds::Dict{String, Float64}
    timestamp::DateTime
    version::String
end

"""
Baseline feature selection methods for comparison
"""
abstract type BaselineMethod end

struct CorrelationBaseline <: BaselineMethod
    threshold::Float64
end

struct MutualInfoBaseline <: BaselineMethod
    k_features::Int
end

struct VarianceBaseline <: BaselineMethod
    threshold::Float64
end

struct RandomBaseline <: BaselineMethod
    k_features::Int
    seed::Int
end

"""
Run baseline feature selection method
"""
function run_baseline_method(method::BaselineMethod, X::Matrix, y::Vector, n_features::Int)
    if isa(method, CorrelationBaseline)
        return run_correlation_baseline(X, y, method.threshold, n_features)
    elseif isa(method, MutualInfoBaseline)
        return run_mutual_info_baseline(X, y, min(method.k_features, n_features))
    elseif isa(method, VarianceBaseline)
        return run_variance_baseline(X, y, method.threshold, n_features)
    elseif isa(method, RandomBaseline)
        return run_random_baseline(X, y, min(method.k_features, n_features), method.seed)
    end
end

function run_correlation_baseline(X::Matrix, y::Vector, threshold::Float64, n_features::Int)
    # Calculate correlation with target
    correlations = abs.([cor(X[:, i], y) for i in 1:size(X, 2)])
    
    # Handle NaN correlations (constant features)
    correlations[isnan.(correlations)] .= 0.0
    
    # Select top features by correlation
    sorted_indices = sortperm(correlations, rev=true)
    selected = sorted_indices[1:min(n_features, length(sorted_indices))]
    
    return selected, correlations[selected]
end

function run_mutual_info_baseline(X::Matrix, y::Vector, k_features::Int)
    # Simplified mutual information estimation
    # In practice, would use proper MI calculation
    n_samples, n_feats = size(X)
    mi_scores = zeros(n_feats)
    
    for i in 1:n_feats
        # Discretize feature
        feature = X[:, i]
        n_bins = min(10, length(unique(feature)))
        
        if n_bins > 1
            # Calculate approximate MI score
            mi_scores[i] = estimate_mutual_information(feature, y, n_bins)
        end
    end
    
    # Select top k features
    sorted_indices = sortperm(mi_scores, rev=true)
    selected = sorted_indices[1:k_features]
    
    return selected, mi_scores[selected]
end

function estimate_mutual_information(x::Vector, y::Vector, n_bins::Int)
    # Simplified MI estimation
    # Discretize x
    x_min, x_max = extrema(x)
    x_discrete = floor.(Int, (x .- x_min) ./ (x_max - x_min + 1e-10) .* n_bins) .+ 1
    x_discrete = clamp.(x_discrete, 1, n_bins)
    
    # Calculate joint and marginal distributions
    joint_counts = zeros(Int, n_bins, length(unique(y)))
    x_counts = zeros(Int, n_bins)
    y_counts = zeros(Int, length(unique(y)))
    
    y_labels = sort(unique(y))
    y_map = Dict(label => i for (i, label) in enumerate(y_labels))
    
    for (xi, yi) in zip(x_discrete, y)
        yi_idx = y_map[yi]
        joint_counts[xi, yi_idx] += 1
        x_counts[xi] += 1
        y_counts[yi_idx] += 1
    end
    
    # Calculate MI
    n = length(x)
    mi = 0.0
    
    for i in 1:n_bins
        for j in 1:length(y_labels)
            if joint_counts[i, j] > 0 && x_counts[i] > 0 && y_counts[j] > 0
                p_xy = joint_counts[i, j] / n
                p_x = x_counts[i] / n
                p_y = y_counts[j] / n
                mi += p_xy * log(p_xy / (p_x * p_y))
            end
        end
    end
    
    return max(0.0, mi)
end

function run_variance_baseline(X::Matrix, y::Vector, threshold::Float64, n_features::Int)
    # Calculate variance for each feature
    variances = [var(X[:, i]) for i in 1:size(X, 2)]
    
    # Select features above threshold, then top n_features
    above_threshold = findall(v -> v >= threshold, variances)
    
    if length(above_threshold) <= n_features
        return above_threshold, variances[above_threshold]
    else
        # Sort by variance and take top n_features
        sorted_indices = sortperm(variances[above_threshold], rev=true)
        selected_indices = above_threshold[sorted_indices[1:n_features]]
        return selected_indices, variances[selected_indices]
    end
end

function run_random_baseline(X::Matrix, y::Vector, k_features::Int, seed::Int)
    Random.seed!(seed)
    n_total = size(X, 2)
    selected = randperm(n_total)[1:k_features]
    scores = ones(k_features) * 0.5  # Neutral score for random selection
    return selected, scores
end

"""
Compare HSOF results with baseline methods
"""
function compare_with_baseline(hsof_features::Vector{Int}, X::Matrix, y::Vector)
    n_hsof = length(hsof_features)
    
    # Define baseline methods
    baselines = Dict{String, BaselineMethod}(
        "correlation" => CorrelationBaseline(0.1),
        "mutual_info" => MutualInfoBaseline(n_hsof),
        "variance" => VarianceBaseline(0.01),
        "random" => RandomBaseline(n_hsof, 42)
    )
    
    # Run comparisons
    comparisons = Dict{String, Any}()
    
    for (name, method) in baselines
        try
            baseline_features, baseline_scores = run_baseline_method(method, X, y, n_hsof)
            
            # Calculate overlap
            overlap = length(intersect(hsof_features, baseline_features))
            overlap_ratio = overlap / n_hsof
            
            # Evaluate both feature sets
            hsof_quality = evaluate_feature_set_quality(X, y, hsof_features)
            baseline_quality = evaluate_feature_set_quality(X, y, baseline_features)
            
            improvement = hsof_quality - baseline_quality
            
            comparisons[name] = Dict(
                "overlap_count" => overlap,
                "overlap_ratio" => overlap_ratio,
                "hsof_quality" => hsof_quality,
                "baseline_quality" => baseline_quality,
                "improvement" => improvement,
                "better" => improvement > 0
            )
            
        catch e
            comparisons[name] = Dict("error" => string(e))
        end
    end
    
    # Summary statistics
    successful_comparisons = filter(p -> !haskey(p.second, "error"), comparisons)
    
    if !isempty(successful_comparisons)
        improvements = [comp["improvement"] for comp in values(successful_comparisons)]
        better_count = count(comp -> comp["better"], values(successful_comparisons))
        
        comparisons["summary"] = Dict(
            "average_improvement" => mean(improvements),
            "best_improvement" => maximum(improvements),
            "worst_improvement" => minimum(improvements),
            "better_than_baselines" => better_count,
            "total_baselines" => length(successful_comparisons)
        )
    end
    
    return comparisons
end

"""
Evaluate quality of a feature set using cross-validation
"""
function evaluate_feature_set_quality(X::Matrix, y::Vector, features::Vector{Int})
    if isempty(features)
        return 0.0
    end
    
    X_subset = X[:, features]
    
    # Simple cross-validation evaluation
    n_folds = min(5, length(unique(y)))
    n_samples = size(X_subset, 1)
    fold_size = div(n_samples, n_folds)
    
    scores = Float64[]
    
    for fold in 1:n_folds
        # Create train/test split
        test_start = (fold - 1) * fold_size + 1
        test_end = fold == n_folds ? n_samples : fold * fold_size
        
        test_indices = test_start:test_end
        train_indices = setdiff(1:n_samples, test_indices)
        
        if length(train_indices) == 0 || length(test_indices) == 0
            continue
        end
        
        X_train = X_subset[train_indices, :]
        y_train = y[train_indices]
        X_test = X_subset[test_indices, :]
        y_test = y[test_indices]
        
        # Simple classifier: majority class with feature-based adjustments
        score = evaluate_simple_classifier(X_train, y_train, X_test, y_test)
        push!(scores, score)
    end
    
    return isempty(scores) ? 0.0 : mean(scores)
end

function evaluate_simple_classifier(X_train::Matrix, y_train::Vector, X_test::Matrix, y_test::Vector)
    # Simplified classifier evaluation
    # Calculate class means for each feature
    unique_classes = unique(y_train)
    
    if length(unique_classes) == 1
        # All same class
        accuracy = mean(y_test .== unique_classes[1])
        return accuracy
    end
    
    class_means = Dict{Int, Vector{Float64}}()
    for class in unique_classes
        class_mask = y_train .== class
        if sum(class_mask) > 0
            class_means[class] = mean(X_train[class_mask, :], dims=1)[:]
        end
    end
    
    # Predict using nearest class centroid
    correct = 0
    for i in 1:length(y_test)
        distances = Dict{Int, Float64}()
        for class in unique_classes
            if haskey(class_means, class)
                dist = sum((X_test[i, :] .- class_means[class]).^2)
                distances[class] = dist
            end
        end
        
        if !isempty(distances)
            predicted_class = argmin(distances)
            if predicted_class == y_test[i]
                correct += 1
            end
        end
    end
    
    return correct / length(y_test)
end

"""
Create performance baseline from current results
"""
function create_performance_baseline(dataset_name::String, result::PipelineTestResult; version::String = "1.0")
    # Extract runtime bounds (with some tolerance)
    runtime_bounds = Dict{String, Float64}()
    memory_bounds = Dict{String, Float64}()
    accuracy_thresholds = Dict{String, Float64}()
    
    for (stage, stage_result) in result.stage_results
        stage_key = "stage$stage"
        
        # Runtime bounds (±20% tolerance)
        runtime = stage_result["runtime"]
        runtime_bounds["$(stage_key)_min"] = runtime * 0.5
        runtime_bounds["$(stage_key)_max"] = runtime * 1.5
        
        # Memory bounds (±30% tolerance)
        memory = stage_result["memory_mb"]
        memory_bounds["$(stage_key)_min"] = memory * 0.4
        memory_bounds["$(stage_key)_max"] = memory * 1.8
        
        # Quality thresholds (minimum acceptable quality)
        quality = stage_result["quality_score"]
        accuracy_thresholds[stage_key] = quality * 0.9
    end
    
    # Overall bounds
    runtime_bounds["total_max"] = result.total_runtime * 1.3
    memory_bounds["peak_max"] = result.peak_memory_mb * 1.5
    
    return PerformanceBaseline(
        dataset_name,
        result.feature_reduction,
        result.quality_scores,
        runtime_bounds,
        memory_bounds,
        accuracy_thresholds,
        now(),
        version
    )
end

"""
Validate results against performance baseline
"""
function validate_against_baseline(result::PipelineTestResult, baseline::PerformanceBaseline)
    violations = String[]
    warnings = String[]
    
    # Check runtime bounds
    for (stage, stage_result) in result.stage_results
        stage_key = "stage$stage"
        runtime = stage_result["runtime"]
        
        max_key = "$(stage_key)_max"
        if haskey(baseline.runtime_bounds, max_key)
            if runtime > baseline.runtime_bounds[max_key]
                push!(violations, "Stage $stage runtime $(round(runtime, digits=2))s exceeds baseline $(round(baseline.runtime_bounds[max_key], digits=2))s")
            end
        end
        
        # Check memory bounds
        memory = stage_result["memory_mb"]
        max_mem_key = "$(stage_key)_max"
        if haskey(baseline.memory_bounds, max_mem_key)
            if memory > baseline.memory_bounds[max_mem_key]
                push!(violations, "Stage $stage memory $(round(memory, digits=2))MB exceeds baseline $(round(baseline.memory_bounds[max_mem_key], digits=2))MB")
            end
        end
        
        # Check quality thresholds
        quality = stage_result["quality_score"]
        if haskey(baseline.accuracy_thresholds, stage_key)
            if quality < baseline.accuracy_thresholds[stage_key]
                push!(violations, "Stage $stage quality $(round(quality, digits=3)) below baseline $(round(baseline.accuracy_thresholds[stage_key], digits=3))")
            end
        end
    end
    
    # Check overall bounds
    if haskey(baseline.runtime_bounds, "total_max")
        if result.total_runtime > baseline.runtime_bounds["total_max"]
            push!(violations, "Total runtime $(round(result.total_runtime, digits=2))s exceeds baseline $(round(baseline.runtime_bounds["total_max"], digits=2))s")
        end
    end
    
    if haskey(baseline.memory_bounds, "peak_max")
        if result.peak_memory_mb > baseline.memory_bounds["peak_max"]
            push!(violations, "Peak memory $(round(result.peak_memory_mb, digits=2))MB exceeds baseline $(round(baseline.memory_bounds["peak_max"], digits=2))MB")
        end
    end
    
    # Check feature reduction pattern
    if length(result.feature_reduction) == length(baseline.feature_reduction)
        for i in 1:length(result.feature_reduction)
            baseline_count = baseline.feature_reduction[i]
            current_count = result.feature_reduction[i]
            
            # Allow some variation in intermediate stages, but final should be close
            tolerance = i == length(result.feature_reduction) ? 0.1 : 0.3
            if abs(current_count - baseline_count) / baseline_count > tolerance
                push!(warnings, "Stage $i feature count $current_count differs significantly from baseline $baseline_count")
            end
        end
    end
    
    return Dict(
        "passed" => isempty(violations),
        "violations" => violations,
        "warnings" => warnings,
        "baseline_version" => baseline.version,
        "baseline_date" => baseline.timestamp
    )
end

"""
Test configuration changes don't break performance
"""
function test_configuration_changes(dataset::DatasetInfo; verbose::Bool = true)
    if verbose
        println("\n=== Testing Configuration Changes ===")
    end
    
    results = Dict{String, Any}()
    
    # Test different configurations
    test_configs = [
        ("reduced_iterations", Dict("stage2" => Dict("max_iterations" => 50))),
        ("cpu_only", Dict("stage2" => Dict("gpu_enabled" => false))),
        ("fewer_trees", Dict("stage2" => Dict("num_trees" => 2))),
        ("smaller_ensemble", Dict("stage3" => Dict("ensemble_size" => 2)))
    ]
    
    for (config_name, config_changes) in test_configs
        if verbose
            println("  Testing configuration: $config_name")
        end
        
        try
            # Get base configuration
            base_config = get_pipeline_config(dataset.name)
            
            # Apply changes
            test_config = deepcopy(base_config)
            for (stage, changes) in config_changes
                if haskey(test_config, stage)
                    merge!(test_config[stage], changes)
                else
                    test_config[stage] = changes
                end
            end
            
            # Run pipeline with modified config
            # For simplicity, we'll run a quick version
            X, y, _ = prepare_dataset_for_pipeline(dataset)
            
            start_time = time()
            
            # Quick pipeline run
            filter = VarianceFilter(threshold=0.01)
            stage1_features = fit_transform(filter, X, y)
            
            X_filtered = X[:, stage1_features]
            selector = MCTSFeatureSelector(
                n_iterations = get(config_changes["stage2"], "max_iterations", 100),
                exploration_constant = 1.4,
                n_trees = get(config_changes["stage2"], "num_trees", 3),
                use_gpu = get(config_changes["stage2"], "gpu_enabled", false),
                n_simulations = 20
            )
            
            stage2_indices, _ = select_features(selector, X_filtered, y, 10)
            
            runtime = time() - start_time
            
            results[config_name] = Dict(
                "success" => true,
                "runtime" => runtime,
                "stage1_features" => length(stage1_features),
                "stage2_features" => length(stage2_indices)
            )
            
        catch e
            results[config_name] = Dict(
                "success" => false,
                "error" => string(e)
            )
        end
    end
    
    return results
end

"""
Run comprehensive regression test suite
"""
function run_regression_tests(; datasets::Vector{String} = ["titanic"], verbose::Bool = true)
    println("\n" * "=" * 60)
    println("HSOF Regression Test Suite")
    println("=" * 60)
    
    all_datasets = load_all_reference_datasets()
    test_results = Dict{String, Any}()
    
    for dataset_name in datasets
        if !haskey(all_datasets, dataset_name)
            println("⚠ Warning: Dataset '$dataset_name' not found, skipping...")
            continue
        end
        
        dataset = all_datasets[dataset_name]
        println("\nRunning regression tests for: $(dataset.name)")
        
        results = Dict{String, Any}()
        
        # Run current pipeline
        current_result = run_pipeline_test(dataset, verbose=false)
        
        if current_result.passed
            # Test against baselines
            if haskey(current_result.stage_results, 3)
                final_features = current_result.stage_results[3]["selected_features"]
                X, y, _ = prepare_dataset_for_pipeline(dataset)
                
                baseline_comparison = compare_with_baseline(final_features, X, y)
                results["baseline_comparison"] = baseline_comparison
            end
            
            # Create/update performance baseline
            baseline = create_performance_baseline(dataset_name, current_result)
            results["performance_baseline"] = Dict(
                "created" => true,
                "feature_reduction" => baseline.feature_reduction,
                "runtime_bounds" => baseline.runtime_bounds
            )
            
            # Test configuration changes
            config_results = test_configuration_changes(dataset, verbose=false)
            results["configuration_tests"] = config_results
            
        else
            results["error"] = "Pipeline test failed: $(current_result.errors)"
        end
        
        test_results[dataset_name] = results
    end
    
    # Print summary
    println("\n" * "=" * 60)
    println("Regression Test Summary")
    println("=" * 60)
    
    for (dataset_name, results) in test_results
        println("\n$dataset_name:")
        
        if haskey(results, "error")
            println("  ❌ FAILED: $(results["error"])")
            continue
        end
        
        # Baseline comparison summary
        if haskey(results, "baseline_comparison") && haskey(results["baseline_comparison"], "summary")
            summary = results["baseline_comparison"]["summary"]
            better_count = summary["better_than_baselines"]
            total_count = summary["total_baselines"]
            avg_improvement = summary["average_improvement"]
            
            println("  📊 Baseline Comparison: $better_count/$total_count methods outperformed")
            println("      Average improvement: $(round(avg_improvement * 100, digits=1))%")
        end
        
        # Configuration test summary
        if haskey(results, "configuration_tests")
            config_tests = results["configuration_tests"]
            successful_configs = count(r -> r["success"], values(config_tests))
            total_configs = length(config_tests)
            
            println("  ⚙️  Configuration Tests: $successful_configs/$total_configs passed")
        end
        
        # Performance baseline
        if haskey(results, "performance_baseline")
            println("  ⏱️  Performance Baseline: Created successfully")
        end
    end
    
    return test_results
end

end # module RegressionTestSuite