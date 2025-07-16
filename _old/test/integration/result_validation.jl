module ResultValidation

using Statistics
using LinearAlgebra

export FeatureQualityMetrics, calculate_feature_quality
export validate_feature_consistency, validate_feature_stability
export compare_with_baseline, generate_validation_report

"""
Feature quality metrics
"""
struct FeatureQualityMetrics
    relevance_score::Float64      # How well features predict target
    redundancy_score::Float64     # How much features overlap
    stability_score::Float64      # Consistency across runs
    coverage_score::Float64       # Information coverage
    diversity_score::Float64      # Feature diversity
    overall_score::Float64        # Combined metric
end

"""
Calculate comprehensive feature quality metrics
"""
function calculate_feature_quality(
    X::Matrix{Float64},
    y::Vector{Int},
    selected_features::Vector{Int};
    original_features::Union{Nothing, Vector{Int}} = nothing
)::FeatureQualityMetrics
    
    X_selected = X[:, selected_features]
    n_features = length(selected_features)
    
    # 1. Relevance Score - mutual information with target
    relevance_scores = Float64[]
    for i in 1:n_features
        mi = mutual_information(X_selected[:, i], y)
        push!(relevance_scores, mi)
    end
    relevance_score = mean(relevance_scores)
    
    # 2. Redundancy Score - average pairwise correlation
    if n_features > 1
        correlations = Float64[]
        for i in 1:n_features-1
            for j in i+1:n_features
                corr = abs(cor(X_selected[:, i], X_selected[:, j]))
                push!(correlations, corr)
            end
        end
        redundancy_score = 1.0 - mean(correlations)  # Lower correlation = higher score
    else
        redundancy_score = 1.0
    end
    
    # 3. Stability Score - if we have original features, check overlap
    if !isnothing(original_features) && !isempty(original_features)
        overlap = length(intersect(selected_features, original_features))
        stability_score = overlap / min(length(selected_features), length(original_features))
    else
        stability_score = 1.0  # Perfect stability if no comparison
    end
    
    # 4. Coverage Score - explained variance
    if size(X, 2) > n_features
        # Compare variance of selected vs all features
        total_variance = sum(var(X, dims=1))
        selected_variance = sum(var(X_selected, dims=1))
        coverage_score = selected_variance / total_variance
    else
        coverage_score = 1.0
    end
    
    # 5. Diversity Score - based on feature value distributions
    diversity_scores = Float64[]
    for i in 1:n_features
        # Measure distribution spread
        values = X_selected[:, i]
        iqr = quantile(values, 0.75) - quantile(values, 0.25)
        std_val = std(values)
        diversity = iqr > 0 ? std_val / iqr : std_val
        push!(diversity_scores, min(diversity, 1.0))
    end
    diversity_score = mean(diversity_scores)
    
    # 6. Overall Score - weighted combination
    weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # Relevance weighted highest
    scores = [relevance_score, redundancy_score, stability_score, coverage_score, diversity_score]
    overall_score = sum(weights .* scores)
    
    return FeatureQualityMetrics(
        relevance_score,
        redundancy_score,
        stability_score,
        coverage_score,
        diversity_score,
        overall_score
    )
end

"""
Calculate mutual information between continuous variable and discrete target
"""
function mutual_information(x::Vector{Float64}, y::Vector{Int})
    # Discretize continuous variable
    n_bins = min(10, length(unique(x)))
    x_discrete = discretize(x, n_bins)
    
    # Calculate joint and marginal probabilities
    n = length(x)
    xy_counts = zeros(n_bins, maximum(y) + 1)
    
    for i in 1:n
        xy_counts[x_discrete[i], y[i] + 1] += 1
    end
    
    xy_probs = xy_counts / n
    x_probs = sum(xy_probs, dims=2)
    y_probs = sum(xy_probs, dims=1)
    
    # Calculate MI
    mi = 0.0
    for i in 1:n_bins
        for j in 1:size(xy_probs, 2)
            if xy_probs[i, j] > 0
                mi += xy_probs[i, j] * log(xy_probs[i, j] / (x_probs[i] * y_probs[j]))
            end
        end
    end
    
    return mi
end

"""
Discretize continuous variable into bins
"""
function discretize(x::Vector{Float64}, n_bins::Int)
    edges = range(minimum(x), maximum(x), length=n_bins+1)
    discrete = zeros(Int, length(x))
    
    for i in 1:length(x)
        bin = searchsortedfirst(edges[2:end], x[i])
        discrete[i] = min(bin, n_bins)
    end
    
    return discrete
end

"""
Validate feature consistency across stages
"""
function validate_feature_consistency(
    stage_results::Dict{Int, Dict{String, Any}}
)::Dict{String, Any}
    
    validation = Dict{String, Any}()
    validation["consistent"] = true
    validation["issues"] = String[]
    
    # Check monotonic decrease in features
    feature_counts = Int[]
    for stage in 1:3
        if haskey(stage_results, stage)
            push!(feature_counts, length(stage_results[stage]["selected_features"]))
        end
    end
    
    if length(feature_counts) > 1
        for i in 2:length(feature_counts)
            if feature_counts[i] > feature_counts[i-1]
                push!(validation["issues"], 
                    "Feature count increased from stage $(i-1) to $i: $(feature_counts[i-1]) → $(feature_counts[i])")
                validation["consistent"] = false
            end
        end
    end
    
    # Check quality improvement
    quality_scores = Float64[]
    for stage in 1:3
        if haskey(stage_results, stage)
            push!(quality_scores, stage_results[stage]["quality_score"])
        end
    end
    
    if length(quality_scores) > 1
        for i in 2:length(quality_scores)
            if quality_scores[i] < quality_scores[i-1] * 0.9  # Allow 10% degradation
                push!(validation["issues"],
                    "Quality degraded significantly from stage $(i-1) to $i: $(round(quality_scores[i-1], digits=3)) → $(round(quality_scores[i], digits=3))")
            end
        end
    end
    
    validation["feature_progression"] = feature_counts
    validation["quality_progression"] = quality_scores
    
    return validation
end

"""
Validate feature stability across multiple runs
"""
function validate_feature_stability(
    run_results::Vector{Vector{Int}};
    min_stability::Float64 = 0.7
)::Dict{String, Any}
    
    n_runs = length(run_results)
    if n_runs < 2
        return Dict("stable" => true, "message" => "Need at least 2 runs for stability check")
    end
    
    # Count feature frequencies
    all_features = reduce(union, run_results)
    feature_counts = Dict{Int, Int}()
    
    for features in run_results
        for f in features
            feature_counts[f] = get(feature_counts, f, 0) + 1
        end
    end
    
    # Calculate stability metrics
    stable_features = [f for (f, count) in feature_counts if count >= n_runs * min_stability]
    stability_ratio = length(stable_features) / length(all_features)
    
    # Pairwise Jaccard similarity
    similarities = Float64[]
    for i in 1:n_runs-1
        for j in i+1:n_runs
            intersection = length(intersect(run_results[i], run_results[j]))
            union_size = length(union(run_results[i], run_results[j]))
            similarity = union_size > 0 ? intersection / union_size : 1.0
            push!(similarities, similarity)
        end
    end
    
    avg_similarity = mean(similarities)
    
    validation = Dict{String, Any}(
        "stable" => stability_ratio >= min_stability && avg_similarity >= min_stability,
        "stability_ratio" => stability_ratio,
        "average_similarity" => avg_similarity,
        "stable_features" => stable_features,
        "feature_frequencies" => feature_counts
    )
    
    if !validation["stable"]
        validation["message"] = "Feature selection is unstable across runs"
    end
    
    return validation
end

"""
Compare results with baseline methods
"""
function compare_with_baseline(
    selected_features::Vector{Int},
    X::Matrix{Float64},
    y::Vector{Int};
    baseline_methods::Vector{Symbol} = [:correlation, :mutual_information, :variance]
)::Dict{String, Any}
    
    n_features = length(selected_features)
    comparisons = Dict{String, Any}()
    
    for method in baseline_methods
        baseline_features = select_baseline_features(X, y, n_features, method)
        
        # Calculate overlap
        overlap = length(intersect(selected_features, baseline_features))
        overlap_ratio = overlap / n_features
        
        # Calculate quality difference
        our_quality = calculate_feature_quality(X, y, selected_features)
        baseline_quality = calculate_feature_quality(X, y, baseline_features)
        
        quality_improvement = (our_quality.overall_score - baseline_quality.overall_score) / 
                            baseline_quality.overall_score
        
        comparisons[string(method)] = Dict(
            "overlap" => overlap,
            "overlap_ratio" => overlap_ratio,
            "our_quality" => our_quality.overall_score,
            "baseline_quality" => baseline_quality.overall_score,
            "improvement" => quality_improvement,
            "better" => quality_improvement > 0
        )
    end
    
    # Summary
    avg_improvement = mean([c["improvement"] for c in values(comparisons)])
    better_count = sum([c["better"] for c in values(comparisons)])
    
    comparisons["summary"] = Dict(
        "average_improvement" => avg_improvement,
        "better_than_baselines" => better_count,
        "total_baselines" => length(baseline_methods)
    )
    
    return comparisons
end

"""
Select features using baseline method
"""
function select_baseline_features(
    X::Matrix{Float64},
    y::Vector{Int},
    n_features::Int,
    method::Symbol
)::Vector{Int}
    
    scores = if method == :correlation
        # Correlation with target
        [abs(cor(X[:, i], Float64.(y))) for i in 1:size(X, 2)]
    elseif method == :mutual_information
        # Mutual information with target
        [mutual_information(X[:, i], y) for i in 1:size(X, 2)]
    elseif method == :variance
        # Feature variance
        vec(var(X, dims=1))
    else
        error("Unknown baseline method: $method")
    end
    
    # Select top features
    return sortperm(scores, rev=true)[1:min(n_features, length(scores))]
end

"""
Generate comprehensive validation report
"""
function generate_validation_report(
    test_result::Any,  # PipelineTestResult
    X::Matrix{Float64},
    y::Vector{Int}
)::Dict{String, Any}
    
    report = Dict{String, Any}()
    
    # Basic info
    report["dataset"] = test_result.dataset_name
    report["passed"] = test_result.passed
    report["runtime"] = test_result.total_runtime
    report["memory_mb"] = test_result.peak_memory_mb
    
    # Feature reduction
    report["feature_reduction"] = Dict(
        "counts" => test_result.feature_reduction,
        "reduction_ratio" => test_result.feature_reduction[end] / test_result.feature_reduction[1]
    )
    
    # Quality progression
    report["quality_progression"] = test_result.quality_scores
    
    # Stage consistency
    report["consistency"] = validate_feature_consistency(test_result.stage_results)
    
    # Final feature quality
    if haskey(test_result.stage_results, 3)
        final_features = test_result.stage_results[3]["selected_features"]
        quality_metrics = calculate_feature_quality(X, y, final_features)
        
        report["final_quality"] = Dict(
            "relevance" => quality_metrics.relevance_score,
            "redundancy" => quality_metrics.redundancy_score,
            "stability" => quality_metrics.stability_score,
            "coverage" => quality_metrics.coverage_score,
            "diversity" => quality_metrics.diversity_score,
            "overall" => quality_metrics.overall_score
        )
        
        # Baseline comparison
        report["baseline_comparison"] = compare_with_baseline(final_features, X, y)
    end
    
    # Issues summary
    report["issues"] = Dict(
        "errors" => test_result.errors,
        "warnings" => test_result.warnings,
        "total_issues" => length(test_result.errors) + length(test_result.warnings)
    )
    
    # Performance summary
    report["performance"] = Dict(
        "stages" => Dict()
    )
    
    for (stage, result) in test_result.stage_results
        report["performance"]["stages"][stage] = Dict(
            "runtime" => result["runtime"],
            "memory_mb" => result["memory_mb"],
            "features_selected" => length(result["selected_features"])
        )
    end
    
    return report
end

end # module