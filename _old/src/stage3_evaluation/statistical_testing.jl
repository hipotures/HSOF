module StatisticalTesting

using Statistics
using StatsBase
using Distributions
using HypothesisTests
using DataFrames
using Printf
using Random

export StatisticalTest, TestResult, MultipleTestCorrection
export paired_t_test, wilcoxon_signed_rank_test, mcnemar_test
export calculate_confidence_interval, calculate_effect_size
export bonferroni_correction, benjamini_hochberg_correction, holm_correction
export compare_models, generate_statistical_report
export bootstrap_confidence_interval, permutation_test
export BONFERRONI, BENJAMINI_HOCHBERG, HOLM, NONE

"""
Result structure for statistical tests
"""
struct TestResult
    test_name::String
    statistic::Float64
    p_value::Float64
    confidence_interval::Union{Nothing, Tuple{Float64, Float64}}
    effect_size::Union{Nothing, Float64}
    significant::Bool
    alpha::Float64
    metadata::Dict{Symbol, Any}
end

"""
Multiple test correction methods
"""
@enum MultipleTestCorrection begin
    BONFERRONI
    BENJAMINI_HOCHBERG
    HOLM
    NONE
end

"""
Perform paired t-test on cross-validation results
"""
function paired_t_test(scores1::Vector{<:Real}, scores2::Vector{<:Real}; 
                      alpha::Float64=0.05, alternative::Symbol=:two_sided)
    
    @assert length(scores1) == length(scores2) "Score vectors must have equal length"
    @assert length(scores1) >= 2 "Need at least 2 observations for paired t-test"
    
    # Convert to Float64 for calculations
    s1 = Float64.(scores1)
    s2 = Float64.(scores2)
    
    # Calculate differences
    differences = s1 .- s2
    n = length(differences)
    
    # Basic statistics
    mean_diff = mean(differences)
    std_diff = std(differences)
    
    # Handle case where all differences are zero
    if std_diff == 0.0
        return TestResult(
            "Paired t-test",
            0.0,
            1.0,
            (0.0, 0.0),
            0.0,
            false,
            alpha,
            Dict(:mean_difference => 0.0,
                 :degrees_of_freedom => n - 1,
                 :alternative => alternative,
                 :n_pairs => n)
        )
    end
    
    se_diff = std_diff / sqrt(n)
    
    # t-statistic
    t_stat = mean_diff / se_diff
    
    # Degrees of freedom
    df = n - 1
    
    # Calculate p-value based on alternative hypothesis
    t_dist = TDist(df)
    if alternative == :two_sided
        p_value = 2 * ccdf(t_dist, abs(t_stat))
    elseif alternative == :greater
        p_value = ccdf(t_dist, t_stat)
    elseif alternative == :less
        p_value = cdf(t_dist, t_stat)
    else
        error("Unknown alternative hypothesis: $alternative")
    end
    
    # Confidence interval for mean difference
    t_critical = quantile(t_dist, 1 - alpha/2)
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff
    
    # Effect size (Cohen's d)
    pooled_std = sqrt((var(scores1) + var(scores2)) / 2)
    cohens_d = mean_diff / pooled_std
    
    return TestResult(
        "Paired t-test",
        t_stat,
        p_value,
        (ci_lower, ci_upper),
        cohens_d,
        p_value < alpha,
        alpha,
        Dict(:mean_difference => mean_diff,
             :degrees_of_freedom => df,
             :alternative => alternative,
             :n_pairs => n)
    )
end

"""
Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
"""
function wilcoxon_signed_rank_test(scores1::Vector{Float64}, scores2::Vector{Float64};
                                  alpha::Float64=0.05, alternative::Symbol=:two_sided)
    
    @assert length(scores1) == length(scores2) "Score vectors must have equal length"
    
    # Calculate differences
    differences = scores1 .- scores2
    
    # Remove zero differences
    non_zero_diffs = differences[differences .!= 0]
    n = length(non_zero_diffs)
    
    if n == 0
        # All differences are zero
        return TestResult(
            "Wilcoxon signed-rank test",
            0.0,
            1.0,
            nothing,
            0.0,
            false,
            alpha,
            Dict(:n_pairs => length(scores1),
                 :n_non_zero => 0,
                 :alternative => alternative)
        )
    end
    
    # Rank absolute differences
    abs_diffs = abs.(non_zero_diffs)
    ranks = ordinalrank(abs_diffs)
    
    # Calculate test statistic (sum of positive ranks)
    positive_ranks = ranks[non_zero_diffs .> 0]
    W_plus = isempty(positive_ranks) ? 0.0 : sum(positive_ranks)
    
    # For large samples, use normal approximation
    if n > 20
        # Expected value and variance under null hypothesis
        E_W = n * (n + 1) / 4
        
        # Variance with tie correction
        ties = countmap(abs_diffs)
        tie_correction = 0.0
        for (_, t) in ties
            if t > 1
                tie_correction += t * (t^2 - 1)
            end
        end
        tie_correction /= 48
        Var_W = n * (n + 1) * (2*n + 1) / 24 - tie_correction
        
        # Z-score
        z = (W_plus - E_W) / sqrt(Var_W)
        
        # p-value
        if alternative == :two_sided
            p_value = 2 * ccdf(Normal(), abs(z))
        elseif alternative == :greater
            p_value = ccdf(Normal(), z)
        elseif alternative == :less
            p_value = cdf(Normal(), z)
        else
            error("Unknown alternative hypothesis: $alternative")
        end
    else
        # For small samples, use exact test
        # This is a simplified version - full implementation would use exact distribution
        p_value = exact_wilcoxon_p_value(Float64(W_plus), n, alternative)
    end
    
    # Effect size (rank-biserial correlation)
    W_minus = n * (n + 1) / 2 - W_plus
    r_rb = (W_plus - W_minus) / (n * (n + 1) / 2)
    
    return TestResult(
        "Wilcoxon signed-rank test",
        W_plus,
        p_value,
        nothing,  # No standard CI for Wilcoxon test
        r_rb,
        p_value < alpha,
        alpha,
        Dict(:n_pairs => length(scores1),
             :n_non_zero => n,
             :alternative => alternative)
    )
end

"""
Perform McNemar's test for paired binary classification results
"""
function mcnemar_test(predictions1::Vector{Bool}, predictions2::Vector{Bool}, 
                     y_true::Vector{Bool}; alpha::Float64=0.05)
    
    @assert length(predictions1) == length(predictions2) == length(y_true)
    
    # Create contingency table
    # a: both correct, b: only model1 correct
    # c: only model2 correct, d: both wrong
    a = sum((predictions1 .== y_true) .& (predictions2 .== y_true))
    b = sum((predictions1 .== y_true) .& (predictions2 .!= y_true))
    c = sum((predictions1 .!= y_true) .& (predictions2 .== y_true))
    d = sum((predictions1 .!= y_true) .& (predictions2 .!= y_true))
    
    # McNemar's test statistic
    if b + c == 0
        # No discordant pairs
        chi2_stat = 0.0
        p_value = 1.0
    else
        # Use continuity correction for small samples
        if b + c < 25
            chi2_stat = (abs(b - c) - 1)^2 / (b + c)
        else
            chi2_stat = (b - c)^2 / (b + c)
        end
        
        # p-value from chi-squared distribution with 1 df
        p_value = ccdf(Chisq(1), chi2_stat)
    end
    
    # Odds ratio and confidence interval
    if b == 0 || c == 0
        # Add 0.5 to avoid division by zero
        odds_ratio = (b + 0.5) / (c + 0.5)
        log_se = sqrt(1/(b + 0.5) + 1/(c + 0.5))
    else
        odds_ratio = b / c
        log_se = sqrt(1/b + 1/c)
    end
    
    z_critical = quantile(Normal(), 1 - alpha/2)
    ci_lower = exp(log(odds_ratio) - z_critical * log_se)
    ci_upper = exp(log(odds_ratio) + z_critical * log_se)
    
    return TestResult(
        "McNemar's test",
        chi2_stat,
        p_value,
        (ci_lower, ci_upper),
        log(odds_ratio),  # Log odds ratio as effect size
        p_value < alpha,
        alpha,
        Dict(:contingency_table => [a b; c d],
             :discordant_pairs => b + c,
             :odds_ratio => odds_ratio)
    )
end

"""
Calculate confidence interval for a metric using bootstrap or analytical methods
"""
function calculate_confidence_interval(scores::Vector{<:Real}; 
                                     alpha::Float64=0.05,
                                     method::Symbol=:analytical)
    
    n = length(scores)
    mean_score = mean(scores)
    
    if method == :analytical
        # Analytical CI assuming normal distribution
        se = std(scores) / sqrt(n)
        t_critical = quantile(TDist(n - 1), 1 - alpha/2)
        ci_lower = mean_score - t_critical * se
        ci_upper = mean_score + t_critical * se
    elseif method == :bootstrap
        # Bootstrap CI
        ci_lower, ci_upper = bootstrap_confidence_interval(scores, alpha=alpha)
    else
        error("Unknown confidence interval method: $method")
    end
    
    return (ci_lower, ci_upper)
end

"""
Calculate effect size (Cohen's d) between two groups
"""
function calculate_effect_size(scores1::Vector{<:Real}, scores2::Vector{<:Real};
                             method::Symbol=:cohens_d)
    
    if method == :cohens_d
        # Cohen's d with pooled standard deviation
        mean1, mean2 = mean(scores1), mean(scores2)
        n1, n2 = length(scores1), length(scores2)
        
        # Pooled standard deviation
        pooled_std = sqrt(((n1 - 1) * var(scores1) + (n2 - 1) * var(scores2)) / (n1 + n2 - 2))
        
        cohens_d = (mean1 - mean2) / pooled_std
        
        return cohens_d
    elseif method == :glass_delta
        # Glass's delta (uses control group SD)
        mean1, mean2 = mean(scores1), mean(scores2)
        glass_delta = (mean1 - mean2) / std(scores2)
        
        return glass_delta
    elseif method == :hedges_g
        # Hedges' g (corrected Cohen's d for small samples)
        cohens_d = calculate_effect_size(scores1, scores2, method=:cohens_d)
        n1, n2 = length(scores1), length(scores2)
        
        # Correction factor
        correction = 1 - 3 / (4 * (n1 + n2) - 9)
        hedges_g = cohens_d * correction
        
        return hedges_g
    else
        error("Unknown effect size method: $method")
    end
end

"""
Apply Bonferroni correction to p-values
"""
function bonferroni_correction(p_values::Vector{Float64}, alpha::Float64=0.05)
    n_tests = length(p_values)
    adjusted_alpha = alpha / n_tests
    adjusted_p_values = min.(p_values .* n_tests, 1.0)
    significant = adjusted_p_values .< alpha
    
    return adjusted_p_values, significant, adjusted_alpha
end

"""
Apply Benjamini-Hochberg correction (FDR control)
"""
function benjamini_hochberg_correction(p_values::Vector{Float64}, alpha::Float64=0.05)
    n = length(p_values)
    
    # Sort p-values and keep track of original indices
    sorted_indices = sortperm(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Apply BH procedure
    adjusted_p = zeros(n)
    for i in n:-1:1
        if i == n
            adjusted_p[i] = sorted_p[i]
        else
            adjusted_p[i] = min(adjusted_p[i+1], sorted_p[i] * n / i)
        end
    end
    
    # Ensure adjusted p-values don't exceed 1
    adjusted_p = min.(adjusted_p, 1.0)
    
    # Determine significance
    significant = falses(n)
    for i in n:-1:1
        if sorted_p[i] <= (i / n) * alpha
            significant[1:i] .= true
            break
        end
    end
    
    # Restore original order
    adjusted_p_values = similar(p_values)
    sig_values = similar(significant)
    adjusted_p_values[sorted_indices] = adjusted_p
    sig_values[sorted_indices] = significant
    
    return adjusted_p_values, sig_values, alpha
end

"""
Apply Holm correction to p-values
"""
function holm_correction(p_values::Vector{Float64}, alpha::Float64=0.05)
    n = length(p_values)
    
    # Sort p-values
    sorted_indices = sortperm(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Apply Holm procedure
    adjusted_p = zeros(n)
    significant = falses(n)
    
    for i in 1:n
        adjusted_p[i] = min(sorted_p[i] * (n - i + 1), 1.0)
        if i > 1
            adjusted_p[i] = max(adjusted_p[i], adjusted_p[i-1])
        end
    end
    
    # Test significance separately
    for i in 1:n
        if sorted_p[i] <= alpha / (n - i + 1)
            significant[i] = true
        else
            break  # Stop at first non-significant
        end
    end
    
    # Restore original order
    adjusted_p_values = similar(p_values)
    sig_values = similar(significant)
    adjusted_p_values[sorted_indices] = adjusted_p
    sig_values[sorted_indices] = significant
    
    return adjusted_p_values, sig_values, alpha
end

"""
Bootstrap confidence interval calculation
"""
function bootstrap_confidence_interval(data::Vector{Float64}; 
                                     n_bootstrap::Int=10000,
                                     alpha::Float64=0.05)
    n = length(data)
    bootstrap_means = zeros(n_bootstrap)
    
    for i in 1:n_bootstrap
        # Resample with replacement
        bootstrap_sample = data[rand(1:n, n)]
        bootstrap_means[i] = mean(bootstrap_sample)
    end
    
    # Percentile method
    lower_percentile = alpha / 2
    upper_percentile = 1 - alpha / 2
    
    ci_lower = quantile(bootstrap_means, lower_percentile)
    ci_upper = quantile(bootstrap_means, upper_percentile)
    
    return (ci_lower, ci_upper)
end

"""
Permutation test for comparing two independent samples
"""
function permutation_test(scores1::Vector{Float64}, scores2::Vector{Float64};
                         n_permutations::Int=10000,
                         statistic::Function=x -> mean(x[1]) - mean(x[2]),
                         alternative::Symbol=:two_sided)
    
    n1, n2 = length(scores1), length(scores2)
    combined = vcat(scores1, scores2)
    n_total = n1 + n2
    
    # Observed statistic
    observed_stat = statistic([scores1, scores2])
    
    # Permutation distribution
    perm_stats = zeros(n_permutations)
    
    for i in 1:n_permutations
        # Shuffle combined data
        perm = Random.randperm(n_total)
        perm_group1 = combined[perm[1:n1]]
        perm_group2 = combined[perm[(n1+1):end]]
        
        perm_stats[i] = statistic([perm_group1, perm_group2])
    end
    
    # Calculate p-value
    if alternative == :two_sided
        p_value = mean(abs.(perm_stats) .>= abs(observed_stat))
    elseif alternative == :greater
        p_value = mean(perm_stats .>= observed_stat)
    elseif alternative == :less
        p_value = mean(perm_stats .<= observed_stat)
    else
        error("Unknown alternative hypothesis: $alternative")
    end
    
    return p_value, observed_stat, perm_stats
end

"""
Compare multiple models using appropriate statistical tests
"""
function compare_models(results_dict::Dict{String, <:Vector{<:Real}};
                       test_type::Symbol=:paired_t,
                       correction::MultipleTestCorrection=BONFERRONI,
                       alpha::Float64=0.05,
                       baseline_model::Union{Nothing, String}=nothing)
    
    model_names = collect(keys(results_dict))
    n_models = length(model_names)
    
    # Determine baseline model
    if isnothing(baseline_model)
        baseline_model = model_names[1]
    end
    
    # Perform pairwise comparisons
    comparisons = []
    
    for i in 1:n_models
        for j in (i+1):n_models
            model1, model2 = model_names[i], model_names[j]
            scores1 = Float64.(results_dict[model1])
            scores2 = Float64.(results_dict[model2])
            
            # Choose test based on test_type
            if test_type == :paired_t
                result = paired_t_test(scores1, scores2, alpha=alpha)
            elseif test_type == :wilcoxon
                result = wilcoxon_signed_rank_test(scores1, scores2, alpha=alpha)
            else
                error("Unknown test type: $test_type")
            end
            
            push!(comparisons, (
                model1 = model1,
                model2 = model2,
                test_result = result
            ))
        end
    end
    
    # Extract p-values for correction
    p_values = [comp.test_result.p_value for comp in comparisons]
    
    # Apply multiple testing correction
    if correction == BONFERRONI
        adjusted_p, significant, adj_alpha = bonferroni_correction(p_values, alpha)
    elseif correction == BENJAMINI_HOCHBERG
        adjusted_p, significant, adj_alpha = benjamini_hochberg_correction(p_values, alpha)
    elseif correction == HOLM
        adjusted_p, significant, adj_alpha = holm_correction(p_values, alpha)
    else
        adjusted_p = p_values
        significant = p_values .< alpha
        adj_alpha = alpha
    end
    
    # Update comparisons with adjusted p-values
    for (i, comp) in enumerate(comparisons)
        comp = merge(comp, (
            adjusted_p_value = adjusted_p[i],
            significant_after_correction = significant[i]
        ))
        comparisons[i] = comp
    end
    
    return comparisons, adj_alpha
end

"""
Generate comprehensive statistical report
"""
function generate_statistical_report(comparisons::Vector, 
                                   model_scores::Dict{String, Vector{Float64}};
                                   output_file::Union{Nothing, String}=nothing)
    
    report_lines = String[]
    
    push!(report_lines, "=" ^ 80)
    push!(report_lines, "STATISTICAL ANALYSIS REPORT")
    push!(report_lines, "=" ^ 80)
    push!(report_lines, "")
    
    # Model summary statistics
    push!(report_lines, "MODEL PERFORMANCE SUMMARY")
    push!(report_lines, "-" ^ 40)
    
    for (model, scores) in model_scores
        mean_score = mean(scores)
        std_score = std(scores)
        ci = calculate_confidence_interval(scores)
        
        push!(report_lines, @sprintf("Model: %s", model))
        push!(report_lines, @sprintf("  Mean Score: %.4f ± %.4f", mean_score, std_score))
        push!(report_lines, @sprintf("  95%% CI: [%.4f, %.4f]", ci[1], ci[2]))
        push!(report_lines, @sprintf("  Range: [%.4f, %.4f]", minimum(scores), maximum(scores)))
        push!(report_lines, "")
    end
    
    # Pairwise comparisons
    push!(report_lines, "PAIRWISE MODEL COMPARISONS")
    push!(report_lines, "-" ^ 40)
    
    for comp in comparisons
        result = comp.test_result
        
        push!(report_lines, @sprintf("%s vs %s:", comp.model1, comp.model2))
        push!(report_lines, @sprintf("  Test: %s", result.test_name))
        push!(report_lines, @sprintf("  Test Statistic: %.4f", result.statistic))
        push!(report_lines, @sprintf("  P-value: %.4f", result.p_value))
        push!(report_lines, @sprintf("  Adjusted P-value: %.4f", comp.adjusted_p_value))
        
        if !isnothing(result.effect_size)
            push!(report_lines, @sprintf("  Effect Size: %.4f", result.effect_size))
            
            # Interpret effect size
            abs_effect = abs(result.effect_size)
            if abs_effect < 0.2
                interpretation = "negligible"
            elseif abs_effect < 0.5
                interpretation = "small"
            elseif abs_effect < 0.8
                interpretation = "medium"
            else
                interpretation = "large"
            end
            push!(report_lines, @sprintf("  Effect Size Interpretation: %s", interpretation))
        end
        
        if !isnothing(result.confidence_interval)
            push!(report_lines, @sprintf("  95%% CI for difference: [%.4f, %.4f]", 
                                        result.confidence_interval[1], 
                                        result.confidence_interval[2]))
        end
        
        significance = comp.significant_after_correction ? "YES" : "NO"
        push!(report_lines, @sprintf("  Significant after correction: %s", significance))
        push!(report_lines, "")
    end
    
    # Statistical significance summary
    push!(report_lines, "SIGNIFICANCE SUMMARY")
    push!(report_lines, "-" ^ 40)
    
    sig_pairs = [(c.model1, c.model2) for c in comparisons if c.significant_after_correction]
    
    if isempty(sig_pairs)
        push!(report_lines, "No significant differences found after multiple testing correction.")
    else
        push!(report_lines, "Significant differences found between:")
        for (m1, m2) in sig_pairs
            push!(report_lines, @sprintf("  - %s and %s", m1, m2))
        end
    end
    
    push!(report_lines, "")
    push!(report_lines, "=" ^ 80)
    
    # Join lines into report
    report = join(report_lines, "\n")
    
    # Save to file if specified
    if !isnothing(output_file)
        open(output_file, "w") do f
            write(f, report)
        end
    end
    
    return report
end

"""
Helper function for exact Wilcoxon p-value calculation (simplified)
"""
function exact_wilcoxon_p_value(W::Float64, n::Int, alternative::Symbol)
    # This is a placeholder - full implementation would compute exact distribution
    # For now, use normal approximation even for small samples
    E_W = n * (n + 1) / 4
    Var_W = n * (n + 1) * (2*n + 1) / 24
    z = (W - E_W) / sqrt(Var_W)
    
    if alternative == :two_sided
        return 2 * ccdf(Normal(), abs(z))
    elseif alternative == :greater
        return ccdf(Normal(), z)
    else
        return cdf(Normal(), z)
    end
end

end # module