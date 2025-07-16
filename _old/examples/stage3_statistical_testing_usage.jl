# Example usage of Statistical Testing for Stage 3 Model Comparison

using Random
using Statistics
using DataFrames
using CSV

# Include the modules
include("../src/stage3_evaluation/statistical_testing.jl")

using .StatisticalTesting

# Set random seed
Random.seed!(123)

"""
Example 1: Basic paired t-test for model comparison
"""
function paired_comparison_example()
    println("=== Paired T-Test Model Comparison Example ===\n")
    
    # Simulate cross-validation results for two models
    n_folds = 10
    
    # Model A: XGBoost with mean accuracy ~0.85
    model_a_scores = [0.84, 0.86, 0.85, 0.87, 0.83, 0.85, 0.86, 0.84, 0.85, 0.86]
    
    # Model B: Random Forest with mean accuracy ~0.82
    model_b_scores = [0.81, 0.83, 0.82, 0.84, 0.80, 0.82, 0.83, 0.81, 0.82, 0.83]
    
    println("Model A (XGBoost) scores: ", model_a_scores)
    println("Mean: $(round(mean(model_a_scores), digits=4))")
    println("\nModel B (RandomForest) scores: ", model_b_scores)
    println("Mean: $(round(mean(model_b_scores), digits=4))")
    
    # Perform paired t-test
    result = paired_t_test(model_a_scores, model_b_scores, alpha=0.05)
    
    println("\n--- Paired t-test Results ---")
    println("Test statistic: $(round(result.statistic, digits=4))")
    println("P-value: $(round(result.p_value, digits=6))")
    println("Mean difference: $(round(result.metadata[:mean_difference], digits=4))")
    println("95% CI for difference: [$(round(result.confidence_interval[1], digits=4)), $(round(result.confidence_interval[2], digits=4))]")
    println("Cohen's d effect size: $(round(result.effect_size, digits=4))")
    println("Significant at α=0.05: $(result.significant ? "YES" : "NO")")
    
    # Interpret effect size
    abs_d = abs(result.effect_size)
    if abs_d < 0.2
        effect_interpretation = "negligible"
    elseif abs_d < 0.5
        effect_interpretation = "small"
    elseif abs_d < 0.8
        effect_interpretation = "medium"
    else
        effect_interpretation = "large"
    end
    println("Effect size interpretation: $effect_interpretation")
    
    return result
end

"""
Example 2: Non-parametric comparison with Wilcoxon test
"""
function nonparametric_comparison_example()
    println("\n\n=== Wilcoxon Signed-Rank Test Example ===\n")
    
    # Simulate scores with non-normal distribution (e.g., skewed)
    n_folds = 12
    
    # Model A: Good performance but with outliers
    model_a_scores = [0.90, 0.91, 0.89, 0.92, 0.65, 0.90, 0.91, 0.88, 0.90, 0.93, 0.89, 0.91]
    
    # Model B: More consistent but lower average
    model_b_scores = [0.87, 0.86, 0.88, 0.87, 0.85, 0.86, 0.87, 0.86, 0.87, 0.88, 0.86, 0.87]
    
    println("Model A scores (with outlier): ", model_a_scores)
    println("Mean: $(round(mean(model_a_scores), digits=4)), Median: $(round(median(model_a_scores), digits=4))")
    println("\nModel B scores (consistent): ", model_b_scores)
    println("Mean: $(round(mean(model_b_scores), digits=4)), Median: $(round(median(model_b_scores), digits=4))")
    
    # Perform both parametric and non-parametric tests
    t_result = paired_t_test(model_a_scores, model_b_scores)
    w_result = wilcoxon_signed_rank_test(model_a_scores, model_b_scores)
    
    println("\n--- Comparison of Tests ---")
    println("Paired t-test:")
    println("  P-value: $(round(t_result.p_value, digits=6))")
    println("  Significant: $(t_result.significant ? "YES" : "NO")")
    
    println("\nWilcoxon signed-rank test:")
    println("  P-value: $(round(w_result.p_value, digits=6))")
    println("  Significant: $(w_result.significant ? "YES" : "NO")")
    println("  Rank-biserial correlation: $(round(w_result.effect_size, digits=4))")
    
    println("\nNote: Wilcoxon test is more robust to outliers")
    
    return t_result, w_result
end

"""
Example 3: McNemar's test for classification models
"""
function mcnemar_test_example()
    println("\n\n=== McNemar's Test for Classification Example ===\n")
    
    # Simulate binary classification results on 200 test samples
    n_samples = 200
    Random.seed!(42)
    
    # True labels
    y_true = rand(Bool, n_samples)
    
    # Model A predictions (85% accuracy)
    model_a_pred = copy(y_true)
    errors_a = randperm(n_samples)[1:30]  # 30 errors
    model_a_pred[errors_a] .= .!model_a_pred[errors_a]
    
    # Model B predictions (83% accuracy but different error pattern)
    model_b_pred = copy(y_true)
    errors_b = randperm(n_samples)[15:48]  # 34 errors, some overlap with A
    model_b_pred[errors_b] .= .!model_b_pred[errors_b]
    
    # Calculate accuracies
    acc_a = mean(model_a_pred .== y_true)
    acc_b = mean(model_b_pred .== y_true)
    
    println("Model A accuracy: $(round(acc_a, digits=4))")
    println("Model B accuracy: $(round(acc_b, digits=4))")
    
    # Perform McNemar's test
    result = mcnemar_test(model_a_pred, model_b_pred, y_true)
    
    println("\n--- McNemar's Test Results ---")
    println("Chi-squared statistic: $(round(result.statistic, digits=4))")
    println("P-value: $(round(result.p_value, digits=6))")
    println("Significant at α=0.05: $(result.significant ? "YES" : "NO")")
    
    # Show contingency table
    cont_table = result.metadata[:contingency_table]
    println("\nContingency Table:")
    println("                Model B Correct | Model B Wrong")
    println("Model A Correct      $(cont_table[1,1])         |     $(cont_table[1,2])")
    println("Model A Wrong        $(cont_table[2,1])         |     $(cont_table[2,2])")
    
    println("\nDiscordant pairs: $(result.metadata[:discordant_pairs])")
    println("Odds ratio: $(round(result.metadata[:odds_ratio], digits=4))")
    println("95% CI for OR: [$(round(result.confidence_interval[1], digits=4)), $(round(result.confidence_interval[2], digits=4))]")
    
    return result
end

"""
Example 4: Multiple model comparison with correction
"""
function multiple_comparison_example()
    println("\n\n=== Multiple Model Comparison with Corrections Example ===\n")
    
    # Simulate CV results for 5 different models
    n_folds = 10
    model_scores = Dict(
        "XGBoost" => [0.88, 0.87, 0.89, 0.86, 0.88, 0.90, 0.87, 0.88, 0.89, 0.88],
        "RandomForest" => [0.85, 0.84, 0.86, 0.83, 0.85, 0.87, 0.84, 0.85, 0.86, 0.85],
        "LightGBM" => [0.87, 0.86, 0.88, 0.85, 0.87, 0.89, 0.86, 0.87, 0.88, 0.87],
        "SVM" => [0.82, 0.81, 0.83, 0.80, 0.82, 0.84, 0.81, 0.82, 0.83, 0.82],
        "LogisticReg" => [0.79, 0.78, 0.80, 0.77, 0.79, 0.81, 0.78, 0.79, 0.80, 0.79]
    )
    
    # Display mean scores
    println("Model Performance Summary:")
    for (model, scores) in model_scores
        println("  $model: Mean = $(round(mean(scores), digits=4)), Std = $(round(std(scores), digits=4))")
    end
    
    # Compare all models with different correction methods
    println("\n--- Pairwise Comparisons ---")
    
    # 1. Bonferroni correction
    comparisons_bonf, alpha_bonf = compare_models(model_scores, 
                                                 test_type=:paired_t,
                                                 correction=BONFERRONI,
                                                 alpha=0.05)
    
    println("\nWith Bonferroni correction (α = $(round(alpha_bonf, digits=4))):")
    print_significant_pairs(comparisons_bonf)
    
    # 2. Benjamini-Hochberg (FDR) correction
    comparisons_bh, alpha_bh = compare_models(model_scores,
                                            test_type=:paired_t,
                                            correction=BENJAMINI_HOCHBERG,
                                            alpha=0.05)
    
    println("\nWith Benjamini-Hochberg correction:")
    print_significant_pairs(comparisons_bh)
    
    # 3. Holm correction
    comparisons_holm, alpha_holm = compare_models(model_scores,
                                                test_type=:paired_t,
                                                correction=HOLM,
                                                alpha=0.05)
    
    println("\nWith Holm correction:")
    print_significant_pairs(comparisons_holm)
    
    # 4. No correction (for comparison)
    comparisons_none, _ = compare_models(model_scores,
                                       test_type=:paired_t,
                                       correction=NONE,
                                       alpha=0.05)
    
    println("\nWithout correction (not recommended):")
    print_significant_pairs(comparisons_none)
    
    return comparisons_bonf, comparisons_bh, comparisons_holm
end

"""
Example 5: Bootstrap and permutation testing
"""
function resampling_methods_example()
    println("\n\n=== Bootstrap and Permutation Testing Example ===\n")
    
    # Generate data for a single model's performance
    model_scores = [0.82, 0.85, 0.81, 0.84, 0.83, 0.86, 0.82, 0.84, 0.85, 0.83]
    
    println("Model scores: ", model_scores)
    println("Mean: $(round(mean(model_scores), digits=4))")
    
    # Calculate confidence intervals using different methods
    println("\n--- Confidence Interval Comparison ---")
    
    # Analytical (t-distribution based)
    ci_analytical = calculate_confidence_interval(model_scores, method=:analytical)
    println("Analytical 95% CI: [$(round(ci_analytical[1], digits=4)), $(round(ci_analytical[2], digits=4))]")
    
    # Bootstrap
    ci_bootstrap = calculate_confidence_interval(model_scores, method=:bootstrap)
    println("Bootstrap 95% CI: [$(round(ci_bootstrap[1], digits=4)), $(round(ci_bootstrap[2], digits=4))]")
    
    # Permutation test example
    println("\n--- Permutation Test Example ---")
    
    # Compare two feature selection methods
    method_a_scores = [0.86, 0.87, 0.85, 0.88, 0.86, 0.87, 0.88, 0.86]
    method_b_scores = [0.83, 0.84, 0.82, 0.85, 0.83, 0.84, 0.85, 0.83]
    
    println("\nMethod A scores: Mean = $(round(mean(method_a_scores), digits=4))")
    println("Method B scores: Mean = $(round(mean(method_b_scores), digits=4))")
    
    # Permutation test
    p_value, obs_stat, perm_dist = permutation_test(method_a_scores, method_b_scores,
                                                    n_permutations=5000)
    
    println("\nPermutation test results:")
    println("Observed difference: $(round(obs_stat, digits=4))")
    println("P-value: $(round(p_value, digits=6))")
    println("Significant at α=0.05: $(p_value < 0.05 ? "YES" : "NO")")
    
    # Compare with t-test
    t_test_unpaired = OneSampleTTest(method_a_scores .- method_b_scores, 0.0)
    println("\nFor comparison, unpaired t-test p-value: $(round(pvalue(t_test_unpaired), digits=6))")
    
    return ci_analytical, ci_bootstrap, p_value
end

"""
Example 6: Comprehensive statistical report
"""
function comprehensive_report_example()
    println("\n\n=== Comprehensive Statistical Report Example ===\n")
    
    # Simulate results from final feature selection evaluation
    # 4 models tested with 5-fold CV
    model_scores = Dict(
        "XGBoost_All" => [0.875, 0.882, 0.871, 0.879, 0.884],      # All features
        "XGBoost_Selected" => [0.886, 0.891, 0.883, 0.888, 0.893], # Selected features
        "RF_All" => [0.862, 0.868, 0.859, 0.865, 0.871],
        "RF_Selected" => [0.871, 0.876, 0.869, 0.873, 0.878]
    )
    
    println("Comparing models with all features vs selected features")
    println("Number of features: All = 500, Selected = 20")
    
    # Perform comparisons
    comparisons, _ = compare_models(model_scores,
                                  test_type=:paired_t,
                                  correction=BONFERRONI,
                                  alpha=0.05)
    
    # Generate report
    report = generate_statistical_report(comparisons, model_scores)
    
    # Save to file
    report_file = "statistical_analysis_report.txt"
    generate_statistical_report(comparisons, model_scores, output_file=report_file)
    
    println("\nReport saved to: $report_file")
    println("\n" * "="^60)
    println(report)
    
    # Clean up
    rm(report_file, force=true)
    
    # Additional analysis: Effect sizes
    println("\n\n--- Effect Size Analysis ---")
    
    # Calculate effect sizes for feature selection impact
    xgb_effect = calculate_effect_size(
        model_scores["XGBoost_Selected"],
        model_scores["XGBoost_All"],
        method=:cohens_d
    )
    
    rf_effect = calculate_effect_size(
        model_scores["RF_Selected"],
        model_scores["RF_All"],
        method=:cohens_d
    )
    
    println("Effect of feature selection:")
    println("  XGBoost: Cohen's d = $(round(xgb_effect, digits=4))")
    println("  RandomForest: Cohen's d = $(round(rf_effect, digits=4))")
    
    # Power analysis simulation
    println("\n--- Post-hoc Power Analysis ---")
    effect_size = 0.8  # Large effect
    n_folds = length(first(values(model_scores)))
    
    # Approximate power (simplified)
    power = calculate_approximate_power(effect_size, n_folds, 0.05)
    println("Approximate power for detecting large effect (d=0.8): $(round(power, digits=3))")
    
    return report
end

# Helper functions

function print_significant_pairs(comparisons)
    sig_pairs = [(c.model1, c.model2) for c in comparisons if c.significant_after_correction]
    
    if isempty(sig_pairs)
        println("  No significant differences found")
    else
        for (m1, m2) in sig_pairs
            comp = first(c for c in comparisons if c.model1 == m1 && c.model2 == m2)
            println("  $m1 vs $m2: p = $(round(comp.adjusted_p_value, digits=6)), effect = $(round(comp.test_result.effect_size, digits=3))")
        end
    end
end

function calculate_approximate_power(effect_size::Float64, n::Int, alpha::Float64)
    # Simplified power calculation for paired t-test
    # Uses non-central t-distribution approximation
    
    # Non-centrality parameter
    ncp = effect_size * sqrt(n)
    
    # Critical value for two-tailed test
    t_crit = quantile(TDist(n-1), 1 - alpha/2)
    
    # Power using non-central t-distribution
    # This is an approximation
    power = 1 - cdf(TDist(n-1), t_crit - ncp) + cdf(TDist(n-1), -t_crit - ncp)
    
    return power
end

# Run all examples
if abspath(PROGRAM_FILE) == @__FILE__
    println("Statistical Testing Examples for Model Comparison")
    println("=" ^ 60)
    
    # Run examples
    paired_result = paired_comparison_example()
    t_result, w_result = nonparametric_comparison_example()
    mcnemar_result = mcnemar_test_example()
    comp_bonf, comp_bh, comp_holm = multiple_comparison_example()
    ci_analytical, ci_bootstrap, perm_p = resampling_methods_example()
    report = comprehensive_report_example()
    
    println("\n" * "=" * 60)
    println("All statistical testing examples completed!")
    
    # Summary
    println("\nKey takeaways:")
    println("1. Use paired tests when comparing models on same CV folds")
    println("2. Consider non-parametric tests for non-normal distributions")
    println("3. Always correct for multiple comparisons")
    println("4. Report effect sizes alongside p-values")
    println("5. Bootstrap provides robust confidence intervals")
    println("6. Permutation tests are assumption-free alternatives")
end