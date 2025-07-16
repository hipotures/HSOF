using Test
using Random
using Statistics
using Distributions

# Include the module
include("../../src/stage3_evaluation/statistical_testing.jl")

using .StatisticalTesting
import .StatisticalTesting: BONFERRONI, BENJAMINI_HOCHBERG, HOLM, NONE

# Set random seed for reproducibility
Random.seed!(42)

@testset "Statistical Testing Tests" begin
    
    @testset "TestResult Structure" begin
        result = TestResult(
            "Test Name",
            1.96,
            0.05,
            (0.1, 0.5),
            0.8,
            true,
            0.05,
            Dict(:test => true)
        )
        
        @test result.test_name == "Test Name"
        @test result.statistic == 1.96
        @test result.p_value == 0.05
        @test result.confidence_interval == (0.1, 0.5)
        @test result.effect_size == 0.8
        @test result.significant == true
        @test result.alpha == 0.05
        @test result.metadata[:test] == true
    end
    
    @testset "Paired t-test" begin
        # Generate paired data with known difference
        n = 20
        scores1 = randn(n) .+ 1.0  # Mean = 1
        scores2 = randn(n)         # Mean = 0
        
        # Two-sided test
        result = paired_t_test(scores1, scores2, alpha=0.05)
        
        @test result.test_name == "Paired t-test"
        @test result.p_value < 0.05  # Should be significant
        @test result.significant == true
        @test !isnothing(result.confidence_interval)
        @test result.confidence_interval[1] < result.confidence_interval[2]
        @test !isnothing(result.effect_size)
        @test result.metadata[:n_pairs] == n
        @test result.metadata[:degrees_of_freedom] == n - 1
        
        # One-sided test (greater)
        result_greater = paired_t_test(scores1, scores2, alpha=0.05, alternative=:greater)
        @test result_greater.p_value < result.p_value  # One-sided should have smaller p-value
        
        # Test with identical scores (no difference)
        scores_same = randn(n)
        result_same = paired_t_test(scores_same, scores_same, alpha=0.05)
        @test result_same.p_value ≈ 1.0
        @test result_same.significant == false
        @test abs(result_same.statistic) < 0.001
        
        # Test error handling
        @test_throws AssertionError paired_t_test([1.0], [2.0])  # Too few observations
        @test_throws AssertionError paired_t_test([1.0, 2.0], [1.0])  # Unequal lengths
    end
    
    @testset "Wilcoxon Signed-Rank Test" begin
        # Generate paired data
        n = 15
        scores1 = randn(n) .+ 0.5
        scores2 = randn(n)
        
        # Basic test
        result = wilcoxon_signed_rank_test(scores1, scores2, alpha=0.05)
        
        @test result.test_name == "Wilcoxon signed-rank test"
        @test result.p_value >= 0.0 && result.p_value <= 1.0
        @test isa(result.significant, Bool)
        @test !isnothing(result.effect_size)  # Rank-biserial correlation
        @test abs(result.effect_size) <= 1.0
        
        # Test with all differences zero
        scores_same = ones(n)
        result_same = wilcoxon_signed_rank_test(scores_same, scores_same)
        @test result_same.p_value == 1.0
        @test result_same.significant == false
        @test result_same.metadata[:n_non_zero] == 0
        
        # Test with large sample (normal approximation)
        n_large = 50
        scores1_large = randn(n_large) .+ 0.3
        scores2_large = randn(n_large)
        result_large = wilcoxon_signed_rank_test(scores1_large, scores2_large)
        @test result_large.metadata[:n_pairs] == n_large
    end
    
    @testset "McNemar's Test" begin
        # Generate binary classification results
        n = 100
        y_true = rand(Bool, n)
        
        # Model 1: 80% accuracy
        pred1 = copy(y_true)
        wrong_idx1 = randperm(n)[1:20]
        pred1[wrong_idx1] .= .!pred1[wrong_idx1]
        
        # Model 2: 75% accuracy with some different errors
        pred2 = copy(y_true)
        wrong_idx2 = randperm(n)[1:25]
        pred2[wrong_idx2] .= .!pred2[wrong_idx2]
        
        result = mcnemar_test(pred1, pred2, y_true, alpha=0.05)
        
        @test result.test_name == "McNemar's test"
        @test result.p_value >= 0.0 && result.p_value <= 1.0
        @test !isnothing(result.confidence_interval)  # CI for odds ratio
        @test haskey(result.metadata, :contingency_table)
        @test haskey(result.metadata, :odds_ratio)
        @test size(result.metadata[:contingency_table]) == (2, 2)
        
        # Test with identical predictions
        result_same = mcnemar_test(pred1, pred1, y_true)
        @test result_same.p_value == 1.0
        @test result_same.metadata[:discordant_pairs] == 0
    end
    
    @testset "Confidence Interval Calculation" begin
        # Generate sample data
        scores = randn(30) .+ 2.0
        
        # Analytical CI
        ci_analytical = calculate_confidence_interval(scores, alpha=0.05, method=:analytical)
        @test ci_analytical[1] < mean(scores) < ci_analytical[2]
        @test ci_analytical[1] < ci_analytical[2]
        
        # Bootstrap CI
        ci_bootstrap = calculate_confidence_interval(scores, alpha=0.05, method=:bootstrap)
        @test ci_bootstrap[1] < mean(scores) < ci_bootstrap[2]
        @test ci_bootstrap[1] < ci_bootstrap[2]
        
        # Bootstrap should be similar to analytical for normal data
        @test abs(ci_analytical[1] - ci_bootstrap[1]) < 0.5
        @test abs(ci_analytical[2] - ci_bootstrap[2]) < 0.5
    end
    
    @testset "Effect Size Calculation" begin
        # Generate two groups with known difference
        n1, n2 = 25, 30
        group1 = randn(n1) .+ 1.0  # Mean ≈ 1, SD ≈ 1
        group2 = randn(n2)         # Mean ≈ 0, SD ≈ 1
        
        # Cohen's d
        d = calculate_effect_size(group1, group2, method=:cohens_d)
        @test abs(d - 1.0) < 0.5  # Should be approximately 1
        
        # Glass's delta
        delta = calculate_effect_size(group1, group2, method=:glass_delta)
        @test abs(delta - 1.0) < 0.5
        
        # Hedges' g
        g = calculate_effect_size(group1, group2, method=:hedges_g)
        @test abs(g) < abs(d)  # Hedges' g should be slightly smaller
        
        # Test with no difference
        d_same = calculate_effect_size(group1, group1, method=:cohens_d)
        @test abs(d_same) < 0.001
    end
    
    @testset "Multiple Testing Corrections" begin
        # Generate p-values with some significant
        p_values = [0.001, 0.01, 0.02, 0.04, 0.06, 0.1, 0.5, 0.8]
        alpha = 0.05
        
        # Bonferroni correction
        p_bonf, sig_bonf, alpha_bonf = bonferroni_correction(p_values, alpha)
        @test all(p_bonf .>= p_values)  # Adjusted p-values should be larger
        @test alpha_bonf == alpha / length(p_values)
        @test sum(sig_bonf) <= sum(p_values .< alpha)  # Fewer significant after correction
        
        # Benjamini-Hochberg correction
        p_bh, sig_bh, alpha_bh = benjamini_hochberg_correction(p_values, alpha)
        @test all(p_bh .>= p_values)
        @test sum(sig_bh) >= sum(sig_bonf)  # BH is less conservative than Bonferroni
        
        # Holm correction
        p_holm, sig_holm, alpha_holm = holm_correction(p_values, alpha)
        @test all(p_holm .>= p_values)
        @test sum(sig_holm) >= sum(sig_bonf)  # Holm is less conservative than Bonferroni
        @test sum(sig_holm) <= sum(sig_bh)    # But more conservative than BH
        
        # Test with all p-values > alpha
        p_high = fill(0.1, 5)
        _, sig_high, _ = bonferroni_correction(p_high, 0.05)
        @test sum(sig_high) == 0
    end
    
    @testset "Bootstrap Confidence Interval" begin
        # Generate sample
        data = randn(50) .+ 3.0
        
        # Calculate bootstrap CI
        ci = bootstrap_confidence_interval(data, n_bootstrap=1000, alpha=0.05)
        
        @test ci[1] < mean(data) < ci[2]
        @test ci[2] - ci[1] > 0  # Positive width
        
        # Compare with analytical
        ci_analytical = calculate_confidence_interval(data, method=:analytical)
        @test abs(ci[1] - ci_analytical[1]) < 0.2  # Should be similar
        @test abs(ci[2] - ci_analytical[2]) < 0.2
    end
    
    @testset "Permutation Test" begin
        # Generate two independent groups
        group1 = randn(20) .+ 0.5
        group2 = randn(25)
        
        # Permutation test for mean difference
        p_value, obs_stat, perm_stats = permutation_test(group1, group2, 
                                                         n_permutations=1000)
        
        @test p_value >= 0.0 && p_value <= 1.0
        @test length(perm_stats) == 1000
        @test obs_stat ≈ mean(group1) - mean(group2)
        
        # Test with identical distributions
        group_same = randn(20)
        p_same, _, _ = permutation_test(group_same, group_same, n_permutations=1000)
        @test p_same > 0.5  # Should not be significant
        
        # One-sided test
        p_greater, _, _ = permutation_test(group1, group2, 
                                          n_permutations=1000,
                                          alternative=:greater)
        @test p_greater <= p_value  # One-sided should have smaller p-value
    end
    
    @testset "Model Comparison" begin
        # Generate CV scores for multiple models
        n_folds = 10
        model_scores = Dict(
            "Model_A" => randn(n_folds) .* 0.05 .+ 0.85,  # Mean ≈ 0.85
            "Model_B" => randn(n_folds) .* 0.05 .+ 0.82,  # Mean ≈ 0.82
            "Model_C" => randn(n_folds) .* 0.05 .+ 0.80,  # Mean ≈ 0.80
            "Model_D" => randn(n_folds) .* 0.05 .+ 0.85   # Mean ≈ 0.85 (same as A)
        )
        
        # Compare with paired t-test and Bonferroni correction
        comparisons, adj_alpha = compare_models(model_scores, 
                                               test_type=:paired_t,
                                               correction=BONFERRONI,
                                               alpha=0.05)
        
        @test length(comparisons) == 6  # C(4,2) = 6 pairwise comparisons
        @test adj_alpha < 0.05  # Adjusted alpha should be smaller
        
        # Check structure
        for comp in comparisons
            @test haskey(comp, :model1)
            @test haskey(comp, :model2)
            @test haskey(comp, :test_result)
            @test haskey(comp, :adjusted_p_value)
            @test haskey(comp, :significant_after_correction)
        end
        
        # Compare with Wilcoxon test
        comparisons_wilcox, _ = compare_models(model_scores,
                                              test_type=:wilcoxon,
                                              correction=BENJAMINI_HOCHBERG)
        @test length(comparisons_wilcox) == 6
        
        # Test with no correction
        comparisons_none, _ = compare_models(model_scores,
                                           test_type=:paired_t,
                                           correction=NONE)
        # More comparisons should be significant without correction
        sig_with_correction = sum(c.significant_after_correction for c in comparisons)
        sig_without = sum(c.significant_after_correction for c in comparisons_none)
        @test sig_without >= sig_with_correction
    end
    
    @testset "Statistical Report Generation" begin
        # Setup model scores and comparisons
        model_scores = Dict(
            "XGBoost" => [0.85, 0.87, 0.86, 0.84, 0.88],
            "RandomForest" => [0.82, 0.83, 0.81, 0.84, 0.82],
            "LightGBM" => [0.86, 0.85, 0.87, 0.85, 0.86]
        )
        
        comparisons, _ = compare_models(model_scores, 
                                       test_type=:paired_t,
                                       correction=BONFERRONI)
        
        # Generate report
        report = generate_statistical_report(comparisons, model_scores)
        
        @test isa(report, String)
        @test occursin("STATISTICAL ANALYSIS REPORT", report)
        @test occursin("MODEL PERFORMANCE SUMMARY", report)
        @test occursin("PAIRWISE MODEL COMPARISONS", report)
        @test occursin("SIGNIFICANCE SUMMARY", report)
        
        # Check all models are mentioned
        for model in keys(model_scores)
            @test occursin(model, report)
        end
        
        # Test file output
        temp_file = tempname() * ".txt"
        report_file = generate_statistical_report(comparisons, model_scores,
                                                output_file=temp_file)
        @test isfile(temp_file)
        @test read(temp_file, String) == report_file
        
        # Clean up
        rm(temp_file, force=true)
    end
    
    @testset "Edge Cases and Error Handling" begin
        # Empty vectors
        @test_throws AssertionError paired_t_test(Float64[], Float64[])
        
        # Single observation
        @test_throws AssertionError paired_t_test([1.0], [2.0])
        
        # Unknown alternative hypothesis  
        # Using different values to avoid std_diff = 0 case
        @test_throws ErrorException paired_t_test([1.0,2.5,3.0], [2.0,3.0,4.2], alternative=:unknown)
        
        # Unknown effect size method
        @test_throws ErrorException calculate_effect_size([1,2,3], [2,3,4], method=:unknown)
        
        # Unknown CI method
        @test_throws ErrorException calculate_confidence_interval([1,2,3], method=:unknown)
        
        # Unknown test type in compare_models
        model_scores = Dict("A" => [1,2,3], "B" => [2,3,4])
        @test_throws ErrorException compare_models(model_scores, test_type=:unknown)
    end
end

println("All statistical testing tests passed! ✓")