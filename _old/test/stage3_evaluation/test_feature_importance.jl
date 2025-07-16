using Test
using Random
using Statistics
using DataFrames
using CSV

# Include the modules
include("../../src/stage3_evaluation/feature_importance.jl")
include("../../src/stage3_evaluation/mlj_infrastructure.jl")
include("../../src/stage3_evaluation/cross_validation.jl")

using .FeatureImportance
using .MLJInfrastructure
using .CrossValidation

# Set random seed for reproducibility
Random.seed!(42)

@testset "Feature Importance Tests" begin
    
    @testset "ImportanceResult Structure" begin
        result = ImportanceResult(
            [1, 2, 3],
            ["feat1", "feat2", "feat3"],
            [0.5, 0.3, 0.2],
            [0.4 0.6; 0.2 0.4; 0.1 0.3],
            :shap,
            Dict(:test => true)
        )
        
        @test result.feature_indices == [1, 2, 3]
        @test result.feature_names == ["feat1", "feat2", "feat3"]
        @test result.importance_values == [0.5, 0.3, 0.2]
        @test result.method == :shap
        @test result.metadata[:test] == true
    end
    
    @testset "SHAP Calculator Creation" begin
        calc = SHAPCalculator(model_type=:xgboost, n_samples=100, 
                            baseline_type=:mean, random_state=42)
        
        @test calc.model_type == :xgboost
        @test calc.n_samples == 100
        @test calc.baseline_type == :mean
        @test calc.random_state == 42
        
        # Test defaults
        calc2 = SHAPCalculator()
        @test calc2.model_type == :auto
        @test isnothing(calc2.n_samples)
        @test calc2.baseline_type == :mean
    end
    
    @testset "Permutation Importance Creation" begin
        perm = PermutationImportance(n_repeats=5, 
                                   scoring_function=accuracy,
                                   random_state=42,
                                   n_jobs=2)
        
        @test perm.n_repeats == 5
        @test perm.scoring_function == accuracy
        @test perm.random_state == 42
        @test perm.n_jobs == 2
    end
    
    @testset "Model Type Detection" begin
        # Mock model types
        @test FeatureImportance.detect_model_type("XGBoostClassifier") == :xgboost
        @test_throws ErrorException FeatureImportance.detect_model_type("UnknownModel")
    end
    
    @testset "Data Sampling" begin
        X = randn(1000, 10)
        
        # No sampling
        X_sample, indices = FeatureImportance.sample_data(X, nothing, nothing)
        @test X_sample === X
        @test indices == 1:1000
        
        # With sampling
        X_sample2, indices2 = FeatureImportance.sample_data(X, 100, 42)
        @test size(X_sample2) == (100, 10)
        @test length(indices2) == 100
        @test all(1 .<= indices2 .<= 1000)
        
        # Reproducibility
        X_sample3, indices3 = FeatureImportance.sample_data(X, 100, 42)
        @test indices2 == indices3
    end
    
    @testset "Baseline Calculation" begin
        X = randn(100, 5)
        y = randn(100)
        
        baseline_mean = FeatureImportance.calculate_baseline(X, y, :mean)
        @test baseline_mean ≈ mean(y)
        
        baseline_median = FeatureImportance.calculate_baseline(X, y, :median)
        @test baseline_median ≈ median(y)
        
        baseline_zeros = FeatureImportance.calculate_baseline(X, y, :zeros)
        @test baseline_zeros == 0.0
        
        @test_throws ErrorException FeatureImportance.calculate_baseline(X, y, :unknown)
    end
    
    @testset "Basic SHAP Calculation" begin
        # Generate simple data
        n_samples = 100
        n_features = 5
        X = randn(n_samples, n_features)
        
        # Create target with known feature importance
        # Feature 1 has high importance, feature 5 has low importance
        y = 2 * X[:, 1] + 0.5 * X[:, 2] + 0.1 * randn(n_samples)
        y_binary = Int.(y .> median(y))
        
        # Train a simple model
        model = create_model(:xgboost, :classification, n_estimators=10, max_depth=3)
        fitted_model, machine = fit_model!(model, X, y_binary, verbosity=0)
        
        # Calculate SHAP values
        shap_calc = SHAPCalculator(model_type=:xgboost, n_samples=50, random_state=42)
        shap_result = calculate_shap_values(shap_calc, fitted_model, machine, X, y_binary,
                                          feature_names=["f1", "f2", "f3", "f4", "f5"])
        
        @test length(shap_result.importance_values) == n_features
        @test !isnothing(shap_result.confidence_intervals)
        @test size(shap_result.confidence_intervals) == (n_features, 2)
        @test shap_result.method == :shap
        
        # Feature 1 should have highest importance
        @test argmax(shap_result.importance_values) == 1
        
        # Check metadata
        @test haskey(shap_result.metadata, :n_samples)
        @test shap_result.metadata[:n_samples] == 50
        @test shap_result.metadata[:model_type] == :xgboost
    end
    
    @testset "Permutation Importance Calculation" begin
        # Generate data
        n_samples = 100
        n_features = 5
        X = randn(n_samples, n_features)
        
        # Target depends mostly on first two features
        y = Int.((X[:, 1] + 0.8 * X[:, 2] + 0.1 * randn(n_samples)) .> 0)
        
        # Train model
        model = create_model(:random_forest, :classification, n_estimators=20)
        fitted_model, machine = fit_model!(model, X, y, verbosity=0)
        
        # Calculate permutation importance
        perm_calc = PermutationImportance(n_repeats=5, random_state=42, n_jobs=1)
        perm_result = calculate_permutation_importance(perm_calc, fitted_model, machine, X, y,
                                                     feature_names=["f1", "f2", "f3", "f4", "f5"])
        
        @test length(perm_result.importance_values) == n_features
        @test !isnothing(perm_result.confidence_intervals)
        @test perm_result.method == :permutation
        
        # First two features should have highest importance
        top_2_features = sortperm(perm_result.importance_values, rev=true)[1:2]
        @test 1 in top_2_features
        @test 2 in top_2_features
        
        # Check metadata
        @test perm_result.metadata[:n_repeats] == 5
        @test haskey(perm_result.metadata, :baseline_score)
    end
    
    @testset "Feature Interactions" begin
        # Create data with interaction
        n_samples = 200
        X = randn(n_samples, 4)
        
        # Target includes interaction between features 1 and 2
        y = X[:, 1] .+ X[:, 2] .+ 2 * X[:, 1] .* X[:, 2] .+ 0.1 * randn(n_samples)
        y_binary = Int.(y .> median(y))
        
        # Train model
        model = create_model(:xgboost, :classification, n_estimators=20)
        fitted_model, machine = fit_model!(model, X, y_binary, verbosity=0)
        
        # Calculate interactions
        shap_calc = SHAPCalculator(model_type=:xgboost, n_samples=50, random_state=42)
        interaction_matrix, top_interactions = get_feature_interactions(
            shap_calc, fitted_model, machine, X, y_binary, top_k=3
        )
        
        @test size(interaction_matrix.interaction_values) == (4, 4)
        @test issymmetric(interaction_matrix.interaction_values)
        @test length(top_interactions) <= 3
        
        # Diagonal should be zero (no self-interaction)
        @test all(interaction_matrix.interaction_values[i, i] == 0 for i in 1:4)
    end
    
    @testset "CV Aggregation" begin
        # Create mock importance results from different folds
        results = [
            ImportanceResult([1, 2, 3], nothing, [0.5, 0.3, 0.2], nothing, :shap, Dict()),
            ImportanceResult([1, 2, 3], nothing, [0.6, 0.2, 0.2], nothing, :shap, Dict()),
            ImportanceResult([1, 2, 3], nothing, [0.55, 0.35, 0.1], nothing, :shap, Dict())
        ]
        
        # Test mean aggregation
        agg_result = aggregate_importance_cv(results, aggregation_method=:mean)
        
        @test length(agg_result.importance_values) == 3
        @test agg_result.importance_values[1] ≈ mean([0.5, 0.6, 0.55])
        @test agg_result.importance_values[2] ≈ mean([0.3, 0.2, 0.35])
        @test agg_result.importance_values[3] ≈ mean([0.2, 0.2, 0.1])
        @test agg_result.method == :combined
        @test agg_result.metadata[:n_folds] == 3
        @test agg_result.metadata[:aggregation_method] == :mean
        
        # Test median aggregation
        agg_result_median = aggregate_importance_cv(results, aggregation_method=:median)
        @test agg_result_median.importance_values[1] ≈ median([0.5, 0.6, 0.55])
    end
    
    @testset "Combined Importance Methods" begin
        # Create mock SHAP and permutation results
        shap_result = ImportanceResult(
            [1, 2, 3],
            ["f1", "f2", "f3"],
            [0.6, 0.3, 0.1],
            [0.5 0.7; 0.2 0.4; 0.05 0.15],
            :shap,
            Dict()
        )
        
        perm_result = ImportanceResult(
            [1, 2, 3],
            ["f1", "f2", "f3"],
            [0.5, 0.4, 0.1],
            [0.4 0.6; 0.3 0.5; 0.05 0.15],
            :permutation,
            Dict()
        )
        
        # Combine with equal weights
        combined = combine_importance_methods(shap_result, perm_result, weights=(0.5, 0.5))
        
        @test length(combined.importance_values) == 3
        @test combined.method == :combined
        @test combined.metadata[:shap_weight] == 0.5
        @test combined.metadata[:permutation_weight] == 0.5
        
        # Values should be normalized and combined
        @test all(0 .<= combined.importance_values .<= 1)
        
        # Test with different weights
        combined2 = combine_importance_methods(shap_result, perm_result, weights=(0.7, 0.3))
        @test combined2.metadata[:shap_weight] == 0.7
        @test combined2.metadata[:permutation_weight] == 0.3
    end
    
    @testset "Parallel Permutation Importance" begin
        # Test with multiple threads if available
        if Threads.nthreads() > 1
            n_samples = 100
            n_features = 10
            X = randn(n_samples, n_features)
            y = Int.(X[:, 1] .> 0)
            
            model = create_model(:xgboost, :classification, n_estimators=10)
            fitted_model, machine = fit_model!(model, X, y, verbosity=0)
            
            # Sequential
            perm_seq = PermutationImportance(n_repeats=3, random_state=42, n_jobs=1)
            result_seq = calculate_permutation_importance(perm_seq, fitted_model, machine, X, y)
            
            # Parallel
            perm_par = PermutationImportance(n_repeats=3, random_state=42, n_jobs=2)
            result_par = calculate_permutation_importance(perm_par, fitted_model, machine, X, y)
            
            # Results should be similar (not exact due to parallelization)
            @test length(result_seq.importance_values) == length(result_par.importance_values)
        end
    end
    
    @testset "Scoring Functions" begin
        y_true = [0, 1, 0, 1, 1, 0, 1, 1, 0, 0]
        y_pred = [0, 1, 0, 1, 0, 0, 1, 1, 1, 0]
        
        @test accuracy(y_true, y_pred) == 0.8
        
        y_true_reg = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred_reg = [1.1, 1.9, 3.2, 3.8, 5.1]
        
        @test mse(y_true_reg, y_pred_reg) ≈ mean((y_true_reg .- y_pred_reg).^2)
        @test rmse(y_true_reg, y_pred_reg) ≈ sqrt(mse(y_true_reg, y_pred_reg))
    end
    
    @testset "Export Functionality" begin
        result = ImportanceResult(
            [1, 2, 3, 4, 5],
            ["feat1", "feat2", "feat3", "feat4", "feat5"],
            [0.5, 0.3, 0.2, 0.1, 0.05],
            nothing,
            :shap,
            Dict()
        )
        
        # Test export (without actual file writing in tests)
        temp_file = tempname() * ".csv"
        df_exported = export_importance_plot(result, temp_file, top_k=3)
        
        @test nrow(df_exported) == 3
        @test df_exported.feature[1] == "feat1"  # Should be sorted by importance
        @test df_exported.importance[1] == 0.5
        
        # Clean up
        rm(temp_file, force=true)
    end
    
    @testset "Edge Cases" begin
        # Empty aggregation
        @test_throws ErrorException aggregate_importance_cv(ImportanceResult[])
        
        # Invalid aggregation method
        results = [ImportanceResult([1], nothing, [0.5], nothing, :shap, Dict())]
        @test_throws ErrorException aggregate_importance_cv(results, aggregation_method=:invalid)
        
        # Mismatched feature counts
        shap_res = ImportanceResult([1, 2], nothing, [0.5, 0.3], nothing, :shap, Dict())
        perm_res = ImportanceResult([1, 2, 3], nothing, [0.4, 0.3, 0.2], nothing, :permutation, Dict())
        @test_throws AssertionError combine_importance_methods(shap_res, perm_res)
        
        # Invalid weights
        shap_res2 = ImportanceResult([1, 2], nothing, [0.5, 0.3], nothing, :shap, Dict())
        perm_res2 = ImportanceResult([1, 2], nothing, [0.4, 0.3], nothing, :permutation, Dict())
        @test_throws AssertionError combine_importance_methods(shap_res2, perm_res2, weights=(0.6, 0.6))
    end
end

println("All feature importance tests passed! ✓")