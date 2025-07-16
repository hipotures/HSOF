using Test
using Random
using Statistics

# Include the modules
include("../../src/stage3_evaluation/cross_validation.jl")
include("../../src/stage3_evaluation/mlj_infrastructure.jl")

using .CrossValidation
using .MLJInfrastructure

# Set random seed for reproducibility
Random.seed!(42)

@testset "Cross-Validation Tests" begin
    
    @testset "StratifiedKFold Creation" begin
        # Test basic creation
        cv = StratifiedKFold(n_folds=5)
        @test cv.n_folds == 5
        @test cv.shuffle == true
        @test isnothing(cv.random_state)
        
        # Test with parameters
        cv2 = StratifiedKFold(n_folds=3, shuffle=false, random_state=123)
        @test cv2.n_folds == 3
        @test cv2.shuffle == false
        @test cv2.random_state == 123
        
        # Test invalid parameters
        @test_throws ErrorException StratifiedKFold(n_folds=1)
    end
    
    @testset "Stratified Fold Creation" begin
        # Binary classification data
        y_binary = vcat(zeros(Int, 50), ones(Int, 50))
        cv = StratifiedKFold(n_folds=5, random_state=42)
        cv_iter = create_folds(cv, y_binary)
        
        @test cv_iter.n_folds == 5
        @test length(cv_iter.train_indices) == 5
        @test length(cv_iter.test_indices) == 5
        
        # Check fold sizes
        for i in 1:5
            @test length(cv_iter.train_indices[i]) == 80
            @test length(cv_iter.test_indices[i]) == 20
            @test isempty(intersect(cv_iter.train_indices[i], cv_iter.test_indices[i]))
        end
        
        # Check stratification
        for i in 1:5
            y_test = y_binary[cv_iter.test_indices[i]]
            @test sum(y_test .== 0) == 10
            @test sum(y_test .== 1) == 10
        end
        
        # Multi-class data
        y_multi = vcat(fill(1, 30), fill(2, 30), fill(3, 40))
        cv_iter_multi = create_folds(cv, y_multi)
        
        # Check approximate stratification
        for i in 1:5
            y_test = y_multi[cv_iter_multi.test_indices[i]]
            # Each fold should have roughly 6, 6, and 8 samples
            @test sum(y_test .== 1) >= 5 && sum(y_test .== 1) <= 7
            @test sum(y_test .== 2) >= 5 && sum(y_test .== 2) <= 7
            @test sum(y_test .== 3) >= 7 && sum(y_test .== 3) <= 9
        end
    end
    
    @testset "QuantileStratifiedKFold" begin
        # Regression data
        y_reg = randn(100) .* 10 .+ 50
        
        cv = QuantileStratifiedKFold(n_folds=5, n_quantiles=10, random_state=42)
        cv_iter = create_folds(cv, y_reg)
        
        @test cv_iter.n_folds == 5
        
        # Check that folds are approximately balanced
        fold_sizes = [length(cv_iter.test_indices[i]) for i in 1:5]
        @test all(18 .<= fold_sizes .<= 22)
        
        # Check that values are distributed across folds
        for i in 1:5
            y_test = y_reg[cv_iter.test_indices[i]]
            # Each fold should have a range of values
            @test maximum(y_test) - minimum(y_test) > 10
        end
    end
    
    @testset "Fold Validation" begin
        # Create imbalanced data
        y_imbalanced = vcat(zeros(Int, 90), ones(Int, 10))
        cv = StratifiedKFold(n_folds=10, random_state=42)
        cv_iter = create_folds(cv, y_imbalanced)
        
        # Validate with minimum 1 sample per class
        is_valid, results = validate_folds(cv_iter, y_imbalanced, min_samples_per_class=1, verbose=false)
        @test is_valid == true
        
        # Validate with minimum 2 samples per class (should fail for minority class)
        is_valid2, _ = validate_folds(cv_iter, y_imbalanced, min_samples_per_class=2, verbose=false)
        @test is_valid2 == false
    end
    
    @testset "SMOTE Sampling" begin
        # Create imbalanced dataset
        X_imb = randn(100, 5)
        y_imb = vcat(zeros(Int, 80), ones(Int, 20))
        
        sampler = SMOTESampler(sampling_ratio=0.5, k_neighbors=5, random_state=42)
        X_balanced, y_balanced = apply_smote(sampler, X_imb, y_imb)
        
        # Check that synthetic samples were added
        @test size(X_balanced, 1) > 100
        @test length(y_balanced) == size(X_balanced, 1)
        
        # Check class balance improved
        original_ratio = sum(y_imb .== 1) / length(y_imb)
        balanced_ratio = sum(y_balanced .== 1) / length(y_balanced)
        @test balanced_ratio > original_ratio
    end
    
    @testset "Fold Metrics Calculation" begin
        # Classification metrics
        y_true_class = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
        y_pred_class = [0, 1, 1, 0, 1, 1, 1, 0, 0, 0]
        
        metrics_class = get_fold_metrics(y_true_class, y_pred_class, task_type=:classification)
        
        @test haskey(metrics_class, :accuracy)
        @test haskey(metrics_class, :precision)
        @test haskey(metrics_class, :recall)
        @test haskey(metrics_class, :f1_score)
        @test metrics_class[:accuracy] ≈ 0.8
        
        # Regression metrics
        y_true_reg = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred_reg = [1.1, 1.9, 3.2, 3.8, 5.1]
        
        metrics_reg = get_fold_metrics(y_true_reg, y_pred_reg, task_type=:regression)
        
        @test haskey(metrics_reg, :mse)
        @test haskey(metrics_reg, :rmse)
        @test haskey(metrics_reg, :mae)
        @test haskey(metrics_reg, :r2)
        @test metrics_reg[:mae] < 0.3
        @test metrics_reg[:r2] > 0.95
    end
    
    @testset "Cross-Validation Integration" begin
        # Generate data
        X = randn(100, 5)
        y = Int.(X[:, 1] .+ 0.5 * X[:, 2] .> 0)
        
        # Create model
        model = create_model(:xgboost, :classification, n_estimators=10)
        
        # Run cross-validation
        cv = StratifiedKFold(n_folds=3, random_state=42)
        results = cross_validate_model(model.model, X, y, cv, 
                                     task_type=:classification, 
                                     verbose=false)
        
        @test haskey(results, "fold_metrics")
        @test haskey(results, "aggregated_metrics")
        @test length(results["fold_metrics"]) == 3
        
        # Check aggregated metrics
        agg_metrics = results["aggregated_metrics"]
        @test haskey(agg_metrics, :accuracy_mean)
        @test haskey(agg_metrics, :accuracy_std)
        @test agg_metrics[:n_folds] == 3
        @test agg_metrics[:accuracy_mean] > 0.5
    end
    
    @testset "Reproducibility" begin
        y = vcat(zeros(Int, 50), ones(Int, 50))
        
        # Create two CV splitters with same seed
        cv1 = StratifiedKFold(n_folds=5, random_state=123)
        cv2 = StratifiedKFold(n_folds=5, random_state=123)
        
        iter1 = create_folds(cv1, y)
        iter2 = create_folds(cv2, y)
        
        # Check that folds are identical
        for i in 1:5
            @test iter1.train_indices[i] == iter2.train_indices[i]
            @test iter1.test_indices[i] == iter2.test_indices[i]
        end
        
        # Different seed should give different folds
        cv3 = StratifiedKFold(n_folds=5, random_state=456)
        iter3 = create_folds(cv3, y)
        
        # Very unlikely to be identical
        @test any(iter1.test_indices[i] != iter3.test_indices[i] for i in 1:5)
    end
    
    @testset "Utility Functions" begin
        # Test get_reproducible_cv
        cv_strat = get_reproducible_cv(:stratified, 5, seed=42)
        @test cv_strat isa StratifiedKFold
        @test cv_strat.n_folds == 5
        @test cv_strat.random_state == 42
        
        cv_quant = get_reproducible_cv(:quantile_stratified, 3, seed=123, n_quantiles=20)
        @test cv_quant isa QuantileStratifiedKFold
        @test cv_quant.n_folds == 3
        @test cv_quant.n_quantiles == 20
        
        @test_throws ErrorException get_reproducible_cv(:unknown, 5)
    end
    
    @testset "Edge Cases" begin
        # Very small dataset
        y_small = [0, 1, 0, 1, 0]
        cv = StratifiedKFold(n_folds=2, random_state=42)
        cv_iter = create_folds(cv, y_small)
        
        @test cv_iter.n_folds == 2
        # Each fold should have at least one sample from each class
        for i in 1:2
            y_train = y_small[cv_iter.train_indices[i]]
            @test 0 in y_train
            @test 1 in y_train
        end
        
        # Single class data (should still work but not stratify)
        y_single = ones(Int, 20)
        cv_iter_single = create_folds(cv, y_single)
        @test cv_iter_single.n_folds == 2
        
        # Empty data handling
        @test_throws BoundsError create_folds(cv, Int[])
    end
end

println("All cross-validation tests passed! ✓")