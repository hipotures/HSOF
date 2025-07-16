using Test
using Random
using DataFrames

# Include the modules
include("../../src/stage3_evaluation/mlj_infrastructure.jl")
include("../../src/stage3_evaluation/unified_prediction.jl")

using .MLJInfrastructure
using .UnifiedPrediction

# Set random seed for reproducibility
Random.seed!(42)

@testset "MLJ Infrastructure Tests" begin
    
    @testset "Model Configuration" begin
        # Test default configurations
        xgb_config = get_default_config(:xgboost, :classification)
        @test xgb_config.model_type == :xgboost
        @test xgb_config.task_type == :classification
        @test xgb_config.params[:max_depth] == 6
        @test xgb_config.params[:learning_rate] == 0.3
        
        rf_config = get_default_config(:random_forest, :regression)
        @test rf_config.model_type == :random_forest
        @test rf_config.task_type == :regression
        @test rf_config.params[:n_estimators] == 100
        
        lgbm_config = get_default_config(:lightgbm, :classification)
        @test lgbm_config.model_type == :lightgbm
        @test lgbm_config.params[:num_leaves] == 31
        
        # Test configuration updates
        update_config!(xgb_config, max_depth=10, learning_rate=0.1)
        @test xgb_config.params[:max_depth] == 10
        @test xgb_config.params[:learning_rate] == 0.1
    end
    
    @testset "Model Creation" begin
        # Test XGBoost wrapper creation
        xgb_wrapper = create_model(:xgboost, :classification)
        @test xgb_wrapper isa XGBoostWrapper
        @test !xgb_wrapper.fitted[]
        
        # Test Random Forest wrapper creation
        rf_wrapper = create_model(:random_forest, :regression)
        @test rf_wrapper isa RandomForestWrapper
        @test !rf_wrapper.fitted[]
        
        # Test LightGBM wrapper creation
        lgbm_wrapper = create_model(:lightgbm, :classification)
        @test lgbm_wrapper isa LightGBMWrapper
        @test !lgbm_wrapper.fitted[]
        
        # Test with custom parameters
        custom_xgb = create_model(:xgboost, :regression, max_depth=10, n_estimators=200)
        @test custom_xgb.config.params[:max_depth] == 10
        @test custom_xgb.config.params[:n_estimators] == 200
    end
    
    @testset "Parameter Validation" begin
        # Test invalid parameters
        config = get_default_config(:xgboost, :classification)
        
        # Test invalid max_depth
        config.params[:max_depth] = -1
        @test_throws ErrorException validate_config(config)
        
        # Test invalid learning_rate
        config.params[:max_depth] = 6  # Reset
        config.params[:learning_rate] = 1.5
        @test_throws ErrorException validate_config(config)
        
        # Test valid parameters
        config.params[:learning_rate] = 0.1
        @test validate_config(config) == true
    end
    
    @testset "Model Fitting and Prediction" begin
        # Generate synthetic data
        n_samples = 100
        n_features = 10
        X = randn(n_samples, n_features)
        
        # Classification data
        y_class = rand([0, 1], n_samples)
        
        # Regression data  
        y_reg = X[:, 1] .+ 0.5 .* X[:, 2] .+ 0.1 .* randn(n_samples)
        
        @testset "Classification" begin
            # Create and fit XGBoost classifier
            xgb_model = create_model(:xgboost, :classification, n_estimators=10)
            xgb_model_fitted, xgb_mach = fit_model!(xgb_model, X, y_class, verbosity=0)
            
            @test xgb_model_fitted.fitted[] == true
            
            # Make predictions
            predictions = predict_model(xgb_model_fitted, xgb_mach, X)
            @test length(predictions) == n_samples
            
            # Test unified predictor
            predictor = UnifiedPredictor(xgb_model_fitted, xgb_mach, n_features=n_features)
            result = predict_unified(predictor, X)
            
            @test result.task_type == :classification
            @test length(result.predictions) == n_samples
            @test !isnothing(result.confidence_scores)
        end
        
        @testset "Regression" begin
            # Create and fit Random Forest regressor
            rf_model = create_model(:random_forest, :regression, n_estimators=10)
            rf_model_fitted, rf_mach = fit_model!(rf_model, X, y_reg, verbosity=0)
            
            @test rf_model_fitted.fitted[] == true
            
            # Make predictions
            predictions = predict_model(rf_model_fitted, rf_mach, X)
            @test length(predictions) == n_samples
            
            # Test unified predictor
            predictor = UnifiedPredictor(rf_model_fitted, rf_mach, n_features=n_features)
            result = predict_unified(predictor, X)
            
            @test result.task_type == :regression
            @test length(result.predictions) == n_samples
        end
    end
    
    @testset "Model Serialization" begin
        # Create simple model
        model = create_model(:xgboost, :classification, n_estimators=5)
        
        # Generate data and fit
        X = randn(50, 5)
        y = rand([0, 1], 50)
        model_fitted, mach = fit_model!(model, X, y, verbosity=0)
        
        # Test save and load
        temp_file = tempname() * ".jld2"
        
        try
            # Save model
            save_model(model_fitted, mach, temp_file)
            @test isfile(temp_file)
            
            # Load model
            loaded_model, loaded_mach = load_model(temp_file)
            @test loaded_model.config.model_type == :xgboost
            @test loaded_model.fitted[] == true
            
            # Make predictions with loaded model
            predictions = predict_model(loaded_model, loaded_mach, X)
            @test length(predictions) == 50
        finally
            # Cleanup
            rm(temp_file, force=true)
        end
    end
    
    @testset "Ensemble Predictions" begin
        # Generate data
        X = randn(50, 5)
        y = rand([0, 1], 50)
        
        # Create multiple models
        models = [
            create_model(:xgboost, :classification, n_estimators=5),
            create_model(:random_forest, :classification, n_estimators=5),
            create_model(:lightgbm, :classification, n_estimators=5)
        ]
        
        # Fit models
        predictors = UnifiedPredictor[]
        for model in models
            fitted_model, mach = fit_model!(model, X, y, verbosity=0)
            push!(predictors, UnifiedPredictor(fitted_model, mach))
        end
        
        # Test ensemble prediction
        @testset "Hard Voting" begin
            ensemble_preds = ensemble_predict(predictors, X, voting=:hard)
            @test length(ensemble_preds) == 50
        end
        
        @testset "Soft Voting" begin
            ensemble_preds = ensemble_predict(predictors, X, voting=:soft)
            @test length(ensemble_preds) == 50
        end
        
        @testset "Weighted Voting" begin
            weights = [0.5, 0.3, 0.2]
            ensemble_preds = ensemble_predict(predictors, X, voting=:soft, weights=weights)
            @test length(ensemble_preds) == 50
        end
    end
    
    @testset "Prediction Metrics" begin
        # Classification metrics
        y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
        y_pred = [0, 1, 1, 0, 1, 1, 1, 0, 0, 0]
        
        result = PredictionResult(
            y_pred,
            nothing,
            fill(0.8, 10),
            nothing,
            :xgboost,
            :classification,
            Dict{Symbol, Any}()
        )
        
        metrics = get_prediction_metrics(result, y_true)
        @test haskey(metrics, :accuracy)
        @test haskey(metrics, :precision)
        @test haskey(metrics, :recall)
        @test haskey(metrics, :f1_score)
        @test metrics[:accuracy] ≈ 0.8
        
        # Regression metrics
        y_true_reg = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred_reg = [1.1, 1.9, 3.2, 3.8, 5.1]
        
        result_reg = PredictionResult(
            y_pred_reg,
            nothing,
            nothing,
            nothing,
            :random_forest,
            :regression,
            Dict{Symbol, Any}()
        )
        
        metrics_reg = get_prediction_metrics(result_reg, y_true_reg)
        @test haskey(metrics_reg, :mse)
        @test haskey(metrics_reg, :rmse)
        @test haskey(metrics_reg, :mae)
        @test haskey(metrics_reg, :r2)
        @test metrics_reg[:mae] < 0.3
    end
end

println("All MLJ infrastructure tests passed! ✓")