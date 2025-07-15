using Test
using Random
using Statistics
using DataFrames
using Dates
using JSON
using CSV

# Include the module
include("../../src/stage3_evaluation/model_comparison.jl")

using .ModelComparison

# Set random seed for reproducibility
Random.seed!(42)

@testset "Model Comparison Tests" begin
    
    @testset "Configuration Tests" begin
        # Test ComparisonConfig
        config = ComparisonConfig()
        @test config.metrics == [:accuracy, :f1, :auc]
        @test config.statistical_test == :paired_t
        @test config.confidence_level == 0.95
        @test config.include_variance == true
        @test config.include_timing == true
        
        # Test custom config
        custom_config = ComparisonConfig(
            metrics = [:rmse, :mae],
            statistical_test = :wilcoxon,
            confidence_level = 0.99,
            include_variance = false
        )
        @test custom_config.metrics == [:rmse, :mae]
        @test custom_config.statistical_test == :wilcoxon
        @test custom_config.confidence_level == 0.99
        
        # Test VisualizationConfig
        viz_config = VisualizationConfig()
        @test viz_config.plot_size == (800, 600)
        @test viz_config.color_scheme == :viridis
        @test viz_config.save_plots == true
        @test viz_config.dpi == 300
        
        # Test ExportConfig
        export_config = ExportConfig()
        @test export_config.formats == [:json, :csv]
        @test export_config.include_metadata == true
        @test export_config.timestamp_files == true
    end
    
    @testset "Model Comparison Tests" begin
        # Create mock model results
        n_folds = 5
        model_results = Dict{String, Dict{Symbol, Vector{Float64}}}(
            "XGBoost" => Dict(
                :accuracy => [0.92, 0.91, 0.93, 0.90, 0.92],
                :f1 => [0.91, 0.90, 0.92, 0.89, 0.91],
                :auc => [0.95, 0.94, 0.96, 0.93, 0.95],
                :training_time => [2.5, 2.3, 2.6, 2.4, 2.5]
            ),
            "RandomForest" => Dict(
                :accuracy => [0.89, 0.88, 0.90, 0.87, 0.89],
                :f1 => [0.88, 0.87, 0.89, 0.86, 0.88],
                :auc => [0.92, 0.91, 0.93, 0.90, 0.92],
                :training_time => [3.1, 3.0, 3.2, 3.0, 3.1]
            ),
            "LightGBM" => Dict(
                :accuracy => [0.91, 0.90, 0.92, 0.89, 0.91],
                :f1 => [0.90, 0.89, 0.91, 0.88, 0.90],
                :auc => [0.94, 0.93, 0.95, 0.92, 0.94],
                :training_time => [1.8, 1.7, 1.9, 1.7, 1.8]
            )
        )
        
        # Compare models
        result = compare_models(model_results)
        
        @test isa(result, ModelComparisonResult)
        @test Set(result.model_names) == Set(["XGBoost", "RandomForest", "LightGBM"])
        @test haskey(result.metrics, :accuracy)
        @test haskey(result.metrics, :f1)
        @test haskey(result.metrics, :auc)
        
        # Check aggregated metrics
        @test haskey(result.aggregated_metrics, :accuracy)
        @test length(result.aggregated_metrics[:accuracy]) == 3
        @test result.aggregated_metrics[:accuracy]["XGBoost"] ≈ mean([0.92, 0.91, 0.93, 0.90, 0.92])
        
        # Check confidence intervals
        @test haskey(result.confidence_intervals, :accuracy)
        @test haskey(result.confidence_intervals[:accuracy], "XGBoost")
        ci = result.confidence_intervals[:accuracy]["XGBoost"]
        @test ci[1] < result.aggregated_metrics[:accuracy]["XGBoost"] < ci[2]
        
        # Check timing info
        @test !isempty(result.timing_info)
        @test result.timing_info["LightGBM"] < result.timing_info["XGBoost"]
        
        # Check metadata
        @test haskey(result.metadata, :comparison_date)
        @test result.metadata[:n_models] == 3
    end
    
    @testset "Feature Set Comparison Tests" begin
        # Create mock data
        n_samples, n_features = 100, 20
        X = randn(n_samples, n_features)
        y = rand(Bool, n_samples)
        
        # Define feature sets
        feature_sets = Dict(
            "All_Features" => collect(1:20),
            "Top_10" => collect(1:10),
            "Selected_8" => [1, 3, 5, 7, 9, 11, 13, 15],
            "Random_5" => [2, 7, 11, 15, 19]
        )
        
        # Mock model factory
        model_factory = () -> "MockModel"
        
        # Compare feature sets
        result = compare_feature_sets(feature_sets, X, y, model_factory, cv_folds=3)
        
        @test isa(result, FeatureSetComparison)
        @test result.feature_sets == feature_sets
        @test length(result.performance) == 4
        @test haskey(result.performance, "All_Features")
        @test haskey(result.performance_delta, ("All_Features", "Top_10"))
        @test !isempty(result.best_feature_set)
    end
    
    @testset "Ensemble Model Tests" begin
        # Create mock models
        models = ["Model1", "Model2", "Model3", "Model4"]
        performances = [0.92, 0.89, 0.91, 0.88]
        
        # Create ensemble
        ensemble = create_model_ensemble(models, performances, 
                                       voting_method=:weighted, top_k=3)
        
        @test isa(ensemble, EnsembleModel)
        @test length(ensemble.models) == 3
        @test length(ensemble.weights) == 3
        @test sum(ensemble.weights) ≈ 1.0
        @test ensemble.voting_method == :weighted
        @test ensemble.task_type == :classification
        
        # Test equal weight ensemble
        ensemble_equal = create_model_ensemble(models, performances,
                                             voting_method=:hard, top_k=2)
        @test all(ensemble_equal.weights .≈ 0.5)
    end
    
    @testset "LaTeX Table Generation Tests" begin
        # Create simple comparison result
        model_results = Dict(
            "Model_A" => Dict(:accuracy => [0.85], :f1 => [0.84]),
            "Model_B" => Dict(:accuracy => [0.87], :f1 => [0.86])
        )
        
        result = compare_models(model_results)
        latex_table = generate_latex_tables(result)
        
        @test isa(latex_table, String)
        @test occursin("\\begin{table}", latex_table)
        @test occursin("\\begin{tabular}", latex_table)
        @test occursin("Model\\_A", latex_table)
        @test occursin("Model\\_B", latex_table)
        @test occursin("\\textbf{", latex_table)  # Best model should be bold
        @test occursin("\\hline", latex_table)
        @test occursin("\\end{table}", latex_table)
    end
    
    @testset "Executive Summary Tests" begin
        # Create mock results
        model_results = Dict(
            "XGBoost" => Dict(
                :accuracy => [0.92, 0.91, 0.93],
                :f1 => [0.91, 0.90, 0.92]
            ),
            "RandomForest" => Dict(
                :accuracy => [0.88, 0.87, 0.89],
                :f1 => [0.87, 0.86, 0.88]
            )
        )
        
        comparison_result = compare_models(model_results)
        
        # Generate summary
        summary = generate_executive_summary(comparison_result)
        
        @test isa(summary, String)
        @test occursin("EXECUTIVE SUMMARY", summary)
        @test occursin("KEY FINDINGS", summary)
        @test occursin("Best Performing Model", summary)
        @test occursin("RECOMMENDATIONS", summary)
        @test occursin("TECHNICAL DETAILS", summary)
        @test occursin("XGBoost", summary)
        @test occursin("RandomForest", summary)
        
        # Test with feature selection result
        mock_feature_result = (
            selected_features = [1, 3, 5, 7, 9],
            metadata = Dict(:selection_method => :pareto, :performance => 0.91)
        )
        
        summary_with_features = generate_executive_summary(
            comparison_result, mock_feature_result
        )
        @test occursin("Feature Selection Results", summary_with_features)
        @test occursin("Features selected: 5", summary_with_features)
    end
    
    @testset "Export Functions Tests" begin
        # Create simple comparison result
        model_results = Dict(
            "Model1" => Dict(:accuracy => [0.85, 0.86], :f1 => [0.84, 0.85]),
            "Model2" => Dict(:accuracy => [0.88, 0.89], :f1 => [0.87, 0.88])
        )
        
        result = compare_models(model_results)
        
        # Test export without timestamp
        export_config = ExportConfig(
            formats = [:json, :csv],
            timestamp_files = false
        )
        
        temp_prefix = tempname()
        exported_files = export_results(result, temp_prefix, config=export_config)
        
        @test length(exported_files) == 2
        @test any(endswith(f, ".json") for f in exported_files)
        @test any(endswith(f, ".csv") for f in exported_files)
        
        # Clean up
        for file in exported_files
            rm(file, force=true)
        end
        
        # Test LaTeX export
        export_config_latex = ExportConfig(
            formats = [:latex],
            timestamp_files = false
        )
        
        exported_latex = export_results(result, temp_prefix, config=export_config_latex)
        @test length(exported_latex) == 1
        @test endswith(exported_latex[1], ".tex")
        
        # Read and check LaTeX content
        latex_content = read(exported_latex[1], String)
        @test occursin("\\begin{table}", latex_content)
        
        # Clean up
        rm(exported_latex[1], force=true)
    end
    
    @testset "Helper Functions Tests" begin
        # Test find_best_model
        result = ModelComparisonResult(
            ["A", "B", "C"],
            Dict{Symbol, Matrix{Float64}}(),
            Dict(:accuracy => Dict("A" => 0.85, "B" => 0.90, "C" => 0.88),
                 :rmse => Dict("A" => 0.15, "B" => 0.12, "C" => 0.14)),
            Dict{Symbol, Dict{String, Tuple{Float64, Float64}}}(),
            Dict{Tuple{String, String}, Dict{Symbol, Any}}(),
            Dict{String, Float64}(),
            Dict{Symbol, Any}()
        )
        
        @test ModelComparison.find_best_model(result, :accuracy) == "B"
        @test ModelComparison.find_best_model(result, :rmse) == "B"  # Lower is better
        
        # Test LaTeX escape
        @test ModelComparison.latex_escape("test_name") == "test\\_name"
        @test ModelComparison.latex_escape("50%") == "50\\%"
        @test ModelComparison.latex_escape("A & B") == "A \\& B"
        
        # Test ROC curve calculation
        y_true = [1, 1, 0, 0, 1, 0, 1, 0]
        y_scores = [0.9, 0.8, 0.3, 0.2, 0.7, 0.4, 0.6, 0.1]
        
        fpr, tpr, thresholds = ModelComparison.roc_curve(y_true, y_scores)
        @test length(fpr) == length(tpr)
        @test fpr[1] == 0.0 && fpr[end] == 1.0
        @test tpr[1] == 0.0 && tpr[end] == 1.0
        @test issorted(fpr)
        
        # Test AUC calculation
        auc_score = ModelComparison.auc(fpr, tpr)
        @test 0.0 <= auc_score <= 1.0
        
        # Test confusion matrix
        y_true = [1, 1, 2, 2, 3, 3]
        y_pred = [1, 2, 2, 2, 3, 1]
        cm = ModelComparison.confusion_matrix(y_true, y_pred)
        @test size(cm) == (3, 3)
        @test sum(cm) == length(y_true)
        @test cm[1, 1] == 1  # True positives for class 1
        @test cm[2, 2] == 2  # True positives for class 2
    end
    
    @testset "Integration Tests" begin
        # Full pipeline test
        n_folds = 5
        
        # Create comprehensive model results
        model_results = Dict{String, Dict{Symbol, Vector{Float64}}}()
        
        for model in ["XGBoost", "RandomForest", "LightGBM", "SVM"]
            base_acc = model == "XGBoost" ? 0.92 : 
                      model == "RandomForest" ? 0.89 :
                      model == "LightGBM" ? 0.91 : 0.87
            
            model_results[model] = Dict(
                :accuracy => [base_acc + 0.01*randn() for _ in 1:n_folds],
                :f1 => [base_acc - 0.01 + 0.01*randn() for _ in 1:n_folds],
                :auc => [base_acc + 0.03 + 0.01*randn() for _ in 1:n_folds],
                :precision => [base_acc + 0.01*randn() for _ in 1:n_folds],
                :recall => [base_acc - 0.02 + 0.01*randn() for _ in 1:n_folds],
                :training_time => [2.0 + 0.5*randn() for _ in 1:n_folds]
            )
        end
        
        # Configure comparison
        config = ComparisonConfig(
            metrics = [:accuracy, :f1, :auc, :precision, :recall],
            statistical_test = :paired_t,
            confidence_level = 0.95,
            include_variance = true,
            include_timing = true
        )
        
        # Compare models
        comparison_result = compare_models(model_results, config=config)
        
        @test isa(comparison_result, ModelComparisonResult)
        @test length(comparison_result.model_names) == 4
        @test all(haskey(comparison_result.aggregated_metrics, m) for m in config.metrics)
        
        # Create ensemble from top models
        performances = [comparison_result.aggregated_metrics[:accuracy][m] 
                       for m in comparison_result.model_names]
        
        ensemble = create_model_ensemble(
            comparison_result.model_names,
            performances,
            voting_method = :weighted,
            top_k = 3
        )
        
        @test length(ensemble.models) == 3
        @test ensemble.weights[1] >= ensemble.weights[2] >= ensemble.weights[3]
        
        # Generate executive summary
        summary = generate_executive_summary(comparison_result)
        @test length(summary) > 100  # Non-trivial summary
        
        # Export results
        temp_dir = mktempdir()
        export_config = ExportConfig(
            formats = [:json, :csv, :latex],
            timestamp_files = false
        )
        
        exported = export_results(
            comparison_result,
            joinpath(temp_dir, "results"),
            config = export_config
        )
        
        @test length(exported) == 3
        @test isfile(joinpath(temp_dir, "results.json"))
        @test isfile(joinpath(temp_dir, "results.csv"))
        @test isfile(joinpath(temp_dir, "results.tex"))
        
        # Read and verify JSON
        json_data = JSON.parsefile(joinpath(temp_dir, "results.json"))
        @test haskey(json_data, "model_names")
        @test haskey(json_data, "aggregated_metrics")
        @test length(json_data["model_names"]) == 4
        
        # Read and verify CSV
        csv_data = CSV.read(joinpath(temp_dir, "results.csv"), DataFrame)
        @test size(csv_data, 1) == 4  # 4 models
        @test "Model" in names(csv_data)
        @test "accuracy" in names(csv_data)
        
        # Clean up
        rm(temp_dir, recursive=true)
    end
end

println("All model comparison tests passed! ✓")