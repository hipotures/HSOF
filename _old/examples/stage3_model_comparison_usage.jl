# Example usage of Model Comparison and Results Export System

using Random
using Statistics
using DataFrames
using CSV
using JSON
using Dates

# Include the modules
include("../src/stage3_evaluation/model_comparison.jl")

using .ModelComparison

# Set random seed
Random.seed!(123)

"""
Example 1: Basic model comparison with multiple metrics
"""
function basic_model_comparison_example()
    println("=== Basic Model Comparison Example ===\n")
    
    # Simulate cross-validation results for different models
    n_folds = 5
    
    model_results = Dict{String, Dict{Symbol, Vector{Float64}}}(
        "XGBoost" => Dict(
            :accuracy => [0.924, 0.917, 0.931, 0.912, 0.926],
            :f1 => [0.918, 0.911, 0.925, 0.906, 0.920],
            :auc => [0.956, 0.949, 0.963, 0.944, 0.957],
            :precision => [0.932, 0.925, 0.939, 0.920, 0.934],
            :recall => [0.905, 0.898, 0.912, 0.893, 0.907],
            :training_time => [2.34, 2.41, 2.28, 2.52, 2.37]
        ),
        "RandomForest" => Dict(
            :accuracy => [0.896, 0.889, 0.903, 0.884, 0.898],
            :f1 => [0.890, 0.883, 0.897, 0.878, 0.892],
            :auc => [0.928, 0.921, 0.935, 0.916, 0.929],
            :precision => [0.904, 0.897, 0.911, 0.892, 0.906],
            :recall => [0.877, 0.870, 0.884, 0.865, 0.879],
            :training_time => [3.56, 3.48, 3.62, 3.71, 3.55]
        ),
        "LightGBM" => Dict(
            :accuracy => [0.915, 0.908, 0.922, 0.903, 0.917],
            :f1 => [0.909, 0.902, 0.916, 0.897, 0.911],
            :auc => [0.947, 0.940, 0.954, 0.935, 0.948],
            :precision => [0.923, 0.916, 0.930, 0.911, 0.925],
            :recall => [0.896, 0.889, 0.903, 0.884, 0.898],
            :training_time => [1.82, 1.76, 1.89, 1.94, 1.85]
        ),
        "SVM" => Dict(
            :accuracy => [0.878, 0.871, 0.885, 0.866, 0.880],
            :f1 => [0.872, 0.865, 0.879, 0.860, 0.874],
            :auc => [0.910, 0.903, 0.917, 0.898, 0.911],
            :precision => [0.886, 0.879, 0.893, 0.874, 0.888],
            :recall => [0.859, 0.852, 0.866, 0.847, 0.861],
            :training_time => [5.23, 5.35, 5.18, 5.41, 5.29]
        )
    )
    
    # Configure comparison
    config = ComparisonConfig(
        metrics = [:accuracy, :f1, :auc, :precision, :recall],
        statistical_test = :paired_t,
        confidence_level = 0.95,
        include_variance = true,
        include_timing = true
    )
    
    # Compare models
    result = compare_models(model_results, config=config)
    
    # Display results
    println("Model Performance Summary:")
    println("-" * 60)
    println("Model         | Accuracy | F1 Score |   AUC   | Time (s)")
    println("-" * 60)
    
    for model in result.model_names
        acc = result.aggregated_metrics[:accuracy][model]
        f1 = result.aggregated_metrics[:f1][model]
        auc = result.aggregated_metrics[:auc][model]
        time = get(result.timing_info, model, 0.0)
        
        @printf("%-13s | %8.3f | %8.3f | %7.3f | %7.2f\n", 
                model, acc, f1, auc, time)
    end
    
    println("\nBest performing model: $(ModelComparison.find_best_model(result, :accuracy))")
    
    return result
end

"""
Example 2: Feature set comparison
"""
function feature_set_comparison_example()
    println("\n\n=== Feature Set Comparison Example ===\n")
    
    # Create synthetic data
    n_samples, n_features = 1000, 50
    X = randn(n_samples, n_features)
    
    # Create target with known important features
    important_features = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    y_continuous = sum(X[:, f] * randn() for f in important_features) + 0.1 * randn(n_samples)
    y = y_continuous .> median(y_continuous)
    
    # Define different feature sets
    feature_sets = Dict(
        "All_Features" => collect(1:50),
        "Top_20_Univariate" => [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 2, 6, 11, 16, 21, 26, 31, 36, 41, 46],
        "Selected_15_LASSO" => [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 3, 8, 13, 18, 23],
        "Random_10" => sort(randperm(50)[1:10]),
        "Domain_Expert_12" => [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 48, 49]
    )
    
    # Model factory (returns a simple classifier)
    model_factory = () -> "SimpleClassifier"
    
    # Compare feature sets
    comparison_config = ComparisonConfig(
        metrics = [:accuracy, :f1, :auc],
        statistical_test = :paired_t
    )
    
    result = compare_feature_sets(
        feature_sets, X, y, model_factory,
        cv_folds=5, config=comparison_config
    )
    
    println("Feature Set Performance:")
    println("-" * 50)
    println("Feature Set          | Size | Accuracy |   F1")
    println("-" * 50)
    
    for (name, features) in feature_sets
        perf = result.performance[name]
        @printf("%-20s | %4d | %8.3f | %7.3f\n",
                name, length(features), 
                get(perf, :accuracy, 0.0),
                get(perf, :f1, 0.0))
    end
    
    println("\nBest feature set: $(result.best_feature_set)")
    
    # Show performance deltas
    println("\nPerformance Deltas (vs All_Features):")
    for (name, _) in feature_sets
        if name != "All_Features"
            delta_key = ("All_Features", name)
            if haskey(result.performance_delta, delta_key)
                delta = result.performance_delta[delta_key]
                @printf("  %s: Accuracy %+.3f\n", name, get(delta, :accuracy, 0.0))
            end
        end
    end
    
    return result
end

"""
Example 3: Creating and using model ensembles
"""
function ensemble_model_example()
    println("\n\n=== Model Ensemble Example ===\n")
    
    # Mock models and their performances
    models = ["XGBoost_1", "XGBoost_2", "RandomForest", "LightGBM", "ExtraTrees"]
    cv_performances = [0.925, 0.922, 0.898, 0.918, 0.895]
    
    println("Individual Model Performances:")
    for (model, perf) in zip(models, cv_performances)
        println("  $model: $(round(perf, digits=3))")
    end
    
    # Create weighted ensemble from top 3 models
    ensemble = create_model_ensemble(
        models, cv_performances,
        voting_method = :weighted,
        top_k = 3,
        task_type = :classification
    )
    
    println("\nEnsemble Configuration:")
    println("  Selected models: $(join(ensemble.model_names, ", "))")
    println("  Weights: $(round.(ensemble.weights, digits=3))")
    println("  Voting method: $(ensemble.voting_method)")
    
    # Simulate ensemble prediction
    n_test = 100
    X_test = randn(n_test, 20)
    
    # Note: This would use actual predictions in practice
    println("\nEnsemble prediction on test set:")
    println("  Test samples: $n_test")
    println("  Prediction type: $(ensemble.task_type)")
    
    return ensemble
end

"""
Example 4: Visualization exports
"""
function visualization_example()
    println("\n\n=== Visualization Export Example ===\n")
    
    # Create output directory
    output_dir = "model_comparison_plots"
    mkpath(output_dir)
    
    viz_config = VisualizationConfig(
        plot_size = (800, 600),
        color_scheme = :viridis,
        save_plots = true,
        output_dir = output_dir,
        dpi = 300
    )
    
    # Example 1: ROC Curves
    println("Generating ROC curves...")
    
    # Simulate predictions
    n_samples = 500
    y_true = rand(Bool, n_samples)
    
    predictions = Dict(
        "XGBoost" => y_true .+ 0.3 * randn(n_samples),
        "RandomForest" => y_true .+ 0.4 * randn(n_samples),
        "LightGBM" => y_true .+ 0.35 * randn(n_samples)
    )
    
    # Normalize to probabilities
    for (model, preds) in predictions
        predictions[model] = 1 ./ (1 .+ exp.(-preds))
    end
    
    # Plot ROC curves
    roc_plot = plot_roc_curves(y_true, predictions, config=viz_config)
    println("  ROC curves saved to: $(viz_config.output_dir)/roc_curves.png")
    
    # Example 2: Confusion Matrix
    println("\nGenerating confusion matrix...")
    
    # Binary predictions
    y_pred = predictions["XGBoost"] .> 0.5
    class_names = ["Negative", "Positive"]
    
    cm_plot = plot_confusion_matrix(
        y_true, y_pred, class_names,
        model_name = "XGBoost",
        config = viz_config
    )
    println("  Confusion matrix saved to: $(viz_config.output_dir)/confusion_matrix_XGBoost.png")
    
    # Example 3: Feature Importance
    println("\nGenerating feature importance comparison...")
    
    feature_names = ["Feature_$i" for i in 1:50]
    importance_dict = Dict(
        "XGBoost" => abs.(randn(50)) .* [i <= 10 ? 2.0 : 1.0 for i in 1:50],
        "RandomForest" => abs.(randn(50)) .* [i <= 10 ? 1.8 : 0.9 for i in 1:50],
        "LightGBM" => abs.(randn(50)) .* [i <= 10 ? 1.9 : 0.95 for i in 1:50]
    )
    
    # Normalize importances
    for (model, imp) in importance_dict
        importance_dict[model] = imp / sum(imp)
    end
    
    imp_plot = plot_feature_importance(
        importance_dict, feature_names,
        top_k = 15,
        config = viz_config
    )
    println("  Feature importance saved to: $(viz_config.output_dir)/feature_importance_comparison.png")
    
    # Clean up (comment out to keep plots)
    rm(output_dir, recursive=true, force=true)
    
    return viz_config
end

"""
Example 5: Export results to multiple formats
"""
function export_results_example()
    println("\n\n=== Results Export Example ===\n")
    
    # Create mock comparison result
    model_results = Dict(
        "XGBoost" => Dict(
            :accuracy => [0.92, 0.91, 0.93],
            :f1 => [0.91, 0.90, 0.92],
            :auc => [0.95, 0.94, 0.96]
        ),
        "RandomForest" => Dict(
            :accuracy => [0.89, 0.88, 0.90],
            :f1 => [0.88, 0.87, 0.89],
            :auc => [0.92, 0.91, 0.93]
        )
    )
    
    result = compare_models(model_results)
    
    # Configure export
    export_config = ExportConfig(
        formats = [:json, :csv, :latex, :html],
        include_metadata = true,
        timestamp_files = false
    )
    
    # Export to temporary directory
    temp_dir = mktempdir()
    filepath_prefix = joinpath(temp_dir, "model_comparison")
    
    exported_files = export_results(result, filepath_prefix, config=export_config)
    
    println("Exported files:")
    for file in exported_files
        size_kb = filesize(file) / 1024
        println("  - $(basename(file)) ($(round(size_kb, digits=1)) KB)")
    end
    
    # Show JSON content
    println("\nJSON content preview:")
    json_file = first(filter(f -> endswith(f, ".json"), exported_files))
    json_data = JSON.parsefile(json_file)
    println("  Models: $(join(json_data["model_names"], ", "))")
    println("  Metrics: $(join(keys(json_data["aggregated_metrics"]), ", "))")
    
    # Show CSV content
    println("\nCSV content preview:")
    csv_file = first(filter(f -> endswith(f, ".csv"), exported_files))
    df = CSV.read(csv_file, DataFrame)
    println(first(df, 5))
    
    # Show LaTeX content preview
    println("\nLaTeX content preview (first 5 lines):")
    latex_file = first(filter(f -> endswith(f, ".tex"), exported_files))
    latex_lines = readlines(latex_file)
    for line in latex_lines[1:min(5, length(latex_lines))]
        println("  $line")
    end
    
    # Clean up
    rm(temp_dir, recursive=true)
    
    return exported_files
end

"""
Example 6: Comprehensive executive summary
"""
function executive_summary_example()
    println("\n\n=== Executive Summary Example ===\n")
    
    # Create comprehensive model comparison
    n_folds = 10
    model_results = Dict{String, Dict{Symbol, Vector{Float64}}}()
    
    # Simulate results for multiple models
    models = ["XGBoost", "RandomForest", "LightGBM", "SVM", "LogisticRegression"]
    base_performances = [0.92, 0.89, 0.91, 0.87, 0.85]
    
    for (i, model) in enumerate(models)
        base = base_performances[i]
        model_results[model] = Dict(
            :accuracy => [base + 0.01*randn() for _ in 1:n_folds],
            :f1 => [base - 0.01 + 0.01*randn() for _ in 1:n_folds],
            :auc => [base + 0.03 + 0.01*randn() for _ in 1:n_folds],
            :precision => [base + 0.01 + 0.01*randn() for _ in 1:n_folds],
            :recall => [base - 0.02 + 0.01*randn() for _ in 1:n_folds],
            :training_time => [2.0 * i + 0.5*randn() for _ in 1:n_folds]
        )
    end
    
    # Compare models with comprehensive configuration
    config = ComparisonConfig(
        metrics = [:accuracy, :f1, :auc, :precision, :recall],
        statistical_test = :paired_t,
        correction_method = BONFERRONI,
        confidence_level = 0.95,
        include_variance = true,
        include_timing = true
    )
    
    comparison_result = compare_models(model_results, config=config)
    
    # Create mock feature selection result
    feature_selection_result = (
        selected_features = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        metadata = Dict(
            :selection_method => :pareto,
            :performance => 0.918,
            :original_features => 50
        )
    )
    
    # Generate executive summary
    summary = generate_executive_summary(
        comparison_result,
        feature_selection_result
    )
    
    # Display summary
    println(summary)
    
    # Save to file
    summary_file = "executive_summary.txt"
    generate_executive_summary(
        comparison_result,
        feature_selection_result,
        output_file = summary_file
    )
    
    println("\n\nSummary also saved to: $summary_file")
    
    # Clean up
    rm(summary_file, force=true)
    
    return comparison_result
end

"""
Example 7: Complete pipeline with statistical testing
"""
function complete_pipeline_example()
    println("\n\n=== Complete Model Comparison Pipeline Example ===\n")
    
    # Step 1: Prepare model results
    println("Step 1: Collecting cross-validation results...")
    
    n_folds = 5
    models = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees"]
    
    # Simulate realistic CV results
    model_results = Dict{String, Dict{Symbol, Vector{Float64}}}()
    
    # XGBoost - typically best
    model_results["XGBoost"] = Dict(
        :accuracy => [0.924, 0.918, 0.932, 0.921, 0.927],
        :f1 => [0.919, 0.913, 0.927, 0.916, 0.922],
        :auc => [0.958, 0.952, 0.966, 0.955, 0.961],
        :training_time => [12.3, 11.8, 12.9, 12.1, 12.5]
    )
    
    # LightGBM - fast and good
    model_results["LightGBM"] = Dict(
        :accuracy => [0.919, 0.913, 0.927, 0.916, 0.922],
        :f1 => [0.914, 0.908, 0.922, 0.911, 0.917],
        :auc => [0.953, 0.947, 0.961, 0.950, 0.956],
        :training_time => [4.2, 4.0, 4.5, 4.1, 4.3]
    )
    
    # CatBoost - handles categoricals well
    model_results["CatBoost"] = Dict(
        :accuracy => [0.922, 0.916, 0.930, 0.919, 0.925],
        :f1 => [0.917, 0.911, 0.925, 0.914, 0.920],
        :auc => [0.956, 0.950, 0.964, 0.953, 0.959],
        :training_time => [15.7, 15.2, 16.3, 15.5, 15.9]
    )
    
    # RandomForest - reliable baseline
    model_results["RandomForest"] = Dict(
        :accuracy => [0.908, 0.902, 0.916, 0.905, 0.911],
        :f1 => [0.903, 0.897, 0.911, 0.900, 0.906],
        :auc => [0.942, 0.936, 0.950, 0.939, 0.945],
        :training_time => [8.9, 8.5, 9.3, 8.7, 9.0]
    )
    
    # ExtraTrees - similar to RF
    model_results["ExtraTrees"] = Dict(
        :accuracy => [0.905, 0.899, 0.913, 0.902, 0.908],
        :f1 => [0.900, 0.894, 0.908, 0.897, 0.903],
        :auc => [0.939, 0.933, 0.947, 0.936, 0.942],
        :training_time => [7.2, 6.9, 7.6, 7.0, 7.3]
    )
    
    # Step 2: Compare models
    println("\nStep 2: Performing statistical comparison...")
    
    config = ComparisonConfig(
        metrics = [:accuracy, :f1, :auc],
        statistical_test = :paired_t,
        correction_method = BONFERRONI,
        confidence_level = 0.95,
        include_variance = true,
        include_timing = true
    )
    
    comparison_result = compare_models(model_results, config=config)
    
    # Step 3: Create ensemble
    println("\nStep 3: Creating model ensemble...")
    
    accuracies = [comparison_result.aggregated_metrics[:accuracy][m] 
                  for m in comparison_result.model_names]
    
    ensemble = create_model_ensemble(
        comparison_result.model_names,
        accuracies,
        voting_method = :weighted,
        top_k = 3
    )
    
    println("  Ensemble members: $(join(ensemble.model_names, ", "))")
    println("  Ensemble weights: $(round.(ensemble.weights, digits=3))")
    
    # Step 4: Generate LaTeX tables
    println("\nStep 4: Generating LaTeX tables for paper...")
    
    latex_table = generate_latex_tables(
        comparison_result,
        caption = "Cross-validation performance comparison of gradient boosting and tree ensemble methods",
        label = "tab:model_comparison"
    )
    
    println("LaTeX table preview:")
    println("-" * 60)
    latex_lines = split(latex_table, "\n")
    for line in latex_lines[1:min(10, length(latex_lines))]
        println(line)
    end
    println("...")
    
    # Step 5: Export comprehensive results
    println("\nStep 5: Exporting results...")
    
    temp_dir = mktempdir()
    export_config = ExportConfig(
        formats = [:json, :csv, :latex],
        include_metadata = true,
        timestamp_files = false
    )
    
    exported = export_results(
        comparison_result,
        joinpath(temp_dir, "final_results"),
        config = export_config
    )
    
    println("  Exported $(length(exported)) files")
    
    # Step 6: Generate executive summary
    println("\nStep 6: Generating executive summary...")
    
    summary = generate_executive_summary(comparison_result)
    summary_lines = split(summary, "\n")
    println("\nExecutive Summary Preview:")
    println("-" * 60)
    for line in summary_lines[1:min(20, length(summary_lines))]
        println(line)
    end
    println("...\n")
    
    # Clean up
    rm(temp_dir, recursive=true)
    
    return comparison_result, ensemble
end

# Run all examples
if abspath(PROGRAM_FILE) == @__FILE__
    println("Model Comparison and Results Export Examples")
    println("="^60)
    
    # Run examples
    basic_result = basic_model_comparison_example()
    feature_result = feature_set_comparison_example()
    ensemble = ensemble_model_example()
    viz_config = visualization_example()
    export_files = export_results_example()
    exec_result = executive_summary_example()
    final_result, final_ensemble = complete_pipeline_example()
    
    println("\n" * "="^60)
    println("All model comparison examples completed!")
    
    # Summary
    println("\nKey Takeaways:")
    println("1. Comprehensive model comparison with multiple metrics and statistical tests")
    println("2. Feature set comparison to evaluate different selection strategies")
    println("3. Model ensemble creation for improved performance")
    println("4. Rich visualizations including ROC curves and confusion matrices")
    println("5. Multiple export formats (JSON, CSV, LaTeX, HTML)")
    println("6. Executive summaries for stakeholder communication")
    println("7. Complete pipeline from CV results to publication-ready tables")
end