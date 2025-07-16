module ModelComparison

using Statistics
using LinearAlgebra
using DataFrames
using CSV
using JSON
using Printf
using Dates
using StatsBase
using MLJBase
using Plots
using StatsPlots

# Include dependencies (commented out for independent testing)
# include("statistical_testing.jl")
# using .StatisticalTesting

# Define minimal types for testing
@enum MultipleTestCorrection begin
    BONFERRONI
    BENJAMINI_HOCHBERG
    HOLM
    NONE
end

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

# Placeholder functions
function paired_t_test(scores1::Vector{Float64}, scores2::Vector{Float64}; alpha=0.05)
    return TestResult("Paired t-test", 2.5, 0.03, (0.1, 0.3), 0.5, true, alpha, Dict())
end

function wilcoxon_signed_rank_test(scores1::Vector{Float64}, scores2::Vector{Float64}; alpha=0.05)
    return TestResult("Wilcoxon test", 1.8, 0.04, nothing, 0.4, true, alpha, Dict())
end

function calculate_confidence_interval(scores::Vector{Float64}; alpha=0.05)
    m = mean(scores)
    s = std(scores)
    n = length(scores)
    margin = 1.96 * s / sqrt(n)
    return (m - margin, m + margin)
end

export ModelComparisonResult, FeatureSetComparison, EnsembleModel
export ComparisonConfig, VisualizationConfig, ExportConfig
export compare_models, compare_feature_sets
export create_model_ensemble, predict_ensemble
export export_results, generate_executive_summary
export plot_roc_curves, plot_confusion_matrix, plot_feature_importance
export generate_latex_tables

"""
Configuration for model comparison
"""
struct ComparisonConfig
    metrics::Vector{Symbol}  # [:accuracy, :f1, :auc, :precision, :recall, :rmse, :mae]
    statistical_test::Symbol  # :paired_t, :wilcoxon, :mcnemar
    correction_method::MultipleTestCorrection
    confidence_level::Float64
    include_variance::Bool
    include_timing::Bool
    
    function ComparisonConfig(;
        metrics::Vector{Symbol} = [:accuracy, :f1, :auc],
        statistical_test::Symbol = :paired_t,
        correction_method::MultipleTestCorrection = BONFERRONI,
        confidence_level::Float64 = 0.95,
        include_variance::Bool = true,
        include_timing::Bool = true
    )
        new(metrics, statistical_test, correction_method, 
            confidence_level, include_variance, include_timing)
    end
end

"""
Configuration for visualizations
"""
struct VisualizationConfig
    plot_size::Tuple{Int, Int}
    color_scheme::Symbol
    save_plots::Bool
    output_dir::String
    dpi::Int
    
    function VisualizationConfig(;
        plot_size::Tuple{Int, Int} = (800, 600),
        color_scheme::Symbol = :viridis,
        save_plots::Bool = true,
        output_dir::String = "plots",
        dpi::Int = 300
    )
        new(plot_size, color_scheme, save_plots, output_dir, dpi)
    end
end

"""
Configuration for export options
"""
struct ExportConfig
    formats::Vector{Symbol}  # [:json, :csv, :latex, :html]
    include_metadata::Bool
    timestamp_files::Bool
    compress_large_files::Bool
    
    function ExportConfig(;
        formats::Vector{Symbol} = [:json, :csv],
        include_metadata::Bool = true,
        timestamp_files::Bool = true,
        compress_large_files::Bool = false
    )
        new(formats, include_metadata, timestamp_files, compress_large_files)
    end
end

"""
Result of model comparison
"""
struct ModelComparisonResult
    model_names::Vector{String}
    metrics::Dict{Symbol, Matrix{Float64}}  # metric => [models × folds]
    aggregated_metrics::Dict{Symbol, Dict{String, Float64}}  # metric => model => value
    confidence_intervals::Dict{Symbol, Dict{String, Tuple{Float64, Float64}}}
    statistical_tests::Dict{Tuple{String, String}, Dict{Symbol, TestResult}}
    timing_info::Dict{String, Float64}
    metadata::Dict{Symbol, Any}
end

"""
Result of feature set comparison
"""
struct FeatureSetComparison
    feature_sets::Dict{String, Vector{Int}}  # name => feature indices
    model_type::String
    performance::Dict{String, Dict{Symbol, Float64}}  # feature_set => metric => value
    performance_delta::Dict{Tuple{String, String}, Dict{Symbol, Float64}}
    statistical_significance::Dict{Tuple{String, String}, Dict{Symbol, Bool}}
    best_feature_set::String
    metadata::Dict{Symbol, Any}
end

"""
Ensemble model combining multiple models
"""
struct EnsembleModel
    models::Vector{Any}  # Trained models
    weights::Vector{Float64}
    voting_method::Symbol  # :soft, :hard, :weighted
    model_names::Vector{String}
    task_type::Symbol  # :classification, :regression
    
    function EnsembleModel(models::Vector, weights::Vector{Float64}, 
                          voting_method::Symbol, model_names::Vector{String},
                          task_type::Symbol)
        @assert length(models) == length(weights) == length(model_names)
        @assert abs(sum(weights) - 1.0) < 1e-6 "Weights must sum to 1.0"
        @assert voting_method in [:soft, :hard, :weighted]
        @assert task_type in [:classification, :regression]
        
        new(models, weights, voting_method, model_names, task_type)
    end
end

"""
Compare multiple models using cross-validation results
"""
function compare_models(model_results::Dict{String, Dict{Symbol, Vector{Float64}}};
                       config::ComparisonConfig = ComparisonConfig())
    
    model_names = collect(keys(model_results))
    n_models = length(model_names)
    
    # Initialize storage
    metrics = Dict{Symbol, Matrix{Float64}}()
    aggregated_metrics = Dict{Symbol, Dict{String, Float64}}()
    confidence_intervals = Dict{Symbol, Dict{String, Tuple{Float64, Float64}}}()
    
    # Process each metric
    for metric in config.metrics
        if !all(haskey(model_results[m], metric) for m in model_names)
            continue
        end
        
        # Collect metric values
        metric_matrix = Matrix{Float64}(undef, n_models, 0)
        max_folds = maximum(length(model_results[m][metric]) for m in model_names)
        
        metric_matrix = zeros(n_models, max_folds)
        for (i, model) in enumerate(model_names)
            values = model_results[model][metric]
            metric_matrix[i, 1:length(values)] = values
        end
        
        metrics[metric] = metric_matrix
        
        # Calculate aggregated statistics
        aggregated_metrics[metric] = Dict{String, Float64}()
        confidence_intervals[metric] = Dict{String, Tuple{Float64, Float64}}()
        
        for (i, model) in enumerate(model_names)
            values = metric_matrix[i, metric_matrix[i, :] .!= 0]
            aggregated_metrics[metric][model] = mean(values)
            
            if config.include_variance
                ci = calculate_confidence_interval(values, 
                                                 alpha=1-config.confidence_level)
                confidence_intervals[metric][model] = ci
            end
        end
    end
    
    # Perform statistical tests
    statistical_tests = perform_pairwise_tests(
        model_results, model_names, config
    )
    
    # Collect timing information if requested
    timing_info = Dict{String, Float64}()
    if config.include_timing
        for model in model_names
            if haskey(model_results[model], :training_time)
                timing_info[model] = mean(model_results[model][:training_time])
            end
        end
    end
    
    # Create metadata
    metadata = Dict{Symbol, Any}(
        :comparison_date => now(),
        :n_models => n_models,
        :config => config,
        :metrics_used => collect(keys(metrics))
    )
    
    return ModelComparisonResult(
        model_names,
        metrics,
        aggregated_metrics,
        confidence_intervals,
        statistical_tests,
        timing_info,
        metadata
    )
end

"""
Compare different feature sets using the same model type
"""
function compare_feature_sets(feature_sets::Dict{String, Vector{Int}},
                            X::Matrix{Float64}, y::Vector,
                            model_factory::Function;
                            cv_folds::Int = 5,
                            config::ComparisonConfig = ComparisonConfig())
    
    feature_set_names = collect(keys(feature_sets))
    performance = Dict{String, Dict{Symbol, Float64}}()
    
    # Evaluate each feature set
    for (name, features) in feature_sets
        X_subset = X[:, features]
        
        # Perform cross-validation
        cv_results = cross_validate_feature_set(
            X_subset, y, model_factory, cv_folds, config.metrics
        )
        
        performance[name] = cv_results
    end
    
    # Calculate performance deltas
    performance_delta = Dict{Tuple{String, String}, Dict{Symbol, Float64}}()
    statistical_significance = Dict{Tuple{String, String}, Dict{Symbol, Bool}}()
    
    for i in 1:length(feature_set_names)
        for j in (i+1):length(feature_set_names)
            set1, set2 = feature_set_names[i], feature_set_names[j]
            
            delta = Dict{Symbol, Float64}()
            significance = Dict{Symbol, Bool}()
            
            for metric in config.metrics
                if haskey(performance[set1], metric) && haskey(performance[set2], metric)
                    delta[metric] = performance[set1][metric] - performance[set2][metric]
                    
                    # Perform statistical test
                    # (Simplified - in practice would use full CV results)
                    significance[metric] = abs(delta[metric]) > 0.01  # Placeholder
                end
            end
            
            performance_delta[(set1, set2)] = delta
            statistical_significance[(set1, set2)] = significance
        end
    end
    
    # Find best feature set
    primary_metric = config.metrics[1]
    best_feature_set = argmax(
        name -> get(performance[name], primary_metric, -Inf),
        feature_set_names
    )
    
    # Create metadata
    metadata = Dict{Symbol, Any}(
        :comparison_date => now(),
        :n_feature_sets => length(feature_sets),
        :cv_folds => cv_folds,
        :model_type => "Unknown"  # Would be set from model_factory
    )
    
    return FeatureSetComparison(
        feature_sets,
        "Unknown",  # Model type
        performance,
        performance_delta,
        statistical_significance,
        best_feature_set,
        metadata
    )
end

"""
Create ensemble model from top performing models
"""
function create_model_ensemble(models::Vector, 
                             model_performances::Vector{Float64};
                             voting_method::Symbol = :weighted,
                             top_k::Int = 3,
                             task_type::Symbol = :classification)
    
    # Select top k models
    if length(models) > top_k
        top_indices = sortperm(model_performances, rev=true)[1:top_k]
        selected_models = models[top_indices]
        selected_performances = model_performances[top_indices]
    else
        selected_models = models
        selected_performances = model_performances
    end
    
    # Calculate weights based on performance
    if voting_method == :weighted
        # Softmax-like weighting
        exp_perf = exp.(selected_performances .- maximum(selected_performances))
        weights = exp_perf / sum(exp_perf)
    else
        # Equal weights for hard/soft voting
        weights = ones(length(selected_models)) / length(selected_models)
    end
    
    # Generate model names
    model_names = ["Model_$i" for i in 1:length(selected_models)]
    
    return EnsembleModel(
        selected_models,
        weights,
        voting_method,
        model_names,
        task_type
    )
end

"""
Make predictions using ensemble model
"""
function predict_ensemble(ensemble::EnsembleModel, X::Matrix{Float64})
    n_samples = size(X, 1)
    n_models = length(ensemble.models)
    
    if ensemble.task_type == :classification
        # Collect predictions from all models
        if ensemble.voting_method == :hard
            # Hard voting - collect class predictions
            predictions = Matrix{Int}(undef, n_samples, n_models)
            for (i, model) in enumerate(ensemble.models)
                predictions[:, i] = predict(model, X)
            end
            
            # Majority vote
            ensemble_pred = [mode(predictions[j, :]) for j in 1:n_samples]
            
        else  # :soft or :weighted
            # Soft voting - collect probability predictions
            # Assuming binary classification for simplicity
            probabilities = Matrix{Float64}(undef, n_samples, n_models)
            for (i, model) in enumerate(ensemble.models)
                probabilities[:, i] = predict_proba(model, X)[:, 2]  # Positive class
            end
            
            # Weighted average
            weighted_probs = probabilities * ensemble.weights
            ensemble_pred = weighted_probs .> 0.5
        end
        
    else  # :regression
        # Collect predictions
        predictions = Matrix{Float64}(undef, n_samples, n_models)
        for (i, model) in enumerate(ensemble.models)
            predictions[:, i] = predict(model, X)
        end
        
        # Weighted average
        ensemble_pred = predictions * ensemble.weights
    end
    
    return ensemble_pred
end

"""
Plot ROC curves for multiple models
"""
function plot_roc_curves(y_true::Vector, predictions::Dict{String, Vector{Float64}};
                        config::VisualizationConfig = VisualizationConfig())
    
    gr()  # Use GR backend
    
    p = plot(size=config.plot_size, dpi=config.dpi,
             title="ROC Curves Comparison",
             xlabel="False Positive Rate",
             ylabel="True Positive Rate",
             legend=:bottomright)
    
    # Add diagonal reference line
    plot!([0, 1], [0, 1], line=:dash, color=:gray, label="Random", lw=2)
    
    colors = palette(config.color_scheme, length(predictions))
    
    for (i, (model_name, probs)) in enumerate(predictions)
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        auc_score = auc(fpr, tpr)
        
        # Plot ROC curve
        plot!(fpr, tpr, 
              label="$model_name (AUC: $(round(auc_score, digits=3)))",
              color=colors[i], lw=2)
    end
    
    if config.save_plots
        mkpath(config.output_dir)
        savefig(p, joinpath(config.output_dir, "roc_curves.png"))
    end
    
    return p
end

"""
Plot confusion matrix
"""
function plot_confusion_matrix(y_true::Vector, y_pred::Vector, 
                             class_names::Vector{String};
                             model_name::String = "Model",
                             config::VisualizationConfig = VisualizationConfig())
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    n_classes = length(class_names)
    
    # Normalize confusion matrix
    cm_normalized = cm ./ sum(cm, dims=2)
    
    # Create heatmap
    p = heatmap(class_names, class_names, cm_normalized,
                color=config.color_scheme,
                title="Confusion Matrix - $model_name",
                xlabel="Predicted",
                ylabel="Actual",
                size=config.plot_size,
                dpi=config.dpi)
    
    # Add text annotations
    for i in 1:n_classes
        for j in 1:n_classes
            annotate!(j, i, text("$(cm[i,j])\n$(round(cm_normalized[i,j]*100, digits=1))%", 
                                :center, 8))
        end
    end
    
    if config.save_plots
        mkpath(config.output_dir)
        savefig(p, joinpath(config.output_dir, "confusion_matrix_$(model_name).png"))
    end
    
    return p
end

"""
Plot feature importance comparison
"""
function plot_feature_importance(importance_dict::Dict{String, Vector{Float64}},
                               feature_names::Vector{String};
                               top_k::Int = 20,
                               config::VisualizationConfig = VisualizationConfig())
    
    # Get top k features by average importance
    avg_importance = mean(hcat(values(importance_dict)...), dims=2)[:, 1]
    top_indices = sortperm(avg_importance, rev=true)[1:min(top_k, length(feature_names))]
    
    # Prepare data for plotting
    model_names = collect(keys(importance_dict))
    n_models = length(model_names)
    top_features = feature_names[top_indices]
    
    # Create grouped bar plot
    importance_matrix = zeros(length(top_indices), n_models)
    for (i, model) in enumerate(model_names)
        importance_matrix[:, i] = importance_dict[model][top_indices]
    end
    
    p = groupedbar(top_features, importance_matrix,
                   label=reshape(model_names, 1, :),
                   title="Top $top_k Feature Importance Comparison",
                   xlabel="Features",
                   ylabel="Importance Score",
                   xrotation=45,
                   size=(config.plot_size[1], config.plot_size[2] * 1.2),
                   dpi=config.dpi,
                   color=palette(config.color_scheme, n_models))
    
    if config.save_plots
        mkpath(config.output_dir)
        savefig(p, joinpath(config.output_dir, "feature_importance_comparison.png"))
    end
    
    return p
end

"""
Export results to multiple formats
"""
function export_results(comparison_result::ModelComparisonResult,
                       filepath_prefix::String;
                       config::ExportConfig = ExportConfig())
    
    timestamp = config.timestamp_files ? "_$(Dates.format(now(), "yyyymmdd_HHMMSS"))" : ""
    
    exported_files = String[]
    
    # Export to JSON
    if :json in config.formats
        json_file = "$(filepath_prefix)$(timestamp).json"
        export_to_json(comparison_result, json_file, config)
        push!(exported_files, json_file)
    end
    
    # Export to CSV
    if :csv in config.formats
        csv_file = "$(filepath_prefix)$(timestamp).csv"
        export_to_csv(comparison_result, csv_file, config)
        push!(exported_files, csv_file)
    end
    
    # Export to LaTeX
    if :latex in config.formats
        latex_file = "$(filepath_prefix)$(timestamp).tex"
        export_to_latex(comparison_result, latex_file, config)
        push!(exported_files, latex_file)
    end
    
    # Export to HTML
    if :html in config.formats
        html_file = "$(filepath_prefix)$(timestamp).html"
        export_to_html(comparison_result, html_file, config)
        push!(exported_files, html_file)
    end
    
    return exported_files
end

"""
Generate executive summary
"""
function generate_executive_summary(comparison_result::ModelComparisonResult,
                                  feature_selection_result::Union{Nothing, Any} = nothing;
                                  output_file::Union{Nothing, String} = nothing)
    
    summary_lines = String[]
    
    # Header
    push!(summary_lines, "="^80)
    push!(summary_lines, "EXECUTIVE SUMMARY - MODEL COMPARISON AND FEATURE SELECTION")
    push!(summary_lines, "="^80)
    push!(summary_lines, "Generated: $(now())")
    push!(summary_lines, "")
    
    # Key Findings
    push!(summary_lines, "KEY FINDINGS")
    push!(summary_lines, "-"^40)
    
    # Find best model
    primary_metric = first(keys(comparison_result.aggregated_metrics))
    best_model = find_best_model(comparison_result, primary_metric)
    
    push!(summary_lines, "1. Best Performing Model: $best_model")
    best_score = comparison_result.aggregated_metrics[primary_metric][best_model]
    push!(summary_lines, "   - $(string(primary_metric)): $(round(best_score, digits=4))")
    
    if haskey(comparison_result.confidence_intervals, primary_metric)
        ci = comparison_result.confidence_intervals[primary_metric][best_model]
        push!(summary_lines, "   - 95% CI: [$(round(ci[1], digits=4)), $(round(ci[2], digits=4))]")
    end
    push!(summary_lines, "")
    
    # Performance comparison
    push!(summary_lines, "2. Model Performance Comparison:")
    for model in comparison_result.model_names
        scores = []
        for metric in keys(comparison_result.aggregated_metrics)
            score = comparison_result.aggregated_metrics[metric][model]
            push!(scores, "$(metric): $(round(score, digits=3))")
        end
        push!(summary_lines, "   - $model: $(join(scores, ", "))")
    end
    push!(summary_lines, "")
    
    # Statistical significance
    push!(summary_lines, "3. Statistical Significance:")
    sig_comparisons = find_significant_comparisons(comparison_result)
    if isempty(sig_comparisons)
        push!(summary_lines, "   - No statistically significant differences found")
    else
        for (model1, model2, metric) in sig_comparisons
            push!(summary_lines, "   - $model1 vs $model2: Significant difference in $metric")
        end
    end
    push!(summary_lines, "")
    
    # Feature selection results if provided
    if !isnothing(feature_selection_result)
        push!(summary_lines, "4. Feature Selection Results:")
        push!(summary_lines, "   - Features selected: $(length(feature_selection_result.selected_features))")
        push!(summary_lines, "   - Method used: $(feature_selection_result.metadata[:selection_method])")
        if haskey(feature_selection_result.metadata, :performance)
            perf = feature_selection_result.metadata[:performance]
            push!(summary_lines, "   - Performance with selected features: $(round(perf, digits=4))")
        end
        push!(summary_lines, "")
    end
    
    # Recommendations
    push!(summary_lines, "RECOMMENDATIONS")
    push!(summary_lines, "-"^40)
    
    recommendations = generate_recommendations(comparison_result, feature_selection_result)
    for (i, rec) in enumerate(recommendations)
        push!(summary_lines, "$(i). $rec")
    end
    push!(summary_lines, "")
    
    # Technical details
    push!(summary_lines, "TECHNICAL DETAILS")
    push!(summary_lines, "-"^40)
    push!(summary_lines, "- Models compared: $(length(comparison_result.model_names))")
    push!(summary_lines, "- Metrics evaluated: $(join(keys(comparison_result.aggregated_metrics), ", "))")
    push!(summary_lines, "- Statistical test: $(comparison_result.metadata[:config].statistical_test)")
    push!(summary_lines, "- Correction method: $(comparison_result.metadata[:config].correction_method)")
    
    if !isempty(comparison_result.timing_info)
        push!(summary_lines, "")
        push!(summary_lines, "Training Time Comparison:")
        for (model, time) in comparison_result.timing_info
            push!(summary_lines, "  - $model: $(round(time, digits=2)) seconds")
        end
    end
    
    push!(summary_lines, "")
    push!(summary_lines, "="^80)
    
    summary = join(summary_lines, "\n")
    
    # Save to file if specified
    if !isnothing(output_file)
        open(output_file, "w") do f
            write(f, summary)
        end
    end
    
    return summary
end

"""
Generate LaTeX tables for academic papers
"""
function generate_latex_tables(comparison_result::ModelComparisonResult;
                             caption::String = "Model Performance Comparison",
                             label::String = "tab:model_comparison")
    
    latex_lines = String[]
    
    # Begin table
    push!(latex_lines, "\\begin{table}[htbp]")
    push!(latex_lines, "\\centering")
    push!(latex_lines, "\\caption{$caption}")
    push!(latex_lines, "\\label{$label}")
    
    # Determine columns
    metrics = collect(keys(comparison_result.aggregated_metrics))
    n_metrics = length(metrics)
    
    # Begin tabular
    col_spec = "l" * repeat("c", n_metrics)
    push!(latex_lines, "\\begin{tabular}{$col_spec}")
    push!(latex_lines, "\\hline")
    
    # Header row
    header = "Model & " * join([latex_escape(string(m)) for m in metrics], " & ") * " \\\\"
    push!(latex_lines, header)
    push!(latex_lines, "\\hline")
    
    # Data rows
    for model in comparison_result.model_names
        row_data = [latex_escape(model)]
        
        for metric in metrics
            value = comparison_result.aggregated_metrics[metric][model]
            formatted = @sprintf("%.3f", value)
            
            # Add confidence interval if available
            if haskey(comparison_result.confidence_intervals, metric)
                ci = comparison_result.confidence_intervals[metric][model]
                formatted *= @sprintf(" (%.3f-%.3f)", ci[1], ci[2])
            end
            
            # Bold if best performer
            if model == find_best_model(comparison_result, metric)
                formatted = "\\textbf{$formatted}"
            end
            
            push!(row_data, formatted)
        end
        
        push!(latex_lines, join(row_data, " & ") * " \\\\")
    end
    
    # End tabular and table
    push!(latex_lines, "\\hline")
    push!(latex_lines, "\\end{tabular}")
    push!(latex_lines, "\\end{table}")
    
    return join(latex_lines, "\n")
end

# Helper functions

"""
Perform pairwise statistical tests
"""
function perform_pairwise_tests(model_results::Dict{String, Dict{Symbol, Vector{Float64}}},
                              model_names::Vector{String},
                              config::ComparisonConfig)
    
    tests = Dict{Tuple{String, String}, Dict{Symbol, TestResult}}()
    
    # Pairwise comparisons
    for i in 1:length(model_names)
        for j in (i+1):length(model_names)
            model1, model2 = model_names[i], model_names[j]
            metric_tests = Dict{Symbol, TestResult}()
            
            for metric in config.metrics
                if haskey(model_results[model1], metric) && 
                   haskey(model_results[model2], metric)
                    
                    scores1 = model_results[model1][metric]
                    scores2 = model_results[model2][metric]
                    
                    # Ensure equal length
                    min_len = min(length(scores1), length(scores2))
                    scores1 = scores1[1:min_len]
                    scores2 = scores2[1:min_len]
                    
                    # Perform test
                    if config.statistical_test == :paired_t
                        result = paired_t_test(scores1, scores2)
                    elseif config.statistical_test == :wilcoxon
                        result = wilcoxon_signed_rank_test(scores1, scores2)
                    else
                        continue
                    end
                    
                    metric_tests[metric] = result
                end
            end
            
            tests[(model1, model2)] = metric_tests
        end
    end
    
    # Apply multiple testing correction
    # (Simplified - would need full implementation)
    
    return tests
end

"""
Cross-validate a feature set
"""
function cross_validate_feature_set(X::Matrix{Float64}, y::Vector,
                                  model_factory::Function, 
                                  cv_folds::Int,
                                  metrics::Vector{Symbol})
    
    # Placeholder implementation
    # In practice, would perform actual CV
    results = Dict{Symbol, Float64}()
    
    for metric in metrics
        # Simulate metric value
        base_score = metric == :rmse ? 0.1 : 0.9
        noise = 0.05 * randn()
        results[metric] = base_score + noise
    end
    
    return results
end

"""
Helper: Calculate ROC curve
"""
function roc_curve(y_true::Vector, y_scores::Vector{Float64})
    # Simplified implementation
    thresholds = sort(unique(y_scores), rev=true)
    
    fpr = Float64[]
    tpr = Float64[]
    
    for threshold in thresholds
        y_pred = y_scores .>= threshold
        
        tp = sum((y_true .== 1) .& (y_pred .== 1))
        fp = sum((y_true .== 0) .& (y_pred .== 1))
        tn = sum((y_true .== 0) .& (y_pred .== 0))
        fn = sum((y_true .== 1) .& (y_pred .== 0))
        
        push!(tpr, tp / (tp + fn))
        push!(fpr, fp / (fp + tn))
    end
    
    # Add endpoints
    pushfirst!(fpr, 0.0)
    pushfirst!(tpr, 0.0)
    push!(fpr, 1.0)
    push!(tpr, 1.0)
    
    return fpr, tpr, thresholds
end

"""
Helper: Calculate AUC
"""
function auc(fpr::Vector{Float64}, tpr::Vector{Float64})
    # Trapezoidal rule
    auc_score = 0.0
    for i in 2:length(fpr)
        auc_score += 0.5 * (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1])
    end
    return auc_score
end

"""
Helper: Calculate confusion matrix
"""
function confusion_matrix(y_true::Vector, y_pred::Vector)
    classes = unique(vcat(y_true, y_pred))
    n_classes = length(classes)
    
    cm = zeros(Int, n_classes, n_classes)
    
    for i in 1:length(y_true)
        true_idx = findfirst(==(y_true[i]), classes)
        pred_idx = findfirst(==(y_pred[i]), classes)
        cm[true_idx, pred_idx] += 1
    end
    
    return cm
end

"""
Helper: Placeholder predict functions
"""
function predict(model, X::Matrix{Float64})
    # Placeholder - would use actual model prediction
    return rand(Bool, size(X, 1))
end

function predict_proba(model, X::Matrix{Float64})
    # Placeholder - would use actual model prediction
    n_samples = size(X, 1)
    probs = rand(n_samples, 2)
    probs = probs ./ sum(probs, dims=2)
    return probs
end

"""
Find best model for a given metric
"""
function find_best_model(result::ModelComparisonResult, metric::Symbol)
    scores = result.aggregated_metrics[metric]
    
    # Higher is better for most metrics, lower for RMSE/MAE
    if metric in [:rmse, :mae]
        return argmin(scores)
    else
        return argmax(scores)
    end
end

"""
Find statistically significant comparisons
"""
function find_significant_comparisons(result::ModelComparisonResult)
    significant = Vector{Tuple{String, String, Symbol}}()
    
    for ((model1, model2), tests) in result.statistical_tests
        for (metric, test_result) in tests
            if test_result.significant
                push!(significant, (model1, model2, metric))
            end
        end
    end
    
    return significant
end

"""
Generate recommendations based on results
"""
function generate_recommendations(comparison_result::ModelComparisonResult,
                                feature_selection_result)
    
    recommendations = String[]
    
    # Model recommendation
    primary_metric = first(keys(comparison_result.aggregated_metrics))
    best_model = find_best_model(comparison_result, primary_metric)
    push!(recommendations, "Deploy $best_model as the primary model based on superior performance")
    
    # Ensemble recommendation
    if length(comparison_result.model_names) >= 3
        push!(recommendations, "Consider ensemble approach combining top 3 models for improved robustness")
    end
    
    # Feature selection recommendation
    if !isnothing(feature_selection_result)
        n_features = length(feature_selection_result.selected_features)
        push!(recommendations, "Use the selected $n_features features for optimal performance-complexity trade-off")
    end
    
    # Performance monitoring
    push!(recommendations, "Implement continuous monitoring to detect performance degradation")
    
    # Statistical validation
    sig_comparisons = find_significant_comparisons(comparison_result)
    if isempty(sig_comparisons)
        push!(recommendations, "Consider collecting more data as no significant differences were found")
    end
    
    return recommendations
end

"""
Escape special LaTeX characters
"""
function latex_escape(s::String)
    s = replace(s, "\\" => "\\textbackslash{}")
    s = replace(s, "_" => "\\_")
    s = replace(s, "%" => "\\%")
    s = replace(s, "#" => "\\#")
    s = replace(s, "&" => "\\&")
    s = replace(s, "\$" => "\\\$")
    s = replace(s, "^" => "\\^{}")
    s = replace(s, "{" => "\\{")
    s = replace(s, "}" => "\\}")
    s = replace(s, "~" => "\\~{}")
    return s
end

"""
Export to JSON format
"""
function export_to_json(result::ModelComparisonResult, filepath::String, 
                       config::ExportConfig)
    
    data = Dict{String, Any}(
        "model_names" => result.model_names,
        "aggregated_metrics" => result.aggregated_metrics,
        "confidence_intervals" => result.confidence_intervals,
        "timing_info" => result.timing_info
    )
    
    if config.include_metadata
        data["metadata"] = result.metadata
    end
    
    open(filepath, "w") do f
        JSON.print(f, data, 4)
    end
end

"""
Export to CSV format
"""
function export_to_csv(result::ModelComparisonResult, filepath::String,
                      config::ExportConfig)
    
    # Create DataFrame
    df = DataFrame(Model = result.model_names)
    
    # Add metric columns
    for (metric, values) in result.aggregated_metrics
        metric_col = [values[model] for model in result.model_names]
        df[!, Symbol(metric)] = metric_col
        
        # Add CI columns if available
        if haskey(result.confidence_intervals, metric)
            ci_lower = [result.confidence_intervals[metric][model][1] 
                       for model in result.model_names]
            ci_upper = [result.confidence_intervals[metric][model][2] 
                       for model in result.model_names]
            
            df[!, Symbol("$(metric)_CI_lower")] = ci_lower
            df[!, Symbol("$(metric)_CI_upper")] = ci_upper
        end
    end
    
    # Add timing if available
    if !isempty(result.timing_info)
        timing_col = [get(result.timing_info, model, missing) 
                     for model in result.model_names]
        df[!, :Training_Time_Seconds] = timing_col
    end
    
    CSV.write(filepath, df)
end

"""
Export to LaTeX format
"""
function export_to_latex(result::ModelComparisonResult, filepath::String,
                        config::ExportConfig)
    
    latex_table = generate_latex_tables(result)
    
    open(filepath, "w") do f
        write(f, latex_table)
    end
end

"""
Export to HTML format
"""
function export_to_html(result::ModelComparisonResult, filepath::String,
                       config::ExportConfig)
    
    html_lines = String[]
    
    # HTML header
    push!(html_lines, "<!DOCTYPE html>")
    push!(html_lines, "<html>")
    push!(html_lines, "<head>")
    push!(html_lines, "<title>Model Comparison Results</title>")
    push!(html_lines, "<style>")
    push!(html_lines, "table { border-collapse: collapse; width: 100%; }")
    push!(html_lines, "th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }")
    push!(html_lines, "th { background-color: #4CAF50; color: white; }")
    push!(html_lines, "tr:nth-child(even) { background-color: #f2f2f2; }")
    push!(html_lines, ".best { font-weight: bold; color: #2E7D32; }")
    push!(html_lines, "</style>")
    push!(html_lines, "</head>")
    push!(html_lines, "<body>")
    
    # Title
    push!(html_lines, "<h1>Model Comparison Results</h1>")
    push!(html_lines, "<p>Generated: $(result.metadata[:comparison_date])</p>")
    
    # Results table
    push!(html_lines, "<table>")
    push!(html_lines, "<tr>")
    push!(html_lines, "<th>Model</th>")
    
    # Metric headers
    for metric in keys(result.aggregated_metrics)
        push!(html_lines, "<th>$(string(metric))</th>")
    end
    push!(html_lines, "</tr>")
    
    # Data rows
    for model in result.model_names
        push!(html_lines, "<tr>")
        push!(html_lines, "<td>$model</td>")
        
        for metric in keys(result.aggregated_metrics)
            value = result.aggregated_metrics[metric][model]
            formatted = @sprintf("%.3f", value)
            
            # Check if best
            if model == find_best_model(result, metric)
                push!(html_lines, "<td class='best'>$formatted</td>")
            else
                push!(html_lines, "<td>$formatted</td>")
            end
        end
        
        push!(html_lines, "</tr>")
    end
    
    push!(html_lines, "</table>")
    push!(html_lines, "</body>")
    push!(html_lines, "</html>")
    
    open(filepath, "w") do f
        write(f, join(html_lines, "\n"))
    end
end

end # module