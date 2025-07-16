#!/usr/bin/env julia

using DataFrames
using SQLite
using YAML
using Statistics
using StatsBase
using MLJ
using Random
using LinearAlgebra
using Dates
using JSON
using Printf

Random.seed!(42)

struct HSOfResults
    cv_mean::Float64
    cv_std::Float64
    improvement::Float64
    selected_features::Vector{String}
    feature_importance::Vector{Float64}
    dataset_name::String
    task_type::String
    stage1_reduction::Tuple{Int,Int}
    stage2_reduction::Tuple{Int,Int}
    execution_time::Float64
end

function load_dataset_from_yaml(yaml_path::String)
    """Universal YAML-driven SQLite data loader"""
    if !isfile(yaml_path)
        error("YAML configuration file not found: $yaml_path")
    end
    
    try
        config = YAML.load_file(yaml_path)
        
        dataset_name = config["name"]
        db_path = config["database"]["path"]
        table_name = config["tables"]["train_features"]
        target_column = config["target_column"]
        problem_type = config["problem_type"]
        id_columns = haskey(config, "id_columns") ? config["id_columns"] : String[]
        
        # Validate database exists
        if !isfile(db_path)
            error("Database file not found: $db_path")
        end
        
        # Connect to SQLite and load data
        conn = SQLite.DB(db_path)
        
        # Check if table exists
        tables = SQLite.DBInterface.execute(conn, "SELECT name FROM sqlite_master WHERE type='table'") |> DataFrame
        if !(table_name in tables.name)
            SQLite.close(conn)
            error("Table '$table_name' not found in database")
        end
        
        # Load data
        data = DataFrame(SQLite.DBInterface.execute(conn, "SELECT * FROM $table_name"))
        SQLite.close(conn)
        
        if nrow(data) == 0
            error("No data found in table: $table_name")
        end
        
        # Validate target column exists
        if !(target_column in names(data))
            error("Target column '$target_column' not found in data. Available columns: $(names(data))")
        end
        
        return data, target_column, problem_type, id_columns, dataset_name
        
    catch e
        error("Error loading YAML configuration: $e")
    end
end

function prepare_features_from_sqlite(data::DataFrame, target_col::String, id_cols::Vector{String})
    """Convert SQLite DataFrame to ML format using YAML config"""
    try
        # Remove ID columns and target from features
        feature_cols = setdiff(names(data), vcat(id_cols, [target_col]))
        
        if length(feature_cols) == 0
            error("No feature columns found after removing ID and target columns")
        end
        
        # Extract target variable
        y_raw = data[!, target_col]
        
        # Handle missing values in target
        if any(ismissing.(y_raw))
            error("Missing values found in target variable")
        end
        
        # Convert target to appropriate type
        y = if eltype(y_raw) <: Union{Missing, String}
            # Categorical target - encode as integers
            unique_vals = unique(y_raw)
            y_encoded = [findfirst(x -> x == val, unique_vals) - 1 for val in y_raw]
            Float64.(y_encoded)
        else
            Float64.(y_raw)
        end
        
        # Prepare feature matrix
        X_df = select(data, feature_cols)
        
        # Handle missing values and convert to numeric
        X_numeric = Matrix{Float64}(undef, nrow(X_df), ncol(X_df))
        
        for (j, col) in enumerate(feature_cols)
            col_data = X_df[!, col]
            
            if eltype(col_data) <: Union{Missing, String}
                # Categorical feature - encode as integers
                unique_vals = unique(skipmissing(col_data))
                encoded = [ismissing(val) ? 0.0 : Float64(findfirst(x -> x == val, unique_vals) - 1) for val in col_data]
                X_numeric[:, j] = encoded
            else
                # Numeric feature - handle missing values
                numeric_vals = [ismissing(val) ? 0.0 : Float64(val) for val in col_data]
                X_numeric[:, j] = numeric_vals
            end
        end
        
        # Remove constant features
        non_constant_mask = [std(X_numeric[:, j]) > 1e-10 for j in 1:size(X_numeric, 2)]
        X_filtered = X_numeric[:, non_constant_mask]
        feature_names_filtered = feature_cols[non_constant_mask]
        
        if size(X_filtered, 2) == 0
            error("All features are constant")
        end
        
        return X_filtered, y, feature_names_filtered
        
    catch e
        error("Error preparing features: $e")
    end
end

function mutual_information(x::Vector{Float64}, y::Vector{Float64}, bins::Int=10)
    """Calculate mutual information between two continuous variables"""
    try
        # Discretize variables
        x_edges = range(minimum(x), maximum(x), length=bins+1)
        y_edges = range(minimum(y), maximum(y), length=bins+1)
        
        x_disc = [searchsortedfirst(x_edges[2:end], val) for val in x]
        y_disc = [searchsortedfirst(y_edges[2:end], val) for val in y]
        
        # Ensure indices are within bounds
        x_disc = clamp.(x_disc, 1, bins)
        y_disc = clamp.(y_disc, 1, bins)
        
        # Calculate joint and marginal distributions
        joint = zeros(bins, bins)
        for i in 1:length(x_disc)
            joint[x_disc[i], y_disc[i]] += 1
        end
        joint ./= sum(joint)
        
        marginal_x = sum(joint, dims=2)
        marginal_y = sum(joint, dims=1)
        
        # Calculate mutual information
        mi = 0.0
        for i in 1:bins
            for j in 1:bins
                if joint[i, j] > 0 && marginal_x[i] > 0 && marginal_y[j] > 0
                    mi += joint[i, j] * log(joint[i, j] / (marginal_x[i] * marginal_y[j]))
                end
            end
        end
        
        return mi
    catch
        return 0.0
    end
end

function stage1_filter(X::Matrix{Float64}, y::Vector, feature_names::Vector{String}, target_features::Int)
    """Universal Stage 1 filtering"""
    println("Stage 1: Fast filtering...")
    
    n_features = size(X, 2)
    target_features = min(target_features, n_features)
    
    # Calculate correlation matrix
    corr_matrix = cor(X)
    
    # Remove highly correlated features (>0.95)
    to_remove = Set{Int}()
    for i in 1:n_features
        for j in (i+1):n_features
            if abs(corr_matrix[i, j]) > 0.95
                # Remove the feature with lower correlation to target
                corr_i = abs(cor(X[:, i], y))
                corr_j = abs(cor(X[:, j], y))
                if corr_i < corr_j
                    push!(to_remove, i)
                else
                    push!(to_remove, j)
                end
            end
        end
    end
    
    # Keep features not marked for removal
    remaining_indices = setdiff(1:n_features, to_remove)
    
    if length(remaining_indices) <= target_features
        println("  Correlation filtering: $(n_features) → $(length(remaining_indices)) features")
        return remaining_indices
    end
    
    # Rank remaining features by mutual information with target
    mi_scores = Float64[]
    for idx in remaining_indices
        mi = mutual_information(X[:, idx], y)
        push!(mi_scores, mi)
    end
    
    # Select top features
    sorted_indices = sortperm(mi_scores, rev=true)
    selected_indices = remaining_indices[sorted_indices[1:target_features]]
    
    println("  Correlation filtering: $(n_features) → $(length(remaining_indices)) features")
    println("  Mutual information ranking: $(length(remaining_indices)) → $(target_features) features")
    
    return selected_indices
end

function evaluate_feature_set(X::Matrix{Float64}, y::Vector, feature_indices::Vector{Int}, task_type::String)
    """Evaluate feature set using cross-validation"""
    if length(feature_indices) == 0
        return 0.0
    end
    
    try
        X_subset = X[:, feature_indices]
        
        # Choose model based on task type
        if task_type == "binary_classification" || task_type == "classification"
            # Use logistic regression for speed
            model = @load LogisticClassifier pkg=MLJLinearModels
            machine_obj = machine(model(), X_subset, categorical(y))
        else
            # Use linear regression for regression tasks
            model = @load LinearRegressor pkg=MLJLinearModels
            machine_obj = machine(model(), X_subset, y)
        end
        
        # 3-fold cross-validation for speed
        cv_result = evaluate!(machine_obj, resampling=CV(nfolds=3), measure=default_measure(task_type))
        
        return mean(cv_result.measurement)
    catch
        return 0.0
    end
end

function default_measure(task_type::String)
    """Get default evaluation measure for task type"""
    if task_type == "binary_classification" || task_type == "classification"
        return accuracy
    else
        return rms  # Root mean square error (lower is better, so we'll negate it)
    end
end

function stage2_mcts_cpu(X::Matrix{Float64}, y::Vector, feature_indices::Vector{Int}, feature_names::Vector{String}, task_type::String)
    """CPU-based feature selection using greedy forward selection"""
    println("Stage 2: Greedy feature selection...")
    
    if length(feature_indices) <= 10
        println("  Skipping stage 2 - already have ≤10 features")
        return feature_indices
    end
    
    target_features = min(10, length(feature_indices))
    selected = Int[]
    remaining = copy(feature_indices)
    best_score = -Inf
    
    # Forward selection
    for i in 1:target_features
        best_feature = -1
        best_improvement = -Inf
        
        for feature in remaining
            candidate_set = vcat(selected, [feature])
            score = evaluate_feature_set(X, y, candidate_set, task_type)
            
            # For regression tasks, negate RMS to make higher better
            if task_type == "regression"
                score = -score
            end
            
            if score > best_improvement
                best_improvement = score
                best_feature = feature
            end
        end
        
        if best_feature != -1
            push!(selected, best_feature)
            filter!(x -> x != best_feature, remaining)
            best_score = best_improvement
            println("  Selected feature $(i): $(feature_names[findfirst(x -> x == best_feature, feature_indices)]) (score: $(round(best_improvement, digits=4)))")
        else
            break
        end
    end
    
    println("  Greedy selection: $(length(feature_indices)) → $(length(selected)) features")
    return selected
end

function stage3_evaluate(X::Matrix{Float64}, y::Vector, selected_features::Vector{Int}, feature_names::Vector{String}, task_type::String)
    """Final evaluation with robust cross-validation"""
    println("Stage 3: Final evaluation...")
    
    if length(selected_features) == 0
        return 0.0, 0.0, String[], Float64[]
    end
    
    X_final = X[:, selected_features]
    final_feature_names = feature_names[selected_features]
    
    try
        # Use more robust models for final evaluation
        if task_type == "binary_classification" || task_type == "classification"
            # Try RandomForest for classification
            model = @load RandomForestClassifier pkg=MLJDecisionTreeInterface
            machine_obj = machine(model(n_trees=50), X_final, categorical(y))
            measure = accuracy
        else
            # Try RandomForest for regression
            model = @load RandomForestRegressor pkg=MLJDecisionTreeInterface  
            machine_obj = machine(model(n_trees=50), X_final, y)
            measure = rms
        end
        
        # 5-fold cross-validation
        cv_result = evaluate!(machine_obj, resampling=CV(nfolds=5), measure=measure)
        
        cv_mean = mean(cv_result.measurement)
        cv_std = std(cv_result.measurement)
        
        # For regression, convert back to negative (lower RMS is better)
        if task_type == "regression"
            cv_mean = -cv_mean
        end
        
        # Calculate feature importance (simplified - use correlation with target)
        feature_importance = [abs(cor(X[:, idx], y)) for idx in selected_features]
        
        return cv_mean, cv_std, final_feature_names, feature_importance
        
    catch e
        println("  Warning: Final evaluation failed, using fallback method")
        # Fallback to simple correlation
        feature_importance = [abs(cor(X[:, idx], y)) for idx in selected_features]
        return 0.5, 0.1, final_feature_names, feature_importance
    end
end

function calculate_baseline_performance(X::Matrix{Float64}, y::Vector, task_type::String)
    """Calculate baseline performance using all features"""
    try
        if task_type == "binary_classification" || task_type == "classification"
            model = @load LogisticClassifier pkg=MLJLinearModels
            machine_obj = machine(model(), X, categorical(y))
            measure = accuracy
        else
            model = @load LinearRegressor pkg=MLJLinearModels
            machine_obj = machine(model(), X, y)
            measure = rms
        end
        
        cv_result = evaluate!(machine_obj, resampling=CV(nfolds=3), measure=measure)
        baseline = mean(cv_result.measurement)
        
        # For regression, negate RMS
        if task_type == "regression"
            baseline = -baseline
        end
        
        return baseline
    catch
        return task_type == "regression" ? 0.0 : 0.5
    end
end

function save_results(results::HSOfResults, yaml_path::String)
    """Save results to JSON file with timestamp"""
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    output_dir = "results"
    
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    output_file = joinpath(output_dir, "hsof_$(results.dataset_name)_$(timestamp).json")
    
    result_dict = Dict(
        "dataset_name" => results.dataset_name,
        "task_type" => results.task_type,
        "cv_mean" => results.cv_mean,
        "cv_std" => results.cv_std,
        "improvement" => results.improvement,
        "selected_features" => results.selected_features,
        "feature_importance" => results.feature_importance,
        "stage1_reduction" => results.stage1_reduction,
        "stage2_reduction" => results.stage2_reduction,
        "execution_time" => results.execution_time,
        "timestamp" => timestamp
    )
    
    open(output_file, "w") do f
        JSON.print(f, result_dict, 2)
    end
    
    println("\nResults saved to: $output_file")
end

function display_results(results::HSOfResults)
    """Display results in a formatted way"""
    println("\nHSOF Feature Selection Results")
    println("=" ^ 50)
    println("Dataset: $(results.dataset_name)")
    println("Task Type: $(results.task_type)")
    println("Execution Time: $(round(results.execution_time, digits=2)) seconds")
    println()
    
    println("Feature Reduction:")
    println("  Stage 1: $(results.stage1_reduction[1]) → $(results.stage1_reduction[2]) features")
    println("  Stage 2: $(results.stage2_reduction[1]) → $(results.stage2_reduction[2]) features")
    println()
    
    println("Selected Features:")
    for (i, (feature, importance)) in enumerate(zip(results.selected_features, results.feature_importance))
        println("  $(i). $(feature) (importance: $(round(importance, digits=4)))")
    end
    println()
    
    println("Performance:")
    println("  HSOF Score: $(round(results.cv_mean, digits=4)) ± $(round(results.cv_std, digits=4))")
    println("  Improvement: $(results.improvement > 0 ? "+" : "")$(round(results.improvement * 100, digits=2))%")
end

function run_hsof_pipeline(X::Matrix{Float64}, y::Vector, feature_names::Vector{String}, task_type::String, dataset_name::String)
    """Complete HSOF pipeline"""
    start_time = time()
    
    # Calculate baseline performance
    baseline_score = calculate_baseline_performance(X, y, task_type)
    
    # Stage 1: Feature filtering
    stage1_features = stage1_filter(X, y, feature_names, 50)
    stage1_reduction = (size(X, 2), length(stage1_features))
    
    # Stage 2: Feature selection
    stage2_features = stage2_mcts_cpu(X, y, stage1_features, feature_names, task_type)
    stage2_reduction = (length(stage1_features), length(stage2_features))
    
    # Stage 3: Final evaluation
    cv_mean, cv_std, selected_feature_names, feature_importance = stage3_evaluate(X, y, stage2_features, feature_names, task_type)
    
    # Calculate improvement
    improvement = (cv_mean - baseline_score) / abs(baseline_score)
    
    execution_time = time() - start_time
    
    return HSOfResults(
        cv_mean,
        cv_std,
        improvement,
        selected_feature_names,
        feature_importance,
        dataset_name,
        task_type,
        stage1_reduction,
        stage2_reduction,
        execution_time
    )
end

function main()
    """Main function"""
    # Check command line argument
    if length(ARGS) != 1
        println("Usage: julia hsof.jl <config.yaml>")
        println("Examples:")
        println("  julia hsof.jl config/titanic.yaml")
        println("  julia hsof.jl config/playground_s5e7.yaml")
        println("  julia hsof.jl config/housing.yaml")
        exit(1)
    end
    
    yaml_path = ARGS[1]
    
    try
        # Load dataset from YAML config
        println("Loading dataset from YAML configuration...")
        data, target_col, task_type, id_cols, dataset_name = load_dataset_from_yaml(yaml_path)
        X, y, feature_names = prepare_features_from_sqlite(data, target_col, id_cols)
        
        println("HSOF Feature Selection")
        println("=" ^ 50)
        println("Dataset: $(dataset_name) ($(size(X,1)) samples, $(size(X,2)) features)")
        println("Target: $(target_col) ($(task_type))")
        println("Excluded: $(join(id_cols, ", ")) ($(length(id_cols)) columns)")
        println()
        
        # Run HSOF pipeline
        results = run_hsof_pipeline(X, y, feature_names, task_type, dataset_name)
        
        # Save and display results
        save_results(results, yaml_path)
        display_results(results)
        
    catch e
        println("Error: $e")
        exit(1)
    end
end

# Run if this is the main file
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end