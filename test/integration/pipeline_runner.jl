module PipelineRunner

using Test
using Statistics
using Dates
using CUDA

# Include necessary modules
push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
include("../../src/core/models.jl")
include("../../src/stage1/statistical_filter.jl")
include("../../src/stage2/mcts_feature_selection.jl")
include("../../src/stage3/ensemble_optimizer.jl")
include("../../src/stage3/feature_importance.jl")
include("../../src/stage3/model_evaluation.jl")
include("data/dataset_loaders.jl")
include("fixtures/expected_outputs.jl")

using .Models
using .StatisticalFilter
using .MCTSFeatureSelection
using .EnsembleOptimizer
using .FeatureImportance
using .ModelEvaluation
using .DatasetLoaders
using .ExpectedOutputs

export run_pipeline_test, PipelineTestResult, run_all_pipeline_tests

"""
Result of a pipeline test run
"""
struct PipelineTestResult
    dataset_name::String
    total_runtime::Float64
    peak_memory_mb::Float64
    stage_results::Dict{Int, Dict{String, Any}}
    feature_reduction::Vector{Int}  # [initial, after_stage1, after_stage2, final]
    quality_scores::Vector{Float64}
    passed::Bool
    errors::Vector{String}
    warnings::Vector{String}
end

"""
Memory tracking utilities
"""
mutable struct MemoryTracker
    initial_memory::Float64
    peak_memory::Float64
    current_stage::Int
    measurements::Vector{Float64}
end

function track_memory!(tracker::MemoryTracker)
    current = get_memory_usage()
    push!(tracker.measurements, current)
    tracker.peak_memory = max(tracker.peak_memory, current)
    return current - tracker.initial_memory
end

function get_memory_usage()
    # Get process memory usage in MB
    if Sys.islinux()
        pid = getpid()
        statm_file = "/proc/$pid/statm"
        if isfile(statm_file)
            pages = parse(Int, split(read(statm_file, String))[1])
            page_size = 4096  # Usually 4KB on Linux
            return pages * page_size / 1024^2
        end
    end
    # Fallback: estimate based on Julia's GC
    return Base.gc_live_bytes() / 1024^2
end

"""
Run complete pipeline test for a dataset
"""
function run_pipeline_test(
    dataset::DatasetInfo;
    verbose::Bool = true,
    validate_gpu::Bool = true
)::PipelineTestResult
    
    if verbose
        println("\n" * "=" * 60)
        println("Running Pipeline Test: $(dataset.name)")
        println("=" * 60)
        println("Dataset: $(dataset.n_samples) samples × $(dataset.n_features) features")
    end
    
    # Initialize tracking
    start_time = time()
    memory_tracker = MemoryTracker(get_memory_usage(), 0.0, 0, Float64[])
    errors = String[]
    warnings = String[]
    stage_results = Dict{Int, Dict{String, Any}}()
    feature_reduction = [dataset.n_features]
    quality_scores = Float64[]
    
    # Prepare dataset
    X, y, metadata = prepare_dataset_for_pipeline(dataset)
    
    # Get pipeline configuration
    config = get_pipeline_config(dataset.name)
    
    # Check GPU availability if needed
    if validate_gpu && get(config["stage2"], "gpu_enabled", false)
        if !CUDA.functional()
            push!(warnings, "GPU requested but CUDA not functional - falling back to CPU")
            config["stage2"]["gpu_enabled"] = false
        end
    end
    
    try
        # Stage 1: Statistical Filtering
        if verbose
            println("\n--- Stage 1: Statistical Filtering ---")
        end
        
        stage1_result = run_stage1(X, y, config["stage1"], memory_tracker, verbose)
        stage_results[1] = stage1_result
        push!(feature_reduction, length(stage1_result["selected_features"]))
        push!(quality_scores, stage1_result["quality_score"])
        
        # Validate Stage 1
        expected1 = get_expected_output(dataset.name, 1)
        validation1 = validate_stage_output(
            stage1_result["selected_features"],
            stage1_result["runtime"],
            stage1_result["memory_mb"],
            stage1_result["quality_score"],
            expected1
        )
        
        if !validation1["passed"]
            append!(errors, validation1["errors"])
        end
        append!(warnings, validation1["warnings"])
        
        # Stage 2: MCTS Feature Selection
        if verbose
            println("\n--- Stage 2: MCTS Feature Selection ---")
        end
        
        # Prepare filtered data
        X_filtered = X[:, stage1_result["selected_features"]]
        
        stage2_result = run_stage2(
            X_filtered, y, 
            stage1_result["selected_features"],
            config["stage2"], 
            memory_tracker, 
            verbose
        )
        stage_results[2] = stage2_result
        push!(feature_reduction, length(stage2_result["selected_features"]))
        push!(quality_scores, stage2_result["quality_score"])
        
        # Validate Stage 2
        expected2 = get_expected_output(dataset.name, 2)
        validation2 = validate_stage_output(
            stage2_result["selected_features"],
            stage2_result["runtime"],
            stage2_result["memory_mb"],
            stage2_result["quality_score"],
            expected2
        )
        
        if !validation2["passed"]
            append!(errors, validation2["errors"])
        end
        append!(warnings, validation2["warnings"])
        
        # Stage 3: Ensemble Optimization
        if verbose
            println("\n--- Stage 3: Ensemble Optimization ---")
        end
        
        # Map features back to original indices
        stage2_global_features = stage1_result["selected_features"][stage2_result["selected_features"]]
        X_stage2 = X[:, stage2_global_features]
        
        stage3_result = run_stage3(
            X_stage2, y,
            stage2_global_features,
            config["stage3"],
            memory_tracker,
            verbose
        )
        stage_results[3] = stage3_result
        push!(feature_reduction, length(stage3_result["selected_features"]))
        push!(quality_scores, stage3_result["quality_score"])
        
        # Validate Stage 3
        expected3 = get_expected_output(dataset.name, 3)
        validation3 = validate_stage_output(
            stage3_result["selected_features"],
            stage3_result["runtime"],
            stage3_result["memory_mb"],
            stage3_result["quality_score"],
            expected3
        )
        
        if !validation3["passed"]
            append!(errors, validation3["errors"])
        end
        append!(warnings, validation3["warnings"])
        
    catch e
        push!(errors, "Pipeline failed: $(sprint(showerror, e))")
        if verbose
            println("\n❌ Pipeline failed with error:")
            showerror(stdout, e)
            println()
        end
    end
    
    # Calculate total metrics
    total_runtime = time() - start_time
    peak_memory_mb = memory_tracker.peak_memory - memory_tracker.initial_memory
    
    # Determine if test passed
    passed = isempty(errors)
    
    # Print summary
    if verbose
        println("\n" * "=" * 60)
        println("Pipeline Test Summary: $(dataset.name)")
        println("=" * 60)
        println("Status: $(passed ? "✓ PASSED" : "✗ FAILED")")
        println("Total runtime: $(round(total_runtime, digits=2)) seconds")
        println("Peak memory: $(round(peak_memory_mb, digits=2)) MB")
        println("Feature reduction: $(join(feature_reduction, " → "))")
        println("Quality scores: $(join(round.(quality_scores, digits=3), " → "))")
        
        if !isempty(errors)
            println("\nErrors:")
            for error in errors
                println("  ❌ $error")
            end
        end
        
        if !isempty(warnings)
            println("\nWarnings:")
            for warning in warnings
                println("  ⚠ $warning")
            end
        end
    end
    
    return PipelineTestResult(
        dataset.name,
        total_runtime,
        peak_memory_mb,
        stage_results,
        feature_reduction,
        quality_scores,
        passed,
        errors,
        warnings
    )
end

"""
Run Stage 1: Statistical Filtering
"""
function run_stage1(X, y, config, memory_tracker, verbose)
    start_time = time()
    initial_memory = track_memory!(memory_tracker)
    
    # Create filter based on method
    method = get(config, "method", "variance")
    filter = if method == "variance"
        VarianceFilter(threshold=config["threshold"])
    elseif method == "correlation"
        CorrelationFilter(
            target_correlation_threshold=config["threshold"],
            feature_correlation_threshold=0.95
        )
    else
        MutualInformationFilter(threshold=config["threshold"])
    end
    
    # Fit and transform
    selected_features = fit_transform(filter, X, y)
    
    # Calculate quality score
    importance_scores = get_feature_importance(filter)
    quality_score = mean(importance_scores[selected_features])
    
    # Track metrics
    runtime = time() - start_time
    memory_mb = track_memory!(memory_tracker) - initial_memory
    
    if verbose
        println("  Method: $method")
        println("  Selected features: $(length(selected_features)) / $(size(X, 2))")
        println("  Runtime: $(round(runtime, digits=2))s")
        println("  Memory: $(round(memory_mb, digits=2)) MB")
        println("  Quality score: $(round(quality_score, digits=3))")
    end
    
    return Dict(
        "selected_features" => selected_features,
        "importance_scores" => importance_scores,
        "runtime" => runtime,
        "memory_mb" => memory_mb,
        "quality_score" => quality_score,
        "method" => method
    )
end

"""
Run Stage 2: MCTS Feature Selection
"""
function run_stage2(X, y, feature_indices, config, memory_tracker, verbose)
    start_time = time()
    initial_memory = track_memory!(memory_tracker)
    
    # Configure MCTS
    n_features_to_select = get(config, "n_features", 50)
    selector = MCTSFeatureSelector(
        n_iterations = config["max_iterations"],
        exploration_constant = config["exploration_constant"],
        n_trees = config["num_trees"],
        feature_batch_size = min(10, n_features_to_select ÷ 5),
        use_gpu = config["gpu_enabled"],
        n_simulations = 100
    )
    
    # Run selection
    selected_indices, scores = select_features(selector, X, y, n_features_to_select)
    
    # Calculate quality score
    quality_score = mean(scores[selected_indices])
    
    # Track metrics
    runtime = time() - start_time
    memory_mb = track_memory!(memory_tracker) - initial_memory
    
    if verbose
        println("  Trees: $(config["num_trees"])")
        println("  Iterations: $(config["max_iterations"])")
        println("  GPU enabled: $(config["gpu_enabled"])")
        println("  Selected features: $(length(selected_indices)) / $(size(X, 2))")
        println("  Runtime: $(round(runtime, digits=2))s")
        println("  Memory: $(round(memory_mb, digits=2)) MB")
        println("  Quality score: $(round(quality_score, digits=3))")
    end
    
    return Dict(
        "selected_features" => selected_indices,
        "feature_scores" => scores,
        "runtime" => runtime,
        "memory_mb" => memory_mb,
        "quality_score" => quality_score,
        "gpu_used" => config["gpu_enabled"] && CUDA.functional()
    )
end

"""
Run Stage 3: Ensemble Optimization
"""
function run_stage3(X, y, feature_indices, config, memory_tracker, verbose)
    start_time = time()
    initial_memory = track_memory!(memory_tracker)
    
    # Configure ensemble
    n_features_to_select = 15  # Final target
    evaluator = CrossValidationEvaluator(
        n_folds = config["cv_folds"],
        scoring_metric = :f1_weighted
    )
    
    optimizer = EnsembleFeatureOptimizer(
        base_models = [
            RandomForestImportance(n_estimators=100),
            GradientBoostingImportance(n_estimators=50),
            MutualInformationImportance(),
            LassoImportance(alpha=0.1)
        ],
        ensemble_size = config["ensemble_size"],
        consensus_threshold = config["consensus_threshold"],
        evaluator = evaluator
    )
    
    # Run optimization
    final_features, ensemble_result = optimize_features(
        optimizer, X, y, n_features_to_select
    )
    
    # Calculate quality score
    quality_score = ensemble_result.final_score
    
    # Track metrics
    runtime = time() - start_time
    memory_mb = track_memory!(memory_tracker) - initial_memory
    
    if verbose
        println("  Ensemble size: $(config["ensemble_size"])")
        println("  CV folds: $(config["cv_folds"])")
        println("  Consensus threshold: $(config["consensus_threshold"])")
        println("  Selected features: $(length(final_features))")
        println("  Runtime: $(round(runtime, digits=2))s")
        println("  Memory: $(round(memory_mb, digits=2)) MB")
        println("  Quality score: $(round(quality_score, digits=3))")
    end
    
    return Dict(
        "selected_features" => feature_indices[final_features],
        "ensemble_scores" => ensemble_result.consensus_scores,
        "runtime" => runtime,
        "memory_mb" => memory_mb,
        "quality_score" => quality_score,
        "stability_score" => ensemble_result.stability_score
    )
end

"""
Run all pipeline tests
"""
function run_all_pipeline_tests(;
    datasets::Union{Nothing, Vector{String}} = nothing,
    verbose::Bool = true,
    parallel::Bool = false
)
    # Load datasets
    all_datasets = load_all_reference_datasets()
    
    # Filter datasets if specified
    if !isnothing(datasets)
        all_datasets = Dict(k => v for (k, v) in all_datasets if k in datasets)
    end
    
    println("\nRunning Pipeline Tests")
    println("=" * 60)
    println("Datasets: $(join(keys(all_datasets), ", "))")
    println("Parallel: $parallel")
    
    # Run tests
    results = Dict{String, PipelineTestResult}()
    
    if parallel && length(all_datasets) > 1
        # Run in parallel
        tasks = []
        for (name, dataset) in all_datasets
            task = @async run_pipeline_test(dataset, verbose=verbose)
            push!(tasks, (name, task))
        end
        
        for (name, task) in tasks
            results[name] = fetch(task)
        end
    else
        # Run sequentially
        for (name, dataset) in all_datasets
            results[name] = run_pipeline_test(dataset, verbose=verbose)
        end
    end
    
    # Print summary
    println("\n" * "=" * 60)
    println("Pipeline Test Summary")
    println("=" * 60)
    
    total_passed = 0
    for (name, result) in results
        status = result.passed ? "✓ PASSED" : "✗ FAILED"
        println("$name: $status")
        println("  Runtime: $(round(result.total_runtime, digits=2))s")
        println("  Memory: $(round(result.peak_memory_mb, digits=2)) MB")
        println("  Features: $(join(result.feature_reduction, " → "))")
        
        if result.passed
            total_passed += 1
        else
            println("  Errors: $(length(result.errors))")
        end
    end
    
    println("\nTotal: $total_passed / $(length(results)) passed")
    
    return results
end

end # module