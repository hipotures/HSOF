# Example usage of Parallel Training system for Stage 3

using Random
using Statistics
using DataFrames
using Combinatorics

# Include the modules
include("../src/stage3_evaluation/parallel_training.jl")
include("../src/stage3_evaluation/mlj_infrastructure.jl")
include("../src/stage3_evaluation/cross_validation.jl")
include("../src/stage3_evaluation/unified_prediction.jl")

using .ParallelTraining
using .MLJInfrastructure
using .CrossValidation
using .UnifiedPrediction

# Set random seed
Random.seed!(123)

"""
Example 1: Basic parallel training with multiple models
"""
function basic_parallel_example()
    println("=== Basic Parallel Training Example ===\n")
    
    # Generate synthetic data
    n_samples = 500
    n_features = 20
    X = randn(n_samples, n_features)
    
    # Create non-linear target
    y = Int.((
        X[:, 1].^2 + 
        0.5 * X[:, 2] .* X[:, 3] - 
        X[:, 4] .+ 
        0.3 * randn(n_samples)
    ) .> 0)
    
    println("Data shape: $(size(X))")
    println("Class distribution: $(countmap(y))")
    
    # Define feature combinations to test (subset of all possibilities)
    feature_indices = 1:10  # Test first 10 features
    feature_combinations = Vector{Int}[]
    
    # Test different feature subset sizes
    for subset_size in [2, 3, 4]
        for comb in combinations(feature_indices, subset_size)
            push!(feature_combinations, collect(comb))
        end
    end
    
    println("\nTesting $(length(feature_combinations)) feature combinations")
    
    # Define models to evaluate
    model_specs = [
        (:xgboost, Dict(:max_depth => 4, :n_estimators => 50)),
        (:random_forest, Dict(:n_estimators => 50)),
        (:lightgbm, Dict(:num_leaves => 20, :n_estimators => 50))
    ]
    
    # Create work queue
    work_items = create_work_queue(feature_combinations, model_specs)
    println("Created $(length(work_items)) work items")
    
    # Create CV splitter
    cv = StratifiedKFold(n_folds=3, random_state=42)
    
    # Define model factory
    function model_factory(model_type, hyperparams)
        return create_model(model_type, :classification; hyperparams...)
    end
    
    # Define CV function
    function cv_function(model, X_subset, y_subset)
        return cross_validate_model(model, X_subset, y_subset, cv, 
                                  task_type=:classification, verbose=false)
    end
    
    # Create parallel trainer
    n_threads = min(Threads.nthreads(), 4)  # Use up to 4 threads
    println("\nUsing $n_threads threads for parallel training")
    
    trainer = ParallelTrainer(work_items, n_threads=n_threads, 
                            memory_limit_mb=2048, show_progress=true)
    
    # Run parallel training
    println("\nStarting parallel training...")
    start_time = time()
    
    results = train_models_parallel(trainer, X, y, model_factory, cv_function,
                                  save_interval=50)
    
    elapsed_time = time() - start_time
    println("\nCompleted in $(round(elapsed_time, digits=2)) seconds")
    println("Average time per model: $(round(elapsed_time/length(results), digits=3)) seconds")
    
    # Analyze results
    println("\n--- Results Summary ---")
    aggregated = collect_results(results)
    
    for (model_type, stats) in aggregated
        println("\n$model_type:")
        println("  Models trained: $(stats[:count])")
        println("  Errors: $(stats[:error_count])")
        println("  Avg accuracy: $(round(stats[:accuracy_mean_mean], digits=3)) ± $(round(stats[:accuracy_mean_std], digits=3))")
        println("  Avg training time: $(round(stats[:avg_training_time], digits=3))s")
    end
    
    return results, aggregated
end

"""
Example 2: Feature selection using parallel evaluation
"""
function feature_selection_example()
    println("\n\n=== Feature Selection with Parallel Evaluation ===\n")
    
    # Generate data with known important features
    n_samples = 800
    n_features = 50  # Many features
    n_important = 5   # Only 5 are important
    
    X = randn(n_samples, n_features)
    
    # Target depends only on first n_important features
    y = Int.((
        X[:, 1] + 
        0.8 * X[:, 2] - 
        0.6 * X[:, 3] + 
        0.4 * X[:, 4] - 
        0.2 * X[:, 5] + 
        0.2 * randn(n_samples)
    ) .> 0)
    
    println("Data: $n_samples samples, $n_features features")
    println("True important features: 1-$n_important")
    
    # Stage 1: Quick filter to top 20 features using simple correlation
    correlations = [abs(cor(X[:, i], y)) for i in 1:n_features]
    top_20_indices = sortperm(correlations, rev=true)[1:20]
    println("\nStage 1: Selected top 20 features by correlation")
    
    # Stage 2: Evaluate all combinations of size 3-5 from top 20
    feature_combinations = Vector{Int}[]
    for size in 3:5
        for comb in combinations(top_20_indices, size)
            push!(feature_combinations, collect(comb))
        end
    end
    
    println("Stage 2: Evaluating $(length(feature_combinations)) combinations")
    
    # Use only XGBoost for speed
    model_specs = [
        (:xgboost, Dict(:max_depth => 3, :n_estimators => 30, :learning_rate => 0.3))
    ]
    
    # Create work queue with prioritization
    work_items = create_work_queue(feature_combinations, model_specs, 
                                 prioritize_small=true)
    
    # Setup functions
    cv = StratifiedKFold(n_folds=3, random_state=42)
    
    function model_factory(model_type, hyperparams)
        return create_model(model_type, :classification; hyperparams...)
    end
    
    function cv_function(model, X_subset, y_subset)
        return cross_validate_model(model, X_subset, y_subset, cv, 
                                  task_type=:classification, verbose=false)
    end
    
    # Run parallel evaluation
    trainer = ParallelTrainer(work_items, n_threads=Threads.nthreads(), 
                            show_progress=true)
    
    println("\nRunning parallel evaluation...")
    results = train_models_parallel(trainer, X, y, model_factory, cv_function)
    
    # Find best feature combinations
    df_results = results_to_dataframe(results)
    filter!(row -> !row.has_error, df_results)
    sort!(df_results, :accuracy_mean, rev=true)
    
    println("\n--- Top 10 Feature Combinations ---")
    for i in 1:min(10, nrow(df_results))
        row = df_results[i, :]
        features = results[row.work_id].feature_indices
        accuracy = round(row.accuracy_mean, digits=3)
        
        # Check how many true features were found
        true_features_found = sum(f <= n_important for f in features)
        
        println("$i. Features $features: accuracy=$accuracy " *
                "($(true_features_found)/$n_important true features)")
    end
    
    # Analyze feature importance across all models
    feature_counts = zeros(Int, n_features)
    feature_scores = zeros(Float64, n_features)
    
    for result in results
        if isnothing(result.error) && haskey(result.metrics, :accuracy_mean)
            score = result.metrics[:accuracy_mean]
            for idx in result.feature_indices
                feature_counts[idx] += 1
                feature_scores[idx] += score
            end
        end
    end
    
    # Normalize scores
    for i in 1:n_features
        if feature_counts[i] > 0
            feature_scores[i] /= feature_counts[i]
        end
    end
    
    # Show most selected features
    top_features = sortperm(feature_counts, rev=true)[1:10]
    println("\n--- Most Frequently Selected Features ---")
    for (rank, idx) in enumerate(top_features)
        avg_score = feature_counts[idx] > 0 ? round(feature_scores[idx], digits=3) : 0.0
        println("$rank. Feature $idx: selected $(feature_counts[idx]) times, " *
                "avg accuracy=$avg_score")
    end
    
    return df_results, feature_counts
end

"""
Example 3: Large-scale parallel evaluation with interruption
"""
function large_scale_example()
    println("\n\n=== Large-Scale Parallel Evaluation Example ===\n")
    
    # Simulate large dataset
    n_samples = 1000
    n_features = 100
    
    println("Generating large dataset: $n_samples × $n_features")
    X = randn(n_samples, n_features)
    y = rand(0:1, n_samples)
    
    # Create many feature combinations (simulate real workload)
    println("\nCreating feature combinations...")
    feature_combinations = Vector{Int}[]
    
    # Sample random combinations of different sizes
    Random.seed!(42)
    for size in [5, 10, 15, 20]
        for _ in 1:50  # 50 random combinations per size
            indices = randperm(n_features)[1:size]
            push!(feature_combinations, indices)
        end
    end
    
    println("Generated $(length(feature_combinations)) feature combinations")
    
    # Multiple model configurations
    model_specs = [
        (:xgboost, Dict(:max_depth => d, :n_estimators => n)) 
        for d in [3, 5] for n in [30, 50]
    ]
    
    work_items = create_work_queue(feature_combinations, model_specs)
    println("Total work items: $(length(work_items))")
    
    # Estimate time
    time_per_model = 0.5  # seconds (rough estimate)
    total_time_serial = length(work_items) * time_per_model
    n_threads = Threads.nthreads()
    estimated_time_parallel = total_time_serial / n_threads
    
    println("\nEstimated time:")
    println("  Serial: $(round(total_time_serial/60, digits=1)) minutes")
    println("  Parallel ($n_threads threads): $(round(estimated_time_parallel/60, digits=1)) minutes")
    
    # Setup
    cv = StratifiedKFold(n_folds=3, random_state=42)
    
    function model_factory(model_type, hyperparams)
        return create_model(model_type, :classification; hyperparams...)
    end
    
    function cv_function(model, X_subset, y_subset)
        # Add small delay to simulate realistic training
        sleep(0.1 + 0.4 * rand())
        return cross_validate_model(model, X_subset, y_subset, cv, 
                                  task_type=:classification, verbose=false)
    end
    
    # Create trainer with resource monitoring
    trainer = ParallelTrainer(work_items, n_threads=n_threads, 
                            memory_limit_mb=4096, show_progress=true)
    
    # Start resource monitoring in background
    monitor_task = @async monitor_resources(trainer)
    
    println("\nStarting large-scale parallel training...")
    println("Press Ctrl+C to interrupt and save partial results")
    
    start_time = time()
    
    # Run with interruption handling
    try
        results = train_models_parallel(trainer, X, y, model_factory, cv_function,
                                      save_interval=100)
    catch e
        if isa(e, InterruptException)
            interrupt_training(trainer)
            println("\nTraining interrupted by user")
        else
            rethrow(e)
        end
    end
    
    # Stop monitoring
    trainer.interrupt_flag[] = true
    
    elapsed = time() - start_time
    completed = trainer.progress_monitor.completed[]
    
    println("\n--- Execution Summary ---")
    println("Completed: $completed/$(length(work_items)) models")
    println("Time elapsed: $(round(elapsed, digits=1)) seconds")
    println("Models/second: $(round(completed/elapsed, digits=2))")
    println("Effective speedup: $(round(completed/elapsed/(1/time_per_model), digits=1))x")
    
    if completed > 0
        # Get results
        results = trainer.results
        df = results_to_dataframe(results)
        
        # Thread utilization
        thread_counts = countmap(r.thread_id for r in results)
        println("\nThread utilization:")
        for (tid, count) in sort(collect(thread_counts))
            percent = round(100 * count / completed, digits=1)
            println("  Thread $tid: $count models ($percent%)")
        end
        
        # Model type distribution
        model_counts = countmap(r.model_type for r in results)
        println("\nModels by type:")
        for (mtype, count) in model_counts
            println("  $mtype: $count")
        end
    end
    
    return trainer
end

"""
Example 4: Custom work prioritization and load balancing
"""
function advanced_scheduling_example()
    println("\n\n=== Advanced Scheduling Example ===\n")
    
    # Create synthetic workload with known costs
    n_samples = 500
    n_features = 30
    X = randn(n_samples, n_features)
    y = rand(0:1, n_samples)
    
    # Create work items with varying computational costs
    work_items = WorkItem[]
    work_id = 1
    
    # Fast models with small feature sets (high priority)
    for i in 1:5:25
        push!(work_items, WorkItem(
            work_id, :xgboost, [i, i+1], 
            Dict(:max_depth => 3, :n_estimators => 20),
            100  # High priority
        ))
        work_id += 1
    end
    
    # Medium models (medium priority)
    for i in 1:3:25
        push!(work_items, WorkItem(
            work_id, :random_forest, [i, i+1, i+2, i+3],
            Dict(:n_estimators => 30),
            50  # Medium priority
        ))
        work_id += 1
    end
    
    # Slow models with large feature sets (low priority)
    for i in 1:5:20
        push!(work_items, WorkItem(
            work_id, :lightgbm, collect(i:i+9),
            Dict(:num_leaves => 50, :n_estimators => 50),
            10  # Low priority
        ))
        work_id += 1
    end
    
    println("Created workload with $(length(work_items)) items:")
    println("  High priority (fast): $(sum(w.priority == 100 for w in work_items))")
    println("  Medium priority: $(sum(w.priority == 50 for w in work_items))")
    println("  Low priority (slow): $(sum(w.priority == 10 for w in work_items))")
    
    # Setup
    cv = StratifiedKFold(n_folds=3, random_state=42)
    
    function model_factory(model_type, hyperparams)
        return create_model(model_type, :classification; hyperparams...)
    end
    
    function cv_function(model, X_subset, y_subset)
        # Simulate varying costs
        base_time = model isa XGBoostClassifier ? 0.05 : 
                   model isa RandomForestClassifier ? 0.1 : 0.15
        sleep(base_time + 0.05 * rand())
        
        return cross_validate_model(model, X_subset, y_subset, cv, 
                                  task_type=:classification, verbose=false)
    end
    
    # Create trainer
    trainer = ParallelTrainer(work_items, n_threads=Threads.nthreads(), 
                            show_progress=true)
    
    # Add rebalancing task
    rebalance_task = @async begin
        while !trainer.interrupt_flag[]
            sleep(2)  # Rebalance every 2 seconds
            rebalance_work!(trainer)
        end
    end
    
    println("\nRunning with dynamic load balancing...")
    start_time = time()
    
    results = train_models_parallel(trainer, X, y, model_factory, cv_function)
    
    trainer.interrupt_flag[] = true  # Stop rebalancing
    
    elapsed = time() - start_time
    
    # Analyze scheduling efficiency
    println("\n--- Scheduling Analysis ---")
    
    # Group results by priority
    high_priority_times = Float64[]
    medium_priority_times = Float64[]
    low_priority_times = Float64[]
    
    for result in results
        work_item = first(w for w in work_items if w.id == result.work_id)
        if work_item.priority == 100
            push!(high_priority_times, result.training_time)
        elseif work_item.priority == 50
            push!(medium_priority_times, result.training_time)
        else
            push!(low_priority_times, result.training_time)
        end
    end
    
    println("\nAverage completion times by priority:")
    println("  High priority: $(round(mean(high_priority_times), digits=3))s")
    println("  Medium priority: $(round(mean(medium_priority_times), digits=3))s")
    println("  Low priority: $(round(mean(low_priority_times), digits=3))s")
    
    # Check if high priority finished first
    completion_order = sortperm(results, by=r->r.work_id)
    high_priority_positions = Int[]
    
    for (pos, idx) in enumerate(completion_order)
        work_item = first(w for w in work_items if w.id == results[idx].work_id)
        if work_item.priority == 100
            push!(high_priority_positions, pos)
        end
    end
    
    avg_position = mean(high_priority_positions)
    println("\nHigh priority average completion position: $(round(avg_position, digits=1))/$(length(results))")
    println("(Lower is better - should be < $(length(results)/2) for good prioritization)")
    
    return results
end

# Run all examples
if abspath(PROGRAM_FILE) == @__FILE__
    println("Parallel Training System Examples")
    println("=" ^ 50)
    
    # Check thread availability
    println("Available threads: $(Threads.nthreads())")
    if Threads.nthreads() == 1
        println("WARNING: Only 1 thread available. Run Julia with -t auto for parallel execution.")
    end
    println()
    
    # Run examples
    results1, aggregated1 = basic_parallel_example()
    df_results2, feature_counts2 = feature_selection_example()
    
    # Only run large-scale example if enough threads
    if Threads.nthreads() >= 4
        trainer3 = large_scale_example()
    else
        println("\n\nSkipping large-scale example (requires >= 4 threads)")
    end
    
    results4 = advanced_scheduling_example()
    
    println("\n" * "=" ^ 50)
    println("All parallel training examples completed!")
    println("\nKey takeaways:")
    println("- Parallel training provides linear speedup with thread count")
    println("- Work prioritization ensures important models complete first")
    println("- Memory-efficient feature extraction prevents overhead")
    println("- Progress monitoring helps track long-running jobs")
    println("- Interruption handling allows graceful early stopping")
end