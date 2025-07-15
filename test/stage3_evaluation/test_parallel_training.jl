using Test
using Random
using Statistics
using DataFrames

# Include the modules
include("../../src/stage3_evaluation/parallel_training.jl")
include("../../src/stage3_evaluation/mlj_infrastructure.jl")
include("../../src/stage3_evaluation/cross_validation.jl")

using .ParallelTraining
using .MLJInfrastructure
using .CrossValidation

# Set random seed for reproducibility
Random.seed!(42)

# Mock model factory for testing
function test_model_factory(model_type::Symbol, hyperparams::Dict{Symbol, Any})
    # Return a simple mock model that sleeps to simulate training
    if model_type == :fast_model
        return create_model(:xgboost, :classification, n_estimators=10)
    elseif model_type == :slow_model
        return create_model(:random_forest, :classification, n_estimators=20)
    else
        error("Unknown model type: $model_type")
    end
end

# Mock CV function for testing
function test_cv_function(model, X::AbstractMatrix, y::AbstractVector)
    # Simulate CV with random results and variable time
    sleep(0.01 + 0.09 * rand())  # 10-100ms
    
    # Generate random metrics
    accuracy = 0.7 + 0.2 * rand()
    f1_score = 0.6 + 0.3 * rand()
    
    return Dict(
        "aggregated_metrics" => Dict{Symbol, Float64}(
            :accuracy_mean => accuracy,
            :accuracy_std => 0.05 * rand(),
            :f1_score_mean => f1_score,
            :f1_score_std => 0.03 * rand()
        )
    )
end

@testset "Parallel Training Tests" begin
    
    @testset "WorkItem Creation" begin
        item = WorkItem(1, :xgboost, [1, 2, 3], Dict(:max_depth => 5), 10)
        @test item.id == 1
        @test item.model_type == :xgboost
        @test item.feature_indices == [1, 2, 3]
        @test item.hyperparams[:max_depth] == 5
        @test item.priority == 10
    end
    
    @testset "WorkQueue Operations" begin
        items = [
            WorkItem(1, :model1, [1, 2], Dict(), 5),
            WorkItem(2, :model2, [3, 4], Dict(), 10),
            WorkItem(3, :model3, [5, 6], Dict(), 1)
        ]
        
        queue = WorkQueue(items)
        @test queue.total == 3
        @test queue.completed[] == 0
        
        # Check priority ordering (highest first)
        item1 = get_next_work!(queue)
        @test item1.priority == 10
        
        item2 = get_next_work!(queue)
        @test item2.priority == 5
        
        item3 = get_next_work!(queue)
        @test item3.priority == 1
        
        # Queue should be empty
        @test isnothing(get_next_work!(queue))
    end
    
    @testset "Progress Monitor" begin
        monitor = ProgressMonitor(100, show_progress=false)
        @test monitor.total == 100
        @test monitor.completed[] == 0
        
        # Update progress
        for i in 1:10
            atomic_add!(monitor.completed, 1)
            update_progress!(monitor)
        end
        
        @test monitor.completed[] == 10
        @test monitor.results_per_second[] > 0
    end
    
    @testset "Feature Extraction" begin
        X = randn(100, 20)
        indices = [5, 10, 15]
        
        # Test without buffer
        X_subset = extract_features(X, indices)
        @test size(X_subset) == (100, 3)
        @test X_subset[:, 1] == X[:, 5]
        @test X_subset[:, 2] == X[:, 10]
        @test X_subset[:, 3] == X[:, 15]
        
        # Test with buffer
        buffer = Matrix{Float64}(undef, 100, 3)
        X_subset2 = extract_features(X, indices, buffer)
        @test X_subset2 === buffer
        @test X_subset2 == X_subset
    end
    
    @testset "Create Work Queue" begin
        feature_combs = [[1, 2], [3, 4, 5], [6]]
        model_specs = [
            (:fast_model, Dict(:param1 => 1)),
            (:slow_model, Dict(:param2 => 2))
        ]
        
        work_items = create_work_queue(feature_combs, model_specs, prioritize_small=true)
        
        @test length(work_items) == 6  # 3 feature combs × 2 models
        
        # Check prioritization (smaller feature sets first)
        priorities = [item.priority for item in work_items]
        @test issorted(priorities, rev=true)
        
        # First items should be single-feature
        @test length(work_items[1].feature_indices) == 1
    end
    
    @testset "Single Model Training" begin
        X = randn(50, 10)
        y = rand(0:1, 50)
        
        work_item = WorkItem(1, :fast_model, [1, 3, 5], Dict(:max_depth => 3), 1)
        
        result = train_single_model(work_item, X, y, test_model_factory, test_cv_function)
        
        @test result.work_id == 1
        @test result.model_type == :fast_model
        @test result.feature_indices == [1, 3, 5]
        @test haskey(result.metrics, :accuracy_mean)
        @test haskey(result.metrics, :f1_score_mean)
        @test result.training_time > 0
        @test isnothing(result.error)
    end
    
    @testset "Parallel Training Integration" begin
        # Generate test data
        n_samples = 100
        n_features = 15
        X = randn(n_samples, n_features)
        y = rand(0:1, n_samples)
        
        # Create work items
        feature_combs = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10, 11]]
        model_specs = [(:fast_model, Dict()), (:slow_model, Dict())]
        work_items = create_work_queue(feature_combs, model_specs)
        
        # Create trainer with limited threads for testing
        trainer = ParallelTrainer(work_items, n_threads=2, show_progress=false)
        
        # Run training
        results = train_models_parallel(trainer, X, y, test_model_factory, test_cv_function)
        
        @test length(results) == length(work_items)
        @test all(r -> !isnothing(r.metrics), results)
        @test trainer.progress_monitor.completed[] == length(work_items)
        
        # Check thread distribution
        thread_ids = unique(r.thread_id for r in results)
        @test length(thread_ids) <= 2
    end
    
    @testset "Result Collection and Aggregation" begin
        # Create mock results
        results = [
            TrainingResult(1, :model1, [1, 2], Dict(:accuracy_mean => 0.8, :f1_score_mean => 0.75), 1.0, 1, nothing),
            TrainingResult(2, :model1, [3, 4], Dict(:accuracy_mean => 0.85, :f1_score_mean => 0.8), 1.2, 2, nothing),
            TrainingResult(3, :model2, [1, 2], Dict(:accuracy_mean => 0.82, :f1_score_mean => 0.77), 2.0, 1, nothing),
            TrainingResult(4, :model2, [3, 4], Dict(:accuracy_mean => 0.0, :f1_score_mean => 0.0), 0.5, 2, "Error")
        ]
        
        # Test DataFrame conversion
        df = results_to_dataframe(results)
        @test nrow(df) == 4
        @test "accuracy_mean" in names(df)
        @test sum(df.has_error) == 1
        
        # Test aggregation
        aggregated = collect_results(results)
        
        @test haskey(aggregated, :model1)
        @test haskey(aggregated, :model2)
        
        # Model1 should have 2 valid results
        @test aggregated[:model1][:count] == 2
        @test aggregated[:model1][:error_count] == 0
        @test aggregated[:model1][:accuracy_mean_mean] ≈ 0.825
        
        # Model2 should have 1 valid result and 1 error
        @test aggregated[:model2][:count] == 1
        @test aggregated[:model2][:error_count] == 1
    end
    
    @testset "Memory Efficiency" begin
        # Test with larger data
        X = randn(1000, 50)
        y = rand(0:1, 1000)
        
        # Create many small work items
        feature_combs = [[i, i+1] for i in 1:2:49]  # 25 combinations
        model_specs = [(:fast_model, Dict())]
        work_items = create_work_queue(feature_combs, model_specs)
        
        trainer = ParallelTrainer(work_items, n_threads=4, 
                                memory_limit_mb=512, show_progress=false)
        
        # Check memory allocation
        initial_mem = Base.gc_bytes()
        results = train_models_parallel(trainer, X, y, test_model_factory, test_cv_function)
        final_mem = Base.gc_bytes()
        
        # Memory increase should be reasonable
        mem_increase_mb = (final_mem - initial_mem) / 1024 / 1024
        @test mem_increase_mb < 512  # Should stay within limit
        
        @test length(results) == length(work_items)
    end
    
    @testset "Interruption Handling" begin
        # Create long-running work
        X = randn(100, 10)
        y = rand(0:1, 100)
        
        # Many work items
        feature_combs = [[i] for i in 1:20]
        model_specs = [(:slow_model, Dict())]
        work_items = create_work_queue(feature_combs, model_specs)
        
        trainer = ParallelTrainer(work_items, n_threads=2, show_progress=false)
        
        # Start training in a task
        task = @async train_models_parallel(trainer, X, y, test_model_factory, test_cv_function)
        
        # Let it run briefly
        sleep(0.1)
        
        # Interrupt
        interrupt_training(trainer)
        
        # Wait for completion
        wait(task)
        
        # Should have partial results
        @test length(trainer.results) > 0
        @test length(trainer.results) < length(work_items)
        @test trainer.interrupt_flag[]
    end
    
    @testset "Load Balancing" begin
        # Create work items with known different costs
        items = [
            WorkItem(1, :fast_model, [1], Dict(), 1),
            WorkItem(2, :slow_model, [1, 2, 3, 4, 5], Dict(), 1),
            WorkItem(3, :fast_model, [2], Dict(), 1),
            WorkItem(4, :slow_model, [6, 7, 8, 9, 10], Dict(), 1),
        ]
        
        trainer = ParallelTrainer(items, n_threads=2, show_progress=false)
        
        # Add some mock results
        push!(trainer.results, TrainingResult(1, :fast_model, [1], Dict(), 0.1, 1, nothing))
        push!(trainer.results, TrainingResult(2, :slow_model, [1, 2, 3, 4, 5], Dict(), 1.0, 2, nothing))
        
        # Rebalance
        rebalance_work!(trainer)
        
        # Remaining work should be reordered
        # (This is a simplified test - real rebalancing would be more complex)
        @test length(trainer.work_queue.items) == 2
    end
    
    @testset "Error Handling" begin
        # Create a work item that will fail
        function failing_cv_function(model, X, y)
            error("Simulated training failure")
        end
        
        X = randn(50, 5)
        y = rand(0:1, 50)
        
        work_item = WorkItem(1, :fast_model, [1, 2], Dict(), 1)
        result = train_single_model(work_item, X, y, test_model_factory, failing_cv_function)
        
        @test !isnothing(result.error)
        @test occursin("Simulated training failure", result.error)
        @test isempty(result.metrics)
    end
    
    @testset "Thread Safety" begin
        # Test concurrent access to work queue
        items = [WorkItem(i, :model, [i], Dict(), i) for i in 1:100]
        queue = WorkQueue(items)
        
        results = Vector{Union{Nothing, WorkItem}}(nothing, 100)
        
        # Multiple threads pulling from queue
        @threads for i in 1:100
            results[i] = get_next_work!(queue)
        end
        
        # All items should be retrieved exactly once
        retrieved = filter(!isnothing, results)
        @test length(retrieved) == 100
        @test length(unique(r.id for r in retrieved)) == 100
    end
end

println("All parallel training tests passed! ✓")