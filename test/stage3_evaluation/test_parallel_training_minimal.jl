using Test
using Random
using Statistics
using DataFrames
using Base.Threads

# Include only the parallel training module
include("../../src/stage3_evaluation/parallel_training.jl")

using .ParallelTraining
import .ParallelTraining: WorkItem, WorkQueue, TrainingResult, ProgressMonitor
import .ParallelTraining: get_next_work!, update_progress!, extract_features
import .ParallelTraining: results_to_dataframe, collect_results
import .ParallelTraining: train_models_parallel, interrupt_training

# Set random seed for reproducibility
Random.seed!(42)

# Simple mock model and CV functions for testing
struct MockModel
    name::Symbol
    params::Dict{Symbol, Any}
end

function mock_model_factory(model_type::Symbol, hyperparams::Dict{Symbol, Any})
    return MockModel(model_type, hyperparams)
end

function mock_cv_function(model::MockModel, X::AbstractMatrix, y::AbstractVector)
    # Simulate training with variable time
    sleep(0.01 + 0.04 * rand())
    
    # Generate mock metrics
    accuracy = 0.7 + 0.2 * rand()
    f1_score = 0.6 + 0.3 * rand()
    
    return Dict(
        "aggregated_metrics" => Dict{Symbol, Float64}(
            :accuracy_mean => accuracy,
            :accuracy_std => 0.05 * rand(),
            :f1_score_mean => f1_score,
            :f1_score_std => 0.03 * rand(),
            :n_folds => 3
        )
    )
end

@testset "Parallel Training Minimal Tests" begin
    
    @testset "Basic Components" begin
        # Test WorkItem
        item = WorkItem(1, :xgboost, [1, 2, 3], Dict(:max_depth => 5), 10)
        @test item.id == 1
        @test item.model_type == :xgboost
        @test length(item.feature_indices) == 3
        @test item.priority == 10
        
        # Test WorkQueue
        items = [
            WorkItem(1, :model1, [1, 2], Dict(), 5),
            WorkItem(2, :model2, [3, 4], Dict(), 10),
            WorkItem(3, :model3, [5, 6], Dict(), 1)
        ]
        
        queue = WorkQueue(items)
        @test queue.total == 3
        
        # Items should be ordered by priority
        item1 = get_next_work!(queue)
        @test item1.priority == 10
    end
    
    @testset "Progress Monitor" begin
        monitor = ProgressMonitor(50, show_progress=false)
        @test monitor.total == 50
        @test monitor.completed[] == 0
        
        # Simulate progress
        atomic_add!(monitor.completed, 5)
        update_progress!(monitor)
        @test monitor.completed[] == 5
    end
    
    @testset "Feature Extraction" begin
        X = randn(100, 20)
        indices = [5, 10, 15]
        
        X_subset = extract_features(X, indices)
        @test size(X_subset) == (100, 3)
        @test X_subset[:, 1] == X[:, 5]
    end
    
    @testset "Training Result" begin
        result = TrainingResult(
            1, :xgboost, [1, 2, 3],
            Dict(:accuracy => 0.85),
            1.5, 1, nothing
        )
        
        @test result.work_id == 1
        @test result.model_type == :xgboost
        @test result.training_time == 1.5
        @test isnothing(result.error)
    end
    
    @testset "Results to DataFrame" begin
        results = [
            TrainingResult(1, :model1, [1, 2], Dict(:acc => 0.8), 1.0, 1, nothing),
            TrainingResult(2, :model2, [3, 4], Dict(:acc => 0.85), 1.2, 2, nothing),
            TrainingResult(3, :model1, [5, 6], Dict(:acc => 0.0), 0.5, 1, "Error")
        ]
        
        df = results_to_dataframe(results)
        @test nrow(df) == 3
        @test sum(df.has_error) == 1
        @test "n_features" in names(df)
    end
    
    @testset "Result Aggregation" begin
        results = [
            TrainingResult(1, :model1, [1, 2], Dict(:accuracy_mean => 0.8), 1.0, 1, nothing),
            TrainingResult(2, :model1, [3, 4], Dict(:accuracy_mean => 0.85), 1.2, 2, nothing),
            TrainingResult(3, :model2, [1, 2], Dict(:accuracy_mean => 0.82), 2.0, 1, nothing)
        ]
        
        aggregated = collect_results(results)
        
        @test haskey(aggregated, :model1)
        @test haskey(aggregated, :model2)
        @test aggregated[:model1][:count] == 2
        @test aggregated[:model1][:accuracy_mean_mean] ≈ 0.825
    end
    
    @testset "Parallel Execution" begin
        # Generate test data
        X = randn(50, 10)
        y = rand(0:1, 50)
        
        # Create simple work items
        work_items = [
            WorkItem(i, :test_model, [i, i+1], Dict(:param => i), i)
            for i in 1:10
        ]
        
        # Create trainer
        trainer = ParallelTrainer(work_items, n_threads=2, show_progress=false)
        
        # Run training
        results = train_models_parallel(trainer, X, y, mock_model_factory, mock_cv_function)
        
        @test length(results) == 10
        @test all(r -> haskey(r.metrics, :accuracy_mean), results)
        @test all(r -> r.training_time > 0, results)
    end
    
    @testset "Thread Safety" begin
        # Test concurrent queue access
        items = [WorkItem(i, :model, [i], Dict(), i) for i in 1:50]
        queue = WorkQueue(items)
        
        retrieved = Vector{Union{Nothing, WorkItem}}(nothing, 50)
        
        @threads for i in 1:50
            retrieved[i] = get_next_work!(queue)
        end
        
        # All items should be retrieved
        valid_items = filter(!isnothing, retrieved)
        @test length(valid_items) == 50
        @test length(unique(r.id for r in valid_items)) == 50
    end
    
    @testset "Interruption" begin
        X = randn(50, 10)
        y = rand(0:1, 50)
        
        # Many work items
        work_items = [WorkItem(i, :model, [i], Dict(), 1) for i in 1:50]
        trainer = ParallelTrainer(work_items, n_threads=2, show_progress=false)
        
        # Start training
        task = @async train_models_parallel(trainer, X, y, mock_model_factory, mock_cv_function)
        
        # Interrupt quickly
        sleep(0.1)
        interrupt_training(trainer)
        wait(task)
        
        # Should have partial results
        @test length(trainer.results) > 0
        @test length(trainer.results) < 50
    end
end

println("Minimal parallel training tests completed! ✓")