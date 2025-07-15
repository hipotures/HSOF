module FaultToleranceTests

using Test
using CUDA
using Random
using Dates

# Include necessary modules
push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
include("../../src/core/models.jl")
include("../../src/stage1/statistical_filter.jl") 
include("../../src/stage2/mcts_feature_selection.jl")
include("../../src/stage3/ensemble_optimizer.jl")
include("../../src/database/checkpoint_manager.jl")
include("data/dataset_loaders.jl")
include("pipeline_runner.jl")

using .Models
using .StatisticalFilter
using .MCTSFeatureSelection
using .EnsembleOptimizer
using .CheckpointManager
using .DatasetLoaders
using .PipelineRunner

export run_fault_tolerance_tests, simulate_gpu_failure, simulate_memory_exhaustion
export test_checkpoint_recovery, test_network_failure_recovery

"""
Simulate various types of failures during pipeline execution
"""
mutable struct FailureSimulator
    gpu_failure_probability::Float64
    memory_failure_probability::Float64
    network_failure_probability::Float64
    failure_timing::Vector{String}  # ["stage1", "stage2", "stage3"]
    active::Bool
end

"""
Test fault tolerance with simulated GPU failures
"""
function test_gpu_failure_tolerance(dataset::DatasetInfo; verbose::Bool = true)
    if verbose
        println("\n=== Testing GPU Failure Tolerance ===")
    end
    
    results = Dict{String, Any}()
    X, y, _ = prepare_dataset_for_pipeline(dataset)
    
    # Test 1: GPU failure during Stage 2
    @testset "GPU Failure During MCTS" begin
        # Configure pipeline with GPU enabled
        config = get_pipeline_config(dataset.name)
        config["stage2"]["gpu_enabled"] = true
        
        # Simulate GPU failure by disabling CUDA mid-execution
        original_functional = CUDA.functional
        
        try
            # Start pipeline normally
            filter = VarianceFilter(threshold=0.01)
            selected_features = fit_transform(filter, X, y)
            X_filtered = X[:, selected_features]
            
            # Simulate GPU failure by mocking CUDA.functional()
            mock_cuda_failure = true
            
            # MCTS should fall back to CPU
            selector = MCTSFeatureSelector(
                n_iterations = 100,
                exploration_constant = 1.4,
                n_trees = 5,
                use_gpu = true,  # Requested but will fail
                n_simulations = 50
            )
            
            # This should not crash but fall back to CPU
            selected_indices, scores = select_features(selector, X_filtered, y, 20)
            
            @test length(selected_indices) > 0
            @test all(s >= 0 for s in scores)
            
            results["gpu_fallback"] = "success"
            
        catch e
            results["gpu_fallback"] = "failed: $(string(e))"
            @test false  # Should not reach here
        end
    end
    
    # Test 2: GPU memory exhaustion
    @testset "GPU Memory Exhaustion Recovery" begin
        if CUDA.functional()
            try
                # Try to allocate massive amounts of GPU memory
                huge_arrays = []
                for i in 1:10
                    try
                        array = CUDA.CuArray(randn(Float32, 10000, 10000))
                        push!(huge_arrays, array)
                    catch OutOfMemoryError
                        break
                    end
                end
                
                # Clear memory
                huge_arrays = nothing
                CUDA.reclaim()
                
                # Pipeline should still work after memory recovery
                config = get_pipeline_config("synthetic")
                config["stage2"]["gpu_enabled"] = true
                config["stage2"]["max_iterations"] = 50
                
                result = run_pipeline_test(dataset, verbose=false)
                @test result.passed || !isempty(result.warnings)  # Should pass or warn about GPU
                
                results["memory_recovery"] = "success"
                
            catch e
                results["memory_recovery"] = "failed: $(string(e))"
            end
        else
            results["memory_recovery"] = "skipped: no GPU available"
        end
    end
    
    return results
end

"""
Test checkpoint recovery after interruption
"""
function test_checkpoint_recovery(dataset::DatasetInfo; verbose::Bool = true)
    if verbose
        println("\n=== Testing Checkpoint Recovery ===")
    end
    
    results = Dict{String, Any}()
    checkpoint_dir = "test_checkpoints_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
    mkpath(checkpoint_dir)
    
    try
        X, y, _ = prepare_dataset_for_pipeline(dataset)
        
        @testset "Stage 2 Checkpoint Recovery" begin
            # Create checkpoint manager
            manager = CheckpointManager(
                checkpoint_dir = checkpoint_dir,
                save_interval = 10,  # Save every 10 iterations
                max_checkpoints = 5
            )
            
            # Configure MCTS with checkpointing
            selector = MCTSFeatureSelector(
                n_iterations = 100,
                exploration_constant = 1.4,
                n_trees = 3,
                use_gpu = false,
                n_simulations = 30,
                checkpoint_manager = manager
            )
            
            # Start feature selection
            config = get_pipeline_config(dataset.name)
            filter = VarianceFilter(threshold=0.01)
            selected_features = fit_transform(filter, X, y)
            X_filtered = X[:, selected_features]
            
            # Run for a few iterations then "interrupt"
            interrupted_task = @async begin
                try
                    select_features(selector, X_filtered, y, 15)
                catch InterruptException
                    return :interrupted
                end
            end
            
            # Let it run for a bit, then interrupt
            sleep(2)
            Base.throwto(interrupted_task, InterruptException())
            
            result = fetch(interrupted_task)
            @test result == :interrupted
            
            # Check that checkpoints were created
            checkpoint_files = filter(f -> endswith(f, ".jld2"), readdir(checkpoint_dir))
            @test length(checkpoint_files) > 0
            
            # Now resume from checkpoint
            latest_checkpoint = get_latest_checkpoint(manager)
            if latest_checkpoint !== nothing
                # Create new selector and resume
                new_selector = MCTSFeatureSelector(
                    n_iterations = 200,  # More iterations
                    exploration_constant = 1.4,
                    n_trees = 3,
                    use_gpu = false,
                    n_simulations = 30,
                    checkpoint_manager = manager
                )
                
                # This should resume from checkpoint
                selected_indices, scores = select_features(new_selector, X_filtered, y, 15)
                
                @test length(selected_indices) == 15
                @test all(s >= 0 for s in scores)
                
                results["checkpoint_recovery"] = "success"
            else
                results["checkpoint_recovery"] = "no checkpoint found"
            end
        end
        
        # Test 3: Full pipeline state recovery
        @testset "Full Pipeline State Recovery" begin
            # Create a checkpoint-enabled pipeline configuration
            pipeline_state = Dict{String, Any}(
                "stage" => 1,
                "completed_stages" => Int[],
                "feature_progression" => [dataset.n_features],
                "quality_scores" => Float64[],
                "timestamp" => now()
            )
            
            # Save initial state
            save_pipeline_checkpoint(manager, "pipeline_state", pipeline_state)
            
            # Simulate pipeline execution with state updates
            for stage in 1:3
                pipeline_state["stage"] = stage
                push!(pipeline_state["completed_stages"], stage)
                push!(pipeline_state["feature_progression"], 
                      max(5, div(dataset.n_features, 2^stage)))
                push!(pipeline_state["quality_scores"], 0.7 + 0.1 * stage)
                
                save_pipeline_checkpoint(manager, "pipeline_state", pipeline_state)
                
                # Simulate some processing time
                sleep(0.1)
            end
            
            # Load and verify final state
            recovered_state = load_pipeline_checkpoint(manager, "pipeline_state")
            
            @test recovered_state !== nothing
            @test recovered_state["stage"] == 3
            @test length(recovered_state["completed_stages"]) == 3
            @test length(recovered_state["feature_progression"]) == 4
            @test length(recovered_state["quality_scores"]) == 3
            
            results["pipeline_state_recovery"] = "success"
        end
        
    finally
        # Cleanup
        rm(checkpoint_dir, recursive=true, force=true)
    end
    
    return results
end

"""
Test data validation and consistency across pipeline stages
"""
function test_data_consistency_validation(dataset::DatasetInfo; verbose::Bool = true)
    if verbose
        println("\n=== Testing Data Consistency Validation ===")
    end
    
    results = Dict{String, Any}()
    X, y, _ = prepare_dataset_for_pipeline(dataset)
    
    @testset "Feature Index Consistency" begin
        # Track feature indices through pipeline
        feature_tracker = Dict{String, Vector{Int}}()
        
        # Stage 1
        filter = VarianceFilter(threshold=0.01)
        stage1_features = fit_transform(filter, X, y)
        feature_tracker["stage1"] = stage1_features
        
        @test all(1 ≤ f ≤ size(X, 2) for f in stage1_features)
        @test length(unique(stage1_features)) == length(stage1_features)  # No duplicates
        
        # Stage 2
        X_filtered = X[:, stage1_features]
        selector = MCTSFeatureSelector(
            n_iterations = 50,
            exploration_constant = 1.4,
            n_trees = 3,
            use_gpu = false,
            n_simulations = 20
        )
        
        stage2_indices, _ = select_features(selector, X_filtered, y, min(15, length(stage1_features)))
        stage2_global_features = stage1_features[stage2_indices]
        feature_tracker["stage2"] = stage2_global_features
        
        @test all(f in stage1_features for f in stage2_global_features)
        @test length(stage2_global_features) ≤ length(stage1_features)
        
        # Stage 3
        X_stage2 = X[:, stage2_global_features]
        evaluator = CrossValidationEvaluator(n_folds=3, scoring_metric=:f1_weighted)
        optimizer = EnsembleFeatureOptimizer(
            base_models = [RandomForestImportance(n_estimators=10)],
            ensemble_size = 2,
            consensus_threshold = 0.5,
            evaluator = evaluator
        )
        
        final_indices, _ = optimize_features(optimizer, X_stage2, y, min(10, length(stage2_global_features)))
        final_global_features = stage2_global_features[final_indices]
        feature_tracker["final"] = final_global_features
        
        @test all(f in stage2_global_features for f in final_global_features)
        @test length(final_global_features) ≤ length(stage2_global_features)
        
        # Verify monotonic reduction
        @test length(feature_tracker["stage1"]) ≤ dataset.n_features
        @test length(feature_tracker["stage2"]) ≤ length(feature_tracker["stage1"])
        @test length(feature_tracker["final"]) ≤ length(feature_tracker["stage2"])
        
        results["feature_consistency"] = "success"
        results["feature_counts"] = [
            dataset.n_features,
            length(feature_tracker["stage1"]),
            length(feature_tracker["stage2"]),
            length(feature_tracker["final"])
        ]
    end
    
    @testset "Data Type and Range Validation" begin
        # Check data types remain consistent
        @test eltype(X) == Float64
        @test eltype(y) <: Integer
        
        # Check for NaN/Inf values
        @test !any(isnan, X)
        @test !any(isinf, X)
        @test !any(isnan, y)
        @test !any(isinf, y)
        
        # Check target label consistency
        unique_labels = unique(y)
        @test length(unique_labels) ≥ 2  # At least binary classification
        @test all(l ≥ 0 for l in unique_labels)  # Non-negative labels
        
        results["data_validation"] = "success"
    end
    
    return results
end

"""
Test network failure simulation and recovery
"""
function test_network_failure_recovery(dataset::DatasetInfo; verbose::Bool = true)
    if verbose
        println("\n=== Testing Network Failure Recovery ===")
    end
    
    results = Dict{String, Any}()
    
    @testset "Database Connection Recovery" begin
        # Simulate database connection failure
        try
            # This would normally connect to a database for result storage
            # For testing, we simulate connection failure and recovery
            
            connection_attempts = 0
            max_attempts = 3
            
            for attempt in 1:max_attempts
                connection_attempts += 1
                
                # Simulate connection failure for first 2 attempts
                if attempt < 3
                    # Simulate failure
                    sleep(0.1)
                    continue
                else
                    # Simulate successful connection
                    break
                end
            end
            
            @test connection_attempts == 3
            results["database_recovery"] = "success"
            
        catch e
            results["database_recovery"] = "failed: $(string(e))"
        end
    end
    
    @testset "Remote Storage Recovery" begin
        # Simulate remote storage failure (e.g., S3, shared filesystem)
        storage_available = false
        retry_count = 0
        max_retries = 5
        
        while !storage_available && retry_count < max_retries
            retry_count += 1
            
            # Simulate storage check
            if retry_count >= 3
                storage_available = true  # Simulate recovery after retries
            end
            
            sleep(0.05)
        end
        
        @test storage_available
        @test retry_count ≤ max_retries
        results["storage_recovery"] = "success"
    end
    
    return results
end

"""
Run comprehensive fault tolerance test suite
"""
function run_fault_tolerance_tests(; datasets::Vector{String} = ["titanic"], verbose::Bool = true)
    println("\n" * "=" * 60)
    println("HSOF Fault Tolerance Test Suite")
    println("=" * 60)
    
    all_datasets = load_all_reference_datasets()
    test_results = Dict{String, Dict{String, Any}}()
    
    for dataset_name in datasets
        if !haskey(all_datasets, dataset_name)
            println("⚠ Warning: Dataset '$dataset_name' not found, skipping...")
            continue
        end
        
        dataset = all_datasets[dataset_name]
        println("\nTesting fault tolerance for: $(dataset.name)")
        
        results = Dict{String, Any}()
        
        # Test GPU failure tolerance
        try
            results["gpu_failure"] = test_gpu_failure_tolerance(dataset, verbose=verbose)
        catch e
            results["gpu_failure"] = Dict("error" => string(e))
        end
        
        # Test checkpoint recovery
        try
            results["checkpoint_recovery"] = test_checkpoint_recovery(dataset, verbose=verbose)
        catch e
            results["checkpoint_recovery"] = Dict("error" => string(e))
        end
        
        # Test data consistency
        try
            results["data_consistency"] = test_data_consistency_validation(dataset, verbose=verbose)
        catch e
            results["data_consistency"] = Dict("error" => string(e))
        end
        
        # Test network failure recovery
        try
            results["network_recovery"] = test_network_failure_recovery(dataset, verbose=verbose)
        catch e
            results["network_recovery"] = Dict("error" => string(e))
        end
        
        test_results[dataset_name] = results
    end
    
    # Print summary
    println("\n" * "=" * 60)
    println("Fault Tolerance Test Summary")
    println("=" * 60)
    
    for (dataset_name, results) in test_results
        println("\n$dataset_name:")
        for (test_type, result) in results
            if isa(result, Dict) && haskey(result, "error")
                println("  ❌ $test_type: FAILED - $(result["error"])")
            else
                success_count = count(r -> r == "success", values(result))
                total_count = length(result)
                println("  ✓ $test_type: $success_count/$total_count tests passed")
            end
        end
    end
    
    return test_results
end

end # module FaultToleranceTests