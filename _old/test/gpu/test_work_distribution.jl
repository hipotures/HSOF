using Test
using CUDA
using Dates

# Include the GPU module first
include("../../src/gpu/GPU.jl")
using .GPU

# Include work distribution module
include("../../src/gpu/work_distribution.jl")
using .WorkDistribution

@testset "Work Distribution Tests" begin
    
    @testset "WorkDistributor Creation" begin
        # Test single GPU setup
        distributor_single = WorkDistributor(total_trees=100)
        @test distributor_single.total_trees == 100
        @test distributor_single.rebalance_threshold == 0.05
        
        # Test custom parameters
        distributor_custom = WorkDistributor(
            total_trees=50,
            rebalance_threshold=0.1
        )
        @test distributor_custom.total_trees == 50
        @test distributor_custom.rebalance_threshold == 0.1
    end
    
    @testset "GPU Work Assignment" begin
        distributor = create_work_distributor(total_trees=100)
        
        # Test tree assignment
        @test get_gpu_for_tree(distributor, 1) == 0
        @test get_gpu_for_tree(distributor, 25) == 0
        
        # For multi-GPU setup (if available)
        if length(distributor.gpu_assignments) > 1
            @test get_gpu_for_tree(distributor, 51) == 1
            @test get_gpu_for_tree(distributor, 100) == 1
        end
        
        # Test tree range retrieval
        range_gpu0 = get_tree_range(distributor, 0)
        @test first(range_gpu0) == 1
        if length(distributor.gpu_assignments) == 1
            @test last(range_gpu0) == 100
        else
            @test last(range_gpu0) == 50
        end
    end
    
    @testset "Batch Tree Assignment" begin
        distributor = create_work_distributor(total_trees=100)
        
        # Assign batch of trees
        tree_indices = collect(1:20)
        assignments = assign_tree_work(distributor, tree_indices)
        
        # All trees should go to GPU 0
        @test haskey(assignments, 0)
        @test assignments[0] == tree_indices
        
        # Test mixed assignment for multi-GPU
        if length(distributor.gpu_assignments) > 1
            mixed_trees = [10, 20, 60, 70]
            mixed_assignments = assign_tree_work(distributor, mixed_trees)
            @test mixed_assignments[0] == [10, 20]
            @test mixed_assignments[1] == [60, 70]
        end
    end
    
    @testset "Metamodel Work Assignment" begin
        distributor = create_work_distributor()
        
        # Test metamodel assignment
        training_gpu = assign_metamodel_work(distributor, :training)
        inference_gpu = assign_metamodel_work(distributor, :inference)
        
        if length(distributor.gpu_assignments) == 1
            # Single GPU handles both
            @test training_gpu == 0
            @test inference_gpu == 0
        else
            # Dual GPU setup
            @test training_gpu == 0
            @test inference_gpu == 1
        end
    end
    
    @testset "Workload Metrics" begin
        distributor = create_work_distributor()
        
        # Update metrics
        update_metrics!(distributor, 0, 
            trees_processed=10,
            metamodel_operations=5,
            time_ms=1000.0,
            utilization=75.0,
            memory_mb=2048.0
        )
        
        metrics = distributor.workload_metrics[0]
        @test metrics.trees_processed == 10
        @test metrics.metamodel_operations == 5
        @test metrics.total_time_ms == 1000.0
        @test metrics.utilization_percent == 75.0
        @test metrics.memory_used_mb == 2048.0
        
        # Test cumulative updates
        update_metrics!(distributor, 0, trees_processed=5)
        @test distributor.workload_metrics[0].trees_processed == 15
    end
    
    @testset "Load Balance Calculation" begin
        distributor = create_work_distributor()
        
        # Initial balance should be perfect
        @test get_load_balance_ratio(distributor) == 1.0
        
        # Update metrics to create imbalance
        update_metrics!(distributor, 0, trees_processed=100)
        
        if length(distributor.gpu_assignments) > 1
            update_metrics!(distributor, 1, trees_processed=80)
            
            # Balance ratio should reflect the imbalance
            ratio = get_load_balance_ratio(distributor)
            @test ratio ≈ 0.8 atol=0.01
        end
    end
    
    @testset "Rebalancing Logic" begin
        distributor = WorkDistributor(
            total_trees=100,
            rebalance_threshold=0.2  # 20% threshold for testing
        )
        
        if length(distributor.gpu_assignments) > 1
            # Create significant imbalance
            update_metrics!(distributor, 0, 
                trees_processed=100,
                total_time_ms=1000.0
            )
            update_metrics!(distributor, 1, 
                trees_processed=50,
                total_time_ms=1000.0
            )
            
            # Should trigger rebalancing
            needs_rebalance = rebalance_if_needed!(distributor)
            @test needs_rebalance == true
            
            # Check that trees were redistributed
            # GPU 0 should have more trees since it's faster
            range0 = get_tree_range(distributor, 0)
            range1 = get_tree_range(distributor, 1)
            @test length(range0) > length(range1)
        end
    end
    
    @testset "Work Summary" begin
        distributor = create_work_distributor(total_trees=100)
        
        # Update some metrics
        update_metrics!(distributor, 0,
            trees_processed=25,
            metamodel_operations=10,
            utilization=85.0,
            memory_mb=3072.0
        )
        
        summary = get_work_summary(distributor)
        
        @test summary["total_trees"] == 100
        @test summary["num_gpus"] == length(distributor.gpu_assignments)
        @test haskey(summary["assignments"], 0)
        
        gpu0_summary = summary["assignments"][0]
        @test gpu0_summary["trees_processed"] == 25
        @test gpu0_summary["metamodel_ops"] == 10
        @test gpu0_summary["utilization"] == 85.0
        @test gpu0_summary["memory_mb"] == 3072.0
    end
    
    @testset "GPU Affinity" begin
        if CUDA.functional()
            # Test setting GPU affinity
            original_device = CUDA.device()
            
            set_gpu_affinity(0)
            @test CUDA.device() == collect(CUDA.devices())[1]
            
            # Restore original device
            CUDA.device!(original_device)
        else
            @test_skip "CUDA not functional"
        end
    end
    
    @testset "Edge Cases" begin
        # Test with odd number of trees
        distributor_odd = WorkDistributor(total_trees=99)
        
        if length(distributor_odd.gpu_assignments) > 1
            range0 = get_tree_range(distributor_odd, 0)
            range1 = get_tree_range(distributor_odd, 1)
            
            # Trees should be distributed as evenly as possible
            @test abs(length(range0) - length(range1)) <= 1
            @test length(range0) + length(range1) == 99
        end
        
        # Test empty tree assignment
        empty_assignments = assign_tree_work(distributor_odd, Int[])
        @test isempty(empty_assignments)
        
        # Test invalid tree index
        @test get_gpu_for_tree(distributor_odd, 150) == 0  # Defaults to GPU 0
    end
    
    @testset "Execute on GPU" begin
        # Test work execution wrapper
        function dummy_work(x)
            return x * 2
        end
        
        result = execute_on_gpu(0, dummy_work, 21)
        @test result == 42
        
        # Test with GPU memory allocation
        if CUDA.functional()
            function gpu_work()
                arr = CUDA.zeros(100)
                return length(arr)
            end
            
            result = execute_on_gpu(0, gpu_work)
            @test result == 100
        end
    end

end

# Print summary
println("\nWork Distribution Test Summary:")
println("==============================")
if CUDA.functional()
    println("✓ CUDA functional - GPU tests executed")
    println("  GPUs detected: $(length(CUDA.devices()))")
else
    println("⚠ CUDA not functional - GPU tests skipped")
end
println("\nAll work distribution tests completed!")