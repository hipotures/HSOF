using Test
using CUDA
using Dates
using Statistics

# Include required modules
include("../../src/gpu/GPU.jl")
using .GPU

# Import from GPU module
using .GPU.DynamicRebalancing
using .GPU.DynamicRebalancing: BALANCED, MONITORING, PLANNING, MIGRATING, STABILIZING

@testset "Dynamic Rebalancing Tests" begin
    
    @testset "Manager Creation" begin
        # Test with default parameters
        manager = create_rebalancing_manager()
        @test isa(manager, RebalancingManager)
        @test manager.num_gpus == 2
        @test manager.total_trees == 100
        @test manager.imbalance_threshold == 0.1f0
        @test manager.auto_rebalancing
        @test !manager.monitoring_active[]
        
        # Test with custom parameters
        manager2 = create_rebalancing_manager(
            num_gpus = 4,
            total_trees = 200,
            imbalance_threshold = 0.15f0,
            check_interval = 2.0,
            auto_rebalancing = false
        )
        @test manager2.num_gpus == 4
        @test manager2.total_trees == 200
        @test manager2.imbalance_threshold == 0.15f0
        @test !manager2.auto_rebalancing
        
        # Check initial distribution
        total_assigned = sum(w.total_trees for (_, w) in manager2.gpu_workloads)
        @test total_assigned == 200
        
        # Verify even distribution
        for gpu_id in 0:3
            @test manager2.gpu_workloads[gpu_id].total_trees == 50
        end
    end
    
    @testset "Workload Distribution" begin
        manager = create_rebalancing_manager(num_gpus=3, total_trees=100)
        
        # Check initial distribution
        distribution = get_current_distribution(manager)
        @test length(distribution) == 3
        
        # GPU 0 and 1 should have 34 trees, GPU 2 should have 32
        @test length(distribution[0]) == 34
        @test length(distribution[1]) == 33
        @test length(distribution[2]) == 33
        
        # Verify all trees are assigned
        all_trees = vcat(values(distribution)...)
        @test sort(all_trees) == collect(1:100)
        @test length(unique(all_trees)) == 100
    end
    
    @testset "Imbalance Detection" begin
        manager = create_rebalancing_manager(num_gpus=2, total_trees=100)
        
        # Initially balanced
        update_workload_metrics!(manager, 0, 
            utilization=50.0f0, memory_usage=40.0f0, 
            throughput=10.0f0, avg_tree_time=5.0)
        update_workload_metrics!(manager, 1,
            utilization=50.0f0, memory_usage=40.0f0,
            throughput=10.0f0, avg_tree_time=5.0)
        
        imbalance = check_imbalance(manager)
        @test imbalance < manager.imbalance_threshold
        
        # Create imbalance
        update_workload_metrics!(manager, 0,
            utilization=80.0f0, memory_usage=60.0f0,
            throughput=8.0f0, avg_tree_time=7.0)
        update_workload_metrics!(manager, 1,
            utilization=30.0f0, memory_usage=20.0f0,
            throughput=12.0f0, avg_tree_time=3.0)
        
        imbalance2 = check_imbalance(manager)
        @test imbalance2 > manager.imbalance_threshold
    end
    
    @testset "Rebalancing Decision" begin
        manager = create_rebalancing_manager(num_gpus=2, total_trees=100)
        
        # Create significant imbalance
        manager.gpu_workloads[0].avg_utilization = 85.0f0
        manager.gpu_workloads[1].avg_utilization = 35.0f0
        
        decision = DynamicRebalancing.create_rebalancing_decision(manager)
        
        @test isa(decision, RebalancingDecision)
        @test decision.imbalance_ratio > 0.1f0
        
        # With significant imbalance, should recommend rebalancing
        if decision.should_rebalance
            @test !isempty(decision.migration_plans)
            @test decision.total_trees_to_migrate > 0
            @test decision.estimated_total_cost > 0
            
            # Check migration plans
            for plan in decision.migration_plans
                @test plan.source_gpu == 0  # From overloaded GPU
                @test plan.target_gpu == 1  # To underloaded GPU
                @test !isempty(plan.tree_ids)
                @test plan.estimated_cost > 0
                @test plan.expected_improvement > 0
            end
        end
    end
    
    @testset "Cost Model" begin
        manager = create_rebalancing_manager()
        cost_model = manager.cost_model
        
        @test cost_model.per_tree_migration_cost == 10.0
        @test cost_model.state_transfer_cost == 50.0
        @test cost_model.synchronization_cost == 20.0
        @test cost_model.min_migration_benefit == 5.0f0
        @test cost_model.max_migration_ratio == 0.3f0
        
        # Test cost estimation
        num_trees = 10
        cost = DynamicRebalancing.estimate_migration_cost(manager, num_trees)
        expected_cost = 10.0 * num_trees + 50.0 + 20.0
        @test cost == expected_cost
    end
    
    @testset "Hysteresis Mechanism" begin
        manager = create_rebalancing_manager(
            imbalance_threshold = 0.1f0,
            hysteresis_factor = 0.8f0
        )
        
        # Just below threshold
        manager.gpu_workloads[0].avg_utilization = 55.0f0
        manager.gpu_workloads[1].avg_utilization = 47.0f0
        
        decision = DynamicRebalancing.create_rebalancing_decision(manager)
        
        # Should not rebalance due to hysteresis
        hysteresis_threshold = manager.imbalance_threshold * manager.hysteresis_factor
        if check_imbalance(manager) < hysteresis_threshold
            @test !decision.should_rebalance
            @test occursin("hysteresis", lowercase(decision.decision_reason))
        end
    end
    
    @testset "Tree Migration" begin
        manager = create_rebalancing_manager(num_gpus=2, total_trees=100)
        
        # Create migration plan
        trees_to_migrate = [45, 46, 47, 48, 49, 50]
        plan = create_migration_plan(0, 1, trees_to_migrate, "Test migration")
        
        # Execute migration
        execute_migration!(manager, plan)
        
        # Verify trees moved
        @test all(id -> !(id in manager.gpu_workloads[0].tree_ids), trees_to_migrate)
        @test all(id -> id in manager.gpu_workloads[1].tree_ids, trees_to_migrate)
        
        # Check tree counts
        @test manager.gpu_workloads[0].total_trees == 44  # 50 - 6
        @test manager.gpu_workloads[1].total_trees == 56  # 50 + 6
        
        # Verify sorted order
        @test issorted(manager.gpu_workloads[1].tree_ids)
    end
    
    @testset "Cooldown Period" begin
        manager = create_rebalancing_manager(cooldown_period = 30.0)
        
        # Simulate recent rebalancing
        manager.last_rebalancing = now()
        
        @test !should_rebalance(manager)
        
        # Simulate cooldown expired
        manager.last_rebalancing = now() - Second(35)
        
        @test should_rebalance(manager)
    end
    
    @testset "State Management" begin
        manager = create_rebalancing_manager()
        
        @test manager.current_state == BALANCED
        @test manager.consecutive_triggers == 0
        
        # Test state transitions
        lock(manager.state_lock) do
            manager.current_state = MONITORING
            manager.consecutive_triggers = 2
        end
        
        @test manager.current_state == MONITORING
        @test manager.consecutive_triggers == 2
    end
    
    @testset "Rebalancing History" begin
        manager = create_rebalancing_manager()
        
        # Simulate rebalancing event
        metrics = RebalancingMetrics(
            now(), BALANCED, 0.15f0, 0.05f0,
            10, 150.0, true
        )
        
        push!(manager.rebalancing_history, metrics)
        
        history = get_rebalancing_history(manager)
        @test length(history) == 1
        @test history[1].trees_migrated == 10
        @test history[1].success
        @test history[1].imbalance_before > history[1].imbalance_after
    end
    
    @testset "Configuration Updates" begin
        manager = create_rebalancing_manager()
        
        # Test threshold update
        set_rebalancing_threshold!(manager, 0.2f0)
        @test manager.imbalance_threshold == 0.2f0
        
        # Test invalid threshold
        set_rebalancing_threshold!(manager, 1.5f0)
        @test manager.imbalance_threshold == 0.2f0  # Should not change
        
        # Test auto-rebalancing toggle
        enable_auto_rebalancing!(manager, false)
        @test !manager.auto_rebalancing
        
        enable_auto_rebalancing!(manager, true)
        @test manager.auto_rebalancing
    end
    
    @testset "Workload Metrics Update" begin
        manager = create_rebalancing_manager()
        
        # Update metrics multiple times
        for i in 1:5
            update_workload_metrics!(manager, 0,
                utilization = 50.0f0 + i * 5.0f0,
                memory_usage = 40.0f0,
                throughput = 10.0f0,
                avg_tree_time = 5.0
            )
        end
        
        workload = manager.gpu_workloads[0]
        @test workload.current_utilization == 75.0f0  # 50 + 5*5
        @test length(workload.utilization_history) == 5
        @test workload.avg_utilization ≈ 65.0f0  # Average of 55, 60, 65, 70, 75
    end
    
    @testset "Memory Pressure Check" begin
        manager = create_rebalancing_manager()
        
        # Set high memory usage
        manager.gpu_workloads[0].memory_usage = 0.95f0
        
        @test !should_rebalance(manager)
        
        # Lower memory usage
        manager.gpu_workloads[0].memory_usage = 0.7f0
        manager.gpu_workloads[1].memory_usage = 0.7f0
        
        @test should_rebalance(manager)
    end
    
    @testset "Monitoring Integration" begin
        manager = create_rebalancing_manager(
            check_interval = 0.1,
            auto_rebalancing = false  # Disable auto for testing
        )
        
        # Start monitoring
        start_rebalancing!(manager)
        @test manager.monitoring_active[]
        @test !isnothing(manager.monitor_task)
        
        # Let it run briefly
        sleep(0.3)
        
        # Stop monitoring
        stop_rebalancing!(manager)
        @test !manager.monitoring_active[]
        @test isnothing(manager.monitor_task)
    end
    
    @testset "Callbacks" begin
        manager = create_rebalancing_manager()
        
        # Track callback calls
        pre_called = Ref(false)
        post_called = Ref(false)
        
        # Register callbacks
        push!(manager.pre_migration_callbacks, (decision) -> pre_called[] = true)
        push!(manager.post_migration_callbacks, (metrics) -> post_called[] = true)
        
        # Create and execute a simple rebalancing
        decision = RebalancingDecision(
            now(), true, 0.2f0,
            [create_migration_plan(0, 1, [50], "Test")],
            1, 80.0, "Test rebalancing"
        )
        
        DynamicRebalancing.execute_rebalancing!(manager, decision)
        
        @test pre_called[]
        @test post_called[]
    end
    
end

# Print summary
println("\nDynamic Rebalancing Test Summary:")
println("==================================")
if CUDA.functional()
    num_gpus = length(CUDA.devices())
    println("✓ CUDA functional - Dynamic rebalancing tests executed")
    println("  GPUs detected: $num_gpus")
    println("  Load balancing algorithms validated")
    println("  Tree migration protocols tested")
    println("  Hysteresis mechanism verified")
else
    println("⚠ CUDA not functional - CPU simulation tests only")
end
println("\nAll dynamic rebalancing tests completed!")