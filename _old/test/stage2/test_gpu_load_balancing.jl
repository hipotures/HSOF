"""
Test Suite for Dynamic Load Balancing Across GPUs
Validates GPU utilization monitoring, work stealing algorithms, tree migration protocols,
load prediction models, and adaptive batch sizing for dual RTX 4090 configuration.
"""

using Test
using Random
using Statistics
using Dates

# Include the GPU load balancing module
include("../../src/stage2/gpu_load_balancing.jl")
using .GPULoadBalancing

@testset "Dynamic Load Balancing Across GPUs Tests" begin
    
    Random.seed!(42)  # For reproducible tests
    
    @testset "GPU Device Creation Tests" begin
        # Test default RTX 4090 device
        device = create_gpu_device(0)
        
        @test device.device_id == 0
        @test device.device_name == "RTX 4090"
        @test device.total_memory == 24 * 1024^3  # 24GB
        @test device.compute_capability == (8, 9)
        @test device.multiprocessor_count == 128
        @test device.max_threads_per_block == 1024
        @test device.memory_bandwidth == 1008.0
        @test device.is_available == true
        
        # Test custom device
        custom_device = create_gpu_device(
            1, 
            device_name = "Custom GPU",
            total_memory = 16 * 1024^3,
            compute_capability = (7, 5)
        )
        
        @test custom_device.device_id == 1
        @test custom_device.device_name == "Custom GPU"
        @test custom_device.total_memory == 16 * 1024^3
        @test custom_device.compute_capability == (7, 5)
        
        println("  ✅ GPU device creation tests passed")
    end
    
    @testset "GPU Metrics Creation Tests" begin
        metrics = create_gpu_metrics(0)
        
        @test metrics.device_id == 0
        @test metrics.utilization_gpu == 0.0f0
        @test metrics.utilization_memory == 0.0f0
        @test metrics.memory_used == 0
        @test metrics.memory_free == 0
        @test metrics.temperature == 0.0f0
        @test metrics.power_draw == 0.0f0
        @test metrics.fan_speed == 0.0f0
        @test metrics.sm_clock_mhz == 0.0f0
        @test metrics.memory_clock_mhz == 0.0f0
        @test metrics.throughput_samples_per_sec == 0.0
        @test metrics.active_trees == 0
        @test metrics.pending_operations == 0
        @test metrics.completed_operations == 0
        @test metrics.update_count == 0
        @test metrics.average_update_interval == 0.0
        
        println("  ✅ GPU metrics creation tests passed")
    end
    
    @testset "Tree Workload Creation Tests" begin
        workload = GPULoadBalancing.create_tree_workload(42)
        
        @test workload.tree_id == 42
        @test workload.estimated_compute_time == 0.0
        @test workload.memory_requirement == 0
        @test workload.priority == 1.0f0
        @test workload.complexity_score == 1.0f0
        @test workload.last_update == workload.creation_time
        @test workload.migration_count == 0
        @test workload.is_migrating == false
        @test workload.target_gpu == nothing
        @test workload.migration_progress == 0.0f0
        @test workload.avg_execution_time == 0.0
        @test workload.total_execution_time == 0.0
        @test workload.execution_count == 0
        
        println("  ✅ Tree workload creation tests passed")
    end
    
    @testset "Load Balancing Configuration Tests" begin
        # Test default configuration
        config = GPULoadBalancing.create_load_balancing_config()
        
        @test config.monitoring_interval_ms == 100
        @test config.load_imbalance_threshold == 0.3f0
        @test config.work_stealing_enabled == true
        @test config.work_stealing_strategy == GPULoadBalancing.GREEDY_STEALING
        @test config.migration_safety_level == GPULoadBalancing.SAFE_MIGRATION
        @test config.max_trees_per_gpu == 50
        @test config.min_trees_per_gpu == 5
        @test config.enable_load_prediction == true
        @test config.prediction_model == GPULoadBalancing.LINEAR_PREDICTION
        @test config.adaptive_batch_sizing == true
        @test config.memory_usage_threshold == 0.85f0
        @test config.temperature_threshold == 85.0f0
        @test config.enable_fault_tolerance == true
        
        # Test custom configuration
        custom_config = GPULoadBalancing.create_load_balancing_config(
            monitoring_interval_ms = 50,
            load_imbalance_threshold = 0.2f0,
            work_stealing_strategy = GPULoadBalancing.ROUND_ROBIN_STEALING,
            migration_safety_level = GPULoadBalancing.IMMEDIATE_MIGRATION
        )
        
        @test custom_config.monitoring_interval_ms == 50
        @test custom_config.load_imbalance_threshold == 0.2f0
        @test custom_config.work_stealing_strategy == GPULoadBalancing.ROUND_ROBIN_STEALING
        @test custom_config.migration_safety_level == GPULoadBalancing.IMMEDIATE_MIGRATION
        
        println("  ✅ Load balancing configuration tests passed")
    end
    
    @testset "Load Balancer Initialization Tests" begin
        config = GPULoadBalancing.create_load_balancing_config(
            target_gpu_count = 2,
            max_trees_per_gpu = 10
        )
        balancer = GPULoadBalancing.initialize_gpu_load_balancer(config)
        
        @test balancer.config == config
        @test length(balancer.gpu_devices) == 2
        @test length(balancer.gpu_metrics) == 2
        @test length(balancer.gpu_workloads) == 2
        @test length(balancer.metrics_history) == 2
        @test isempty(balancer.current_assignments)
        @test balancer.total_trees == 0
        @test balancer.balancer_state == "active"
        @test balancer.stats.total_trees_processed == 0
        @test balancer.stats.total_migrations == 0
        @test balancer.stats.work_stealing_events == 0
        @test balancer.stats.load_balancing_rounds == 0
        
        # Check individual GPU setup
        for device_id in 0:1
            @test haskey(balancer.gpu_devices, device_id)
            @test haskey(balancer.gpu_metrics, device_id)
            @test haskey(balancer.gpu_workloads, device_id)
            @test haskey(balancer.metrics_history, device_id)
            @test isempty(balancer.metrics_history[device_id])
            @test isempty(balancer.gpu_workloads[device_id])
        end
        
        println("  ✅ Load balancer initialization tests passed")
    end
    
    @testset "Tree Assignment Tests" begin
        config = GPULoadBalancing.create_load_balancing_config(target_gpu_count = 2)
        balancer = GPULoadBalancing.initialize_gpu_load_balancer(config)
        
        # Test initial assignment
        success = GPULoadBalancing.assign_tree_to_gpu!(balancer, 1, 0)
        @test success == true
        @test haskey(balancer.current_assignments, 1)
        @test balancer.current_assignments[1] == 0
        @test length(balancer.gpu_workloads[0]) == 1
        @test 1 in keys(balancer.gpu_workloads[0])
        @test balancer.total_trees == 1
        
        # Test assignment to second GPU
        success = GPULoadBalancing.assign_tree_to_gpu!(balancer, 2, 1)
        @test success == true
        @test balancer.current_assignments[2] == 1
        @test length(balancer.gpu_workloads[1]) == 1
        @test balancer.total_trees == 2
        
        # Test duplicate assignment (should fail)
        success = GPULoadBalancing.assign_tree_to_gpu!(balancer, 1, 1)
        @test success == false
        @test balancer.current_assignments[1] == 0  # Should remain on original GPU
        
        # Test assignment to invalid GPU
        success = GPULoadBalancing.assign_tree_to_gpu!(balancer, 3, 5)
        @test success == false
        
        println("  ✅ Tree assignment tests passed")
    end
    
    @testset "Metrics Update Tests" begin
        config = GPULoadBalancing.create_load_balancing_config(target_gpu_count = 2)
        balancer = GPULoadBalancing.initialize_gpu_load_balancer(config)
        
        # Add some trees for realistic metrics
        GPULoadBalancing.assign_tree_to_gpu!(balancer, 1, 0)
        GPULoadBalancing.assign_tree_to_gpu!(balancer, 2, 0)
        GPULoadBalancing.assign_tree_to_gpu!(balancer, 3, 1)
        
        # Update metrics
        initial_update_count = balancer.gpu_metrics[0].update_count
        GPULoadBalancing.update_gpu_metrics!(balancer)
        
        # Check that metrics were updated
        @test balancer.gpu_metrics[0].update_count == initial_update_count + 1
        @test balancer.gpu_metrics[1].update_count == initial_update_count + 1
        @test balancer.stats.total_monitoring_updates == 1
        
        # Check that workload affects simulated metrics
        @test balancer.gpu_metrics[0].active_trees == 2
        @test balancer.gpu_metrics[1].active_trees == 1
        @test balancer.gpu_metrics[0].utilization_gpu > balancer.gpu_metrics[1].utilization_gpu
        
        # Check metrics history
        @test length(balancer.metrics_history[0]) == 1
        @test length(balancer.metrics_history[1]) == 1
        
        println("  ✅ Metrics update tests passed")
    end
    
    @testset "Load Imbalance Detection Tests" begin
        config = GPULoadBalancing.create_load_balancing_config(
            target_gpu_count = 2,
            load_imbalance_threshold = 0.3f0
        )
        balancer = GPULoadBalancing.initialize_gpu_load_balancer(config)
        
        # Create balanced load
        GPULoadBalancing.assign_tree_to_gpu!(balancer, 1, 0)
        GPULoadBalancing.assign_tree_to_gpu!(balancer, 2, 1)
        GPULoadBalancing.update_gpu_metrics!(balancer)
        
        imbalanced = GPULoadBalancing.detect_load_imbalance(balancer)
        @test imbalanced == false  # Should be balanced
        
        # Create imbalanced load
        for i in 3:8
            GPULoadBalancing.assign_tree_to_gpu!(balancer, i, 0)
        end
        GPULoadBalancing.update_gpu_metrics!(balancer)
        
        imbalanced = GPULoadBalancing.detect_load_imbalance(balancer)
        @test imbalanced == true  # Should be imbalanced now
        
        println("  ✅ Load imbalance detection tests passed")
    end
    
    @testset "Work Stealing Selection Tests" begin
        config = GPULoadBalancing.create_load_balancing_config(target_gpu_count = 2)
        balancer = GPULoadBalancing.initialize_gpu_load_balancer(config)
        
        # Create imbalanced scenario
        for i in 1:6
            GPULoadBalancing.assign_tree_to_gpu!(balancer, i, 0)
        end
        GPULoadBalancing.assign_tree_to_gpu!(balancer, 7, 1)
        GPULoadBalancing.update_gpu_metrics!(balancer)
        
        # Test GPU selection for work stealing
        source_gpu, target_gpu = GPULoadBalancing.select_gpus_for_stealing(balancer)
        @test !isnothing(source_gpu)
        @test !isnothing(target_gpu)
        @test source_gpu == 0  # Overloaded GPU
        @test target_gpu == 1  # Underloaded GPU
        
        # Test tree selection for stealing
        trees_to_steal = GPULoadBalancing.select_trees_for_stealing(balancer, source_gpu, target_gpu)
        @test !isempty(trees_to_steal)
        @test length(trees_to_steal) <= 3  # Should not steal too many at once
        
        # All selected trees should be from source GPU
        for tree_id in trees_to_steal
            @test balancer.current_assignments[tree_id] == source_gpu
        end
        
        println("  ✅ Work stealing selection tests passed")
    end
    
    @testset "Tree Migration Tests" begin
        config = GPULoadBalancing.create_load_balancing_config(target_gpu_count = 2)
        balancer = GPULoadBalancing.initialize_gpu_load_balancer(config)
        
        # Set up initial assignment
        GPULoadBalancing.assign_tree_to_gpu!(balancer, 1, 0)
        @test balancer.current_assignments[1] == 0
        @test 1 in keys(balancer.gpu_workloads[0])
        @test 1 ∉ keys(balancer.gpu_workloads[1])
        
        # Test migration capability check
        can_migrate = GPULoadBalancing.can_migrate_tree(balancer, 1)
        @test can_migrate == true
        
        # Perform migration
        success = GPULoadBalancing.migrate_tree!(balancer, 1, 0, 1)
        @test success == true
        @test balancer.current_assignments[1] == 1
        @test 1 ∉ keys(balancer.gpu_workloads[0])
        @test 1 in keys(balancer.gpu_workloads[1])
        @test balancer.stats.total_migrations == 1
        
        # Check workload migration tracking
        workload = balancer.gpu_workloads[1][1]
        @test workload.migration_count == 1
        
        # Test migration of non-existent tree
        success = GPULoadBalancing.migrate_tree!(balancer, 999, 0, 1)
        @test success == false
        
        println("  ✅ Tree migration tests passed")
    end
    
    @testset "Work Stealing Execution Tests" begin
        config = GPULoadBalancing.create_load_balancing_config(
            target_gpu_count = 2,
            work_stealing_enabled = true
        )
        balancer = GPULoadBalancing.initialize_gpu_load_balancer(config)
        
        # Create significant imbalance
        for i in 1:8
            GPULoadBalancing.assign_tree_to_gpu!(balancer, i, 0)
        end
        GPULoadBalancing.update_gpu_metrics!(balancer)
        
        initial_stealing_events = balancer.stats.work_stealing_events
        initial_gpu0_trees = length(balancer.gpu_workloads[0])
        initial_gpu1_trees = length(balancer.gpu_workloads[1])
        
        # Perform work stealing
        GPULoadBalancing.perform_work_stealing!(balancer)
        
        # Check that work stealing occurred
        final_gpu0_trees = length(balancer.gpu_workloads[0])
        final_gpu1_trees = length(balancer.gpu_workloads[1])
        
        @test final_gpu0_trees < initial_gpu0_trees  # Source GPU should have fewer trees
        @test final_gpu1_trees > initial_gpu1_trees  # Target GPU should have more trees
        @test balancer.stats.work_stealing_events > initial_stealing_events
        @test final_gpu0_trees + final_gpu1_trees == 8  # Total trees preserved
        
        println("  ✅ Work stealing execution tests passed")
    end
    
    @testset "Load Prediction Tests" begin
        config = GPULoadBalancing.create_load_balancing_config(
            target_gpu_count = 2,
            enable_load_prediction = true,
            prediction_model = GPULoadBalancing.LINEAR_PREDICTION
        )
        balancer = GPULoadBalancing.initialize_gpu_load_balancer(config)
        
        # Add trees and update metrics to build history
        for i in 1:4
            GPULoadBalancing.assign_tree_to_gpu!(balancer, i, 0)
        end
        
        for round in 1:5
            GPULoadBalancing.update_gpu_metrics!(balancer)
            sleep(0.01)  # Small delay to differentiate timestamps
        end
        
        # Test load prediction
        predicted_load = GPULoadBalancing.predict_gpu_load(balancer, 0, 1.0)  # Predict 1 second ahead
        @test predicted_load >= 0.0
        @test predicted_load <= 1.0
        
        # Test prediction with different models
        balancer.config = GPULoadBalancing.create_load_balancing_config(
            target_gpu_count = 2,
            prediction_model = GPULoadBalancing.EXPONENTIAL_PREDICTION
        )
        
        predicted_load_exp = GPULoadBalancing.predict_gpu_load(balancer, 0, 1.0)
        @test predicted_load_exp >= 0.0
        @test predicted_load_exp <= 1.0
        
        println("  ✅ Load prediction tests passed")
    end
    
    @testset "Adaptive Batch Sizing Tests" begin
        config = GPULoadBalancing.create_load_balancing_config(
            target_gpu_count = 2,
            adaptive_batch_sizing = true,
            memory_usage_threshold = 0.8f0
        )
        balancer = GPULoadBalancing.initialize_gpu_load_balancer(config)
        
        # Add trees and simulate different GPU states
        for i in 1:6
            GPULoadBalancing.assign_tree_to_gpu!(balancer, i, 0)
        end
        for i in 7:8
            GPULoadBalancing.assign_tree_to_gpu!(balancer, i, 1)
        end
        
        GPULoadBalancing.update_gpu_metrics!(balancer)
        
        # Test batch size calculation for different GPU states
        batch_size_gpu0 = GPULoadBalancing.calculate_adaptive_batch_size(balancer, 0)
        batch_size_gpu1 = GPULoadBalancing.calculate_adaptive_batch_size(balancer, 1)
        
        @test batch_size_gpu0 > 0
        @test batch_size_gpu1 > 0
        @test batch_size_gpu0 <= batch_size_gpu1  # More loaded GPU should have smaller batch size
        
        # Test with high memory usage (should reduce batch size)
        balancer.gpu_metrics[0].utilization_memory = 0.9f0  # High memory usage
        high_mem_batch_size = GPULoadBalancing.calculate_adaptive_batch_size(balancer, 0)
        @test high_mem_batch_size <= batch_size_gpu0
        
        println("  ✅ Adaptive batch sizing tests passed")
    end
    
    @testset "Full Load Balancing Cycle Tests" begin
        config = GPULoadBalancing.create_load_balancing_config(
            target_gpu_count = 2,
            monitoring_interval_ms = 10,  # Fast for testing
            load_imbalance_threshold = 0.2f0
        )
        balancer = GPULoadBalancing.initialize_gpu_load_balancer(config)
        
        # Create initial imbalanced load
        for i in 1:10
            GPULoadBalancing.assign_tree_to_gpu!(balancer, i, 0)
        end
        for i in 11:12
            GPULoadBalancing.assign_tree_to_gpu!(balancer, i, 1)
        end
        
        initial_imbalance = abs(length(balancer.gpu_workloads[0]) - length(balancer.gpu_workloads[1]))
        
        # Run load balancing cycle
        GPULoadBalancing.run_load_balancing_cycle!(balancer)
        
        final_imbalance = abs(length(balancer.gpu_workloads[0]) - length(balancer.gpu_workloads[1]))
        
        # Check that load balancing was effective
        @test final_imbalance < initial_imbalance
        @test balancer.stats.load_balancing_rounds == 1
        @test balancer.stats.total_monitoring_updates > 0
        
        # Check that all trees are still tracked
        total_assigned = sum(length(workloads) for workloads in values(balancer.gpu_workloads))
        @test total_assigned == 12
        
        println("  ✅ Full load balancing cycle tests passed")
    end
    
    @testset "Status and Monitoring Tests" begin
        config = GPULoadBalancing.create_load_balancing_config(target_gpu_count = 2)
        balancer = GPULoadBalancing.initialize_gpu_load_balancer(config)
        
        # Initial status
        status = GPULoadBalancing.get_load_balancer_status(balancer)
        @test haskey(status, "balancer_state")
        @test haskey(status, "total_trees")
        @test haskey(status, "gpu_count")
        @test haskey(status, "load_balancing_rounds")
        @test haskey(status, "total_migrations")
        @test haskey(status, "work_stealing_events")
        
        @test status["balancer_state"] == "active"
        @test status["total_trees"] == 0
        @test status["gpu_count"] == 2
        @test status["load_balancing_rounds"] == 0
        
        # Add some activity
        for i in 1:5
            GPULoadBalancing.assign_tree_to_gpu!(balancer, i, 0)
        end
        GPULoadBalancing.run_load_balancing_cycle!(balancer)
        
        updated_status = GPULoadBalancing.get_load_balancer_status(balancer)
        @test updated_status["total_trees"] == 5
        @test updated_status["load_balancing_rounds"] == 1
        
        println("  ✅ Status and monitoring tests passed")
    end
    
    @testset "Cleanup and Reset Tests" begin
        config = GPULoadBalancing.create_load_balancing_config(target_gpu_count = 2)
        balancer = GPULoadBalancing.initialize_gpu_load_balancer(config)
        
        # Add some data
        for i in 1:5
            GPULoadBalancing.assign_tree_to_gpu!(balancer, i, rand(0:1))
        end
        GPULoadBalancing.run_load_balancing_cycle!(balancer)
        
        @test !isempty(balancer.current_assignments)
        @test balancer.stats.load_balancing_rounds > 0
        
        # Test cleanup
        GPULoadBalancing.cleanup_load_balancer!(balancer)
        
        @test balancer.balancer_state == "shutdown"
        @test isempty(balancer.current_assignments)
        @test balancer.total_trees == 0
        @test all(isempty(workloads) for workloads in values(balancer.gpu_workloads))
        
        println("  ✅ Cleanup and reset tests passed")
    end
    
    @testset "Error Handling Tests" begin
        config = GPULoadBalancing.create_load_balancing_config(target_gpu_count = 2)
        balancer = GPULoadBalancing.initialize_gpu_load_balancer(config)
        
        # Test assignment to invalid GPU
        success = GPULoadBalancing.assign_tree_to_gpu!(balancer, 1, 10)
        @test success == false
        
        # Test migration of non-existent tree
        success = GPULoadBalancing.migrate_tree!(balancer, 999, 0, 1)
        @test success == false
        
        # Test prediction on GPU with no history
        predicted = GPULoadBalancing.predict_gpu_load(balancer, 0, 1.0)
        @test predicted >= 0.0  # Should return reasonable default
        
        # Test batch size calculation on invalid GPU
        batch_size = GPULoadBalancing.calculate_adaptive_batch_size(balancer, 10)
        @test batch_size == 1  # Should return default
        
        println("  ✅ Error handling tests passed")
    end
end

println("All Dynamic Load Balancing Across GPUs tests completed!")
println("✅ GPU device and metrics creation")
println("✅ Tree workload management and tracking")
println("✅ Load balancing configuration and initialization")
println("✅ Load balancer setup with multi-GPU support")
println("✅ Tree assignment and management")
println("✅ Real-time GPU metrics monitoring and updates")
println("✅ Load imbalance detection algorithms")
println("✅ Work stealing GPU and tree selection strategies")
println("✅ Tree migration protocols with safety checks")
println("✅ Work stealing execution and load redistribution")
println("✅ Load prediction models (linear, exponential)")
println("✅ Adaptive batch sizing based on GPU utilization")
println("✅ Complete load balancing cycle execution")
println("✅ Status monitoring and performance tracking")
println("✅ Cleanup and resource management")
println("✅ Error handling for edge cases and failures")
println("✅ Ready for MCTS ensemble GPU load distribution")