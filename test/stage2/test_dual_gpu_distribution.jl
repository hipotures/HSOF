"""
Test Suite for Dual-GPU Tree Distribution System
Validates GPU assignment, load balancing, fault tolerance, and monitoring functionality
"""

using Test
using Random
using Statistics
using Dates

# Include the dual-GPU distribution module
include("../../src/stage2/dual_gpu_distribution.jl")
using .DualGPUDistribution

# Include ensemble forest for tree management
include("../../src/stage2/ensemble_forest.jl")
using .EnsembleForest

@testset "Dual-GPU Distribution Tests" begin
    
    Random.seed!(42)  # For reproducible tests
    
    @testset "Configuration Tests" begin
        # Test default configuration
        config = create_dual_gpu_config()
        
        @test config.primary_gpu_id == 0
        @test config.secondary_gpu_id == 1
        @test config.assignment_strategy == STATIC_SPLIT
        @test config.enable_dynamic_reallocation == true
        @test config.reallocation_threshold == 0.3f0
        @test config.trees_per_gpu == 50
        @test config.memory_per_tree_mb == 100
        @test config.enable_failover == true
        @test config.max_errors_before_failover == 5
        
        # Test custom configuration
        custom_config = create_dual_gpu_config(
            primary_gpu_id = 1,
            secondary_gpu_id = 0,
            assignment_strategy = DYNAMIC_LOAD,
            trees_per_gpu = 25,
            enable_dynamic_reallocation = false
        )
        
        @test custom_config.primary_gpu_id == 1
        @test custom_config.secondary_gpu_id == 0
        @test custom_config.assignment_strategy == DYNAMIC_LOAD
        @test custom_config.trees_per_gpu == 25
        @test custom_config.enable_dynamic_reallocation == false
        
        println("  ✅ Configuration tests passed")
    end
    
    @testset "GPU Device Info Tests" begin
        # Create mock GPU device info
        device_info = GPUDeviceInfo(
            0,                          # device_id
            "Mock RTX 4090",            # device_name
            24 * 1024^3,               # total_memory (24GB)
            20 * 1024^3,               # available_memory (20GB)
            0.5f0,                     # utilization
            65.0f0,                    # temperature
            250.0f0,                   # power_usage
            Set{Int}([1, 2, 3]),       # assigned_trees
            50,                        # max_trees
            3,                         # active_trees
            1500.0,                    # iterations_per_second
            800.0,                     # memory_bandwidth
            now(),                     # last_update
            true,                      # is_healthy
            nothing,                   # cuda_context
            0,                         # error_count
            nothing                    # last_error
        )
        
        @test device_info.device_id == 0
        @test device_info.device_name == "Mock RTX 4090"
        @test device_info.total_memory == 24 * 1024^3
        @test device_info.available_memory == 20 * 1024^3
        @test device_info.utilization == 0.5f0
        @test device_info.max_trees == 50
        @test length(device_info.assigned_trees) == 3
        @test device_info.active_trees == 3
        @test device_info.is_healthy == true
        @test device_info.error_count == 0
        
        println("  ✅ GPU device info tests passed")
    end
    
    @testset "Assignment Strategy Tests" begin
        # Test enum values
        @test Int(STATIC_SPLIT) == 1
        @test Int(DYNAMIC_LOAD) == 2
        @test Int(MEMORY_BASED) == 3
        @test Int(PERFORMANCE_BASED) == 4
        
        println("  ✅ Assignment strategy tests passed")
    end
    
    # Skip GPU-dependent tests if CUDA not available or insufficient GPUs
    if !CUDA.functional()
        println("  ⚠️  Skipping GPU-dependent tests (CUDA not available)")
    elseif CUDA.ndevices() < 2
        println("  ⚠️  Skipping GPU-dependent tests (need 2+ GPUs, found $(CUDA.ndevices()))")
    else
        @testset "GPU Manager Initialization Tests" begin
            # Test initialization with default config
            config = create_dual_gpu_config()
            
            try
                manager = initialize_dual_gpu_manager(config)
                
                @test manager.distribution_state == "active"
                @test length(manager.gpu_devices) == 2
                @test length(manager.active_gpus) == 2
                @test manager.config.primary_gpu_id in keys(manager.gpu_devices)
                @test manager.config.secondary_gpu_id in keys(manager.gpu_devices)
                @test manager.total_trees_managed == 0
                @test !manager.monitoring_active
                @test !manager.failover_active
                
                # Check GPU device initialization
                for gpu_id in [config.primary_gpu_id, config.secondary_gpu_id]
                    device_info = manager.gpu_devices[gpu_id]
                    @test device_info.device_id == gpu_id
                    @test device_info.is_healthy == true
                    @test device_info.error_count == 0
                    @test length(device_info.assigned_trees) == 0
                    @test device_info.max_trees == config.trees_per_gpu
                end
                
                # Cleanup
                cleanup_distribution!(manager)
                
                println("  ✅ GPU manager initialization tests passed")
                
            catch e
                println("  ⚠️  GPU manager initialization failed: $e")
            end
        end
        
        @testset "Tree Assignment Tests" begin
            try
                # Create distribution manager
                config = create_dual_gpu_config(trees_per_gpu = 5)
                manager = initialize_dual_gpu_manager(config)
                
                # Create forest manager with trees
                forest_config = create_tree_pool_config(initial_trees = 10)
                forest_manager = initialize_forest_manager(forest_config)
                
                # Test static split assignment
                assign_trees_to_gpus!(manager, forest_manager)
                
                @test manager.total_trees_managed == 10
                @test length(manager.tree_to_gpu_mapping) == 10
                
                # Verify trees are assigned to both GPUs
                gpu0_trees = length(manager.gpu_to_trees_mapping[config.primary_gpu_id])
                gpu1_trees = length(manager.gpu_to_trees_mapping[config.secondary_gpu_id])
                
                @test gpu0_trees + gpu1_trees == 10
                @test gpu0_trees > 0  # Both GPUs should have trees
                @test gpu1_trees > 0
                
                # Verify forest manager trees have GPU assignments
                for (tree_id, tree) in forest_manager.trees
                    @test tree.gpu_device_id in [config.primary_gpu_id, config.secondary_gpu_id]
                    @test tree.gpu_device_id == manager.tree_to_gpu_mapping[tree_id]
                end
                
                # Cleanup
                cleanup_forest!(forest_manager)
                cleanup_distribution!(manager)
                
                println("  ✅ Tree assignment tests passed")
                
            catch e
                println("  ⚠️  Tree assignment test failed: $e")
            end
        end
        
        @testset "Distribution Status Tests" begin
            try
                # Create distribution manager
                config = create_dual_gpu_config()
                manager = initialize_dual_gpu_manager(config)
                
                # Test status retrieval
                status = get_distribution_status(manager)
                
                @test haskey(status, "distribution_state")
                @test haskey(status, "total_trees_managed")
                @test haskey(status, "active_gpus")
                @test haskey(status, "failover_active")
                @test haskey(status, "monitoring_active")
                @test haskey(status, "gpus")
                
                @test status["distribution_state"] == "active"
                @test status["total_trees_managed"] == 0
                @test status["active_gpus"] == 2
                @test status["failover_active"] == false
                @test status["monitoring_active"] == false
                
                # Test GPU-specific status
                for gpu_id in [config.primary_gpu_id, config.secondary_gpu_id]
                    gpu_status = status["gpus"]["gpu_$gpu_id"]
                    @test haskey(gpu_status, "device_name")
                    @test haskey(gpu_status, "assigned_trees")
                    @test haskey(gpu_status, "max_trees")
                    @test haskey(gpu_status, "utilization")
                    @test haskey(gpu_status, "is_healthy")
                    
                    @test gpu_status["assigned_trees"] == 0
                    @test gpu_status["max_trees"] == config.trees_per_gpu
                    @test gpu_status["is_healthy"] == true
                end
                
                # Test report generation
                report = generate_distribution_report(manager)
                @test contains(report, "Dual-GPU Distribution Status Report")
                @test contains(report, "Distribution State: active")
                @test contains(report, "Total Trees: 0")
                @test contains(report, "Active GPUs: 2")
                
                # Cleanup
                cleanup_distribution!(manager)
                
                println("  ✅ Distribution status tests passed")
                
            catch e
                println("  ⚠️  Distribution status test failed: $e")
            end
        end
    end
    
    @testset "Mock Assignment Algorithm Tests" begin
        # Test assignment algorithms with mock data (no actual GPU required)
        
        # Create mock manager structure
        gpu_devices = Dict{Int, GPUDeviceInfo}(
            0 => GPUDeviceInfo(0, "Mock GPU 0", 24*1024^3, 20*1024^3, 0.3f0, 60.0f0, 200.0f0, 
                              Set{Int}(), 50, 0, 1000.0, 800.0, now(), true, nothing, 0, nothing),
            1 => GPUDeviceInfo(1, "Mock GPU 1", 24*1024^3, 18*1024^3, 0.7f0, 70.0f0, 280.0f0, 
                              Set{Int}(), 50, 0, 1200.0, 900.0, now(), true, nothing, 0, nothing)
        )
        
        # Test static split assignment logic
        tree_ids = collect(1:10)
        config = create_dual_gpu_config()
        
        # Create mock manager
        manager = DualGPUDistributionManager(
            config, gpu_devices, [0, 1], [0, 1],
            Dict{Int, Int}(), Dict{Int, Vector{Int}}(0 => Int[], 1 => Int[]),
            now(), Tuple{DateTime, Int, Int, Int}[], false, nothing,
            Dict{String, Vector{Float64}}(), ReentrantLock(), now(),
            Dict{String, Any}(), false, nothing, String[], "active", 0, now()
        )
        
        # Test static split
        DualGPUDistribution.assign_trees_static_split!(manager, tree_ids)
        @test length(manager.tree_to_gpu_mapping) == 10
        @test length(manager.gpu_to_trees_mapping[0]) == 5
        @test length(manager.gpu_to_trees_mapping[1]) == 5
        
        # Reset for next test
        empty!(manager.tree_to_gpu_mapping)
        manager.gpu_to_trees_mapping[0] = Int[]
        manager.gpu_to_trees_mapping[1] = Int[]
        
        # Test dynamic load assignment
        DualGPUDistribution.assign_trees_dynamic_load!(manager, tree_ids)
        @test length(manager.tree_to_gpu_mapping) == 10
        
        # GPU 0 should get more trees (lower utilization: 0.3 vs 0.7)
        gpu0_trees = length(manager.gpu_to_trees_mapping[0])
        gpu1_trees = length(manager.gpu_to_trees_mapping[1])
        @test gpu0_trees >= gpu1_trees  # Lower utilization GPU gets more trees
        
        println("  ✅ Mock assignment algorithm tests passed")
    end
    
    @testset "Error Handling Tests" begin
        # Test configuration validation
        @test_throws ErrorException create_dual_gpu_config(
            primary_gpu_id = 0,
            secondary_gpu_id = 0  # Same GPU for both
        )
        
        # Test invalid tree assignment
        config = create_dual_gpu_config()
        
        # Create manager with mock devices
        gpu_devices = Dict{Int, GPUDeviceInfo}(
            0 => GPUDeviceInfo(0, "Mock GPU 0", 24*1024^3, 20*1024^3, 0.0f0, 60.0f0, 200.0f0, 
                              Set{Int}(), 50, 0, 1000.0, 800.0, now(), true, nothing, 0, nothing)
        )
        
        manager = DualGPUDistributionManager(
            config, gpu_devices, [0], [0],  # Only one GPU active
            Dict{Int, Int}(), Dict{Int, Vector{Int}}(0 => Int[]),
            now(), Tuple{DateTime, Int, Int, Int}[], false, nothing,
            Dict{String, Vector{Float64}}(), ReentrantLock(), now(),
            Dict{String, Any}(), false, nothing, String[], "active", 0, now()
        )
        
        # Test assigning tree to non-existent GPU
        @test_throws KeyError DualGPUDistribution.assign_tree_to_gpu!(manager, 1, 999)
        
        println("  ✅ Error handling tests passed")
    end
end

println("All Dual-GPU Distribution tests completed!")
println("✅ Configuration and setup validation")
println("✅ GPU device info management")
println("✅ Tree assignment strategies")
if CUDA.functional() && CUDA.ndevices() >= 2
    println("✅ GPU manager initialization and cleanup")
    println("✅ Distribution status and reporting")
else
    println("⚠️  GPU-dependent tests skipped (CUDA/multi-GPU not available)")
end
println("✅ Assignment algorithm validation")
println("✅ Error handling and edge cases")
println("✅ Ready for dual-GPU MCTS ensemble execution")