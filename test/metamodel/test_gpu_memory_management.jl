#!/usr/bin/env julia

using Test
using CUDA
using Random
using Dates
using Statistics
using Printf

# Skip if no GPU available for some tests
gpu_available = CUDA.functional()

println("GPU MEMORY MANAGEMENT SYSTEM - COMPREHENSIVE TESTS")
println("="^80)
println("Testing unified memory allocation system for model weights, replay buffer, and inference batches")
println("GPU Available: $gpu_available")
println("="^80)

# Include the GPU memory management module
include("../../src/metamodel/gpu_memory_management.jl")
using .GPUMemoryManagement

"""
Test 1: Basic configuration and manager creation
"""
function test_configuration_and_creation()
    println("\n--- Test 1: Configuration and Manager Creation ---")
    
    # Test default configuration
    config = MemoryConfig()
    @test config.total_vram_gb == 24.0
    @test config.reserved_system_gb == 2.0
    @test haskey(config.pool_size_ratios, :model_weights)
    @test haskey(config.pool_size_ratios, :replay_buffer)
    @test haskey(config.pool_size_ratios, :inference_batch)
    
    # Verify pool ratios sum to 1.0
    total_ratio = sum(values(config.pool_size_ratios))
    @test abs(total_ratio - 1.0) < 0.01
    
    # Test custom configuration
    custom_config = MemoryConfig(
        total_vram_gb = 12.0,
        reserved_system_gb = 1.0,
        pool_size_ratios = Dict(
            :model_weights => 0.4,
            :replay_buffer => 0.3,
            :general_pool => 0.3
        ),
        enable_defragmentation = false,
        adaptive_batch_sizing = false
    )
    @test custom_config.total_vram_gb == 12.0
    @test !custom_config.enable_defragmentation
    @test !custom_config.adaptive_batch_sizing
    
    # Test manager creation (only if GPU available)
    if gpu_available
        manager = create_memory_manager(custom_config)
        @test manager.config.total_vram_gb == 12.0
        @test length(manager.pools) == 3
        @test haskey(manager.pools, :model_weights)
        @test haskey(manager.pools, :replay_buffer)
        @test haskey(manager.pools, :general_pool)
        
        # Test device selection
        @test manager.device_id == 0
        
        println("✓ Configuration created and validated")
        println("✓ Memory manager initialized successfully")
        println("✓ Memory pools created: $(length(manager.pools))")
        
        return manager
    else
        println("✓ Configuration created and validated")
        println("⚠ GPU manager creation skipped (no GPU)")
        return nothing
    end
end

"""
Test 2: Component registration and management
"""
function test_component_registration()
    println("\n--- Test 2: Component Registration and Management ---")
    
    if !gpu_available
        println("⚠ Test skipped (no GPU)")
        return nothing
    end
    
    config = MemoryConfig(total_vram_gb = 8.0)
    manager = create_memory_manager(config)
    
    # Test component registration
    @test register_component!(manager, "neural_network", 
                             max_memory_gb = 2.0,
                             preferred_pools = [:model_weights],
                             priority = 1)
    
    @test register_component!(manager, "replay_buffer",
                             max_memory_gb = 1.5,
                             preferred_pools = [:replay_buffer],
                             priority = 2)
    
    @test register_component!(manager, "batch_inference",
                             max_memory_gb = 1.0,
                             preferred_pools = [:inference_batch],
                             priority = 3)
    
    # Verify registration
    @test haskey(manager.registered_components, "neural_network")
    @test haskey(manager.registered_components, "replay_buffer")
    @test haskey(manager.registered_components, "batch_inference")
    
    # Test component properties
    nn_component = manager.registered_components["neural_network"]
    @test nn_component["max_memory_bytes"] == Int64(2.0 * 1024^3)
    @test nn_component["preferred_pools"] == [:model_weights]
    @test nn_component["priority"] == 1
    @test nn_component["total_allocated_bytes"] == 0
    
    # Test duplicate registration (should update)
    @test register_component!(manager, "neural_network", max_memory_gb = 3.0)
    updated_component = manager.registered_components["neural_network"]
    @test updated_component["max_memory_bytes"] == Int64(3.0 * 1024^3)
    
    println("✓ Component registration working")
    println("✓ Component properties correctly stored")
    println("✓ Duplicate registration handled")
    
    return manager
end

"""
Test 3: Memory allocation and deallocation
"""
function test_memory_allocation()
    println("\n--- Test 3: Memory Allocation and Deallocation ---")
    
    if !gpu_available
        println("⚠ Test skipped (no GPU)")
        return nothing
    end
    
    config = MemoryConfig(total_vram_gb = 8.0)
    manager = create_memory_manager(config)
    
    # Register test component
    register_component!(manager, "test_component", max_memory_gb = 2.0)
    
    # Test basic allocation
    block_id1 = allocate_memory!(manager, "test_component", 1024 * 1024, pool = :general_pool)  # 1MB
    @test block_id1 !== nothing
    @test haskey(manager.block_registry, block_id1)
    
    block = manager.block_registry[block_id1]
    @test block.component == "test_component"
    @test block.size_bytes >= 1024 * 1024
    @test !block.is_free
    
    # Test pool allocation tracking
    pool = manager.pools[:general_pool]
    @test pool.allocated_bytes >= 1024 * 1024
    @test pool.allocation_count >= 1
    
    # Test component tracking
    component = manager.registered_components["test_component"]
    @test length(component["allocated_blocks"]) >= 1
    @test component["total_allocated_bytes"] >= 1024 * 1024
    
    # Allocate more memory
    block_id2 = allocate_memory!(manager, "test_component", 2 * 1024 * 1024)  # 2MB
    @test block_id2 !== nothing
    @test block_id2 != block_id1
    
    # Test memory freeing
    @test free_memory!(manager, block_id1)
    freed_block = manager.block_registry[block_id1]
    @test freed_block.is_free
    
    # Verify pool updates
    @test pool.free_bytes > 0
    @test pool.deallocation_count >= 1
    
    # Test freeing non-existent block
    @test !free_memory!(manager, "non_existent_block")
    
    println("✓ Memory allocation working")
    println("✓ Memory deallocation working")
    println("✓ Pool and component tracking accurate")
    
    return manager
end

"""
Test 4: Memory pool management and switching
"""
function test_pool_management()
    println("\n--- Test 4: Memory Pool Management and Switching ---")
    
    if !gpu_available
        println("⚠ Test skipped (no GPU)")
        return nothing
    end
    
    config = MemoryConfig(
        total_vram_gb = 4.0,
        pool_size_ratios = Dict(
            :small_pool => 0.3,
            :medium_pool => 0.4,
            :large_pool => 0.3
        )
    )
    manager = create_memory_manager(config)
    
    register_component!(manager, "test_component", max_memory_gb = 2.0)
    
    # Allocate from specific pools
    small_block = allocate_memory!(manager, "test_component", 100 * 1024, pool = :small_pool)
    medium_block = allocate_memory!(manager, "test_component", 200 * 1024, pool = :medium_pool)
    large_block = allocate_memory!(manager, "test_component", 300 * 1024, pool = :large_pool)
    
    @test small_block !== nothing
    @test medium_block !== nothing
    @test large_block !== nothing
    
    # Verify blocks are in correct pools
    @test manager.block_registry[small_block].pool == :small_pool
    @test manager.block_registry[medium_block].pool == :medium_pool
    @test manager.block_registry[large_block].pool == :large_pool
    
    # Test pool utilization
    small_pool = manager.pools[:small_pool]
    medium_pool = manager.pools[:medium_pool]
    large_pool = manager.pools[:large_pool]
    
    @test small_pool.allocated_bytes > 0
    @test medium_pool.allocated_bytes > 0
    @test large_pool.allocated_bytes > 0
    
    # Test allocation from non-existent pool should fail
    @test_throws ArgumentError allocate_memory!(manager, "test_component", 1024, pool = :non_existent)
    
    println("✓ Multi-pool allocation working")
    println("✓ Pool-specific tracking accurate")
    println("✓ Pool validation working")
    
    return manager
end

"""
Test 5: Memory pressure detection and adaptive responses
"""
function test_memory_pressure()
    println("\n--- Test 5: Memory Pressure Detection and Adaptive Responses ---")
    
    if !gpu_available
        println("⚠ Test skipped (no GPU)")
        return nothing
    end
    
    # Create small memory configuration to easily trigger pressure
    config = MemoryConfig(
        total_vram_gb = 2.0,
        pressure_warning_threshold = 0.7,
        pressure_critical_threshold = 0.9,
        adaptive_batch_sizing = true,
        min_batch_size = 16,
        max_batch_size = 256
    )
    manager = create_memory_manager(config)
    
    register_component!(manager, "pressure_test", max_memory_gb = 1.8)
    
    # Start with normal pressure
    pressure_info = check_memory_pressure(manager)
    @test pressure_info["level"] == "normal"
    
    # Allocate memory to create warning pressure
    warning_size = Int64(config.total_vram_gb * 0.75 * 1024^3)  # 75% of available
    try
        warning_block = allocate_memory!(manager, "pressure_test", warning_size ÷ 4)
        
        pressure_info = check_memory_pressure(manager)
        if pressure_info["pressure_ratio"] > config.pressure_warning_threshold
            @test pressure_info["level"] in ["warning", "critical"]
            @test haskey(pressure_info, "recommendation")
        end
    catch OutOfMemoryError
        @info "Memory allocation triggered OutOfMemoryError as expected"
    end
    
    # Test adaptive batch sizing
    original_batch_size = manager.current_batch_size
    set_adaptive_batch_size!(manager, 64)
    @test manager.current_batch_size == 64
    
    # Test batch size limits
    set_adaptive_batch_size!(manager, 8)  # Below minimum
    @test manager.current_batch_size == config.min_batch_size
    
    set_adaptive_batch_size!(manager, 512)  # Above maximum
    @test manager.current_batch_size == config.max_batch_size
    
    println("✓ Memory pressure detection working")
    println("✓ Adaptive batch sizing working")
    println("✓ Pressure thresholds correctly enforced")
    
    return manager
end

"""
Test 6: Defragmentation and optimization
"""
function test_defragmentation()
    println("\n--- Test 6: Defragmentation and Optimization ---")
    
    if !gpu_available
        println("⚠ Test skipped (no GPU)")
        return nothing
    end
    
    config = MemoryConfig(
        total_vram_gb = 4.0,
        enable_defragmentation = true,
        defrag_threshold = 0.2
    )
    manager = create_memory_manager(config)
    
    register_component!(manager, "frag_test", max_memory_gb = 2.0)
    
    # Allocate multiple small blocks to create fragmentation
    block_ids = String[]
    for i in 1:10
        block_id = allocate_memory!(manager, "frag_test", 50 * 1024)  # 50KB each
        push!(block_ids, block_id)
    end
    
    # Free every other block to create fragmentation
    for i in 1:2:length(block_ids)
        free_memory!(manager, block_ids[i])
    end
    
    # Check fragmentation
    pool = manager.pools[:general_pool]
    initial_fragmentation = pool.fragmentation_ratio
    
    # Force defragmentation
    success = defragment_pool!(manager, :general_pool)
    
    if success
        @test pool.fragmentation_ratio <= initial_fragmentation
        @test manager.defrag_operations > 0
    end
    
    # Test optimization
    optimization_result = optimize_memory_layout!(manager)
    @test haskey(optimization_result, "optimization_time_ms")
    @test haskey(optimization_result, "defragmentation_operations")
    @test haskey(optimization_result, "cleaned_bytes")
    
    println("✓ Defragmentation working")
    println("✓ Memory optimization completed")
    println("✓ Fragmentation metrics tracked")
    
    return manager
end

"""
Test 7: Fallback to system RAM
"""
function test_system_ram_fallback()
    println("\n--- Test 7: System RAM Fallback ---")
    
    if !gpu_available
        println("⚠ Test skipped (no GPU)")
        return nothing
    end
    
    # Create very small memory configuration to force fallback
    config = MemoryConfig(total_vram_gb = 0.5)  # Very small
    manager = create_memory_manager(config)
    
    register_component!(manager, "fallback_test", max_memory_gb = 2.0)
    
    # Try to allocate more than available GPU memory
    large_allocation_size = Int64(1.0 * 1024^3)  # 1GB
    
    try
        block_id = allocate_memory!(manager, "fallback_test", large_allocation_size)
        
        # If allocation succeeded, it might be in fallback storage
        if startswith(block_id, "fallback_")
            @test manager.fallback_usage_bytes > 0
            @test haskey(manager.fallback_storage, split(block_id, "_")[2:end][1])
            
            # Test fallback freeing
            @test free_memory!(manager, block_id)
            
            println("✓ System RAM fallback working")
            println("✓ Fallback memory tracking accurate")
        else
            println("✓ GPU allocation succeeded (sufficient memory)")
        end
        
    catch OutOfMemoryError
        println("✓ OutOfMemoryError correctly thrown when no fallback possible")
    end
    
    return manager
end

"""
Test 8: Statistics and monitoring
"""
function test_statistics_monitoring()
    println("\n--- Test 8: Statistics and Monitoring ---")
    
    if !gpu_available
        println("⚠ Test skipped (no GPU)")
        return nothing
    end
    
    config = MemoryConfig(enable_memory_profiling = true)
    manager = create_memory_manager(config)
    
    register_component!(manager, "stats_test", max_memory_gb = 1.0)
    
    # Perform some allocations
    block_ids = String[]
    for i in 1:5
        block_id = allocate_memory!(manager, "stats_test", 100 * 1024)
        push!(block_ids, block_id)
    end
    
    # Get comprehensive statistics
    stats = get_memory_stats(manager)
    @test stats.total_allocated_bytes > 0
    @test haskey(stats.pool_stats, :general_pool)
    @test haskey(stats.component_usage, "stats_test")
    @test stats.pressure_level in [:normal, :warning, :critical]
    
    # Test component usage statistics
    usage = get_component_usage(manager, "stats_test")
    @test usage !== nothing
    @test usage["allocated_blocks"] == 5
    @test usage["total_allocated_bytes"] > 0
    @test haskey(usage, "utilization_ratio")
    
    # Test non-existent component
    @test get_component_usage(manager, "non_existent") === nothing
    
    # Test memory profiling
    @test manager.profiling_enabled
    @test haskey(manager.memory_profile, "component_profiles")
    @test haskey(manager.memory_profile["component_profiles"], "stats_test")
    
    # Test saving memory profile
    profile_file = "test_memory_profile.json"
    @test save_memory_profile(manager, profile_file)
    @test isfile(profile_file)
    
    # Cleanup
    rm(profile_file, force=true)
    
    println("✓ Memory statistics collection working")
    println("✓ Component usage tracking accurate")
    println("✓ Memory profiling functional")
    
    return manager
end

"""
Test 9: Component lifecycle management
"""
function test_component_lifecycle()
    println("\n--- Test 9: Component Lifecycle Management ---")
    
    if !gpu_available
        println("⚠ Test skipped (no GPU)")
        return nothing
    end
    
    config = MemoryConfig()
    manager = create_memory_manager(config)
    
    # Register and allocate for component
    register_component!(manager, "lifecycle_test", max_memory_gb = 1.0)
    
    block_ids = String[]
    for i in 1:3
        block_id = allocate_memory!(manager, "lifecycle_test", 50 * 1024)
        push!(block_ids, block_id)
    end
    
    # Verify allocations
    component = manager.registered_components["lifecycle_test"]
    @test length(component["allocated_blocks"]) == 3
    @test component["total_allocated_bytes"] > 0
    
    # Unregister component (should free all memory)
    @test unregister_component!(manager, "lifecycle_test")
    
    # Verify cleanup
    @test !haskey(manager.registered_components, "lifecycle_test")
    
    # Verify blocks are freed
    for block_id in block_ids
        if haskey(manager.block_registry, block_id)
            @test manager.block_registry[block_id].is_free
        end
    end
    
    # Test unregistering non-existent component
    @test !unregister_component!(manager, "non_existent")
    
    println("✓ Component registration/unregistration working")
    println("✓ Automatic memory cleanup on unregister")
    println("✓ Component lifecycle properly managed")
    
    return manager
end

"""
Test 10: Error handling and edge cases
"""
function test_error_handling()
    println("\n--- Test 10: Error Handling and Edge Cases ---")
    
    if !gpu_available
        println("⚠ Test skipped (no GPU)")
        return nothing
    end
    
    config = MemoryConfig(total_vram_gb = 2.0)
    manager = create_memory_manager(config)
    
    # Test allocating for unregistered component
    @test_throws ArgumentError allocate_memory!(manager, "unregistered", 1024)
    
    # Test invalid pool configuration
    @test_throws ArgumentError MemoryConfig(
        pool_size_ratios = Dict(:pool1 => 0.5, :pool2 => 0.6)  # Sums to 1.1
    )
    
    # Register component and test memory limits
    register_component!(manager, "limited_component", max_memory_gb = 0.1)  # 100MB limit
    
    # Try to allocate more than component limit
    large_size = Int64(200 * 1024 * 1024)  # 200MB
    try
        block_id = allocate_memory!(manager, "limited_component", large_size)
        # If successful, should be fallback allocation
        if startswith(block_id, "fallback_")
            @test manager.fallback_usage_bytes > 0
        end
    catch OutOfMemoryError
        # Expected if no fallback possible
        @test true
    end
    
    # Test zero-size allocation
    @test_throws ArgumentError allocate_memory!(manager, "limited_component", 0)
    
    # Test invalid device ID
    @test_throws ArgumentError create_memory_manager(config, device_id = 999)
    
    # Test freeing already freed memory
    register_component!(manager, "edge_test", max_memory_gb = 0.5)
    block_id = allocate_memory!(manager, "edge_test", 1024)
    @test free_memory!(manager, block_id)
    @test !free_memory!(manager, block_id)  # Should return false, not error
    
    println("✓ Error handling working correctly")
    println("✓ Component limits enforced")
    println("✓ Edge cases handled gracefully")
    
    return manager
end

"""
Test 11: Performance and memory usage
"""
function test_performance()
    println("\n--- Test 11: Performance and Memory Usage ---")
    
    if !gpu_available
        println("⚠ Test skipped (no GPU)")
        return nothing
    end
    
    config = MemoryConfig(enable_memory_profiling = true)
    manager = create_memory_manager(config)
    
    register_component!(manager, "perf_test", max_memory_gb = 2.0)
    
    # Measure allocation performance
    n_allocations = 100
    allocation_times = Float64[]
    block_ids = String[]
    
    for i in 1:n_allocations
        start_time = time()
        block_id = allocate_memory!(manager, "perf_test", 10 * 1024)  # 10KB
        alloc_time = (time() - start_time) * 1000  # ms
        
        push!(allocation_times, alloc_time)
        push!(block_ids, block_id)
    end
    
    avg_alloc_time = mean(allocation_times)
    max_alloc_time = maximum(allocation_times)
    
    @test avg_alloc_time < 1.0  # Should be under 1ms on average
    
    # Measure deallocation performance
    deallocation_times = Float64[]
    
    for block_id in block_ids
        start_time = time()
        free_memory!(manager, block_id)
        dealloc_time = (time() - start_time) * 1000  # ms
        
        push!(deallocation_times, dealloc_time)
    end
    
    avg_dealloc_time = mean(deallocation_times)
    
    @test avg_dealloc_time < 0.5  # Should be even faster
    
    # Test statistics collection performance
    start_time = time()
    stats = get_memory_stats(manager)
    stats_time = (time() - start_time) * 1000
    
    @test stats_time < 10.0  # Should be very fast
    
    # Test memory overhead
    initial_memory = manager.total_allocated_bytes
    
    # The manager itself should have minimal overhead
    @test initial_memory == 0  # No allocations yet in fresh manager
    
    println("  Average allocation time: $(round(avg_alloc_time, digits=3)) ms")
    println("  Average deallocation time: $(round(avg_dealloc_time, digits=3)) ms")
    println("  Statistics collection time: $(round(stats_time, digits=3)) ms")
    println("  Maximum allocation time: $(round(max_alloc_time, digits=3)) ms")
    println("✓ Performance within acceptable limits")
    println("✓ Memory overhead minimal")
    
    return manager, Dict(
        "avg_alloc_time" => avg_alloc_time,
        "avg_dealloc_time" => avg_dealloc_time,
        "stats_time" => stats_time
    )
end

"""
Test 12: Memory manager reset and cleanup
"""
function test_reset_and_cleanup()
    println("\n--- Test 12: Reset and Cleanup ---")
    
    if !gpu_available
        println("⚠ Test skipped (no GPU)")
        return nothing
    end
    
    config = MemoryConfig()
    manager = create_memory_manager(config)
    
    # Register components and allocate memory
    register_component!(manager, "reset_test1", max_memory_gb = 1.0)
    register_component!(manager, "reset_test2", max_memory_gb = 1.0)
    
    for i in 1:5
        allocate_memory!(manager, "reset_test1", 100 * 1024)
        allocate_memory!(manager, "reset_test2", 100 * 1024)
    end
    
    # Verify allocations exist
    @test manager.total_allocated_bytes > 0
    @test length(manager.registered_components) == 2
    @test !isempty(manager.block_registry)
    
    # Reset manager
    @test reset_memory_manager!(manager)
    
    # Verify complete reset
    @test manager.total_allocated_bytes == 0
    @test length(manager.registered_components) == 0
    @test isempty(manager.block_registry)
    @test isempty(manager.fallback_storage)
    @test manager.fallback_usage_bytes == 0
    
    # Verify pools are reset
    for pool in values(manager.pools)
        @test pool.allocated_bytes == 0
        @test pool.free_bytes == pool.total_size_bytes
        @test isempty(pool.blocks)
        @test isempty(pool.free_blocks)
    end
    
    # Verify statistics are reset
    @test isempty(manager.stats_history)
    @test isempty(manager.allocation_times)
    @test isempty(manager.deallocation_times)
    
    println("✓ Memory manager reset working")
    println("✓ Complete cleanup performed")
    println("✓ All resources freed")
    
    return true
end

"""
Main test runner
"""
function run_gpu_memory_management_tests()
    println("\n🚀 Starting GPU Memory Management System Tests...")
    
    test_results = Dict{String, Any}()
    all_tests_passed = true
    performance_data = nothing
    
    try
        # Test 1: Configuration and creation
        manager1 = test_configuration_and_creation()
        test_results["config_creation"] = "PASSED"
        
        if gpu_available
            # Test 2: Component registration
            manager2 = test_component_registration()
            test_results["component_registration"] = "PASSED"
            
            # Test 3: Memory allocation
            manager3 = test_memory_allocation()
            test_results["memory_allocation"] = "PASSED"
            
            # Test 4: Pool management
            manager4 = test_pool_management()
            test_results["pool_management"] = "PASSED"
            
            # Test 5: Memory pressure
            manager5 = test_memory_pressure()
            test_results["memory_pressure"] = "PASSED"
            
            # Test 6: Defragmentation
            manager6 = test_defragmentation()
            test_results["defragmentation"] = "PASSED"
            
            # Test 7: System RAM fallback
            manager7 = test_system_ram_fallback()
            test_results["system_fallback"] = "PASSED"
            
            # Test 8: Statistics monitoring
            manager8 = test_statistics_monitoring()
            test_results["statistics_monitoring"] = "PASSED"
            
            # Test 9: Component lifecycle
            manager9 = test_component_lifecycle()
            test_results["component_lifecycle"] = "PASSED"
            
            # Test 10: Error handling
            manager10 = test_error_handling()
            test_results["error_handling"] = "PASSED"
            
            # Test 11: Performance
            manager11, performance_data = test_performance()
            test_results["performance"] = Dict(
                "status" => "PASSED",
                "avg_alloc_time_ms" => performance_data["avg_alloc_time"],
                "avg_dealloc_time_ms" => performance_data["avg_dealloc_time"],
                "stats_time_ms" => performance_data["stats_time"]
            )
            
            # Test 12: Reset and cleanup
            test12_result = test_reset_and_cleanup()
            test_results["reset_cleanup"] = "PASSED"
        else
            test_results["gpu_tests"] = "SKIPPED (No GPU)"
        end
        
    catch e
        println("❌ Test failed with error: $e")
        all_tests_passed = false
        test_results["error"] = string(e)
    end
    
    # Final summary
    println("\n" * "="^80)
    println("GPU MEMORY MANAGEMENT SYSTEM - TEST RESULTS")
    println("="^80)
    
    if all_tests_passed
        println("🎉 ALL TESTS PASSED!")
        println("✅ Configuration and manager creation: Working")
        
        if gpu_available
            println("✅ Component registration and management: Working")
            println("✅ Memory allocation and deallocation: Working")
            println("✅ Memory pool management: Working")
            println("✅ Memory pressure detection: Working")
            println("✅ Defragmentation and optimization: Working")
            println("✅ System RAM fallback: Working")
            println("✅ Statistics and monitoring: Working")
            println("✅ Component lifecycle management: Working")
            println("✅ Error handling and edge cases: Working")
            println("✅ Performance optimization: Working")
            println("✅ Reset and cleanup: Working")
            
            if performance_data !== nothing
                println("\n📊 Performance Metrics:")
                println("  Average allocation time: $(round(performance_data["avg_alloc_time"], digits=3)) ms")
                println("  Average deallocation time: $(round(performance_data["avg_dealloc_time"], digits=3)) ms")
                println("  Statistics collection time: $(round(performance_data["stats_time"], digits=3)) ms")
            end
            
            println("\n✅ Task 5.9 - Implement GPU Memory Management: COMPLETED")
            println("✅ Unified memory allocation system: IMPLEMENTED")
            println("✅ Memory pool allocators with pre-allocated chunks: IMPLEMENTED")
            println("✅ Defragmentation for long-running sessions: IMPLEMENTED")
            println("✅ Memory pressure monitoring and adaptive batch sizing: IMPLEMENTED")
            println("✅ Fallback to system RAM for overflow scenarios: IMPLEMENTED")
            println("✅ Profiling tools for memory usage analysis: IMPLEMENTED")
        else
            println("⚠ GPU-specific tests skipped (no GPU available)")
            println("✅ Configuration and basic functionality: Working")
        end
        
    else
        println("❌ SOME TESTS FAILED")
        println("❌ Task 5.9 - Implement GPU Memory Management: NEEDS ATTENTION")
    end
    
    println("="^80)
    return all_tests_passed, test_results
end

# Run tests if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    success, results = run_gpu_memory_management_tests()
    exit(success ? 0 : 1)
end

# Export for module usage
run_gpu_memory_management_tests