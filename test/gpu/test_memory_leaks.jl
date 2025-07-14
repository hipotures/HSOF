using Test
using CUDA
using Printf

# Include modules  
include("../../src/gpu/kernels/mcts_types.jl")
include("../../src/gpu/kernels/memory_pool.jl")

using .MCTSTypes
using .MemoryPool

"""
Memory leak detection framework for GPU operations
"""
struct MemorySnapshot
    free_memory::Int
    total_memory::Int
    allocated_buffers::Int
    timestamp::Float64
end

function take_memory_snapshot()
    info = CUDA.MemoryInfo()
    return MemorySnapshot(
        info.free_bytes,
        info.total_bytes,
        info.total_bytes - info.free_bytes,
        time()
    )
end

function report_memory_diff(before::MemorySnapshot, after::MemorySnapshot, test_name::String)
    leaked_bytes = before.free_memory - after.free_memory
    leaked_mb = leaked_bytes / (1024 * 1024)
    
    if abs(leaked_mb) > 1.0  # More than 1MB difference
        @warn "Potential memory leak detected in $test_name" leaked_mb
        return false
    else
        @info "Memory test passed for $test_name" leaked_mb
        return true
    end
end

@testset "GPU Memory Leak Detection" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping memory leak tests"
        return
    end
    
    # Force garbage collection before tests
    GC.gc(true)
    CUDA.reclaim()
    
    @testset "Tree Allocation/Deallocation Cycles" begin
        # Take baseline snapshot
        baseline = take_memory_snapshot()
        
        # Run multiple allocation/deallocation cycles
        for cycle in 1:10
            tree = MCTSTreeSoA(CUDA.device())
            
            # Allocate many nodes
            CUDA.@allowscalar begin
                tree.next_free_node[1] = 1
                tree.total_nodes[1] = 0
                
                for i in 1:10000
                    node = allocate_node!(tree)
                    if node > 0
                        tree.node_states[node] = NODE_ACTIVE
                        tree.visit_counts[node] = i
                        tree.total_scores[node] = Float32(i) * 0.1f0
                    end
                end
            end
            
            # Tree goes out of scope and should be freed
            tree = nothing
            CUDA.unsafe_free!
        end
        
        # Force cleanup
        GC.gc(true)
        CUDA.reclaim()
        sleep(0.1)  # Give time for async operations
        
        # Take final snapshot
        final = take_memory_snapshot()
        
        # Check for leaks
        @test report_memory_diff(baseline, final, "Tree Allocation Cycles")
    end
    
    @testset "Memory Pool Manager Lifecycle" begin
        baseline = take_memory_snapshot()
        
        for cycle in 1:5
            tree = MCTSTreeSoA(CUDA.device())
            manager = MemoryPoolManager(tree)
            
            # Allocate and free nodes
            CUDA.@allowscalar begin
                tree.next_free_node[1] = 1
                tree.total_nodes[1] = 0
                
                allocated = Int32[]
                for i in 1:5000
                    node = allocate_node!(tree)
                    if node > 0
                        push!(allocated, node)
                    end
                end
                
                # Free half
                for i in 1:2:length(allocated)
                    free_node!(tree, allocated[i])
                end
                
                # Defragment
                defragment!(manager)
            end
            
            # Clean up
            tree = nothing
            manager = nothing
            CUDA.unsafe_free!
        end
        
        GC.gc(true)
        CUDA.reclaim()
        sleep(0.1)
        
        final = take_memory_snapshot()
        @test report_memory_diff(baseline, final, "Memory Pool Manager")
    end
    
    @testset "Feature Mask Operations" begin
        baseline = take_memory_snapshot()
        
        for cycle in 1:10
            tree = MCTSTreeSoA(CUDA.device())
            
            # Set many features
            CUDA.@allowscalar begin
                tree.next_free_node[1] = 1
                tree.total_nodes[1] = 0
                
                for i in 1:1000
                    node = allocate_node!(tree)
                    if node > 0
                        # Set random features
                        for j in 1:100
                            feature_idx = rand(1:MAX_FEATURES)
                            set_feature!(tree.feature_masks, node, feature_idx)
                        end
                    end
                end
            end
            
            tree = nothing
            CUDA.unsafe_free!
        end
        
        GC.gc(true)
        CUDA.reclaim()
        sleep(0.1)
        
        final = take_memory_snapshot()
        @test report_memory_diff(baseline, final, "Feature Mask Operations")
    end
    
    @testset "Synchronization Structures" begin
        baseline = take_memory_snapshot()
        
        for cycle in 1:10
            # Create and destroy sync structures
            grid_barrier = GridBarrier(Int32(108))
            phase_sync = PhaseSynchronizer(Int32(108))
            tree_sync = TreeSynchronizer(Int32(MAX_NODES))
            
            # Use them briefly
            CUDA.@allowscalar begin
                grid_barrier.generation[1] = cycle
                phase_sync.current_phase[1] = UInt8(cycle % 4)
                tree_sync.operation_counter[1] = cycle * 1000
            end
            
            # Clean up
            grid_barrier = nothing
            phase_sync = nothing  
            tree_sync = nothing
            CUDA.unsafe_free!
        end
        
        GC.gc(true)
        CUDA.reclaim()
        sleep(0.1)
        
        final = take_memory_snapshot()
        @test report_memory_diff(baseline, final, "Synchronization Structures")
    end
    
    @testset "Large Array Allocations" begin
        baseline = take_memory_snapshot()
        
        for cycle in 1:5
            # Allocate large arrays
            arrays = []
            
            # Different sized allocations
            push!(arrays, CUDA.zeros(Float32, 1_000_000))
            push!(arrays, CUDA.ones(Int32, 500_000))
            push!(arrays, CUDA.fill(UInt8(1), 2_000_000))
            
            # Modify arrays
            for arr in arrays
                arr .= arr .+ 1
            end
            
            # Clean up
            arrays = nothing
            CUDA.unsafe_free!
        end
        
        GC.gc(true)
        CUDA.reclaim()
        sleep(0.1)
        
        final = take_memory_snapshot()
        @test report_memory_diff(baseline, final, "Large Array Allocations")
    end
    
    @testset "Kernel Launch Memory" begin
        baseline = take_memory_snapshot()
        
        # Simple kernel
        function dummy_kernel(data)
            tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            if tid <= length(data)
                data[tid] = data[tid] * 2.0f0
            end
            return nothing
        end
        
        for cycle in 1:100
            data = CUDA.rand(Float32, 10000)
            
            # Launch kernel many times
            for i in 1:10
                @cuda threads=256 blocks=40 dummy_kernel(data)
            end
            
            CUDA.synchronize()
            
            # Clean up
            data = nothing
            CUDA.unsafe_free!
        end
        
        GC.gc(true)
        CUDA.reclaim()
        sleep(0.1)
        
        final = take_memory_snapshot()
        @test report_memory_diff(baseline, final, "Kernel Launch Memory")
    end
    
    @testset "Memory Fragmentation Test" begin
        baseline = take_memory_snapshot()
        
        # Create fragmentation pattern
        for cycle in 1:3
            allocations = []
            
            # Allocate many different sized arrays
            for i in 1:100
                size = rand(1000:100000)
                push!(allocations, CUDA.zeros(Float32, size))
            end
            
            # Free every other allocation
            for i in 1:2:length(allocations)
                allocations[i] = nothing
            end
            CUDA.unsafe_free!
            
            # Allocate more in the gaps
            for i in 1:50
                size = rand(500:50000)
                push!(allocations, CUDA.ones(Float32, size))
            end
            
            # Clean up all
            allocations = nothing
            CUDA.unsafe_free!
        end
        
        GC.gc(true)
        CUDA.reclaim()
        sleep(0.1)
        
        final = take_memory_snapshot()
        @test report_memory_diff(baseline, final, "Memory Fragmentation")
    end
    
    # Final memory report
    @testset "Final Memory Status" begin
        GC.gc(true)
        CUDA.reclaim()
        
        final_snapshot = take_memory_snapshot()
        used_memory_mb = final_snapshot.allocated_buffers / (1024 * 1024)
        free_memory_mb = final_snapshot.free_memory / (1024 * 1024)
        total_memory_mb = final_snapshot.total_memory / (1024 * 1024)
        
        @info "Final GPU Memory Status" used_mb=used_memory_mb free_mb=free_memory_mb total_mb=total_memory_mb
        
        # Ensure we're not using too much memory after all tests
        @test used_memory_mb < 1000  # Less than 1GB used after cleanup
    end
end

println("\n✅ Memory leak detection tests completed!")