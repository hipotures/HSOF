using Test
using CUDA
using Random

# Include modules  
include("../../src/gpu/kernels/mcts_types.jl")
include("../../src/gpu/kernels/memory_pool.jl")
include("../../src/gpu/kernels/synchronization.jl")

using .MCTSTypes
using .MemoryPool
using .Synchronization

@testset "Basic MCTS Tree Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU tree tests"
        return
    end
    
    @testset "Tree Structure Basics" begin
        tree = MCTSTreeSoA(CUDA.device())
        
        # Test initial state
        CUDA.@allowscalar begin
            @test tree.next_free_node[1] == 1
            @test tree.total_nodes[1] == 0
            @test tree.max_depth[1] == 0
            @test tree.free_list_size[1] == 0
        end
    end
    
    @testset "Feature Mask Operations" begin
        tree = MCTSTreeSoA(CUDA.device())
        
        # Test feature operations in GPU kernel
        function test_features_kernel!(tree)
            tid = threadIdx().x
            node_idx = Int32(tid)
            
            if tid <= 10
                # Set some features
                set_feature!(tree.feature_masks, node_idx, tid * 10)
                set_feature!(tree.feature_masks, node_idx, tid * 20)
                
                # Check if set correctly
                has1 = has_feature(tree.feature_masks, node_idx, tid * 10)
                has2 = has_feature(tree.feature_masks, node_idx, tid * 20)
                has3 = has_feature(tree.feature_masks, node_idx, tid * 30)
                
                # Store results
                tree.visit_counts[tid] = Int32(has1 && has2 && !has3)
            end
            
            return nothing
        end
        
        @cuda threads=32 test_features_kernel!(tree)
        
        # Verify results
        CUDA.@allowscalar begin
            for i in 1:10
                @test tree.visit_counts[i] == 1
            end
        end
    end
    
    @testset "UCB Score Calculation" begin
        # Test UCB1 score calculation
        total_score = 10.0f0
        visits = Int32(5)
        parent_visits = Int32(20)
        c = 1.414f0
        prior = 0.1f0
        
        score = ucb1_score(total_score, visits, parent_visits, c, prior)
        
        # Check score is reasonable
        @test score > 0
        @test score < 100
        
        # Test edge cases
        @test ucb1_score(0.0f0, Int32(0), Int32(1), c, 0.0f0) == Inf32
        @test ucb1_score(0.0f0, Int32(1), Int32(1), c, 0.0f0) < Inf32
    end
    
    @testset "GPU Node Allocation" begin
        tree = MCTSTreeSoA(CUDA.device())
        
        # Test allocation in GPU kernel
        function allocation_kernel!(tree, results)
            tid = threadIdx().x
            
            if tid <= length(results)
                node_idx = allocate_node!(tree)
                results[tid] = node_idx
            end
            
            return nothing
        end
        
        num_threads = 32
        results = CUDA.fill(Int32(-1), num_threads)
        
        @cuda threads=num_threads allocation_kernel!(tree, results)
        
        # Verify results
        results_host = Array(results)
        valid_allocations = filter(x -> x > 0, results_host)
        
        # All allocations should be unique
        @test length(unique(valid_allocations)) == length(valid_allocations)
        
        # Check tree state
        CUDA.@allowscalar begin
            @test tree.total_nodes[1] == length(valid_allocations)
        end
    end
    
    @testset "Tree Synchronization Primitives" begin
        tree_sync = TreeSynchronizer(Int32(100))
        
        # Test lock operations in kernel
        function test_locks_kernel!(tree_sync, results)
            tid = threadIdx().x
            node_idx = Int32(1)
            
            if tid <= 32
                if tid % 2 == 0
                    # Even threads read
                    acquire_read_lock!(tree_sync.read_locks, node_idx)
                    results[tid] = 1
                    release_read_lock!(tree_sync.read_locks, node_idx)
                else
                    # Odd threads write
                    acquire_write_lock!(tree_sync.read_locks, tree_sync.write_locks, node_idx)
                    results[tid] = 2
                    release_write_lock!(tree_sync.write_locks, node_idx)
                end
            end
            
            return nothing
        end
        
        results = CUDA.zeros(Int32, 32)
        @cuda threads=32 test_locks_kernel!(tree_sync, results)
        
        results_host = Array(results)
        # All threads should have completed
        @test all(r > 0 for r in results_host)
    end
    
    @testset "Memory Pool Free List" begin
        tree = MCTSTreeSoA(CUDA.device())
        
        # Test free/allocate cycle in kernel
        function free_alloc_kernel!(tree)
            tid = threadIdx().x
            
            # Phase 1: Allocate
            sync_threads()
            
            if tid <= 50
                node_idx = allocate_node!(tree)
                tree.visit_counts[tid] = node_idx
            end
            
            sync_threads()
            
            # Phase 2: Free half
            if tid <= 25
                node_to_free = tree.visit_counts[tid]
                if node_to_free > 0
                    free_node!(tree, node_to_free)
                end
            end
            
            sync_threads()
            
            # Phase 3: Reallocate
            if tid > 50 && tid <= 75
                new_node = allocate_node!(tree)
                tree.visit_counts[tid] = new_node
            end
            
            return nothing
        end
        
        @cuda threads=128 free_alloc_kernel!(tree)
        
        # Verify free list was used
        CUDA.@allowscalar begin
            # Some nodes should have been reused
            reused_nodes = 0
            for i in 51:75
                node = tree.visit_counts[i]
                if node > 0 && node <= 25
                    reused_nodes += 1
                end
            end
            @test reused_nodes > 0
        end
    end
    
    @testset "Tree Depth Tracking" begin
        tree = MCTSTreeSoA(CUDA.device())
        
        # Build tree and track depth
        function build_tree_kernel!(tree)
            tid = threadIdx().x
            
            if tid == 1
                # Thread 1 builds a chain
                node1 = allocate_node!(tree)
                tree.parent_ids[node1] = -1
                tree.node_states[node1] = NODE_ACTIVE
                
                node2 = allocate_node!(tree)
                tree.parent_ids[node2] = node1
                tree.node_states[node2] = NODE_ACTIVE
                
                node3 = allocate_node!(tree)
                tree.parent_ids[node3] = node2
                tree.node_states[node3] = NODE_ACTIVE
                
                # Update max depth
                CUDA.atomic_max!(pointer(tree.max_depth), Int32(3))
            end
            
            return nothing
        end
        
        @cuda threads=32 build_tree_kernel!(tree)
        
        CUDA.@allowscalar begin
            @test tree.max_depth[1] == 3
            @test tree.total_nodes[1] == 3
        end
    end
    
    @testset "Grid Barrier Synchronization" begin
        num_blocks = Int32(4)
        barrier = GridBarrier(num_blocks)
        
        # Test barrier synchronization
        function barrier_kernel!(barrier, counter, num_blocks)
            tid = threadIdx().x
            bid = blockIdx().x
            
            # Phase 1: Increment counter
            if tid == 1
                CUDA.atomic_add!(pointer(counter), Int32(1))
            end
            
            # Synchronize all blocks
            grid_barrier_sync!(
                barrier.arrival_count,
                barrier.release_count,
                barrier.generation,
                barrier.target_count,
                bid, tid
            )
            
            # Phase 2: Check all blocks reached barrier
            if tid == 1 && bid == 1
                counter[2] = counter[1]  # Store phase 1 result
                counter[1] = 0  # Reset for phase 2
            end
            
            # Another barrier
            grid_barrier_sync!(
                barrier.arrival_count,
                barrier.release_count,
                barrier.generation,
                barrier.target_count,
                bid, tid
            )
            
            # Phase 3: Increment again
            if tid == 1
                CUDA.atomic_add!(pointer(counter), Int32(1))
            end
            
            return nothing
        end
        
        counter = CUDA.zeros(Int32, 2)
        @cuda threads=32 blocks=num_blocks barrier_kernel!(barrier, counter, num_blocks)
        
        counter_host = Array(counter)
        @test counter_host[2] == num_blocks  # All blocks participated in phase 1
        @test counter_host[1] == num_blocks  # All blocks participated in phase 3
    end
end

println("\n✅ Basic tree tests completed!")