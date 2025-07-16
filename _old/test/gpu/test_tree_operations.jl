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

# Helper functions
function initialize_root!(tree::MCTSTreeSoA)
    CUDA.@allowscalar begin
        tree.next_free_node[1] = 1
        tree.total_nodes[1] = 0
        tree.max_depth[1] = 0
        tree.free_list_size[1] = 0
        
        # Allocate root
        root_idx = allocate_node!(tree)
        tree.node_ids[root_idx] = 1
        tree.parent_ids[root_idx] = -1
        tree.visit_counts[root_idx] = 1
        tree.node_states[root_idx] = NODE_ACTIVE
    end
end

function expand_node!(tree::MCTSTreeSoA, node_idx::Int32, num_children::Int32)
    CUDA.@allowscalar begin
        if tree.node_states[node_idx] != NODE_ACTIVE
            return
        end
        
        first_child = tree.next_free_node[1]
        
        # Allocate children
        for i in 1:num_children
            child_idx = allocate_node!(tree)
            if child_idx > 0
                tree.parent_ids[child_idx] = node_idx
                tree.node_states[child_idx] = NODE_ACTIVE
                tree.visit_counts[child_idx] = 0
                
                # Copy parent features
                for j in 1:FEATURE_CHUNKS
                    tree.feature_masks[j, child_idx] = tree.feature_masks[j, node_idx]
                end
            end
        end
        
        tree.first_child_idx[node_idx] = first_child
        tree.num_children[node_idx] = num_children
        tree.node_states[node_idx] = NODE_EXPANDED
    end
end

function verify_tree_consistency(tree::MCTSTreeSoA)
    CUDA.@allowscalar begin
        total_nodes = tree.total_nodes[1]
        
        for i in 1:total_nodes
            if tree.node_states[i] == NODE_ACTIVE || tree.node_states[i] == NODE_EXPANDED
                # Check parent relationship
                parent_idx = tree.parent_ids[i]
                if parent_idx > 0
                    @test parent_idx < i  # Parent should come before child
                    @test tree.node_states[parent_idx] == NODE_EXPANDED
                end
                
                # Check children
                if tree.node_states[i] == NODE_EXPANDED
                    num_children = tree.num_children[i]
                    first_child = tree.first_child_idx[i]
                    
                    @test num_children > 0
                    @test first_child > i  # Children come after parent
                    
                    # Verify all children
                    for j in 0:(num_children-1)
                        child_idx = first_child + j
                        @test tree.parent_ids[child_idx] == i
                    end
                end
            end
        end
    end
end

@testset "MCTS Tree Operations Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU tree operation tests"
        return
    end
    
    @testset "Tree Structure Initialization" begin
        tree = MCTSTreeSoA(CUDA.device())
        
        # Test initial state
        CUDA.@allowscalar begin
            @test tree.next_free_node[1] == 1
            @test tree.total_nodes[1] == 0
            @test tree.max_depth[1] == 0
            @test tree.free_list_size[1] == 0
        end
        
        # Initialize with root
        initialize_root!(tree)
        
        CUDA.@allowscalar begin
            @test tree.next_free_node[1] == 2
            @test tree.total_nodes[1] == 1
            @test tree.node_ids[1] == 1
            @test tree.node_states[1] == NODE_ACTIVE
            @test tree.parent_ids[1] == -1
            @test tree.visit_counts[1] == 1
        end
    end
    
    @testset "Concurrent Node Allocation" begin
        tree = MCTSTreeSoA(CUDA.device())
        initialize_root!(tree)
        
        # Test kernel for concurrent allocation
        function concurrent_allocation_kernel!(tree, results, num_threads)
            tid = threadIdx().x
            
            # Each thread tries to allocate a node
            node_idx = allocate_node!(tree)
            
            # Store result
            if tid <= length(results)
                results[tid] = node_idx
            end
            
            return nothing
        end
        
        # Launch with 32 threads
        num_threads = 32
        results = CUDA.fill(Int32(-1), num_threads)
        
        @cuda threads=num_threads concurrent_allocation_kernel!(tree, results, num_threads)
        
        # Verify results
        results_host = Array(results)
        valid_allocations = filter(x -> x > 0, results_host)
        
        # All allocations should be unique
        @test length(unique(valid_allocations)) == length(valid_allocations)
        
        # Should have allocated correct number of nodes
        CUDA.@allowscalar @test tree.total_nodes[1] == 1 + length(valid_allocations)
    end
    
    @testset "Tree Expansion Operations" begin
        tree = MCTSTreeSoA(CUDA.device())
        initialize_root!(tree)
        
        # Expand root node
        num_children = 5
        expand_node!(tree, Int32(1), Int32(num_children))
        
        CUDA.@allowscalar begin
            @test tree.num_children[1] == num_children
            @test tree.first_child_idx[1] == 2
            @test tree.node_states[1] == NODE_EXPANDED
            
            # Check children
            for i in 1:num_children
                child_idx = 1 + i
                @test tree.parent_ids[child_idx] == 1
                @test tree.node_states[child_idx] == NODE_ACTIVE
            end
        end
    end
    
    @testset "Feature Mask Operations" begin
        tree = MCTSTreeSoA(CUDA.device())
        initialize_root!(tree)
        
        # Test setting and getting features
        node_idx = Int32(1)
        test_features = [1, 50, 100, 500, 1000, 4999]
        
        for feature in test_features
            set_feature!(tree.feature_masks, node_idx, feature)
        end
        
        # Verify features are set
        for feature in test_features
            @test has_feature(tree.feature_masks, node_idx, feature)
        end
        
        # Verify other features are not set
        @test !has_feature(tree.feature_masks, node_idx, 25)
        @test !has_feature(tree.feature_masks, node_idx, 2000)
        
        # Test feature clearing
        clear_feature!(tree.feature_masks, node_idx, 50)
        @test !has_feature(tree.feature_masks, node_idx, 50)
        @test has_feature(tree.feature_masks, node_idx, 100)  # Others still set
    end
    
    @testset "Memory Pool Management" begin
        tree = MCTSTreeSoA(CUDA.device())
        memory_manager = MemoryPoolManager(tree)
        
        # Allocate many nodes
        allocated_nodes = Int32[]
        for i in 1:100
            node = allocate_node!(tree)
            if node > 0
                push!(allocated_nodes, node)
            end
        end
        
        @test length(allocated_nodes) == 100
        CUDA.@allowscalar @test tree.total_nodes[1] == 100
        
        # Free some nodes
        nodes_to_free = allocated_nodes[1:50]
        for node in nodes_to_free
            free_node!(tree, node)
        end
        
        CUDA.@allowscalar @test tree.free_list_size[1] == 50
        
        # Reallocate - should reuse freed nodes
        new_allocations = Int32[]
        for i in 1:25
            node = allocate_node!(tree)
            push!(new_allocations, node)
        end
        
        # Should have reused some freed nodes
        @test any(n in nodes_to_free for n in new_allocations)
        CUDA.@allowscalar @test tree.free_list_size[1] == 25
    end
    
    @testset "Tree Synchronization" begin
        tree = MCTSTreeSoA(CUDA.device())
        tree_sync = TreeSynchronizer(Int32(MAX_NODES))
        
        # Test read/write lock functionality
        function test_sync_kernel!(tree_sync, results)
            tid = threadIdx().x
            node_idx = Int32(1)  # All threads access same node
            
            if tid % 2 == 0
                # Even threads read
                acquire_read_lock!(tree_sync.read_locks, node_idx)
                results[tid] = 1  # Mark successful read
                release_read_lock!(tree_sync.read_locks, node_idx)
            else
                # Odd threads write
                acquire_write_lock!(tree_sync.read_locks, tree_sync.write_locks, node_idx)
                results[tid] = 2  # Mark successful write
                release_write_lock!(tree_sync.write_locks, node_idx)
            end
            
            return nothing
        end
        
        results = CUDA.zeros(Int32, 32)
        @cuda threads=32 test_sync_kernel!(tree_sync, results)
        
        results_host = Array(results)
        # All threads should have completed their operations
        @test all(r > 0 for r in results_host)
    end
    
    @testset "Tree Consistency Checks" begin
        tree = MCTSTreeSoA(CUDA.device())
        initialize_root!(tree)
        
        # Build a small tree
        expand_node!(tree, Int32(1), Int32(3))  # Root has 3 children
        expand_node!(tree, Int32(2), Int32(2))  # First child has 2 children
        
        # Check parent-child consistency
        CUDA.@allowscalar begin
            # Root's children
            @test tree.parent_ids[2] == 1
            @test tree.parent_ids[3] == 1
            @test tree.parent_ids[4] == 1
            
            # Grandchildren
            @test tree.parent_ids[5] == 2
            @test tree.parent_ids[6] == 2
            
            # Check first_child pointers
            @test tree.first_child_idx[1] == 2
            @test tree.first_child_idx[2] == 5
            
            # Check num_children
            @test tree.num_children[1] == 3
            @test tree.num_children[2] == 2
        end
    end
    
    @testset "Stress Test - Large Tree Construction" begin
        tree = MCTSTreeSoA(CUDA.device())
        initialize_root!(tree)
        
        # Build a tree with many nodes
        target_nodes = 10000
        nodes_created = 1  # Start with root
        
        current_level = [Int32(1)]
        next_level = Int32[]
        
        while nodes_created < target_nodes && !isempty(current_level)
            for node_idx in current_level
                if nodes_created >= target_nodes
                    break
                end
                
                # Expand with random number of children
                num_children = rand(2:5)
                first_child = CUDA.@allowscalar tree.next_free_node[1]
                
                if first_child + num_children - 1 <= MAX_NODES
                    expand_node!(tree, node_idx, Int32(num_children))
                    
                    for i in 0:(num_children-1)
                        push!(next_level, first_child + i)
                    end
                    
                    nodes_created += num_children
                end
            end
            
            current_level = next_level
            next_level = Int32[]
        end
        
        # Verify tree statistics
        CUDA.@allowscalar begin
            @test tree.total_nodes[1] >= target_nodes / 2  # At least half target
            @test tree.max_depth[1] > 0
        end
        
        # Verify all parent-child relationships
        verify_tree_consistency(tree)
    end
    
    @testset "Memory Defragmentation" begin
        tree = MCTSTreeSoA(CUDA.device())
        memory_manager = MemoryPoolManager(tree, defrag_threshold = 0.3f0)
        
        # Create fragmented memory
        allocated = Int32[]
        for i in 1:100
            node = allocate_node!(tree)
            push!(allocated, node)
        end
        
        # Free every other node
        for i in 1:2:100
            free_node!(tree, allocated[i])
        end
        
        # Check fragmentation
        @test should_defragment(memory_manager)
        
        # Defragment
        original_total = CUDA.@allowscalar tree.total_nodes[1]
        new_total = defragment!(memory_manager)
        
        @test new_total < original_total
        CUDA.@allowscalar @test tree.free_list_size[1] == 0
        
        # Verify remaining nodes are still valid
        verify_tree_consistency(tree)
    end
    
    @testset "UCB Score Calculation" begin
        # Test UCB1 score calculation
        total_score = 10.0f0
        visits = Int32(5)
        parent_visits = Int32(20)
        c = 1.414f0
        prior = 0.1f0
        
        score = ucb1_score(total_score, visits, parent_visits, c, prior)
        
        # Manually calculate expected score
        avg_score = total_score / visits
        exploration = c * sqrt(log(Float32(parent_visits)) / visits)
        expected = avg_score + exploration + prior
        
        @test score ≈ expected
        
        # Test edge cases
        @test ucb1_score(0.0f0, Int32(0), Int32(1), c, 0.0f0) == Inf32
        @test ucb1_score(0.0f0, Int32(1), Int32(1), c, 0.0f0) < Inf32
    end
end

println("\n✅ Tree operations tests completed!")