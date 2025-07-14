using Test
using CUDA
using Random
using Statistics
using BenchmarkTools

# Include modules  
include("../../src/gpu/kernels/mcts_types.jl")
include("../../src/gpu/kernels/memory_pool.jl")
include("../../src/gpu/kernels/synchronization.jl")
include("../../src/gpu/kernels/persistent_kernel.jl")

using .MCTSTypes
using .MemoryPool
using .Synchronization
using .PersistentKernel

@testset "MCTS Tree Stress Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU tree stress tests"
        return
    end
    
    @testset "Concurrent Read-Write Stress" begin
        tree = MCTSTreeSoA(CUDA.device())
        tree_sync = TreeSynchronizer(Int32(MAX_NODES))
        
        # Initialize tree with some nodes
        CUDA.@allowscalar begin
            tree.next_free_node[1] = 1
            tree.total_nodes[1] = 0
            
            for i in 1:100
                node = allocate_node!(tree)
                tree.node_states[node] = NODE_ACTIVE
                tree.visit_counts[node] = 0
                tree.total_scores[node] = 0.0f0
            end
        end
        
        # Stress test kernel - many threads reading/writing
        function stress_kernel!(tree, tree_sync, iterations)
            tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            
            # Random operations on random nodes
            rng_state = UInt32(tid)
            
            for iter in 1:iterations
                # Simple LCG for thread-local random
                rng_state = rng_state * 1664525 + 1013904223
                operation = rng_state % 3
                
                rng_state = rng_state * 1664525 + 1013904223
                node_idx = Int32((rng_state % 100) + 1)
                
                if operation == 0
                    # Read operation
                    acquire_read_lock!(tree_sync.read_locks, node_idx)
                    score = tree.total_scores[node_idx]
                    visits = tree.visit_counts[node_idx]
                    release_read_lock!(tree_sync.read_locks, node_idx)
                    
                elseif operation == 1
                    # Write operation
                    acquire_write_lock!(tree_sync.read_locks, tree_sync.write_locks, node_idx)
                    tree.total_scores[node_idx] += 0.1f0
                    CUDA.atomic_add!(pointer(tree.visit_counts, node_idx), Int32(1))
                    release_write_lock!(tree_sync.write_locks, node_idx)
                    
                else
                    # Update operation
                    acquire_write_lock!(tree_sync.read_locks, tree_sync.write_locks, node_idx)
                    old_score = tree.total_scores[node_idx]
                    tree.total_scores[node_idx] = old_score * 0.9f0
                    release_write_lock!(tree_sync.write_locks, node_idx)
                end
            end
            
            return nothing
        end
        
        # Run stress test with many threads
        threads = 256
        blocks = 32
        iterations = 100
        
        @cuda threads=threads blocks=blocks stress_kernel!(tree, tree_sync, iterations)
        CUDA.synchronize()
        
        # Verify tree is still consistent
        CUDA.@allowscalar begin
            total_visits = sum(tree.visit_counts[1:100])
            @test total_visits > 0  # Some writes should have succeeded
            
            # Check no corruption
            for i in 1:100
                @test tree.visit_counts[i] >= 0
                @test !isnan(tree.total_scores[i])
                @test !isinf(tree.total_scores[i])
            end
        end
    end
    
    @testset "Memory Pool Exhaustion Test" begin
        tree = MCTSTreeSoA(CUDA.device())
        
        # Allocate until exhaustion
        allocated_count = 0
        last_valid = Int32(0)
        
        CUDA.@allowscalar begin
            tree.next_free_node[1] = 1
            tree.total_nodes[1] = 0
            
            while true
                node = allocate_node!(tree)
                if node <= 0
                    break
                end
                allocated_count += 1
                last_valid = node
            end
        end
        
        @test allocated_count == MAX_NODES
        @test last_valid == MAX_NODES
        
        # Verify pool is exhausted
        CUDA.@allowscalar begin
            @test tree.next_free_node[1] > MAX_NODES
            @test allocate_node!(tree) == -1  # Should fail
        end
        
        # Free some nodes and verify reallocation works
        CUDA.@allowscalar begin
            for i in 1:100
                free_node!(tree, Int32(i))
            end
            
            @test tree.free_list_size[1] == 100
            
            # Should be able to allocate again
            new_node = allocate_node!(tree)
            @test new_node > 0
            @test new_node <= 100  # Should reuse freed node
        end
    end
    
    @testset "Rapid Expansion-Contraction Cycles" begin
        tree = MCTSTreeSoA(CUDA.device())
        memory_manager = MemoryPoolManager(tree)
        
        # Perform rapid expansion and contraction
        for cycle in 1:10
            allocated = Int32[]
            
            # Expansion phase
            CUDA.@allowscalar begin
                tree.next_free_node[1] = 1
                tree.total_nodes[1] = 0
                tree.free_list_size[1] = 0
                
                for i in 1:1000
                    node = allocate_node!(tree)
                    if node > 0
                        push!(allocated, node)
                        tree.node_states[node] = NODE_ACTIVE
                    end
                end
            end
            
            # Contraction phase - free 80% of nodes
            to_free = Int(length(allocated) * 0.8)
            shuffle!(allocated)
            
            CUDA.@allowscalar begin
                for i in 1:to_free
                    free_node!(tree, allocated[i])
                end
            end
            
            # Verify memory state
            CUDA.@allowscalar begin
                @test tree.free_list_size[1] == to_free
                @test tree.total_nodes[1] == length(allocated)
            end
            
            # Defragment if needed
            if should_defragment(memory_manager)
                defragment!(memory_manager)
            end
        end
    end
    
    @testset "Parallel Tree Traversal Stress" begin
        tree = MCTSTreeSoA(CUDA.device())
        config = PersistentKernelConfig()
        
        # Build a deep tree
        CUDA.@allowscalar begin
            tree.next_free_node[1] = 1
            tree.total_nodes[1] = 0
            
            # Create root
            root = allocate_node!(tree)
            tree.node_states[root] = NODE_EXPANDED
            tree.parent_ids[root] = -1
            tree.visit_counts[root] = 1000
            tree.num_children[root] = 4
            tree.first_child_idx[root] = 2
            
            # Create tree levels
            current_idx = Int32(2)
            for level in 1:5
                nodes_in_level = 4^level
                for i in 1:nodes_in_level
                    if current_idx > MAX_NODES - 10
                        break
                    end
                    
                    node = allocate_node!(tree)
                    if node > 0
                        parent_idx = Int32(1 + (node - 2) ÷ 4)
                        tree.parent_ids[node] = parent_idx
                        tree.node_states[node] = level < 5 ? NODE_EXPANDED : NODE_ACTIVE
                        tree.visit_counts[node] = 1000 ÷ (level + 1)
                        tree.total_scores[node] = rand(Float32)
                        
                        if level < 5
                            tree.num_children[node] = 4
                            tree.first_child_idx[node] = current_idx + nodes_in_level + (i-1)*4
                        end
                    end
                    current_idx += 1
                end
            end
        end
        
        # Parallel traversal kernel
        function traverse_kernel!(tree, config, paths, path_lengths)
            tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            
            if tid > length(path_lengths)
                return
            end
            
            # Start from root
            current = Int32(1)
            depth = 0
            max_depth = 20
            
            # Traverse down selecting best UCB child
            while depth < max_depth
                num_children = tree.num_children[current]
                if num_children == 0 || tree.node_states[current] != NODE_EXPANDED
                    break
                end
                
                # Find best child
                best_child = select_best_child_ucb1(tree, current, config)
                if best_child <= 0
                    break
                end
                
                # Record path
                if depth < 20
                    paths[depth + 1, tid] = best_child
                end
                
                current = best_child
                depth += 1
            end
            
            path_lengths[tid] = depth
            
            return nothing
        end
        
        # Run parallel traversal
        num_traversals = 1024
        paths = CUDA.zeros(Int32, 20, num_traversals)
        path_lengths = CUDA.zeros(Int32, num_traversals)
        
        threads = 256
        blocks = cld(num_traversals, threads)
        
        @cuda threads=threads blocks=blocks traverse_kernel!(tree, config, paths, path_lengths)
        
        # Verify traversals
        path_lengths_host = Array(path_lengths)
        @test all(l > 0 for l in path_lengths_host)
        @test maximum(path_lengths_host) <= 6  # Tree depth is 5
    end
    
    @testset "Performance Benchmarks" begin
        tree = MCTSTreeSoA(CUDA.device())
        
        # Benchmark node allocation
        allocation_time = CUDA.@elapsed begin
            CUDA.@sync begin
                @cuda threads=256 blocks=40 allocation_benchmark_kernel!(tree, 1000)
            end
        end
        
        nodes_allocated = CUDA.@allowscalar tree.total_nodes[1]
        allocation_rate = nodes_allocated / allocation_time
        
        @info "Node allocation performance" rate_per_second=allocation_rate total_allocated=nodes_allocated
        @test allocation_rate > 1_000_000  # Should allocate >1M nodes/sec
        
        # Benchmark tree traversal
        config = PersistentKernelConfig()
        build_benchmark_tree!(tree)
        
        traversal_time = CUDA.@elapsed begin
            CUDA.@sync begin
                @cuda threads=256 blocks=40 traversal_benchmark_kernel!(tree, config, 100)
            end
        end
        
        traversals_completed = 256 * 40 * 100
        traversal_rate = traversals_completed / traversal_time
        
        @info "Tree traversal performance" rate_per_second=traversal_rate
        @test traversal_rate > 100_000  # Should complete >100K traversals/sec
    end
end

# Benchmark kernels
function allocation_benchmark_kernel!(tree, iterations)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    for i in 1:iterations
        node = allocate_node!(tree)
        if node > 0
            tree.node_states[node] = NODE_ACTIVE
        end
    end
    
    return nothing
end

function traversal_benchmark_kernel!(tree, config, iterations)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    for i in 1:iterations
        current = Int32(1)
        depth = 0
        
        while depth < 10
            num_children = tree.num_children[current]
            if num_children == 0
                break
            end
            
            # Simple child selection
            child_idx = tree.first_child_idx[current] + Int32(tid % num_children)
            current = child_idx
            depth += 1
        end
    end
    
    return nothing
end

function build_benchmark_tree!(tree)
    CUDA.@allowscalar begin
        tree.next_free_node[1] = 1
        tree.total_nodes[1] = 0
        
        # Build balanced tree
        queue = Int32[1]
        
        while !isempty(queue) && tree.total_nodes[1] < 10000
            node_idx = popfirst!(queue)
            
            if node_idx == 1
                allocate_node!(tree)
                tree.node_states[1] = NODE_EXPANDED
                tree.parent_ids[1] = -1
            end
            
            # Add 4 children
            first_child = tree.next_free_node[1]
            for i in 1:4
                child = allocate_node!(tree)
                if child > 0
                    tree.parent_ids[child] = node_idx
                    tree.node_states[child] = NODE_EXPANDED
                    tree.visit_counts[child] = 100
                    tree.total_scores[child] = rand(Float32) * 100
                    push!(queue, child)
                end
            end
            
            tree.num_children[node_idx] = 4
            tree.first_child_idx[node_idx] = first_child
        end
    end
end

println("\n✅ Tree stress tests completed!")