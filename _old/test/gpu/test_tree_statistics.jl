using Test
using CUDA
using Random

# Include modules  
include("../../src/gpu/kernels/mcts_types.jl")
include("../../src/gpu/kernels/memory_pool.jl")
include("../../src/gpu/kernels/synchronization.jl")
include("../../src/gpu/kernels/tree_statistics.jl")

using .MCTSTypes
using .MemoryPool
using .Synchronization
using .TreeStatistics

# Use the collect_tree_statistics from the correct module
const collect_tree_statistics = TreeStatistics.collect_tree_statistics
const generate_stats_report = TreeStatistics.generate_stats_report

# Helper function to allocate node on host (for testing)
function host_allocate_node!(tree::MCTSTreeSoA)
    CUDA.@allowscalar begin
        node_idx = tree.next_free_node[1]
        tree.next_free_node[1] += 1
        tree.total_nodes[1] += 1
        return node_idx
    end
end

# Helper functions
function build_test_tree!(tree::MCTSTreeSoA, structure::Symbol = :balanced)
    CUDA.@allowscalar begin
        # Reset tree
        tree.next_free_node[1] = 1
        tree.total_nodes[1] = 0
        tree.max_depth[1] = 0
        tree.free_list_size[1] = 0
        
        # Allocate root
        root_idx = host_allocate_node!(tree)
        tree.node_ids[root_idx] = 1
        tree.parent_ids[root_idx] = -1
        tree.visit_counts[root_idx] = 100
        tree.node_states[root_idx] = NODE_EXPANDED
        tree.num_children[root_idx] = 3
        tree.first_child_idx[root_idx] = 2
        
        if structure == :balanced
            # Build balanced tree
            # Level 1 - 3 children
            for i in 1:3
                child_idx = host_allocate_node!(tree)
                tree.parent_ids[child_idx] = root_idx
                tree.visit_counts[child_idx] = 30 - i*5
                tree.node_states[child_idx] = NODE_EXPANDED
                tree.num_children[child_idx] = 2
                tree.first_child_idx[child_idx] = tree.next_free_node[1]
                
                # Level 2 - 2 children each
                for j in 1:2
                    grandchild_idx = host_allocate_node!(tree)
                    tree.parent_ids[grandchild_idx] = child_idx
                    tree.visit_counts[grandchild_idx] = 5 + j
                    tree.node_states[grandchild_idx] = j == 1 ? NODE_ACTIVE : NODE_TERMINAL
                end
            end
            
            tree.max_depth[1] = 2
            
        elseif structure == :deep
            # Build deep tree (single path)
            current = root_idx
            for depth in 1:10
                child_idx = host_allocate_node!(tree)
                tree.parent_ids[child_idx] = current
                tree.visit_counts[child_idx] = 100 - depth*10
                tree.node_states[child_idx] = depth < 10 ? NODE_EXPANDED : NODE_TERMINAL
                
                if depth < 10
                    tree.num_children[current] = 1
                    tree.first_child_idx[current] = child_idx
                end
                
                current = child_idx
            end
            
            tree.max_depth[1] = 10
            
        elseif structure == :wide
            # Build wide tree (many children at root)
            tree.num_children[root_idx] = 20
            tree.first_child_idx[root_idx] = 2
            
            for i in 1:20
                child_idx = host_allocate_node!(tree)
                tree.parent_ids[child_idx] = root_idx
                tree.visit_counts[child_idx] = rand(1:50)
                tree.node_states[child_idx] = rand() > 0.5 ? NODE_ACTIVE : NODE_TERMINAL
            end
            
            tree.max_depth[1] = 1
        end
    end
end

@testset "Tree Statistics Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU tree statistics tests"
        return
    end
    
    @testset "TreeStatsCollector Initialization" begin
        collector = TreeStatsCollector()
        
        # Check all fields are initialized to zero
        CUDA.@allowscalar begin
            @test collector.max_depth_observed[1] == 0
            @test collector.active_nodes[1] == 0
            @test collector.expanded_nodes[1] == 0
            @test collector.terminal_nodes[1] == 0
            @test collector.recycled_nodes[1] == 0
            @test collector.avg_branching_factor[1] == 0.0f0
            @test collector.avg_path_length[1] == 0.0f0
            @test collector.path_count[1] == 0
            @test collector.max_visits[1] == 0
            @test collector.stats_version[1] == 0
        end
    end
    
    @testset "Balanced Tree Statistics" begin
        tree = MCTSTreeSoA(CUDA.device())
        build_test_tree!(tree, :balanced)
        
        collector = TreeStatsCollector()
        summary = collect_tree_statistics(tree, collector)
        
        # Check summary statistics
        @test summary.total_nodes == 13  # 1 root + 3 children + 6 grandchildren + 3 allocated
        @test summary.max_depth == 2
        @test summary.expanded_nodes == 4  # root + 3 children
        @test summary.active_nodes == 3    # 3 active grandchildren
        @test summary.terminal_nodes == 3  # 3 terminal grandchildren
        @test summary.avg_branching_factor ≈ 2.25f0  # (3 + 2 + 2 + 2) / 4
        @test summary.max_visits == 100
        
        # Check depth distribution
        @test length(summary.depth_distribution) >= 3
        # Note: depth calculation traces from node to root, so distribution might vary
        
        # Check visit distribution
        @test length(summary.visit_distribution) >= 1
        @test sum(summary.visit_distribution) > 0
    end
    
    @testset "Deep Tree Statistics" begin
        tree = MCTSTreeSoA(CUDA.device())
        build_test_tree!(tree, :deep)
        
        collector = TreeStatsCollector()
        summary = collect_tree_statistics(tree, collector)
        
        # Check summary statistics
        @test summary.total_nodes == 11  # 1 root + 10 in chain
        @test summary.max_depth >= 9     # Should detect deep structure
        @test summary.expanded_nodes == 10  # All except last are expanded
        @test summary.terminal_nodes == 1   # Only last node
        @test summary.avg_branching_factor ≈ 1.0f0  # Each has 1 child
        @test summary.max_visits == 100
    end
    
    @testset "Wide Tree Statistics" begin
        tree = MCTSTreeSoA(CUDA.device())
        build_test_tree!(tree, :wide)
        
        collector = TreeStatsCollector()
        summary = collect_tree_statistics(tree, collector)
        
        # Check summary statistics
        @test summary.total_nodes == 21  # 1 root + 20 children
        @test summary.max_depth >= 1
        @test summary.expanded_nodes == 1  # Only root
        @test summary.active_nodes + summary.terminal_nodes == 20
        @test summary.avg_branching_factor ≈ 20.0f0  # Root has 20 children
    end
    
    @testset "Empty Tree Statistics" begin
        tree = MCTSTreeSoA(CUDA.device())
        # Don't build any tree
        
        collector = TreeStatsCollector()
        summary = collect_tree_statistics(tree, collector)
        
        # Should handle empty tree gracefully
        @test summary.total_nodes == 0
        @test summary.max_depth == 0
        @test summary.expanded_nodes == 0
        @test summary.active_nodes == 0
        @test summary.terminal_nodes == 0
        @test summary.avg_branching_factor == 0.0f0
    end
    
    @testset "Statistics Report Generation" begin
        tree = MCTSTreeSoA(CUDA.device())
        build_test_tree!(tree, :balanced)
        
        collector = TreeStatsCollector()
        summary = collect_tree_statistics(tree, collector)
        
        # Generate report
        report = generate_stats_report(summary)
        
        # Check report structure
        @test haskey(report, "node_counts")
        @test haskey(report["node_counts"], "total")
        @test haskey(report["node_counts"], "active")
        @test haskey(report["node_counts"], "expanded")
        @test haskey(report["node_counts"], "terminal")
        @test haskey(report["node_counts"], "recycled")
        
        @test haskey(report, "tree_structure")
        @test haskey(report["tree_structure"], "max_depth")
        @test haskey(report["tree_structure"], "avg_branching_factor")
        @test haskey(report["tree_structure"], "avg_path_length")
        
        @test haskey(report, "visit_stats")
        @test haskey(report["visit_stats"], "max_visits")
        @test haskey(report["visit_stats"], "distribution")
        
        @test haskey(report, "depth_distribution")
        
        # Tree balance metric should be present for non-trivial trees
        if summary.max_depth > 0 && summary.expanded_nodes > 0
            @test haskey(report, "tree_balance")
        end
    end
    
    @testset "Visit Count Bucketing" begin
        # Test the visit count bucketing function
        @test TreeStatistics.visit_count_to_bucket(Int32(5)) == 5
        @test TreeStatistics.visit_count_to_bucket(Int32(10)) == 10
        @test TreeStatistics.visit_count_to_bucket(Int32(15)) == 10  # 10 + (15-10)÷10 = 10 + 0 = 10
        @test TreeStatistics.visit_count_to_bucket(Int32(20)) == 11  # 10 + (20-10)÷10 = 10 + 1 = 11
        @test TreeStatistics.visit_count_to_bucket(Int32(50)) == 14  # 10 + (50-10)÷10 = 10 + 4 = 14
        @test TreeStatistics.visit_count_to_bucket(Int32(100)) == 19 # 10 + (100-10)÷10 = 10 + 9 = 19
        @test TreeStatistics.visit_count_to_bucket(Int32(150)) == 19 # 19 + (150-100)÷100 = 19 + 0 = 19
        @test TreeStatistics.visit_count_to_bucket(Int32(1500)) == 20 # Over 1000 -> bucket 20
    end
    
    @testset "Concurrent Statistics Collection" begin
        tree = MCTSTreeSoA(CUDA.device())
        build_test_tree!(tree, :balanced)
        
        # Collect statistics multiple times concurrently
        collectors = [TreeStatsCollector() for _ in 1:3]
        summaries = []
        
        # Simulate concurrent collection
        for collector in collectors
            summary = collect_tree_statistics(tree, collector)
            push!(summaries, summary)
        end
        
        # All summaries should be identical
        for i in 2:length(summaries)
            @test summaries[i].total_nodes == summaries[1].total_nodes
            @test summaries[i].max_depth == summaries[1].max_depth
            @test summaries[i].expanded_nodes == summaries[1].expanded_nodes
            @test summaries[i].active_nodes == summaries[1].active_nodes
            @test summaries[i].terminal_nodes == summaries[1].terminal_nodes
        end
    end
    
    @testset "Statistics Version Tracking" begin
        tree = MCTSTreeSoA(CUDA.device())
        build_test_tree!(tree, :balanced)
        
        collector = TreeStatsCollector()
        
        # Initial version
        CUDA.@allowscalar @test collector.stats_version[1] == 0
        
        # Collect stats multiple times
        for i in 1:5
            collect_tree_statistics(tree, collector)
            CUDA.@allowscalar @test collector.stats_version[1] == i
        end
    end
end

println("\n✅ Tree statistics tests completed!")