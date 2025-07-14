using Test
using CUDA

# Include the main MCTS GPU module instead of individual modules
include("../../src/gpu/mcts_gpu.jl")

using .MCTSGPU

@testset "Tree Statistics Integration Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU tree statistics tests"
        return
    end
    
    @testset "Basic Tree Statistics Collection" begin
        # Create engine
        engine = MCTSGPUEngine(
            block_size = 256,
            grid_size = 108,
            max_iterations = 100
        )
        
        # Initialize engine
        initialize!(engine, [1, 2, 3])
        
        # Collect initial statistics
        stats_summary = MCTSGPU.collect_tree_stats!(engine)
        
        # Check initial state
        @test stats_summary.total_nodes == 1  # Just root
        @test stats_summary.active_nodes == 1
        @test stats_summary.expanded_nodes == 0
        @test stats_summary.terminal_nodes == 0
        @test stats_summary.max_depth == 0
        
        # Get tree summary
        summary2 = MCTSGPU.get_tree_summary(engine)
        @test summary2.total_nodes == stats_summary.total_nodes
    end
    
    @testset "Performance Report with Tree Statistics" begin
        # Create engine
        engine = MCTSGPUEngine()
        initialize!(engine)
        
        # Collect some statistics
        MCTSGPU.collect_tree_stats!(engine)
        
        # Get performance report
        report = get_performance_report(engine)
        
        # Check report structure
        @test haskey(report, "mcts_metrics")
        @test haskey(report, "tree_statistics")
        @test haskey(report["tree_statistics"], "node_counts")
        @test haskey(report["tree_statistics"], "tree_structure")
    end
    
    # Skip kernel execution tests due to struct passing limitation
    @testset "Tree Statistics During Execution" begin
        @test_skip "Skipping execution tests due to CUDA.jl struct passing limitation"
    end
end

println("\n✅ Tree statistics integration tests completed!")