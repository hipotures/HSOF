using Test
using CUDA

# Include MCTS GPU module
include("../../src/gpu/mcts_gpu.jl")
using .MCTSGPU

@testset "Simple MCTS GPU Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU tests"
        return
    end
    
    @testset "Basic Engine Creation and Init" begin
        engine = MCTSGPUEngine(max_iterations = 10)
        @test !isnothing(engine)
        
        # Initialize
        initialize!(engine)
        
        # Check root node using Array to copy from GPU
        CUDA.@allowscalar begin
            @test engine.tree.total_nodes[1] == 1
            @test engine.tree.node_states[1] == MCTSGPU.MCTSTypes.NODE_ACTIVE
            @test engine.tree.parent_ids[1] == -1
            @test engine.tree.visit_counts[1] == 1
        end
    end
    
    @testset "Feature Initialization" begin
        engine = MCTSGPUEngine()
        initial_features = [10, 20, 30]
        initialize!(engine, initial_features)
        
        # Check features are set
        CUDA.@allowscalar begin
            for feature in initial_features
                @test MCTSGPU.MCTSTypes.has_feature(engine.tree.feature_masks, Int32(1), Int32(feature))
            end
        end
    end
    
    @testset "UCB1 Score Function" begin
        # Test on CPU
        score = MCTSGPU.MCTSTypes.ucb1_score(10.0f0, Int32(5), Int32(100), 1.414f0, 0.5f0)
        @test score > 0
        @test !isinf(score)
        
        # Test unvisited
        score_unvisited = MCTSGPU.MCTSTypes.ucb1_score(0.0f0, Int32(0), Int32(100), 1.414f0, 0.5f0)
        @test isinf(score_unvisited)
    end
    
    @testset "Statistics" begin
        engine = MCTSGPUEngine()
        initialize!(engine)
        
        stats = get_statistics(engine)
        @test haskey(stats, "nodes_allocated")
        @test stats["nodes_allocated"] == 1  # Just root
        @test haskey(stats, "device")
        @test haskey(stats, "memory_used_mb")
    end
end

println("\n✅ Simple MCTS GPU tests passed!")