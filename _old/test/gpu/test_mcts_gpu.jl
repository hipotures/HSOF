using Test
using CUDA
using Random

# Include MCTS GPU module
include("../../src/gpu/mcts_gpu.jl")
using .MCTSGPU

@testset "MCTS GPU Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU tests"
        return
    end
    
    @testset "Basic Engine Creation" begin
        engine = MCTSGPUEngine()
        
        @test !isnothing(engine.tree)
        @test !isnothing(engine.config)
        @test !isnothing(engine.memory_manager)
        @test !engine.is_running
    end
    
    @testset "Tree Initialization" begin
        engine = MCTSGPUEngine()
        
        # Initialize with empty features
        initialize!(engine)
        
        # Check root node
        CUDA.@allowscalar begin
            @test engine.tree.node_states[1] == MCTSGPU.MCTSTypes.NODE_ACTIVE
            @test engine.tree.parent_ids[1] == -1
            @test engine.tree.visit_counts[1] == 1
            @test engine.tree.total_nodes[1] == 1
        end
        
        # Initialize with initial features
        initial_features = [10, 20, 30, 40, 50]
        initialize!(engine, initial_features)
        
        CUDA.@allowscalar begin
            # Check features are set
            for feature in initial_features
                @test MCTSGPU.MCTSTypes.has_feature(engine.tree.feature_masks, 1, Int32(feature))
            end
        end
    end
    
    @testset "Memory Pool Operations" begin
        engine = MCTSGPUEngine()
        initialize!(engine)
        
        # Test node allocation
        CUDA.@allowscalar begin
            # Allocate some nodes
            node_indices = Int32[]
            for i in 1:10
                idx = MCTSGPU.MemoryPool.allocate_node!(engine.tree)
                @test idx > 0
                push!(node_indices, idx)
            end
            
            @test engine.tree.total_nodes[1] == 11  # Including root
            
            # Free some nodes
            for idx in node_indices[1:5]
                MCTSGPU.MemoryPool.free_node!(engine.tree, idx)
            end
            
            # Allocate again - should reuse freed nodes
            new_idx = MCTSGPU.MemoryPool.allocate_node!(engine.tree)
            @test new_idx in node_indices[1:5]
        end
    end
    
    @testset "Feature Mask Operations" begin
        engine = MCTSGPUEngine()
        initialize!(engine)
        
        CUDA.@allowscalar begin
            # Test setting features
            node_idx = Int32(1)
            features_to_set = [Int32(i) for i in 100:10:200]
            
            for feature in features_to_set
                MCTSGPU.MCTSTypes.set_feature!(engine.tree.feature_masks, node_idx, feature)
            end
            
            # Verify features are set
            for feature in features_to_set
                @test MCTSGPU.MCTSTypes.has_feature(engine.tree.feature_masks, node_idx, feature)
            end
            
            # Count features
            count = MCTSGPU.MCTSTypes.count_features(engine.tree.feature_masks, node_idx)
            @test count == length(features_to_set) + 5  # Plus initial features
            
            # Unset a feature
            MCTSGPU.MCTSTypes.unset_feature!(engine.tree.feature_masks, node_idx, features_to_set[1])
            @test !MCTSGPU.MCTSTypes.has_feature(engine.tree.feature_masks, node_idx, features_to_set[1])
        end
    end
    
    @testset "UCB1 Score Calculation" begin
        # Test UCB1 formula
        total_score = 10.0f0
        visit_count = Int32(5)
        parent_visits = Int32(100)
        exploration_constant = 1.414f0
        prior_score = 0.5f0
        
        score = MCTSGPU.MCTSTypes.ucb1_score(
            total_score, visit_count, parent_visits,
            exploration_constant, prior_score
        )
        
        @test score > 0
        @test !isinf(score)
        
        # Test unvisited node
        score_unvisited = MCTSGPU.MCTSTypes.ucb1_score(
            0.0f0, Int32(0), parent_visits,
            exploration_constant, prior_score
        )
        
        @test isinf(score_unvisited)
    end
    
    @testset "Kernel Launch and Stop" begin
        engine = MCTSGPUEngine(max_iterations = 10)
        initialize!(engine)
        
        # Start kernel
        start!(engine)
        @test engine.is_running
        
        # Let it run briefly
        sleep(0.1)
        
        # Stop kernel
        stop!(engine)
        @test !engine.is_running
        
        # Check some work was done
        stats = get_statistics(engine)
        @test stats["nodes_allocated"] >= 1
        @test haskey(stats, "iterations")
    end
    
    @testset "Feature Selection Small Test" begin
        engine = MCTSGPUEngine(
            max_iterations = 100,
            block_size = 32,
            grid_size = 4
        )
        
        # Run feature selection
        num_features_to_select = 10
        selected = select_features(engine, num_features_to_select, 100)
        
        @test length(selected) <= num_features_to_select
        @test all(f -> 1 <= f <= MCTSGPU.MCTSTypes.MAX_FEATURES, selected)
        
        # Get statistics
        stats = get_statistics(engine)
        @test stats["nodes_allocated"] > 1
        @test stats["iterations"] > 0
    end
    
    @testset "Memory Defragmentation" begin
        engine = MCTSGPUEngine(defrag_threshold = 0.3f0)
        initialize!(engine)
        
        CUDA.@allowscalar begin
            # Allocate many nodes
            nodes = Int32[]
            for i in 1:100
                idx = MCTSGPU.MemoryPool.allocate_node!(engine.tree)
                if idx > 0
                    push!(nodes, idx)
                end
            end
            
            # Free every other node to create fragmentation
            for i in 1:2:length(nodes)
                MCTSGPU.MemoryPool.free_node!(engine.tree, nodes[i])
            end
            
            # Check if defragmentation is needed
            @test MCTSGPU.MemoryPool.should_defragment(engine.memory_manager)
            
            # Run defragmentation
            initial_total = engine.tree.total_nodes[1]
            new_total = MCTSGPU.MemoryPool.defragment!(engine.memory_manager)
            
            @test new_total < length(nodes)  # Should be compacted
        end
    end
    
    @testset "Statistics Collection" begin
        engine = MCTSGPUEngine(max_iterations = 50)
        initialize!(engine)
        
        start!(engine)
        sleep(0.2)
        stop!(engine)
        
        stats = get_statistics(engine)
        
        # Check all expected fields
        @test haskey(stats, "nodes_allocated")
        @test haskey(stats, "nodes_recycled")
        @test haskey(stats, "max_depth")
        @test haskey(stats, "gpu_utilization")
        @test haskey(stats, "elapsed_time")
        @test haskey(stats, "device")
        @test haskey(stats, "memory_used_mb")
        @test haskey(stats, "iterations")
        
        # Validate ranges
        @test stats["nodes_allocated"] >= 1
        @test stats["gpu_utilization"] >= 0
        @test stats["elapsed_time"] > 0
        @test stats["memory_used_mb"] > 0
    end
    
    @testset "Progress Monitoring" begin
        engine = MCTSGPUEngine(max_iterations = 100)
        initialize!(engine)
        
        # Check initial progress
        @test get_progress(engine) == 0.0
        
        # Start and monitor
        start!(engine)
        sleep(0.1)
        
        progress = get_progress(engine)
        @test 0.0 <= progress <= 1.0
        
        stop!(engine)
    end
end

# Run benchmarks if requested
if get(ENV, "RUN_BENCHMARKS", "false") == "true"
    println("\n" * "="^60)
    println("MCTS GPU Benchmarks")
    println("="^60)
    
    engine = MCTSGPUEngine(
        max_iterations = 10000,
        block_size = 256,
        grid_size = 108
    )
    
    # Benchmark feature selection
    println("\nBenchmarking feature selection...")
    
    for num_features in [10, 50, 100]
        println("\nSelecting $num_features features:")
        
        start_time = time()
        selected = select_features(engine, num_features, 10000)
        elapsed = time() - start_time
        
        stats = get_statistics(engine)
        
        println("  Time: $(round(elapsed, digits=2))s")
        println("  Nodes: $(stats["nodes_allocated"])")
        println("  GPU Utilization: $(round(stats["gpu_utilization"], digits=1))%")
        println("  Throughput: $(round(stats["iterations"] / elapsed, digits=0)) iter/s")
    end
end

println("\n✅ All MCTS GPU tests passed!")