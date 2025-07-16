using Test
using CUDA
using Random

# Test the memory-efficient MCTS implementation
include("../../src/gpu/mcts_gpu.jl")
using .MCTSGPU

@testset "Memory-Efficient MCTS Tests" begin
    # Skip if no CUDA device available
    if !CUDA.functional()
        @warn "CUDA not available, skipping memory-efficient MCTS tests"
        return
    end
    
    device = CUDA.device()
    
    @testset "Compressed Node Storage" begin
        # Test compressed node encoding/decoding
        @test encode_visit_count(100) == UInt8(100)
        @test decode_visit_count(UInt8(100)) == 100
        
        # Test logarithmic encoding for large values
        @test decode_visit_count(encode_visit_count(1000)) >= 512  # Approximate due to log encoding
        
        # Test Q-value encoding
        test_q = 0.75f0
        encoded_q = encode_q_value(test_q)
        decoded_q = decode_q_value(encoded_q)
        @test abs(decoded_q - test_q) < 0.01f0  # Small error due to fixed-point
        
        # Test prior encoding
        test_prior = 0.3f0
        encoded_prior = encode_prior(test_prior)
        decoded_prior = decode_prior(encoded_prior)
        @test abs(decoded_prior - test_prior) < 0.001f0
        
        # Test compressed tree creation
        compressed_tree = CompressedTreeSoA(device, 10, 1000)
        @test compressed_tree.max_trees_per_gpu == 10
        @test compressed_tree.max_nodes_per_tree == 1000
        
        # Test node allocation
        node_id = allocate_compressed_node!(compressed_tree, Int32(1))
        @test node_id > 0
        
        # Test node initialization
        init_compressed_node!(compressed_tree, Int32(1), node_id, UInt16(0))
        
        # Test node field access
        set_visit_count!(compressed_tree, Int32(1), node_id, 50)
        @test get_visit_count(compressed_tree.nodes[node_id, 1]) == 50
        
        set_q_value!(compressed_tree, Int32(1), node_id, 0.8f0)
        q_val = get_q_value(compressed_tree.nodes[node_id, 1])
        @test abs(q_val - 0.8f0) < 0.01f0
        
        # Test node flags
        add_flag!(compressed_tree, Int32(1), node_id, NODE_FLAG_EXPANDED)
        @test has_flag(compressed_tree.nodes[node_id, 1], NODE_FLAG_EXPANDED)
        
        remove_flag!(compressed_tree, Int32(1), node_id, NODE_FLAG_EXPANDED)
        @test !has_flag(compressed_tree.nodes[node_id, 1], NODE_FLAG_EXPANDED)
        
        @info "Compressed node storage tests passed"
    end
    
    @testset "Shared Feature Storage" begin
        # Test feature pool creation
        feature_pool = SharedFeatureStorage.SharedFeaturePool(device, 0.8f0)
        @test feature_pool.gc_threshold == 0.8f0
        
        # Test feature hashing
        test_features = ntuple(i -> i == 1 ? UInt64(0x123456789ABCDEF0) : UInt64(0), FEATURE_CHUNKS)
        hash_val = hash_feature_set(test_features)
        @test hash_val != 0
        
        # Test feature storage
        feature_id = store_feature_set!(feature_pool, test_features, UInt16(1))
        @test feature_id > 0
        
        # Test feature retrieval
        retrieved_features = get_feature_set(feature_pool, feature_id)
        @test retrieved_features == test_features
        
        # Test reference counting
        @test add_reference!(feature_pool, feature_id)
        @test remove_reference!(feature_pool, feature_id)
        
        # Test deduplication
        duplicate_id = store_feature_set!(feature_pool, test_features, UInt16(1))
        @test duplicate_id == feature_id  # Should return same ID
        
        # Test pool statistics
        stats = get_pool_statistics(feature_pool)
        @test haskey(stats, "total_entries")
        @test haskey(stats, "utilization")
        
        @info "Shared feature storage tests passed"
    end
    
    @testset "Lazy Expansion" begin
        # Create dependencies
        feature_pool = SharedFeatureStorage.SharedFeaturePool(device, 0.8f0)
        lazy_manager = LazyExpansionManager(device, feature_pool, 10, 1000)
        
        # Test lazy context initialization
        init_lazy_context!(lazy_manager, Int32(1), UInt16(1), UInt8(5))
        
        # Test child descriptor addition
        success = add_child_descriptor!(
            lazy_manager, Int32(1), UInt16(1), UInt8(1),
            UInt16(10), 0.5f0, UInt32(0x12345678)
        )
        @test success
        
        # Test child descriptor retrieval
        descriptor = get_child_descriptor(lazy_manager, Int32(1), UInt16(1), UInt8(1))
        @test descriptor.action_id == UInt16(10)
        @test descriptor.feature_hash == UInt32(0x12345678)
        
        # Test expansion state
        @test !is_expanded(lazy_manager, Int32(1), UInt16(1))
        @test !is_expanding(lazy_manager, Int32(1), UInt16(1))
        
        # Test expansion start
        @test try_start_expansion!(lazy_manager, Int32(1), UInt16(1))
        @test is_expanding(lazy_manager, Int32(1), UInt16(1))
        
        # Test expansion completion
        complete_expansion!(lazy_manager, Int32(1), UInt16(1), UInt8(3))
        @test is_expanded(lazy_manager, Int32(1), UInt16(1))
        @test get_allocated_children(lazy_manager, Int32(1), UInt16(1)) == UInt8(3)
        
        # Test lazy expansion statistics
        stats = get_lazy_expansion_stats(lazy_manager)
        @test haskey(stats, "expansions_performed")
        @test haskey(stats, "memory_saved_mb")
        
        @info "Lazy expansion tests passed"
    end
    
    @testset "Memory-Efficient Ensemble" begin
        # Test ensemble creation
        ensemble = MemoryEfficientTreeEnsemble(device, max_trees=5, max_nodes_per_tree=100)
        @test ensemble.max_trees == 5
        @test ensemble.max_nodes_per_tree == 100
        
        # Test ensemble initialization
        initial_features = [Int[] for _ in 1:5]  # Empty initial features
        initialize_ensemble!(ensemble, initial_features)
        
        # Check that trees are initialized
        for tree_id in 1:5
            @test ensemble.tree.tree_states[tree_id] == UInt8(1)  # Active
            @test ensemble.tree.root_nodes[tree_id] == UInt16(1)
        end
        
        # Test dummy evaluation function
        dummy_eval = function(features::Vector{Int})
            return 0.5 + 0.1 * length(features)  # Simple scoring
        end
        
        # Test actions and priors
        actions = [UInt16(i) for i in 1:10]
        priors = [0.1f0 for _ in 1:10]
        
        # Test single MCTS simulation
        score = run_mcts_simulation!(ensemble, Int32(1), dummy_eval, actions, priors)
        @test score >= 0.0f0
        
        # Test short ensemble run
        run_ensemble_mcts!(ensemble, 10, dummy_eval, actions, priors)
        
        # Test feature extraction
        best_features = get_best_features_ensemble(ensemble, 5)
        @test length(best_features) <= 5
        
        # Test statistics
        stats = get_ensemble_statistics(ensemble)
        @test haskey(stats, "memory_stats")
        @test haskey(stats, "pool_stats")
        @test haskey(stats, "lazy_stats")
        @test haskey(stats, "tree_stats")
        
        # Check memory efficiency
        memory_stats = stats["memory_stats"]
        @test memory_stats["total_mb"] > 0
        @test memory_stats["compression_ratio"] > 1.0  # Should be compressed
        
        @info "Memory-efficient ensemble tests passed"
        @info "Total memory usage: $(memory_stats["total_mb"]) MB"
        @info "Compression ratio: $(memory_stats["compression_ratio"])"
    end
    
    @testset "Memory Pressure Test" begin
        # Test with larger ensemble to verify memory efficiency
        large_ensemble = MemoryEfficientTreeEnsemble(device, max_trees=20, max_nodes_per_tree=1000)
        
        # Initialize with random features
        Random.seed!(42)
        initial_features = [rand(1:100, 10) for _ in 1:20]  # 10 random features per tree
        initialize_ensemble!(large_ensemble, initial_features)
        
        # Get initial memory stats
        initial_stats = get_ensemble_statistics(large_ensemble)
        initial_memory = initial_stats["memory_stats"]["total_mb"]
        
        # Run longer simulation
        dummy_eval = features -> 0.5 + 0.01 * length(features)
        actions = [UInt16(i) for i in 1:50]
        priors = [1.0f0 / 50.0f0 for _ in 1:50]
        
        run_ensemble_mcts!(large_ensemble, 100, dummy_eval, actions, priors)
        
        # Check final memory usage
        final_stats = get_ensemble_statistics(large_ensemble)
        final_memory = final_stats["memory_stats"]["total_mb"]
        
        @test final_memory < 100.0  # Should be well under 100MB for 20 trees
        
        # Check that lazy expansion saved memory
        lazy_savings = final_stats["lazy_stats"]["memory_saved_mb"]
        @test lazy_savings > 0
        
        # Check feature sharing efficiency
        pool_stats = final_stats["pool_stats"]
        @test pool_stats["utilization"] < 0.5  # Should be sharing features effectively
        
        @info "Memory pressure test passed"
        @info "20 trees with 1000 nodes each: $(final_memory) MB"
        @info "Lazy expansion saved: $(lazy_savings) MB"
        @info "Feature pool utilization: $(pool_stats["utilization"])"
    end
    
    @testset "Performance Comparison" begin
        # Compare memory usage between original and compressed
        
        # Original implementation (approximate)
        original_node_size = 256  # bytes
        original_50_trees = 50 * 20000 * original_node_size / (1024^2)  # MB
        
        # Compressed implementation
        compressed_ensemble = MemoryEfficientTreeEnsemble(device, max_trees=50, max_nodes_per_tree=20000)
        initialize_ensemble!(compressed_ensemble, [Int[] for _ in 1:50])
        
        stats = get_ensemble_statistics(compressed_ensemble)
        actual_memory = stats["memory_stats"]["total_mb"]
        compression_ratio = stats["memory_stats"]["compression_ratio"]
        
        @test actual_memory < original_50_trees * 0.5  # Should be at least 50% better
        @test compression_ratio > 2.0  # Should be at least 2x compression
        
        @info "Performance comparison:"
        @info "Original (estimated): $(original_50_trees) MB"
        @info "Compressed (actual): $(actual_memory) MB"
        @info "Compression ratio: $(compression_ratio)"
        @info "Memory reduction: $(100 * (1 - actual_memory / original_50_trees))%"
    end
end

@info "All memory-efficient MCTS tests completed successfully!"