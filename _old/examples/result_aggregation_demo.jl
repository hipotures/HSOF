#!/usr/bin/env julia

# Result Aggregation Demo
# Demonstrates unified result collection and ensemble consensus across multiple GPUs

using CUDA
using Printf
using Random
using Dates
using Statistics

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

# Load GPU modules
include("../src/gpu/GPU.jl")
using .GPU

# Import from GPU module
using .GPU.ResultAggregation

"""
Generate mock tree results for demonstration
"""
function generate_mock_results(
    aggregator::ResultAggregator,
    num_trees::Int,
    feature_pool::Vector{Int},
    consensus_features::Vector{Int} = Int[]
)
    Random.seed!(12345)  # For reproducible results
    
    trees_per_gpu = num_trees ÷ aggregator.num_gpus
    remaining = num_trees % aggregator.num_gpus
    
    tree_id = 1
    for gpu_id in 0:(aggregator.num_gpus-1)
        trees_for_gpu = trees_per_gpu + (gpu_id < remaining ? 1 : 0)
        
        for _ in 1:trees_for_gpu
            # Select features for this tree
            num_features = rand(20:50)
            
            # Ensure consensus features are included with high probability
            selected_features = Int[]
            for cf in consensus_features
                if rand() < 0.9  # 90% chance to include consensus feature
                    push!(selected_features, cf)
                end
            end
            
            # Add random features
            while length(selected_features) < num_features
                f = rand(feature_pool)
                if !(f in selected_features)
                    push!(selected_features, f)
                end
            end
            
            # Generate feature scores
            feature_scores = Dict{Int, Float32}()
            for f in selected_features
                # Consensus features get higher scores
                base_score = f in consensus_features ? 0.8f0 : 0.5f0
                feature_scores[f] = base_score + rand(Float32) * 0.2f0
            end
            
            # Create tree result
            confidence = 0.8f0 + rand(Float32) * 0.2f0
            iterations = 1000 + rand(1:500)
            compute_time = 100.0 + rand() * 50.0
            
            result = TreeResult(
                tree_id,
                gpu_id,
                selected_features,
                feature_scores,
                confidence,
                iterations,
                compute_time
            )
            
            submit_tree_result!(aggregator, result)
            tree_id += 1
        end
    end
end

"""
Display aggregation results
"""
function display_results(ensemble_result::EnsembleResult)
    println("\nEnsemble Aggregation Results:")
    println("=" ^ 60)
    
    println("Total trees aggregated: $(ensemble_result.total_trees)")
    println("Participating GPUs: $(ensemble_result.participating_gpus)")
    println("Aggregation time: $(round(ensemble_result.aggregation_time, digits=2))ms")
    println("Ensemble confidence: $(round(ensemble_result.confidence * 100, digits=1))%")
    
    println("\nConsensus Features ($(length(ensemble_result.consensus_features)) total):")
    if length(ensemble_result.consensus_features) <= 20
        println("  $(ensemble_result.consensus_features)")
    else
        println("  First 20: $(ensemble_result.consensus_features[1:20])")
        println("  ... and $(length(ensemble_result.consensus_features) - 20) more")
    end
    
    println("\nTop 20 Features by Ranking:")
    println("  Rank | Feature | Score")
    println("  -----|---------|-------")
    for (i, (feature_id, score)) in enumerate(ensemble_result.feature_rankings[1:min(20, end)])
        println(@sprintf("  %4d | %7d | %.3f", i, feature_id, score))
    end
end

"""
Demo result aggregation system
"""
function demo_result_aggregation()
    println("GPU Result Aggregation Demo")
    println("=" ^ 60)
    
    # Create aggregator
    aggregator = create_result_aggregator(
        num_gpus = 2,
        total_trees = 100,
        consensus_threshold = 0.6f0,  # 60% threshold
        top_k_features = 50,
        caching_enabled = true
    )
    
    println("\nAggregator Configuration:")
    println("  Number of GPUs: $(aggregator.num_gpus)")
    println("  Total trees: $(aggregator.total_trees)")
    println("  Consensus threshold: $(aggregator.consensus_threshold * 100)%")
    println("  Top K features: $(aggregator.top_k_features)")
    println("  Caching enabled: $(aggregator.caching_enabled)")
    
    # Demo 1: Basic Aggregation
    println("\n1. Basic Feature Selection Aggregation:")
    println("-" ^ 40)
    
    # Define feature pool and consensus features
    feature_pool = collect(1:1000)
    consensus_features = [10, 25, 42, 78, 99, 150, 200, 333, 500, 750]
    
    println("Generating mock results from 100 trees...")
    println("Expected consensus features: $consensus_features")
    
    generate_mock_results(aggregator, 100, feature_pool, consensus_features)
    
    # Aggregate results
    ensemble_result = aggregate_results(aggregator)
    display_results(ensemble_result)
    
    # Demo 2: Caching Performance
    println("\n2. Caching Performance Test:")
    println("-" ^ 40)
    
    # First aggregation already done
    cache_stats = get_cache_stats(aggregator)
    println("Initial cache stats:")
    println("  Cache size: $(cache_stats["cache_size"])")
    println("  Cache hits: $(cache_stats["cache_hits"])")
    println("  Cache misses: $(cache_stats["cache_misses"])")
    
    # Second aggregation - should hit cache
    println("\nRunning aggregation again (should hit cache)...")
    start_time = time()
    result2 = aggregate_results(aggregator)
    cache_time = (time() - start_time) * 1000
    
    cache_stats2 = get_cache_stats(aggregator)
    println("After second aggregation:")
    println("  Time: $(round(cache_time, digits=2))ms (cached)")
    println("  Cache hits: $(cache_stats2["cache_hits"])")
    println("  Hit rate: $(round(cache_stats2["hit_rate"] * 100, digits=1))%")
    
    # Demo 3: Threshold Adjustment
    println("\n3. Consensus Threshold Adjustment:")
    println("-" ^ 40)
    
    println("Changing threshold from 60% to 80%...")
    set_consensus_threshold!(aggregator, 0.8f0)
    
    result3 = aggregate_results(aggregator)
    println("\nResults with 80% threshold:")
    println("  Consensus features: $(length(result3.consensus_features))")
    println("  (Previous: $(length(ensemble_result.consensus_features)))")
    
    # Demo 4: GPU Imbalance Detection
    println("\n4. GPU Balance Analysis:")
    println("-" ^ 40)
    
    gpu_stats = ResultAggregation.get_gpu_statistics(aggregator)
    
    for (gpu_id, stats) in sort(collect(gpu_stats))
        println("\nGPU $gpu_id:")
        println("  Trees processed: $(stats["tree_count"])")
        if haskey(stats, "avg_response_time")
            println("  Avg response time: $(round(stats["avg_response_time"], digits=2))ms")
            println("  Min response time: $(round(stats["min_response_time"], digits=2))ms")
            println("  Max response time: $(round(stats["max_response_time"], digits=2))ms")
        end
    end
    
    # Demo 5: Partial Results (Fault Tolerance)
    println("\n5. Partial Results Test:")
    println("-" ^ 40)
    
    # Clear and submit only partial results
    clear_results!(aggregator)
    println("Cleared all results")
    
    println("Submitting results from only 30 trees...")
    generate_mock_results(aggregator, 30, feature_pool, consensus_features)
    
    partial_result = ResultAggregation.create_partial_result(aggregator, 20)
    if !isnothing(partial_result)
        println("\nPartial aggregation successful:")
        println("  Trees: $(partial_result.total_trees)")
        println("  Consensus features: $(length(partial_result.consensus_features))")
        println("  Confidence: $(round(partial_result.confidence * 100, digits=1))%")
    end
    
    # Demo 6: Feature Score Analysis
    println("\n6. Feature Score Distribution:")
    println("-" ^ 40)
    
    # Submit fresh results
    clear_results!(aggregator)
    generate_mock_results(aggregator, 100, feature_pool, consensus_features)
    
    # Analyze different scoring scenarios
    println("\nTesting feature ranking with different score distributions...")
    
    # Add some results with very different scores
    for i in 101:105
        # High confidence for specific features
        special_features = [42, 99, 777]
        scores = Dict{Int, Float32}()
        for f in special_features
            scores[f] = 0.95f0 + rand(Float32) * 0.05f0
        end
        
        result = TreeResult(i, i % 2, special_features, scores, 0.99f0, 1500, 120.0)
        submit_tree_result!(aggregator, result)
    end
    
    final_result = aggregate_results(aggregator)
    
    println("\nFeature ranking changes after high-confidence submissions:")
    top_10 = final_result.feature_rankings[1:10]
    for (i, (fid, score)) in enumerate(top_10)
        special = fid in [42, 99, 777] ? " *" : ""
        println(@sprintf("  %2d. Feature %4d: %.3f%s", i, fid, score, special))
    end
    
    # Demo 7: Performance Summary
    println("\n7. Performance Metrics:")
    println("-" ^ 40)
    
    total_aggregations = aggregator.aggregation_count[]
    println("Total aggregations performed: $total_aggregations")
    
    if aggregator.caching_enabled
        final_cache_stats = get_cache_stats(aggregator)
        println("Cache performance:")
        println("  Total requests: $(final_cache_stats["cache_hits"] + final_cache_stats["cache_misses"])")
        println("  Hit rate: $(round(final_cache_stats["hit_rate"] * 100, digits=1))%")
        println("  Cache size: $(final_cache_stats["cache_size"])/$(final_cache_stats["max_size"])")
    end
    
    # Demo 8: Real-world Scenario
    println("\n8. Real-world Feature Selection Scenario:")
    println("-" ^ 40)
    
    # Simulate a more realistic scenario
    clear_results!(aggregator)
    set_consensus_threshold!(aggregator, 0.7f0)  # 70% threshold
    
    println("Simulating feature selection for 5000 features...")
    large_feature_pool = collect(1:5000)
    
    # Define tiers of features
    tier1_features = [100, 200, 300, 400, 500]  # Very important
    tier2_features = [1000, 1500, 2000, 2500, 3000]  # Important
    tier3_features = collect(3500:3520)  # Moderately important
    
    for tree_id in 1:100
        gpu_id = (tree_id - 1) % 2
        
        # Select features based on importance tiers
        selected = Int[]
        
        # Tier 1: 95% chance
        for f in tier1_features
            if rand() < 0.95
                push!(selected, f)
            end
        end
        
        # Tier 2: 75% chance
        for f in tier2_features
            if rand() < 0.75
                push!(selected, f)
            end
        end
        
        # Tier 3: 50% chance
        for f in tier3_features
            if rand() < 0.50
                push!(selected, f)
            end
        end
        
        # Random features: fill to 30-50 features
        target_size = rand(30:50)
        while length(selected) < target_size
            f = rand(large_feature_pool)
            if !(f in selected)
                push!(selected, f)
            end
        end
        
        # Generate scores
        scores = Dict{Int, Float32}()
        for f in selected
            if f in tier1_features
                scores[f] = 0.9f0 + rand(Float32) * 0.1f0
            elseif f in tier2_features
                scores[f] = 0.7f0 + rand(Float32) * 0.2f0
            elseif f in tier3_features
                scores[f] = 0.5f0 + rand(Float32) * 0.3f0
            else
                scores[f] = rand(Float32) * 0.5f0
            end
        end
        
        result = TreeResult(tree_id, gpu_id, selected, scores, 0.85f0, 1200, 110.0)
        submit_tree_result!(aggregator, result)
    end
    
    final_ensemble = aggregate_results(aggregator)
    
    println("\nFinal ensemble results:")
    println("  Total features ranked: $(length(final_ensemble.feature_rankings))")
    println("  Consensus features: $(length(final_ensemble.consensus_features))")
    
    # Check if tier features made it to consensus
    tier1_in_consensus = count(f -> f in final_ensemble.consensus_features, tier1_features)
    tier2_in_consensus = count(f -> f in final_ensemble.consensus_features, tier2_features)
    
    println("\nTier analysis:")
    println("  Tier 1 features in consensus: $tier1_in_consensus/$(length(tier1_features))")
    println("  Tier 2 features in consensus: $tier2_in_consensus/$(length(tier2_features))")
    
    println("\n" * "=" ^ 60)
    println("Demo completed!")
end

# Utility function for interactive exploration
function explore_aggregation_interactive()
    println("\nInteractive Result Aggregation Explorer")
    println("=" ^ 40)
    
    aggregator = create_result_aggregator(
        num_gpus = 2,
        total_trees = 20,
        consensus_threshold = 0.5f0
    )
    
    println("Small aggregator created (20 trees)")
    println("\nCommands:")
    println("  1. Add random results")
    println("  2. Add targeted results") 
    println("  3. Show current aggregation")
    println("  4. Clear all results")
    println("  5. Change consensus threshold")
    println("  6. Show cache stats")
    println("  7. Exit")
    
    # Simple command loop would go here
    println("\n(Interactive mode not implemented in this demo)")
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    demo_result_aggregation()
    
    # Uncomment to run interactive explorer
    # explore_aggregation_interactive()
end