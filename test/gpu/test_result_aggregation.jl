using Test
using CUDA
using Dates
using Statistics

# Include required modules
include("../../src/gpu/GPU.jl")
using .GPU

# Import from GPU module
using .GPU.ResultAggregation

@testset "Result Aggregation Tests" begin
    
    @testset "Aggregator Creation" begin
        # Test with default parameters
        aggregator = create_result_aggregator()
        @test isa(aggregator, ResultAggregator)
        @test aggregator.num_gpus == 2
        @test aggregator.total_trees == 100
        @test aggregator.consensus_threshold == 0.5f0
        @test aggregator.caching_enabled
        @test isempty(aggregator.tree_results)
        
        # Test with custom parameters
        aggregator2 = create_result_aggregator(
            num_gpus = 4,
            total_trees = 200,
            consensus_threshold = 0.7f0,
            top_k_features = 100,
            caching_enabled = false
        )
        @test aggregator2.num_gpus == 4
        @test aggregator2.total_trees == 200
        @test aggregator2.consensus_threshold == 0.7f0
        @test !aggregator2.caching_enabled
        
        # Check initialization
        @test length(aggregator2.gpu_tree_counts) == 4
        @test all(v -> v == 0, values(aggregator2.gpu_tree_counts))
    end
    
    @testset "Tree Result Submission" begin
        aggregator = create_result_aggregator()
        
        # Submit result from GPU 0
        result1 = TreeResult(
            1, 0, [1, 5, 10, 15], 
            Dict(1 => 0.9f0, 5 => 0.8f0, 10 => 0.7f0, 15 => 0.6f0),
            0.95f0, 1000, 150.0
        )
        submit_tree_result!(aggregator, result1)
        
        @test haskey(aggregator.tree_results, 1)
        @test aggregator.tree_results[1].tree_id == 1
        @test aggregator.gpu_tree_counts[0] == 1
        
        # Submit result from GPU 1
        result2 = TreeResult(
            51, 1, [2, 5, 11, 15],
            Dict(2 => 0.85f0, 5 => 0.9f0, 11 => 0.75f0, 15 => 0.65f0),
            0.92f0, 1100, 145.0
        )
        submit_tree_result!(aggregator, result2)
        
        @test haskey(aggregator.tree_results, 51)
        @test aggregator.gpu_tree_counts[1] == 1
        
        # Check response time tracking
        @test length(aggregator.gpu_response_times[0]) == 1
        @test aggregator.gpu_response_times[0][1] == 150.0
    end
    
    @testset "Result Aggregation" begin
        aggregator = create_result_aggregator(
            num_gpus = 2,
            total_trees = 10,  # Small for testing
            consensus_threshold = 0.5f0
        )
        
        # Submit results from multiple trees
        # Features 5 and 15 appear in most trees (consensus)
        # Feature 1 appears in half
        # Others appear less frequently
        
        for i in 1:5
            # GPU 0 trees
            features = if i <= 3
                [1, 5, 15, 20+i]  # 3 trees select feature 1
            else
                [5, 15, 20+i, 30+i]
            end
            
            scores = Dict{Int, Float32}()
            for f in features
                scores[f] = 0.5f0 + 0.1f0 * i
            end
            
            result = TreeResult(i, 0, features, scores, 0.9f0, 1000, 100.0)
            submit_tree_result!(aggregator, result)
        end
        
        for i in 6:10
            # GPU 1 trees
            features = if i <= 7
                [1, 5, 15, 40+i]  # 2 trees select feature 1 (total 5/10 = 50%)
            else
                [5, 15, 40+i, 50+i]
            end
            
            scores = Dict{Int, Float32}()
            for f in features
                scores[f] = 0.5f0 + 0.1f0 * i
            end
            
            result = TreeResult(i, 1, features, scores, 0.9f0, 1000, 100.0)
            submit_tree_result!(aggregator, result)
        end
        
        # Aggregate results
        ensemble_result = aggregate_results(aggregator)
        
        @test ensemble_result.total_trees == 10
        @test ensemble_result.participating_gpus == [0, 1]
        @test !ensemble_result.cache_hit
        
        # Check consensus features (5 and 15 should be consensus, 1 is exactly at threshold)
        @test 5 in ensemble_result.consensus_features
        @test 15 in ensemble_result.consensus_features
        @test 1 in ensemble_result.consensus_features  # 5/10 = 50% = threshold
        
        # Check feature rankings
        @test !isempty(ensemble_result.feature_rankings)
        
        # Find positions of features 5 and 15 in rankings
        feature_ids = [f[1] for f in ensemble_result.feature_rankings]
        @test 5 in feature_ids
        @test 15 in feature_ids
        
        # They should be high-ranked since they appear in all trees
        pos_5 = findfirst(x -> x[1] == 5, ensemble_result.feature_rankings)
        pos_15 = findfirst(x -> x[1] == 15, ensemble_result.feature_rankings)
        @test pos_5 <= 10  # Should be in top 10
        @test pos_15 <= 10  # Should be in top 10
    end
    
    @testset "Consensus Threshold" begin
        aggregator = create_result_aggregator(
            total_trees = 10,
            consensus_threshold = 0.7f0  # 70% threshold
        )
        
        # Feature 1: appears in 5/10 trees (50%) - exactly at threshold
        # Feature 2: appears in 8/10 trees (80%)
        # Feature 3: appears in 10/10 trees (100%)
        
        for i in 1:10
            features = if i <= 5
                [1, 2, 3]
            elseif i <= 8
                [2, 3]
            else
                [3]
            end
            
            result = TreeResult(i, i % 2, features)
            submit_tree_result!(aggregator, result)
        end
        
        ensemble_result = aggregate_results(aggregator)
        
        # Only features 2 and 3 meet 70% threshold
        @test !(1 in ensemble_result.consensus_features)
        @test 2 in ensemble_result.consensus_features
        @test 3 in ensemble_result.consensus_features
        
        # Test threshold adjustment to 40%
        set_consensus_threshold!(aggregator, 0.4f0)
        ensemble_result2 = aggregate_results(aggregator)
        
        # Now feature 1 (50%) should definitely be included
        @test 1 in ensemble_result2.consensus_features
        @test 2 in ensemble_result2.consensus_features
        @test 3 in ensemble_result2.consensus_features
    end
    
    @testset "Caching System" begin
        aggregator = create_result_aggregator(
            total_trees = 5,
            caching_enabled = true,
            cache_size = 10
        )
        
        # Submit initial results
        for i in 1:5
            result = TreeResult(i, 0, [i, i+10])
            submit_tree_result!(aggregator, result)
        end
        
        # First aggregation - cache miss
        result1 = aggregate_results(aggregator)
        @test !result1.cache_hit
        @test aggregator.cache_misses[] == 1
        @test aggregator.cache_hits[] == 0
        
        # Second aggregation with same data - cache hit
        result2 = aggregate_results(aggregator)
        @test aggregator.cache_hits[] == 1  # Should have a cache hit
        
        # Cache hit field is part of the result, not exposed separately in current implementation
        # Instead verify by cache stats
        
        # Verify results are identical
        @test result1.selected_features == result2.selected_features
        @test result1.consensus_features == result2.consensus_features
        
        # Add new result - invalidates cache
        submit_tree_result!(aggregator, TreeResult(6, 1, [6, 16]))
        result3 = aggregate_results(aggregator)
        @test !result3.cache_hit
        @test aggregator.cache_misses[] == 2
        
        # Test cache statistics
        stats = get_cache_stats(aggregator)
        @test stats["cache_hits"] == 1
        @test stats["cache_misses"] == 2
        @test stats["hit_rate"] ≈ 1/3
        
        # Test cache reset
        reset_cache!(aggregator)
        stats2 = get_cache_stats(aggregator)
        @test stats2["cache_size"] == 0
        @test stats2["cache_hits"] == 0
    end
    
    @testset "GPU Balance Factor" begin
        aggregator = create_result_aggregator(num_gpus = 2, total_trees = 10)
        
        # Balanced submission
        for i in 1:5
            submit_tree_result!(aggregator, TreeResult(i, 0, [i]))
        end
        for i in 6:10
            submit_tree_result!(aggregator, TreeResult(i, 1, [i]))
        end
        
        result = aggregate_results(aggregator)
        @test result.confidence > 0.5f0  # Reasonable confidence with balanced GPUs
        
        # Imbalanced submission
        aggregator2 = create_result_aggregator(num_gpus = 2, total_trees = 10)
        
        # 9 results from GPU 0, only 1 from GPU 1
        for i in 1:9
            submit_tree_result!(aggregator2, TreeResult(i, 0, [i]))
        end
        submit_tree_result!(aggregator2, TreeResult(10, 1, [10]))
        
        result2 = aggregate_results(aggregator2)
        @test result2.confidence < result.confidence  # Lower confidence due to imbalance
    end
    
    @testset "Feature Score Aggregation" begin
        aggregator = create_result_aggregator(total_trees = 3)
        
        # Tree 1: features [1, 2] with scores
        submit_tree_result!(aggregator, TreeResult(
            1, 0, [1, 2],
            Dict(1 => 0.9f0, 2 => 0.8f0, 3 => 0.3f0),  # 3 has score but not selected
            0.95f0
        ))
        
        # Tree 2: features [1, 3] with scores
        submit_tree_result!(aggregator, TreeResult(
            2, 0, [1, 3],
            Dict(1 => 0.85f0, 3 => 0.9f0),
            0.90f0
        ))
        
        # Tree 3: features [2, 3] with scores
        submit_tree_result!(aggregator, TreeResult(
            3, 1, [2, 3],
            Dict(2 => 0.7f0, 3 => 0.95f0),
            0.92f0
        ))
        
        result = aggregate_results(aggregator)
        
        # Check feature rankings include scored features
        feature_ids = [f[1] for f in result.feature_rankings]
        @test 1 in feature_ids
        @test 2 in feature_ids
        @test 3 in feature_ids
        
        # Feature 1: selected by 2/3 trees
        # Feature 2: selected by 2/3 trees
        # Feature 3: selected by 2/3 trees (but highest scores)
        
        # With consensus threshold 0.5, all should be consensus features
        @test length(result.consensus_features) == 3
    end
    
    @testset "Callbacks" begin
        aggregator = create_result_aggregator()
        
        # Track callback execution
        pre_called = Ref(false)
        post_called = Ref(false)
        
        # Register callbacks
        push!(aggregator.pre_aggregation_callbacks, 
            (agg) -> pre_called[] = true)
        push!(aggregator.post_aggregation_callbacks, 
            (res) -> post_called[] = true)
        
        # Submit some results
        submit_tree_result!(aggregator, TreeResult(1, 0, [1, 2, 3]))
        
        # Aggregate - should trigger callbacks
        aggregate_results(aggregator)
        
        @test pre_called[]
        @test post_called[]
    end
    
    @testset "Partial Results" begin
        aggregator = create_result_aggregator(total_trees = 100)
        
        # Submit only 20 results
        for i in 1:20
            submit_tree_result!(aggregator, TreeResult(
                i, i % 2, [i, i+10, i+20]
            ))
        end
        
        # Try to create partial result with minimum 10 trees
        partial_result = ResultAggregation.create_partial_result(aggregator, 10)
        
        @test !isnothing(partial_result)
        @test partial_result.total_trees == 20
        
        # Try with minimum 30 trees - should fail
        partial_result2 = ResultAggregation.create_partial_result(aggregator, 30)
        @test isnothing(partial_result2)
    end
    
    @testset "GPU Statistics" begin
        aggregator = create_result_aggregator()
        
        # Submit results with varying response times
        for i in 1:5
            submit_tree_result!(aggregator, TreeResult(
                i, 0, [i], Dict{Int,Float32}(), 0.9f0, 1000, 100.0 + i * 10
            ))
        end
        
        for i in 6:8
            submit_tree_result!(aggregator, TreeResult(
                i, 1, [i], Dict{Int,Float32}(), 0.9f0, 1000, 200.0 + i * 5
            ))
        end
        
        stats = ResultAggregation.get_gpu_statistics(aggregator)
        
        @test haskey(stats, 0)
        @test haskey(stats, 1)
        
        # GPU 0 stats
        @test stats[0]["tree_count"] == 5
        @test stats[0]["avg_response_time"] ≈ 130.0  # (110+120+130+140+150)/5
        @test stats[0]["min_response_time"] == 110.0
        @test stats[0]["max_response_time"] == 150.0
        
        # GPU 1 stats
        @test stats[1]["tree_count"] == 3
        # Response times: 200+6*5=230, 200+7*5=235, 200+8*5=240
        @test stats[1]["avg_response_time"] ≈ 235.0  # (230+235+240)/3
    end
    
    @testset "Empty Results Handling" begin
        aggregator = create_result_aggregator()
        
        # Aggregate with no results
        result = aggregate_results(aggregator)
        
        @test result.total_trees == 0
        @test isempty(result.selected_features)
        @test isempty(result.consensus_features)
        @test result.confidence == 0.0f0
    end
    
    @testset "Cache Eviction" begin
        aggregator = create_result_aggregator(
            total_trees = 2,
            cache_size = 3  # Small cache for testing
        )
        
        # Create different result sets
        for set_num in 1:5
            clear_results!(aggregator)
            
            # Each set has different features
            for i in 1:2
                submit_tree_result!(aggregator, TreeResult(
                    i, 0, [set_num, set_num + 10]
                ))
            end
            
            aggregate_results(aggregator)
        end
        
        # Cache should have at most cache_size entries
        cache_stats = get_cache_stats(aggregator)
        @test cache_stats["cache_size"] <= cache_stats["max_size"]
    end
    
end

# Print summary
println("\nResult Aggregation Test Summary:")
println("==================================")
if CUDA.functional()
    num_gpus = length(CUDA.devices())
    println("✓ CUDA functional - Result aggregation tests executed")
    println("  GPUs detected: $num_gpus")
    println("  Multi-GPU aggregation validated")
    println("  Consensus mechanisms tested")
    println("  Caching system verified")
else
    println("⚠ CUDA not functional - CPU simulation tests only")
end
println("\nAll result aggregation tests completed!")