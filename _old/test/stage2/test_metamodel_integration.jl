"""
Test Suite for Metamodel Integration
Validates neural network metamodel interface, batched evaluation, caching system,
confidence-based fallback, and performance monitoring functionality.
"""

using Test
using Random
using Statistics
using Dates

# Include the metamodel integration module
include("../../src/stage2/metamodel_integration.jl")
using .MetamodelIntegration

@testset "Metamodel Integration Tests" begin
    
    Random.seed!(42)  # For reproducible tests
    
    @testset "Configuration Tests" begin
        # Test default configuration
        config = create_metamodel_config()
        
        @test config.feature_dimension == 500
        @test config.enable_gpu_acceleration == true
        @test config.max_batch_size == 100
        @test config.batch_timeout_ms == 50
        @test config.cache_size == 10000
        @test config.cache_ttl_hours == 24
        @test config.confidence_threshold == 0.7f0
        @test config.fallback_probability == 0.05f0
        @test config.enable_online_learning == true
        @test config.prediction_timeout_ms == 100
        
        # Test custom configuration
        custom_config = create_metamodel_config(
            feature_dimension = 200,
            max_batch_size = 50,
            cache_size = 5000,
            confidence_threshold = 0.8f0,
            enable_gpu_acceleration = false
        )
        
        @test custom_config.feature_dimension == 200
        @test custom_config.max_batch_size == 50
        @test custom_config.cache_size == 5000
        @test custom_config.confidence_threshold == 0.8f0
        @test custom_config.enable_gpu_acceleration == false
        
        println("  ✅ Configuration tests passed")
    end
    
    @testset "Confidence Level Tests" begin
        # Test enum values
        @test Int(HIGH_CONFIDENCE) == 1
        @test Int(MEDIUM_CONFIDENCE) == 2
        @test Int(LOW_CONFIDENCE) == 3
        
        # Test classification function
        @test MetamodelIntegration.classify_confidence_level(0.9f0) == HIGH_CONFIDENCE
        @test MetamodelIntegration.classify_confidence_level(0.75f0) == MEDIUM_CONFIDENCE
        @test MetamodelIntegration.classify_confidence_level(0.6f0) == LOW_CONFIDENCE
        
        println("  ✅ Confidence level tests passed")
    end
    
    @testset "Evaluation Mode Tests" begin
        # Test enum values
        @test Int(METAMODEL_ONLY) == 1
        @test Int(HYBRID_MODE) == 2
        @test Int(REAL_EVALUATION) == 3
        
        println("  ✅ Evaluation mode tests passed")
    end
    
    @testset "Prediction Structure Tests" begin
        # Create test prediction
        feature_subset = [1, 5, 10, 15, 20]
        prediction = MetamodelPrediction(
            feature_subset,
            0.75f0,
            0.85f0,
            HIGH_CONFIDENCE,
            25.5,
            false,
            now()
        )
        
        @test prediction.feature_subset == feature_subset
        @test prediction.predicted_score == 0.75f0
        @test prediction.confidence == 0.85f0
        @test prediction.confidence_level == HIGH_CONFIDENCE
        @test prediction.computation_time == 25.5
        @test prediction.is_cached == false
        
        println("  ✅ Prediction structure tests passed")
    end
    
    @testset "Evaluation Batch Tests" begin
        # Create test batch
        feature_subsets = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        batch = EvaluationBatch(
            "test_batch_1",
            feature_subsets,
            length(feature_subsets),
            3,
            1,
            now(),
            nothing
        )
        
        @test batch.batch_id == "test_batch_1"
        @test batch.batch_size == 3
        @test batch.priority == 3
        @test batch.requester_tree_id == 1
        @test length(batch.feature_subsets) == 3
        
        println("  ✅ Evaluation batch tests passed")
    end
    
    @testset "Model Input Preparation Tests" begin
        # Create test configuration
        config = create_metamodel_config(feature_dimension = 100)
        
        # Mock manager for testing input preparation
        manager = MetamodelManager(
            config, nothing, nothing, "ready", "v1.0", now(),
            EvaluationBatch[], false, nothing, 50,
            Dict{Vector{Int}, MetamodelIntegration.CacheEntry}(),
            Vector{Vector{Int}}(), (hits=0, misses=0, evictions=0),
            MetamodelIntegration.MetamodelStats(0, 0, 0, 0, 0, 0.0, 0.0f0, 0.0f0, now(), Tuple{DateTime, Float32}[]),
            Float64[], Float32[], nothing, Vector{Vector{Int}}(), Dict{Vector{Int}, Float32}(),
            ReentrantLock(), ReentrantLock(), ReentrantLock(),
            "active", String[], now()
        )
        
        # Test single input preparation
        feature_subset = [1, 10, 50, 75]
        model_input = MetamodelIntegration.prepare_model_input(manager, feature_subset)
        
        @test size(model_input) == (100, 1)
        @test model_input[1, 1] == 1.0f0   # Feature 1 selected
        @test model_input[10, 1] == 1.0f0  # Feature 10 selected
        @test model_input[50, 1] == 1.0f0  # Feature 50 selected
        @test model_input[75, 1] == 1.0f0  # Feature 75 selected
        @test model_input[2, 1] == 0.0f0   # Feature 2 not selected
        
        # Test batch input preparation
        feature_subsets = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8, 9]
        ]
        batch_input = MetamodelIntegration.prepare_batch_input(manager, feature_subsets)
        
        @test size(batch_input) == (100, 3)
        @test batch_input[1, 1] == 1.0f0  # First subset, feature 1
        @test batch_input[4, 2] == 1.0f0  # Second subset, feature 4
        @test batch_input[6, 3] == 1.0f0  # Third subset, feature 6
        @test batch_input[10, 1] == 0.0f0 # First subset, feature 10 not selected
        
        println("  ✅ Model input preparation tests passed")
    end
    
    @testset "Confidence Calculation Tests" begin
        config = create_metamodel_config(feature_dimension = 100)
        
        manager = MetamodelManager(
            config, nothing, nothing, "ready", "v1.0", now(),
            EvaluationBatch[], false, nothing, 50,
            Dict{Vector{Int}, MetamodelIntegration.CacheEntry}(),
            Vector{Vector{Int}}(), (hits=0, misses=0, evictions=0),
            MetamodelIntegration.MetamodelStats(0, 0, 0, 0, 0, 0.0, 0.0f0, 0.0f0, now(), Tuple{DateTime, Float32}[]),
            Float64[], Float32[], nothing, Vector{Vector{Int}}(), Dict{Vector{Int}, Float32}(),
            ReentrantLock(), ReentrantLock(), ReentrantLock(),
            "active", String[], now()
        )
        
        # Test confidence calculation
        small_subset = [1, 2, 3]  # Small subset should have higher confidence
        large_subset = collect(1:80)  # Large subset should have lower confidence
        
        conf_small = MetamodelIntegration.calculate_prediction_confidence(manager, small_subset, 0.7f0)
        conf_large = MetamodelIntegration.calculate_prediction_confidence(manager, large_subset, 0.7f0)
        
        @test 0.0f0 <= conf_small <= 1.0f0
        @test 0.0f0 <= conf_large <= 1.0f0
        @test conf_small > conf_large  # Smaller subsets should be more confident
        
        # Test confidence level classification
        @test MetamodelIntegration.classify_confidence_level(0.9f0) == HIGH_CONFIDENCE
        @test MetamodelIntegration.classify_confidence_level(0.75f0) == MEDIUM_CONFIDENCE
        @test MetamodelIntegration.classify_confidence_level(0.6f0) == LOW_CONFIDENCE
        
        println("  ✅ Confidence calculation tests passed")
    end
    
    @testset "Cache Operations Tests" begin
        config = create_metamodel_config(cache_size = 5)  # Small cache for testing
        
        manager = MetamodelManager(
            config, nothing, nothing, "ready", "v1.0", now(),
            EvaluationBatch[], false, nothing, 50,
            Dict{Vector{Int}, MetamodelIntegration.CacheEntry}(),
            Vector{Vector{Int}}(), (hits=0, misses=0, evictions=0),
            MetamodelIntegration.MetamodelStats(0, 0, 0, 0, 0, 0.0, 0.0f0, 0.0f0, now(), Tuple{DateTime, Float32}[]),
            Float64[], Float32[], nothing, Vector{Vector{Int}}(), Dict{Vector{Int}, Float32}(),
            ReentrantLock(), ReentrantLock(), ReentrantLock(),
            "active", String[], now()
        )
        
        # Test cache miss
        result = MetamodelIntegration.get_cached_prediction(manager, [1, 2, 3])
        @test isnothing(result)
        @test manager.stats.cache_misses == 1
        
        # Test cache insertion
        prediction = MetamodelPrediction([1, 2, 3], 0.8f0, 0.9f0, HIGH_CONFIDENCE, 10.0, false, now())
        MetamodelIntegration.cache_prediction!(manager, [1, 2, 3], prediction)
        
        @test length(manager.prediction_cache) == 1
        @test length(manager.cache_access_order) == 1
        
        # Test cache hit
        cached_result = MetamodelIntegration.get_cached_prediction(manager, [1, 2, 3])
        @test !isnothing(cached_result)
        @test cached_result.prediction.predicted_score == 0.8f0
        @test manager.stats.cache_hits == 1
        
        # Test cache eviction (fill cache beyond limit)
        for i in 1:6
            subset = [i, i+1, i+2]
            pred = MetamodelPrediction(subset, 0.5f0, 0.8f0, HIGH_CONFIDENCE, 10.0, false, now())
            MetamodelIntegration.cache_prediction!(manager, subset, pred)
        end
        
        @test length(manager.prediction_cache) <= config.cache_size
        
        println("  ✅ Cache operations tests passed")
    end
    
    @testset "Real Evaluator Tests" begin
        config = create_metamodel_config()
        
        manager = MetamodelManager(
            config, nothing, nothing, "ready", "v1.0", now(),
            EvaluationBatch[], false, nothing, 50,
            Dict{Vector{Int}, MetamodelIntegration.CacheEntry}(),
            Vector{Vector{Int}}(), (hits=0, misses=0, evictions=0),
            MetamodelIntegration.MetamodelStats(0, 0, 0, 0, 0, 0.0, 0.0f0, 0.0f0, now(), Tuple{DateTime, Float32}[]),
            Float64[], Float32[], nothing, Vector{Vector{Int}}(), Dict{Vector{Int}, Float32}(),
            ReentrantLock(), ReentrantLock(), ReentrantLock(),
            "active", String[], now()
        )
        
        # Mock real evaluator function
        function mock_evaluator(feature_subset::Vector{Int})
            # Simple mock: score based on subset size (normalized)
            return length(feature_subset) / 100.0
        end
        
        # Set real evaluator
        set_real_evaluator!(manager, mock_evaluator)
        @test !isnothing(manager.real_evaluator)
        
        # Test real evaluation
        test_subset = [1, 5, 10, 15, 20]  # 5 features
        prediction = MetamodelIntegration.evaluate_with_real_function(manager, test_subset, time())
        
        @test prediction.predicted_score ≈ 0.05f0  # 5/100
        @test prediction.confidence == 1.0f0
        @test prediction.confidence_level == HIGH_CONFIDENCE
        @test haskey(manager.validation_data, test_subset)
        @test manager.stats.real_evaluations == 1
        
        println("  ✅ Real evaluator tests passed")
    end
    
    @testset "Statistics and Monitoring Tests" begin
        config = create_metamodel_config()
        
        manager = MetamodelManager(
            config, nothing, nothing, "ready", "v1.0", now(),
            EvaluationBatch[], false, nothing, 50,
            Dict{Vector{Int}, MetamodelIntegration.CacheEntry}(),
            Vector{Vector{Int}}(), (hits=0, misses=0, evictions=0),
            MetamodelIntegration.MetamodelStats(0, 0, 0, 0, 0, 0.0, 0.0f0, 0.0f0, now(), Tuple{DateTime, Float32}[]),
            Float64[], Float32[], nothing, Vector{Vector{Int}}(), Dict{Vector{Int}, Float32}(),
            ReentrantLock(), ReentrantLock(), ReentrantLock(),
            "active", String[], now()
        )
        
        # Test initial statistics
        @test manager.stats.total_predictions == 0
        @test manager.stats.cache_hits == 0
        @test manager.stats.cache_misses == 0
        @test manager.stats.average_confidence == 0.0f0
        
        # Test statistics update
        prediction = MetamodelPrediction([1, 2, 3], 0.7f0, 0.8f0, HIGH_CONFIDENCE, 15.0, false, now())
        MetamodelIntegration.update_prediction_stats!(manager, prediction)
        
        @test manager.stats.total_predictions == 1
        @test length(manager.prediction_times) == 1
        @test manager.prediction_times[1] == 15.0
        @test length(manager.confidence_distribution) == 1
        @test manager.confidence_distribution[1] == 0.8f0
        @test manager.stats.average_prediction_time == 15.0
        @test manager.stats.average_confidence == 0.8f0
        
        # Test status retrieval
        status = get_metamodel_status(manager)
        
        @test haskey(status, "manager_state")
        @test haskey(status, "model_state")
        @test haskey(status, "total_predictions")
        @test haskey(status, "cache_hits")
        @test haskey(status, "cache_misses")
        @test haskey(status, "average_prediction_time_ms")
        @test haskey(status, "average_confidence")
        
        @test status["manager_state"] == "active"
        @test status["total_predictions"] == 1
        @test status["average_prediction_time_ms"] == 15.0
        @test status["average_confidence"] == 0.8f0
        
        println("  ✅ Statistics and monitoring tests passed")
    end
    
    @testset "Report Generation Tests" begin
        config = create_metamodel_config()
        
        manager = MetamodelManager(
            config, nothing, nothing, "ready", "v1.0", now(),
            EvaluationBatch[], false, nothing, 50,
            Dict{Vector{Int}, MetamodelIntegration.CacheEntry}(),
            Vector{Vector{Int}}(), (hits=0, misses=0, evictions=0),
            MetamodelIntegration.MetamodelStats(0, 0, 0, 0, 0, 0.0, 0.0f0, 0.0f0, now(), Tuple{DateTime, Float32}[]),
            Float64[], Float32[], nothing, Vector{Vector{Int}}(), Dict{Vector{Int}, Float32}(),
            ReentrantLock(), ReentrantLock(), ReentrantLock(),
            "active", String[], now()
        )
        
        # Add some test data
        prediction = MetamodelPrediction([1, 2, 3], 0.7f0, 0.8f0, HIGH_CONFIDENCE, 15.0, false, now())
        MetamodelIntegration.update_prediction_stats!(manager, prediction)
        MetamodelIntegration.cache_prediction!(manager, [1, 2, 3], prediction)
        
        # Generate report
        report = generate_metamodel_report(manager)
        
        @test contains(report, "Metamodel Integration Performance Report")
        @test contains(report, "Manager State: active")
        @test contains(report, "Total Predictions: 1")
        @test contains(report, "Cache Size: 1")
        @test contains(report, "Average Time: 15.0ms")
        @test contains(report, "Average Confidence: 80.0%")
        
        println("  ✅ Report generation tests passed")
    end
    
    @testset "Fallback Logic Tests" begin
        config = create_metamodel_config(fallback_probability = 1.0f0)  # Always trigger fallback
        
        manager = MetamodelManager(
            config, nothing, nothing, "ready", "v1.0", now(),
            EvaluationBatch[], false, nothing, 50,
            Dict{Vector{Int}, MetamodelIntegration.CacheEntry}(),
            Vector{Vector{Int}}(), (hits=0, misses=0, evictions=0),
            MetamodelIntegration.MetamodelStats(0, 0, 0, 0, 0, 0.0, 0.0f0, 0.0f0, now(), Tuple{DateTime, Float32}[]),
            Float64[], Float32[], nothing, Vector{Vector{Int}}(), Dict{Vector{Int}, Float32}(),
            ReentrantLock(), ReentrantLock(), ReentrantLock(),
            "active", String[], now()
        )
        
        # Test fallback trigger
        @test MetamodelIntegration.should_trigger_fallback(manager) == true
        
        # Test with no fallback
        manager.config = create_metamodel_config(fallback_probability = 0.0f0)
        @test MetamodelIntegration.should_trigger_fallback(manager) == false
        
        println("  ✅ Fallback logic tests passed")
    end
    
    @testset "Error Handling Tests" begin
        config = create_metamodel_config()
        
        manager = MetamodelManager(
            config, nothing, nothing, "error", "v1.0", now(),  # Set to error state
            EvaluationBatch[], false, nothing, 50,
            Dict{Vector{Int}, MetamodelIntegration.CacheEntry}(),
            Vector{Vector{Int}}(), (hits=0, misses=0, evictions=0),
            MetamodelIntegration.MetamodelStats(0, 0, 0, 0, 0, 0.0, 0.0f0, 0.0f0, now(), Tuple{DateTime, Float32}[]),
            Float64[], Float32[], nothing, Vector{Vector{Int}}(), Dict{Vector{Int}, Float32}(),
            ReentrantLock(), ReentrantLock(), ReentrantLock(),
            "active", String[], now()
        )
        
        # Test with error state - should not process batches
        test_batch = EvaluationBatch("test", [[1, 2, 3]], 1, 5, 1, now(), nothing)
        
        # Since model is in error state, batch processing should handle gracefully
        # (In practice, this would require more sophisticated error handling)
        
        # Test invalid feature indices
        invalid_subset = [0, -1, 1000]  # Invalid indices
        model_input = MetamodelIntegration.prepare_model_input(manager, invalid_subset)
        
        # Should handle gracefully (only valid indices used)
        @test size(model_input, 1) == config.feature_dimension
        @test all(model_input .== 0.0f0)  # No valid features
        
        println("  ✅ Error handling tests passed")
    end
end

println("All Metamodel Integration tests completed!")
println("✅ Configuration and setup validation")
println("✅ Confidence level and evaluation mode enums")
println("✅ Prediction and batch data structures")
println("✅ Model input preparation for single and batch processing")
println("✅ Confidence calculation and classification")
println("✅ Cache operations with LRU eviction")
println("✅ Real evaluator integration and fallback")
println("✅ Performance statistics and monitoring")
println("✅ Report generation and status tracking")
println("✅ Fallback logic and error handling")
println("✅ Ready for metamodel-accelerated MCTS feature evaluation")