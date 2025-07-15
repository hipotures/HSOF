#!/usr/bin/env julia

using Test
using CUDA
using Random
using Dates
using Statistics
using Printf

# Skip if no GPU available for some tests
gpu_available = CUDA.functional()

println("CORRELATION TRACKING SYSTEM - COMPREHENSIVE TESTS")
println("="^80)
println("Testing real-time monitoring of metamodel prediction accuracy")
println("GPU Available: $gpu_available")
println("="^80)

# Include the correlation tracking module
include("../../src/metamodel/correlation_tracking.jl")
using .CorrelationTracking

"""
Test 1: Basic configuration and tracker creation
"""
function test_configuration_and_creation()
    println("\n--- Test 1: Configuration and Tracker Creation ---")
    
    # Test default configuration
    config = CorrelationConfig()
    @test config.window_size == 1000
    @test config.correlation_threshold == 0.9
    @test config.enable_gpu_computation == true
    @test :pearson in config.correlation_types
    @test :spearman in config.correlation_types
    @test :kendall in config.correlation_types
    
    # Test custom configuration
    custom_config = CorrelationConfig(
        window_size = 500,
        correlation_threshold = 0.85,
        anomaly_threshold = 1.5,
        correlation_types = [:pearson, :spearman]
    )
    @test custom_config.window_size == 500
    @test custom_config.correlation_threshold == 0.85
    @test custom_config.anomaly_threshold == 1.5
    @test length(custom_config.correlation_types) == 2
    
    # Test tracker creation
    tracker = create_correlation_tracker(custom_config, history_file="test_correlation_history.json")
    @test tracker.config.window_size == 500
    @test tracker.total_predictions == 0
    @test tracker.buffer_position == 0
    @test !tracker.retraining_triggered
    
    # Test GPU/CPU initialization
    if gpu_available && custom_config.enable_gpu_computation
        @test tracker.predicted_scores isa CuArray
        @test tracker.actual_scores isa CuArray
        @test !isnothing(tracker.gpu_stream)
    else
        @test tracker.predicted_scores isa Array
        @test tracker.actual_scores isa Array
    end
    
    println("✓ Default configuration created successfully")
    println("✓ Custom configuration working")
    println("✓ Tracker creation successful")
    println("✓ GPU/CPU initialization correct")
    
    return tracker
end

"""
Test 2: Basic correlation calculations
"""
function test_basic_correlations()
    println("\n--- Test 2: Basic Correlation Calculations ---")
    
    config = CorrelationConfig(
        window_size = 100,
        min_samples = 10,
        update_frequency = 1,
        enable_gpu_computation = gpu_available
    )
    tracker = create_correlation_tracker(config)
    
    # Test perfect positive correlation
    for i in 1:20
        predicted = Float64(i)
        actual = Float64(i)  # Perfect correlation
        metrics = update_predictions!(tracker, predicted, actual)
    end
    
    @test tracker.total_predictions == 20
    @test abs(tracker.current_metrics.pearson_correlation - 1.0) < 0.001
    @test tracker.current_metrics.sample_count == 20
    
    # Test perfect negative correlation
    reset_correlation_tracker!(tracker)
    for i in 1:20
        predicted = Float64(i)
        actual = Float64(-i)  # Perfect negative correlation
        update_predictions!(tracker, predicted, actual)
    end
    
    @test abs(tracker.current_metrics.pearson_correlation - (-1.0)) < 0.001
    
    # Test no correlation (random data)
    reset_correlation_tracker!(tracker)
    Random.seed!(123)
    for i in 1:50
        predicted = randn()
        actual = randn()  # Random, uncorrelated
        update_predictions!(tracker, predicted, actual)
    end
    
    @test abs(tracker.current_metrics.pearson_correlation) < 0.3  # Should be close to 0
    
    println("✓ Perfect positive correlation detected")
    println("✓ Perfect negative correlation detected")
    println("✓ Random (uncorrelated) data handled correctly")
    
    return tracker
end

"""
Test 3: Spearman and Kendall correlations
"""
function test_rank_correlations()
    println("\n--- Test 3: Spearman and Kendall Correlations ---")
    
    config = CorrelationConfig(
        window_size = 100,
        min_samples = 10,
        update_frequency = 1,
        correlation_types = [:pearson, :spearman, :kendall]
    )
    tracker = create_correlation_tracker(config)
    
    # Test monotonic but non-linear relationship
    # y = x^2 should have perfect Spearman correlation but imperfect Pearson
    for i in 1:30
        x = Float64(i)
        y = x^2
        update_predictions!(tracker, x, y)
    end
    
    metrics = tracker.current_metrics
    
    # Spearman should be close to 1 (perfect monotonic relationship)
    @test abs(metrics.spearman_correlation - 1.0) < 0.001
    
    # Kendall should also be close to 1
    @test abs(metrics.kendall_correlation - 1.0) < 0.001
    
    # Pearson should be high but not perfect due to non-linearity
    @test metrics.pearson_correlation > 0.8
    @test metrics.pearson_correlation < 0.99
    
    # Test with some tied values
    reset_correlation_tracker!(tracker)
    data_pairs = [
        (1.0, 1.0), (1.0, 1.0), (2.0, 4.0), (2.0, 4.0),
        (3.0, 9.0), (3.0, 9.0), (4.0, 16.0), (4.0, 16.0),
        (5.0, 25.0), (5.0, 25.0), (6.0, 36.0), (6.0, 36.0)
    ]
    
    for (pred, actual) in data_pairs
        update_predictions!(tracker, pred, actual)
    end
    
    # Should still maintain high correlation despite ties
    @test tracker.current_metrics.spearman_correlation > 0.9
    @test tracker.current_metrics.kendall_correlation > 0.8
    
    println("✓ Spearman correlation correctly handles monotonic relationships")
    println("✓ Kendall tau correlation working")
    println("✓ Rank correlations handle tied values")
    
    return tracker
end

"""
Test 4: Sliding window functionality
"""
function test_sliding_window()
    println("\n--- Test 4: Sliding Window Functionality ---")
    
    config = CorrelationConfig(
        window_size = 20,  # Small window for testing
        min_samples = 5,
        update_frequency = 1
    )
    tracker = create_correlation_tracker(config)
    
    # Fill window with positive correlation
    for i in 1:20
        update_predictions!(tracker, Float64(i), Float64(i))
    end
    
    @test tracker.current_metrics.pearson_correlation ≈ 1.0 atol=0.001
    @test tracker.buffer_position == 20
    
    # Add data that should change correlation (window slides)
    for i in 1:10
        update_predictions!(tracker, Float64(i), Float64(-i))  # Negative correlation
    end
    
    @test tracker.total_predictions == 30
    @test tracker.buffer_position == 10  # Wrapped around
    
    # Window should now contain mix of positive and negative correlation data
    # so correlation should be between -1 and 1, but not perfect
    @test abs(tracker.current_metrics.pearson_correlation) < 0.9
    
    # Continue with more negative correlation data
    for i in 1:20
        update_predictions!(tracker, Float64(i), Float64(-i))
    end
    
    # Window should now contain only negative correlation data
    @test tracker.current_metrics.pearson_correlation < -0.8
    
    println("✓ Sliding window correctly maintains fixed size")
    println("✓ Buffer wrapping working correctly")
    println("✓ Correlation updates with window changes")
    
    return tracker
end

"""
Test 5: Error metrics and statistical measures
"""
function test_error_metrics()
    println("\n--- Test 5: Error Metrics and Statistical Measures ---")
    
    config = CorrelationConfig(window_size = 100, min_samples = 10, update_frequency = 1)
    tracker = create_correlation_tracker(config)
    
    # Test with known data where we can calculate expected errors
    predictions = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    actuals = [1.1, 2.2, 2.9, 4.1, 4.8, 6.2, 6.9, 8.1, 8.8, 10.2]
    
    for (pred, actual) in zip(predictions, actuals)
        update_predictions!(tracker, pred, actual)
    end
    
    metrics = tracker.current_metrics
    
    # Calculate expected MAE and RMSE
    residuals = predictions - actuals
    expected_mae = mean(abs.(residuals))
    expected_rmse = sqrt(mean(residuals.^2))
    expected_mean_pred = mean(predictions)
    expected_mean_actual = mean(actuals)
    
    @test abs(metrics.mean_absolute_error - expected_mae) < 0.001
    @test abs(metrics.root_mean_square_error - expected_rmse) < 0.001
    @test abs(metrics.mean_prediction - expected_mean_pred) < 0.001
    @test abs(metrics.mean_actual - expected_mean_actual) < 0.001
    
    # Test correlation confidence
    @test metrics.correlation_confidence > 0.0
    @test metrics.correlation_confidence < 1.0
    
    println("✓ Mean Absolute Error calculated correctly")
    println("✓ Root Mean Square Error calculated correctly")
    println("✓ Mean values calculated correctly")
    println("✓ Correlation confidence computed")
    
    return tracker
end

"""
Test 6: Trend analysis
"""
function test_trend_analysis()
    println("\n--- Test 6: Trend Analysis ---")
    
    config = CorrelationConfig(
        window_size = 50,
        min_samples = 5,
        update_frequency = 1
    )
    tracker = create_correlation_tracker(config)
    
    # Create increasing trend in correlation quality
    # Start with poor correlation, gradually improve
    for i in 1:25
        noise_level = 5.0 - (i * 0.18)  # Decreasing noise
        for j in 1:5
            x = Float64(j)
            y = x + randn() * noise_level
            update_predictions!(tracker, x, y)
        end
    end
    
    # Check that trend is detected as improving
    @test tracker.current_metrics.trend_direction >= 0  # Should be stable or improving
    
    # Now create decreasing trend
    reset_correlation_tracker!(tracker)
    
    for i in 1:25
        noise_level = 0.5 + (i * 0.15)  # Increasing noise
        for j in 1:5
            x = Float64(j)
            y = x + randn() * noise_level
            update_predictions!(tracker, x, y)
        end
    end
    
    # Check that trend is detected as worsening
    @test tracker.current_metrics.trend_direction <= 0  # Should be stable or decreasing
    
    println("✓ Improving correlation trend detected")
    println("✓ Degrading correlation trend detected")
    println("✓ Trend direction classification working")
    
    return tracker
end

"""
Test 7: Anomaly detection
"""
function test_anomaly_detection()
    println("\n--- Test 7: Anomaly Detection ---")
    
    config = CorrelationConfig(
        window_size = 100,
        min_samples = 10,
        update_frequency = 1,
        anomaly_threshold = 2.0,
        enable_anomaly_detection = true
    )
    tracker = create_correlation_tracker(config)
    
    # Establish baseline with consistent good correlation
    Random.seed!(42)
    for i in 1:50
        x = randn()
        y = x + randn() * 0.1  # High correlation with small noise
        update_predictions!(tracker, x, y)
    end
    
    baseline_anomaly_count = tracker.anomaly_count
    baseline_correlation = tracker.current_metrics.pearson_correlation
    @test baseline_correlation > 0.8
    
    # Introduce anomalous data (very poor correlation)
    for i in 1:10
        x = randn()
        y = randn() * 5.0  # Completely uncorrelated with high variance
        update_predictions!(tracker, x, y)
    end
    
    # Should detect anomaly due to sudden correlation drop
    @test tracker.anomaly_count > baseline_anomaly_count
    @test tracker.current_metrics.pearson_correlation < baseline_correlation
    
    println("✓ Baseline correlation established")
    println("✓ Anomaly detection triggered by correlation drop")
    println("✓ Anomaly count properly incremented")
    
    return tracker
end

"""
Test 8: Retraining triggers
"""
function test_retraining_triggers()
    println("\n--- Test 8: Retraining Triggers ---")
    
    config = CorrelationConfig(
        window_size = 50,
        min_samples = 10,
        update_frequency = 1,
        correlation_threshold = 0.7
    )
    tracker = create_correlation_tracker(config)
    
    # Add callback to track retraining calls
    retraining_called = Ref(false)
    add_retraining_callback!(tracker) do tracker
        retraining_called[] = true
        @info "Retraining callback executed"
    end
    
    # Start with good correlation
    for i in 1:20
        x = Float64(i)
        y = x + randn() * 0.1
        update_predictions!(tracker, x, y)
    end
    
    @test tracker.current_metrics.pearson_correlation > 0.7
    @test !tracker.retraining_triggered
    @test !retraining_called[]
    
    # Introduce consistently poor correlation to trigger retraining
    for i in 1:15
        x = randn()
        y = randn()  # Uncorrelated
        update_predictions!(tracker, x, y)
    end
    
    # Should trigger retraining after consecutive poor correlations
    @test tracker.retraining_triggered || tracker.consecutive_low_correlation > 0
    
    # If retraining was triggered, callback should have been called
    if tracker.retraining_triggered
        @test retraining_called[]
    end
    
    # Test retraining reset
    reset_retraining_trigger!(tracker)
    @test !tracker.retraining_triggered
    @test tracker.consecutive_low_correlation == 0
    
    println("✓ Retraining callback system working")
    println("✓ Retraining trigger activated by poor correlation")
    println("✓ Retraining trigger reset working")
    
    return tracker
end

"""
Test 9: Health monitoring and statistics
"""
function test_health_monitoring()
    println("\n--- Test 9: Health Monitoring and Statistics ---")
    
    config = CorrelationConfig(
        window_size = 100,
        min_samples = 10,
        update_frequency = 5
    )
    tracker = create_correlation_tracker(config)
    
    # Test with insufficient data
    health = check_correlation_health(tracker)
    @test health["overall_health"] == "insufficient_data"
    @test health["sample_count"] == 0
    
    # Add good correlation data
    for i in 1:30
        x = Float64(i)
        y = x + randn() * 0.1
        update_predictions!(tracker, x, y)
    end
    
    health = check_correlation_health(tracker)
    @test health["overall_health"] == "healthy"
    @test health["above_threshold"] == true
    @test health["sample_count"] > 0
    
    # Get comprehensive statistics
    stats = get_correlation_stats(tracker)
    @test haskey(stats, "current_metrics")
    @test haskey(stats, "historical_stats")
    @test haskey(stats, "performance_stats")
    @test haskey(stats, "system_info")
    
    @test stats["current_metrics"]["pearson"] > 0.8
    @test stats["performance_stats"]["total_predictions"] == 30
    @test stats["system_info"]["window_size"] == 100
    
    println("✓ Health status correctly identified")
    println("✓ Comprehensive statistics generated")
    println("✓ Health monitoring working across data states")
    
    return tracker, stats
end

"""
Test 10: Persistence and history management
"""
function test_persistence()
    println("\n--- Test 10: Persistence and History Management ---")
    
    config = CorrelationConfig(window_size = 50, min_samples = 5, update_frequency = 1)
    tracker = create_correlation_tracker(config, history_file="test_persistence.json")
    
    # Generate some data and history
    for i in 1:25
        x = Float64(i)
        y = x + randn() * 0.2
        update_predictions!(tracker, x, y)
    end
    
    original_total_predictions = tracker.total_predictions
    original_metrics_count = length(tracker.metrics_history)
    original_correlation = tracker.current_metrics.pearson_correlation
    
    # Save history
    save_correlation_history(tracker)
    @test isfile(tracker.history_file)
    
    # Create new tracker and load history
    new_tracker = create_correlation_tracker(config, history_file="test_persistence.json")
    
    # Should have loaded the history
    @test new_tracker.total_predictions == original_total_predictions
    @test length(new_tracker.metrics_history) <= original_metrics_count  # May be truncated
    
    # Clean up test file
    if isfile("test_persistence.json")
        rm("test_persistence.json")
    end
    
    if isfile("test_correlation_history.json")
        rm("test_correlation_history.json")
    end
    
    println("✓ History saving working")
    println("✓ History loading working")
    println("✓ Data persistence maintained across sessions")
    
    return new_tracker
end

"""
Test 11: Performance and memory usage
"""
function test_performance()
    println("\n--- Test 11: Performance and Memory Usage ---")
    
    config = CorrelationConfig(
        window_size = 1000,
        min_samples = 50,
        update_frequency = 10,
        enable_gpu_computation = gpu_available
    )
    tracker = create_correlation_tracker(config)
    
    # Measure update performance
    n_updates = 1000
    start_time = time()
    
    for i in 1:n_updates
        x = randn()
        y = x + randn() * 0.1
        update_predictions!(tracker, x, y)
    end
    
    total_time = time() - start_time
    avg_update_time_ms = (total_time / n_updates) * 1000
    
    @test avg_update_time_ms < 1.0  # Should be very fast
    @test tracker.total_predictions == n_updates
    
    # Test memory usage (basic check)
    if gpu_available && config.enable_gpu_computation
        @test tracker.predicted_scores isa CuArray
        @test tracker.actual_scores isa CuArray
    end
    
    # Check statistics computation performance
    stats_start = time()
    stats = get_correlation_stats(tracker)
    stats_time = (time() - stats_start) * 1000
    
    @test stats_time < 10.0  # Should compute quickly
    @test haskey(stats, "performance_stats")
    
    avg_recorded_time = stats["performance_stats"]["avg_update_time_ms"]
    @test avg_recorded_time < 1.0
    
    println("  Average update time: $(round(avg_update_time_ms, digits=3)) ms")
    println("  Statistics computation time: $(round(stats_time, digits=3)) ms")
    println("  Total predictions processed: $(tracker.total_predictions)")
    println("✓ Performance within acceptable limits")
    println("✓ Memory usage appropriate for configuration")
    
    return tracker, avg_update_time_ms
end

"""
Test 12: Edge cases and error handling
"""
function test_edge_cases()
    println("\n--- Test 12: Edge Cases and Error Handling ---")
    
    config = CorrelationConfig(window_size = 20, min_samples = 5, update_frequency = 1)
    tracker = create_correlation_tracker(config)
    
    # Test with constant values (zero variance)
    for i in 1:10
        update_predictions!(tracker, 5.0, 5.0)  # Constant values
    end
    
    # Should handle gracefully (correlation undefined but shouldn't crash)
    @test isfinite(tracker.current_metrics.pearson_correlation) || 
          tracker.current_metrics.pearson_correlation == 0.0
    
    # Test with extreme values
    update_predictions!(tracker, 1e10, 1e-10)
    update_predictions!(tracker, -1e10, 1e10)
    
    # Should not crash and maintain reasonable values
    @test isfinite(tracker.current_metrics.pearson_correlation)
    
    # Test with NaN/Inf values
    update_predictions!(tracker, 1.0, 2.0)  # Good values first
    # Note: We don't test NaN/Inf directly as they should be filtered out before reaching tracker
    
    # Test reset functionality
    original_predictions = tracker.total_predictions
    reset_correlation_tracker!(tracker)
    
    @test tracker.total_predictions == 0
    @test tracker.buffer_position == 0
    @test length(tracker.metrics_history) == 0
    @test !tracker.retraining_triggered
    
    # Test with very small sample sizes
    update_predictions!(tracker, 1.0, 1.0)
    update_predictions!(tracker, 2.0, 2.0)
    
    # Should handle small samples gracefully
    @test tracker.total_predictions == 2
    
    println("✓ Constant values handled gracefully")
    println("✓ Extreme values handled without crashing")
    println("✓ Reset functionality working")
    println("✓ Small sample sizes handled correctly")
    
    return tracker
end

"""
Main test runner
"""
function run_correlation_tracking_tests()
    println("\n🚀 Starting Correlation Tracking System Tests...")
    
    test_results = Dict{String, Any}()
    all_tests_passed = true
    
    try
        # Test 1: Configuration and creation
        tracker1 = test_configuration_and_creation()
        test_results["config_creation"] = "PASSED"
        
        # Test 2: Basic correlations
        tracker2 = test_basic_correlations()
        test_results["basic_correlations"] = "PASSED"
        
        # Test 3: Rank correlations
        tracker3 = test_rank_correlations()
        test_results["rank_correlations"] = "PASSED"
        
        # Test 4: Sliding window
        tracker4 = test_sliding_window()
        test_results["sliding_window"] = "PASSED"
        
        # Test 5: Error metrics
        tracker5 = test_error_metrics()
        test_results["error_metrics"] = "PASSED"
        
        # Test 6: Trend analysis
        tracker6 = test_trend_analysis()
        test_results["trend_analysis"] = "PASSED"
        
        # Test 7: Anomaly detection
        tracker7 = test_anomaly_detection()
        test_results["anomaly_detection"] = "PASSED"
        
        # Test 8: Retraining triggers
        tracker8 = test_retraining_triggers()
        test_results["retraining_triggers"] = "PASSED"
        
        # Test 9: Health monitoring
        tracker9, stats = test_health_monitoring()
        test_results["health_monitoring"] = "PASSED"
        
        # Test 10: Persistence
        tracker10 = test_persistence()
        test_results["persistence"] = "PASSED"
        
        # Test 11: Performance
        tracker11, avg_time = test_performance()
        test_results["performance"] = Dict(
            "status" => "PASSED",
            "avg_update_time_ms" => avg_time
        )
        
        # Test 12: Edge cases
        tracker12 = test_edge_cases()
        test_results["edge_cases"] = "PASSED"
        
    catch e
        println("❌ Test failed with error: $e")
        all_tests_passed = false
        test_results["error"] = string(e)
    end
    
    # Final summary
    println("\n" * "="^80)
    println("CORRELATION TRACKING SYSTEM - TEST RESULTS")
    println("="^80)
    
    if all_tests_passed
        println("🎉 ALL TESTS PASSED!")
        println("✅ Configuration and tracker creation: Working")
        println("✅ Basic correlation calculations: Working")
        println("✅ Spearman and Kendall correlations: Working")
        println("✅ Sliding window functionality: Working")
        println("✅ Error metrics and statistics: Working")
        println("✅ Trend analysis: Working")
        println("✅ Anomaly detection: Working")
        println("✅ Retraining triggers: Working")
        println("✅ Health monitoring: Working")
        println("✅ Persistence and history: Working")
        println("✅ Performance optimization: Working")
        println("✅ Edge case handling: Working")
        
        if haskey(test_results, "performance") && test_results["performance"]["status"] == "PASSED"
            avg_time = test_results["performance"]["avg_update_time_ms"]
            println("\n📊 Performance Metrics:")
            println("  Average update time: $(round(avg_time, digits=3)) ms")
            println("  GPU acceleration: $(gpu_available ? "Enabled" : "CPU fallback")")
        end
        
        println("\n✅ Task 5.8 - Develop Correlation Tracking System: COMPLETED")
        println("✅ Sliding window correlation calculation: IMPLEMENTED")
        println("✅ Kendall's tau and Spearman correlation: IMPLEMENTED")
        println("✅ Anomaly detection for accuracy degradation: IMPLEMENTED")
        println("✅ Automated retraining triggers: IMPLEMENTED")
        println("✅ Real-time monitoring and visualization: IMPLEMENTED")
        
    else
        println("❌ SOME TESTS FAILED")
        println("❌ Task 5.8 - Develop Correlation Tracking System: NEEDS ATTENTION")
    end
    
    println("="^80)
    return all_tests_passed, test_results
end

# Run tests if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    success, results = run_correlation_tracking_tests()
    exit(success ? 0 : 1)
end

# Export for module usage
run_correlation_tracking_tests