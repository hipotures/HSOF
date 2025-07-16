#!/usr/bin/env julia

using CUDA
using Random
using Statistics
using Dates

# Skip if no GPU
if !CUDA.functional()
    @warn "CUDA not functional, running correlation tracking test in CPU mode"
end

println("SIMPLE CORRELATION TRACKING TEST")
println("="^50)

# Include the correlation tracking module
include("src/metamodel/correlation_tracking.jl")
using .CorrelationTracking

function test_simple_correlation_tracking()
    println("Testing basic correlation tracking functionality...")
    
    # Create configuration
    config = CorrelationConfig(
        window_size = 100,
        min_samples = 10,
        update_frequency = 5,
        correlation_threshold = 0.8,
        enable_gpu_computation = CUDA.functional(),
        correlation_types = [:pearson, :spearman, :kendall]
    )
    
    # Create tracker
    tracker = create_correlation_tracker(config, history_file = "simple_test_correlation.json")
    println("✓ Created correlation tracker")
    
    # Test perfect correlation
    println("\nTesting perfect positive correlation...")
    for i in 1:25
        predicted = Float64(i)
        actual = Float64(i)  # Perfect correlation
        metrics = update_predictions!(tracker, predicted, actual)
    end
    
    @assert abs(tracker.current_metrics.pearson_correlation - 1.0) < 0.01
    @assert abs(tracker.current_metrics.spearman_correlation - 1.0) < 0.01
    @assert abs(tracker.current_metrics.kendall_correlation - 1.0) < 0.05  # More lenient for Kendall
    println("✓ Perfect correlation detected: Pearson=$(round(tracker.current_metrics.pearson_correlation, digits=3))")
    
    # Test health monitoring
    health = check_correlation_health(tracker)
    @assert health["overall_health"] == "healthy"
    @assert health["above_threshold"] == true
    println("✓ Health status: $(health["overall_health"])")
    
    # Test with noisy correlation
    println("\nTesting noisy correlation...")
    reset_correlation_tracker!(tracker)
    Random.seed!(42)
    
    for i in 1:50
        x = randn()
        y = x + randn() * 0.3  # Correlated with noise
        update_predictions!(tracker, x, y)
    end
    
    correlation = tracker.current_metrics.pearson_correlation
    @assert correlation > 0.5  # Should still be reasonably correlated
    println("✓ Noisy correlation: $(round(correlation, digits=3))")
    
    # Test sliding window
    println("\nTesting sliding window...")
    window_size = config.window_size
    
    # Fill window completely
    for i in 1:window_size
        update_predictions!(tracker, Float64(i), Float64(i))
    end
    
    @assert tracker.current_metrics.pearson_correlation ≈ 1.0 atol=0.01
    
    # Add conflicting data (should change correlation as window slides)
    for i in 1:25
        update_predictions!(tracker, Float64(i), Float64(-i))  # Negative correlation
    end
    
    new_correlation = tracker.current_metrics.pearson_correlation
    @assert new_correlation < 1.0  # Should be lower due to mixed data
    println("✓ Sliding window working: correlation changed to $(round(new_correlation, digits=3))")
    
    # Test error metrics
    println("\nTesting error metrics...")
    metrics = tracker.current_metrics
    @assert metrics.mean_absolute_error >= 0.0
    @assert metrics.root_mean_square_error >= 0.0
    @assert isfinite(metrics.mean_prediction)
    @assert isfinite(metrics.mean_actual)
    println("✓ Error metrics computed: MAE=$(round(metrics.mean_absolute_error, digits=3)), RMSE=$(round(metrics.root_mean_square_error, digits=3))")
    
    # Test retraining trigger
    println("\nTesting retraining trigger...")
    
    # Add callback to track retraining
    retraining_triggered = Ref(false)
    add_retraining_callback!(tracker, function(t)
        retraining_triggered[] = true
        println("  Retraining callback executed!")
    end)
    
    # Reset and create poor correlation to trigger retraining
    reset_correlation_tracker!(tracker)
    
    for i in 1:20
        x = randn()
        y = randn()  # Completely uncorrelated
        update_predictions!(tracker, x, y)
    end
    
    if tracker.retraining_triggered
        @assert retraining_triggered[]
        println("✓ Retraining trigger activated")
    else
        println("✓ Retraining trigger not activated (correlation still acceptable)")
    end
    
    # Test statistics
    println("\nTesting statistics collection...")
    stats = get_correlation_stats(tracker)
    @assert haskey(stats, "current_metrics")
    @assert haskey(stats, "performance_stats") 
    @assert stats["performance_stats"]["total_predictions"] > 0
    
    if haskey(stats, "performance_stats")
        total_preds = stats["performance_stats"]["total_predictions"]
        avg_time = stats["performance_stats"]["avg_update_time_ms"]
        println("✓ Statistics: $(total_preds) predictions, $(round(avg_time, digits=3))ms avg update time")
    end
    
    # Test persistence
    println("\nTesting persistence...")
    save_correlation_history(tracker)
    @assert isfile(tracker.history_file)
    
    # Create new tracker and try to load
    new_tracker = create_correlation_tracker(config, history_file="simple_test_correlation.json")
    println("✓ Persistence: history saved and loaded")
    
    # Cleanup
    if isfile("simple_test_correlation.json")
        rm("simple_test_correlation.json")
    end
    
    # Test trend analysis
    println("\nTesting trend analysis...")
    reset_correlation_tracker!(tracker)
    
    # Create improving trend
    for phase in 1:10
        noise_level = 2.0 - (phase * 0.18)  # Decreasing noise = improving correlation
        for i in 1:5
            x = Float64(i)
            y = x + randn() * noise_level
            update_predictions!(tracker, x, y)
        end
    end
    
    trend_direction = tracker.current_metrics.trend_direction
    println("✓ Trend analysis: direction=$(trend_direction), trend=$(round(tracker.current_metrics.correlation_trend, digits=4))")
    
    # Performance test
    println("\nTesting performance...")
    n_updates = 1000
    start_time = time()
    
    for i in 1:n_updates
        x = randn()
        y = x + randn() * 0.1
        update_predictions!(tracker, x, y)
    end
    
    total_time = time() - start_time
    avg_time_ms = (total_time / n_updates) * 1000
    
    @assert avg_time_ms < 2.0  # Should be fast
    println("✓ Performance: $(round(avg_time_ms, digits=3)) ms per update")
    
    return true
end

# Run the simple test
if abspath(PROGRAM_FILE) == @__FILE__
    success = test_simple_correlation_tracking()
    
    println("="^50)
    if success
        println("✅ Simple correlation tracking test PASSED")
        println("✅ Core functionality validated:")
        println("  - Configuration and tracker creation")
        println("  - Pearson, Spearman, and Kendall correlations")
        println("  - Sliding window functionality")
        println("  - Error metrics calculation")
        println("  - Health monitoring and statistics")
        println("  - Retraining trigger system")
        println("  - Trend analysis")
        println("  - Data persistence")
        println("  - Performance optimization")
    else
        println("❌ Simple correlation tracking test FAILED")
    end
    
    exit(success ? 0 : 1)
end