module CorrelationTracking

using CUDA
using Statistics
using Random
using Dates
using Printf
using JSON3

export CorrelationTracker, CorrelationMetrics, CorrelationConfig
export create_correlation_tracker, update_predictions!, get_current_correlation
export check_correlation_health, trigger_retraining_if_needed, get_correlation_stats
export save_correlation_history, load_correlation_history, reset_correlation_tracker!
export add_retraining_callback!, reset_retraining_trigger!

"""
Configuration for correlation tracking system
"""
struct CorrelationConfig
    window_size::Int                    # Size of sliding window for correlation calculation
    min_samples::Int                    # Minimum samples needed before calculating correlation
    correlation_threshold::Float64      # Threshold below which retraining is triggered (0.9)
    anomaly_threshold::Float64          # Z-score threshold for anomaly detection
    update_frequency::Int               # How often to update correlations (in predictions)
    save_frequency::Int                 # How often to save correlation history
    enable_gpu_computation::Bool        # Use GPU for correlation calculations
    enable_anomaly_detection::Bool      # Enable anomaly detection
    correlation_types::Vector{Symbol}   # Types of correlation to compute [:pearson, :spearman, :kendall]
end

"""
Default correlation configuration
"""
function CorrelationConfig(;
    window_size::Int = 1000,
    min_samples::Int = 50,
    correlation_threshold::Float64 = 0.9,
    anomaly_threshold::Float64 = 2.0,
    update_frequency::Int = 10,
    save_frequency::Int = 100,
    enable_gpu_computation::Bool = true,
    enable_anomaly_detection::Bool = true,
    correlation_types::Vector{Symbol} = [:pearson, :spearman, :kendall]
)
    CorrelationConfig(
        window_size, min_samples, correlation_threshold, anomaly_threshold,
        update_frequency, save_frequency, enable_gpu_computation,
        enable_anomaly_detection, correlation_types
    )
end

"""
Correlation metrics for different correlation types
"""
mutable struct CorrelationMetrics
    pearson_correlation::Float64
    spearman_correlation::Float64
    kendall_correlation::Float64
    
    # Statistical measures
    mean_absolute_error::Float64
    root_mean_square_error::Float64
    mean_prediction::Float64
    mean_actual::Float64
    
    # Confidence measures
    pearson_pvalue::Float64
    correlation_confidence::Float64
    
    # Trend analysis
    correlation_trend::Float64      # Recent trend in correlation
    trend_direction::Int8           # -1 (decreasing), 0 (stable), 1 (increasing)
    
    # Metadata
    timestamp::DateTime
    sample_count::Int64
    window_position::Int64
end

"""
Default correlation metrics
"""
function CorrelationMetrics()
    CorrelationMetrics(
        0.0, 0.0, 0.0,          # correlations
        0.0, 0.0, 0.0, 0.0,     # errors and means
        1.0, 0.0,               # p-value and confidence
        0.0, 0,                 # trend
        now(), 0, 0             # metadata
    )
end

"""
Main correlation tracker with sliding window and GPU acceleration
"""
mutable struct CorrelationTracker
    config::CorrelationConfig
    
    # Sliding window buffers (GPU arrays if GPU enabled)
    predicted_scores::Union{CuArray{Float32}, Array{Float32}}
    actual_scores::Union{CuArray{Float32}, Array{Float32}}
    timestamps::Vector{DateTime}
    
    # Current position in circular buffer
    buffer_position::Int64
    total_predictions::Int64
    
    # Current metrics
    current_metrics::CorrelationMetrics
    metrics_history::Vector{CorrelationMetrics}
    
    # Anomaly detection
    correlation_history_buffer::Vector{Float64}
    anomaly_count::Int64
    consecutive_low_correlation::Int64
    
    # Retraining triggers
    retraining_triggered::Bool
    last_retraining_time::DateTime
    retraining_callbacks::Vector{Function}
    
    # Performance tracking
    update_count::Int64
    total_update_time::Float64
    gpu_stream::Union{CuStream, Nothing}
    
    # Persistence
    history_file::String
    last_save_time::DateTime
end

"""
Create correlation tracker with specified configuration
"""
function create_correlation_tracker(config::CorrelationConfig; history_file::String = "correlation_history.json")
    # Initialize buffers based on GPU availability and configuration
    if config.enable_gpu_computation && CUDA.functional()
        predicted_scores = CUDA.zeros(Float32, config.window_size)
        actual_scores = CUDA.zeros(Float32, config.window_size)
        gpu_stream = CuStream()
        @info "Correlation tracker initialized with GPU acceleration"
    else
        predicted_scores = zeros(Float32, config.window_size)
        actual_scores = zeros(Float32, config.window_size)
        gpu_stream = nothing
        @info "Correlation tracker initialized with CPU computation"
    end
    
    timestamps = Vector{DateTime}(undef, config.window_size)
    fill!(timestamps, now())
    
    tracker = CorrelationTracker(
        config,
        predicted_scores,
        actual_scores,
        timestamps,
        0,                                          # buffer_position
        0,                                          # total_predictions
        CorrelationMetrics(),                       # current_metrics
        Vector{CorrelationMetrics}(),              # metrics_history
        Vector{Float64}(),                         # correlation_history_buffer
        0, 0,                                      # anomaly_count, consecutive_low_correlation
        false, DateTime(0),                        # retraining_triggered, last_retraining_time
        Vector{Function}(),                        # retraining_callbacks
        0, 0.0,                                    # update_count, total_update_time
        gpu_stream,                                # gpu_stream
        history_file,                              # history_file
        now()                                      # last_save_time
    )
    
    # Try to load existing history
    try
        load_correlation_history!(tracker)
        @info "Loaded existing correlation history from $(history_file)"
    catch e
        @info "Starting with fresh correlation history: $e"
    end
    
    return tracker
end

"""
Update tracker with new prediction and actual score
"""
function update_predictions!(tracker::CorrelationTracker, predicted::Real, actual::Real)
    start_time = time()
    
    # Update circular buffer position
    tracker.buffer_position = (tracker.buffer_position % tracker.config.window_size) + 1
    tracker.total_predictions += 1
    
    # Store new values
    if tracker.predicted_scores isa CuArray
        # Use copyto! for GPU arrays to avoid scalar indexing
        CUDA.@allowscalar begin
            tracker.predicted_scores[tracker.buffer_position] = Float32(predicted)
            tracker.actual_scores[tracker.buffer_position] = Float32(actual)
        end
    else
        tracker.predicted_scores[tracker.buffer_position] = Float32(predicted)
        tracker.actual_scores[tracker.buffer_position] = Float32(actual)
    end
    tracker.timestamps[tracker.buffer_position] = now()
    
    # Update correlations if we have enough samples and it's time to update
    if (tracker.total_predictions >= tracker.config.min_samples && 
        tracker.total_predictions % tracker.config.update_frequency == 0)
        
        update_correlations!(tracker)
        
        # Check for anomalies and retraining triggers
        if tracker.config.enable_anomaly_detection
            check_anomalies!(tracker)
        end
        
        check_retraining_trigger!(tracker)
        
        # Save history periodically
        if tracker.total_predictions % tracker.config.save_frequency == 0
            save_correlation_history(tracker)
        end
    end
    
    # Update performance metrics
    tracker.update_count += 1
    tracker.total_update_time += time() - start_time
    
    return tracker.current_metrics
end

"""
Update correlation calculations using current window data
"""
function update_correlations!(tracker::CorrelationTracker)
    config = tracker.config
    
    # Determine how many samples we actually have
    n_samples = min(tracker.total_predictions, config.window_size)
    
    if n_samples < config.min_samples
        return
    end
    
    # Extract data from circular buffer
    if tracker.total_predictions <= config.window_size
        # Buffer not full yet, use from beginning
        predicted_data = tracker.predicted_scores[1:n_samples]
        actual_data = tracker.actual_scores[1:n_samples]
    else
        # Buffer is full, extract in correct order
        if tracker.config.enable_gpu_computation
            predicted_data = get_circular_buffer_data(tracker.predicted_scores, tracker.buffer_position, n_samples)
            actual_data = get_circular_buffer_data(tracker.actual_scores, tracker.buffer_position, n_samples)
        else
            predicted_data = get_circular_buffer_data_cpu(tracker.predicted_scores, tracker.buffer_position, n_samples)
            actual_data = get_circular_buffer_data_cpu(tracker.actual_scores, tracker.buffer_position, n_samples)
        end
    end
    
    # Convert to CPU arrays for correlation calculation if needed
    if tracker.config.enable_gpu_computation && predicted_data isa CuArray
        predicted_cpu = Array(predicted_data)
        actual_cpu = Array(actual_data)
    else
        predicted_cpu = predicted_data
        actual_cpu = actual_data
    end
    
    # Calculate correlations
    metrics = CorrelationMetrics()
    
    # Pearson correlation
    if :pearson in config.correlation_types
        try
            metrics.pearson_correlation = cor(predicted_cpu, actual_cpu)
            if isnan(metrics.pearson_correlation)
                metrics.pearson_correlation = 0.0
            end
        catch
            metrics.pearson_correlation = 0.0
        end
    end
    
    # Spearman rank correlation
    if :spearman in config.correlation_types
        try
            metrics.spearman_correlation = spearman_correlation(predicted_cpu, actual_cpu)
        catch
            metrics.spearman_correlation = 0.0
        end
    end
    
    # Kendall's tau
    if :kendall in config.correlation_types
        try
            metrics.kendall_correlation = kendall_tau(predicted_cpu, actual_cpu)
        catch
            metrics.kendall_correlation = 0.0
        end
    end
    
    # Error metrics
    residuals = predicted_cpu - actual_cpu
    metrics.mean_absolute_error = mean(abs.(residuals))
    metrics.root_mean_square_error = sqrt(mean(residuals.^2))
    metrics.mean_prediction = mean(predicted_cpu)
    metrics.mean_actual = mean(actual_cpu)
    
    # Confidence and trend analysis
    metrics.correlation_confidence = calculate_correlation_confidence(metrics.pearson_correlation, n_samples)
    update_correlation_trend!(tracker, metrics.pearson_correlation)
    
    # Update metadata
    metrics.timestamp = now()
    metrics.sample_count = n_samples
    metrics.window_position = tracker.buffer_position
    
    # Store current metrics
    tracker.current_metrics = metrics
    push!(tracker.metrics_history, metrics)
    
    # Maintain correlation history buffer for anomaly detection
    push!(tracker.correlation_history_buffer, metrics.pearson_correlation)
    if length(tracker.correlation_history_buffer) > config.window_size ÷ 10
        popfirst!(tracker.correlation_history_buffer)
    end
    
    return metrics
end

"""
Extract data from circular buffer in correct chronological order (GPU version)
"""
function get_circular_buffer_data(buffer::CuArray, position::Int, n_samples::Int)
    if n_samples <= length(buffer) - position
        # Data doesn't wrap around
        return buffer[position+1:position+n_samples]
    else
        # Data wraps around
        first_part = buffer[position+1:end]
        second_part = buffer[1:n_samples-(length(buffer)-position)]
        return vcat(first_part, second_part)
    end
end

"""
Extract data from circular buffer in correct chronological order (CPU version)
"""
function get_circular_buffer_data_cpu(buffer::Array, position::Int, n_samples::Int)
    if n_samples <= length(buffer) - position
        # Data doesn't wrap around
        return buffer[position+1:position+n_samples]
    else
        # Data wraps around
        first_part = buffer[position+1:end]
        second_part = buffer[1:n_samples-(length(buffer)-position)]
        return vcat(first_part, second_part)
    end
end

"""
Calculate Spearman rank correlation
"""
function spearman_correlation(x::AbstractVector, y::AbstractVector)
    n = length(x)
    if n != length(y) || n < 2
        return 0.0
    end
    
    # Rank the data
    rank_x = compute_ranks(x)
    rank_y = compute_ranks(y)
    
    # Calculate Pearson correlation of ranks
    return cor(rank_x, rank_y)
end

"""
Calculate Kendall's tau correlation
"""
function kendall_tau(x::AbstractVector, y::AbstractVector)
    n = length(x)
    if n != length(y) || n < 2
        return 0.0
    end
    
    concordant = 0
    discordant = 0
    
    for i in 1:n-1
        for j in i+1:n
            sign_x = sign(x[j] - x[i])
            sign_y = sign(y[j] - y[i])
            
            if sign_x == sign_y && sign_x != 0
                concordant += 1
            elseif sign_x != sign_y && sign_x != 0 && sign_y != 0
                discordant += 1
            end
        end
    end
    
    total_pairs = n * (n - 1) ÷ 2
    if total_pairs == 0
        return 0.0
    end
    
    return (concordant - discordant) / total_pairs
end

"""
Compute ranks for Spearman correlation
"""
function compute_ranks(x::AbstractVector)
    n = length(x)
    sorted_indices = sortperm(x)
    ranks = similar(x, Float64)
    
    i = 1
    while i <= n
        j = i
        # Find tied values
        while j < n && x[sorted_indices[j]] == x[sorted_indices[j+1]]
            j += 1
        end
        
        # Assign average rank to tied values
        avg_rank = (i + j) / 2
        for k in i:j
            ranks[sorted_indices[k]] = avg_rank
        end
        
        i = j + 1
    end
    
    return ranks
end

"""
Calculate correlation confidence based on sample size
"""
function calculate_correlation_confidence(correlation::Float64, n::Int)
    if n < 3
        return 0.0
    end
    
    # Fisher's z-transformation for confidence calculation
    z = 0.5 * log((1 + abs(correlation)) / (1 - abs(correlation)))
    se = 1 / sqrt(n - 3)
    
    # 95% confidence interval
    z_critical = 1.96
    margin = z_critical * se
    
    # Convert back to correlation space
    lower_z = z - margin
    upper_z = z + margin
    
    lower_r = (exp(2 * lower_z) - 1) / (exp(2 * lower_z) + 1)
    upper_r = (exp(2 * upper_z) - 1) / (exp(2 * upper_z) + 1)
    
    # Return confidence as width of interval (smaller is better)
    return abs(upper_r - lower_r)
end

"""
Update correlation trend analysis
"""
function update_correlation_trend!(tracker::CorrelationTracker, current_correlation::Float64)
    history = tracker.correlation_history_buffer
    
    if length(history) < 5
        tracker.current_metrics.correlation_trend = 0.0
        tracker.current_metrics.trend_direction = 0
        return
    end
    
    # Calculate trend over last few correlation values
    recent_correlations = history[max(1, end-4):end]
    push!(recent_correlations, current_correlation)
    
    # Simple linear trend
    n = length(recent_correlations)
    x = collect(1:n)
    y = recent_correlations
    
    # Calculate slope
    x_mean = mean(x)
    y_mean = mean(y)
    slope = sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean).^2)
    
    tracker.current_metrics.correlation_trend = slope
    
    # Determine trend direction
    if abs(slope) < 0.001
        tracker.current_metrics.trend_direction = 0  # Stable
    elseif slope > 0
        tracker.current_metrics.trend_direction = 1  # Increasing
    else
        tracker.current_metrics.trend_direction = -1 # Decreasing
    end
end

"""
Check for correlation anomalies and degradation
"""
function check_anomalies!(tracker::CorrelationTracker)
    config = tracker.config
    current_corr = tracker.current_metrics.pearson_correlation
    
    # Check against threshold
    if current_corr < config.correlation_threshold
        tracker.consecutive_low_correlation += 1
    else
        tracker.consecutive_low_correlation = 0
    end
    
    # Anomaly detection using Z-score
    if length(tracker.correlation_history_buffer) >= 10
        history = tracker.correlation_history_buffer
        mean_corr = mean(history)
        std_corr = std(history)
        
        if std_corr > 0
            z_score = abs(current_corr - mean_corr) / std_corr
            if z_score > config.anomaly_threshold
                tracker.anomaly_count += 1
                @warn "Correlation anomaly detected: current=$(round(current_corr, digits=3)), z-score=$(round(z_score, digits=2))"
            end
        end
    end
end

"""
Check if retraining should be triggered
"""
function check_retraining_trigger!(tracker::CorrelationTracker)
    config = tracker.config
    current_corr = tracker.current_metrics.pearson_correlation
    
    # Trigger retraining if correlation is consistently below threshold
    should_retrain = (
        current_corr < config.correlation_threshold &&
        tracker.consecutive_low_correlation >= 3 &&
        !tracker.retraining_triggered
    )
    
    if should_retrain
        tracker.retraining_triggered = true
        tracker.last_retraining_time = now()
        
        @warn "Retraining triggered: correlation $(round(current_corr, digits=3)) < $(config.correlation_threshold) for $(tracker.consecutive_low_correlation) consecutive updates"
        
        # Execute retraining callbacks
        for callback in tracker.retraining_callbacks
            try
                callback(tracker)
            catch e
                @error "Retraining callback failed: $e"
            end
        end
    end
end

"""
Add retraining callback function
"""
function add_retraining_callback!(tracker::CorrelationTracker, callback::Function)
    push!(tracker.retraining_callbacks, callback)
end

"""
Reset retraining trigger (call after successful retraining)
"""
function reset_retraining_trigger!(tracker::CorrelationTracker)
    tracker.retraining_triggered = false
    tracker.consecutive_low_correlation = 0
    @info "Retraining trigger reset"
end

"""
Get current correlation health status
"""
function check_correlation_health(tracker::CorrelationTracker)
    metrics = tracker.current_metrics
    config = tracker.config
    
    health_status = Dict{String, Any}(
        "overall_health" => "unknown",
        "pearson_correlation" => metrics.pearson_correlation,
        "spearman_correlation" => metrics.spearman_correlation,
        "kendall_correlation" => metrics.kendall_correlation,
        "above_threshold" => metrics.pearson_correlation >= config.correlation_threshold,
        "trend_direction" => metrics.trend_direction,
        "correlation_trend" => metrics.correlation_trend,
        "consecutive_low_correlation" => tracker.consecutive_low_correlation,
        "anomaly_count" => tracker.anomaly_count,
        "retraining_needed" => tracker.retraining_triggered,
        "sample_count" => metrics.sample_count,
        "last_update" => metrics.timestamp
    )
    
    # Determine overall health
    if metrics.sample_count < config.min_samples
        health_status["overall_health"] = "insufficient_data"
    elseif tracker.retraining_triggered
        health_status["overall_health"] = "needs_retraining"
    elseif metrics.pearson_correlation >= config.correlation_threshold && tracker.consecutive_low_correlation == 0
        health_status["overall_health"] = "healthy"
    elseif metrics.pearson_correlation >= config.correlation_threshold * 0.95
        health_status["overall_health"] = "warning"
    else
        health_status["overall_health"] = "poor"
    end
    
    return health_status
end

"""
Get comprehensive correlation statistics
"""
function get_correlation_stats(tracker::CorrelationTracker)
    metrics = tracker.current_metrics
    
    # Calculate statistics over metrics history
    if !isempty(tracker.metrics_history)
        pearson_history = [m.pearson_correlation for m in tracker.metrics_history]
        spearman_history = [m.spearman_correlation for m in tracker.metrics_history]
        kendall_history = [m.kendall_correlation for m in tracker.metrics_history]
        
        stats = Dict{String, Any}(
            "current_metrics" => Dict(
                "pearson" => metrics.pearson_correlation,
                "spearman" => metrics.spearman_correlation,
                "kendall" => metrics.kendall_correlation,
                "mae" => metrics.mean_absolute_error,
                "rmse" => metrics.root_mean_square_error,
                "confidence" => metrics.correlation_confidence,
                "trend" => metrics.correlation_trend,
                "trend_direction" => metrics.trend_direction
            ),
            "historical_stats" => Dict(
                "pearson_mean" => mean(pearson_history),
                "pearson_std" => std(pearson_history),
                "pearson_min" => minimum(pearson_history),
                "pearson_max" => maximum(pearson_history),
                "spearman_mean" => mean(spearman_history),
                "kendall_mean" => mean(kendall_history)
            ),
            "performance_stats" => Dict(
                "total_predictions" => tracker.total_predictions,
                "update_count" => tracker.update_count,
                "avg_update_time_ms" => (tracker.total_update_time / tracker.update_count) * 1000,
                "anomaly_count" => tracker.anomaly_count,
                "consecutive_low_correlation" => tracker.consecutive_low_correlation
            ),
            "system_info" => Dict(
                "window_size" => tracker.config.window_size,
                "gpu_enabled" => tracker.config.enable_gpu_computation && !isnothing(tracker.gpu_stream),
                "correlation_types" => tracker.config.correlation_types,
                "last_save" => tracker.last_save_time
            )
        )
    else
        stats = Dict{String, Any}(
            "current_metrics" => Dict(
                "pearson" => metrics.pearson_correlation,
                "spearman" => metrics.spearman_correlation,
                "kendall" => metrics.kendall_correlation
            ),
            "message" => "No historical data available yet"
        )
    end
    
    return stats
end

"""
Save correlation history to file
"""
function save_correlation_history(tracker::CorrelationTracker)
    try
        history_data = Dict{String, Any}(
            "config" => Dict(
                "window_size" => tracker.config.window_size,
                "correlation_threshold" => tracker.config.correlation_threshold,
                "correlation_types" => string.(tracker.config.correlation_types)
            ),
            "metrics_history" => [
                Dict(
                    "timestamp" => string(m.timestamp),
                    "pearson_correlation" => m.pearson_correlation,
                    "spearman_correlation" => m.spearman_correlation,
                    "kendall_correlation" => m.kendall_correlation,
                    "mae" => m.mean_absolute_error,
                    "rmse" => m.root_mean_square_error,
                    "sample_count" => m.sample_count
                ) for m in tracker.metrics_history
            ],
            "summary_stats" => Dict(
                "total_predictions" => tracker.total_predictions,
                "anomaly_count" => tracker.anomaly_count,
                "retraining_triggered" => tracker.retraining_triggered,
                "last_retraining_time" => string(tracker.last_retraining_time)
            )
        )
        
        open(tracker.history_file, "w") do io
            JSON3.write(io, history_data)
        end
        
        tracker.last_save_time = now()
        @info "Saved correlation history to $(tracker.history_file)"
        
    catch e
        @error "Failed to save correlation history: $e"
    end
end

"""
Load correlation history from file
"""
function load_correlation_history!(tracker::CorrelationTracker)
    if !isfile(tracker.history_file)
        return false
    end
    
    try
        history_data = JSON3.read(read(tracker.history_file, String))
        
        # Restore summary stats
        if haskey(history_data, "summary_stats")
            stats = history_data["summary_stats"]
            tracker.total_predictions = get(stats, "total_predictions", 0)
            tracker.anomaly_count = get(stats, "anomaly_count", 0)
            tracker.retraining_triggered = get(stats, "retraining_triggered", false)
            
            if haskey(stats, "last_retraining_time")
                try
                    tracker.last_retraining_time = DateTime(stats["last_retraining_time"])
                catch
                    tracker.last_retraining_time = DateTime(0)
                end
            end
        end
        
        # Restore metrics history (keep only recent entries)
        if haskey(history_data, "metrics_history")
            recent_history = history_data["metrics_history"]
            if length(recent_history) > 100
                recent_history = recent_history[end-99:end]  # Keep last 100 entries
            end
            
            for entry in recent_history
                metrics = CorrelationMetrics()
                metrics.timestamp = DateTime(entry["timestamp"])
                metrics.pearson_correlation = entry["pearson_correlation"]
                metrics.spearman_correlation = get(entry, "spearman_correlation", 0.0)
                metrics.kendall_correlation = get(entry, "kendall_correlation", 0.0)
                metrics.mean_absolute_error = get(entry, "mae", 0.0)
                metrics.root_mean_square_error = get(entry, "rmse", 0.0)
                metrics.sample_count = get(entry, "sample_count", 0)
                
                push!(tracker.metrics_history, metrics)
                push!(tracker.correlation_history_buffer, metrics.pearson_correlation)
            end
        end
        
        @info "Loaded correlation history: $(length(tracker.metrics_history)) entries"
        return true
        
    catch e
        @error "Failed to load correlation history: $e"
        return false
    end
end

"""
Reset correlation tracker to initial state
"""
function reset_correlation_tracker!(tracker::CorrelationTracker)
    # Reset buffers
    fill!(tracker.predicted_scores, 0.0f0)
    fill!(tracker.actual_scores, 0.0f0)
    fill!(tracker.timestamps, now())
    
    # Reset counters
    tracker.buffer_position = 0
    tracker.total_predictions = 0
    tracker.update_count = 0
    tracker.total_update_time = 0.0
    
    # Reset metrics
    tracker.current_metrics = CorrelationMetrics()
    empty!(tracker.metrics_history)
    empty!(tracker.correlation_history_buffer)
    
    # Reset anomaly detection
    tracker.anomaly_count = 0
    tracker.consecutive_low_correlation = 0
    
    # Reset retraining
    tracker.retraining_triggered = false
    tracker.last_retraining_time = DateTime(0)
    
    @info "Correlation tracker reset to initial state"
end

"""
Get current correlation value (primary metric)
"""
function get_current_correlation(tracker::CorrelationTracker)
    return tracker.current_metrics.pearson_correlation
end

"""
Check if retraining is needed
"""
function trigger_retraining_if_needed(tracker::CorrelationTracker)
    return tracker.retraining_triggered
end

end # module CorrelationTracking