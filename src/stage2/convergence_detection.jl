"""
Convergence Detection Algorithm for MCTS Ensemble Feature Selection
Implements convergence detection determining when ensemble has reached stable feature selection,
including convergence metrics tracking, sliding window analysis, early stopping criteria,
and adaptive convergence thresholds for robust ensemble termination.

This module provides intelligent convergence detection across ensemble trees, consensus stability
monitoring, and dynamic threshold adaptation based on problem difficulty and selection patterns.
"""

module ConvergenceDetection

using Random
using Statistics
using Dates
using Printf
using LinearAlgebra

# Import consensus voting for feature tracking
include("consensus_voting.jl")
using .ConsensusVoting

# Import diversity mechanisms for ensemble coordination
include("diversity_mechanisms.jl")
using .DiversityMechanisms

"""
Convergence detection strategy types
"""
@enum ConvergenceStrategy begin
    STABILITY_BASED = 1      # Based on feature selection stability
    CONSENSUS_BASED = 2      # Based on consensus strength
    ENTROPY_BASED = 3        # Based on selection entropy reduction
    HYBRID_CONVERGENCE = 4   # Combines multiple strategies
    ADAPTIVE_CONVERGENCE = 5 # Dynamic strategy selection
end

"""
Early stopping criteria types
"""
@enum StoppingCriteria begin
    FIXED_ITERATIONS = 1     # Fixed number of stable iterations
    STATISTICAL_TEST = 2     # Statistical significance testing
    PLATEAU_DETECTION = 3    # Performance plateau detection
    ENSEMBLE_AGREEMENT = 4   # High ensemble agreement
    ADAPTIVE_STOPPING = 5    # Dynamic criteria adjustment
end

"""
Convergence threshold adaptation types
"""
@enum ThresholdAdaptation begin
    STATIC_THRESHOLD = 1     # Fixed convergence threshold
    LINEAR_ADAPTATION = 2    # Linear threshold adjustment
    EXPONENTIAL_DECAY = 3    # Exponential threshold reduction
    PROBLEM_ADAPTIVE = 4     # Problem-specific adaptation
    PERFORMANCE_BASED = 5    # Based on ensemble performance
end

"""
Convergence metrics for tracking ensemble stability
"""
mutable struct ConvergenceMetrics
    feature_stability::Float64           # Stability of feature selections
    consensus_strength::Float64         # Strength of ensemble consensus
    selection_entropy::Float64          # Entropy of feature selections
    iteration_variance::Float64         # Variance across iterations
    agreement_coefficient::Float64      # Inter-tree agreement measure
    trend_slope::Float64               # Trend of convergence over time
    confidence_interval::Tuple{Float64, Float64}  # Confidence bounds
    last_update::DateTime              # Last metrics update time
    update_count::Int                  # Number of metrics updates
    is_converged::Bool                 # Whether convergence detected
end

"""
Create convergence metrics tracker
"""
function create_convergence_metrics()
    return ConvergenceMetrics(
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        (0.0, 1.0), now(), 0, false
    )
end

"""
Sliding window for convergence analysis
"""
mutable struct ConvergenceWindow
    window_size::Int                    # Size of sliding window
    stability_history::Vector{Float64}  # Historical stability values
    consensus_history::Vector{Float64}  # Historical consensus values
    entropy_history::Vector{Float64}    # Historical entropy values
    timestamps::Vector{DateTime}        # Timestamps for each measurement
    window_variance::Float64            # Variance within current window
    window_trend::Float64              # Trend within current window
    is_stable::Bool                    # Whether window shows stability
    consecutive_stable::Int            # Consecutive stable windows
end

"""
Create convergence sliding window
"""
function create_convergence_window(window_size::Int = 50)
    return ConvergenceWindow(
        window_size,
        Float64[], Float64[], Float64[], DateTime[],
        0.0, 0.0, false, 0
    )
end

"""
Convergence detection configuration
"""
struct ConvergenceConfig
    # Strategy configuration
    convergence_strategy::ConvergenceStrategy
    stopping_criteria::StoppingCriteria
    threshold_adaptation::ThresholdAdaptation
    
    # Convergence thresholds
    stability_threshold::Float32
    consensus_threshold::Float32
    entropy_threshold::Float32
    agreement_threshold::Float32
    
    # Window and timing parameters
    sliding_window_size::Int
    minimum_iterations::Int
    maximum_iterations::Int
    convergence_patience::Int
    
    # Early stopping configuration
    early_stopping_enabled::Bool
    minimum_stable_windows::Int
    stability_tolerance::Float32
    convergence_confidence::Float32
    
    # Adaptive threshold parameters
    initial_threshold::Float32
    threshold_decay_rate::Float32
    minimum_threshold::Float32
    adaptation_interval::Int
    
    # Statistical testing parameters
    significance_level::Float32
    statistical_test_window::Int
    trend_detection_sensitivity::Float32
    variance_stability_factor::Float32
    
    # Visualization and monitoring
    enable_convergence_tracking::Bool
    enable_visualization::Bool
    visualization_update_interval::Int
    enable_detailed_logging::Bool
    
    # Performance optimization
    enable_parallel_analysis::Bool
    batch_analysis_size::Int
    cache_convergence_calculations::Bool
    enable_incremental_updates::Bool
end

"""
Create convergence detection configuration
"""
function create_convergence_config(;
    convergence_strategy::ConvergenceStrategy = HYBRID_CONVERGENCE,
    stopping_criteria::StoppingCriteria = ADAPTIVE_STOPPING,
    threshold_adaptation::ThresholdAdaptation = PROBLEM_ADAPTIVE,
    stability_threshold::Float32 = 0.95f0,
    consensus_threshold::Float32 = 0.9f0,
    entropy_threshold::Float32 = 0.1f0,
    agreement_threshold::Float32 = 0.85f0,
    sliding_window_size::Int = 100,
    minimum_iterations::Int = 1000,
    maximum_iterations::Int = 50000,
    convergence_patience::Int = 10,
    early_stopping_enabled::Bool = true,
    minimum_stable_windows::Int = 5,
    stability_tolerance::Float32 = 0.02f0,
    convergence_confidence::Float32 = 0.95f0,
    initial_threshold::Float32 = 0.95f0,
    threshold_decay_rate::Float32 = 0.99f0,
    minimum_threshold::Float32 = 0.8f0,
    adaptation_interval::Int = 500,
    significance_level::Float32 = 0.05f0,
    statistical_test_window::Int = 200,
    trend_detection_sensitivity::Float32 = 0.01f0,
    variance_stability_factor::Float32 = 0.1f0,
    enable_convergence_tracking::Bool = true,
    enable_visualization::Bool = false,
    visualization_update_interval::Int = 100,
    enable_detailed_logging::Bool = true,
    enable_parallel_analysis::Bool = true,
    batch_analysis_size::Int = 50,
    cache_convergence_calculations::Bool = true,
    enable_incremental_updates::Bool = true
)
    return ConvergenceConfig(
        convergence_strategy, stopping_criteria, threshold_adaptation,
        stability_threshold, consensus_threshold, entropy_threshold, agreement_threshold,
        sliding_window_size, minimum_iterations, maximum_iterations, convergence_patience,
        early_stopping_enabled, minimum_stable_windows, stability_tolerance, convergence_confidence,
        initial_threshold, threshold_decay_rate, minimum_threshold, adaptation_interval,
        significance_level, statistical_test_window, trend_detection_sensitivity, variance_stability_factor,
        enable_convergence_tracking, enable_visualization, visualization_update_interval, enable_detailed_logging,
        enable_parallel_analysis, batch_analysis_size, cache_convergence_calculations, enable_incremental_updates
    )
end

"""
Convergence detection statistics
"""
mutable struct ConvergenceStats
    total_iterations::Int
    convergence_iterations::Int
    false_convergences::Int
    early_stopping_triggers::Int
    threshold_adaptations::Int
    average_convergence_time::Float64
    stability_score::Float64
    consensus_evolution::Vector{Float64}
    convergence_history::Vector{DateTime}
    last_convergence_check::DateTime
end

"""
Initialize convergence statistics
"""
function initialize_convergence_stats()
    return ConvergenceStats(
        0, 0, 0, 0, 0, 0.0, 0.0,
        Float64[], DateTime[], now()
    )
end

"""
Convergence detector managing ensemble convergence analysis
"""
mutable struct ConvergenceDetector
    config::ConvergenceConfig
    metrics::ConvergenceMetrics
    sliding_window::ConvergenceWindow
    stats::ConvergenceStats
    
    # Threshold management
    current_thresholds::Dict{String, Float32}
    adaptive_thresholds::Vector{Float32}
    threshold_history::Vector{Tuple{DateTime, Float32}}
    
    # Convergence state
    is_converged::Bool
    convergence_confidence::Float64
    convergence_timestamp::Union{DateTime, Nothing}
    consecutive_stable_iterations::Int
    
    # Analysis cache
    feature_stability_cache::Dict{Vector{Int}, Float64}
    consensus_cache::Dict{Vector{Int}, Float64}
    entropy_cache::Dict{Vector{Int}, Float64}
    
    # Synchronization
    detector_lock::ReentrantLock
    analysis_times::Vector{Float64}
    
    # Status and logging
    detector_state::String
    convergence_log::Vector{String}
    last_analysis_time::DateTime
end

"""
Initialize convergence detector
"""
function initialize_convergence_detector(config::ConvergenceConfig = create_convergence_config())
    detector = ConvergenceDetector(
        config,
        create_convergence_metrics(),
        create_convergence_window(config.sliding_window_size),
        initialize_convergence_stats(),
        Dict{String, Float32}(
            "stability" => config.stability_threshold,
            "consensus" => config.consensus_threshold,
            "entropy" => config.entropy_threshold,
            "agreement" => config.agreement_threshold
        ),
        Float32[config.initial_threshold],
        Tuple{DateTime, Float32}[(now(), config.initial_threshold)],
        false,
        0.0,
        nothing,
        0,
        Dict{Vector{Int}, Float64}(),
        Dict{Vector{Int}, Float64}(),
        Dict{Vector{Int}, Float64}(),
        ReentrantLock(),
        Float64[],
        "active",
        String[],
        now()
    )
    
    @info "Convergence detector initialized with strategy: $(config.convergence_strategy)"
    return detector
end

"""
Calculate feature selection stability across ensemble
"""
function calculate_feature_stability(detector::ConvergenceDetector, 
                                   feature_selections::Vector{Vector{Int}})::Float64
    if length(feature_selections) < 2
        return 0.0
    end
    
    # Check cache first
    if detector.config.cache_convergence_calculations
        cached_stability = get(detector.feature_stability_cache, feature_selections, nothing)
        if !isnothing(cached_stability)
            return cached_stability
        end
    end
    
    start_time = time()
    
    # Calculate pairwise Jaccard similarity between feature sets
    similarities = Float64[]
    
    for i in 1:(length(feature_selections)-1)
        for j in (i+1):length(feature_selections)
            set1 = Set(feature_selections[i])
            set2 = Set(feature_selections[j])
            
            intersection_size = length(intersect(set1, set2))
            union_size = length(union(set1, set2))
            
            jaccard_similarity = union_size > 0 ? intersection_size / union_size : 0.0
            push!(similarities, jaccard_similarity)
        end
    end
    
    # Calculate overall stability as mean similarity
    stability = length(similarities) > 0 ? mean(similarities) : 0.0
    
    # Cache result
    if detector.config.cache_convergence_calculations
        detector.feature_stability_cache[feature_selections] = stability
    end
    
    # Track analysis time
    analysis_time = (time() - start_time) * 1000
    push!(detector.analysis_times, analysis_time)
    if length(detector.analysis_times) > 1000
        deleteat!(detector.analysis_times, 1)
    end
    
    return stability
end

"""
Calculate consensus strength across ensemble
"""
function calculate_consensus_strength(detector::ConvergenceDetector,
                                    consensus_manager::ConsensusManager)::Float64
    status = get_consensus_status(consensus_manager)
    return status["consensus_strength"]
end

"""
Calculate selection entropy measure
"""
function calculate_selection_entropy(detector::ConvergenceDetector,
                                   feature_selections::Vector{Vector{Int}})::Float64
    if isempty(feature_selections)
        return 1.0  # Maximum entropy for empty selections
    end
    
    # Check cache
    if detector.config.cache_convergence_calculations
        cached_entropy = get(detector.entropy_cache, feature_selections, nothing)
        if !isnothing(cached_entropy)
            return cached_entropy
        end
    end
    
    # Count feature frequency across all selections
    feature_counts = Dict{Int, Int}()
    total_selections = 0
    
    for selection in feature_selections
        for feature_id in selection
            feature_counts[feature_id] = get(feature_counts, feature_id, 0) + 1
            total_selections += 1
        end
    end
    
    if total_selections == 0
        return 1.0
    end
    
    # Calculate Shannon entropy
    entropy = 0.0
    for count in values(feature_counts)
        if count > 0
            probability = count / total_selections
            entropy -= probability * log2(probability)
        end
    end
    
    # Normalize entropy by maximum possible entropy
    max_entropy = log2(length(feature_counts))
    normalized_entropy = max_entropy > 0 ? entropy / max_entropy : 0.0
    
    # Cache result
    if detector.config.cache_convergence_calculations
        detector.entropy_cache[feature_selections] = normalized_entropy
    end
    
    return normalized_entropy
end

"""
Calculate inter-tree agreement coefficient
"""
function calculate_agreement_coefficient(detector::ConvergenceDetector,
                                       feature_selections::Vector{Vector{Int}})::Float64
    if length(feature_selections) < 2
        return 1.0
    end
    
    # Calculate agreement as inverse of coefficient of variation in selection sizes
    selection_sizes = [length(selection) for selection in feature_selections]
    
    if isempty(selection_sizes)
        return 1.0
    end
    
    mean_size = mean(selection_sizes)
    std_size = std(selection_sizes)
    
    # Calculate coefficient of variation
    cv = mean_size > 0 ? std_size / mean_size : 0.0
    
    # Convert to agreement coefficient (higher CV = lower agreement)
    agreement = 1.0 / (1.0 + cv)
    
    return agreement
end

"""
Update convergence metrics based on current ensemble state
"""
function update_convergence_metrics!(detector::ConvergenceDetector,
                                   feature_selections::Vector{Vector{Int}},
                                   consensus_manager::ConsensusManager)
    lock(detector.detector_lock) do
        metrics = detector.metrics
        
        # Calculate current metrics
        metrics.feature_stability = calculate_feature_stability(detector, feature_selections)
        metrics.consensus_strength = calculate_consensus_strength(detector, consensus_manager)
        metrics.selection_entropy = calculate_selection_entropy(detector, feature_selections)
        metrics.agreement_coefficient = calculate_agreement_coefficient(detector, feature_selections)
        
        # Calculate iteration variance if we have history
        window = detector.sliding_window
        if length(window.stability_history) > 1
            metrics.iteration_variance = var(window.stability_history)
        end
        
        # Calculate trend slope if we have sufficient history
        if length(window.stability_history) >= 10
            x_values = collect(1:length(window.stability_history))
            y_values = window.stability_history
            
            # Simple linear regression for trend
            n = length(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x_values .* y_values)
            sum_x2 = sum(x_values .^ 2)
            
            denominator = n * sum_x2 - sum_x^2
            if denominator != 0
                metrics.trend_slope = (n * sum_xy - sum_x * sum_y) / denominator
            end
        end
        
        # Update confidence interval (simple approach using stability variance)
        if metrics.iteration_variance > 0
            margin = 1.96 * sqrt(metrics.iteration_variance)  # 95% confidence
            metrics.confidence_interval = (
                max(0.0, metrics.feature_stability - margin),
                min(1.0, metrics.feature_stability + margin)
            )
        else
            metrics.confidence_interval = (metrics.feature_stability, metrics.feature_stability)
        end
        
        metrics.last_update = now()
        metrics.update_count += 1
        
        detector.stats.total_iterations += 1
        detector.last_analysis_time = now()
        
        @debug "Convergence metrics updated: stability=$(round(metrics.feature_stability, digits=3)), consensus=$(round(metrics.consensus_strength, digits=3)), entropy=$(round(metrics.selection_entropy, digits=3))"
    end
end

"""
Update sliding window with current convergence measurements
"""
function update_sliding_window!(detector::ConvergenceDetector)
    window = detector.sliding_window
    metrics = detector.metrics
    
    # Add current measurements to window
    push!(window.stability_history, metrics.feature_stability)
    push!(window.consensus_history, metrics.consensus_strength)
    push!(window.entropy_history, metrics.selection_entropy)
    push!(window.timestamps, now())
    
    # Maintain window size
    while length(window.stability_history) > window.window_size
        deleteat!(window.stability_history, 1)
        deleteat!(window.consensus_history, 1)
        deleteat!(window.entropy_history, 1)
        deleteat!(window.timestamps, 1)
    end
    
    # Calculate window statistics
    if length(window.stability_history) > 1
        window.window_variance = var(window.stability_history)
        
        # Calculate trend within window
        if length(window.stability_history) >= 5
            recent_values = window.stability_history[end-4:end]
            x_vals = collect(1:5)
            
            # Simple trend calculation
            sum_x = sum(x_vals)
            sum_y = sum(recent_values)
            sum_xy = sum(x_vals .* recent_values)
            sum_x2 = sum(x_vals .^ 2)
            
            denominator = 5 * sum_x2 - sum_x^2
            if denominator != 0
                window.window_trend = (5 * sum_xy - sum_x * sum_y) / denominator
            end
        end
    end
    
    # Check if window shows stability
    current_threshold = detector.current_thresholds["stability"]
    window.is_stable = (
        length(window.stability_history) >= detector.config.minimum_stable_windows &&
        all(s >= current_threshold - detector.config.stability_tolerance 
            for s in window.stability_history[end-detector.config.minimum_stable_windows+1:end])
    )
    
    # Update consecutive stable windows counter
    if window.is_stable
        window.consecutive_stable += 1
    else
        window.consecutive_stable = 0
    end
end

"""
Adapt convergence thresholds based on problem difficulty
"""
function adapt_convergence_thresholds!(detector::ConvergenceDetector)
    if detector.config.threshold_adaptation == STATIC_THRESHOLD
        return
    end
    
    current_iteration = detector.stats.total_iterations
    
    # Check if adaptation interval has passed
    if current_iteration % detector.config.adaptation_interval != 0
        return
    end
    
    metrics = detector.metrics
    
    if detector.config.threshold_adaptation == LINEAR_ADAPTATION
        # Linear decay towards minimum threshold
        progress = current_iteration / detector.config.maximum_iterations
        new_threshold = detector.config.initial_threshold - 
                       progress * (detector.config.initial_threshold - detector.config.minimum_threshold)
        
    elseif detector.config.threshold_adaptation == EXPONENTIAL_DECAY
        # Exponential decay
        new_threshold = detector.config.initial_threshold * 
                       (detector.config.threshold_decay_rate ^ (current_iteration / detector.config.adaptation_interval))
        
    elseif detector.config.threshold_adaptation == PROBLEM_ADAPTIVE
        # Adapt based on current stability trends
        if metrics.trend_slope > 0.001  # Improving stability
            new_threshold = detector.current_thresholds["stability"] * 1.01f0
        elseif metrics.trend_slope < -0.001  # Decreasing stability
            new_threshold = detector.current_thresholds["stability"] * 0.99f0
        else
            new_threshold = detector.current_thresholds["stability"]
        end
        
    elseif detector.config.threshold_adaptation == PERFORMANCE_BASED
        # Adapt based on ensemble performance
        if metrics.consensus_strength > 0.9
            new_threshold = detector.current_thresholds["stability"] * 0.98f0
        else
            new_threshold = detector.current_thresholds["stability"]
        end
    end
    
    # Apply threshold bounds
    new_threshold = clamp(new_threshold, detector.config.minimum_threshold, detector.config.initial_threshold)
    
    # Update threshold if changed significantly
    if abs(new_threshold - detector.current_thresholds["stability"]) > 0.01
        detector.current_thresholds["stability"] = new_threshold
        push!(detector.adaptive_thresholds, new_threshold)
        push!(detector.threshold_history, (now(), new_threshold))
        detector.stats.threshold_adaptations += 1
        
        @debug "Convergence threshold adapted to: $(round(new_threshold, digits=3))"
    end
end

"""
Check for convergence based on configured strategy
"""
function check_convergence(detector::ConvergenceDetector)::Bool
    metrics = detector.metrics
    window = detector.sliding_window
    config = detector.config
    
    # Must meet minimum iteration requirement
    if detector.stats.total_iterations < config.minimum_iterations
        return false
    end
    
    # Check maximum iteration limit
    if detector.stats.total_iterations >= config.maximum_iterations
        @info "Maximum iterations reached, forcing convergence"
        return true
    end
    
    convergence_checks = Bool[]
    
    if config.convergence_strategy in [STABILITY_BASED, HYBRID_CONVERGENCE]
        stability_converged = (
            metrics.feature_stability >= detector.current_thresholds["stability"] &&
            window.consecutive_stable >= config.minimum_stable_windows
        )
        push!(convergence_checks, stability_converged)
    end
    
    if config.convergence_strategy in [CONSENSUS_BASED, HYBRID_CONVERGENCE]
        consensus_converged = metrics.consensus_strength >= detector.current_thresholds["consensus"]
        push!(convergence_checks, consensus_converged)
    end
    
    if config.convergence_strategy in [ENTROPY_BASED, HYBRID_CONVERGENCE]
        entropy_converged = metrics.selection_entropy <= detector.current_thresholds["entropy"]
        push!(convergence_checks, entropy_converged)
    end
    
    if config.convergence_strategy == HYBRID_CONVERGENCE
        # Require majority of criteria to be met
        convergence_rate = sum(convergence_checks) / length(convergence_checks)
        return convergence_rate >= 0.6  # At least 60% of criteria met
    elseif config.convergence_strategy == ADAPTIVE_CONVERGENCE
        # Dynamic convergence based on ensemble state
        if metrics.feature_stability > 0.95 && metrics.consensus_strength > 0.9
            return true
        elseif metrics.selection_entropy < 0.1 && window.consecutive_stable >= 3
            return true
        else
            return false
        end
    else
        # Single strategy convergence
        return !isempty(convergence_checks) && all(convergence_checks)
    end
end

"""
Determine if early stopping should be triggered
"""
function should_stop_early(detector::ConvergenceDetector)::Bool
    if !detector.config.early_stopping_enabled
        return false
    end
    
    config = detector.config
    metrics = detector.metrics
    window = detector.sliding_window
    
    if config.stopping_criteria == FIXED_ITERATIONS
        return window.consecutive_stable >= config.convergence_patience
        
    elseif config.stopping_criteria == STATISTICAL_TEST
        # Simple statistical test for stability
        if length(window.stability_history) >= config.statistical_test_window
            recent_stability = window.stability_history[end-config.statistical_test_window+1:end]
            stability_variance = var(recent_stability)
            return stability_variance < config.variance_stability_factor
        end
        return false
        
    elseif config.stopping_criteria == PLATEAU_DETECTION
        # Detect when improvement plateaus
        if length(window.stability_history) >= 20
            recent_trend = window.window_trend
            return abs(recent_trend) < config.trend_detection_sensitivity
        end
        return false
        
    elseif config.stopping_criteria == ENSEMBLE_AGREEMENT
        return metrics.agreement_coefficient >= detector.current_thresholds["agreement"]
        
    elseif config.stopping_criteria == ADAPTIVE_STOPPING
        # Combination of multiple criteria
        stability_ok = window.consecutive_stable >= 3
        consensus_ok = metrics.consensus_strength >= 0.85
        entropy_ok = metrics.selection_entropy <= 0.2
        trend_ok = abs(window.window_trend) < 0.01
        
        criteria_met = sum([stability_ok, consensus_ok, entropy_ok, trend_ok])
        return criteria_met >= 3  # At least 3 out of 4 criteria
    end
    
    return false
end

"""
Run convergence detection cycle
"""
function run_convergence_detection!(detector::ConvergenceDetector,
                                  feature_selections::Vector{Vector{Int}},
                                  consensus_manager::ConsensusManager)::Bool
    start_time = time()
    
    # Update metrics
    update_convergence_metrics!(detector, feature_selections, consensus_manager)
    
    # Update sliding window
    update_sliding_window!(detector)
    
    # Adapt thresholds if needed
    adapt_convergence_thresholds!(detector)
    
    # Check for convergence
    converged = check_convergence(detector)
    
    if converged
        if !detector.is_converged
            # First time convergence detected
            detector.is_converged = true
            detector.convergence_timestamp = now()
            detector.stats.convergence_iterations = detector.stats.total_iterations
            detector.stats.early_stopping_triggers += 1
            
            @info "Convergence detected after $(detector.stats.total_iterations) iterations"
            push!(detector.convergence_log, "Convergence detected at $(now())")
        end
        
        detector.consecutive_stable_iterations += 1
    else
        if detector.is_converged
            # False convergence - reset
            detector.is_converged = false
            detector.convergence_timestamp = nothing
            detector.stats.false_convergences += 1
            detector.consecutive_stable_iterations = 0
            
            @debug "False convergence detected, resetting convergence state"
        end
    end
    
    # Update convergence confidence
    if detector.is_converged
        detector.convergence_confidence = min(1.0, 
            detector.consecutive_stable_iterations / detector.config.convergence_patience)
    else
        detector.convergence_confidence = 0.0
    end
    
    # Track analysis time
    analysis_time = (time() - start_time) * 1000
    push!(detector.analysis_times, analysis_time)
    if length(detector.analysis_times) > 1000
        deleteat!(detector.analysis_times, 1)
    end
    
    detector.stats.last_convergence_check = now()
    
    return detector.is_converged
end

"""
Get convergence detection status
"""
function get_convergence_status(detector::ConvergenceDetector)
    return Dict{String, Any}(
        "detector_state" => detector.detector_state,
        "is_converged" => detector.is_converged,
        "convergence_confidence" => detector.convergence_confidence,
        "convergence_timestamp" => detector.convergence_timestamp,
        "consecutive_stable_iterations" => detector.consecutive_stable_iterations,
        "total_iterations" => detector.stats.total_iterations,
        "convergence_iterations" => detector.stats.convergence_iterations,
        "false_convergences" => detector.stats.false_convergences,
        "early_stopping_triggers" => detector.stats.early_stopping_triggers,
        "threshold_adaptations" => detector.stats.threshold_adaptations,
        "current_stability" => detector.metrics.feature_stability,
        "current_consensus" => detector.metrics.consensus_strength,
        "current_entropy" => detector.metrics.selection_entropy,
        "current_agreement" => detector.metrics.agreement_coefficient,
        "stability_threshold" => detector.current_thresholds["stability"],
        "window_consecutive_stable" => detector.sliding_window.consecutive_stable,
        "average_analysis_time_ms" => length(detector.analysis_times) > 0 ? mean(detector.analysis_times) : 0.0,
        "last_convergence_check" => detector.stats.last_convergence_check
    )
end

"""
Generate convergence detection report
"""
function generate_convergence_report(detector::ConvergenceDetector)
    status = get_convergence_status(detector)
    
    report = String[]
    
    push!(report, "=== Convergence Detection Report ===")
    push!(report, "Generated: $(now())")
    push!(report, "Detector State: $(status["detector_state"])")
    push!(report, "")
    
    # Convergence status
    push!(report, "Convergence Status:")
    push!(report, "  Converged: $(status["is_converged"] ? "Yes" : "No")")
    if status["is_converged"]
        push!(report, "  Convergence Time: $(status["convergence_timestamp"])")
        push!(report, "  Convergence Iterations: $(status["convergence_iterations"])")
        push!(report, "  Confidence: $(round(status["convergence_confidence"], digits=3))")
    end
    push!(report, "  Total Iterations: $(status["total_iterations"])")
    push!(report, "  Consecutive Stable: $(status["consecutive_stable_iterations"])")
    push!(report, "")
    
    # Current metrics
    push!(report, "Current Metrics:")
    push!(report, "  Feature Stability: $(round(status["current_stability"], digits=3))")
    push!(report, "  Consensus Strength: $(round(status["current_consensus"], digits=3))")
    push!(report, "  Selection Entropy: $(round(status["current_entropy"], digits=3))")
    push!(report, "  Agreement Coefficient: $(round(status["current_agreement"], digits=3))")
    push!(report, "")
    
    # Threshold information
    push!(report, "Convergence Thresholds:")
    push!(report, "  Stability Threshold: $(round(status["stability_threshold"], digits=3))")
    push!(report, "  Threshold Adaptations: $(status["threshold_adaptations"])")
    push!(report, "")
    
    # Performance statistics
    push!(report, "Performance Statistics:")
    push!(report, "  False Convergences: $(status["false_convergences"])")
    push!(report, "  Early Stopping Triggers: $(status["early_stopping_triggers"])")
    push!(report, "  Average Analysis Time: $(round(status["average_analysis_time_ms"], digits=2))ms")
    push!(report, "")
    
    # Configuration summary
    push!(report, "Configuration:")
    push!(report, "  Strategy: $(detector.config.convergence_strategy)")
    push!(report, "  Stopping Criteria: $(detector.config.stopping_criteria)")
    push!(report, "  Threshold Adaptation: $(detector.config.threshold_adaptation)")
    push!(report, "  Window Size: $(detector.config.sliding_window_size)")
    push!(report, "  Early Stopping: $(detector.config.early_stopping_enabled)")
    
    push!(report, "=== End Convergence Report ===")
    
    return join(report, "\n")
end

"""
Reset convergence detector state
"""
function reset_convergence_detector!(detector::ConvergenceDetector)
    lock(detector.detector_lock) do
        detector.is_converged = false
        detector.convergence_confidence = 0.0
        detector.convergence_timestamp = nothing
        detector.consecutive_stable_iterations = 0
        
        # Reset metrics
        detector.metrics = create_convergence_metrics()
        
        # Reset window
        detector.sliding_window = create_convergence_window(detector.config.sliding_window_size)
        
        # Reset thresholds to initial values
        detector.current_thresholds["stability"] = detector.config.stability_threshold
        detector.current_thresholds["consensus"] = detector.config.consensus_threshold
        detector.current_thresholds["entropy"] = detector.config.entropy_threshold
        detector.current_thresholds["agreement"] = detector.config.agreement_threshold
        
        # Clear caches
        empty!(detector.feature_stability_cache)
        empty!(detector.consensus_cache)
        empty!(detector.entropy_cache)
        
        # Reset stats
        detector.stats = initialize_convergence_stats()
        
        @info "Convergence detector reset"
    end
end

"""
Cleanup convergence detector
"""
function cleanup_convergence_detector!(detector::ConvergenceDetector)
    lock(detector.detector_lock) do
        reset_convergence_detector!(detector)
        empty!(detector.analysis_times)
        empty!(detector.convergence_log)
    end
    
    detector.detector_state = "shutdown"
    @info "Convergence detector cleaned up"
end

# Export main types and functions
export ConvergenceStrategy, StoppingCriteria, ThresholdAdaptation
export STABILITY_BASED, CONSENSUS_BASED, ENTROPY_BASED, HYBRID_CONVERGENCE, ADAPTIVE_CONVERGENCE
export FIXED_ITERATIONS, STATISTICAL_TEST, PLATEAU_DETECTION, ENSEMBLE_AGREEMENT, ADAPTIVE_STOPPING
export STATIC_THRESHOLD, LINEAR_ADAPTATION, EXPONENTIAL_DECAY, PROBLEM_ADAPTIVE, PERFORMANCE_BASED
export ConvergenceMetrics, ConvergenceWindow, ConvergenceConfig, ConvergenceDetector
export create_convergence_config, initialize_convergence_detector
export update_convergence_metrics!, run_convergence_detection!, should_stop_early
export get_convergence_status, generate_convergence_report
export reset_convergence_detector!, cleanup_convergence_detector!

end # module ConvergenceDetection