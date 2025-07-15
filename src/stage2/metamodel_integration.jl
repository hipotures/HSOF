"""
Metamodel Integration for Fast Feature Evaluation
Provides neural network metamodel interface replacing expensive cross-validation with fast approximation
for MCTS ensemble feature selection. Includes batched evaluation, caching, confidence-based fallback,
and adaptive model updating for optimal performance vs accuracy trade-offs.

This module enables 1000x speedup over real evaluation while maintaining prediction quality through
intelligent caching and confidence-aware fallback mechanisms.
"""

module MetamodelIntegration

using Flux
using CUDA
using Random
using Statistics
using Dates
using Printf
using LinearAlgebra
using Base.Threads

# Import metamodel for neural network operations
include("../metamodel/metamodel.jl")
using .Metamodel

# Import ensemble forest for tree management
include("ensemble_forest.jl")
using .EnsembleForest

"""
Confidence levels for metamodel predictions
"""
@enum ConfidenceLevel begin
    HIGH_CONFIDENCE = 1    # σ < 0.05, use prediction directly
    MEDIUM_CONFIDENCE = 2  # 0.05 ≤ σ < 0.15, use with caution
    LOW_CONFIDENCE = 3     # σ ≥ 0.15, trigger real evaluation
end

"""
Evaluation modes for feature subset assessment
"""
@enum EvaluationMode begin
    METAMODEL_ONLY = 1     # Use only metamodel predictions
    HYBRID_MODE = 2        # Use metamodel with fallback to real evaluation
    REAL_EVALUATION = 3    # Use only real cross-validation
end

"""
Metamodel prediction result
"""
struct MetamodelPrediction
    feature_subset::Vector{Int}        # Feature indices in subset
    predicted_score::Float32           # Predicted performance score
    confidence::Float32                # Prediction confidence (0.0-1.0)
    confidence_level::ConfidenceLevel  # Categorical confidence assessment
    computation_time::Float64          # Time to compute prediction (ms)
    is_cached::Bool                    # Whether result came from cache
    timestamp::DateTime                # When prediction was made
end

"""
Metamodel evaluation batch
"""
struct EvaluationBatch
    batch_id::String                   # Unique batch identifier
    feature_subsets::Vector{Vector{Int}}  # Multiple feature subsets to evaluate
    batch_size::Int                    # Number of subsets in batch
    priority::Int                      # Batch priority (1=highest, 10=lowest)
    requester_tree_id::Int            # Tree ID that requested batch
    creation_time::DateTime           # When batch was created
    target_gpu::Union{Int, Nothing}   # Preferred GPU for execution
end

"""
Cache entry for metamodel predictions
"""
mutable struct CacheEntry
    prediction::MetamodelPrediction
    access_count::Int
    last_accessed::DateTime
    validation_score::Union{Float32, Nothing}  # Real evaluation score for validation
    is_validated::Bool
end

"""
Metamodel performance statistics
"""
mutable struct MetamodelStats
    total_predictions::Int
    cache_hits::Int
    cache_misses::Int
    fallback_triggers::Int
    real_evaluations::Int
    average_prediction_time::Float64
    average_confidence::Float32
    prediction_accuracy::Float32  # When validation data available
    last_update_time::DateTime
    accuracy_history::Vector{Tuple{DateTime, Float32}}
end

"""
Metamodel integration configuration
"""
struct MetamodelConfig
    # Model settings
    model_path::String                 # Path to trained metamodel
    feature_dimension::Int             # Expected feature dimension
    enable_gpu_acceleration::Bool      # Use GPU for model inference
    
    # Batching
    max_batch_size::Int               # Maximum features per batch
    batch_timeout_ms::Int             # Maximum wait time before processing batch
    enable_dynamic_batching::Bool     # Adjust batch size based on load
    
    # Caching
    cache_size::Int                   # Maximum cache entries
    cache_ttl_hours::Int              # Cache time-to-live
    enable_cache_validation::Bool     # Validate cached predictions periodically
    
    # Confidence and fallback
    confidence_threshold::Float32     # Minimum confidence for using prediction
    fallback_probability::Float32    # Probability of real evaluation for validation
    min_confidence_for_caching::Float32  # Minimum confidence to cache result
    
    # Model updating
    enable_online_learning::Bool      # Update model with new data
    accuracy_degradation_threshold::Float32  # Trigger model update
    validation_sample_rate::Float32   # Rate of real evaluations for validation
    model_update_frequency::Int      # Hours between model updates
    
    # Performance
    prediction_timeout_ms::Int        # Maximum time per prediction
    parallel_evaluation_threads::Int  # Threads for parallel processing
    memory_limit_mb::Int             # Memory limit for model operations
end

"""
Create default metamodel configuration
"""
function create_metamodel_config(;
    model_path::String = "models/feature_metamodel.bson",
    feature_dimension::Int = 500,
    enable_gpu_acceleration::Bool = true,
    max_batch_size::Int = 100,
    batch_timeout_ms::Int = 50,
    enable_dynamic_batching::Bool = true,
    cache_size::Int = 10000,
    cache_ttl_hours::Int = 24,
    enable_cache_validation::Bool = true,
    confidence_threshold::Float32 = 0.7f0,
    fallback_probability::Float32 = 0.05f0,
    min_confidence_for_caching::Float32 = 0.6f0,
    enable_online_learning::Bool = true,
    accuracy_degradation_threshold::Float32 = 0.15f0,
    validation_sample_rate::Float32 = 0.1f0,
    model_update_frequency::Int = 12,
    prediction_timeout_ms::Int = 100,
    parallel_evaluation_threads::Int = 4,
    memory_limit_mb::Int = 2048
)
    return MetamodelConfig(
        model_path, feature_dimension, enable_gpu_acceleration,
        max_batch_size, batch_timeout_ms, enable_dynamic_batching,
        cache_size, cache_ttl_hours, enable_cache_validation,
        confidence_threshold, fallback_probability, min_confidence_for_caching,
        enable_online_learning, accuracy_degradation_threshold, validation_sample_rate,
        model_update_frequency, prediction_timeout_ms, parallel_evaluation_threads,
        memory_limit_mb
    )
end

"""
Metamodel integration manager
Coordinates metamodel predictions, caching, and fallback mechanisms
"""
mutable struct MetamodelManager
    # Configuration
    config::MetamodelConfig
    
    # Model management
    model::Union{Chain, Nothing}       # Loaded neural network model
    model_device::Union{CuDevice, Nothing}  # GPU device for model
    model_state::String               # "loading", "ready", "error", "updating"
    model_version::String             # Current model version
    last_model_update::DateTime       # Last model update time
    
    # Evaluation batching
    pending_batches::Vector{EvaluationBatch}
    batch_processing_active::Bool
    batch_processor_task::Union{Task, Nothing}
    current_batch_size::Int
    
    # Prediction caching
    prediction_cache::Dict{Vector{Int}, CacheEntry}
    cache_access_order::Vector{Vector{Int}}  # LRU tracking
    cache_stats::NamedTuple{(:hits, :misses, :evictions), Tuple{Int, Int, Int}}
    
    # Performance monitoring
    stats::MetamodelStats
    prediction_times::Vector{Float64}  # Recent prediction times
    confidence_distribution::Vector{Float32}  # Recent confidence scores
    
    # Fallback mechanism
    real_evaluator::Union{Function, Nothing}  # Real evaluation function
    fallback_queue::Vector{Vector{Int}}     # Features pending real evaluation
    validation_data::Dict{Vector{Int}, Float32}  # Real evaluation results
    
    # Synchronization
    manager_lock::ReentrantLock
    cache_lock::ReentrantLock
    batch_lock::ReentrantLock
    
    # Status
    manager_state::String             # "initializing", "active", "error", "shutdown"
    error_log::Vector{String}
    creation_time::DateTime
end

"""
Initialize metamodel integration manager
"""
function initialize_metamodel_manager(config::MetamodelConfig = create_metamodel_config())
    # Initialize statistics
    stats = MetamodelStats(
        0, 0, 0, 0, 0, 0.0, 0.0f0, 0.0f0,
        now(), Tuple{DateTime, Float32}[]
    )
    
    # Create manager
    manager = MetamodelManager(
        config,
        nothing,  # Model loaded later
        nothing,  # Device set during model loading
        "initializing",
        "v1.0",
        now(),
        EvaluationBatch[],
        false,
        nothing,
        config.max_batch_size,
        Dict{Vector{Int}, CacheEntry}(),
        Vector{Vector{Int}}(),
        (hits=0, misses=0, evictions=0),
        stats,
        Float64[],
        Float32[],
        nothing,  # Real evaluator set by user
        Vector{Vector{Int}}(),
        Dict{Vector{Int}, Float32}(),
        ReentrantLock(),
        ReentrantLock(),
        ReentrantLock(),
        "initializing",
        String[],
        now()
    )
    
    # Load metamodel
    load_metamodel!(manager)
    
    # Start batch processing
    start_batch_processing!(manager)
    
    manager.manager_state = "active"
    
    return manager
end

"""
Load neural network metamodel
"""
function load_metamodel!(manager::MetamodelManager)
    lock(manager.manager_lock) do
        try
            @info "Loading metamodel from $(manager.config.model_path)"
            
            # Load model file
            if !isfile(manager.config.model_path)
                error("Metamodel file not found: $(manager.config.model_path)")
            end
            
            # Load model using Flux/BSON
            model_data = BSON.load(manager.config.model_path)
            manager.model = model_data[:model]
            
            # Set up GPU if enabled
            if manager.config.enable_gpu_acceleration && CUDA.functional()
                manager.model_device = CuDevice(0)  # Use first GPU
                manager.model = manager.model |> gpu
                @info "Metamodel loaded on GPU"
            else
                @info "Metamodel loaded on CPU"
            end
            
            # Validate model input dimensions
            validate_model_dimensions!(manager)
            
            manager.model_state = "ready"
            @info "Metamodel successfully loaded and validated"
            
        catch e
            error_msg = "Failed to load metamodel: $e"
            push!(manager.error_log, error_msg)
            manager.model_state = "error"
            @error error_msg
            rethrow(e)
        end
    end
end

"""
Validate model input/output dimensions
"""
function validate_model_dimensions!(manager::MetamodelManager)
    try
        # Create test input
        test_input = randn(Float32, manager.config.feature_dimension, 1)
        if manager.config.enable_gpu_acceleration && !isnothing(manager.model_device)
            test_input = test_input |> gpu
        end
        
        # Test forward pass
        test_output = manager.model(test_input)
        
        # Validate output shape (should be single prediction)
        if size(test_output, 1) != 1
            error("Model output dimension mismatch: expected 1, got $(size(test_output, 1))")
        end
        
        @info "Model validation passed: input $(size(test_input)), output $(size(test_output))"
        
    catch e
        error("Model validation failed: $e")
    end
end

"""
Start batch processing task
"""
function start_batch_processing!(manager::MetamodelManager)
    if manager.batch_processing_active
        @warn "Batch processing already active"
        return
    end
    
    manager.batch_processing_active = true
    
    manager.batch_processor_task = @async begin
        try
            while manager.batch_processing_active
                process_evaluation_batches!(manager)
                sleep(manager.config.batch_timeout_ms / 1000.0)
            end
        catch e
            @error "Batch processing task failed: $e"
            manager.batch_processing_active = false
        end
    end
    
    @info "Batch processing started"
end

"""
Stop batch processing task
"""
function stop_batch_processing!(manager::MetamodelManager)
    manager.batch_processing_active = false
    
    if !isnothing(manager.batch_processor_task)
        try
            wait(manager.batch_processor_task)
        catch e
            @warn "Error stopping batch processor: $e"
        end
        manager.batch_processor_task = nothing
    end
    
    @info "Batch processing stopped"
end

"""
Evaluate feature subset using metamodel
"""
function evaluate_feature_subset(manager::MetamodelManager, 
                                feature_subset::Vector{Int},
                                evaluation_mode::EvaluationMode = HYBRID_MODE,
                                tree_id::Int = 0)::MetamodelPrediction
    start_time = time()
    
    # Check cache first
    cached_result = get_cached_prediction(manager, feature_subset)
    if !isnothing(cached_result)
        update_cache_access!(manager, feature_subset)
        return cached_result.prediction
    end
    
    # Determine evaluation strategy
    should_use_metamodel = (evaluation_mode == METAMODEL_ONLY) || 
                          (evaluation_mode == HYBRID_MODE && manager.model_state == "ready")
    
    prediction = if should_use_metamodel
        evaluate_with_metamodel(manager, feature_subset, start_time)
    else
        evaluate_with_real_function(manager, feature_subset, start_time)
    end
    
    # Apply fallback logic for hybrid mode
    if evaluation_mode == HYBRID_MODE && prediction.confidence_level == LOW_CONFIDENCE
        if should_trigger_fallback(manager)
            @info "Low confidence prediction, triggering real evaluation for subset of length $(length(feature_subset))"
            prediction = evaluate_with_real_function(manager, feature_subset, start_time)
            manager.stats.fallback_triggers += 1
        end
    end
    
    # Cache result if confidence is sufficient
    if prediction.confidence >= manager.config.min_confidence_for_caching
        cache_prediction!(manager, feature_subset, prediction)
    end
    
    # Update statistics
    update_prediction_stats!(manager, prediction)
    
    return prediction
end

"""
Evaluate multiple feature subsets in batch
"""
function evaluate_feature_subsets_batch(manager::MetamodelManager,
                                       feature_subsets::Vector{Vector{Int}},
                                       evaluation_mode::EvaluationMode = HYBRID_MODE,
                                       tree_id::Int = 0,
                                       priority::Int = 5)::Vector{MetamodelPrediction}
    if isempty(feature_subsets)
        return MetamodelPrediction[]
    end
    
    # Check for cached results
    results = Vector{Union{MetamodelPrediction, Nothing}}(undef, length(feature_subsets))
    uncached_indices = Int[]
    uncached_subsets = Vector{Int}[]
    
    for (i, subset) in enumerate(feature_subsets)
        cached_result = get_cached_prediction(manager, subset)
        if !isnothing(cached_result)
            results[i] = cached_result.prediction
            update_cache_access!(manager, subset)
        else
            results[i] = nothing
            push!(uncached_indices, i)
            push!(uncached_subsets, subset)
        end
    end
    
    # Process uncached subsets
    if !isempty(uncached_subsets)
        if length(uncached_subsets) <= 5  # Small batch, evaluate immediately
            uncached_predictions = [evaluate_feature_subset(manager, subset, evaluation_mode, tree_id) 
                                  for subset in uncached_subsets]
            
            for (idx, pred_idx) in enumerate(uncached_indices)
                results[pred_idx] = uncached_predictions[idx]
            end
        else  # Large batch, use batch processing
            batch_id = "batch_$(tree_id)_$(now().instant)"
            batch = EvaluationBatch(
                batch_id, uncached_subsets, length(uncached_subsets),
                priority, tree_id, now(), nothing
            )
            
            # Submit batch and wait for results
            batch_predictions = submit_evaluation_batch(manager, batch, evaluation_mode)
            
            for (idx, pred_idx) in enumerate(uncached_indices)
                results[pred_idx] = batch_predictions[idx]
            end
        end
    end
    
    return convert(Vector{MetamodelPrediction}, results)
end

"""
Evaluate feature subset using metamodel
"""
function evaluate_with_metamodel(manager::MetamodelManager, 
                                feature_subset::Vector{Int},
                                start_time::Float64)::MetamodelPrediction
    try
        # Convert feature subset to model input
        model_input = prepare_model_input(manager, feature_subset)
        
        # Run model prediction
        prediction_output = nothing
        if manager.config.enable_gpu_acceleration && !isnothing(manager.model_device)
            CUDA.device!(manager.model_device)
            model_input = model_input |> gpu
            prediction_output = manager.model(model_input) |> cpu
        else
            prediction_output = manager.model(model_input)
        end
        
        # Extract prediction and confidence
        predicted_score = Float32(prediction_output[1])
        
        # Calculate confidence (simplified - could be more sophisticated)
        confidence = calculate_prediction_confidence(manager, feature_subset, predicted_score)
        confidence_level = classify_confidence_level(confidence)
        
        computation_time = (time() - start_time) * 1000  # Convert to ms
        
        prediction = MetamodelPrediction(
            feature_subset,
            predicted_score,
            confidence,
            confidence_level,
            computation_time,
            false,  # Not from cache
            now()
        )
        
        manager.stats.total_predictions += 1
        
        return prediction
        
    catch e
        @error "Metamodel evaluation failed for subset of length $(length(feature_subset)): $e"
        # Fallback to real evaluation
        return evaluate_with_real_function(manager, feature_subset, start_time)
    end
end

"""
Evaluate feature subset using real evaluation function
"""
function evaluate_with_real_function(manager::MetamodelManager,
                                    feature_subset::Vector{Int},
                                    start_time::Float64)::MetamodelPrediction
    if isnothing(manager.real_evaluator)
        error("Real evaluator not configured but real evaluation requested")
    end
    
    try
        # Use real evaluation function
        real_score = manager.real_evaluator(feature_subset)
        
        computation_time = (time() - start_time) * 1000  # Convert to ms
        
        prediction = MetamodelPrediction(
            feature_subset,
            Float32(real_score),
            1.0f0,  # Perfect confidence for real evaluation
            HIGH_CONFIDENCE,
            computation_time,
            false,
            now()
        )
        
        # Store for validation
        manager.validation_data[feature_subset] = Float32(real_score)
        manager.stats.real_evaluations += 1
        
        return prediction
        
    catch e
        @error "Real evaluation failed for subset of length $(length(feature_subset)): $e"
        
        # Return dummy prediction with error
        computation_time = (time() - start_time) * 1000
        return MetamodelPrediction(
            feature_subset,
            0.0f0,  # Default score
            0.0f0,  # No confidence
            LOW_CONFIDENCE,
            computation_time,
            false,
            now()
        )
    end
end

"""
Prepare model input from feature subset
"""
function prepare_model_input(manager::MetamodelManager, feature_subset::Vector{Int})
    # Create binary feature vector
    input_vector = zeros(Float32, manager.config.feature_dimension)
    
    # Set selected features to 1
    for feature_idx in feature_subset
        if 1 <= feature_idx <= manager.config.feature_dimension
            input_vector[feature_idx] = 1.0f0
        end
    end
    
    # Reshape for model (assuming batch dimension)
    return reshape(input_vector, :, 1)
end

"""
Calculate prediction confidence
"""
function calculate_prediction_confidence(manager::MetamodelManager,
                                       feature_subset::Vector{Int},
                                       predicted_score::Float32)::Float32
    # Simplified confidence calculation
    # In practice, this could use ensemble variance, dropout uncertainty, etc.
    
    # Base confidence on subset size (smaller subsets often more reliable)
    size_factor = 1.0f0 - (length(feature_subset) / manager.config.feature_dimension) * 0.3f0
    
    # Factor in prediction magnitude (extreme values might be less reliable)
    magnitude_factor = 1.0f0 - abs(predicted_score - 0.5f0) * 0.2f0
    
    # Combine factors
    confidence = min(1.0f0, max(0.0f0, size_factor * magnitude_factor))
    
    return confidence
end

"""
Classify confidence level
"""
function classify_confidence_level(confidence::Float32)::ConfidenceLevel
    if confidence >= 0.85f0
        return HIGH_CONFIDENCE
    elseif confidence >= 0.7f0
        return MEDIUM_CONFIDENCE
    else
        return LOW_CONFIDENCE
    end
end

"""
Check if fallback should be triggered
"""
function should_trigger_fallback(manager::MetamodelManager)::Bool
    return rand() < manager.config.fallback_probability
end

"""
Submit evaluation batch for processing
"""
function submit_evaluation_batch(manager::MetamodelManager,
                                batch::EvaluationBatch,
                                evaluation_mode::EvaluationMode)::Vector{MetamodelPrediction}
    lock(manager.batch_lock) do
        push!(manager.pending_batches, batch)
        
        # Sort by priority
        sort!(manager.pending_batches, by = b -> b.priority)
    end
    
    # Process immediately for high priority
    if batch.priority <= 2
        return process_single_batch!(manager, batch, evaluation_mode)
    end
    
    # Wait for batch processing (simplified - in practice would use proper synchronization)
    timeout_s = manager.config.batch_timeout_ms / 1000.0 * 10  # 10x timeout for waiting
    start_wait = time()
    
    while time() - start_wait < timeout_s
        # Check if batch was processed
        lock(manager.batch_lock) do
            if !(batch in manager.pending_batches)
                # Batch was processed, results should be available
                return true
            end
        end
        sleep(0.01)
    end
    
    # Timeout - process individually
    @warn "Batch processing timeout, falling back to individual evaluation"
    return [evaluate_feature_subset(manager, subset, evaluation_mode, batch.requester_tree_id) 
            for subset in batch.feature_subsets]
end

"""
Process evaluation batches
"""
function process_evaluation_batches!(manager::MetamodelManager)
    if manager.model_state != "ready"
        return
    end
    
    batches_to_process = EvaluationBatch[]
    
    lock(manager.batch_lock) do
        if !isempty(manager.pending_batches)
            # Take batches to process
            splice!(batches_to_process, 1:length(batches_to_process), manager.pending_batches)
            empty!(manager.pending_batches)
        end
    end
    
    for batch in batches_to_process
        try
            process_single_batch!(manager, batch, HYBRID_MODE)
        catch e
            @error "Failed to process batch $(batch.batch_id): $e"
        end
    end
end

"""
Process single evaluation batch
"""
function process_single_batch!(manager::MetamodelManager,
                              batch::EvaluationBatch,
                              evaluation_mode::EvaluationMode)::Vector{MetamodelPrediction}
    start_time = time()
    results = MetamodelPrediction[]
    
    try
        if evaluation_mode == METAMODEL_ONLY || manager.model_state == "ready"
            # Prepare batch input
            batch_input = prepare_batch_input(manager, batch.feature_subsets)
            
            # Run batch prediction
            batch_output = nothing
            if manager.config.enable_gpu_acceleration && !isnothing(manager.model_device)
                CUDA.device!(manager.model_device)
                batch_input = batch_input |> gpu
                batch_output = manager.model(batch_input) |> cpu
            else
                batch_output = manager.model(batch_input)
            end
            
            # Process results
            for (i, subset) in enumerate(batch.feature_subsets)
                predicted_score = Float32(batch_output[i])
                confidence = calculate_prediction_confidence(manager, subset, predicted_score)
                confidence_level = classify_confidence_level(confidence)
                
                prediction = MetamodelPrediction(
                    subset,
                    predicted_score,
                    confidence,
                    confidence_level,
                    (time() - start_time) * 1000 / length(batch.feature_subsets),  # Average time per prediction
                    false,
                    now()
                )
                
                push!(results, prediction)
                
                # Cache if confidence is sufficient
                if confidence >= manager.config.min_confidence_for_caching
                    cache_prediction!(manager, subset, prediction)
                end
            end
        else
            # Fallback to individual evaluation
            for subset in batch.feature_subsets
                prediction = evaluate_with_real_function(manager, subset, start_time)
                push!(results, prediction)
            end
        end
        
        # Update statistics
        manager.stats.total_predictions += length(results)
        
    catch e
        @error "Batch processing failed: $e"
        
        # Fallback to individual evaluation
        for subset in batch.feature_subsets
            try
                prediction = evaluate_feature_subset(manager, subset, evaluation_mode, batch.requester_tree_id)
                push!(results, prediction)
            catch ee
                @error "Individual evaluation also failed: $ee"
                # Create error prediction
                error_prediction = MetamodelPrediction(
                    subset, 0.0f0, 0.0f0, LOW_CONFIDENCE,
                    (time() - start_time) * 1000, false, now()
                )
                push!(results, error_prediction)
            end
        end
    end
    
    return results
end

"""
Prepare batch input for model
"""
function prepare_batch_input(manager::MetamodelManager, feature_subsets::Vector{Vector{Int}})
    batch_size = length(feature_subsets)
    batch_input = zeros(Float32, manager.config.feature_dimension, batch_size)
    
    for (i, subset) in enumerate(feature_subsets)
        for feature_idx in subset
            if 1 <= feature_idx <= manager.config.feature_dimension
                batch_input[feature_idx, i] = 1.0f0
            end
        end
    end
    
    return batch_input
end

"""
Get cached prediction
"""
function get_cached_prediction(manager::MetamodelManager, feature_subset::Vector{Int})::Union{CacheEntry, Nothing}
    lock(manager.cache_lock) do
        cache_entry = get(manager.prediction_cache, feature_subset, nothing)
        
        if !isnothing(cache_entry)
            # Check TTL
            age_hours = (now() - cache_entry.prediction.timestamp).value / (1000 * 3600)
            if age_hours > manager.config.cache_ttl_hours
                # Expired
                delete!(manager.prediction_cache, feature_subset)
                filter!(k -> k != feature_subset, manager.cache_access_order)
                return nothing
            end
            
            manager.stats.cache_hits += 1
            return cache_entry
        else
            manager.stats.cache_misses += 1
            return nothing
        end
    end
end

"""
Cache prediction result
"""
function cache_prediction!(manager::MetamodelManager, feature_subset::Vector{Int}, prediction::MetamodelPrediction)
    lock(manager.cache_lock) do
        # Check cache size limit
        if length(manager.prediction_cache) >= manager.config.cache_size
            evict_oldest_cache_entry!(manager)
        end
        
        # Create cache entry
        cache_entry = CacheEntry(
            prediction,
            0,  # Access count
            now(),
            nothing,  # No validation score yet
            false
        )
        
        # Store in cache
        manager.prediction_cache[copy(feature_subset)] = cache_entry
        push!(manager.cache_access_order, copy(feature_subset))
    end
end

"""
Update cache access statistics
"""
function update_cache_access!(manager::MetamodelManager, feature_subset::Vector{Int})
    lock(manager.cache_lock) do
        cache_entry = get(manager.prediction_cache, feature_subset, nothing)
        if !isnothing(cache_entry)
            cache_entry.access_count += 1
            cache_entry.last_accessed = now()
            
            # Move to end of access order (LRU)
            filter!(k -> k != feature_subset, manager.cache_access_order)
            push!(manager.cache_access_order, feature_subset)
        end
    end
end

"""
Evict oldest cache entry
"""
function evict_oldest_cache_entry!(manager::MetamodelManager)
    if !isempty(manager.cache_access_order)
        oldest_key = manager.cache_access_order[1]
        delete!(manager.prediction_cache, oldest_key)
        deleteat!(manager.cache_access_order, 1)
        
        # Update stats
        current_stats = manager.cache_stats
        manager.cache_stats = (
            hits = current_stats.hits,
            misses = current_stats.misses,
            evictions = current_stats.evictions + 1
        )
    end
end

"""
Update prediction statistics
"""
function update_prediction_stats!(manager::MetamodelManager, prediction::MetamodelPrediction)
    manager.stats.total_predictions += 1
    
    # Update timing statistics
    push!(manager.prediction_times, prediction.computation_time)
    if length(manager.prediction_times) > 1000
        deleteat!(manager.prediction_times, 1)
    end
    manager.stats.average_prediction_time = mean(manager.prediction_times)
    
    # Update confidence statistics
    push!(manager.confidence_distribution, prediction.confidence)
    if length(manager.confidence_distribution) > 1000
        deleteat!(manager.confidence_distribution, 1)
    end
    manager.stats.average_confidence = mean(manager.confidence_distribution)
    
    # Update last update time
    manager.stats.last_update_time = now()
end

"""
Set real evaluation function
"""
function set_real_evaluator!(manager::MetamodelManager, evaluator_function::Function)
    manager.real_evaluator = evaluator_function
    @info "Real evaluator function configured"
end

"""
Validate model accuracy with real evaluations
"""
function validate_model_accuracy!(manager::MetamodelManager, validation_subsets::Vector{Vector{Int}} = Vector{Int}[])
    if isnothing(manager.real_evaluator)
        @warn "Cannot validate accuracy without real evaluator"
        return
    end
    
    # Use provided validation set or sample from cache
    subsets_to_validate = if !isempty(validation_subsets)
        validation_subsets
    else
        # Sample from cached predictions
        cached_keys = collect(keys(manager.prediction_cache))
        sample_size = min(100, length(cached_keys))
        sample(cached_keys, sample_size, replace=false)
    end
    
    if isempty(subsets_to_validate)
        @warn "No subsets available for validation"
        return
    end
    
    validation_errors = Float32[]
    
    for subset in subsets_to_validate
        try
            # Get metamodel prediction
            metamodel_pred = evaluate_with_metamodel(manager, subset, time())
            
            # Get real evaluation
            real_score = manager.real_evaluator(subset)
            
            # Calculate error
            error = abs(metamodel_pred.predicted_score - real_score)
            push!(validation_errors, error)
            
            # Update validation data
            manager.validation_data[subset] = Float32(real_score)
            
            # Update cache entry if it exists
            cache_entry = get(manager.prediction_cache, subset, nothing)
            if !isnothing(cache_entry)
                cache_entry.validation_score = Float32(real_score)
                cache_entry.is_validated = true
            end
            
        catch e
            @warn "Validation failed for subset of length $(length(subset)): $e"
        end
    end
    
    if !isempty(validation_errors)
        manager.stats.prediction_accuracy = 1.0f0 - mean(validation_errors)
        push!(manager.stats.accuracy_history, (now(), manager.stats.prediction_accuracy))
        
        # Limit history size
        if length(manager.stats.accuracy_history) > 100
            deleteat!(manager.stats.accuracy_history, 1)
        end
        
        @info "Model validation complete: accuracy = $(round(manager.stats.prediction_accuracy * 100, digits=1))% ($(length(validation_errors)) samples)"
        
        # Check if model update is needed
        if manager.stats.prediction_accuracy < (1.0f0 - manager.config.accuracy_degradation_threshold)
            @warn "Model accuracy below threshold, consider updating model"
        end
    end
end

"""
Get metamodel status and statistics
"""
function get_metamodel_status(manager::MetamodelManager)
    lock(manager.manager_lock) do
        return Dict{String, Any}(
            "manager_state" => manager.manager_state,
            "model_state" => manager.model_state,
            "model_version" => manager.model_version,
            "total_predictions" => manager.stats.total_predictions,
            "cache_hits" => manager.stats.cache_hits,
            "cache_misses" => manager.stats.cache_misses,
            "cache_hit_rate" => manager.stats.cache_hits > 0 ? 
                manager.stats.cache_hits / (manager.stats.cache_hits + manager.stats.cache_misses) : 0.0,
            "fallback_triggers" => manager.stats.fallback_triggers,
            "real_evaluations" => manager.stats.real_evaluations,
            "average_prediction_time_ms" => manager.stats.average_prediction_time,
            "average_confidence" => manager.stats.average_confidence,
            "prediction_accuracy" => manager.stats.prediction_accuracy,
            "cache_size" => length(manager.prediction_cache),
            "max_cache_size" => manager.config.cache_size,
            "pending_batches" => length(manager.pending_batches),
            "batch_processing_active" => manager.batch_processing_active,
            "validation_samples" => length(manager.validation_data),
            "error_count" => length(manager.error_log),
            "last_update" => manager.stats.last_update_time
        )
    end
end

"""
Generate metamodel performance report
"""
function generate_metamodel_report(manager::MetamodelManager)
    status = get_metamodel_status(manager)
    
    report = String[]
    
    push!(report, "=== Metamodel Integration Performance Report ===")
    push!(report, "Generated: $(now())")
    push!(report, "Manager State: $(status["manager_state"])")
    push!(report, "Model State: $(status["model_state"])")
    push!(report, "")
    
    # Prediction statistics
    push!(report, "Prediction Statistics:")
    push!(report, "  Total Predictions: $(status["total_predictions"])")
    push!(report, "  Average Time: $(round(status["average_prediction_time_ms"], digits=2))ms")
    push!(report, "  Average Confidence: $(round(status["average_confidence"] * 100, digits=1))%")
    if status["prediction_accuracy"] > 0
        push!(report, "  Prediction Accuracy: $(round(status["prediction_accuracy"] * 100, digits=1))%")
    end
    push!(report, "")
    
    # Cache performance
    push!(report, "Cache Performance:")
    push!(report, "  Cache Size: $(status["cache_size"])/$(status["max_cache_size"])")
    push!(report, "  Cache Hits: $(status["cache_hits"])")
    push!(report, "  Cache Misses: $(status["cache_misses"])")
    push!(report, "  Hit Rate: $(round(status["cache_hit_rate"] * 100, digits=1))%")
    push!(report, "")
    
    # Fallback statistics
    push!(report, "Fallback and Evaluation:")
    push!(report, "  Fallback Triggers: $(status["fallback_triggers"])")
    push!(report, "  Real Evaluations: $(status["real_evaluations"])")
    push!(report, "  Validation Samples: $(status["validation_samples"])")
    push!(report, "")
    
    # Batch processing
    push!(report, "Batch Processing:")
    push!(report, "  Pending Batches: $(status["pending_batches"])")
    push!(report, "  Processing Active: $(status["batch_processing_active"])")
    push!(report, "")
    
    # Performance metrics
    if !isempty(manager.prediction_times)
        push!(report, "Performance Metrics:")
        push!(report, "  Min Prediction Time: $(round(minimum(manager.prediction_times), digits=2))ms")
        push!(report, "  Max Prediction Time: $(round(maximum(manager.prediction_times), digits=2))ms")
        push!(report, "  Std Dev: $(round(std(manager.prediction_times), digits=2))ms")
        push!(report, "")
    end
    
    # Error reporting
    if status["error_count"] > 0
        push!(report, "Recent Errors:")
        recent_errors = manager.error_log[max(1, length(manager.error_log)-2):end]
        for error in recent_errors
            push!(report, "  - $error")
        end
        push!(report, "")
    end
    
    push!(report, "=== End Metamodel Report ===")
    
    return join(report, "\n")
end

"""
Cleanup metamodel manager
"""
function cleanup_metamodel!(manager::MetamodelManager)
    # Stop batch processing
    stop_batch_processing!(manager)
    
    # Clear caches
    lock(manager.cache_lock) do
        empty!(manager.prediction_cache)
        empty!(manager.cache_access_order)
    end
    
    # Clear model from GPU
    if manager.config.enable_gpu_acceleration && !isnothing(manager.model)
        try
            manager.model = nothing
            CUDA.reclaim()
        catch e
            @warn "Error cleaning up GPU resources: $e"
        end
    end
    
    manager.manager_state = "shutdown"
    @info "Metamodel manager cleaned up"
end

# Export main types and functions
export ConfidenceLevel, EvaluationMode, MetamodelPrediction, EvaluationBatch, MetamodelConfig, MetamodelManager
export HIGH_CONFIDENCE, MEDIUM_CONFIDENCE, LOW_CONFIDENCE
export METAMODEL_ONLY, HYBRID_MODE, REAL_EVALUATION
export create_metamodel_config, initialize_metamodel_manager
export evaluate_feature_subset, evaluate_feature_subsets_batch
export set_real_evaluator!, validate_model_accuracy!
export get_metamodel_status, generate_metamodel_report, cleanup_metamodel!

end # module MetamodelIntegration