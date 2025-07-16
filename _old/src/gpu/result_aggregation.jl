module ResultAggregation

using CUDA
using Dates
using Printf
using Statistics
using Base.Threads: Atomic, @spawn, ReentrantLock

export ResultAggregator, TreeResult, FeatureScore, EnsembleResult
export create_result_aggregator, submit_tree_result!, aggregate_results
export get_ensemble_consensus, get_feature_rankings, get_aggregated_results
export clear_results!, reset_cache!, get_cache_stats
export set_consensus_threshold!, enable_caching!

"""
Individual tree result from a single GPU
"""
struct TreeResult
    tree_id::Int
    gpu_id::Int
    timestamp::DateTime
    selected_features::Vector{Int}
    feature_scores::Dict{Int, Float32}
    confidence::Float32
    iterations::Int
    compute_time::Float64  # milliseconds
    
    function TreeResult(
        tree_id::Int,
        gpu_id::Int,
        selected_features::Vector{Int},
        feature_scores::Dict{Int, Float32} = Dict{Int, Float32}(),
        confidence::Float32 = 1.0f0,
        iterations::Int = 0,
        compute_time::Float64 = 0.0
    )
        new(
            tree_id,
            gpu_id,
            now(),
            selected_features,
            feature_scores,
            confidence,
            iterations,
            compute_time
        )
    end
end

"""
Aggregated feature score across multiple trees
"""
mutable struct FeatureScore
    feature_id::Int
    total_score::Float32
    selection_count::Int
    average_score::Float32
    confidence::Float32
    gpu_votes::Dict{Int, Int}  # GPU ID -> vote count
    
    function FeatureScore(feature_id::Int)
        new(
            feature_id,
            0.0f0,
            0,
            0.0f0,
            0.0f0,
            Dict{Int, Int}()
        )
    end
end

"""
Final ensemble result combining all trees
"""
struct EnsembleResult
    timestamp::DateTime
    total_trees::Int
    participating_gpus::Vector{Int}
    selected_features::Vector{Int}
    feature_rankings::Vector{Tuple{Int, Float32}}  # (feature_id, score)
    consensus_features::Vector{Int}  # Features above consensus threshold
    confidence::Float32
    aggregation_time::Float64  # milliseconds
    cache_hit::Bool
end

"""
Cache entry for storing aggregated results
"""
mutable struct CacheEntry
    result::EnsembleResult
    tree_results_hash::UInt64
    access_count::Atomic{Int}
    last_access::DateTime
    
    function CacheEntry(result::EnsembleResult, hash::UInt64)
        new(
            result,
            hash,
            Atomic{Int}(1),
            now()
        )
    end
end

"""
Main result aggregator for multi-GPU ensemble
"""
mutable struct ResultAggregator
    # Configuration
    num_gpus::Int
    total_trees::Int
    consensus_threshold::Float32  # Percentage of trees for consensus
    top_k_features::Int
    
    # Result storage
    tree_results::Dict{Int, TreeResult}
    result_lock::ReentrantLock
    
    # Aggregation state
    feature_scores::Dict{Int, FeatureScore}
    last_aggregation::Union{Nothing, EnsembleResult}
    aggregation_count::Atomic{Int}
    
    # Caching
    caching_enabled::Bool
    result_cache::Dict{UInt64, CacheEntry}
    cache_size::Int
    cache_hits::Atomic{Int}
    cache_misses::Atomic{Int}
    
    # GPU tracking
    gpu_tree_counts::Dict{Int, Int}
    gpu_response_times::Dict{Int, Vector{Float64}}
    
    # Callbacks
    pre_aggregation_callbacks::Vector{Function}
    post_aggregation_callbacks::Vector{Function}
    
    function ResultAggregator(;
        num_gpus::Int = 2,
        total_trees::Int = 100,
        consensus_threshold::Float32 = 0.5f0,  # 50% of trees
        top_k_features::Int = 50,
        caching_enabled::Bool = true,
        cache_size::Int = 100
    )
        new(
            num_gpus,
            total_trees,
            consensus_threshold,
            top_k_features,
            Dict{Int, TreeResult}(),
            ReentrantLock(),
            Dict{Int, FeatureScore}(),
            nothing,
            Atomic{Int}(0),
            caching_enabled,
            Dict{UInt64, CacheEntry}(),
            cache_size,
            Atomic{Int}(0),
            Atomic{Int}(0),
            Dict(i => 0 for i in 0:num_gpus-1),
            Dict(i => Float64[] for i in 0:num_gpus-1),
            Function[],
            Function[]
        )
    end
end

"""
Create and initialize result aggregator
"""
function create_result_aggregator(;kwargs...)
    return ResultAggregator(;kwargs...)
end

"""
Submit result from a single tree
"""
function submit_tree_result!(
    aggregator::ResultAggregator,
    tree_result::TreeResult
)
    lock(aggregator.result_lock) do
        # Store result
        aggregator.tree_results[tree_result.tree_id] = tree_result
        
        # Update GPU tracking
        gpu_id = tree_result.gpu_id
        aggregator.gpu_tree_counts[gpu_id] = get(aggregator.gpu_tree_counts, gpu_id, 0) + 1
        
        # Track response time
        if haskey(aggregator.gpu_response_times, gpu_id)
            push!(aggregator.gpu_response_times[gpu_id], tree_result.compute_time)
            
            # Keep only recent times (last 100)
            if length(aggregator.gpu_response_times[gpu_id]) > 100
                popfirst!(aggregator.gpu_response_times[gpu_id])
            end
        end
        
        # Invalidate cache since we have new results
        aggregator.last_aggregation = nothing
    end
    
    @info "Tree result submitted" tree_id=tree_result.tree_id gpu_id=tree_result.gpu_id features=length(tree_result.selected_features)
end

"""
Aggregate results from all submitted trees
"""
function aggregate_results(aggregator::ResultAggregator)::EnsembleResult
    start_time = time()
    
    # Check cache first
    if aggregator.caching_enabled && !isnothing(aggregator.last_aggregation)
        cache_key = compute_results_hash(aggregator)
        
        if haskey(aggregator.result_cache, cache_key)
            # Cache hit
            cache_entry = aggregator.result_cache[cache_key]
            cache_entry.access_count[] += 1
            cache_entry.last_access = now()
            aggregator.cache_hits[] += 1
            
            @info "Cache hit for result aggregation"
            return cache_entry.result
        else
            aggregator.cache_misses[] += 1
        end
    end
    
    # Call pre-aggregation callbacks
    for callback in aggregator.pre_aggregation_callbacks
        try
            callback(aggregator)
        catch e
            @error "Pre-aggregation callback error" exception=e
        end
    end
    
    result = lock(aggregator.result_lock) do
        # Reset feature scores
        empty!(aggregator.feature_scores)
        
        # Collect all tree results
        tree_results = collect(values(aggregator.tree_results))
        
        if isempty(tree_results)
            @warn "No tree results to aggregate"
            return EnsembleResult(
                now(), 0, Int[], Int[], Tuple{Int, Float32}[],
                Int[], 0.0f0, 0.0, false
            )
        end
        
        # Aggregate feature scores
        for tree_result in tree_results
            process_tree_result!(aggregator, tree_result)
        end
        
        # Calculate final rankings
        feature_rankings = calculate_feature_rankings(aggregator)
        
        # Determine consensus features
        consensus_features = find_consensus_features(
            aggregator,
            length(tree_results)
        )
        
        # Select top features
        selected_features = [f[1] for f in feature_rankings[1:min(aggregator.top_k_features, end)]]
        
        # Calculate ensemble confidence
        confidence = calculate_ensemble_confidence(aggregator, tree_results)
        
        # Get participating GPUs
        participating_gpus = unique([tr.gpu_id for tr in tree_results])
        
        # Create result
        ensemble_result = EnsembleResult(
            now(),
            length(tree_results),
            participating_gpus,
            selected_features,
            feature_rankings,
            consensus_features,
            confidence,
            (time() - start_time) * 1000,
            false
        )
        
        # Cache result
        if aggregator.caching_enabled
            cache_key2 = compute_results_hash(aggregator)
            cache_result!(aggregator, ensemble_result, cache_key2)
        end
        
        # Update state
        aggregator.last_aggregation = ensemble_result
        aggregator.aggregation_count[] += 1
        
        ensemble_result
    end
    
    # Call post-aggregation callbacks
    for callback in aggregator.post_aggregation_callbacks
        try
            callback(result)
        catch e
            @error "Post-aggregation callback error" exception=e
        end
    end
    
    @info "Result aggregation completed" trees=result.total_trees features=length(result.selected_features) time_ms=round(result.aggregation_time, digits=1)
    
    return result
end

"""
Process results from a single tree
"""
function process_tree_result!(aggregator::ResultAggregator, tree_result::TreeResult)
    # Process selected features
    for feature_id in tree_result.selected_features
        if !haskey(aggregator.feature_scores, feature_id)
            aggregator.feature_scores[feature_id] = FeatureScore(feature_id)
        end
        
        feature_score = aggregator.feature_scores[feature_id]
        feature_score.selection_count += 1
        
        # Add score if available
        if haskey(tree_result.feature_scores, feature_id)
            score = tree_result.feature_scores[feature_id]
            feature_score.total_score += score * tree_result.confidence
        else
            # Default score if not provided
            feature_score.total_score += tree_result.confidence
        end
        
        # Track GPU votes
        gpu_id = tree_result.gpu_id
        feature_score.gpu_votes[gpu_id] = get(feature_score.gpu_votes, gpu_id, 0) + 1
    end
    
    # Process scored features even if not selected (for ranking)
    for (feature_id, score) in tree_result.feature_scores
        if !haskey(aggregator.feature_scores, feature_id)
            aggregator.feature_scores[feature_id] = FeatureScore(feature_id)
        end
        
        feature_score = aggregator.feature_scores[feature_id]
        feature_score.total_score += score * tree_result.confidence * 0.5f0  # Weight non-selected features lower
    end
end

"""
Calculate final feature rankings
"""
function calculate_feature_rankings(aggregator::ResultAggregator)::Vector{Tuple{Int, Float32}}
    rankings = Tuple{Int, Float32}[]
    
    for (feature_id, feature_score) in aggregator.feature_scores
        # Calculate average score
        if feature_score.selection_count > 0
            feature_score.average_score = feature_score.total_score / feature_score.selection_count
        else
            feature_score.average_score = feature_score.total_score / aggregator.total_trees
        end
        
        # Calculate confidence based on selection frequency
        feature_score.confidence = Float32(feature_score.selection_count) / aggregator.total_trees
        
        # Combined score: weighted average of score and selection frequency
        combined_score = 0.7f0 * feature_score.average_score + 0.3f0 * feature_score.confidence
        
        push!(rankings, (feature_id, combined_score))
    end
    
    # Sort by combined score (descending)
    sort!(rankings, by=x->x[2], rev=true)
    
    return rankings
end

"""
Find features that meet consensus threshold
"""
function find_consensus_features(
    aggregator::ResultAggregator,
    num_trees::Int
)::Vector{Int}
    consensus_features = Int[]
    required_votes = Int(ceil(num_trees * aggregator.consensus_threshold))
    
    for (feature_id, feature_score) in aggregator.feature_scores
        if feature_score.selection_count >= required_votes
            push!(consensus_features, feature_id)
        end
    end
    
    # Sort by selection count
    sort!(consensus_features, 
        by=fid -> aggregator.feature_scores[fid].selection_count, 
        rev=true)
    
    return consensus_features
end

"""
Calculate ensemble confidence
"""
function calculate_ensemble_confidence(
    aggregator::ResultAggregator,
    tree_results::Vector{TreeResult}
)::Float32
    if isempty(tree_results)
        return 0.0f0
    end
    
    # Average tree confidence
    avg_confidence = mean([tr.confidence for tr in tree_results])
    
    # Agreement factor (how much trees agree on top features)
    agreement_factor = calculate_agreement_factor(aggregator)
    
    # GPU balance factor (penalize if results are skewed to one GPU)
    balance_factor = calculate_gpu_balance_factor(aggregator, tree_results)
    
    # Combined confidence
    confidence = avg_confidence * 0.4f0 + agreement_factor * 0.4f0 + balance_factor * 0.2f0
    
    return clamp(confidence, 0.0f0, 1.0f0)
end

"""
Calculate agreement factor between trees
"""
function calculate_agreement_factor(aggregator::ResultAggregator)::Float32
    if isempty(aggregator.feature_scores)
        return 0.0f0
    end
    
    # Get top features by selection count
    top_features = sort(
        collect(aggregator.feature_scores),
        by=kv -> kv[2].selection_count,
        rev=true
    )[1:min(10, length(aggregator.feature_scores))]
    
    # Calculate agreement as ratio of votes for top features
    total_possible_votes = length(top_features) * aggregator.total_trees
    actual_votes = sum(kv[2].selection_count for kv in top_features)
    
    return Float32(actual_votes) / Float32(total_possible_votes)
end

"""
Calculate GPU balance factor
"""
function calculate_gpu_balance_factor(
    aggregator::ResultAggregator,
    tree_results::Vector{TreeResult}
)::Float32
    gpu_counts = Dict{Int, Int}()
    
    for tr in tree_results
        gpu_counts[tr.gpu_id] = get(gpu_counts, tr.gpu_id, 0) + 1
    end
    
    if length(gpu_counts) < 2
        return 0.5f0  # Single GPU, neutral score
    end
    
    # Calculate imbalance
    counts = collect(values(gpu_counts))
    max_count = maximum(counts)
    min_count = minimum(counts)
    avg_count = mean(counts)
    
    if avg_count > 0
        imbalance = (max_count - min_count) / avg_count
        balance_factor = 1.0f0 - imbalance
    else
        balance_factor = 0.0f0
    end
    
    return clamp(balance_factor, 0.0f0, 1.0f0)
end

"""
Compute hash of current results for caching
"""
function compute_results_hash(aggregator::ResultAggregator)::UInt64
    h = hash(length(aggregator.tree_results))
    
    # Include consensus threshold in hash
    h = hash(aggregator.consensus_threshold, h)
    
    # Hash tree IDs and their selected features
    for (tree_id, result) in sort(collect(aggregator.tree_results), by=x->x[1])
        h = hash((tree_id, result.selected_features), h)
    end
    
    return h
end

"""
Cache aggregated result
"""
function cache_result!(
    aggregator::ResultAggregator,
    result::EnsembleResult,
    cache_key::UInt64
)
    # Check cache size
    if length(aggregator.result_cache) >= aggregator.cache_size
        # Evict least recently used
        lru_key = nothing
        lru_time = now()
        
        for (key, entry) in aggregator.result_cache
            if entry.last_access < lru_time
                lru_key = key
                lru_time = entry.last_access
            end
        end
        
        if !isnothing(lru_key)
            delete!(aggregator.result_cache, lru_key)
        end
    end
    
    # Add to cache
    aggregator.result_cache[cache_key] = CacheEntry(result, cache_key)
end

"""
Get ensemble consensus features
"""
function get_ensemble_consensus(aggregator::ResultAggregator)::Vector{Int}
    if isnothing(aggregator.last_aggregation)
        result = aggregate_results(aggregator)
        return result.consensus_features
    end
    
    return aggregator.last_aggregation.consensus_features
end

"""
Get feature rankings
"""
function get_feature_rankings(
    aggregator::ResultAggregator;
    top_k::Union{Nothing, Int} = nothing
)::Vector{Tuple{Int, Float32}}
    if isnothing(aggregator.last_aggregation)
        result = aggregate_results(aggregator)
    else
        result = aggregator.last_aggregation
    end
    
    rankings = result.feature_rankings
    
    if !isnothing(top_k)
        return rankings[1:min(top_k, length(rankings))]
    end
    
    return rankings
end

"""
Get aggregated results
"""
function get_aggregated_results(aggregator::ResultAggregator)::Union{Nothing, EnsembleResult}
    return aggregator.last_aggregation
end

"""
Clear all results
"""
function clear_results!(aggregator::ResultAggregator)
    lock(aggregator.result_lock) do
        empty!(aggregator.tree_results)
        empty!(aggregator.feature_scores)
        aggregator.last_aggregation = nothing
        
        # Reset GPU tracking
        for gpu_id in keys(aggregator.gpu_tree_counts)
            aggregator.gpu_tree_counts[gpu_id] = 0
            empty!(aggregator.gpu_response_times[gpu_id])
        end
    end
    
    @info "All results cleared"
end

"""
Reset cache
"""
function reset_cache!(aggregator::ResultAggregator)
    lock(aggregator.result_lock) do
        empty!(aggregator.result_cache)
        aggregator.cache_hits[] = 0
        aggregator.cache_misses[] = 0
    end
    
    @info "Result cache reset"
end

"""
Get cache statistics
"""
function get_cache_stats(aggregator::ResultAggregator)
    total_requests = aggregator.cache_hits[] + aggregator.cache_misses[]
    hit_rate = total_requests > 0 ? aggregator.cache_hits[] / total_requests : 0.0
    
    return Dict(
        "cache_size" => length(aggregator.result_cache),
        "cache_hits" => aggregator.cache_hits[],
        "cache_misses" => aggregator.cache_misses[],
        "hit_rate" => hit_rate,
        "max_size" => aggregator.cache_size
    )
end

"""
Set consensus threshold
"""
function set_consensus_threshold!(aggregator::ResultAggregator, threshold::Float32)
    if 0.0f0 < threshold <= 1.0f0
        aggregator.consensus_threshold = threshold
        aggregator.last_aggregation = nothing  # Invalidate cached result
        @info "Consensus threshold set to $(threshold * 100)%"
    else
        @error "Invalid threshold: must be between 0 and 1"
    end
end

"""
Enable or disable caching
"""
function enable_caching!(aggregator::ResultAggregator, enabled::Bool)
    aggregator.caching_enabled = enabled
    
    if !enabled
        reset_cache!(aggregator)
    end
    
    @info "Result caching $(enabled ? "enabled" : "disabled")"
end

"""
Get GPU statistics
"""
function get_gpu_statistics(aggregator::ResultAggregator)
    stats = Dict{Int, Dict{String, Any}}()
    
    lock(aggregator.result_lock) do
        for gpu_id in keys(aggregator.gpu_tree_counts)
            tree_count = aggregator.gpu_tree_counts[gpu_id]
            response_times = aggregator.gpu_response_times[gpu_id]
            
            gpu_stats = Dict{String, Any}(
                "tree_count" => tree_count,
                "response_count" => length(response_times)
            )
            
            if !isempty(response_times)
                gpu_stats["avg_response_time"] = mean(response_times)
                gpu_stats["min_response_time"] = minimum(response_times)
                gpu_stats["max_response_time"] = maximum(response_times)
                gpu_stats["std_response_time"] = std(response_times)
            end
            
            stats[gpu_id] = gpu_stats
        end
    end
    
    return stats
end

"""
Create ensemble result from partial results (for fault tolerance)
"""
function create_partial_result(
    aggregator::ResultAggregator,
    min_trees::Int = 1
)::Union{Nothing, EnsembleResult}
    lock(aggregator.result_lock) do
        num_results = length(aggregator.tree_results)
        
        if num_results < min_trees
            @warn "Insufficient results for partial aggregation" have=num_results need=min_trees
            return nothing
        end
        
        @info "Creating partial result from $num_results trees"
        
        # Temporarily adjust total trees for correct calculations
        original_total = aggregator.total_trees
        aggregator.total_trees = num_results
        
        try
            result = aggregate_results(aggregator)
            return result
        finally
            aggregator.total_trees = original_total
        end
    end
end

end # module